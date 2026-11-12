"""生成经过转写核对的助手问答、上下文记忆和静音监督。"""
import argparse
import copy
from difflib import SequenceMatcher
import hashlib
import json
import math
import re
from pathlib import Path
import numpy as np
import soundfile as sf
from .asr import SpeechTranscriber
from .config import ROOT, load_config
from .corpus import write_jsonl
from .dialogue_data import DIALOGUE_PROMPT, HELDOUT, balance_repair_records, MEMORY_EVAL, MEMORY_TRAIN, clean_filler_turns, normalize_audio_text, repair_scenarios, spoken_numbers, spoken_units
from .download import atomic_json
from .model import load_base
from .pipeline import save_yaml
from .synthesis import build_synthetic_record, synthesize_query


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--corpus', default='data/formal/full-corpus-v1')
    parser.add_argument('--output', default='data/dialogue-repair-v5')
    parser.add_argument('--include-reference-semantics', action='store_true', help='兼容实验，默认只在原生声码器中使用音色参考')
    args = parser.parse_args()
    output = ROOT / args.output
    output.mkdir(parents=True, exist_ok=True)
    (output / 'audio').mkdir(exist_ok=True)
    if (output / 'dataset_report.json').is_file():
        existing = load_config(output / 'config.yaml')
        if existing['model'].get('reference_audio_in_context', True) != args.include_reference_semantics:
            raise ValueError('已有数据的参考音频模式不同，请指定新的输出目录')
        print('复用已完成的数据及其原始训练配置', output, flush=True)
        return
    corpus = ROOT / args.corpus
    raw = [json.loads(line) for line in (corpus / 'real_train.jsonl').read_text().splitlines() if line.strip()]
    audit = {'samples': len(raw), 'turns_cleaned': 0}
    references = {}
    for record in raw:
        _, removed = clean_filler_turns(record['units'])
        audit['turns_cleaned'] += removed
        if len(references) < 12 and record['voice_id'] not in references:
            path = Path(record['voice'])
            path = path if path.is_absolute() else corpus / path
            if sf.info(path).duration >= 2:
                references[record['voice_id']] = {'voice_id': record['voice_id'], 'voice': str(path.resolve())}
    voices = list(references.values())
    if len(voices) < 4:
        raise ValueError('至少需要四个独立参考音色片段')
    evaluation_references = {}
    for line in (corpus / 'real_eval.jsonl').read_text().splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        if record['voice_id'] in references or record['voice_id'] in evaluation_references:
            continue
        path = Path(record['voice'])
        path = path if path.is_absolute() else corpus / path
        if sf.info(path).duration >= 2:
            evaluation_references[record['voice_id']] = {'voice_id': record['voice_id'], 'voice': str(path.resolve())}
        if len(evaluation_references) == 6:
            break
    evaluation_voices = list(evaluation_references.values())
    if len(evaluation_voices) < 4:
        raise ValueError('缺少未参与训练的参考音色片段')
    config = copy.deepcopy(load_config(args.config))
    config['model'].update(adapter=None, system_prompt=DIALOGUE_PROMPT, decode_mode='greedy', reference_audio_in_context=args.include_reference_semantics, conversation_floor=True)
    config['training'].update(
        prepared_dir=str(output / 'prepared'), output_dir=str(ROOT / 'runs' / output.name),
        rank=8, alpha=16, target_modules=['q_proj', 'v_proj'], learning_rate=0.000005 if args.include_reference_semantics else 0.00002,
        epochs=8, gradient_accumulation=4, instruction_weight=1, compile_llm_forward=False,
        save_every=25, keep_checkpoints=5, eval_every=25,
        mask_forced_tokens=True,
        backup_dir=str((Path('/root/gpufree-share') if Path('/root/gpufree-share').is_dir() else ROOT / 'runs/backups') / ('VoiceDuplexLab-' + output.name)),
    )
    config['asr'] = {'enabled': True, 'model_dir': str(ROOT / 'models/SenseVoice-int8'), 'num_threads': 2}
    save_yaml(output / 'config.yaml', config)
    transcriber = SpeechTranscriber(config['asr'])
    model = load_base(config, with_adapter=False)
    model.init_tts()
    tokenizer = model.processor.tokenizer
    rejected = []

    def query(text, voice, identifier, check_padding=False):
        digest = hashlib.sha256((text + hashlib.sha256(Path(voice['voice']).read_bytes()).hexdigest() + config['model']['revision']).encode()).hexdigest()[:20]
        path = output / 'queries' / (digest + '.wav')
        path.parent.mkdir(exist_ok=True)
        audio = synthesize_query(model, text, voice['voice'], path)
        actual = transcriber.recognize(audio)
        similarity = SequenceMatcher(None, normalize_audio_text(text), normalize_audio_text(actual)).ratio()
        numbers_match = spoken_numbers(text) == spoken_numbers(actual)
        negation_matches = bool(re.search('[不没别无]', text)) == bool(re.search('[不没别无]', actual))
        if similarity < 0.9 or not numbers_match or not negation_matches:
            rejected.append({'id': identifier, 'audio': str(path), 'expected': text, 'transcript': actual, 'similarity': similarity, 'numbers_match': numbers_match, 'negation_match': negation_matches})
            atomic_json(output / 'rejected.json', rejected)
            return None
        if check_padding:
            padded = transcriber.recognize(np.pad(audio, (0, 10 * 16000)))
            if SequenceMatcher(None, normalize_audio_text(text), normalize_audio_text(padded)).ratio() < 0.9 or spoken_numbers(text) != spoken_numbers(padded):
                rejected.append({'id': identifier, 'audio': str(path), 'expected': text, 'transcript': padded, 'reason': '补十秒静音后转写不稳定'})
                atomic_json(output / 'rejected.json', rejected)
                return None
        return audio

    records = []
    scenarios = repair_scenarios()
    for index in range(len(scenarios) * 2):
        scenario = scenarios[index % len(scenarios)]
        voice = voices[index % len(voices)]
        speaker = voices[(index + 3) % len(voices)]
        identifier = f'repair-{index:04d}'
        audio = query(scenario['user'], speaker, identifier)
        if audio is None:
            audio = query(scenario['user'], voices[(index + 5) % len(voices)], identifier)
        if audio is None:
            continue
        record = build_synthetic_record(
            scenario, audio, voice, identifier, tokenizer,
            config['model']['max_speak_tokens'], output,
            splitter=spoken_units, trailing_listen=10,
        )
        record['prompt'] = DIALOGUE_PROMPT
        record['provenance'].update(paced_output=True, trailing_listen=10)
        records.append(record)
        print(f'完整问答 {index + 1}/{len(scenarios) * 2}', flush=True)
    accepted_questions = {record['provenance']['user_text'] for record in records}
    if len(accepted_questions) < int(len(scenarios) * 0.8):
        raise ValueError('有效完整问答的场景覆盖不足，停止并保留拒绝清单')
    for index, turns in enumerate(MEMORY_TRAIN):
        voice = voices[index % len(voices)]
        speaker = voices[(index + 3) % len(voices)]
        units, audio_parts = [], []
        for question, answer in ((turns[0], turns[1]), (turns[2], turns[3])):
            audio = query(question, speaker, f'memory-{index}')
            if audio is None:
                break
            listen = math.ceil(len(audio) / 16000) + 1
            spoken = spoken_units(answer, tokenizer, config['model']['max_speak_tokens'])
            count = listen + len(spoken) + 8
            audio_parts.append(np.pad(audio, (0, count * 16000 - len(audio))))
            units.extend([{'listen': True} for _ in range(listen)] + spoken + [{'listen': True} for _ in range(8)])
        else:
            identifier = f'memory-{index:04d}'
            path = output / 'audio' / (identifier + '.wav')
            sf.write(path, np.concatenate(audio_parts), 16000, subtype='PCM_16')
            records.append({'id': identifier, **voice, 'audio': str(path), 'units': units, 'prompt': DIALOGUE_PROMPT, 'provenance': {'kind': 'synthetic_instruction', 'scenario': 'memory'}})
    for index in range(24):
        identifier = f'quiet-{index:04d}'
        path = output / 'audio' / (identifier + '.wav')
        sf.write(path, np.zeros(12 * 16000, dtype=np.float32), 16000, subtype='PCM_16')
        records.append({'id': identifier, **voices[index % len(voices)], 'audio': str(path), 'units': [{'listen': True} for _ in range(12)], 'prompt': DIALOGUE_PROMPT, 'provenance': {'kind': 'synthetic_instruction', 'scenario': 'silence'}})
    evaluation = []
    fixtures = []
    for index, scenario in enumerate(HELDOUT):
        voice = evaluation_voices[index % len(evaluation_voices)]
        spoken_user = scenario.get('spoken_user', scenario['user'])
        audio = query(spoken_user, evaluation_voices[(index + 1) % len(evaluation_voices)], f'eval-{index}', check_padding=True)
        if audio is None:
            audio = query(spoken_user, evaluation_voices[(index + 2) % len(evaluation_voices)], f'eval-{index}', check_padding=True)
        if audio is None:
            audio = query(spoken_user, {'voice': str(ROOT / 'models/MiniCPM-o-4_5/assets/HT_ref_audio.wav')}, f'eval-{index}', check_padding=True)
        if audio is None:
            raise ValueError('未参与训练的评估录音转写检查失败，请检查拒绝清单')
        record = build_synthetic_record(scenario | {'kind': 'direct'}, audio, voice, f'eval-{index:04d}', tokenizer, config['model']['max_speak_tokens'], output, splitter=spoken_units, trailing_listen=10)
        record['prompt'] = DIALOGUE_PROMPT
        evaluation.append(record)
        fixtures.append({'id': record['id'], 'question': scenario['user'], 'spoken_question': spoken_user, 'keywords': scenario['keywords'], 'audio': record['audio'], 'voice': record['voice']})
    memory_fixtures = []
    for index, turns in enumerate(MEMORY_EVAL):
        speaker = evaluation_voices[(index + 1) % len(evaluation_voices)]
        paths = []
        for position in (0, 2):
            audio = query(turns[position], speaker, f'memory-eval-{index}-{position}')
            if audio is None:
                raise ValueError('上下文评估录音转写检查失败')
            path = output / 'audio' / f'memory-eval-{index}-{position}.wav'
            sf.write(path, audio, 16000, subtype='PCM_16')
            paths.append(str(path))
        memory_fixtures.append({'id': f'memory-eval-{index}', 'questions': [turns[0], turns[2]], 'audio': paths, 'voice': evaluation_voices[index]['voice'], 'keyword': turns[4]})
    if len(records) < len(scenarios):
        raise ValueError('有效完整问答不足，停止训练并保留拒绝清单')
    unique_records = len(records)
    records = balance_repair_records(records)
    write_jsonl(output / 'train.jsonl', records)
    write_jsonl(output / 'eval.jsonl', evaluation)
    atomic_json(output / 'fixtures.json', fixtures)
    atomic_json(output / 'memory_fixtures.json', memory_fixtures)
    atomic_json(output / 'rejected.json', rejected)
    atomic_json(output / 'dataset_report.json', {
        'train_samples': len(records), 'unique_train_samples': unique_records, 'sampling': '工具、失败、取消和记忆样本重复采样三次，其余一次', 'eval_samples': len(evaluation), 'source_audit': audit,
        'train_source': '完整助手问答、上下文记忆、静音倾听；原始自然聊天仅提供参考音色',
        'output_pacing': '最多六个字符一秒单元', 'post_answer_listen_seconds': 10,
        'rejected_queries': len(rejected), 'train_texts': [row['user'] for row in scenarios],
        'evaluation_reference_clips_held_out': True,
    })
    transcriber.close()
    print('新训练数据完成', output, flush=True)


if __name__ == '__main__':
    main()
