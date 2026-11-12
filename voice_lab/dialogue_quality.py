"""用未参与训练的问题、静音和上下文记忆检查实际语音回复。"""
import argparse
import hashlib
from difflib import SequenceMatcher
import json
import time
import numpy as np
import soundfile as sf
from .asr import SpeechTranscriber
from .config import ROOT, load_config
from .dialogue_data import FILLERS, normalize_audio_text, spoken_numbers
from .download import atomic_json
from .floor import FLOOR_VERSION, is_preface
from .model import VoiceRuntime, read_audio


def observe_turn(runtime, source, voice_state, keywords, min_chars=8, max_frames=45):
    offset = 0
    texts, turns, events = [], [], []
    pending = ''
    ended = None
    answer_turn = None
    first = voice_state['frames'] == 0
    began = time.monotonic()
    for index in range(max_frames):
        needed = voice_state['first_samples'] if first and index == 0 else 16000
        wave = source[offset:offset + needed]
        offset += needed
        wave = np.pad(wave, (0, needed - len(wave)))
        value = runtime.step(wave.astype(np.float32))
        voice_state['samples'] += needed
        voice_state['frames'] += 1
        moment = voice_state['samples'] / 16000
        if value['kind'] not in ('listen', 'speech'):
            return {'passed': False, 'error': '普通问题出现控制输出', 'kind': value['kind']}
        if value['kind'] == 'speech':
            for block in value['blocks']:
                text = block['text']
                texts.append(text)
                pending += text
                audio = block['audio']
                play_at = max(moment, voice_state['play_end'])
                voice_state['audio'].append((play_at, audio.copy()))
                voice_state['play_end'] = play_at + len(audio) / 24000
                events.append({'text': text, 'end': block['end'], 'input_time': moment, 'audio_samples': len(audio)})
                if block['end']:
                    turns.append(pending)
                    normalized = normalize_audio_text(pending)
                    if ended is None and len(normalized) >= min_chars and not FILLERS.fullmatch(normalized) and not is_preface(pending):
                        ended = max(moment, voice_state['play_end'])
                        answer_turn = len(turns)
                    pending = ''
        if ended is not None and moment >= ended + 10 and offset >= len(source):
            break
    text = ''.join(texts)
    filler_turns = [turn for turn in turns if FILLERS.fullmatch(normalize_audio_text(turn))]
    normalized = normalize_audio_text(text)
    keyword_found = keywords is None or any(normalize_audio_text(word) in normalized for word in keywords)
    normalized_turns = [normalize_audio_text(turn) for turn in turns if normalize_audio_text(turn)]
    repeated_turns = len(normalized_turns) - len(set(normalized_turns))
    followup_turns = len(turns) - answer_turn if answer_turn is not None else 0
    return {
        'text': text, 'turns': turns, 'filler_turns': filler_turns, 'events': events,
        'keyword_found': keyword_found, 'response_completed': ended is not None,
        'repeated_turns': repeated_turns,
        'followup_turns': followup_turns,
        'passed': ended is not None and keyword_found and len(filler_turns) <= 1 and repeated_turns == 0 and followup_turns <= 1,
        'wall_seconds': round(time.monotonic() - began, 3),
    }


def audio_track(blocks):
    blocks = [(start, wave) for start, wave in blocks if len(wave)]
    if not blocks:
        return np.empty(0, dtype=np.float32)
    first = min(start for start, wave in blocks if len(wave))
    end = max(start + len(wave) / 24000 for start, wave in blocks)
    output = np.zeros(int((end - first) * 24000) + 2, dtype=np.float32)
    for start, wave in blocks:
        position = int((start - first) * 24000)
        output[position:position + len(wave)] += wave
    return np.clip(output, -1, 1)


def main():
    import torch
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--data', default='data/dialogue-repair-v5')
    parser.add_argument('--adapter', default='none')
    parser.add_argument('--output', required=True)
    parser.add_argument('--limit', type=int, default=20)
    args = parser.parse_args()
    directory = ROOT / args.data
    output = ROOT / args.output
    output.mkdir(parents=True, exist_ok=True)
    config = load_config(args.config)
    config['model']['adapter'] = None if args.adapter == 'none' else str((ROOT / args.adapter).resolve())
    runtime = VoiceRuntime(config)
    transcriber = SpeechTranscriber(config['asr'])
    fixture_bytes = (directory / 'fixtures.json').read_bytes()
    fixtures = json.loads(fixture_bytes)[:args.limit]
    results = []

    def start(reference, seed):
        torch.manual_seed(seed)
        np.random.seed(seed)
        sizes = runtime.start(reference)
        return {'frames': 0, 'samples': 0, 'first_samples': sizes['first_samples'], 'play_end': 0, 'audio': []}

    def save_case(identifier, result, state):
        audio = audio_track(state['audio'])
        if len(audio):
            path = output / (identifier + '.wav')
            sf.write(path, audio, 24000, subtype='PCM_16')
            transcript = transcriber.recognize(audio, 24000)
            expected = normalize_audio_text(result.get('text', ''))
            actual = normalize_audio_text(transcript)
            similarity = SequenceMatcher(None, expected, actual).ratio()
            result.update(audio=str(path), transcript=transcript, audio_text_similarity=similarity)
            numbers_match = spoken_numbers(result.get('text', '')) == spoken_numbers(transcript)
            result['audio_numbers_match'] = numbers_match
            result['passed'] = result['passed'] and similarity >= 0.8 and numbers_match
        else:
            result['passed'] = False
        print(identifier, result['passed'], result.get('text', ''), flush=True)
        return result

    for index, fixture in enumerate(fixtures):
        state = start(fixture['voice'], 42 + index)
        source = read_audio(fixture['audio'])
        minimum = 2 if '十四' in fixture['keywords'] else 8
        result = observe_turn(runtime, source, state, fixture['keywords'], minimum)
        result.update(id=fixture['id'], question=fixture['question'], input_audio=fixture['audio'], spoken_question=fixture.get('spoken_question', fixture['question']), fixture_revision=fixture.get('fixture_revision', 1))
        generated = normalize_audio_text(result.get('text', ''))
        question = normalize_audio_text(fixture['question'])
        if SequenceMatcher(None, generated, question).ratio() > 0.9:
            result['passed'] = False
            result['echoed_question'] = True
        results.append(save_case(fixture['id'], result, state))
        atomic_json(output / 'progress.json', {'completed': len(results), 'passed': sum(row['passed'] for row in results)})
    memory_results = []
    for index, fixture in enumerate(json.loads((directory / 'memory_fixtures.json').read_text())):
        state = start(fixture['voice'], 500 + index)
        initial = observe_turn(runtime, read_audio(fixture['audio'][0]), state, None, 2)
        second = observe_turn(runtime, read_audio(fixture['audio'][1]), state, [fixture['keyword']], 2)
        combined = dict(second)
        combined['id'] = fixture['id']
        combined['text'] = initial.get('text', '') + second.get('text', '')
        combined['first_turn'] = initial
        combined['passed'] = initial['passed'] and second['passed']
        memory_results.append(save_case(fixture['id'], combined, state))
    silence_results = []
    for index, reference in enumerate([fixtures[0]['voice'], fixtures[-1]['voice']]):
        state = start(reference, 900 + index)
        result = observe_turn(runtime, np.zeros(15 * 16000, dtype=np.float32), state, ['不会出现的检验词'], max_frames=15)
        filler = result.get('filler_turns', [])
        text = result.get('text', '')
        quiet = not normalize_audio_text(text)
        voiced_seconds = sum(len(wave) / 24000 for _, wave in state['audio'] if len(wave) and float(np.sqrt(np.mean(wave ** 2))) > 0.001)
        if not text and voiced_seconds > 0.16:
            quiet = False
        silence_results.append({'text': text, 'filler_turns': filler, 'voiced_seconds': voiced_seconds, 'passed': not filler and quiet})
    sequence_results = []
    if len(fixtures) >= 6:
        state = start(fixtures[0]['voice'], 1200)
        for position in (0, 4, 5):
            fixture = fixtures[position]
            minimum = 2 if '十四' in fixture['keywords'] else 8
            result = observe_turn(runtime, read_audio(fixture['audio']), state, fixture['keywords'], minimum)
            result.update(id=fixture['id'], question=fixture['question'], input_audio=fixture['audio'], spoken_question=fixture.get('spoken_question', fixture['question']), fixture_revision=fixture.get('fixture_revision', 1))
            sequence_results.append(result)
        track = audio_track(state['audio'])
        if len(track):
            sf.write(output / 'continuous-three-topics.wav', track, 24000, subtype='PCM_16')
    count = sum(result['passed'] for result in results)
    passed = count >= int(np.ceil(len(results) * 0.9)) and all(row['passed'] for row in memory_results + silence_results + sequence_results)
    report = {
        'fixture_manifest_sha256': hashlib.sha256(fixture_bytes).hexdigest(),
        'passed': passed, 'adapter': config['model']['adapter'], 'ordinary_passed': count,
        'ordinary_total': len(results), 'cases': results, 'memory': memory_results,
        'silence': silence_results, 'continuous_topics': sequence_results,
        'decode_mode': config['model'].get('decode_mode', 'sampling'),
        'reference_audio_in_context': config['model'].get('reference_audio_in_context', True),
        'conversation_floor_enabled': config['model'].get('conversation_floor', False),
        'conversation_floor_version': FLOOR_VERSION if config['model'].get('conversation_floor') else None,
        'system_prompt': config['model'].get('system_prompt'),
        'criteria': '至少90%未训练问答有效回答；语音/文本一致度>=0.8；每轮最多一次纯语气词且无整句重复；记忆、静音和同会话连续三话题均通过',
    }
    atomic_json(output / 'report.json', report)
    transcriber.close()
    print(json.dumps({'passed': passed, 'ordinary_passed': count, 'ordinary_total': len(results)}, ensure_ascii=False), flush=True)
    raise SystemExit(0 if passed else 2)


if __name__ == '__main__':
    main()
