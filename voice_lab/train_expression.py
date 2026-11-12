"""原生speech-token监督训练，支持特征复用、声学LoRA及精确断点恢复。"""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import random
import time

from .acoustic import ACOUSTIC_STYLES, condition_with_style, language_adapter_digest
from .config import load_config
from .download import atomic_json
from .expression import expression_context
from .model import load_base, read_audio
from .prompts import VOICE_PROMPT
from .prepare import Capture


def manifest_rows(path):
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    identifiers = set()
    for row in rows:
        if row['id'] in identifiers or row['style'] not in ACOUSTIC_STYLES or row['split'] not in ('train', 'eval'):
            raise ValueError('情绪训练清单重复或字段无效')
        identifiers.add(row['id'])
        row['audio'] = str((path.parent / row['audio']).resolve())
        if not Path(row['audio']).is_file():
            raise FileNotFoundError(row['audio'])
    return rows


def prepare(args, config):
    import numpy as np
    import s3tokenizer
    import torch
    rows = manifest_rows(args.manifest)
    digest = hashlib.sha256(args.manifest.read_bytes())
    adapter_hash = language_adapter_digest(config)
    digest.update(json.dumps({'revision': config['model']['revision'], 'language_adapter': adapter_hash, 'compiler': 3,
                             'prompts': [VOICE_PROMPT] + [expression_context(style, 'unknown') for style in ACOUSTIC_STYLES]}, sort_keys=True).encode())
    for row in rows:
        digest.update(Path(row['audio']).read_bytes())
    signature = digest.hexdigest()
    output = args.prepared
    output.mkdir(parents=True, exist_ok=True)
    metadata_path = output / 'index.json'
    if metadata_path.exists() and json.loads(metadata_path.read_text())['signature'] != signature:
        raise ValueError('训练输入已变化，请使用新的特征目录')
    model = load_base(config)
    model.init_tts()
    compiled = []
    for number, row in enumerate(rows, 1):
        target = output / (hashlib.sha256(row['id'].encode()).hexdigest()[:20] + '.pt')
        if target.exists():
            saved = torch.load(target, map_location='cpu', weights_only=True)
            if saved['signature'] != signature:
                raise ValueError('已有特征指纹不一致')
        else:
            with torch.no_grad():
                model.reset_session(reset_token2wav_cache=True)
                original_init = model.init_tts
                try:
                    # 声码器已经载入，逐样本重建它会重复读权重并拖慢特征编译。
                    model.init_tts = lambda **kwargs: None
                    duplex = model.as_duplex(generate_audio=False, sliding_window_mode='off')
                finally:
                    model.init_tts = original_init
                capture = Capture(duplex.decoder, 4096, forward=False, storage_device=config['model']['device'])
                duplex.decoder.feed = capture.feed
                style = row['style']
                prompt = VOICE_PROMPT + '\n' + expression_context(style, 'unknown')
                duplex.prepare(prefix_system_prompt=prompt, ref_audio=None)
                result = duplex.streaming_prefill(
                    audio_waveform=np.zeros(duplex.processor.get_streaming_chunk_size(), dtype=np.float32),
                    text_list=['请按指定的情绪与发声方式说出给定内容。'],
                )
                if not result['success']:
                    raise RuntimeError('原生语音预填充失败')
                duplex.decoder.feed(duplex.decoder.embed_token(duplex.speak_token_id))
                tokens = duplex.tokenizer.encode(row['text'], add_special_tokens=False)
                tokens += [duplex.tokenizer.convert_tokens_to_ids('<|turn_eos|>')]
                duplex.decoder.feed(duplex.decoder.embed_tokens(tokens))
                duplex.decoder.feed = capture.original
                # 教师强制条件已完整收集，一次因果前向与逐token前向使用相同位置。
                inputs = torch.cat(capture.embeddings).to(device=model.device, dtype=model.llm.dtype).unsqueeze(0)
                result = model.llm(inputs_embeds=inputs, output_hidden_states=True, use_cache=False, return_dict=True)
                hidden = result.hidden_states[-1][:, -len(tokens):]
                parts = [[token, hidden[:, index:index + 1], False] for index, token in enumerate(tokens)]
                condition = duplex._convert_results_to_tts_input(parts)
                audio = read_audio(row['audio'])
                mels = s3tokenizer.log_mel_spectrogram(torch.from_numpy(audio))
                mels, lengths = s3tokenizer.padding([mels])
                codes, code_lengths = model.tts.audio_tokenizer.audio_tokenizer.quantize(mels.cuda(), lengths.cuda())
                codes = codes[0, :int(code_lengths[0])].long()
                eos = model.tts.config.num_audio_tokens - 1
                if not len(codes) or bool((codes < 0).any()) or bool((codes >= eos).any()):
                    raise ValueError('目标音频token无效')
                saved = {'condition': condition.cpu().to(torch.bfloat16).clone(),
                         'codes': torch.cat((codes.cpu(), torch.tensor([eos]))),
                         'vocal_only': row.get('event') in ('laughter', 'crying'),
                         'style': ACOUSTIC_STYLES.index(style), 'id': row['id'], 'signature': signature}
            temporary = target.with_suffix('.partial')
            torch.save(saved, temporary)
            temporary.replace(target)
        compiled.append({'file': target.name, 'split': row['split'], 'style': row['style'], 'id': row['id']})
        atomic_json(metadata_path, {'signature': signature, 'compiler': 3, 'records': compiled,
                                    'conditioning_version': 2,
                                    'complete': number == len(rows), 'base_revision': config['model']['revision'],
                                    'language_adapter_sha256': adapter_hash})
        if number % 10 == 0 or number == len(rows):
            print(json.dumps({'prepared': number, 'total': len(rows)}, ensure_ascii=False), flush=True)


def speech_loss(tts, embedding, record):
    import torch
    import torch.nn.functional as functional
    device = tts.device
    condition = record['condition'].to(device=device, dtype=tts.emb_text.weight.dtype)
    condition = condition_with_style(condition, embedding, record['style'], vocal_only=record.get('vocal_only', False))
    codes = record['codes'].to(device)
    preceding = tts.emb_code[0](codes[:-1]).unsqueeze(0)
    inputs = torch.cat((condition, preceding), dim=1)
    hidden = tts.model(inputs_embeds=inputs, use_cache=False, return_dict=True).last_hidden_state
    predictions = tts.head_code[0](hidden[:, condition.shape[1] - 1:]).float()
    return functional.cross_entropy(predictions.reshape(-1, predictions.shape[-1]), codes)


def train(args, config):
    import torch
    from peft import LoraConfig, TaskType, get_peft_model
    if args.output.exists() and not args.resume and any(path.name != '.train.lock' for path in args.output.iterdir()):
        raise FileExistsError('训练输出已有内容，请指定--resume或使用新的输出目录')
    index = json.loads((args.prepared / 'index.json').read_text())
    if not index.get('complete'):
        raise ValueError('声学特征尚未完整准备')
    if index['base_revision'] != config['model']['revision'] or index['language_adapter_sha256'] != language_adapter_digest(config):
        raise ValueError('声学特征与当前语音模型不匹配')
    rows = index['records']
    training = [row for row in rows if row['split'] == 'train']
    evaluation = [row for row in rows if row['split'] == 'eval']
    if not training or not evaluation:
        raise ValueError('需要独立的训练集和验证集')
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    model = load_base(config)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    model.tts.model = get_peft_model(model.tts.model, LoraConfig(
        task_type=TaskType.FEATURE_EXTRACTION, r=8, lora_alpha=16,
        target_modules=['q_proj', 'v_proj'], lora_dropout=0.05,
    ))
    embedding = torch.nn.Embedding(len(ACOUSTIC_STYLES), model.tts.config.hidden_size).to(model.tts.device)
    torch.nn.init.zeros_(embedding.weight)
    initialization = None
    if args.initialize:
        from peft import set_peft_model_state_dict
        from safetensors.torch import load_file
        metadata = json.loads((args.initialize / 'acoustic_metadata.json').read_text())
        if metadata['base_revision'] != index['base_revision'] or metadata['language_adapter_sha256'] != index['language_adapter_sha256'] or tuple(metadata['styles']) != ACOUSTIC_STYLES:
            raise ValueError('初始化适配器与当前基座或情绪编号不一致')
        weights = args.initialize / 'tts_lora/adapter_model.safetensors'
        set_peft_model_state_dict(model.tts.model, load_file(str(weights), device='cpu'))
        embedding.load_state_dict(torch.load(args.initialize / 'style_embedding.pt', map_location='cpu', weights_only=True))
        initialization = {'checkpoint': args.initialize.name, 'adapter_sha256': hashlib.sha256(weights.read_bytes()).hexdigest()}
    lora_parameters = [parameter for parameter in model.tts.model.parameters() if parameter.requires_grad]
    parameters = lora_parameters + list(embedding.parameters())
    optimizer = torch.optim.AdamW([
        {'params': lora_parameters, 'lr': args.learning_rate},
        {'params': embedding.parameters(), 'lr': args.style_learning_rate},
    ])
    start_step = 0
    best = float('inf')
    signature_data = {'inputs': index['signature'], 'learning_rate': args.learning_rate,
                      'style_learning_rate': args.style_learning_rate,
                      'seed': args.seed, 'rank': 8, 'alpha': 16, 'compiler': 1}
    if 'conditioning_version' in index:
        signature_data['conditioning_version'] = index['conditioning_version']
    signature = hashlib.sha256(json.dumps(signature_data, sort_keys=True).encode()).hexdigest()
    if args.resume:
        state = torch.load(args.resume / 'state.pt', map_location='cpu', weights_only=True)
        if state['signature'] != signature:
            raise ValueError('续训参数或训练输入与断点不一致')
        from peft import set_peft_model_state_dict
        set_peft_model_state_dict(model.tts.model, state['adapter'])
        embedding.load_state_dict(state['embedding'])
        optimizer.load_state_dict(state['optimizer'])
        random.setstate(state['python_rng'])
        torch.set_rng_state(state['torch_rng'])
        for device, rng in enumerate(state['cuda_rng'][:torch.cuda.device_count()]):
            torch.cuda.set_rng_state(rng, device=device)
        previous_device = state.get('training_cuda_index', 0)
        torch.cuda.set_rng_state(state['cuda_rng'][previous_device], device=torch.cuda.current_device())
        start_step, best = state['step'], state['best_eval']
        initialization = state.get('initialization')
    args.output.mkdir(parents=True, exist_ok=True)
    def load(row):
        return torch.load(args.prepared / row['file'], map_location='cpu', weights_only=True)
    def evaluate():
        model.tts.model.eval()
        losses = []
        with torch.no_grad():
            for row in evaluation:
                losses.append(float(speech_loss(model.tts, embedding, load(row))))
        model.tts.model.train()
        return sum(losses) / len(losses)
    def save(step, score):
        from peft import get_peft_model_state_dict
        target = args.output / f'step-{step:07d}'
        temporary = args.output / f'.step-{step:07d}.partial'
        if target.exists() or temporary.exists():
            raise FileExistsError('断点目录已存在，请保留原结果并选择新输出位置')
        temporary.mkdir()
        model.tts.model.save_pretrained(temporary / 'tts_lora', safe_serialization=True)
        torch.save(embedding.state_dict(), temporary / 'style_embedding.pt')
        metadata = {'base_revision': index['base_revision'], 'language_adapter_sha256': index['language_adapter_sha256'],
                    'styles': list(ACOUSTIC_STYLES), 'step': step, 'eval_loss': score,
                    'training_signature': signature, 'ready_for_user_acceptance': False,
                    'initialized_from': initialization,
                    'conditioning_version': index.get('conditioning_version', 1),
                    'scope': '原生TTS的speech-token交叉熵、情绪条件与LoRA；需要独立生成验收'}
        atomic_json(temporary / 'acoustic_metadata.json', metadata)
        state = {'adapter': get_peft_model_state_dict(model.tts.model), 'embedding': embedding.state_dict(),
                 'optimizer': optimizer.state_dict(), 'python_rng': random.getstate(),
                 'torch_rng': torch.get_rng_state(), 'cuda_rng': torch.cuda.get_rng_state_all(),
                 'training_cuda_index': torch.cuda.current_device(),
                 'initialization': initialization,
                 'step': step, 'best_eval': best, 'signature': signature}
        torch.save(state, temporary / 'state.pt')
        temporary.replace(target)
        return target
    initial = evaluate()
    best = min(best, initial)
    print(json.dumps({'step': start_step, 'initial_eval_loss': initial}), flush=True)
    began = time.monotonic()
    last_saved = start_step
    model.tts.model.train()
    for step in range(start_step + 1, args.max_steps + 1):
        record = load(random.choice(training))
        optimizer.zero_grad(set_to_none=True)
        loss = speech_loss(model.tts, embedding, record)
        if not torch.isfinite(loss):
            raise ValueError('声学损失出现非有限值')
        loss.backward()
        torch.nn.utils.clip_grad_norm_(parameters, 1.0)
        optimizer.step()
        metrics = {'step': step, 'train_speech_token_loss': float(loss.detach()), 'elapsed': time.monotonic() - began}
        timed_out = metrics['elapsed'] >= args.max_seconds
        if step % args.save_every == 0 or step == args.max_steps or timed_out:
            score = evaluate()
            metrics['eval_speech_token_loss'] = score
            improved = score < best
            best = min(best, score)
            checkpoint = save(step, score)
            last_saved = step
            atomic_json(args.output / 'latest.json', {'checkpoint': checkpoint.name, 'step': step, 'eval_loss': score})
            if improved:
                atomic_json(args.output / 'best.json', {'checkpoint': checkpoint.name, 'step': step, 'eval_loss': score})
        with (args.output / 'metrics.jsonl').open('a') as stream:
            stream.write(json.dumps(metrics) + '\n')
        if step % 10 == 0 or timed_out:
            print(json.dumps(metrics), flush=True)
        if timed_out:
            break
    print(json.dumps({'last_saved_step': last_saved, 'best_eval': best}), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=['prepare', 'train'])
    parser.add_argument('--config', required=True)
    parser.add_argument('--manifest', type=Path)
    parser.add_argument('--prepared', required=True, type=Path)
    parser.add_argument('--output', type=Path, default=Path('runs/expression-acoustic'))
    parser.add_argument('--resume', type=Path)
    parser.add_argument('--initialize', type=Path)
    parser.add_argument('--max-steps', type=int, default=200)
    parser.add_argument('--max-seconds', type=int, default=1800)
    parser.add_argument('--save-every', type=int, default=50)
    parser.add_argument('--learning-rate', type=float, default=0.00005)
    parser.add_argument('--style-learning-rate', type=float, default=0.001)
    parser.add_argument('--seed', type=int, default=20261112)
    args = parser.parse_args()
    if args.resume and args.initialize:
        parser.error('--resume恢复原作业，--initialize从已有权重创建新数据作业，两者不能同时使用')
    if min(args.max_steps, args.max_seconds, args.save_every) <= 0:
        parser.error('步数和时限必须为正数')
    if args.mode == 'prepare' and args.manifest is None:
        parser.error('准备特征需要--manifest')
    config = copy.deepcopy(load_config(args.config))
    config['model'].pop('acoustic_adapter', None)
    import fcntl
    directory = args.prepared if args.mode == 'prepare' else args.output
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / ('.' + args.mode + '.lock')).open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        (prepare if args.mode == 'prepare' else train)(args, config)


if __name__ == '__main__':
    main()
