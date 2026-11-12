"""联合训练主语言骨干与原生语音解码器，保留语音损失到语言隐藏状态的梯度。"""
import argparse
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import random
import signal

from .config import load_config
from .download import atomic_json
from .model import load_base, read_audio
from .prepare import Capture, target_ids, target_labels
from .prompts import NATIVE_VOICE_PROMPT

NATIVE_COMPILER_VERSION = 4


def read_rows(path):
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    identifiers = set()
    for row in rows:
        if row['id'] in identifiers or row['split'] not in ('train', 'eval'):
            raise ValueError('样本编号重复或数据划分无效')
        identifiers.add(row['id'])
        for key in ('user_audio', 'reference_audio'):
            row[key] = str((path.parent / row[key]).resolve())
            if not Path(row[key]).is_file():
                raise FileNotFoundError(row[key])
        if not row.get('outputs') and not row.get('units'):
            raise ValueError('每条记录需要助手输出或完整流式对话单元')
        for unit in row.get('outputs', []):
            if not isinstance(unit['text'], str) or not unit['text']:
                raise ValueError('助手文字不能为空')
            if unit.get('audio'):
                unit['audio'] = str((path.parent / unit['audio']).resolve())
                if not Path(unit['audio']).is_file():
                    raise FileNotFoundError(unit['audio'])
    if not rows:
        raise ValueError('训练清单为空')
    return rows


def native_config(config):
    config = deepcopy(config)
    for key in ('acoustic_adapter', 'vocoder_adapter'):
        if config['model'].get(key):
            raise ValueError('联合训练不能装入旧版情绪分支')
    return config


def compile_record(model, config, row):
    import numpy as np
    import s3tokenizer
    import torch

    if row.get('units'):
        from .prepare import prepare_record
        inherited = deepcopy(config)
        inherited['model']['reference_audio_in_context'] = row.get('reference_audio_in_context', False)
        inherited['training']['compile_llm_forward'] = False
        inherited['training']['capture_device'] = config['model']['device']
        record = {'id': row['id'], 'audio': row['user_audio'], 'voice': row['reference_audio'],
                  'prompt': NATIVE_VOICE_PROMPT, 'units': row['units']}
        compiled = prepare_record(model, inherited, record)
        return {'inputs_embeds': compiled['inputs_embeds'].cpu(), 'labels': compiled['labels'],
                'spans': [], 'id': row['id'], 'controls': [], 'audio_units': [], 'language_weight': 1.0}

    model.reset_session(reset_token2wav_cache=True)
    original_init = model.init_tts
    try:
        model.init_tts = lambda **kwargs: None
        d = model.as_duplex(generate_audio=False, sliding_window_mode='off')
    finally:
        model.init_tts = original_init
    capture = Capture(d.decoder, config['training']['max_tokens'], forward=False,
                      storage_device=config['model']['device'])
    d.decoder.feed = capture.feed
    spans = []
    audio_units = []
    try:
        with torch.no_grad():
            d.prepare(prefix_system_prompt=NATIVE_VOICE_PROMPT,
                      ref_audio=read_audio(row['reference_audio']))
            audio = read_audio(row['user_audio'])
            offset = 0
            while offset < len(audio):
                required = max(0, d.processor.get_streaming_chunk_size() - len(d.audio_buffer))
                if required == 0:
                    raise RuntimeError('音频分块未前进')
                chunk = audio[offset:offset + required]
                offset += len(chunk)
                if len(chunk) < required:
                    chunk = np.pad(chunk, (0, required - len(chunk)))
                before = len(capture.embeddings)
                result = d.streaming_prefill(audio_waveform=chunk)
                if not result['success']:
                    raise RuntimeError('用户音频编码失败')
                if row.get('controls'):
                    audio_units.append(torch.cat(capture.embeddings[before:]))
                if offset < len(audio):
                    ids = target_ids({'listen': True}, d.tokenizer)
                    capture.target = target_labels({'listen': True}, d.tokenizer, mask_forced_tokens=True)
                    d.decoder.feed(d.decoder.embed_tokens(ids))
                    capture.target = None
            for number, unit in enumerate(row['outputs']):
                text_ids = d.tokenizer.encode(unit['text'], add_special_tokens=False)
                if number == len(row['outputs']) - 1:
                    text_ids += [d.tokenizer.convert_tokens_to_ids('<|turn_eos|>')]
                speak = d.tokenizer.encode('<|speak|>', add_special_tokens=False)
                closing = d.tokenizer.encode('<|chunk_eos|></unit>', add_special_tokens=False)
                ids = speak + text_ids + closing
                start = capture.count + len(speak)
                capture.target = ids.copy()
                forced = len(d.tokenizer.encode('</unit>', add_special_tokens=False))
                capture.target[-forced:] = [-100] * forced
                d.decoder.feed(d.decoder.embed_tokens(ids))
                capture.target = None
                if unit.get('audio'):
                    mels = s3tokenizer.log_mel_spectrogram(torch.from_numpy(read_audio(unit['audio'])))
                    mels, lengths = s3tokenizer.padding([mels])
                    device = model.tts.device
                    codes, sizes = model.tts.audio_tokenizer.audio_tokenizer.quantize(mels.to(device), lengths.to(device))
                    codes = codes[0, :int(sizes[0])].long().cpu()
                    eos = model.tts.config.num_audio_tokens - 1
                    if not len(codes) or (codes < 0).any() or (codes >= eos).any():
                        raise ValueError('目标语音编码无效')
                    from .native_flow import prepare_flow
                    flow = prepare_flow(model.tts.audio_tokenizer, read_audio(row['reference_audio']),
                                        read_audio(unit['audio']), codes)
                    spans.append({'start': start, 'ids': torch.tensor(text_ids),
                                  'codes': torch.cat((codes, torch.tensor([eos]))), 'flow': flow})
    finally:
        d.decoder.feed = capture.original
        d.decoder.reset()
    return {'inputs_embeds': torch.cat(capture.embeddings).cpu(), 'labels': torch.cat(capture.labels),
            'spans': spans, 'id': row['id'], 'audio_units': [unit.cpu() for unit in audio_units],
            'controls': row.get('controls', []), 'language_weight': row.get('language_weight', 1.0)}


def preparation_signature(rows, config):
    digest = hashlib.sha256(json.dumps({'rows': rows, 'revision': config['model']['revision'],
                                      'prompt': NATIVE_VOICE_PROMPT, 'compiler': NATIVE_COMPILER_VERSION}, sort_keys=True).encode())
    for row in rows:
        for path in [row['user_audio'], row['reference_audio']] + [u['audio'] for u in row.get('outputs', []) if u.get('audio')]:
            digest.update(Path(path).read_bytes())
    from .acoustic import language_adapter_digest
    digest.update(json.dumps(language_adapter_digest(config)).encode())
    return digest.hexdigest()


def prepare(args, config):
    import torch
    from .acoustic import language_adapter_digest
    rows = read_rows(args.manifest)
    signature = preparation_signature(rows, config)
    args.prepared.mkdir(parents=True, exist_ok=True)
    import fcntl
    process_lock = (args.prepared / '.prepare.lock').open('w')
    try:
        fcntl.flock(process_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        process_lock.close()
        raise RuntimeError('同一特征目录已有编译进程') from exc
    index_path = args.prepared / 'index.json'
    if index_path.exists() and json.loads(index_path.read_text())['signature'] != signature:
        raise ValueError('数据或基座变化，请使用新的编译目录')
    model = load_base(config)
    model.init_tts()
    model.tts.audio_tokenizer.flow.eval()
    records = []
    for row in rows:
        name = hashlib.sha256(row['id'].encode()).hexdigest()[:24] + '.pt'
        path = args.prepared / name
        if not path.exists():
            record = compile_record(model, config, row)
            temporary = path.with_suffix('.partial')
            torch.save({**record, 'signature': signature}, temporary)
            temporary.replace(path)
        records.append({'file': name, 'split': row['split'], 'id': row['id'],
                        'domain': row.get('domain', row.get('style', row.get('emotion', 'dialogue')))})
        atomic_json(index_path, {'signature': signature, 'records': records,
                                'base_revision': config['model']['revision'],
                                'language_initialization_sha256': language_adapter_digest(config),
                                'compiler': NATIVE_COMPILER_VERSION, 'prompt': NATIVE_VOICE_PROMPT,
                                'auxiliary_tasks': any(row.get('controls') for row in rows),
                                'coverage': {key: sorted({str(row.get(key, 'unlabelled')) for row in rows})
                                             for key in ('emotion', 'user_emotion', 'speaking_rate', 'speaker')},
                                'complete': len(records) == len(rows)})
        print(json.dumps({'prepared': len(records), 'total': len(rows)}), flush=True)


def auxiliary_loss(model, record):
    import torch
    import torch.nn.functional as functional
    from types import SimpleNamespace
    from .voice_action import action_embeddings
    controls = record.get('controls', [])
    if not controls:
        return torch.zeros((), device=model.device)
    tokenizer = model.processor.tokenizer
    decoder = SimpleNamespace(m=model.llm, tokenizer=tokenizer,
        embed_tokens=lambda ids: model.llm.get_input_embeddings()(torch.tensor(ids, device=model.device)))
    losses = []
    for control in controls:
        inputs = action_embeddings(decoder, record['audio_units'], control['query'])
        output = model.llm(inputs_embeds=inputs, use_cache=False, logits_to_keep=1, return_dict=True)
        ids = [tokenizer.encode(str(number), add_special_tokens=False) for number in range(control['count'])]
        if any(len(value) != 1 for value in ids) or not 0 <= control['label'] < control['count']:
            raise ValueError('辅助动作标签或分词器不兼容')
        logits = output.logits[:, -1, [value[0] for value in ids]].float()
        losses.append(functional.cross_entropy(logits, torch.tensor([control['label']], device=model.device)))
    return torch.stack(losses).mean()


def joint_loss(model, record, speech_weight=1.0, flow_weight=1.0):
    import torch
    import torch.nn.functional as functional

    device = model.device
    language_weight = record.get('language_weight', 1.0)
    if language_weight == 0 and not record['spans']:
        from types import SimpleNamespace
        result = SimpleNamespace(loss=torch.zeros((), device=device), hidden_states=[])
    else:
        result = model.llm(
            inputs_embeds=record['inputs_embeds'].to(device=device, dtype=model.llm.dtype).unsqueeze(0),
            labels=record['labels'].to(device).unsqueeze(0),
            output_hidden_states=bool(record['spans']), use_cache=False, return_dict=True,
        )
    losses = []
    flow_losses = []
    tts = model.tts
    for span in record['spans']:
        ids = span['ids'].to(tts.device)
        # 隐藏状态保留计算图，语音交叉熵同时更新主语言骨干和语音骨干。
        hidden = result.hidden_states[-1][:, span['start']:span['start'] + len(ids)].to(tts.device)
        semantic = functional.normalize(tts.projector_semantic(hidden), p=2, dim=-1)
        text = tts.emb_text(ids).unsqueeze(0) + semantic
        bos = tts.emb_text(torch.tensor([tts.audio_bos_token_id], device=tts.device)).unsqueeze(0)
        condition = torch.cat((text, bos), dim=1)
        codes = span['codes'].to(tts.device)
        preceding = tts.emb_code[0](codes[:-1]).unsqueeze(0)
        inputs = torch.cat((condition, preceding), dim=1)
        hidden = tts.model(inputs_embeds=inputs, use_cache=False, return_dict=True).last_hidden_state
        logits = tts.head_code[0](hidden[:, condition.shape[1] - 1:]).float()
        losses.append(functional.cross_entropy(logits.reshape(-1, logits.shape[-1]), codes))
        if 'flow' in span:
            from .native_flow import flow_loss
            flow_losses.append(flow_loss(tts.audio_tokenizer.flow.decoder, span['flow']))
    speech = torch.stack(losses).mean() if losses else result.loss * 0
    flow = torch.stack(flow_losses).mean() if flow_losses else result.loss * 0
    controls = auxiliary_loss(model, record)
    language = result.loss * language_weight
    return language + speech_weight * speech + flow_weight * flow + controls, language, speech, flow, controls


def train(args, config):
    import torch
    from peft import LoraConfig, TaskType, get_peft_model, get_peft_model_state_dict, set_peft_model_state_dict

    index = json.loads((args.prepared / 'index.json').read_text())
    from .acoustic import language_adapter_digest
    if index['language_initialization_sha256'] != language_adapter_digest(config):
        raise ValueError('编译特征与语言初始化权重不匹配')
    if not index['complete'] or index['base_revision'] != config['model']['revision']:
        raise ValueError('特征编译不完整或模型版本不符')
    training = [r for r in index['records'] if r['split'] == 'train']
    evaluation = [r for r in index['records'] if r['split'] == 'eval']
    if not training or not evaluation:
        raise ValueError('训练集与验证集均需提供')
    args.output.mkdir(parents=True, exist_ok=True)
    import fcntl
    process_lock = (args.output / '.train.lock').open('w')
    try:
        fcntl.flock(process_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        process_lock.close()
        raise RuntimeError('同一输出目录已有训练进程') from exc
    random.seed(42)
    torch.manual_seed(42)
    model = load_base(config)
    model.init_tts()
    model.requires_grad_(False)
    model.tts.audio_tokenizer.flow.requires_grad_(False)
    model.llm = get_peft_model(model.llm, LoraConfig(task_type=TaskType.CAUSAL_LM, r=8,
                              lora_alpha=16, target_modules=['q_proj', 'v_proj'], lora_dropout=0.05))
    model.tts.model = get_peft_model(model.tts.model, LoraConfig(task_type=TaskType.FEATURE_EXTRACTION,
                                    r=8, lora_alpha=16, target_modules=['q_proj', 'v_proj'], lora_dropout=0.05))
    model.tts.projector_semantic.requires_grad_(True)
    flow = model.tts.audio_tokenizer.flow
    flow.decoder.estimator = get_peft_model(flow.decoder.estimator, LoraConfig(
        r=8, lora_alpha=16, target_modules=['to_q', 'to_v'], lora_dropout=0.0))
    model.llm.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
    parameters = [p for p in model.parameters() if p.requires_grad]
    included = {id(p) for p in parameters}
    parameters += [p for p in flow.parameters() if p.requires_grad and id(p) not in included]
    optimizer_groups = parameters
    if args.language_learning_rate is not None or args.flow_learning_rate is not None:
        language_parameters = [p for p in model.llm.parameters() if p.requires_grad]
        flow_parameters = [p for p in flow.parameters() if p.requires_grad]
        separated = {id(p) for p in language_parameters + flow_parameters}
        acoustic_parameters = [p for p in parameters if id(p) not in separated]
        optimizer_groups = [
            {'params': language_parameters, 'lr': args.language_learning_rate or args.learning_rate},
            {'params': acoustic_parameters, 'lr': args.learning_rate},
            {'params': flow_parameters, 'lr': args.flow_learning_rate or args.learning_rate},
        ]
    optimizer = torch.optim.AdamW(optimizer_groups, lr=args.learning_rate)
    signature_data = {'inputs': index['signature'], 'lr': args.learning_rate,
                                          'speech_weight': args.speech_weight, 'flow_weight': args.flow_weight,
                                          'version': 2}
    if args.language_learning_rate is not None or args.flow_learning_rate is not None:
        signature_data.update(language_lr=args.language_learning_rate, flow_lr=args.flow_learning_rate)
    if index.get('auxiliary_tasks'):
        signature_data['auxiliary_version'] = 1
    signature = hashlib.sha256(json.dumps(signature_data).encode()).hexdigest()
    step = 0
    initialization = None
    parent = args.resume or args.initialize
    if parent:
        state = torch.load(parent / 'state.pt', map_location='cpu', weights_only=True)
        metadata = json.loads((parent / 'native_metadata.json').read_text())
        if metadata['base_revision'] != index['base_revision'] or metadata['language_initialization_sha256'] != index['language_initialization_sha256']:
            raise ValueError('初始化权重与当前基座或语言初始化不一致')
        if args.resume and state['signature'] != signature:
            raise ValueError('断点与当前数据或训练参数不符')
        set_peft_model_state_dict(model.llm, state['llm'])
        set_peft_model_state_dict(model.tts.model, state['tts'])
        model.tts.projector_semantic.load_state_dict(state['projector'])
        set_peft_model_state_dict(flow.decoder.estimator, state['flow'])
        initialization = metadata.get('initialized_from')
        if args.resume:
            optimizer.load_state_dict(state['optimizer'])
            random.setstate(state['python_rng'])
            torch.set_rng_state(state['torch_rng'])
            torch.cuda.set_rng_state(state['cuda_rng'], device=model.device)
            step = state['step']
        else:
            initialization = {'checkpoint': parent.name, 'signature': state['signature']}
    stopping = False
    def stop(signum, frame):
        nonlocal stopping
        stopping = True
    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    def load(row):
        record = torch.load(args.prepared / row['file'], map_location='cpu', weights_only=True)
        if record['signature'] != index['signature']:
            raise ValueError('样本指纹与清单不符')
        return record
    def save():
        target = args.output / f'step-{step:07d}'
        temporary = args.output / f'.step-{step:07d}.partial'
        if target.exists() or temporary.exists():
            raise FileExistsError('检查点已存在，请保留旧结果')
        temporary.mkdir()
        model.llm.save_pretrained(temporary / 'llm_lora', safe_serialization=True)
        model.tts.model.save_pretrained(temporary / 'tts_lora', safe_serialization=True)
        flow.decoder.estimator.save_pretrained(temporary / 'flow_lora', safe_serialization=True)
        torch.save(model.tts.projector_semantic.state_dict(), temporary / 'projector.pt')
        atomic_json(temporary / 'native_metadata.json', {
            'base_revision': index['base_revision'], 'training_signature': signature,
            'step': step, 'architecture': 'native_e2e', 'capabilities_verified': False,
            'frozen': ['audio_encoder', 'input_embeddings', 'flow_encoder', 'hifigan'],
            'language_initialization_sha256': index['language_initialization_sha256'],
            'initialized_from': initialization,
        })
        torch.save({'step': step, 'signature': signature,
                    'llm': get_peft_model_state_dict(model.llm),
                    'tts': get_peft_model_state_dict(model.tts.model),
                    'flow': get_peft_model_state_dict(flow.decoder.estimator),
                    'projector': model.tts.projector_semantic.state_dict(),
                    'optimizer': optimizer.state_dict(), 'python_rng': random.getstate(),
                    'torch_rng': torch.get_rng_state(),
                    'cuda_rng': torch.cuda.get_rng_state(model.device)}, temporary / 'state.pt')
        temporary.rename(target)
        return target
    while step < args.steps and not stopping:
        model.llm.train()
        model.tts.model.train()
        flow.decoder.estimator.train()
        optimizer.zero_grad(set_to_none=True)
        loss, language, speech, flow_value, controls = joint_loss(model, load(random.choice(training)), args.speech_weight, args.flow_weight)
        if not torch.isfinite(loss):
            raise RuntimeError('联合损失无效，停止更新')
        loss.backward()
        torch.nn.utils.clip_grad_norm_(parameters, 1.0, error_if_nonfinite=True)
        optimizer.step()
        step += 1
        report = {'step': step, 'loss': float(loss.detach()),
                  'language_loss': float(language.detach()), 'speech_loss': float(speech.detach()),
                  'flow_loss': float(flow_value.detach()), 'control_loss': float(controls.detach())}
        if step % args.save_every == 0 or step == args.steps or stopping:
            # 先原子保存可恢复状态，再做耗时验证，关机信号到达时优先保住更新。
            report['checkpoint'] = str(save())
            model.llm.eval()
            model.tts.model.eval()
            flow.decoder.estimator.eval()
            with torch.no_grad(), torch.random.fork_rng(devices=[model.device]):
                torch.manual_seed(20261112)
                scores = []
                domain_scores = {}
                for row in evaluation:
                    if stopping:
                        break
                    values = torch.stack(joint_loss(model, load(row), args.speech_weight, args.flow_weight)).detach().cpu().tolist()
                    scores.append(values[0])
                    domain_scores.setdefault(row.get('domain', 'dialogue'), []).append(values)
                if len(scores) == len(evaluation):
                    report['eval_loss'] = sum(scores) / len(scores)
                    report['eval_domains'] = {}
                    for domain, items in domain_scores.items():
                        report['eval_domains'][domain] = {
                            key: sum(item[number] for item in items) / len(items)
                            for number, key in enumerate(('loss', 'language', 'speech', 'flow', 'control'))
                        }
                    atomic_json(args.output / f'step-{step:07d}' / 'metrics.json', report)
        print(json.dumps(report), flush=True)
    if step and not (args.output / f'step-{step:07d}').exists():
        save()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['prepare', 'train'])
    parser.add_argument('--config', type=Path)
    parser.add_argument('--manifest', type=Path)
    parser.add_argument('--prepared', type=Path, required=True)
    parser.add_argument('--output', type=Path, default=Path('runs/native-joint'))
    parser.add_argument('--resume', type=Path)
    parser.add_argument('--initialize', type=Path)
    parser.add_argument('--steps', type=int, default=1000)
    parser.add_argument('--save-every', type=int, default=50)
    parser.add_argument('--learning-rate', type=float, default=2e-5)
    parser.add_argument('--language-learning-rate', type=float)
    parser.add_argument('--flow-learning-rate', type=float)
    parser.add_argument('--speech-weight', type=float, default=1.0)
    parser.add_argument('--flow-weight', type=float, default=1.0)
    args = parser.parse_args()
    values = [args.speech_weight, args.flow_weight, args.learning_rate]
    values += [value for value in (args.language_learning_rate, args.flow_learning_rate) if value is not None]
    if args.steps < 1 or args.save_every < 1 or any(not math.isfinite(value) or value <= 0 for value in values):
        parser.error('训练步数、保存间隔和损失权重必须为正值')
    if args.resume and args.initialize:
        parser.error('--resume与--initialize不能同时使用')
    if args.action == 'prepare' and args.manifest is None:
        parser.error('编译需要 --manifest')
    config = native_config(load_config(args.config))
    if config['model'].get('native_adapter'):
        parser.error('续训请使用 --resume，初始化配置不要填写 native_adapter')
    if args.action == 'prepare':
        prepare(args, config)
    else:
        train(args, config)


if __name__ == '__main__':
    main()
