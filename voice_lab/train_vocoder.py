"""原生声码器的完整序列流匹配适配，仅用于明确的非语言发声动作。"""
import argparse
import json
from pathlib import Path
import random
import time
import tempfile
import numpy as np


def main():
    import torch
    import torch.nn.functional as functional
    import torchaudio.compliance.kaldi as kaldi
    from peft import LoraConfig, get_peft_model
    from scipy.signal import resample_poly
    from stepaudio2 import Token2wav
    from stepaudio2.flashcosyvoice.utils.audio import mel_spectrogram
    from .download import atomic_json
    from .model import read_audio
    parser = argparse.ArgumentParser()
    parser.add_argument('--base', required=True, type=Path)
    parser.add_argument('--data', required=True, type=Path)
    parser.add_argument('--features', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--device', type=int, default=1)
    parser.add_argument('--steps', type=int, default=300)
    parser.add_argument('--max-seconds', type=int, default=180)
    parser.add_argument('--resume', type=Path)
    parser.add_argument('--initialize', type=Path)
    args = parser.parse_args()
    torch.cuda.set_device(args.device)
    torch.manual_seed(20261112)
    random.seed(20261112)
    args.output.mkdir(parents=True, exist_ok=True)
    import fcntl
    process_lock = (args.output / '.train.lock').open('w')
    try:
        fcntl.flock(process_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        process_lock.close()
        raise RuntimeError('该输出目录已有声码器训练进程，请等待完成') from exc
    if args.resume and args.initialize:
        parser.error('--resume与--initialize不能同时使用')
    if (args.output/'latest.json').exists() and not args.resume:
        raise FileExistsError('请保留现有实验，选择新的输出目录')
    engine = Token2wav(str(args.base/'assets/token2wav'))
    engine.flow.eval()
    for parameter in engine.flow.parameters():
        parameter.requires_grad_(False)
    rows = [json.loads(line) for line in (args.data/'manifest.jsonl').read_text().splitlines()]
    index = json.loads((args.features/'index.json').read_text())
    locations = {row['id']: args.features/row['file'] for row in index['records']}
    references = {}
    for row in rows:
        if row['style'] == 'neutral':
            references.setdefault(row['speaker'], row)
    selected = []
    for style in ('laugh', 'cry'):
        for split, limit in (('train', 48), ('eval', 8)):
            selected.extend([row for row in rows if row['style']==style and row['split']==split and row['speaker'] in references][:limit])
    reference_cache = {}
    def mel(audio, frames):
        wave = torch.from_numpy(resample_poly(audio,3,2).astype(np.float32)).cuda().unsqueeze(0)
        value = mel_spectrogram(wave)
        if value.shape[-1] < frames:
            value = functional.pad(value,(0,frames-value.shape[-1]),mode='replicate')
        return value[:,:,:frames]
    prepared = []
    with torch.no_grad():
        for row in selected:
            reference = references[row['speaker']]
            if reference['id'] not in reference_cache:
                audio = read_audio(args.data/reference['audio'])
                codes = torch.load(locations[reference['id']],map_location='cpu',weights_only=True)['codes'][:-1].cuda()
                feat = kaldi.fbank(torch.from_numpy(audio).unsqueeze(0),num_mel_bins=80,dither=0,sample_frequency=16000)
                feat -= feat.mean(dim=0,keepdim=True)
                vector = engine.spk_model.run(None,{engine.spk_model.get_inputs()[0].name:feat.unsqueeze(0).numpy()})[0]
                vector = engine.flow.spk_embed_affine_layer(functional.normalize(torch.from_numpy(vector).cuda(),dim=1))
                reference_cache[reference['id']] = (codes,mel(audio,len(codes)*2),vector)
            reference_codes,reference_mel,speaker = reference_cache[reference['id']]
            codes = torch.load(locations[row['id']],map_location='cpu',weights_only=True)['codes'][:-1].cuda()
            target = mel(read_audio(args.data/row['audio']),len(codes)*2)
            tokens = torch.cat((reference_codes,codes)).unsqueeze(0).long()
            encoded,_ = engine.flow.encoder(engine.flow.input_embedding(tokens),torch.tensor([tokens.shape[1]],device='cuda'))
            mu = engine.flow.encoder_proj(encoded).transpose(1,2)
            target = torch.cat((reference_mel,target),dim=2)
            condition = torch.zeros_like(target)
            condition[:,:,:reference_mel.shape[-1]] = reference_mel
            mask = torch.ones(1,1,target.shape[-1],device='cuda')
            loss_mask = mask.clone()
            loss_mask[:,:,:reference_mel.shape[-1]] = 0
            prepared.append({'mu':mu.cpu(),'target':target.cpu(),'condition':condition.cpu(),
                             'speaker':speaker.cpu(),'mask':mask.cpu(),'loss_mask':loss_mask.cpu(),
                             'split':row['split'],'id':row['id'],'vocal_style':0 if row['style']=='laugh' else 1})
    temporary_inputs = args.output / '.prepared.partial'
    torch.save(prepared, temporary_inputs)
    temporary_inputs.replace(args.output / 'prepared.pt')
    engine.flow.decoder.estimator = get_peft_model(engine.flow.decoder.estimator,LoraConfig(
        r=8,lora_alpha=16,target_modules=['to_q','to_v'],lora_dropout=0.0))
    estimator = engine.flow.decoder.estimator
    style_embedding = torch.nn.Embedding(2,80).cuda()
    torch.nn.init.zeros_(style_embedding.weight)
    adapter_parameters = [parameter for parameter in estimator.parameters() if parameter.requires_grad]
    parameters = adapter_parameters + list(style_embedding.parameters())
    optimizer = torch.optim.AdamW([{'params':adapter_parameters,'lr':0.0001},
                                  {'params':style_embedding.parameters(),'lr':0.001}])
    start_step = 0
    parent = args.resume or args.initialize
    if parent:
        from safetensors.torch import load_file
        from peft import set_peft_model_state_dict
        metadata = json.loads((parent/'flow_metadata.json').read_text())
        if metadata['feature_signature'] != index['signature']:
            raise ValueError('声码器输入与原断点不匹配')
        set_peft_model_state_dict(estimator,load_file(str(parent/'flow_lora/adapter_model.safetensors'),device='cpu'))
        if (parent/'flow_style.pt').exists():
            style_embedding.load_state_dict(torch.load(parent/'flow_style.pt',map_location='cpu',weights_only=True))
        if args.resume:
            state = torch.load(parent/'state.pt',map_location='cpu',weights_only=True)
            optimizer.load_state_dict(state['optimizer'])
            random.setstate(state['python_rng'])
            torch.set_rng_state(state['torch_rng'])
            torch.cuda.set_rng_state(state['cuda_rng'])
            start_step = state['step']
    training = [row for row in prepared if row['split']=='train']
    evaluation = [row for row in prepared if row['split']=='eval']
    def loss(row, generator=None):
        target = row['target'].cuda()
        noise = torch.randn(target.shape,device='cuda',generator=generator)
        t = torch.rand((1,1,1),device='cuda',generator=generator)
        sigma = float(getattr(engine.flow.decoder,'sigma_min',1e-6))
        x = (1-(1-sigma)*t)*noise + t*target
        offset = style_embedding(torch.tensor([row['vocal_style']],device='cuda')).unsqueeze(2)
        mu = row['mu'].cuda() + offset*row['loss_mask'].cuda()
        prediction = estimator(x=x,mask=row['mask'].cuda(),mu=mu,
                               t=t.reshape(1),spks=row['speaker'].cuda(),cond=row['condition'].cuda())
        difference = (prediction-(target-(1-sigma)*noise)).square()*row['loss_mask'].cuda()
        return difference.sum()/(row['loss_mask'].sum().cuda()*target.shape[1])
    def evaluate():
        estimator.eval()
        generator = torch.Generator(device='cuda').manual_seed(20261112)
        with torch.no_grad():
            values = [float(loss(row,generator)) for row in evaluation]
        estimator.train()
        return sum(values)/len(values)
    initial = evaluate()
    print(json.dumps({'initial_eval_flow_loss':initial,'train':len(training),'eval':len(evaluation)}),flush=True)
    started = time.monotonic()
    for step in range(start_step+1,args.steps+1):
        optimizer.zero_grad(set_to_none=True)
        value = loss(random.choice(training))
        if not torch.isfinite(value):
            raise ValueError('流匹配损失出现非有限值')
        value.backward()
        torch.nn.utils.clip_grad_norm_(parameters,1.0)
        optimizer.step()
        timed_out = time.monotonic()-started>=args.max_seconds
        if step%50==0 or step==args.steps or timed_out:
            score = evaluate()
            checkpoint = args.output/f'step-{step:07d}'
            if checkpoint.exists():
                raise FileExistsError('断点已存在，请恢复最新断点或选择新的输出目录')
            temporary = Path(tempfile.mkdtemp(prefix=f'.step-{step:07d}-',dir=args.output))
            estimator.save_pretrained(temporary/'flow_lora',safe_serialization=True)
            torch.save(style_embedding.state_dict(),temporary/'flow_style.pt')
            torch.save({'optimizer':optimizer.state_dict(),'step':step,'python_rng':random.getstate(),
                        'torch_rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state()},temporary/'state.pt')
            atomic_json(temporary/'flow_metadata.json',{'step':step,'initial_eval_loss':initial,'eval_loss':score,
                'feature_signature':index['signature'],'scope':'原生声码器完整序列CFM；仅非语言动作启用，流式说话使用基座声码器',
                'flow_conditioning_version':2,'base_revision':index['base_revision'],
                'license':'SMIIP-NV / CC BY-NC-SA 4.0','acceptance_pending':True})
            temporary.replace(checkpoint)
            atomic_json(args.output/'latest.json',{'step':step,'checkpoint':checkpoint.name,'eval_loss':score})
            print(json.dumps({'step':step,'train_loss':float(value.detach()),'eval_loss':score,'elapsed':time.monotonic()-started}),flush=True)
        if timed_out:
            break


if __name__=='__main__':
    main()
