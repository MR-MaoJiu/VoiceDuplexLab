"""统一原生声码器的教师强制流匹配，普通语句与情绪语句共用权重。"""
import numpy as np


def prepare_flow(engine, reference_audio, audio, codes):
    import torch
    import torch.nn.functional as functional
    import torchaudio.compliance.kaldi as kaldi
    import s3tokenizer
    from scipy.signal import resample_poly
    from stepaudio2.flashcosyvoice.utils.audio import mel_spectrogram

    device = next(engine.flow.parameters()).device
    mels = s3tokenizer.log_mel_spectrogram(torch.from_numpy(reference_audio))
    mels, lengths = s3tokenizer.padding([mels])
    reference_codes, sizes = engine.audio_tokenizer.quantize(mels.to(device), lengths.to(device))
    reference_codes = reference_codes[0, :int(sizes[0])].long()
    features = kaldi.fbank(torch.from_numpy(reference_audio).unsqueeze(0), num_mel_bins=80,
                          dither=0, sample_frequency=16000)
    features -= features.mean(dim=0, keepdim=True)
    vector = engine.spk_model.run(None, {engine.spk_model.get_inputs()[0].name: features.unsqueeze(0).numpy()})[0]
    speaker = engine.flow.spk_embed_affine_layer(functional.normalize(torch.from_numpy(vector).to(device), dim=1))
    def mel(wave, frames):
        wave = torch.from_numpy(resample_poly(wave, 3, 2).astype(np.float32)).to(device).unsqueeze(0)
        value = mel_spectrogram(wave)
        if value.shape[-1] < frames:
            value = functional.pad(value, (0, frames - value.shape[-1]), mode='replicate')
        return value[:, :, :frames]
    reference_mel = mel(reference_audio, len(reference_codes) * 2)
    target_mel = mel(audio, len(codes) * 2)
    tokens = torch.cat((reference_codes, codes.to(device))).unsqueeze(0).long()
    encoded, _ = engine.flow.encoder(engine.flow.input_embedding(tokens), torch.tensor([tokens.shape[1]], device=device))
    mu = engine.flow.encoder_proj(encoded).transpose(1, 2)
    target = torch.cat((reference_mel, target_mel), dim=2)
    condition = torch.zeros_like(target)
    condition[:, :, :reference_mel.shape[-1]] = reference_mel
    mask = torch.ones(1, 1, target.shape[-1], device=device)
    loss_mask = mask.clone()
    loss_mask[:, :, :reference_mel.shape[-1]] = 0
    return {key: value.cpu() for key, value in {
        'mu': mu, 'target': target, 'condition': condition, 'speaker': speaker,
        'mask': mask, 'loss_mask': loss_mask,
    }.items()}


def flow_loss(decoder, record):
    import torch
    device = next(decoder.parameters()).device
    row = {key: value.to(device) for key, value in record.items()}
    target = row['target']
    noise = torch.randn_like(target)
    time = torch.rand((1, 1, 1), device=device)
    sigma = float(getattr(decoder, 'sigma_min', 1e-6))
    sample = (1 - (1 - sigma) * time) * noise + time * target
    prediction = decoder.estimator(x=sample, mask=row['mask'], mu=row['mu'],
                                   t=time.reshape(1), spks=row['speaker'], cond=row['condition'])
    error = (prediction - (target - (1 - sigma) * noise)).square() * row['loss_mask']
    return error.sum() / (row['loss_mask'].sum() * target.shape[1])


def load_flow_weights(model, config):
    from peft import PeftModel
    from .config import local_path
    path = config['model'].get('native_adapter')
    if path and (local_path(path) / 'flow_lora').is_dir():
        flow = model.tts.audio_tokenizer.flow
        flow.decoder.estimator = PeftModel.from_pretrained(
            flow.decoder.estimator, str(local_path(path) / 'flow_lora')
        ).merge_and_unload()
