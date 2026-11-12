"""原生语音解码器的情绪适配器，条件直接作用于端到端隐藏状态。"""
import hashlib
import json
from pathlib import Path

ACOUSTIC_STYLES = ('happy', 'sad', 'angry', 'laugh', 'cry', 'neutral')


def language_adapter_digest(config):
    from .config import local_path
    path = config['model'].get('adapter')
    if not path:
        return None
    return hashlib.sha256((local_path(path) / 'adapter_model.safetensors').read_bytes()).hexdigest()


def acoustic_style(expression, vocal=False):
    style = expression['style']
    if style == 'laugh':
        return 'laugh' if vocal else 'happy'
    if style == 'cry':
        return 'cry' if vocal else 'sad'
    if style == 'excited':
        return 'happy'
    if style == 'auto' and expression['emotion'] == 'happy' and expression.get('emotion_source') == 'self_report':
        return 'happy'
    return style if style in ACOUSTIC_STYLES and style != 'neutral' else None


def condition_with_style(condition, embedding, index, vocal_only=False):
    import torch
    offset = embedding(torch.tensor([index], device=condition.device)).to(condition.dtype).unsqueeze(1)
    if vocal_only:
        return torch.cat((offset, condition[:, -1:]), dim=1)
    # 保留最后的Audio BOS，只调整文本与语义条件。
    return torch.cat((condition[:, :-1] + offset, condition[:, -1:]), dim=1)


def load_acoustic_adapter(model, config, directory):
    import torch
    from peft import PeftModel
    directory = Path(directory)
    metadata = json.loads((directory / 'acoustic_metadata.json').read_text())
    model._voice_lab_acoustic_version = metadata.get('conditioning_version', 1)
    if metadata['base_revision'] != config['model']['revision'] or metadata['language_adapter_sha256'] != language_adapter_digest(config):
        raise ValueError('声学适配器的基座或语言适配器不匹配')
    if tuple(metadata['styles']) != ACOUSTIC_STYLES:
        raise ValueError('声学适配器的情绪编号不匹配')
    model.tts.model = PeftModel.from_pretrained(model.tts.model, directory / 'tts_lora').eval()
    embedding = torch.nn.Embedding(len(ACOUSTIC_STYLES), model.tts.config.hidden_size).to(model.tts.device)
    embedding.load_state_dict(torch.load(directory / 'style_embedding.pt', map_location=model.tts.device, weights_only=True))
    return embedding.eval()


def load_vocoder_adapter(model, config):
    import torch
    from peft import PeftModel
    from .config import local_path
    value = config['model'].get('vocoder_adapter')
    if not value:
        return None, None
    directory = local_path(value)
    metadata = json.loads((directory / 'flow_metadata.json').read_text())
    if metadata['base_revision'] != config['model']['revision']:
        raise ValueError('声码器适配器与语音基座不匹配')
    decoder = model.tts.audio_tokenizer.flow.decoder
    decoder.estimator = PeftModel.from_pretrained(decoder.estimator, directory / 'flow_lora').eval()
    embedding = torch.nn.Embedding(2, 80).to(model.tts.device)
    embedding.load_state_dict(torch.load(directory / 'flow_style.pt', map_location=model.tts.device, weights_only=True))
    return decoder.estimator, embedding.eval()
