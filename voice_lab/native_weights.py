"""同一基座内的联合适配权重，装入后合并到原生生成路径。"""
import json

from .acoustic import language_adapter_digest


def load_native_weights(model, config, path):
    import torch
    from peft import PeftModel

    metadata = json.loads((path / 'native_metadata.json').read_text())
    if metadata['base_revision'] != config['model']['revision'] or metadata['architecture'] != 'native_e2e':
        raise ValueError('联合权重与原生基座不匹配')
    if metadata['language_initialization_sha256'] != language_adapter_digest(config):
        raise ValueError('联合权重依赖的语言初始化权重不匹配')
    model.llm = PeftModel.from_pretrained(model.llm, str(path / 'llm_lora')).merge_and_unload()
    model.tts.model = PeftModel.from_pretrained(model.tts.model, str(path / 'tts_lora')).merge_and_unload()
    projector = torch.load(path / 'projector.pt', map_location='cpu', weights_only=True)
    model.tts.projector_semantic.load_state_dict(projector)
