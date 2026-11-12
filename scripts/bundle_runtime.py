"""从本地归档生成可迁移的网页推理配置。"""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from voice_lab.config import ROOT, load_config
from voice_lab.pipeline import save_yaml


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("bundle", type=Path)
    parser.add_argument('--native-checkpoint', type=Path, help='显式选择待验收的原生联合权重目录')
    args = parser.parse_args()
    bundle = args.bundle.resolve()
    config_path = ROOT / "configs/bundle-runtime.yaml"
    config = load_config(config_path if config_path.is_file() else ROOT / "configs/default.yaml")
    metadata_path = bundle / "PROVENANCE.json"
    metadata = json.loads(metadata_path.read_text()) if metadata_path.is_file() else {}
    if metadata.get('runtime_config'):
        config = load_config(bundle / metadata['runtime_config'])
    adapter = metadata.get("active_adapter", "artifacts/adapter-step0001133")
    config["model"].update(
        path=str(bundle / "models/MiniCPM-o-4_5"),
        adapter=str(bundle / adapter),
        decode_mode="greedy",
    )
    config['model']['architecture'] = metadata.get('architecture',
        'legacy_v6' if metadata.get('active_acoustic_adapter') else 'native_e2e')
    config['model'].pop('native_adapter', None)
    if config['model']['architecture'] == 'native_e2e':
        config['model'].pop('acoustic_adapter', None)
        config['model'].pop('vocoder_adapter', None)
    if metadata.get('active_native_adapter'):
        native = bundle / metadata['active_native_adapter']
        if not native.is_dir():
            raise FileNotFoundError(native)
        config['model']['native_adapter'] = str(native)
    if args.native_checkpoint:
        native = args.native_checkpoint
        if not native.is_absolute():
            native = bundle / native
        if not (native / 'native_metadata.json').is_file():
            raise FileNotFoundError(native)
        config['model']['architecture'] = 'native_e2e'
        config['model'].pop('acoustic_adapter', None)
        config['model'].pop('vocoder_adapter', None)
        config['model']['native_adapter'] = str(native)
    if metadata.get('active_acoustic_adapter'):
        acoustic = bundle / metadata['active_acoustic_adapter']
        if not acoustic.is_dir():
            raise FileNotFoundError(acoustic)
        config['model']['acoustic_adapter'] = str(acoustic)
    if metadata.get('active_vocoder_adapter'):
        vocoder = bundle / metadata['active_vocoder_adapter']
        if not vocoder.is_dir():
            raise FileNotFoundError(vocoder)
        config['model']['vocoder_adapter'] = str(vocoder)
    if config.get('asr', {}).get('enabled'):
        config['asr']['model_dir'] = str(bundle / 'models/SenseVoice-int8')
    config["backend"]["weights_path"] = str(bundle / "models/Qwen3-8B")
    config["backend"]["knowledge_dir"] = str(ROOT / 'data/knowledge')
    config["server"]["voice_store"] = str(bundle / "local/user_voices")
    for key in ("path", "adapter"):
        if not Path(config["model"][key]).is_dir():
            raise FileNotFoundError(config["model"][key])
    save_yaml(config_path, config)
    print(config_path)


if __name__ == "__main__":
    main()
