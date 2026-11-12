from pathlib import Path
import os
import yaml

ROOT = Path(__file__).resolve().parents[1]


def load_config(path=None):
    path = Path(path or os.environ.get("VOICE_CONFIG", ROOT / "configs/default.yaml"))
    with path.open(encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    return config


def local_path(value):
    path = Path(value).expanduser()
    return path if path.is_absolute() else ROOT / path
