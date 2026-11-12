import json
import numpy as np
import pytest
import soundfile as sf
from voice_lab.data import load_manifest


def write_manifest(tmp_path, units):
    sf.write(tmp_path / "audio.wav", np.zeros(16000), 16000)
    path = tmp_path / "train.jsonl"
    path.write_text(
        json.dumps(
            {"id": "one", "voice": "audio.wav", "audio": "audio.wav", "units": units}
        ),
        encoding="utf-8",
    )
    return path


def test_manifest_resolves_paths(tmp_path):
    path = write_manifest(tmp_path, [{"listen": True}, {"text": "你好", "end": True}])
    assert load_manifest(path)[0]["audio"] == str(tmp_path / "audio.wav")


def test_reject_unfinished_turn(tmp_path):
    path = write_manifest(tmp_path, [{"text": "你好", "end": False}, {"listen": True}])
    with pytest.raises(ValueError):
        load_manifest(path)


def test_reject_missing_voice(tmp_path):
    path = write_manifest(tmp_path, [{"listen": True}])
    (tmp_path / "audio.wav").unlink()
    with pytest.raises(FileNotFoundError):
        load_manifest(path)
