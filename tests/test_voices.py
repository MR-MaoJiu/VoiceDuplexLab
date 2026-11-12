import io
import numpy as np
import pytest
import soundfile as sf
from voice_lab.voices import VoiceStore, principal


def wav(rate=22050):
    stream = io.BytesIO()
    t = np.arange(rate) / rate
    sf.write(stream, np.sin(2 * np.pi * 220 * t) * 0.1, rate, format="WAV")
    return stream.getvalue()


def test_voice_ownership_and_resampling(tmp_path):
    store = VoiceStore(tmp_path)
    result = store.put("alice", wav(), "爱丽丝")
    path = store.get("alice", result["id"])
    assert sf.info(path).samplerate == 16000
    assert len(store.list("alice")) == 1
    assert store.list("bob") == []
    with pytest.raises(FileNotFoundError):
        store.get("bob", result["id"])


def test_voice_rejects_path_traversal(tmp_path):
    with pytest.raises(FileNotFoundError):
        VoiceStore(tmp_path).get("alice", "../../etc/passwd")


def test_user_tokens(monkeypatch):
    config = {"server": {"users": {"alice": "ALICE_TOKEN", "bob": "BOB_TOKEN"}}}
    monkeypatch.setenv("ALICE_TOKEN", "alice-secret")
    assert principal(config, "Bearer alice-secret") == "alice"
    assert principal(config, "Bearer wrong") is None
    assert principal(config, "") is None


def test_new_reference_resets_model_audio_cache(monkeypatch):
    from voice_lab import model as module
    from voice_lab.config import load_config

    events = []

    class Duplex:
        def prepare(self, **kwargs):
            events.append(("prepare", kwargs["prompt_wav_path"]))

    class Model:
        def reset_session(self, **kwargs):
            events.append(("reset", kwargs["reset_token2wav_cache"]))

        def as_duplex(self, **kwargs):
            events.append(("create", None))
            return Duplex()

    monkeypatch.setattr(
        module, "read_audio", lambda path: np.ones(16000, dtype=np.float32)
    )
    module.prepare_duplex(Model(), load_config(), "alice.wav")
    module.prepare_duplex(Model(), load_config(), "bob.wav")
    assert events == [
        ("reset", True),
        ("create", None),
        ("prepare", "alice.wav"),
        ("reset", True),
        ("create", None),
        ("prepare", "bob.wav"),
    ]
