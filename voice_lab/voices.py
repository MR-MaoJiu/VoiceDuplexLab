import hashlib
import hmac
import io
import json
import math
import os
import uuid
import numpy as np
import soundfile as sf
from .config import local_path


def principal(config, authorization):
    token = (
        authorization.removeprefix("Bearer ")
        if authorization.startswith("Bearer ")
        else ""
    )
    users = config["server"].get("users", {})
    if users:
        for user, environment in users.items():
            expected = os.environ.get(environment, "")
            if expected and hmac.compare_digest(token, expected):
                return user
        return None
    expected = os.environ.get("VOICE_SERVER_TOKEN", config["server"].get("token", ""))
    if expected:
        return "default" if hmac.compare_digest(token, expected) else None
    return "local"


class VoiceStore:
    """参考音频只按用户和随机音色 ID 存取，客户端无法指定文件路径。"""

    def __init__(self, directory):
        self.directory = local_path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)

    def user_dir(self, user):
        return self.directory / hashlib.sha256(user.encode()).hexdigest()[:24]

    def put(self, user, body, name):
        if len(body) > 20_000_000:
            raise ValueError("参考音频文件超过 20 MB")
        stream = io.BytesIO(body)
        info = sf.info(stream)
        if not 0 < info.duration <= 60 or info.channels > 8 or info.samplerate > 192000:
            raise ValueError(
                "参考音频需要在 60 秒以内、8 声道以内且采样率不超过 192 kHz"
            )
        stream.seek(0)
        audio, rate = sf.read(stream, dtype="float32", always_2d=True)
        audio = audio.mean(axis=1)
        if not np.isfinite(audio).all() or np.max(np.abs(audio)) < 1e-5:
            raise ValueError("参考音频没有有效声音")
        if rate != 16000:
            from scipy.signal import resample_poly

            divisor = math.gcd(rate, 16000)
            audio = resample_poly(audio, 16000 // divisor, rate // divisor)
        identifier = uuid.uuid4().hex
        directory = self.user_dir(user)
        directory.mkdir(parents=True, exist_ok=True)
        target = directory / f"{identifier}.wav"
        temporary = directory / f"{identifier}.tmp"
        sf.write(temporary, audio, 16000, format="WAV", subtype="PCM_16")
        temporary.replace(target)
        metadata = {
            "id": identifier,
            "name": str(name)[:80],
            "seconds": round(len(audio) / 16000, 3),
            "sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
        }
        metadata_path = target.with_suffix(".json")
        metadata_path.write_text(
            json.dumps(metadata, ensure_ascii=False), encoding="utf-8"
        )
        return metadata

    def get(self, user, identifier):
        if len(identifier) != 32 or any(
            char not in "0123456789abcdef" for char in identifier
        ):
            raise FileNotFoundError("音色不存在")
        path = self.user_dir(user) / f"{identifier}.wav"
        if not path.is_file():
            raise FileNotFoundError("音色不存在或不属于当前用户")
        return path

    def list(self, user):
        return [
            json.loads(path.read_text(encoding="utf-8"))
            for path in sorted(self.user_dir(user).glob("*.json"))
        ]
