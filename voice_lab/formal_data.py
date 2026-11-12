"""恢复已预处理的完整录音，避免续训时重新转换输入。"""

import hashlib
import json
from pathlib import Path
import tarfile

from .config import ROOT
from .corpus import convert_corpus
from .data import load_manifest
from .download import atomic_json


def main():
    target = ROOT / "data/formal/full-corpus-v1"
    receipt = target / ".complete.json"
    if receipt.is_file():
        return
    if (target / "real_train.jsonl").is_file() and (target / "real_eval.jsonl").is_file():
        try:
            load_manifest(target / "real_train.jsonl")
            load_manifest(target / "real_eval.jsonl")
        except FileNotFoundError:
            pass
        else:
            atomic_json(receipt, {"complete": True})
            return
    shared = Path("/root/gpufree-share")
    marker = shared / "VoiceDuplexLab-完整中文训练输入.tar.sha256.json"
    if marker.is_file():
        metadata = json.loads(marker.read_text())
        if Path(metadata["file"]).name != metadata["file"]:
            raise ValueError("训练输入归档名称不正确")
        archive = shared / metadata["file"]
        checksum = hashlib.sha256()
        with archive.open("rb") as stream:
            for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
                checksum.update(chunk)
        if archive.stat().st_size != metadata["bytes"] or checksum.hexdigest() != metadata["sha256"]:
            raise ValueError("完整训练输入归档校验失败")
        with tarfile.open(archive) as source:
            for item in source.getmembers():
                parts = Path(item.name).parts
                if not item.isfile() or not parts or parts[0] != target.name or ".." in parts:
                    raise ValueError("完整训练输入包含异常路径")
            target.parent.mkdir(parents=True, exist_ok=True)
            source.extractall(target.parent, filter="data")
        print("完整训练输入已恢复", flush=True)
    else:
        convert_corpus(ROOT / "data/raw/smoothconv-full-2ch", target, max_windows=0)
    atomic_json(receipt, {"complete": True})


if __name__ == "__main__":
    main()
