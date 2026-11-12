#!/usr/bin/env python3
"""在缺少 pip 时用标准库下载并校验固定版本 uv。"""

import hashlib
from pathlib import Path
import platform
import shutil
import tarfile
import time
import urllib.error
import urllib.request

VERSION = "0.9.10"


def fetch(url, target):
    for attempt in range(4):
        try:
            with urllib.request.urlopen(url, timeout=180) as response:
                with target.open("wb") as stream:
                    shutil.copyfileobj(response, stream)
            return
        except (urllib.error.URLError, TimeoutError) as exc:
            if attempt == 3:
                raise
            delay = 20 if getattr(exc, "code", None) == 429 else 2 ** (attempt + 1)
            print("下载 uv 遇到网络错误，{} 秒后重试".format(delay), flush=True)
            time.sleep(delay)


def main():
    architecture = {
        "x86_64": "x86_64",
        "amd64": "x86_64",
        "aarch64": "aarch64",
        "arm64": "aarch64",
    }.get(platform.machine().lower())
    if platform.system() != "Linux" or architecture is None:
        raise RuntimeError("此引导脚本适用于 Linux x86_64 或 aarch64")
    directory = Path(__file__).resolve().parents[1] / ".bootstrap"
    directory.mkdir(parents=True, exist_ok=True)
    name = "uv-{}-unknown-linux-gnu.tar.gz".format(architecture)
    url = "https://github.com/astral-sh/uv/releases/download/{}/{}".format(
        VERSION, name
    )
    archive = directory / name
    checksum = directory / (name + ".sha256")
    fetch(url + ".sha256", checksum)
    expected = checksum.read_text().split()[0]
    if len(expected) != 64 or any(char not in "0123456789abcdef" for char in expected):
        raise ValueError("uv 校验文件格式异常")
    fetch(url, archive)
    digest = hashlib.sha256()
    with archive.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    if digest.hexdigest() != expected:
        raise ValueError("uv 下载文件 SHA256 校验失败，请重新执行")
    destination = directory / "bin"
    destination.mkdir(exist_ok=True)
    with tarfile.open(archive, "r:gz") as bundle:
        for binary in ("uv", "uvx"):
            member = next(
                (
                    item
                    for item in bundle.getmembers()
                    if item.isfile() and Path(item.name).name == binary
                ),
                None,
            )
            if member is None:
                raise ValueError("uv 归档缺少可执行文件")
            temporary = destination / (binary + ".tmp")
            with bundle.extractfile(member) as source:
                with temporary.open("wb") as target:
                    shutil.copyfileobj(source, target)
            temporary.chmod(0o755)
            temporary.replace(destination / binary)
    archive.unlink()
    checksum.unlink()
    print("uv {} 已安装至 {}".format(VERSION, destination))


if __name__ == "__main__":
    main()
