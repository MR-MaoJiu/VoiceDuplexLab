import argparse
import os
from pathlib import Path
import httpx


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("file")
    parser.add_argument("--name", default="我的音色")
    parser.add_argument("--url", default="http://127.0.0.1:8765")
    parser.add_argument("--token-env", default="VOICE_TOKEN")
    args = parser.parse_args()
    token = os.environ.get(args.token_env, "")
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    path = Path(args.file)
    if path.stat().st_size > 20_000_000:
        raise ValueError("参考音频文件超过 20 MB")
    response = httpx.post(
        args.url.rstrip("/") + "/voices",
        params={"name": args.name},
        content=path.read_bytes(),
        headers=headers,
        timeout=60,
    )
    response.raise_for_status()
    result = response.json()
    print(f"音色已保存，ID 为 {result['id']}，时长 {result['seconds']} 秒")
    print(f"启动客户端时添加 --voice {result['id']}")


if __name__ == "__main__":
    main()
