#!/usr/bin/env python3
"""通过算力自由原生分块接口上传准备好的资源。"""

import argparse
import concurrent.futures
import fcntl
import getpass
import hashlib
import json
import math
import os
import sys
from pathlib import Path
import stat
import signal
import threading
import time
from urllib.parse import urlparse

import requests
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from voice_lab.resources import default_resource_directory

CHUNK_SIZE = 5 * 1024 * 1024
DEFAULT_BASE = "https://ah01.storage.gpufree.cn:8443"
PRINT_LOCK = threading.Lock()


class AuthenticationError(RuntimeError):
    pass


def log(message):
    with PRINT_LOCK:
        print(message, flush=True)


def md5_file(path):
    checksum = hashlib.md5()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            checksum.update(chunk)
    return checksum.hexdigest()


def atomic_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    temporary.replace(path)


def watch_existing(root):
    """已有上传任务时显示其日志，返回是否观察到了运行中的任务。"""
    root.mkdir(parents=True, exist_ok=True)
    with (root / ".upload-task.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            return False
        except BlockingIOError:
            pass
        log("已有上传任务正在运行，正在显示实时进度。按 Ctrl+C 关闭此窗口的进度查看。")
        path = root / "logs/manual-upload.log"
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a+", encoding="utf-8") as stream:
            stream.seek(0, os.SEEK_END)
            journal = root / "logs/api-upload-progress.json"
            if journal.is_file():
                progress = json.loads(journal.read_text(encoding="utf-8"))
                log(f"当前已完成 {progress['completed']}/{progress['expected']} 个文件，后续进度将在此显示。")
            while True:
                print(stream.read(), end="", flush=True)
                try:
                    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    print(stream.read(), end="", flush=True)
                    log("上传进程已结束，请查看上方结果。")
                    return True
                except BlockingIOError:
                    time.sleep(1)


def validate_base(value):
    parsed = urlparse(value)
    if (
        parsed.scheme != "https"
        or not parsed.hostname
        or not parsed.hostname.endswith(".storage.gpufree.cn")
        or parsed.port not in (None, 443, 8443)
        or parsed.username
        or parsed.password
        or parsed.query
        or parsed.fragment
    ):
        raise ValueError("上传地址需要是算力自由的 HTTPS 存储地址")
    return value.rstrip("/")


def authorization(args):
    if args.auth_pipe:
        path = Path(args.auth_pipe)
        information = path.stat()
        if (
            not stat.S_ISFIFO(information.st_mode)
            or information.st_uid != os.getuid()
            or information.st_mode & 0o077
        ):
            raise ValueError("认证通道需要为当前用户私有的命名管道")
        log("等待浏览器登录认证")
        with path.open(encoding="utf-8") as stream:
            value = json.loads(stream.readline())["authorization"]
        path.unlink(missing_ok=True)
    elif args.token_file:
        value = Path(args.token_file).expanduser().read_text(encoding="utf-8-sig")
    else:
        token_path = Path(args.resources).expanduser() / ".credentials/gpufree-token.txt"
        value = os.environ.get("GPUFREE_AUTHORIZATION")
        if not value and token_path.is_file():
            value = token_path.read_text(encoding="utf-8-sig")
        if not value:
            value = getpass.getpass(
                "粘贴当前平台 Authorization 登录令牌，输入过程隐藏："
            )
    value = value.strip()
    if not value:
        raise ValueError("登录令牌为空")
    if "\n" in value or "\r" in value:
        raise ValueError("令牌文件只能包含一行登录令牌")
    return value if value.lower().startswith("bearer ") else "Bearer " + value


class StorageClient:
    def __init__(self, base, credential, retries=5, session=None):
        self.base = validate_base(base)
        self.retries = retries
        self.session = session or requests.Session()
        self.session.headers.update(
            {
                "Authorization": credential,
                "Accept": "application/json, text/plain, */*",
                "Origin": "https://www.gpufree.cn",
                "Referer": "https://www.gpufree.cn/",
                "User-Agent": "VoiceDuplexLab-Uploader/1.0",
            }
        )

    def request(self, method, endpoint, **kwargs):
        for attempt in range(self.retries):
            try:
                response = self.session.request(
                    method,
                    self.base + "/api/v1/storage/" + endpoint,
                    timeout=(20, 180),
                    allow_redirects=False,
                    **kwargs,
                )
                if response.status_code in (401, 403):
                    raise AuthenticationError(
                        "平台登录失效或当前账户无上传权限，请更新 Authorization 后重跑"
                    )
                if response.status_code == 429:
                    if attempt + 1 == self.retries:
                        response.raise_for_status()
                    log("平台限流，20秒后重试")
                    time.sleep(20)
                    continue
                response.raise_for_status()
                result = response.json()
                if result.get("code") != 200:
                    raise RuntimeError(
                        f"平台返回业务错误 code={result.get('code')}，{str(result.get('message', result.get('msg', '')))[:160]}"
                    )
                return result.get("data")
            except AuthenticationError:
                raise
            except (requests.RequestException, ValueError, RuntimeError) as exc:
                if attempt + 1 == self.retries:
                    raise
                delay = min(2 ** (attempt + 1), 20)
                log(f"请求遇到 {type(exc).__name__}，{delay}秒后重试")
                time.sleep(delay)
        raise RuntimeError("请求重试结束")

    def files(self, directory=""):
        values = self.request("GET", "ls", params={"path": "/" + directory.strip("/")})
        if not isinstance(values, list):
            raise ValueError("平台文件列表格式发生变化")
        return {item["name"]: item for item in values if not item.get("is_dir")}

    def completed_chunks(self, checksum):
        values = self.request("GET", "file_check", params={"md5": checksum})
        if not isinstance(values, list):
            raise ValueError("平台断点查询格式发生变化")
        return {int(value) for value in values}

    def send_chunk(self, name, data, index, total, directory, checksum):
        # 平台 md5 字段对应整个文件，file_md5 字段对应当前分块。
        fields = [
            ("file", ("blob", data, "application/octet-stream")),
            ("chunk_offset", (None, str(index))),
            ("chunks", (None, str(total))),
            ("dir", (None, directory)),
            ("file_name", (None, name)),
            ("md5", (None, checksum)),
            ("file_md5", (None, hashlib.md5(data).hexdigest())),
        ]
        self.request("POST", "upload", files=fields)

    def upload(self, path, name, directory=""):
        size = path.stat().st_size
        checksum = md5_file(path)
        complete = self.completed_chunks(checksum)
        chunks = max(1, math.ceil(size / CHUNK_SIZE))
        reused = sum(1 for index in range(chunks) if index in complete)
        log(f"上传 {name}，{size / 1024**2:.1f} MiB，共{chunks}块，复用{reused}块")
        previous_progress = -1
        with path.open("rb") as stream:
            for index in range(chunks):
                if index in complete:
                    stream.seek(CHUNK_SIZE, 1)
                else:
                    data = stream.read(CHUNK_SIZE)
                    if not data and size:
                        raise ValueError("本地文件在上传期间发生变化")
                    self.send_chunk(name, data, index, chunks, directory, checksum)
                progress = (index + 1) * 10 // chunks
                if progress != previous_progress:
                    log(f"{name}  {min(100, progress * 10)}%")
                    previous_progress = progress
        if path.stat().st_size != size:
            raise ValueError("本地文件在上传期间发生变化")
        for attempt in range(10):
            entry = self.files(directory).get(name)
            if entry and entry["size"] == size:
                return {
                    "name": name,
                    "bytes": size,
                    "md5": checksum,
                    "verified_size": True,
                }
            time.sleep(2)
        raise RuntimeError(f"上传后文件大小未匹配 {name}，重新运行可查询断点并继续")


def upload_plan(root, code_archive=None):
    values = {}
    browser_groups = set()
    for manifest_path in sorted(
        (root / "upload-web").glob("*/*.browser.manifest.json")
    ):
        manifest = json.loads(manifest_path.read_text())
        group = manifest["group"]
        browser_groups.add(group)
        values[manifest_path.name] = manifest_path
        for part in manifest["parts"]:
            name = part["name"] + ".zip"
            if name in (".", "..") or Path(name).name != name:
                raise ValueError("分卷文件名格式错误")
            values[name] = manifest_path.parent / name
    for manifest_path in sorted((root / "upload").glob("*/*.manifest.json")):
        manifest = json.loads(manifest_path.read_text())
        if manifest["group"] in browser_groups:
            continue
        values[manifest_path.name] = manifest_path
        for part in manifest["parts"]:
            name = part["name"]
            if name in (".", "..") or Path(name).name != name:
                raise ValueError("分卷文件名格式错误")
            values[name] = manifest_path.parent / name
    if code_archive:
        values[code_archive.name] = code_archive
    if not values:
        raise ValueError("没有找到完成打包的资源清单")
    for name, path in values.items():
        if not path.is_file():
            raise FileNotFoundError(f"清单对应文件缺失 {path}")
    return values


def main():
    parser = argparse.ArgumentParser(
        description="自动分块上传训练资源，跳过完整文件并复用平台断点"
    )
    parser.add_argument(
        "--resources", default=default_resource_directory()
    )
    parser.add_argument("--base-url", default=DEFAULT_BASE)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--auth-pipe", help=argparse.SUPPRESS)
    parser.add_argument("--token-file", help="从本地文本文件读取登录令牌")
    parser.add_argument("--file", help="单文件验证入口")
    parser.add_argument("--remote-name")
    parser.add_argument("--code-archive")
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument("--watch-existing", action="store_true", help="显示已运行任务的实时日志")
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("workers 必须为正整数")
    root = Path(args.resources).expanduser().resolve()
    if args.watch_existing:
        try:
            raise SystemExit(0 if watch_existing(root) else 3)
        except KeyboardInterrupt:
            return
    if args.file:
        path = Path(args.file).expanduser().resolve()
        plan = {args.remote_name or path.name: path}
    else:
        code = Path(args.code_archive).resolve() if args.code_archive else None
        plan = upload_plan(root, code)
    if any(name in (".", "..") or Path(name).name != name for name in plan):
        parser.error("远端文件名需要使用文件名本身")
    total = sum(path.stat().st_size for path in plan.values())
    log(f"上传计划共{len(plan)}个文件，合计{total / 1024**3:.2f} GiB")
    if total > 20 * 1024**3:
        log("平台共享文件免费额度为20GB，500GB是容量上限；本次全量计划超出免费额度，将按平台当日峰值产生存储费。")
    if args.plan_only:
        for name in plan:
            log(name)
        return
    root.mkdir(parents=True, exist_ok=True)

    def stop(signum, frame):
        log("上传已停止，平台上的完整分块会在下次运行时复用。")
        os._exit(130)

    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)
    with (root / ".upload-task.lock").open("w") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            parser.error("当前资源目录已有自动上传任务")
        credential = authorization(args)
        client = StorageClient(args.base_url, credential)
        existing = client.files()
        results = []
        pending = []
        for name, path in plan.items():
            if name in existing and existing[name]["size"] == path.stat().st_size:
                results.append(
                    {
                        "name": name,
                        "bytes": path.stat().st_size,
                        "skipped_complete": True,
                    }
                )
                log(f"复用已完整上传文件 {name}")
            else:
                pending.append((name, path))
        journal = root / "logs/api-upload-progress.json"

        def save(status="uploading"):
            atomic_json(
                journal,
                {
                    "status": status,
                    "expected": len(plan),
                    "completed": len(results),
                    "files": results,
                },
            )

        save()

        def send(item):
            name, path = item
            worker = StorageClient(args.base_url, credential)
            try:
                return worker.upload(path, name)
            finally:
                worker.session.close()

        executor = concurrent.futures.ThreadPoolExecutor(max_workers=args.workers)
        futures = [executor.submit(send, item) for item in pending]
        try:
            for future in concurrent.futures.as_completed(futures):
                results.append(future.result())
                save()
                log(f"文件完成 {len(results)}/{len(plan)}")
        except BaseException:
            for future in futures:
                future.cancel()
            save("failed")
            raise
        finally:
            executor.shutdown(wait=True, cancel_futures=True)
        final = client.files()
        missing = [
            name
            for name, path in plan.items()
            if name not in final or final[name]["size"] != path.stat().st_size
        ]
        if missing:
            raise RuntimeError("最终文件核对发现缺失或大小异常 " + ", ".join(missing))
        save("complete")
        client.session.close()
        log(f"全部{len(plan)}个资源文件上传完成，数量和字节大小核对通过。")


if __name__ == "__main__":
    main()
