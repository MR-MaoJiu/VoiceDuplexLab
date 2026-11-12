#!/usr/bin/env python3
"""在本机续传固定版本训练资源，并生成供手动上传的分卷。"""

import argparse
import concurrent.futures
import fcntl
import os
from pathlib import Path
import shutil
import signal
import sys

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

from huggingface_hub import HfApi, snapshot_download
from voice_lab.download import atomic_json, download_dataset, retry
from voice_lab.resources import CATALOG, default_resource_directory, pack_group


def stop(signum, frame):
    print("\n下载已停止，原始下载断点保留。再次运行同一脚本即可继续。", flush=True)
    os._exit(130)


def model_files(root, group):
    item = CATALOG[group]
    information = retry(
        lambda: HfApi().model_info(
            item["repository"], revision=item["revision"], files_metadata=True
        )
    )
    folder = root / item["root"]
    total = sum(file.size or 0 for file in information.siblings)
    complete = sum(
        min((folder / file.rfilename).stat().st_size, file.size or 0)
        for file in information.siblings
        if (folder / file.rfilename).is_file()
    )
    partial = (
        sum(path.stat().st_size for path in folder.rglob("*.incomplete"))
        if folder.exists()
        else 0
    )
    return total, max(0, total - complete - partial)


def finish_package(root, group, part_mib):
    directory = root / "upload" / group
    manifest = directory / f"{group}.manifest.json"
    if not manifest.exists() and directory.exists():
        # 打包中断后的分卷是派生临时文件，原始模型及录音仍保存在资源目录。
        for path in directory.glob(f"{group}.tar*.part*"):
            if path.is_file():
                path.unlink()
    return pack_group(
        root, directory, group, part_mib * 1024**2, compress=group != "SmoothConv"
    )


def main():
    parser = argparse.ArgumentParser(
        description="本机下载训练资源，自动续传并生成手动上传分卷"
    )
    parser.add_argument(
        "--destination", default=default_resource_directory()
    )
    parser.add_argument("--clips", type=int, default=16)
    parser.add_argument("--part-mib", type=int, default=1024)
    parser.add_argument(
        "--download-only",
        action="store_true",
        help="完成下载后结束，稍后可重新运行进行打包",
    )
    args = parser.parse_args()
    if args.clips < 1 or args.part_mib < 1:
        parser.error("clips 和 part-mib 应为正整数")
    root = Path(args.destination).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)
    with (root / ".download-task.lock").open("w") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            parser.error("同一资源目录已有下载任务，请查看另一个终端")
        quantities = [model_files(root, group) for group in ("MiniCPM", "Qwen3")]
        total = sum(item[0] for item in quantities)
        remaining = sum(item[1] for item in quantities)
        required = remaining + (total if not args.download_only else 0) + 2 * 1024**3
        free = shutil.disk_usage(root).free
        print(f"资源目录 {root}", flush=True)
        print(
            f"两个模型合计约 {total / 1024**3:.2f} GiB，预计待下载约 {remaining / 1024**3:.2f} GiB",
            flush=True,
        )
        print(
            f"可用空间 {free / 1024**3:.1f} GiB，下载与打包预算 {required / 1024**3:.1f} GiB",
            flush=True,
        )
        if free < required:
            raise RuntimeError(
                "目标磁盘剩余空间不足，请使用 --destination 指定更大的磁盘"
            )

        def download_model(group):
            item = CATALOG[group]
            print(f"开始或继续下载 {item['repository']}", flush=True)
            path = retry(
                lambda: snapshot_download(
                    item["repository"],
                    revision=item["revision"],
                    local_dir=root / item["root"],
                    max_workers=3,
                )
            )
            receipt = {
                "repository": item["repository"],
                "revision": item["revision"],
                "path": str(path),
            }
            atomic_json(
                root / "models" / f"{Path(item['root']).name}.receipt.json", receipt
            )
            print(f"模型下载完成 {group}", flush=True)
            if not args.download_only:
                finish_package(root, group, args.part_mib)
                print(f"模型上传分卷已生成 {group}", flush=True)
            return receipt

        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
            futures = [
                executor.submit(download_model, group) for group in ("MiniCPM", "Qwen3")
            ]
            download_dataset(args.clips, 42, root / CATALOG["SmoothConv"]["root"])
            print(f"真实中文录音下载完成，共 {args.clips} 段", flush=True)
            if not args.download_only:
                finish_package(root, "SmoothConv", args.part_mib)
            models = [future.result() for future in futures]
        atomic_json(
            root / "resources.json",
            {
                "models": models,
                "dataset": str(root / CATALOG["SmoothConv"]["root"]),
                "clips": args.clips,
            },
        )
        print("\n本地资源准备完成。", flush=True)
        print(f"手动上传目录为 {root / 'upload'}", flush=True)
        print(
            "每组文件夹中的所有 part 分卷和 manifest.json 清单需要一并上传。",
            flush=True,
        )
        print("请以完整打包后生成的清单为准，分卷校验后即可在服务器恢复。", flush=True)


if __name__ == "__main__":
    main()
