"""下载扩充训练素材，复用已下载录音并生成独立上传分卷。"""

import argparse
import concurrent.futures
import fcntl
import json
import os
from pathlib import Path
import shutil

from .download import DATASET_REPO, DATASET_REVISION, atomic_json, retry
from .resources import CATALOG, default_resource_directory, digest_file, pack_group

TOOL_REPO = "Team-ACE/ToolACE"
TOOL_REVISION = "6bda777c88d21e5a204703c1ee45597a8fa4f734"


def inventory(info):
    files = {}
    for item in info.siblings:
        lfs = item.lfs
        checksum = (
            lfs.get("sha256") if isinstance(lfs, dict) else getattr(lfs, "sha256", None)
        )
        files[item.rfilename] = {"size": item.size or 0, "sha256": checksum}
    return files


def select_full_pairs(files):
    pairs, seen = [], set()
    for name in sorted(files):
        annotation = name[:-4] + ".json"
        if not name.endswith(".wav") or "_2ch/" not in name or annotation not in files:
            continue
        # 音频内容完全相同的文件只选一份，避免重复训练和重复下载。
        identity = files[name].get("sha256") or name
        if identity in seen:
            continue
        seen.add(identity)
        pairs.append({"audio": name, "annotation": annotation})
    if not pairs:
        raise ValueError("没有找到配套标注的双声道录音")
    return pairs


def valid_local(path, metadata):
    return (
        path.is_file()
        and path.stat().st_size == metadata["size"]
        and (not metadata.get("sha256") or digest_file(path) == metadata["sha256"])
    )


def reusable_file(source, destination, metadata):
    if valid_local(destination, metadata):
        return True
    if not valid_local(source, metadata):
        return False
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)
    return True


def download_file(repo, revision, directory, name, metadata, previous=None):
    from huggingface_hub import hf_hub_download

    path = directory / name
    if previous is not None and reusable_file(previous / name, path, metadata):
        return
    if valid_local(path, metadata):
        return
    downloaded = retry(
        lambda: hf_hub_download(
            repo,
            filename=name,
            repo_type="dataset",
            revision=revision,
            local_dir=directory,
        )
    )
    if not valid_local(Path(downloaded), metadata):
        raise ValueError(f"下载后的文件校验失败 {name}")


def download_smoothconv(root, files, pairs, workers):
    destination = root / CATALOG["SmoothConvFull"]["root"]
    previous = root / CATALOG["SmoothConv"]["root"]
    destination.mkdir(parents=True, exist_ok=True)
    download_file(
        DATASET_REPO, DATASET_REVISION, destination, "README.md", files["README.md"]
    )
    atomic_json(
        destination / "download_plan.json",
        {"repository": DATASET_REPO, "revision": DATASET_REVISION, "pairs": pairs},
    )

    def pair_download(pair):
        for key in ("annotation", "audio"):
            name = pair[key]
            download_file(
                DATASET_REPO, DATASET_REVISION, destination, name, files[name], previous
            )

    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(pair_download, pair) for pair in pairs]
        for index, future in enumerate(concurrent.futures.as_completed(futures), 1):
            future.result()
            if index == 1 or index % 10 == 0 or index == len(pairs):
                print(f"真实对话下载完成 {index}/{len(pairs)}", flush=True)
    atomic_json(
        destination / "download_manifest.json",
        {
            "repository": DATASET_REPO,
            "revision": DATASET_REVISION,
            "license": "CC-BY-NC-4.0",
            "profile": "full-2ch",
            "seed": 42,
            "pairs": pairs,
            "selection": "全部双声道、音频内容去重、配套人工标注",
        },
    )
    return destination


def download_toolace(root, files):
    destination = root / CATALOG["ToolACE"]["root"]
    destination.mkdir(parents=True, exist_ok=True)
    for name, metadata in files.items():
        if not name.startswith("."):
            download_file(TOOL_REPO, TOOL_REVISION, destination, name, metadata)
    data = json.loads((destination / "data.json").read_text(encoding="utf-8"))
    count = len(data) if isinstance(data, list) else None
    atomic_json(
        destination / "download_manifest.json",
        {
            "repository": TOOL_REPO,
            "revision": TOOL_REVISION,
            "license": "Apache-2.0",
            "type": "synthetic_tool_text",
            "records": count,
            "usage": "后续筛选中文指令并转换为语音委派、工具结果和口语整合训练样本",
        },
    )
    print(f"ToolACE 文本素材下载完成，记录数 {count}", flush=True)
    return destination


def pack_materials(root, group, part_mib):
    output = root / "upload" / group
    if not (output / f"{group}.manifest.json").exists() and output.exists():
        for path in output.glob(f"{group}.tar*.part*"):
            if path.is_file():
                path.unlink()
    return pack_group(root, output, group, part_mib * 1024**2, compress=True)


def stop(signum, frame):
    print("\n素材下载已停止，再次运行相同脚本即可续传。", flush=True)
    os._exit(130)


def main():
    from huggingface_hub import HfApi, snapshot_download
    import signal

    parser = argparse.ArgumentParser(
        description="下载扩充版中文双声道录音和工具训练文本"
    )
    parser.add_argument(
        "--destination", default=default_resource_directory()
    )
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--part-mib", type=int, default=1024)
    parser.add_argument("--download-only", action="store_true")
    parser.add_argument("--plan-only", action="store_true")
    args = parser.parse_args()
    if args.workers < 1 or args.part_mib < 1:
        parser.error("workers 和 part-mib 应为正整数")
    root = Path(args.destination).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)
    print("等待当前资源任务完成后继续，共用同一目录的任务会串行执行。", flush=True)
    with (root / ".download-task.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        api = HfApi()
        smooth = inventory(
            retry(
                lambda: api.dataset_info(
                    DATASET_REPO, revision=DATASET_REVISION, files_metadata=True
                )
            )
        )
        tools = inventory(
            retry(
                lambda: api.dataset_info(
                    TOOL_REPO, revision=TOOL_REVISION, files_metadata=True
                )
            )
        )
        aligner = CATALOG["ForcedAligner"]
        aligner_files = inventory(
            retry(
                lambda: api.model_info(
                    aligner["repository"],
                    revision=aligner["revision"],
                    files_metadata=True,
                )
            )
        )
        pairs = select_full_pairs(smooth)
        audio_bytes = sum(smooth[item["audio"]]["size"] for item in pairs)
        raw_bytes = (
            sum(
                smooth[item[key]]["size"]
                for item in pairs
                for key in ("audio", "annotation")
            )
            + sum(item["size"] for item in tools.values())
            + sum(item["size"] for item in aligner_files.values())
        )
        destination = root / CATALOG["SmoothConvFull"]["root"]
        previous = root / CATALOG["SmoothConv"]["root"]
        reusable = sum(
            smooth[pair[key]]["size"]
            for pair in pairs
            for key in ("audio", "annotation")
            if valid_local(destination / pair[key], smooth[pair[key]])
            or valid_local(previous / pair[key], smooth[pair[key]])
        )
        required = (
            max(0, raw_bytes - reusable)
            + (raw_bytes if not args.download_only else 0)
            + 2 * 1024**3
        )
        free = shutil.disk_usage(root).free
        print(
            f"扩充素材 {len(pairs)} 段双声道录音，WAV 约 {audio_bytes / 1024**3:.2f} GiB",
            flush=True,
        )
        print(
            f"ToolACE 约 {sum(item['size'] for item in tools.values()) / 1024**2:.1f} MiB，已下载录音和标注可复用约 {reusable / 1024**3:.2f} GiB",
            flush=True,
        )
        print(
            f"当前可用 {free / 1024**3:.1f} GiB，本次下载打包保守预算 {required / 1024**3:.1f} GiB",
            flush=True,
        )
        plan = {
            "profile": "full-2ch",
            "recordings": len(pairs),
            "audio_bytes": audio_bytes,
            "tool_repo": TOOL_REPO,
            "tool_revision": TOOL_REVISION,
            "aligner_repository": aligner["repository"],
            "aligner_revision": aligner["revision"],
            "raw_bytes": raw_bytes,
            "required_free_bytes": required,
        }
        if args.plan_only:
            return
        if free < required:
            raise RuntimeError(
                "目标磁盘空间不足，请调整 destination 或使用 download-only"
            )
        atomic_json(root / "training_materials_plan.json", plan)
        download_smoothconv(root, smooth, pairs, args.workers)
        download_toolace(root, tools)
        print("下载用于后续时间戳细化的强制对齐模型", flush=True)
        retry(
            lambda: snapshot_download(
                aligner["repository"],
                revision=aligner["revision"],
                local_dir=root / aligner["root"],
                max_workers=3,
            )
        )
        if not args.download_only:
            for group in ("SmoothConvFull", "ToolACE", "ForcedAligner"):
                print(f"生成扩充素材上传分卷 {group}", flush=True)
                pack_materials(root, group, args.part_mib)
        atomic_json(root / "training_materials.json", {**plan, "complete": True})
        print(
            "扩充素材准备完成，请上传 upload 中 SmoothConvFull、ToolACE、ForcedAligner 的分卷及清单。",
            flush=True,
        )


if __name__ == "__main__":
    main()
