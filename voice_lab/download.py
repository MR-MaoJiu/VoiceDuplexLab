import argparse
import hashlib
import json
from pathlib import Path
import time
from .config import ROOT, load_config

DATASET_REPO = "qualialabsAI/SmoothConv"
DATASET_REVISION = "cd74b4fca285a66d6ac8c16228d0953ff1e0cda2"
BACKEND_REVISION = "b968826d9c46dd6066d109eabc6255188de91218"


def retry(function, attempts=4):
    for attempt in range(attempts):
        try:
            return function()
        except Exception as exc:
            if attempt + 1 == attempts:
                raise
            status = getattr(getattr(exc, "response", None), "status_code", None)
            delay = 20 if status == 429 else min(2 ** (attempt + 1), 20)
            print(f"下载遇到 {type(exc).__name__}，{delay} 秒后重试", flush=True)
            time.sleep(delay)


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    temporary.replace(path)


def choose_pairs(files, clips, seed):
    names = set(files)
    waves = [
        name
        for name in names
        if name.startswith("none_edu_")
        and "_2ch/" in name
        and name.endswith(".wav")
        and name[:-4] + ".json" in names
    ]
    waves.sort(key=lambda name: hashlib.sha256(f"{seed}:{name}".encode()).hexdigest())
    if clips < 1 or clips > len(waves):
        raise ValueError(f"clips 应在 1 到 {len(waves)} 之间")
    return [(name, name[:-4] + ".json") for name in waves[:clips]]


def download_dataset(clips=16, seed=42, directory=None):
    from huggingface_hub import HfApi, hf_hub_download

    directory = Path(directory or ROOT / "data/raw/smoothconv")
    directory.mkdir(parents=True, exist_ok=True)
    cached = directory / "download_manifest.json"
    if cached.is_file():
        value = json.loads(cached.read_text(encoding="utf-8"))
        if (
            value.get("repository") == DATASET_REPO
            and value.get("revision") == DATASET_REVISION
            and value.get("seed") == seed
            and len(value.get("pairs", [])) >= clips
        ):
            selected = value["pairs"][:clips]
            if all(
                (directory / pair[key]).is_file()
                and (directory / pair[key]).stat().st_size > 0
                for pair in selected
                for key in ("audio", "annotation")
            ):
                if len(value["pairs"]) != clips:
                    atomic_json(
                        directory / "download_manifest.json",
                        {**value, "pairs": selected},
                    )
                print("复用已完整下载的固定版本中文语料", flush=True)
                return directory
    files = retry(
        lambda: HfApi().list_repo_files(
            DATASET_REPO, repo_type="dataset", revision=DATASET_REVISION
        )
    )
    pairs = choose_pairs(files, clips, seed)

    def fetch(name):
        return retry(
            lambda: hf_hub_download(
                DATASET_REPO,
                filename=name,
                repo_type="dataset",
                revision=DATASET_REVISION,
                local_dir=directory,
            )
        )

    fetch("README.md")
    for index, (audio, annotation) in enumerate(pairs, 1):
        print(f"下载真实对话 {index}/{len(pairs)} {audio}", flush=True)
        fetch(annotation)
        fetch(audio)
    manifest = {
        "repository": DATASET_REPO,
        "revision": DATASET_REVISION,
        "license": "CC-BY-NC-4.0",
        "seed": seed,
        "pairs": [
            {"audio": audio, "annotation": annotation} for audio, annotation in pairs
        ],
    }
    atomic_json(directory / "download_manifest.json", manifest)
    return directory


def download_models(config):
    from huggingface_hub import snapshot_download

    models = [
        (config["model"]["path"], config["model"]["revision"]),
        (
            config["backend"].get("weights_path", config["backend"]["model"]),
            config["backend"].get("revision", BACKEND_REVISION),
        ),
    ]
    results = []
    for repository, revision in models:
        local = Path(repository).expanduser()
        if not local.is_absolute():
            local = ROOT / local
        if local.is_dir():
            path = str(local)
        else:
            print(f"下载模型 {repository}，固定版本 {revision}", flush=True)
            path = retry(lambda: snapshot_download(repository, revision=revision))
        results.append({"repository": repository, "revision": revision, "path": path})
    atomic_json(ROOT / "models/downloaded_models.json", {"models": results})
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config")
    parser.add_argument("--clips", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dataset-only", action="store_true")
    parser.add_argument("--models-only", action="store_true")
    args = parser.parse_args()
    if args.dataset_only and args.models_only:
        parser.error("dataset-only 与 models-only 只能选择一个")
    if not args.dataset_only:
        download_models(load_config(args.config))
    if not args.models_only:
        print(f"真实中文对话已下载至 {download_dataset(args.clips, args.seed)}")


if __name__ == "__main__":
    main()
