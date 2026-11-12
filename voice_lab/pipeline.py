import argparse
import copy
import gzip
import hashlib
import json
import random
import shutil
from pathlib import Path
import subprocess
import sys
import tarfile
import time
import yaml
from .config import ROOT, load_config
from .corpus import convert_corpus, write_jsonl
from .data import load_manifest
from .download import atomic_json, download_dataset, download_models, DATASET_REVISION


def save_yaml(path, config):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".yaml.tmp")
    temporary.write_text(
        yaml.safe_dump(config, allow_unicode=True, sort_keys=False), encoding="utf-8"
    )
    temporary.replace(path)


def read_jsonl(path):
    return [
        json.loads(line)
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def backup_job_inputs(job, config):
    """保存不可重新随机生成的指令录音和清单，供实例更换后续训。"""
    directory = config["training"].get("backup_dir")
    if not directory:
        return
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    archive = directory / (job.name + ".inputs.tar.gz")
    temporary = archive.with_suffix(".tmp")
    with temporary.open("wb") as stream:
        with gzip.GzipFile(fileobj=stream, mode="wb", compresslevel=1, mtime=0) as compressed:
            with tarfile.open(fileobj=compressed, mode="w|") as target:
                for path in sorted(job.rglob("*")):
                    relative = path.relative_to(job)
                    if path.is_file() and relative.parts[0] not in ("prepared", "eval_prepared") and not path.name.startswith("."):
                        target.add(path, arcname=str(Path(job.name) / relative), recursive=False)
    temporary.replace(archive)
    checksum = hashlib.sha256()
    with archive.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            checksum.update(chunk)
    atomic_json(directory / (job.name + ".inputs.json"), {
        "file": archive.name, "sha256": checksum.hexdigest(), "job": job.name,
    })
    print(f"续训所需指令录音和清单已备份 {archive}", flush=True)


def restore_job_inputs(job, config):
    directory = config["training"].get("backup_dir")
    if not directory or (job / "instructions.jsonl").exists():
        return
    directory = Path(directory)
    metadata = directory / (job.name + ".inputs.json")
    if not metadata.is_file():
        return
    value = json.loads(metadata.read_text())
    if value["job"] != job.name or Path(value["file"]).name != value["file"]:
        raise ValueError("续训数据备份清单不匹配")
    archive = directory / value["file"]
    checksum = hashlib.sha256()
    with archive.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            checksum.update(chunk)
    if checksum.hexdigest() != value["sha256"]:
        raise ValueError("续训指令录音备份校验失败")
    with tarfile.open(archive) as source:
        for item in source.getmembers():
            parts = Path(item.name).parts
            if not item.isfile() or not parts or parts[0] != job.name or ".." in parts:
                raise ValueError("续训备份包含异常路径")
        source.extractall(job.parent, filter="data")
    print("已恢复同一任务原始指令录音", flush=True)


def filter_records(records, tokenizer, max_tokens):
    from .prepare import target_ids

    accepted, rejected = [], []
    for record in records:
        if any(
            len(target_ids(unit, tokenizer)) > max_tokens
            for unit in record["units"]
            if not unit.get("listen")
        ):
            rejected.append(record["id"])
        else:
            accepted.append(record)
    if not accepted:
        raise ValueError("没有适配当前分块长度的训练样本")
    return accepted, rejected


def training_checkpoint(output, max_steps):
    output = Path(output)
    if (output / "final" / "adapter_config.json").is_file():
        return output / "final", True
    checkpoints = sorted(
        path
        for path in output.glob("step-*")
        if path.is_dir() and not path.is_symlink()
        and path.name.removeprefix("step-").isdigit()
        and (path / "training_state.pt").is_file()
    )
    if not checkpoints:
        return None, False
    latest = checkpoints[-1]
    reached = max_steps > 0 and int(latest.name.split("-")[-1]) >= max_steps
    return latest, reached


def restore_training_checkpoint(output, config):
    backup = config["training"].get("backup_dir")
    if not backup or training_checkpoint(output, 0)[0] is not None:
        return
    checkpoint, _ = training_checkpoint(backup, 0)
    if checkpoint is None:
        return
    metadata = json.loads((checkpoint / "voice_lab_metadata.json").read_text())
    if metadata["config"]["training"]["output_dir"] != str(output):
        return
    output.mkdir(parents=True, exist_ok=True)
    temporary = output / (checkpoint.name + ".tmp")
    if temporary.exists():
        shutil.rmtree(temporary)
    shutil.copytree(checkpoint, temporary)
    temporary.rename(output / checkpoint.name)
    print(f"已从共享存储恢复训练检查点 {checkpoint.name}", flush=True)


def run_module(module, *arguments):
    command = [sys.executable, "-m", "voice_lab." + module, *map(str, arguments)]
    print("执行 " + " ".join(command), flush=True)
    subprocess.run(command, cwd=ROOT, check=True)


def main():
    parser = argparse.ArgumentParser(
        description="一键下载真实语料与模型、生成指令数据并训练"
    )
    parser.add_argument("--clips", type=int, default=16)
    parser.add_argument("--instructions", type=int, default=24)
    parser.add_argument("--window-seconds", type=int, default=24)
    parser.add_argument("--max-windows", type=int, default=6)
    parser.add_argument(
        "--max-steps",
        type=int,
        default=20,
        help="默认进行20步实验训练，0表示完成配置中的所有轮次",
    )
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--config")
    parser.add_argument("--download-only", action="store_true")
    parser.add_argument("--lazy", action="store_true", help="边编译特征边训练")
    parser.add_argument("--max-seconds", type=float, help="本次运行的训练时限，退出后可继续")
    parser.add_argument("--corpus-dir", help="复用已转换的完整语料目录")
    parser.add_argument("--eval-samples", type=int, default=0, help="本轮评估样本上限，0表示全部")
    args = parser.parse_args()
    started = time.monotonic()
    if (
        args.clips < 1
        or args.instructions < 1
        or args.window_seconds < 2
        or args.max_windows < 0
        or args.epochs < 1
        or args.max_steps < 0
        or args.eval_samples < 0
        or (args.max_seconds is not None and args.max_seconds <= 0)
    ):
        parser.error("数据规模、窗口和训练轮数需要为正数，max-steps 可以为0")
    config = copy.deepcopy(load_config(args.config))
    signature = {
        "version": 3,
        "dataset_revision": DATASET_REVISION,
        "clips": args.clips,
        "instructions": args.instructions,
        "window": args.window_seconds,
        "max_windows": args.max_windows,
        "seed": args.seed,
        "epochs": args.epochs,
        "model": config["model"],
        "training": config["training"],
        "raw_dir": config.get("data", {}).get("raw_dir"),
        "corpus_dir": str(Path(args.corpus_dir).resolve()) if args.corpus_dir else None,
        "lazy": args.lazy,
        "eval_samples": args.eval_samples,
    }
    key = hashlib.sha256(json.dumps(signature, sort_keys=True).encode()).hexdigest()[
        :12
    ]
    job = ROOT / "data/automatic" / key
    job.mkdir(parents=True, exist_ok=True)
    if config["training"].get("backup_dir"):
        config["training"]["backup_dir"] = str(Path(config["training"]["backup_dir"]) / key)
    # 同一任务使用文件锁，防止两个一键进程同时写特征或检查点。
    import fcntl

    with (job / ".pipeline.lock").open("w") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("同一个一键任务正在执行，请查看原进程") from exc
        atomic_json(job / "job.json", signature)
        restore_job_inputs(job, config)
        models = download_models(config)
        local_raw = config.get("data", {}).get("raw_dir")
        if args.corpus_dir:
            raw = Path(args.corpus_dir)
        elif local_raw:
            raw = Path(local_raw)
            if not (raw / "download_manifest.json").is_file():
                raise FileNotFoundError("指定语料目录缺少完整下载清单")
        else:
            raw = download_dataset(args.clips, args.seed)
        if args.download_only:
            print("基座、后台模型及真实中文录音下载完成")
            return
        config["model"]["path"] = models[0]["path"]
        config["training"].update(
            prepared_dir=str(job / "prepared"),
            output_dir=str(ROOT / "runs" / ("automatic-" + key)),
            epochs=args.epochs,
            seed=args.seed,
        )
        if not args.lazy:
            config["training"]["save_every"] = min(config["training"]["save_every"], 5)
        config_path = job / "config.yaml"
        save_yaml(config_path, config)
        if (job / "corpus_report.json").is_file():
            real_train, evaluation = (
                read_jsonl(job / "real_train.jsonl"),
                read_jsonl(job / "real_eval.jsonl"),
            )
        elif args.corpus_dir:
            directory = Path(args.corpus_dir)
            real_train = load_manifest(directory / "real_train.jsonl")
            evaluation = load_manifest(directory / "real_eval.jsonl")
            write_jsonl(job / "real_train.jsonl", real_train)
            write_jsonl(job / "real_eval.jsonl", evaluation)
            atomic_json(job / "corpus_report.json", json.loads((directory / "corpus_report.json").read_text()))
        else:
            real_train, evaluation = convert_corpus(
                raw, job, args.window_seconds, args.max_windows
            )
        if not (job / "instructions.jsonl").is_file():
            run_module(
                "synthesis",
                "--manifest",
                job / "real_train.jsonl",
                "--output",
                job,
                "--count",
                args.instructions,
                "--config",
                config_path,
            )
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(
            models[0]["path"], trust_remote_code=True
        )
        training, rejected = filter_records(
            real_train + read_jsonl(job / "instructions.jsonl"),
            tokenizer,
            config["model"]["max_speak_tokens"],
        )
        write_jsonl(job / "train.jsonl", training)
        if evaluation:
            if args.eval_samples:
                evaluation = random.Random(args.seed).sample(evaluation, min(len(evaluation), args.eval_samples))
            evaluation, eval_rejected = filter_records(
                evaluation, tokenizer, config["model"]["max_speak_tokens"]
            )
            rejected += eval_rejected
            write_jsonl(job / "eval.jsonl", evaluation)
        atomic_json(
            job / "dataset_report.json",
            {
                "train": len(training),
                "eval": len(evaluation),
                "reference_voices": len({item["voice_id"] for item in training}),
                "rejected_long_units": rejected,
                "real_samples": len(
                    [
                        item
                        for item in training
                        if item["provenance"]["kind"] == "real_conversation"
                    ]
                ),
                "synthetic_samples": len(
                    [
                        item
                        for item in training
                        if item["provenance"]["kind"] == "synthetic_instruction"
                    ]
                ),
                "dataset_license": "CC-BY-NC-4.0",
            },
        )
        load_manifest(job / "train.jsonl")
        backup_job_inputs(job, config)
        if not args.lazy:
            run_module("prepare", job / "train.jsonl", "--config", config_path, "--resume")
        evaluation_arguments = []
        if evaluation:
            run_module(
                "prepare",
                job / "eval.jsonl",
                "--config",
                config_path,
                "--output",
                job / "eval_prepared",
                "--resume",
            )
            evaluation_arguments = ["--eval-dir", job / "eval_prepared"]
        output = Path(config["training"]["output_dir"])
        restore_training_checkpoint(output, config)
        checkpoint, reached = training_checkpoint(output, args.max_steps)
        if not reached:
            arguments = ["--config", config_path, *evaluation_arguments]
            if args.lazy:
                arguments += ["--manifest", job / "train.jsonl"]
            if checkpoint:
                arguments += ["--resume", checkpoint]
            if args.max_steps:
                arguments += ["--max-steps", args.max_steps]
            if args.max_seconds is not None:
                remaining = args.max_seconds - (time.monotonic() - started)
                if remaining <= 0:
                    print("本轮准备工作已保存，时间预算已用完，可用相同命令继续。", flush=True)
                    return
                arguments += ["--max-seconds", remaining]
            run_module("train", *arguments)
        checkpoint, _ = training_checkpoint(output, args.max_steps)
        if checkpoint is None:
            raise RuntimeError("训练结束后没有找到适配器检查点")
        serving = copy.deepcopy(config)
        serving["model"]["adapter"] = str(checkpoint)
        serving["backend"]["base_url"] = config["backend"]["base_url"]
        active = ROOT / "configs/automatic-latest.yaml"
        save_yaml(active, serving)
        atomic_json(
            job / "completion.json",
            {
                "checkpoint": str(checkpoint),
                "serving_config": str(active),
                "training_samples": len(training),
                "evaluation_samples": len(evaluation),
                "max_steps": args.max_steps,
                "status": "complete" if checkpoint.name == "final" else "paused",
            },
        )
        print(f"本轮流程结束，状态 {'全部轮次完成' if checkpoint.name == 'final' else '已保存断点'}，适配器位于 {checkpoint}")
        print(f"启动语音服务可执行 VOICE_CONFIG={active} ./quickstart.sh serve")


if __name__ == "__main__":
    main()
