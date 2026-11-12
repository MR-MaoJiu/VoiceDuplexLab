"""真实 GPU 上用少量真实录音验证训练与检查点重载。"""

import argparse
import copy
from datetime import datetime, timezone
import json
import subprocess
import sys

from .config import ROOT, load_config, local_path
from .corpus import convert_corpus, write_jsonl
from .download import atomic_json
from .pipeline import save_yaml


def execute(module, *arguments):
    subprocess.run(
        [sys.executable, "-m", "voice_lab." + module, *map(str, arguments)],
        cwd=ROOT,
        check=True,
    )


def main():
    import torch
    from safetensors.torch import load_file

    parser = argparse.ArgumentParser(
        description="用2条真实样本验证GPU训练、保存和重新加载"
    )
    parser.add_argument("--config", default=str(ROOT / "configs/resources-local.yaml"))
    args = parser.parse_args()
    config = copy.deepcopy(load_config(args.config))
    if not torch.cuda.is_available():
        raise RuntimeError("真实训练检查需要可用的 NVIDIA CUDA GPU")
    torch.cuda.set_device(config["model"]["device"])
    run = (
        ROOT
        / "runs"
        / ("gpu-smoke-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ"))
    )
    raw = local_path(config.get("data", {}).get("raw_dir", "data/raw/smoothconv"))
    source = json.loads((raw / "download_manifest.json").read_text())
    pairs = [
        {key: str((raw / item[key]).resolve()) for key in ("audio", "annotation")}
        for item in source["pairs"][:2]
    ]
    if not pairs:
        raise ValueError("没有可用于训练验证的真实录音")
    small_raw = run / "raw"
    atomic_json(small_raw / "download_manifest.json", {**source, "pairs": pairs})
    train, evaluation = convert_corpus(small_raw, run / "converted", max_windows=2)
    write_jsonl(run / "train.jsonl", train[:2])
    if evaluation:
        write_jsonl(run / "eval.jsonl", evaluation[:1])
    config["model"]["adapter"] = None
    config["training"].update(
        prepared_dir=str(run / "prepared"),
        output_dir=str(run / "checkpoints"),
        gradient_accumulation=1,
        epochs=2,
        save_every=1,
    )
    config_path = run / "config.yaml"
    save_yaml(config_path, config)
    execute("prepare", run / "train.jsonl", "--config", config_path)
    evaluation_args = []
    if evaluation:
        execute(
            "prepare",
            run / "eval.jsonl",
            "--config",
            config_path,
            "--output",
            run / "eval_prepared",
        )
        evaluation_args = ["--eval-dir", str(run / "eval_prepared")]
    execute("train", "--config", config_path, "--max-steps", "2", *evaluation_args)
    checkpoints = sorted((run / "checkpoints").glob("step-*"))
    checkpoint = checkpoints[-1]
    state = torch.load(
        checkpoint / "training_state.pt", map_location="cpu", weights_only=True
    )
    weights = load_file(str(checkpoint / "adapter_model.safetensors"))
    changed = any(
        "lora_B" in name and bool(value.abs().sum() > 0)
        for name, value in weights.items()
    )
    if not changed or state["step"] < 1:
        raise RuntimeError("检查点没有记录有效的 LoRA 参数更新")
    del weights
    from .model import load_base, model_fingerprint
    from .train import load_sample, read_index

    config["model"]["adapter"] = str(checkpoint)
    model = load_base(config)
    samples, _ = read_index(run / "prepared", model_fingerprint(config))
    embeds, labels = load_sample(
        run / "prepared",
        samples[0],
        model.llm.device,
        next(model.llm.parameters()).dtype,
        model_fingerprint(config),
        config["training"]["max_tokens"],
    )
    with torch.inference_mode():
        loss = model.llm(inputs_embeds=embeds, labels=labels, use_cache=False).loss
    if not torch.isfinite(loss):
        raise RuntimeError("重载适配器后的损失出现非有限数值")
    report = {
        "gpu": torch.cuda.get_device_name(),
        "steps": state["step"],
        "training_samples": min(2, len(train)),
        "evaluation_samples": min(1, len(evaluation)),
        "adapter_updated": changed,
        "reloaded_loss": loss.item(),
        "checkpoint": str(checkpoint),
    }
    atomic_json(run / "verification.json", report)
    print("真实GPU训练验证完成 " + json.dumps(report, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
