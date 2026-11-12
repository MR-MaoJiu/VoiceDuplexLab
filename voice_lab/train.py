import argparse
import hashlib
import json
import random
import shutil
import copy
import signal
import time
from .config import load_config, local_path
from .model import load_base, model_fingerprint


def read_index(directory, fingerprint, expected_compiler=None):
    path = directory / "index.json"
    raw = path.read_bytes()
    index = json.loads(raw)
    if index["fingerprint"] != fingerprint or not index["samples"]:
        raise ValueError("训练特征与当前基座不匹配，或数据集为空")
    if expected_compiler is not None and index.get('compiler', 4) != expected_compiler:
        raise ValueError('训练特征的标签版本不匹配，请在新目录重新编译特征')
    return index["samples"], hashlib.sha256(raw).hexdigest()


def atomic_checkpoint(model, optimizer, output, state, config, name):
    import torch

    target = output / name
    if target.exists():
        raise FileExistsError(f"检查点已存在 {target}")
    temporary = output / (name + ".tmp")
    if temporary.exists():
        shutil.rmtree(temporary)
    temporary.mkdir(parents=True)
    model.save_pretrained(temporary, safe_serialization=True)
    safe_config = copy.deepcopy(config)
    safe_config["server"]["token"] = ""
    metadata = {
        "base_revision": config["model"]["revision"],
        "fingerprint": model_fingerprint(config),
        "scope": "native-duplex-language-lora",
        "voice_conditioning": "per-sample-reference-audio" if config["model"].get("reference_audio_in_context", True) else "native-decoder-reference-only",
        "config": safe_config,
    }
    (temporary / "voice_lab_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    torch.save(
        {
            **state,
            "optimizer": optimizer.state_dict(),
            "torch_rng": torch.get_rng_state(),
            "cuda_rng": torch.cuda.get_rng_state_all(),
        },
        temporary / "training_state.pt",
    )
    temporary.rename(target)
    print(f"保存检查点 {target}", flush=True)
    options = config["training"]
    backup = options.get("backup_dir")
    if backup:
        backup = local_path(backup)
        backup.mkdir(parents=True, exist_ok=True)
        staging = backup / (name + ".tmp")
        if staging.exists():
            shutil.rmtree(staging)
        shutil.copytree(target, staging)
        staging.rename(backup / name)
        print(f"共享存储检查点备份完成 {backup / name}", flush=True)
    keep = options.get("keep_checkpoints", 3)
    if keep < 1:
        raise ValueError("至少需要保留一个检查点")
    for directory in [output] + ([backup] if backup else []):
        previous = sorted(
            path for path in directory.glob("step-*")
            if path.is_dir() and not path.is_symlink()
            and path.name.removeprefix("step-").isdigit()
            and (path / "training_state.pt").is_file()
        )
        for path in previous[:-keep]:
            shutil.rmtree(path)


def load_sample(directory, item, device, dtype, fingerprint, max_tokens, expected_compiler=None):
    import torch

    sample = torch.load(directory / item["file"], map_location="cpu", weights_only=True)
    if expected_compiler is not None and sample.get('compiler', 4) != expected_compiler:
        raise ValueError('样本使用旧标签版本，请在新目录重新编译特征')
    labels = sample["labels"]
    embeddings = sample["inputs_embeds"]
    if sample["fingerprint"] != fingerprint or len(labels) > max_tokens:
        raise ValueError("特征版本不匹配或样本过长")
    if (
        embeddings.ndim != 2
        or len(embeddings) != len(labels)
        or not (labels[1:] != -100).any()
    ):
        raise ValueError("训练样本形状不正确或没有监督目标")
    return embeddings.unsqueeze(0).to(device=device, dtype=dtype), labels.unsqueeze(
        0
    ).to(device)


def main():
    import torch
    from peft import LoraConfig, PeftModel, get_peft_model
    from torch.utils.tensorboard import SummaryWriter

    parser = argparse.ArgumentParser()
    parser.add_argument("--config")
    parser.add_argument("--resume")
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--eval-dir")
    parser.add_argument("--manifest", help="按需编译清单中的录音特征并立即训练")
    parser.add_argument("--max-seconds", type=float, help="到达运行时限后保存检查点退出")
    args = parser.parse_args()
    if args.max_steps is not None and args.max_steps < 0:
        parser.error("max-steps 需要大于等于0，0表示完成全部轮次")
    if args.max_seconds is not None and args.max_seconds <= 0:
        parser.error("max-seconds 需要大于0")
    config = load_config(args.config)
    from .prepare import compiler_version
    compiler = compiler_version(config)
    options = config["training"]
    random.seed(options["seed"])
    torch.manual_seed(options["seed"])
    fingerprint = model_fingerprint(config)
    directory = local_path(options["prepared_dir"])
    records = None
    if args.manifest:
        from .data import load_manifest
        from .prepare import source_signature, initialize_preparation

        records = load_manifest(args.manifest)
        signature = source_signature(args.manifest, records, config)
        initialize_preparation(directory, signature, resume=True)
        samples = [{"id": record["id"]} for record in records]
        dataset_hash = hashlib.sha256((signature + fingerprint).encode()).hexdigest()
    else:
        samples, dataset_hash = read_index(directory, fingerprint, compiler)
    output = local_path(options["output_dir"])
    output.mkdir(parents=True, exist_ok=True)
    if list(output.glob("step-*")) and not args.resume:
        raise FileExistsError("已有训练结果，请使用 --resume 或配置新的 output_dir")
    model = load_base(config, with_adapter=False)
    llm = model.llm
    if records is None:
        del model
    torch.cuda.empty_cache()
    llm.requires_grad_(False)
    if args.resume:
        checkpoint = local_path(args.resume)
        metadata = json.loads((checkpoint / "voice_lab_metadata.json").read_text())
        if metadata["fingerprint"] != fingerprint:
            raise ValueError("检查点基座不一致")
        llm = PeftModel.from_pretrained(llm, checkpoint, is_trainable=True)
    else:
        adapter = LoraConfig(
            r=options["rank"],
            lora_alpha=options["alpha"],
            lora_dropout=0.0,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=options.get("target_modules", [
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
            ]),
        )
        llm = get_peft_model(llm, adapter)
    llm.config.use_cache = False
    llm.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )
    llm.train()
    if records is not None:
        # 原生解码器读取 model.embed_tokens，保留底层 Qwen 层级并共享已注入的 LoRA 权重。
        model.llm = llm.get_base_model()
    llm.print_trainable_parameters()
    optimizer = torch.optim.AdamW(
        (parameter for parameter in llm.parameters() if parameter.requires_grad),
        lr=options["learning_rate"],
    )
    state = {
        "epoch": 0,
        "next_index": 0,
        "step": 0,
        "dataset_hash": dataset_hash,
        "training_options": options,
    }
    if args.resume:
        restored = torch.load(
            checkpoint / "training_state.pt", map_location="cpu", weights_only=True
        )
        if (
            restored["dataset_hash"] != dataset_hash
            or restored["training_options"] != options
        ):
            raise ValueError("恢复训练时数据或训练配置已发生变化")
        optimizer.load_state_dict(restored["optimizer"])
        torch.set_rng_state(restored["torch_rng"])
        torch.cuda.set_rng_state_all(restored["cuda_rng"])
        state = {key: restored[key] for key in state}
    writer = SummaryWriter(str(output / "tensorboard"))
    accumulation = options["gradient_accumulation"]
    if accumulation < 1 or options["epochs"] < 1 or options["save_every"] < 1:
        raise ValueError("训练步数、轮数和累积步数必须为正整数")
    evaluation = None
    if args.eval_dir:
        eval_directory = local_path(args.eval_dir)
        evaluation, _ = read_index(eval_directory, fingerprint, compiler)
        if {item["id"] for item in evaluation} & {item["id"] for item in samples}:
            raise ValueError("训练集与评估集存在重复 id")
    start_epoch = state["epoch"]
    started = time.monotonic()
    stopped = False

    def request_stop(signum, frame):
        nonlocal stopped
        stopped = True
        print("收到停止请求，将在当前更新完成后保存可恢复检查点。", flush=True)

    previous_handlers = {
        number: signal.signal(number, request_stop)
        for number in (signal.SIGTERM, signal.SIGINT)
    }

    def save_progress(status, reason=None):
        from .download import atomic_json

        value = {key: state[key] for key in ("epoch", "next_index", "step", "dataset_hash")}
        value.update(status=status, reason=reason, samples=len(samples))
        atomic_json(output / "progress.json", value)
        if options.get("backup_dir"):
            atomic_json(local_path(options["backup_dir"]) / "progress.json", value)

    def save_paused(reason):
        name = f"step-{state['step']:07d}"
        if not (output / name).exists():
            atomic_checkpoint(llm, optimizer, output, state, config, name)
        save_progress("paused", reason)
        print(f"训练已保存，可从 {output / name} 继续。停止原因 {reason}", flush=True)

    def training_sample(position):
        item = samples[position]
        if records is not None:
            from .prepare import cached_record

            was_training = llm.training
            was_cache = llm.config.use_cache
            llm.eval()
            llm.config.use_cache = True
            try:
                item = cached_record(records[position], config, directory, signature, lambda: model)
            finally:
                llm.config.use_cache = was_cache
                llm.train(was_training)
        return load_sample(directory, item, llm.device, next(llm.parameters()).dtype,
                           fingerprint, options["max_tokens"], compiler)

    def evaluate():
        if not evaluation:
            return
        llm.eval()
        total_loss, total_tokens = 0.0, 0
        with torch.no_grad():
            for item in evaluation:
                embeds, labels = load_sample(
                    eval_directory, item, llm.device, next(llm.parameters()).dtype,
                    fingerprint, options["max_tokens"], compiler,
                )
                loss = llm(inputs_embeds=embeds, labels=labels, use_cache=False).loss
                if not torch.isfinite(loss):
                    raise FloatingPointError("评估损失出现非有限数值")
                count = (labels[:, 1:] != -100).sum().item()
                total_loss += loss.item() * count
                total_tokens += count
        value = total_loss / total_tokens
        writer.add_scalar("eval/loss", value, state["step"])
        print(json.dumps({"step": state["step"], "eval_loss": value,
                          "evaluation_samples": len(evaluation)}), flush=True)
        llm.train()

    try:
        if options.get("eval_every", 0):
            evaluate()
        for epoch in range(start_epoch, options["epochs"]):
            order = list(range(len(samples)))
            if records is not None:
                weight = options.get("instruction_weight", 1)
                if weight < 1:
                    raise ValueError("指令样本权重至少为1")
                order += [index for index, record in enumerate(records)
                          if record.get("provenance", {}).get("kind") == "synthetic_instruction"
                          for _ in range(weight - 1)]
            random.Random(options["seed"] + epoch).shuffle(order)
            start = state["next_index"] if epoch == start_epoch else 0
            for group_start in range(start, len(order), accumulation):
                if shutil.disk_usage(directory).free < options.get("min_free_gib", 5) * 1024**3:
                    save_paused("disk_space")
                    return
                group = order[group_start : group_start + accumulation]
                optimizer.zero_grad(set_to_none=True)
                total = 0.0
                for position in group:
                    embeds, labels = training_sample(position)
                    result = llm(inputs_embeds=embeds, labels=labels, use_cache=False)
                    if not torch.isfinite(result.loss):
                        raise FloatingPointError("训练损失出现非有限数值")
                    total += result.loss.item()
                    (result.loss / len(group)).backward()
                norm = torch.nn.utils.clip_grad_norm_(llm.parameters(), 1.0)
                if not torch.isfinite(norm):
                    raise FloatingPointError("梯度出现非有限数值")
                optimizer.step()
                state.update(
                    epoch=epoch,
                    next_index=group_start + len(group),
                    step=state["step"] + 1,
                )
                loss = total / len(group)
                writer.add_scalar("train/loss", loss, state["step"])
                print(
                    json.dumps(
                        {"step": state["step"], "epoch": epoch, "loss": loss,
                         "next_index": state["next_index"], "epoch_samples": len(order)},
                        ensure_ascii=False,
                    ),
                    flush=True,
                )
                stop = (stopped
                        or (args.max_seconds is not None and time.monotonic() - started >= args.max_seconds)
                        or (args.max_steps is not None and args.max_steps > 0 and state["step"] >= args.max_steps))
                if state["step"] % options["save_every"] == 0 or stop:
                    atomic_checkpoint(
                        llm,
                        optimizer,
                        output,
                        state,
                        config,
                        f"step-{state['step']:07d}",
                    )
                    save_progress("running")
                if stop:
                    reason = "signal" if stopped else (
                        "time_limit" if args.max_seconds is not None and time.monotonic() - started >= args.max_seconds
                        else "step_limit"
                    )
                    save_progress("paused", reason)
                    return
                if options.get("eval_every", 0) and state["step"] % options["eval_every"] == 0:
                    evaluate()
            state.update(epoch=epoch + 1, next_index=0)
            evaluate()
        atomic_checkpoint(llm, optimizer, output, state, config, "final")
        save_progress("complete")
    finally:
        writer.close()
        for number, handler in previous_handlers.items():
            signal.signal(number, handler)


if __name__ == "__main__":
    main()
