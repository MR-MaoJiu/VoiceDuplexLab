import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from .config import load_config, local_path
from .data import load_manifest
from .model import load_base, model_fingerprint, read_audio
from .prompts import VOICE_PROMPT
from .backend import result_context

COMPILER_VERSION = 5


def compiler_version(config):
    # 未包含新选项的历史配置仍使用原版本，以便恢复旧实验。
    if config['training'].get('mask_forced_tokens', False):
        return COMPILER_VERSION if config.get('model', {}).get('reference_audio_in_context', True) else 6
    if not config.get('model', {}).get('reference_audio_in_context', True):
        raise ValueError('不含参考录音语义的新特征必须启用强制标记掩码')
    return 4


def target_ids(unit, tokenizer):
    if unit.get("listen"):
        pieces = ["<|listen|>"]
    else:
        text = unit["text"]
        if unit.get("delegate"):
            text = "@@TASK " + text + " @@END"
        if unit.get("cancel"):
            text = "@@CANCEL@@"
        pieces = ["<|speak|>", text]
        if unit.get("end", False):
            pieces.append("<|turn_eos|>")
        pieces.append("<|chunk_eos|>")
    ids = []
    for piece in pieces:
        ids.extend(tokenizer.encode(piece, add_special_tokens=False))
    ids.extend(tokenizer.encode("</unit>", add_special_tokens=False))
    return ids


def target_labels(unit, tokenizer, mask_forced_tokens=False):
    labels = target_ids(unit, tokenizer)
    if mask_forced_tokens:
        closing = tokenizer.encode('</unit>', add_special_tokens=False)
        if not closing:
            raise ValueError('分词器缺少单元结束标记')
        # 原生生成器直接插入该结构标记，不经过语言头采样。
        labels[-len(closing):] = [-100] * len(closing)
    return labels


class Capture:
    """捕获原生全双工预填充的输入嵌入，监督目标只标记助手输出。"""

    def __init__(self, decoder, max_tokens, forward=True, storage_device='cpu'):
        self.original = decoder.feed
        self.embeddings, self.labels = [], []
        self.target = None
        self.max_tokens = max_tokens
        self.count = 0
        self.forward = forward
        self.storage_device = storage_device

    def feed(self, embeddings, *args, **kwargs):
        import torch

        length = embeddings.shape[0]
        self.count += length
        if self.count > self.max_tokens:
            raise ValueError(f"样本超过 {self.max_tokens} token，请按完整对话边界拆分")
        self.embeddings.append(
            embeddings.detach().to(device=self.storage_device, dtype=torch.bfloat16).clone()
        )
        label = torch.full((length,), -100, dtype=torch.long)
        if self.target is not None:
            if len(self.target) != length:
                raise ValueError("监督标签与输入嵌入长度不匹配")
            label[:] = torch.tensor(self.target)
        self.labels.append(label)
        if self.forward:
            return self.original(embeddings, *args, **kwargs)
        # 特征编译已提供全部教师强制标签，后续不会读取语言模型的预测值。
        return (None, None) if kwargs.get("return_logits") else None


def prepare_record(model, config, record):
    import torch

    # 每条训练样本都重新建立参考音色和流式音频状态，禁止跨样本复用音色缓存。
    model.reset_session(reset_token2wav_cache=True)
    # 特征编译只使用输入编码和语言骨干，跳过重复的输出声码器初始化。
    original_init_tts = model.init_tts
    try:
        model.init_tts = lambda **kwargs: None
        duplex = model.as_duplex(generate_audio=False, sliding_window_mode="off")
    finally:
        model.init_tts = original_init_tts
    capture = Capture(
        duplex.decoder, config["training"]["max_tokens"],
        forward=config["training"].get("compile_llm_forward", True),
        storage_device=config['training'].get('capture_device', 'cpu'),
    )
    duplex.decoder.feed = capture.feed
    try:
        with torch.no_grad():
            duplex.prepare(
                prefix_system_prompt=record.get("prompt", VOICE_PROMPT),
                ref_audio=read_audio(record["voice"]) if config["model"].get("reference_audio_in_context", True) else None,
            )
            audio = read_audio(record["audio"])
            offset = 0
            for index, unit in enumerate(record["units"]):
                # 首帧保留 CNN 前后文，后续每次再消费约一秒音频。
                needed = duplex.processor.get_streaming_chunk_size()
                required = max(0, needed - len(duplex.audio_buffer))
                chunk = audio[offset : offset + required]
                offset += required
                if len(chunk) < required:
                    chunk = np.pad(chunk, (0, required - len(chunk)))
                context = None
                if "backend" in unit:
                    context = [result_context(unit["backend"])]
                result = duplex.streaming_prefill(
                    audio_waveform=chunk, text_list=context
                )
                if not result["success"]:
                    raise ValueError(
                        f"样本 {record['id']} 单元 {index} 音频预处理失败 {result.get('reason')}"
                    )
                ids = target_ids(unit, duplex.tokenizer)
                if len(ids) > config["model"]["max_speak_tokens"] and not unit.get(
                    "listen"
                ):
                    raise ValueError(
                        f"样本 {record['id']} 单元 {index} 文本过长，请拆成更多一秒单元"
                    )
                capture.target = target_labels(
                    unit, duplex.tokenizer,
                    mask_forced_tokens=config['training'].get('mask_forced_tokens', False),
                )
                embeddings = duplex.decoder.embed_tokens(ids)
                duplex.decoder.feed(embeddings)
                capture.target = None
    finally:
        duplex.decoder.feed = capture.original
        duplex.decoder.reset()
    return {
        "inputs_embeds": torch.cat(capture.embeddings),
        "labels": torch.cat(capture.labels),
        "fingerprint": model_fingerprint(config),
        "compiler": compiler_version(config),
        "id": record["id"],
        "voice_id": record.get(
            "voice_id",
            hashlib.sha256(Path(record["voice"]).read_bytes()).hexdigest()[:16],
        ),
    }


def source_signature(manifest, records, config):
    digest = hashlib.sha256()
    digest.update(Path(manifest).read_bytes())
    digest.update(
        json.dumps(
            {
                "compiler": compiler_version(config),
                "compile_llm_forward": config["training"].get("compile_llm_forward", True),
                "model": model_fingerprint(config),
                "dtype": config["model"]["dtype"],
                "max_tokens": config["training"]["max_tokens"],
                "prompt": VOICE_PROMPT,
            },
            sort_keys=True,
        ).encode()
    )
    for filename in sorted(
        {item[key] for item in records for key in ("voice", "audio")}
    ):
        digest.update(filename.encode())
        with Path(filename).open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


def initialize_preparation(output, signature, resume):
    output.mkdir(parents=True, exist_ok=True)
    metadata_path = output / "preparation.json"
    populated = list(output.glob("*.pt")) or metadata_path.exists() or (output / "index.json").exists()
    if populated and not resume:
        raise FileExistsError("输出目录已有训练特征，请使用 --resume 或新的 --output 目录")
    if metadata_path.exists():
        if json.loads(metadata_path.read_text())["signature"] != signature:
            raise ValueError("原始录音、清单或编译配置已变化，请指定新的特征目录")
    elif populated:
        raise ValueError("现有特征缺少来源签名，请使用新的特征目录")
    else:
        from .download import atomic_json

        atomic_json(metadata_path, {"signature": signature})


def cached_record(record, config, output, signature, model_loader):
    """复用校验过的特征，缺失时只编译当前样本并原子保存。"""
    import torch

    filename = hashlib.sha256(record["id"].encode()).hexdigest()[:20] + ".pt"
    target = output / filename
    if target.is_file():
        value = torch.load(target, map_location="cpu", weights_only=True)
        if value.get("source_signature") != signature or value["id"] != record["id"]:
            raise ValueError(f"已保存样本来源不一致 {record['id']}")
    else:
        value = prepare_record(model_loader(), config, record)
        value["source_signature"] = signature
        temporary = target.with_suffix(".tmp")
        torch.save(value, temporary)
        temporary.replace(target)
    return {
        "file": filename,
        "id": record["id"],
        "voice_id": value["voice_id"],
        "tokens": len(value["labels"]),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest")
    parser.add_argument("--config")
    parser.add_argument("--output")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config)
    records = load_manifest(args.manifest)
    output = local_path(args.output or config["training"]["prepared_dir"])
    signature = source_signature(args.manifest, records, config)
    initialize_preparation(output, signature, args.resume)
    model = None

    def model_loader():
        nonlocal model
        if model is None:
            model = load_base(config, with_adapter=False)
        return model

    index = []
    for record in records:
        item = cached_record(record, config, output, signature, model_loader)
        index.append(item)
        print(
            f"完成 {record['id']}，{item['tokens']} token，音色 {item['voice_id']}",
            flush=True,
        )
    temporary = output / "index.json.tmp"
    temporary.write_text(
        json.dumps(
            {"fingerprint": model_fingerprint(config), "compiler": compiler_version(config), "samples": index},
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    temporary.replace(output / "index.json")
    print(f"训练特征已保存至 {output}")


if __name__ == "__main__":
    main()
