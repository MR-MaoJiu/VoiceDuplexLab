import argparse
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
import re
import numpy as np
import soundfile as sf
from scipy.signal import resample_poly
from .download import atomic_json


def clean_segments(annotation, channel):
    segments = []
    for item in annotation.get("instances", []):
        text = re.sub(r"<[^>]*>", "", item.get("text", "")).strip()
        if item.get("channelIndex") != channel or not text:
            continue
        start, end = float(item["start"]), float(item["end"])
        if start < 0 or not end > start or not math.isfinite(end):
            raise ValueError("标注包含无效时间戳")
        segments.append(
            {
                "start": start,
                "end": end,
                "text": text,
                "speaker": item.get("attributes", {}).get("speaker", "unknown"),
                "turn": item.get("attributes", {}).get("turn", "unknown"),
            }
        )
    return sorted(segments, key=lambda item: item["start"])


def timeline_units(segments, start, end):
    """按段级时间戳等比分配字符，保留原始时间轴并记录近似对齐来源。"""
    count = max(1, math.ceil(end - start))
    parts = defaultdict(list)
    for segment in segments:
        if segment["start"] < start or segment["end"] > end:
            continue
        text = segment["text"]
        duration = segment["end"] - segment["start"]
        for index, char in enumerate(text):
            moment = segment["start"] + duration * (index + 0.5) / len(text)
            position = min(count - 1, max(0, int(moment - start)))
            parts[position].append(char)
    units = []
    for index in range(count):
        text = "".join(parts[index])
        if text:
            units.append({"text": text, "end": not bool(parts[index + 1])})
        else:
            units.append({"listen": True})
    return units


def reference_audio(audio, rate, channel, segments):
    candidates = [item for item in segments if item["end"] - item["start"] >= 1.5]
    if not candidates:
        return None
    segment = max(candidates, key=lambda item: min(8, item["end"] - item["start"]))
    start = segment["start"]
    end = min(segment["end"], start + 8)
    return audio[int(start * rate) : int(end * rate), channel], start, end


def convert_pair(audio_path, annotation_path, output, window_seconds=24, max_windows=6):
    if window_seconds <= 0 or max_windows < 0:
        raise ValueError("窗口时长必须大于0，max_windows 可以为0以使用全部窗口")
    annotation = json.loads(Path(annotation_path).read_text(encoding="utf-8"))
    audio, rate = sf.read(audio_path, dtype="float32", always_2d=True)
    if audio.shape[1] != 2:
        raise ValueError("当前转换器要求独立的双声道对话录音")
    if rate != 16000:
        divisor = math.gcd(rate, 16000)
        audio = resample_poly(audio, 16000 // divisor, rate // divisor, axis=0)
        rate = 16000
    duration = len(audio) / rate
    group = hashlib.sha256(str(Path(annotation_path).name).encode()).hexdigest()[:16]
    records = []
    for channel in (0, 1):
        segments = clean_segments(annotation, channel)
        speakers = {
            item["speaker"] for item in segments if item["speaker"] != "unknown"
        }
        if len(speakers) != 1:
            continue
        reference = reference_audio(audio, rate, channel, segments)
        if reference is None:
            continue
        voice, ref_start, ref_end = reference
        voice_path = output / "voices" / f"{group}-ch{channel}.wav"
        voice_path.parent.mkdir(parents=True, exist_ok=True)
        sf.write(voice_path, voice, rate, subtype="PCM_16")
        windows = 0
        # 以真实发言边界结束窗口，避免把半句截断后标为完整回答。
        start = 0.0
        while start < duration and (max_windows == 0 or windows < max_windows):
            choices = [
                item["end"]
                for item in segments
                if start < item["end"] <= min(start + window_seconds, duration)
            ]
            if not choices:
                start += window_seconds
                continue
            end = max(choices)
            chosen = [
                item
                for item in segments
                if item["start"] >= start and item["end"] <= end
            ]
            # 参考片段与监督窗口分离，避免直接复述参考录音。
            if not chosen or (start < ref_end and end > ref_start):
                start = end + 0.05
                continue
            units = timeline_units(chosen, start, end)
            if not any(unit.get("text") for unit in units):
                start = end + 0.05
                continue
            identifier = f"smooth-{group}-ch{channel}-{windows:03d}"
            input_audio = audio[int(start * rate) : int(end * rate), 1 - channel]
            needed = int((len(units) + 0.1) * rate)
            input_audio = np.pad(input_audio, (0, max(0, needed - len(input_audio))))
            path = output / "audio" / f"{identifier}.wav"
            path.parent.mkdir(parents=True, exist_ok=True)
            sf.write(path, input_audio, rate, subtype="PCM_16")
            records.append(
                {
                    "id": identifier,
                    "group": group,
                    "voice_id": f"{group}-ch{channel}",
                    "voice": str(voice_path.resolve()),
                    "audio": str(path.resolve()),
                    "units": units,
                    "provenance": {
                        "kind": "real_conversation",
                        "audio": str(Path(audio_path).name),
                        "channel": channel,
                        "start": start,
                        "end": end,
                        "alignment": "segment-proportional",
                        "turn_mapping": "adjacent-one-second-text-units",
                    },
                }
            )
            windows += 1
            start = end + 0.05
    return records


def write_jsonl(path, records):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        "".join(json.dumps(item, ensure_ascii=False) + "\n" for item in records),
        encoding="utf-8",
    )
    temporary.replace(path)


def convert_corpus(raw, output, window_seconds=24, max_windows=6):
    raw, output = Path(raw), Path(output)
    source = json.loads((raw / "download_manifest.json").read_text())
    records, skipped = [], []
    for pair in source["pairs"]:
        values = convert_pair(
            raw / pair["audio"],
            raw / pair["annotation"],
            output,
            window_seconds,
            max_windows,
        )
        records.extend(values)
        if not values:
            skipped.append(pair["audio"])
    if not records:
        raise ValueError("选中的录音没有转换出有效样本，请增加 clips 或检查标注")
    groups = sorted({item["group"] for item in records})
    evaluation_groups = (
        set(groups[-max(1, len(groups) // 5) :]) if len(groups) > 1 else set()
    )
    train = [item for item in records if item["group"] not in evaluation_groups]
    evaluation = [item for item in records if item["group"] in evaluation_groups]
    write_jsonl(output / "real_train.jsonl", train)
    write_jsonl(output / "real_eval.jsonl", evaluation)
    atomic_json(
        output / "corpus_report.json",
        {
            "source": source,
            "train_samples": len(train),
            "eval_samples": len(evaluation),
            "reference_voices": len({item["voice_id"] for item in records}),
            "skipped": skipped,
            "alignment": "段级时间戳内按字符等比分配，秒级近似对齐",
            "turn_mapping": "相邻秒内连续文本组成一个监督说话轮次",
        },
    )
    return train, evaluation


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", default="data/raw/smoothconv")
    parser.add_argument("--output", default="data/automatic")
    parser.add_argument("--window-seconds", type=int, default=24)
    parser.add_argument(
        "--max-windows", type=int, default=6, help="每位说话人的窗口上限，0表示全部"
    )
    args = parser.parse_args()
    train, evaluation = convert_corpus(
        args.raw, args.output, args.window_seconds, args.max_windows
    )
    print(f"真实对话转换完成，训练 {len(train)} 条，评估 {len(evaluation)} 条")


if __name__ == "__main__":
    main()
