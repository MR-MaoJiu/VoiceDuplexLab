import json
from pathlib import Path
import soundfile as sf


def load_manifest(path):
    path = Path(path).resolve()
    records, seen = [], set()
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), 1
    ):
        if not line.strip():
            continue
        item = json.loads(line)
        identifier = item.get("id")
        if not isinstance(identifier, str) or not identifier or identifier in seen:
            raise ValueError(f"第 {line_number} 行 id 缺失或重复")
        seen.add(identifier)
        for key in ("audio", "voice"):
            if not isinstance(item.get(key), str):
                raise ValueError(f"{identifier} 缺少 {key}")
            value = Path(item[key]).expanduser()
            value = value if value.is_absolute() else path.parent / value
            if not value.is_file():
                raise FileNotFoundError(f"{identifier} 的 {key} 不存在 {value}")
            info = sf.info(value)
            if info.duration <= 0:
                raise ValueError(f"{identifier} 的 {key} 为空")
            item[key] = str(value.resolve())
        units = item.get("units")
        if not isinstance(units, list) or not units:
            raise ValueError(f"{identifier} 缺少时间单元")
        last_end = True
        for index, unit in enumerate(units):
            if not isinstance(unit, dict):
                raise ValueError("时间单元必须是对象")
            if unit.get("listen") is True:
                if unit.get("text") or not last_end:
                    raise ValueError(
                        f"{identifier} 单元 {index} 在回复尚未结束时进入 listen"
                    )
            else:
                text = unit.get("text")
                if not isinstance(text, str) or not text.strip():
                    raise ValueError(f"{identifier} 单元 {index} 缺少回复文本")
                if unit.get("delegate"):
                    if not unit.get("end", False) or not last_end:
                        raise ValueError("委派应为独立、完整的输出轮次")
                last_end = bool(unit.get("end", False))
            if "backend" in unit and not isinstance(unit["backend"], dict):
                raise ValueError("后台结果必须是对象")
        if not last_end:
            raise ValueError(f"{identifier} 最后一个说话单元需要 end=true")
        records.append(item)
    if not records:
        raise ValueError("数据集为空")
    return records
