import argparse
from collections import Counter
from .data import load_manifest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest")
    args = parser.parse_args()
    records = load_manifest(args.manifest)
    voices = Counter(item.get("voice_id", item["voice"]) for item in records)
    units = [unit for item in records for unit in item["units"]]
    print(
        f"校验通过，{len(records)} 段对话，{len(units)} 个时间单元，{len(voices)} 种参考音色"
    )
    print(
        f"委派单元 {sum(bool(unit.get('delegate')) for unit in units)}，直接说话单元 {sum(bool(unit.get('text')) and not unit.get('delegate') for unit in units)}"
    )
    for voice, count in voices.items():
        print(f"音色 {voice}，{count} 个样本")
    if len(voices) == 1:
        print(
            "提示：当前数据只有一种音色，面向多用户时应加入不同参考音色的训练和留出评估样本。"
        )


if __name__ == "__main__":
    main()
