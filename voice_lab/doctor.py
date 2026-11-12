import argparse
from importlib.metadata import PackageNotFoundError, version
import platform
import sys
from .config import load_config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", action="store_true")
    parser.add_argument("--config")
    args = parser.parse_args()
    config = load_config(args.config)
    failures = []
    print(f"系统 {platform.system()}，Python {platform.python_version()}")
    if not (3, 11) <= sys.version_info[:2] <= (3, 12):
        failures.append("需要 Python 3.11 或 3.12")
    packages = ["fastapi", "httpx", "websockets", "numpy", "soundfile", "scipy"]
    if args.gpu:
        packages += ["torch", "transformers", "peft", "minicpmo-utils"]
    for name in packages:
        try:
            print(f"{name} {version(name)}")
        except PackageNotFoundError:
            failures.append(f"缺少依赖 {name}")
    if args.gpu:
        try:
            import torch
            from transformers import Qwen3ForCausalLM

            print(f"模型架构 {Qwen3ForCausalLM.__name__}")

            if not torch.cuda.is_available():
                failures.append("CUDA 不可用")
            for index in range(torch.cuda.device_count()):
                free, total = torch.cuda.mem_get_info(index)
                print(
                    f"GPU {index} {torch.cuda.get_device_name(index)}，空闲 {free / 2**30:.1f} GiB，总计 {total / 2**30:.1f} GiB"
                )
        except ImportError as exc:
            failures.append(f"GPU 依赖无法导入 {exc}")
    print(f"语音基座版本 {config['model']['revision']}")
    print(
        "数据与模型会保存在项目目录及 Hugging Face 缓存中，请预留下载和训练检查点空间。"
    )
    if failures:
        for reason in failures:
            print("待处理 " + reason)
        raise SystemExit(1)
    print("检查通过")


if __name__ == "__main__":
    main()
