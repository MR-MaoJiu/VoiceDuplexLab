"""在真实 GPU 上比较特征编译路径，确认省略解码不会改变训练输入。"""

import argparse
import copy
import time

from .config import ROOT, load_config
from .data import load_manifest
from .download import atomic_json
from .model import load_base, model_fingerprint
from .prepare import prepare_record, compiler_version


def main():
    import torch

    parser = argparse.ArgumentParser()
    parser.add_argument("manifest")
    parser.add_argument("--config")
    args = parser.parse_args()
    config = load_config(args.config)
    model = load_base(config, with_adapter=False)
    available = load_manifest(args.manifest)
    records = [available[0]]
    other = next((item for item in available if item.get("voice_id") != records[0].get("voice_id")), None)
    if other:
        records.append(other)
    results = []
    for record in records:
        values = []
        for forward in (True, False):
            options = copy.deepcopy(config)
            options["training"]["compile_llm_forward"] = forward
            torch.cuda.synchronize()
            started = time.monotonic()
            value = prepare_record(model, options, record)
            torch.cuda.synchronize()
            values.append((value, time.monotonic() - started))
        eager, fast = values[0][0], values[1][0]
        identical = torch.equal(eager["labels"], fast["labels"]) and torch.equal(eager["inputs_embeds"], fast["inputs_embeds"])
        results.append({"id": record["id"], "tokens": len(fast["labels"]),
                        "identical": identical, "original_seconds": values[0][1], "fast_seconds": values[1][1]})
    atomic_json(ROOT / "runs/compiler_benchmark.json", {
        "fingerprint": model_fingerprint(config),
        "compiler": compiler_version(config),
        "identical": all(item["identical"] for item in results), "samples": results,
    })
    print("特征编译逐元素一致性验证结果", results, flush=True)


if __name__ == "__main__":
    main()
