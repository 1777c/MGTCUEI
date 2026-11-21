#!/usr/bin/env python
"""运行图 5(e) 和表 S12 的图编码器对照实验。"""

import argparse
import json
import os
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mgtcuei.ablations.common import require_cuda
from mgtcuei.ablations.graph_variants import ARMS
from mgtcuei.ablations.seven_metrics import assert_schema, seven
from mgtcuei.ablations.variant_graph_runner import VariantRunner
from mgtcuei.utils import write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arms", default="GE01,GE03,GE04,GE05,GE06", help="可选实验组：" + ",".join(ARMS))
    parser.add_argument("--stages", default="A,B,FINAL", help="选择 A、B、FINAL 中的阶段")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    arms = [arm.strip() for arm in args.arms.split(",")]
    stages = {stage.strip().upper() for stage in args.stages.split(",")}
    if not set(arms) <= ARMS.keys():
        parser.error(f"--arms 只接受以下实验组：{', '.join(ARMS)}")
    if not stages <= {"A", "B", "FINAL"}:
        parser.error("--stages 只接受 A、B 或 FINAL")

    llm_root = os.environ.get("LLM_ROOT")
    if stages & {"B", "FINAL"} and (not llm_root or not Path(llm_root).is_dir()):
        parser.error("请将 LLM_ROOT 设为本地 MMed-Llama-3-8B 模型目录")
    require_cuda(bf16=bool(stages & {"B", "FINAL"}))
    for arm in arms:
        runner = VariantRunner(ROOT, arm, llm_root)
        if "B" in stages:
            print(json.dumps({"arm": arm, **runner.fit(resume=args.resume)}))
        elif "A" in stages:
            print(json.dumps({"arm": arm, "prefit": str(runner.prefit(resume=args.resume))}))
        if "FINAL" in stages:
            predictions, threshold = runner.dev_final()
            metrics = seven(pd.read_parquet(predictions), threshold)
            assert_schema(metrics)
            result = {"arm": arm, "threshold": threshold, **metrics}
            write_json(runner.selection.with_name("metrics.json"), result)
            print(json.dumps(result))


if __name__ == "__main__":
    main()
