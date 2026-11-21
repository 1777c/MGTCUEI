#!/usr/bin/env python
"""运行图 5(f) 和表 S13 的一个语言骨干对照组。"""

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mgtcuei.ablations.common import require_cuda
from mgtcuei.ablations.language.language_arms import load_arms, model_path
from mgtcuei.ablations.language.llama_runner import LanguageArmRunner
from mgtcuei.ablations.language.metrics import seven_metrics
from mgtcuei.ablations.seven_metrics import assert_schema
from mgtcuei.utils import read_json, write_json


def main():
    arms = load_arms(ROOT)
    aliases = {arm["backbone"]: arm_id for arm_id, arm in arms.items()}
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", required=True, choices=[*aliases, *arms])
    parser.add_argument("--stage", required=True, choices=["status", "plan", "fit", "dev-final", "report"])
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    arm = arms[aliases.get(args.arm, args.arm)]
    arm_id = arm["arm_id"]
    path = model_path(ROOT, arm)
    if args.stage == "status":
        result = {"arm": arm_id, "model_directory": str(path), "model_directory_exists": path.is_dir()}
    elif args.stage == "plan":
        matrix = read_json(ROOT / "configs/LANGUAGE_ABLATION_MATRIX.json")
        result = {"arm": arm_id, "partitions": matrix["partitions"], "candidate": arm["candidate"]}
    elif args.stage == "report":
        report = ROOT / "results/ablations/language" / arm_id / "metrics.json"
        if not report.is_file():
            parser.error("请先运行 --stage dev-final，再生成报告")
        result = read_json(report)
    else:
        if not path.is_dir():
            parser.error(f"请将 {arm['root_env']} 设为本地模型目录")
        require_cuda(bf16=arm["backbone"] != "BIOBERT")
        if arm["backbone"] == "BIOBERT":
            from mgtcuei.ablations.language.biobert_runner import BioBERTArmRunner

            runner = BioBERTArmRunner(ROOT, arm)
        else:
            runner = LanguageArmRunner(ROOT, arm)
        if args.stage == "fit":
            result = runner.fit(resume=args.resume)
        else:
            predictions, threshold = runner.dev_final()
            frame = pd.read_parquet(predictions)
            metrics = seven_metrics(frame.label, frame.score, threshold)
            assert_schema(metrics)
            result = {"arm": arm_id, "threshold": threshold, **metrics}
            write_json(runner.selection.with_name("metrics.json"), result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
