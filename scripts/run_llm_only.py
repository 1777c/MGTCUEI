#!/usr/bin/env python
"""复现一个仅使用冻结 LLM 的对照实验。

此路线对冻结的 MMed-Llama-3-8B 执行前向计算，再用冻结的 lm_head True/False 两行读出。没有优化器更新或证据注入。请使用对应结果的协议、折号和提示版本。
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mgtcuei.data import Data  # noqa: E402
from mgtcuei.llm_only import LLMOnlyScorer  # noqa: E402
from mgtcuei.metrics import measure  # noqa: E402
from mgtcuei.threshold import select_threshold  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--protocol", required=True,
                   choices=["warm_k10", "drug_cold_k10", "disease_cold_k10", "strict_both_k5"])
    p.add_argument("--fold", type=int, required=True)
    p.add_argument("--prompt", default="semantic_compact_v1",
                   choices=["basic_name_only_v1", "semantic_compact_v1"])
    p.add_argument("--llm-root", default=os.environ.get("LLM_ROOT"))
    p.add_argument("--device", default="cuda")
    p.add_argument("--output", type=Path, default=None)
    args = p.parse_args()
    if not args.llm_root:
        raise SystemExit("请通过 --llm-root 或 LLM_ROOT 指定本地模型目录")
    data = Data(ROOT)
    val = data.partition("val", args.protocol, args.fold)
    test = data.partition("test", args.protocol, args.fold)
    scorer = LLMOnlyScorer(args.llm_root, prompt_style=args.prompt, device=args.device)
    val_pred = scorer.score(data, val)
    threshold = select_threshold(val_pred)["selected_threshold"]
    test_pred = scorer.score(data, test)
    metrics = measure(test_pred, threshold)
    result = {
        "protocol": args.protocol,
        "fold": args.fold,
        "prompt": args.prompt,
        "threshold": threshold,
        "metrics": metrics,
    }
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        test_pred.to_parquet(args.output, index=False)
    print(json.dumps({k: v for k, v in result.items() if k != "metrics"}, indent=2))
    print(json.dumps({k: metrics[k] for k in
                      ("average_precision", "roc_auc", "f1", "accuracy", "precision", "recall")},
                     indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
