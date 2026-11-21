#!/usr/bin/env python
"""复现正式基准实验矩阵，涵盖四种协议及三组模型。

P6、P7、P8 各拟合 30 次，P9 拟合 15 次，共 105 次。每折拟合一次上游 GCN-ORIG，作为 GNN-only 组及两组证据模型的初始化，再分别拟合两组证据模型。各组以验证集冻结阈值在本折测试集评估一次。

完整运行需要 GPU 和冻结的 8B LLM。使用 --dry-run 可在不加载模型的情况下查看计划。
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mgtcuei.config import PROTOCOLS  # noqa: E402

ARMS = ("gnn_only", "evidence_basic", "evidence_semantic")


def plan():
    rows = []
    for protocol, cfg in PROTOCOLS.items():
        for fold in range(cfg["folds"]):
            for arm in ARMS:
                rows.append({"protocol": protocol, "phase": cfg["phase"], "fold": fold, "arm": arm})
    return rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--results-root", type=Path, default=ROOT / "results/formal_benchmark")
    p.add_argument("--llm-root", default=os.environ.get("LLM_ROOT"))
    p.add_argument("--protocol", action="append", default=None,
                   help="仅运行指定协议，可重复传入，默认包含全部四种协议")
    p.add_argument("--fold", type=int, action="append", default=None,
                   help="仅运行指定折，可重复传入")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()
    rows = plan()
    if args.protocol:
        rows = [r for r in rows if r["protocol"] in args.protocol]
    if args.fold is not None:
        rows = [r for r in rows if r["fold"] in args.fold]
    if args.dry_run:
        print(json.dumps({"fits": len(rows), "plan": rows}, indent=2))
        return 0
    if not args.llm_root:
        raise SystemExit("请通过 --llm-root 或 LLM_ROOT 指定本地模型目录")
    from mgtcuei.pipeline import Pipeline
    from mgtcuei.utils import read_json

    pipe = Pipeline(ROOT, llm_root=args.llm_root)
    records = []
    for row in rows:
        fitted, receipt, directory = pipe.run_arm(
            row["arm"], row["protocol"], row["fold"], args.results_root, block="test"
        )
        metrics = read_json(directory / "receipt.json")["metrics"]["test"]
        records.append({
            **row,
            "run": str(fitted["run"]),
            "evaluation": str(directory),
            "checkpoint_sha256": receipt["checkpoint_sha256"],
            "best_epoch": receipt["best_epoch"],
            "threshold": receipt["threshold"]["selected_threshold"],
            "average_precision": metrics["average_precision"],
            "roc_auc": metrics["roc_auc"],
            "f1": metrics["f1"],
            "accuracy": metrics["accuracy"],
            "precision": metrics["precision"],
            "recall": metrics["recall"],
        })
        out = args.results_root / "reproduction_summary.csv"
        out.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(records).to_csv(out, index=False)
        print(json.dumps(records[-1], indent=2), flush=True)
    print(f"已写入 {len(records)} 行 -> {args.results_root / 'reproduction_summary.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
