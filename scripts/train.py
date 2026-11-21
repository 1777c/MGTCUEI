#!/usr/bin/env python
"""在指定协议和折上拟合一组 MGTCUEI 模型，并评估一次。

Drug-Cold 第 0 折主线，先拟合上游 GCN，再拟合语义提示证据模型。
    python scripts/train.py --protocol drug_cold_k10 --fold 0 --arm evidence_semantic
同一折的 GNN-only 对照。
    python scripts/train.py --protocol drug_cold_k10 --fold 0 --arm gnn_only
在合并的 Dev-FINAL 上运行开发集消融。
    python scripts/train.py --protocol cold --arm evidence_basic         --eval-block Dev-FINAL --cuei-span U30-1L --reader non_query --gate unit --memory fixed_mixed_q1
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mgtcuei.pipeline import Pipeline  # noqa: E402
from mgtcuei.utils import read_json, write_json  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--protocol", required=True,
                   choices=["cold", "warm_k10", "drug_cold_k10",
                            "disease_cold_k10", "strict_both_k5"])
    p.add_argument("--fold", type=int, default=0)
    p.add_argument("--arm", required=True,
                   choices=["gnn_only", "evidence_basic", "evidence_semantic"])
    p.add_argument("--results-root", type=Path, default=ROOT / "results")
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--llm-root", default=os.environ.get("LLM_ROOT"))
    p.add_argument("--cuei-span", default="U29-2L",
                   choices=["U30-1L", "U29-2L", "U28-3L"])
    p.add_argument("--reader", default="cross_attention",
                   choices=["cross_attention", "non_query", "mean"])
    p.add_argument("--gate", default="learned", choices=["learned", "unit"])
    p.add_argument("--memory", default="typed_ddi", choices=["typed_ddi", "fixed_mixed_q1"])
    p.add_argument("--prompt", default=None,
                   choices=["basic_name_only_v1", "semantic_compact_v1"])
    p.add_argument("--eval-block", default=None)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--synthetic", action="store_true", help="在 CPU 上快速检查张量形状和梯度")
    args = p.parse_args()

    protocol = args.protocol
    if protocol == "cold":
        block = args.eval_block or "Dev-FINAL"
    else:
        block = args.eval_block or "test"

    pipe = Pipeline(ROOT, llm_root=args.llm_root, synthetic=args.synthetic)
    fitted, receipt, directory = pipe.run_arm(
        args.arm, protocol, args.fold, args.results_root,
        seed=args.seed, prompt=args.prompt, block=block,
        cuei_span=args.cuei_span, reader=args.reader, gate=args.gate,
        memory=args.memory, resume=args.resume,
    )
    result = {
        "status": "COMPLETE",
        "protocol": protocol,
        "fold": args.fold,
        "arm": args.arm,
        "block": block,
        "run": str(fitted["run"]),
        "evaluation": str(directory),
        "spec_id": receipt["spec_id"],
        "best_epoch": receipt["best_epoch"],
        "checkpoint_sha256": receipt["checkpoint_sha256"],
        "threshold": receipt["threshold"]["selected_threshold"],
        "metrics": read_json(directory / "receipt.json")["metrics"][block],
    }
    write_json(Path(directory) / "public_summary.json", result)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
