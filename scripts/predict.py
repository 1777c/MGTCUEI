#!/usr/bin/env python
"""用训练后的 MGTCUEI 检查点评估候选配对。

输入 CSV 须包含 pair_id、drug_idx 和 disease_idx。每个配对须位于训练所用提示版本的冻结缓存中。
输出新增 logit_false、logit_true、score、probability、predicted_label 和 threshold。
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mgtcuei.inference import Scorer  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--protocol", required=True)
    p.add_argument("--fold", type=int, default=0)
    p.add_argument("--pairs", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--llm-root", default=os.environ.get("LLM_ROOT"))
    p.add_argument("--device", default="cuda")
    p.add_argument("--batch-size", type=int, default=32)
    args = p.parse_args()
    frame = pd.read_csv(args.pairs)
    scorer = Scorer(ROOT, args.run, protocol=args.protocol, fold=args.fold,
                    llm_root=args.llm_root, device=args.device)
    result = scorer.score(frame, batch_size=args.batch_size)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output, index=False)
    print(result.head(20).to_string(index=False))
    print(f"已写入 {len(result)} 行 -> {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
