#!/usr/bin/env python
"""在已登记的数据分区上重新评估完成的 MGTCUEI 运行，无需重新拟合。

沿用训练记录中的冻结阈值。分区须对应训练所用协议和折，测试集不重新搜索阈值。
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
from mgtcuei.utils import read_json  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--protocol", required=True)
    p.add_argument("--fold", type=int, default=0)
    p.add_argument("--block", required=True)
    p.add_argument("--llm-root", default=os.environ.get("LLM_ROOT"))
    p.add_argument("--synthetic", action="store_true")
    args = p.parse_args()
    pipe = Pipeline(ROOT, llm_root=args.llm_root, synthetic=args.synthetic)
    fitted = pipe.load_fitted(args.run, args.protocol, args.fold)
    receipt, directory = pipe.evaluate(fitted, args.block, args.protocol, args.fold)
    print(json.dumps({
        "status": receipt["status"],
        "block": args.block,
        "threshold_used": receipt["threshold_used"],
        "test_threshold_search_performed": receipt["test_threshold_search_performed"],
        "metrics": receipt["metrics"][args.block],
        "evaluation": str(directory),
    }, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
