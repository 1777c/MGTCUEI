#!/usr/bin/env python
"""验证冻结的 MGTCUEI 数据及其依赖，并输出报告。"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mgtcuei.data import Data  # noqa: E402
from mgtcuei.utils import write_json  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, default=ROOT)
    p.add_argument("--output", type=Path, default=None)
    p.add_argument("--no-strict", action="store_true", help="报告问题时不返回非零退出码")
    args = p.parse_args()
    data = Data(args.root)
    try:
        report = data.verify()
    except Exception as exc:  # noqa: BLE001
        report = {"status": "FAIL", "error": "%s: %s" % (type(exc).__name__, exc)}
        print(json.dumps(report, indent=2, ensure_ascii=False))
        return 1 if not args.no_strict else 0
    report["node_features_sha256"] = data.files["data/node_features.npy"]
    report["a_ss_sha256"] = data.files["data/A_ss_sparse.npz"]
    report["prompt_rows"] = {
        style: len(data.prompts(style)) for style in ("basic_name_only_v1", "semantic_compact_v1")
    }
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if args.output:
        write_json(args.output, report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
