#!/usr/bin/env python3
"""补充实验的七项报告指标。

与阈值无关的指标包括基于原始分数的 roc_auc、梯形积分 AUPR 和 sklearn.average_precision_score。AP 使用阶梯积分，与按插值积分的 AUPR 分别报告。

accuracy、precision、recall 和 f1 使用冻结阈值 t*。阈值位于概率空间，预测规则为 sigmoid(score) >= t*，与 mgtcuei.metrics.classification 一致。
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.special import expit
from sklearn import metrics as sk

SEVEN = ("roc_auc", "aupr", "average_precision", "accuracy", "precision", "recall", "f1")
THRESHOLD_FREE = ("roc_auc", "aupr", "average_precision")
THRESHOLD_DEPENDENT = ("accuracy", "precision", "recall", "f1")
#: PR 曲线按 sklearn 的阈值顺序处理并列分数。梯形 AUPR 基于排序，logits 与对应概率给出相同结果。
AUPR_DEFINITION = "trapezoidal area under the full precision-recall curve"


def aupr_trapezoid(y: np.ndarray, s: np.ndarray) -> float:
    precision, recall, _ = sk.precision_recall_curve(y, s)
    # sklearn 返回降序召回率。按升序做梯形积分可得到正面积，并纳入末端锚点 (recall=0, precision=1)。
    trapezoid = np.trapezoid if hasattr(np, "trapezoid") else np.trapz
    return float(trapezoid(precision[::-1], recall[::-1]))


def seven(frame: pd.DataFrame, threshold: float) -> dict:
    if frame.empty or frame.pair_id.duplicated().any():
        raise ValueError("prediction schema: empty or duplicated pair_id")
    y = frame.label.to_numpy()
    z = frame.score.to_numpy(dtype=float)
    if not np.isfinite(z).all():
        raise ValueError("nonfinite scores")
    if len(np.unique(y)) != 2:
        raise ValueError("both classes required for a ranking metric")
    predicted = expit(z) >= float(threshold)
    return {
        "roc_auc": float(sk.roc_auc_score(y, z)),
        "aupr": aupr_trapezoid(y, z),
        "average_precision": float(sk.average_precision_score(y, z)),
        "accuracy": float(sk.accuracy_score(y, predicted)),
        "precision": float(sk.precision_score(y, predicted, zero_division=0)),
        "recall": float(sk.recall_score(y, predicted, zero_division=0)),
        "f1": float(sk.f1_score(y, predicted, zero_division=0)),
    }


def schema_keys() -> tuple:
    return SEVEN


def assert_schema(record: dict) -> bool:
    """检查七项报告指标均为有限浮点数。"""
    missing = [k for k in SEVEN if k not in record]
    if missing:
        raise ValueError(f"seven-metric schema incomplete: missing {missing}")
    for k in SEVEN:
        v = record[k]
        if not isinstance(v, float) or not np.isfinite(v):
            raise ValueError(f"{k} is not a finite float: {v!r}")
    return True
