"""使用 Dev-A 冻结阈值计算合并后的语言骨干预测指标。"""

import numpy as np

from ...utils import ContractError


def aupr_trapezoidal(y_true, score) -> float:
    """计算 PR 曲线下的梯形面积 AUPR。"""
    from sklearn.metrics import auc, precision_recall_curve

    precision, recall, _ = precision_recall_curve(y_true, score)
    return float(auc(recall, precision))


def probability(score):
    """将已保存的 logit 分数转为预测概率。

    predictions/*.parquet 保存 logit。阈值搜索和分类评估均使用概率空间，决策条件为 sigmoid(score) >= threshold。直接用 logit 与概率阈值比较，会影响 Accuracy、Precision、Recall 和 F1。排序指标不受单调变换影响。
    """
    from scipy.special import expit

    return expit(np.asarray(score, dtype=np.float64))


def seven_metrics(y_true, score, threshold: float) -> dict:
    """使用同一个冻结阈值计算七项报告指标。"""
    from sklearn.metrics import (
        accuracy_score,
        average_precision_score,
        f1_score,
        precision_score,
        recall_score,
        roc_auc_score,
    )

    y_true = np.asarray(y_true)
    score = np.asarray(score, dtype=np.float64)
    if y_true.ndim != 1 or score.shape != y_true.shape:
        raise ContractError("metric input shape")
    if not np.isfinite(score).all():
        raise ContractError("non-finite scores")
    if len(np.unique(y_true)) < 2:
        raise ContractError("both classes required to score")
    prob = probability(score)
    pred = (prob >= threshold).astype(int)
    return {
        "roc_auc": float(roc_auc_score(y_true, score)),
        "aupr": aupr_trapezoidal(y_true, score),
        "average_precision": float(average_precision_score(y_true, score)),
        "accuracy": float(accuracy_score(y_true, pred)),
        "precision": float(precision_score(y_true, pred, zero_division=0)),
        "recall": float(recall_score(y_true, pred, zero_division=0)),
        "f1": float(f1_score(y_true, pred, zero_division=0)),
        "threshold": float(threshold),
        "threshold_source": "validation-derived (sealed training receipt)",
        "score_space": "logit; threshold applied to sigmoid(score)",
        "n": int(len(y_true)),
        "n_positive": int((y_true == 1).sum()),
        "n_negative": int((y_true == 0).sum()),
        "predicted_positive_rate": float(pred.mean()),
    }
