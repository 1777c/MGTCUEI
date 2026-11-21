"""DDA5.2 的正式评估指标。

ranking 计算与阈值无关的 AUROC 和 AP 等指标。AP 使用 sklearn 的 average_precision_score，即按阶梯积分的精确率–召回率面积。classification 使用冻结阈值。threshold.select_threshold 仅在检查点选择分区选阈值。

measure 汇总排序指标及正式阈值下的分类指标，并额外记录阈值 0.5 的诊断值。
"""

import numpy as np
from scipy.special import expit
from sklearn import metrics as sk
from .utils import ContractError

SUPPLEMENTARY_THRESHOLD = 0.5


def _schema(frame):
    if frame.empty or frame.pair_id.duplicated().any() or not frame.label.isin([0, 1]).all():
        raise ContractError("prediction schema")
    model = frame.sort_values("pair_id", kind="stable")
    if not np.isfinite(model.score.to_numpy(dtype=float)).all():
        raise ContractError("nonfinite scores")
    return model


def ece15(prob, y, predicted):
    """在给定分类决策下计算 DDA5.0 的 15 分箱置信度 ECE。"""
    confidence = np.maximum(prob, 1 - prob)
    total = 0.0
    edges = np.linspace(0, 1, 16)
    for lo, hi in zip(edges[:-1], edges[1:]):
        take = (confidence >= lo) & ((confidence <= hi) if hi == 1 else (confidence < hi))
        if take.any():
            total += take.mean() * abs(confidence[take].mean() - (predicted[take] == y[take]).mean())
    return float(total)


def ranking(frame):
    """计算与阈值无关的排序指标。"""
    frame = _schema(frame)
    z = frame.score.to_numpy(dtype=float)
    y = frame.label.to_numpy()
    per = []
    for drug, f in frame.groupby("drug_idx", sort=True):
        f = f.sort_values(["score", "pair_id"], ascending=[False, True], kind="stable")
        yy = f.label.to_numpy()
        ss = f.score.to_numpy()
        positives = yy.sum()
        if positives == 0:
            continue
        rank = np.flatnonzero(yy) + 1
        k = min(10, len(f))
        discount = 1 / np.log2(np.arange(2, k + 2))
        ideal = discount[: min(k, int(positives))].sum()
        per.append(
            {
                "drug_idx": int(drug),
                "ap": float(sk.average_precision_score(yy, ss)),
                "ndcg10": float((yy[:k] * discount).sum() / ideal),
                "recall10": float(yy[:k].sum() / positives),
                "recall5": float(yy[:5].sum() / positives),
                "mrr": float(1 / rank[0]),
            }
        )
    if not per:
        raise ContractError("no positive drug: macro AP undefined")
    result = {
        # AP 系列统一使用 sklearn 的 average_precision_score。
        "average_precision": float(sk.average_precision_score(y, z)),
        "drug_macro_ap": float(np.mean([r["ap"] for r in per])),
        "roc_auc": float(sk.roc_auc_score(y, z)) if len(np.unique(y)) == 2 else None,
        "ndcg10": float(np.mean([r["ndcg10"] for r in per])),
        "recall10": float(np.mean([r["recall10"] for r in per])),
        "recall5": float(np.mean([r["recall5"] for r in per])),
        "mrr": float(np.mean([r["mrr"] for r in per])),
        "positive_drugs": len(per),
        "all_drugs": int(frame.drug_idx.nunique()),
        "coverage": len(per) / frame.drug_idx.nunique(),
        "per_drug": per,
    }
    # 疾病宏平均 AP 沿用药物宏平均 AP 定义并替换实体轴。Dual-Cold 分别报告两个轴。
    diseases = []
    for disease, f in frame.groupby("disease_idx", sort=True):
        f = f.sort_values(["score", "pair_id"], ascending=[False, True], kind="stable")
        yy = f.label.to_numpy()
        positives = int(yy.sum())
        if not positives:
            continue
        k = min(10, len(f))
        discount = 1 / np.log2(np.arange(2, k + 2))
        diseases.append(
            {
                "disease_idx": int(disease),
                "ap": float(sk.average_precision_score(yy, f.score)),
                "ndcg10": float((yy[:k] * discount).sum() / discount[: min(k, positives)].sum()),
                "mrr": float(1 / (np.flatnonzero(yy)[0] + 1)),
                "recall5": float(yy[:5].sum() / positives),
                "recall10": float(yy[:10].sum() / positives),
            }
        )
    result.update(
        disease_macro_ap=float(np.mean([d["ap"] for d in diseases])),
        positive_diseases=len(diseases),
        all_diseases=int(frame.disease_idx.nunique()),
        disease_coverage=len(diseases) / frame.disease_idx.nunique(),
        per_disease=diseases,
        brier=float(sk.brier_score_loss(y, expit(z))),
        threshold_free=True,
        average_precision_implementation="sklearn.metrics.average_precision_score (AP)",
    )
    for metric in ("ndcg10", "mrr", "recall5", "recall10"):
        result["disease_" + metric] = float(np.mean([d[metric] for d in diseases]))
    return result


def classification(frame, threshold, *, suffix=""):
    """计算给定冻结阈值下的分类指标。"""
    frame = _schema(frame)
    if not 0.0 < float(threshold) < 1.0:
        raise ContractError("frozen threshold must lie strictly inside (0,1)")
    z = frame.score.to_numpy(dtype=float)
    y = frame.label.to_numpy()
    prob = expit(z)
    predicted = prob >= float(threshold)
    tp = int(((y == 1) & predicted).sum())
    tn = int(((y == 0) & ~predicted).sum())
    fp = int(((y == 0) & predicted).sum())
    fn = int(((y == 1) & ~predicted).sum())
    return {
        "accuracy" + suffix: float(sk.accuracy_score(y, predicted)),
        "f1" + suffix: float(sk.f1_score(y, predicted, zero_division=0)),
        "precision" + suffix: float(sk.precision_score(y, predicted, zero_division=0)),
        "recall" + suffix: float(sk.recall_score(y, predicted, zero_division=0)),
        "specificity" + suffix: float(tn / (tn + fp)) if (tn + fp) else None,
        "mcc" + suffix: float(sk.matthews_corrcoef(y, predicted)),
        "ece15" + suffix: ece15(prob, y, predicted),
        "confusion" + suffix: {"tp": tp, "tn": tn, "fp": fp, "fn": fn},
        "threshold" + suffix: float(threshold),
    }


def measure(frame, threshold):
    """汇总排序指标、正式阈值 t* 下的分类指标和阈值 0.5 下的补充指标。"""
    result = ranking(frame)
    result.update(classification(frame, threshold))
    result.update(classification(frame, SUPPLEMENTARY_THRESHOLD, suffix="_at_0.5"))
    result.update(
        formal_threshold=float(threshold),
        supplementary_threshold=SUPPLEMENTARY_THRESHOLD,
        supplementary_metrics=["f1_at_0.5", "accuracy_at_0.5", "precision_at_0.5"],
        formal_threshold_dependent_metrics=[
            "f1",
            "precision",
            "recall",
            "specificity",
            "mcc",
            "accuracy",
            "ece15",
        ],
    )
    return result
