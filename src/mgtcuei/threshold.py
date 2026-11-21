"""为每个模型和每折确定冻结分类阈值。

DDA5.2 先完成训练，再在验证分区选择并恢复最优检查点。随后在同一分区的预测中找使 F1 最大的阈值 t*，冻结在训练记录中。测试集只使用 t*。缺少冻结阈值时拒绝评估。
"""

import numpy as np
from scipy.special import expit
from sklearn import metrics as sk
from .utils import ContractError

SELECTION_METRIC = "validation_F1"
TIE_BREAK_RULE = "max validation F1; ties resolved to the HIGHER threshold"
CANDIDATE_RULE = "unique(sigmoid(score)) on the selection partition; every candidate lies in (0,1)"


def f1_curve(prob, label):
    """确定性地计算每个不同预测概率对应的 F1，并处理并列值。"""
    if not len(prob):
        raise ContractError("threshold search requires a non-empty selection partition")
    order = np.argsort(prob, kind="stable")
    prob = prob[order]
    label = label[order]
    positives = float(label.sum())
    negatives = float(len(label) - positives)
    # 从全部判为阳性开始向下扫描，用累计计数计算候选阈值的 F1，避免 O(n^2) 重算。
    tp = np.cumsum(label[::-1])[::-1]
    fp = np.cumsum(1.0 - label[::-1])[::-1]
    fn = positives - tp
    f1 = 2 * tp / np.maximum(1e-30, 2 * tp + fp + fn)
    # 阈值 p 将 prob >= p 的样本判为阳性。等概率连续段首个位置的累计和对应候选阈值，其余位置不作为候选。
    keep = np.r_[True, prob[1:] != prob[:-1]]
    return prob[keep], f1[keep], int(positives), int(negatives)


def select_threshold(frame):
    """在一个验证分区选择 t*。测试集预测不用于该步骤。"""
    if frame.empty or frame.pair_id.duplicated().any() or not frame.label.isin([0, 1]).all():
        raise ContractError("threshold selection prediction schema")
    prob = expit(frame.score.to_numpy(dtype=float))
    candidates, values, positives, negatives = f1_curve(prob, frame.label.to_numpy())
    if not np.isfinite(values).all():
        raise ContractError("nonfinite threshold curve")
    best = values.max()
    tied = candidates[values == best]
    # 并列时确定性地取较高阈值。
    threshold = float(tied.max())
    if not 0.0 < threshold < 1.0:
        raise ContractError("selected threshold outside (0,1)")
    return {
        "selected_threshold": threshold,
        "threshold_selection_metric": SELECTION_METRIC,
        "threshold_tie_break_rule": TIE_BREAK_RULE,
        "threshold_candidate_rule": CANDIDATE_RULE,
        "threshold_candidate_count": int(len(candidates)),
        "threshold_tied_candidate_count": int(len(tied)),
        "threshold_selection_f1": float(best),
        "threshold_selection_f1_at_0.5": float(
            sk.f1_score(frame.label.to_numpy(), prob >= 0.5, zero_division=0)
        ),
        "threshold_selection_positive_count": positives,
        "threshold_selection_negative_count": negatives,
        "threshold_search_partition": "validation",
        "test_threshold_search_forbidden": True,
    }
