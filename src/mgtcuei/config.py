"""MGTCUEI 的冻结模型标识和实验规则。

MGTCUEI 的正式架构为 PURE_CUEI，语义提示配置为 T2_ORIG_SEMANTIC。所有协议使用同一检查点选择指标。
"""

from __future__ import annotations

import json
from pathlib import Path

from .utils import ContractError, read_json

PROJECT_NAME = "MGTCUEI"
PROJECT_LABEL = "MGTCUEI (DDA5.2 PURE-CUEI T2_ORIG_SEMANTIC)"
VERSION = "1.0.0"
SOURCE_RELEASE = "DDA5.2 FINAL_RELEASE_V1"
FROZEN_DDA52_SOURCE_IDENTITY = (
    "a07d0add0a0f3dd934405b183c50d8032b673d71dd4ad9cbc8aa9e99d22e897c"
)

CAMPAIGN = "DDA5_2_PURE_CUEI"
MAINLINE = "T2_ORIG_SEMANTIC"

#: 所有协议和训练阶段使用同一个检查点选择指标。
PRIMARY_SELECTION_METRIC = "average_precision"
CHECKPOINT_SECONDARY_METRIC = "roc_auc"
CHECKPOINT_TIE_BREAK = (
    "1: max validation average_precision (pooled Average Precision, all protocols)",
    "2: max validation roc_auc",
    "3: min validation loss",
    "4: earlier epoch",
)

PROTOCOLS = {
    "warm_k10": {"folds": 10, "phase": "P7", "label": "Warm"},
    "drug_cold_k10": {"folds": 10, "phase": "P6", "label": "Drug-Cold"},
    "disease_cold_k10": {"folds": 10, "phase": "P8", "label": "Disease-Cold"},
    "strict_both_k5": {"folds": 5, "phase": "P9", "label": "Dual-Cold"},
}

PROMPTS = ("basic_name_only_v1", "semantic_compact_v1")
MAINLINE_PROMPT = "semantic_compact_v1"

PR_PRIMARY_METRIC_POLICY = {
    "cold": PRIMARY_SELECTION_METRIC,
    **{name: PRIMARY_SELECTION_METRIC for name in PROTOCOLS},
}

THRESHOLD_POLICY = {
    "selection_partition": (
        "the same partition used for checkpoint selection "
        "(Dev-A for protocol 'cold', fold-local Validation for benchmark folds)"
    ),
    "selection_metric": "F1",
    "tie_break": "max F1; ties resolved to the HIGHER threshold",
    "test_search_forbidden": True,
    "supplementary_debug_threshold": 0.5,
}

TRAINABLE_GROUPS = ("gcn", "memory", "reader", "gate")
FROZEN_GROUPS = ("llm", "lm_head_rows")


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def load_json(relative: str):
    return read_json(project_root() / relative)


def model_config():
    return load_json("configs/model.json")


def training_config():
    return load_json("configs/training.json")


def data_config():
    return load_json("configs/data.json")


def checkpoint_metric_for_protocol(protocol: str) -> str:
    if protocol not in PR_PRIMARY_METRIC_POLICY:
        raise ContractError("no registered checkpoint metric for protocol: " + str(protocol))
    metric = PR_PRIMARY_METRIC_POLICY[protocol]
    if metric != PRIMARY_SELECTION_METRIC:
        raise ContractError(
            "protocol %r declares %r; MGTCUEI selects on %r only"
            % (protocol, metric, PRIMARY_SELECTION_METRIC)
        )
    return metric


def candidate(
    *,
    cuei_span: str = "U29-2L",
    reader: str = "cross_attention",
    gate: str = "learned",
    scale: float = 0.01,
    memory: str = "typed_ddi",
    seed: int = 2026,
    prompt: str = MAINLINE_PROMPT,
):
    """构造 MGTCUEI 候选配置。每个字段对应一个 CUEI 组件。"""
    spans = {"U30-1L": [30], "U29-2L": [29, 30], "U28-3L": [28, 29, 30]}
    if cuei_span not in spans:
        raise ContractError("unregistered CUEI span")
    if reader not in ("cross_attention", "non_query", "mean"):
        raise ContractError("unregistered reader")
    if gate not in ("unit", "learned"):
        raise ContractError("unregistered gate")
    if memory not in ("typed_ddi", "fixed_mixed_q1"):
        raise ContractError("unregistered memory")
    if prompt not in PROMPTS:
        raise ContractError("unregistered prompt")
    if abs(float(scale) - 0.01) > 0:
        raise ContractError("the registered gate initialisation is 0.01")
    return {
        "cuei_span": cuei_span,
        "injection_layers": list(spans[cuei_span]),
        "reader": reader,
        "gate": gate,
        "lambda": float(scale),
        "memory": memory,
        "seed": int(seed),
        "prompt": prompt,
        "gcn_lifecycle": "prefit_then_joint",
    }
