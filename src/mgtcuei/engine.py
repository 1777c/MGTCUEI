"""在公共训练集上拟合候选模型，在 Dev-A 选检查点，再独立评估。

训练标识与评估分区无关，拟合结果可用于多个评估分区。检查点按 config.CHECKPOINT_TIE_BREAK 的四级规则选择，分类阈值在相同分区上选择一次，冻结在训练记录中供评估读取。
"""

import csv
import json
import os
import time
import uuid
from typing import Any
import pandas as pd
import torch
from . import checkpoint
from .config import (
    CHECKPOINT_SECONDARY_METRIC,
    CHECKPOINT_TIE_BREAK,
    checkpoint_metric_for_protocol,
)
from .metrics import measure
from .threshold import select_threshold
from .utils import identity, read_json, write_json, sha256, ContractError

#: 按固定顺序保存每轮指标，并按候选配置的注入层附加门控诊断。
BASE_HISTORY_COLUMNS = [
    "epoch",
    "train_loss",
    "validation_loss",
    "roc_auc",
    "average_precision",
    "drug_macro_ap",
    "disease_macro_ap",
    "precision",
    "recall",
    "specificity",
    "accuracy",
    "f1",
    "mcc",
    "brier",
    "ece15",
    "f1_at_0.5",
    "ndcg10",
    "mrr",
    "recall5",
    "recall10",
    "disease_ndcg10",
    "disease_mrr",
    "disease_recall5",
    "disease_recall10",
    "positive_drugs",
    "drug_coverage",
    "positive_diseases",
    "disease_coverage",
    "checkpoint_metric_name",
    "checkpoint_metric_value",
    "checkpoint_secondary_metric_name",
    "checkpoint_secondary_value",
    "epoch_reporting_threshold",
    "learning_rate",
    "is_best",
    "best_metric_so_far",
    "bad_epochs",
    "gradient_max_norm",
    "gradient_total_norm",
    "parameter_update_norm",
    "epoch_runtime_seconds",
    "peak_vram_bytes",
    "epoch_train_samples",
]

GATE_HISTORY_SUFFIXES = ["lambda", "delta_rms", "delta_hidden_ratio", "cosine", "clamp_rate"]


def atomic_frame(path, frame):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    frame.to_parquet(tmp, index=False)
    os.replace(tmp, path)


def predict(model, frame, forward, batch_size=32):
    model.eval()
    parts = []
    with torch.no_grad():
        for start in range(0, len(frame), batch_size):
            f = frame.iloc[start : start + batch_size]
            logits = forward(f).float().cpu().numpy()
            p = f[["pair_id", "drug_idx", "disease_idx", "label"]].copy()
            p["logit_false"] = logits[:, 0]
            p["logit_true"] = logits[:, 1]
            p["score"] = logits[:, 1] - logits[:, 0]
            parts.append(p)
    return pd.concat(parts, ignore_index=True)


def validation_loss(prediction):
    """复用已有验证 logits 计算平均交叉熵，省去一次 LLM 前向计算。"""
    y = torch.as_tensor(prediction.label.to_numpy(), dtype=torch.long)
    z = torch.as_tensor(
        prediction[["logit_false", "logit_true"]].to_numpy(), dtype=torch.float32
    )
    return float(torch.nn.functional.cross_entropy(z, y))


def resolve_checkpoint_metric(spec):
    """确保运行声明的指标与协议登记的指标一致。"""
    protocol = spec.get("protocol")
    expected = checkpoint_metric_for_protocol(protocol)
    metric = spec.get("checkpoint_metric", expected)
    if metric != expected:
        raise ContractError(
            "checkpoint metric %r does not match protocol %r policy %r" % (metric, protocol, expected)
        )
    return metric


def checkpoint_key(metric, metrics, loss):
    """将规定的四级选择规则写成字典序键。

    依次比较合并 AP、AUROC、较低的验证损失和较早的轮次。AUROC 用于平局裁决和报告，单独于 AP。调用方仅在键严格变大时更新，因此完全平局保留最早一轮。AP 和 AUROC 的计算独立于分类阈值。
    """
    primary = metrics.get(metric)
    secondary = metrics.get(CHECKPOINT_SECONDARY_METRIC)
    if primary is None or not torch.isfinite(torch.tensor(float(primary))):
        raise ContractError("checkpoint primary metric undefined at this epoch")
    return (
        float(primary),
        float(secondary) if secondary is not None else float("-inf"),
        -float(loss),
    )


def _gate_columns(model):
    if not hasattr(model, "gates"):
        return []
    return [
        "gate_%s_%s" % (layer, suffix)
        for layer in sorted(model.gates, key=int)
        for suffix in GATE_HISTORY_SUFFIXES
    ]


def _gate_values(model):
    row = {}
    if not hasattr(model, "gates"):
        return row
    for layer, gate in sorted(model.gates.items(), key=lambda kv: int(kv[0])):
        last = gate.last or {}
        for suffix in GATE_HISTORY_SUFFIXES:
            row["gate_%s_%s" % (layer, suffix)] = last.get(suffix)
    return row


def _read_history(run):
    path = run / "history.jsonl"
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def _write_history(run, rows):
    """仅用已完成轮次重写两份历史文件。

    每轮结束时落盘，保证中断恢复不会重复写入或丢失已完成的记录。
    """
    columns = list(BASE_HISTORY_COLUMNS)
    for row in rows:
        for key in row:
            if key not in columns:
                columns.append(key)
    tmp = run / "history.jsonl.tmp"
    with tmp.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(tmp, run / "history.jsonl")
    csv_tmp = run / "history.csv.tmp"
    with csv_tmp.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in columns})
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(csv_tmp, run / "history.csv")


def _group_grad_norms(model, gradients):
    """记录每组参数在该轮的峰值梯度范数。

    分开记录四组梯度，可辨别证据交叉熵实际更新的组件。联合训练时，gcn_grad_norm 应为非零。
    """
    out = {}
    for group, prefix in (("gcn", "gcn."), ("memory", "memory."), ("reader", "reader."), ("gate", "gates.")):
        found = [v for n, v in gradients.items() if n.startswith(prefix)]
        out[group + "_grad_norm"] = max(found) if found else 0.0
        out[group + "_grad_tensors"] = len(found)
    return out


def train(run, model, train, tune, forward, spec, schedule, *, resume=False):
    """拟合一个候选模型，保存 spec.json、最优和最近检查点、训练历史及训练记录。

    使用 Dev-A 或本折验证集选择检查点，训练过程独立于评估分区。
    """
    metric = resolve_checkpoint_metric(spec)
    sid = identity(spec)
    run.mkdir(parents=True, exist_ok=True)
    if (run / "spec.json").exists():
        if read_json(run / "spec.json") != spec:
            raise ContractError("reuse identity mismatch")
        if (run / "receipt.json").exists():
            receipt = read_json(run / "receipt.json")
            if receipt["spec_id"] != sid or any(sha256(run / p) != h for p, h in receipt["files"].items()):
                raise ContractError("sealed result changed")
            if "threshold" not in receipt:
                raise ContractError("运行记录缺少验证集分类阈值")
            return receipt
        if not resume:
            raise ContractError("incomplete run requires --resume")
    else:
        write_json(run / "spec.json", spec, exclusive=True)
    optim = torch.optim.Adam(
        [p for p in model.parameters() if p.requires_grad],
        lr=schedule["learning_rate"],
        weight_decay=schedule["weight_decay"],
    )
    state: dict[str, Any] = {
        "epoch": 0,
        "cursor": 0,
        "best": None,
        "best_key": None,
        "bad_epochs": 0,
        "best_epoch": None,
        "steps": 0,
        "epoch_loss_sum": 0.0,
        "epoch_samples": 0,
    }
    if (run / "last.pt").exists():
        state = checkpoint.restore(run / "last.pt", model, optim, sid)
        state.setdefault("best", None)
        state.setdefault("best_key", None)
        state.setdefault("best_epoch", None)
        state.setdefault("epoch_loss_sum", 0.0)
        state.setdefault("epoch_samples", 0)
    # 恢复时丢弃恢复点及之后的历史行，避免部分完成的轮次重复落盘。
    history = [row for row in _read_history(run) if row.get("epoch", -1) < state["epoch"]]
    gradients: dict[str, float] = {}
    before: dict[str, torch.Tensor] = {
        n: p.detach().cpu().clone() for n, p in model.named_parameters() if p.requires_grad
    }
    start_time = time.monotonic()
    peak_vram = 0
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    write_json(run / "status.json", {"state": "RUNNING", "spec_id": sid})
    attempt = run / "attempts" / (uuid.uuid4().hex + ".json")
    write_json(attempt, {"state": "RUNNING", "spec_id": sid, "resume": resume}, exclusive=True)
    try:
        for epoch in range(state["epoch"], schedule["max_epochs"]):
            if state["bad_epochs"] >= schedule["patience"]:
                break
            epoch_start = time.monotonic()
            generator = torch.Generator().manual_seed(spec["seed"] + epoch + 701)
            order = torch.randperm(len(train), generator=generator).numpy()
            offset = state["cursor"] if epoch == state["epoch"] else 0
            if offset == 0:
                state["epoch_loss_sum"] = 0.0
                state["epoch_samples"] = 0
            before = {
                n: p.detach().cpu().clone() for n, p in model.named_parameters() if p.requires_grad
            }
            model.train()
            gradient_total = 0.0
            peak_batch_grad = 0.0
            for pos in range(offset, len(train), schedule["batch_size"]):
                batch = train.iloc[order[pos : pos + schedule["batch_size"]]]
                optim.zero_grad(set_to_none=True)
                for i in range(0, len(batch), schedule["microbatch"]):
                    f = batch.iloc[i : i + schedule["microbatch"]]
                    logits = forward(f)
                    y = torch.as_tensor(f.label.to_numpy(), device=logits.device, dtype=torch.long)
                    loss = torch.nn.functional.cross_entropy(logits.float(), y, reduction="sum") / len(batch)
                    if not torch.isfinite(loss):
                        raise ContractError("nonfinite training loss")
                    loss.backward()
                # 累计各批次总损失和样本数，以计算轮次平均损失。
                state["epoch_loss_sum"] += float(loss.detach()) * len(batch)
                state["epoch_samples"] += len(batch)
                for name, p in model.named_parameters():
                    if p.grad is not None:
                        if not torch.isfinite(p.grad).all():
                            raise ContractError("nonfinite gradient " + name)
                        norm = float(p.grad.norm())
                        gradients[name] = max(gradients.get(name, 0.0), norm)
                        peak_batch_grad = max(peak_batch_grad, norm)
                total = torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad],
                    schedule["gradient_clip"],
                    error_if_nonfinite=True,
                )
                gradient_total = max(gradient_total, float(total))
                optim.step()
                state.update(epoch=epoch, cursor=pos + len(batch), steps=state["steps"] + 1)
                if state["steps"] % schedule["checkpoint_every_steps"] == 0:
                    checkpoint.save(run / "last.pt", model, optim, sid, state)
            # 复用验证预测计算各项检查点选择指标。
            tuning_prediction = predict(model, tune, forward, schedule["eval_batch_size"])
            metrics = measure(tuning_prediction, _epoch_report_threshold(tuning_prediction))
            loss_value = validation_loss(tuning_prediction)
            key = checkpoint_key(metric, metrics, loss_value)
            improved = state["best_key"] is None or key > tuple(state["best_key"])
            if improved:
                # 使用严格大于比较，完整平局时保留最早轮次。
                state.update(best=key[0], best_key=list(key), bad_epochs=0, best_epoch=epoch)
            else:
                state["bad_epochs"] += 1
            update_norm = float(
                torch.sqrt(
                    sum(
                        (p.detach().cpu() - before[n]).pow(2).sum()
                        for n, p in model.named_parameters()
                        if p.requires_grad and n in before
                    )
                )
            )
            elapsed = time.monotonic() - epoch_start
            if torch.cuda.is_available():
                peak_vram = max(peak_vram, int(torch.cuda.max_memory_allocated()))
            row = {
                "epoch": epoch,
                "train_loss": state["epoch_loss_sum"] / max(1, state["epoch_samples"]),
                "validation_loss": loss_value,
                "checkpoint_metric_name": metric,
                "checkpoint_metric_value": key[0],
                "checkpoint_secondary_metric_name": CHECKPOINT_SECONDARY_METRIC,
                "checkpoint_secondary_value": key[1],
                "epoch_reporting_threshold": metrics["formal_threshold"],
                "learning_rate": float(optim.param_groups[0]["lr"]),
                "is_best": bool(improved),
                "best_metric_so_far": state["best"],
                "bad_epochs": state["bad_epochs"],
                "gradient_max_norm": peak_batch_grad,
                "gradient_total_norm": gradient_total,
                "parameter_update_norm": update_norm,
                "epoch_runtime_seconds": elapsed,
                "peak_vram_bytes": peak_vram,
                "epoch_train_samples": state["epoch_samples"],
                **_gate_values(model),
                **_group_grad_norms(model, gradients),
            }
            for key_name in (
                "roc_auc",
                "average_precision",
                "drug_macro_ap",
                "disease_macro_ap",
                "precision",
                "recall",
                "specificity",
                "accuracy",
                "f1",
                "mcc",
                "brier",
                "ece15",
                "f1_at_0.5",
                "ndcg10",
                "mrr",
                "recall5",
                "recall10",
                "disease_ndcg10",
                "disease_mrr",
                "disease_recall5",
                "disease_recall10",
                "positive_drugs",
                "drug_coverage",
                "positive_diseases",
                "disease_coverage",
            ):
                row[key_name] = metrics.get(key_name)
            history.append(row)
            _write_history(run, history)
            if improved:
                checkpoint.save(run / "best.pt", model, optim, sid, state)
            state.update(epoch=epoch + 1, cursor=0)
            checkpoint.save(run / "last.pt", model, optim, sid, state)
            if state["bad_epochs"] >= schedule["patience"]:
                break
        checkpoint.restore(run / "best.pt", model, None, sid, rng=False)
        tuning_prediction = predict(model, tune, forward, schedule["eval_batch_size"])
        # 分类阈值仅在检查点选择分区选一次并写入训练记录。测试集直接采用该值。
        threshold = select_threshold(tuning_prediction)
        threshold.update(
            threshold_selection_partition_identity=spec["tune_membership"],
            threshold_selection_partition_role=spec["tune_role"],
            threshold_selection_protocol=spec["protocol"],
            threshold_selection_fold=spec["fold"],
            threshold_selection_is_checkpoint_partition=True,
        )
        best_metrics = measure(tuning_prediction, threshold["selected_threshold"])
        updates = {
            n: float((p.detach().cpu() - before[n]).norm())
            for n, p in model.named_parameters()
            if n in before
        }
        diagnostics = {
            "gradient_max_norm": gradients,
            "parameter_update_norm": updates,
            "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
            "nonzero_gradient_parameters": sum(
                p.numel() for n, p in model.named_parameters() if gradients.get(n, 0) > 0
            ),
            "epochs_completed": len(history),
            "checkpoint_metric_name": metric,
            "checkpoint_secondary_metric": CHECKPOINT_SECONDARY_METRIC,
            "checkpoint_selection_rule": list(CHECKPOINT_TIE_BREAK),
            "threshold": threshold,
        }
        if hasattr(model, "readout"):
            # 冻结的 lm_head True/False 权重行没有可训练参数，加载模型时从冻结 LLM 重建。
            diagnostics["readout"] = {"class": type(model.readout).__name__,
                                      **model.readout_contract()}
            diagnostics["head"] = {
                "present": False,
                "task_head_parameters": 0,
                "replaced_by": "FrozenBinaryReadout",
            }
        if hasattr(model, "gates"):
            diagnostics["residual_by_layer"] = {i: g.last for i, g in model.gates.items()}
            diagnostics["reader_parameter_identity_shared"] = (
                len({id(model.reader_at(i)) for i in model.injection_layers}) <= 1
            )
        if hasattr(model, "reader"):
            diagnostics["reader"] = model.reader.last
        write_json(run / "diagnostics.json", diagnostics)
        paths = ["spec.json", "best.pt", "best.json", "last.pt", "last.json", "diagnostics.json", "history.csv", "history.jsonl"]
        receipt = {
            "status": "COMPLETE",
            "spec_id": sid,
            "metrics": {"tune": best_metrics},
            "scientific_identity": spec.get("scientific_identity")
            or {
                k: spec[k]
                for k in ("source", "data", "schedule", "synthetic", "campaign", "benchmark_binding")
                if k in spec
            },
            "best_epoch": state["best_epoch"],
            # 记录一级指标的最优值和完整的字典序选择键。
            "best_tuning_metric": metric,
            "best_tuning_score": state["best"],
            "best_tuning_key": state["best_key"],
            "best_tuning_secondary_score": state["best_key"][1] if state["best_key"] else None,
            # 最优检查点的实体宏平均 AP 仅供报告。
            "best_tuning_macro_ap_report_only": best_metrics["drug_macro_ap"],
            "checkpoint_metric": metric,
            "checkpoint_secondary_metric": CHECKPOINT_SECONDARY_METRIC,
            "checkpoint_selection_rule": list(CHECKPOINT_TIE_BREAK),
            "threshold": threshold,
            "protocol": spec.get("protocol"),
            "epochs_completed": len(history),
            "history_epochs": [row["epoch"] for row in history],
            "checkpoint_sha256": sha256(run / "best.pt"),
            "runtime_seconds": time.monotonic() - start_time,
            "peak_vram_bytes": peak_vram,
            "files": {p: sha256(run / p) for p in paths},
            "formal_test_inference": 0,
            "synthetic": spec["synthetic"],
        }
        write_json(run / "receipt.json", receipt, exclusive=True)
        write_json(run / "status.json", {"state": "COMPLETE", "spec_id": sid})
        write_json(attempt, {"state": "COMPLETE", "spec_id": sid, "resume": resume})
        return receipt
    except BaseException as exc:
        _write_history(run, history)
        write_json(
            attempt,
            {
                "state": "FAILED",
                "type": type(exc).__name__,
                "message": str(exc),
                "spec_id": sid,
                "resume": resume,
            },
        )
        write_json(run / "failure.json", {"type": type(exc).__name__, "message": str(exc), "spec_id": sid})
        write_json(run / "status.json", {"state": "FAILED", "spec_id": sid})
        raise


def _epoch_report_threshold(prediction):
    """计算每轮诊断用阈值。

    正式阈值在恢复最优检查点后选择一次。并列时与正式规则一致，选择较高阈值。
    """
    return select_threshold(prediction)["selected_threshold"]


def evaluate(run, model, forward, frames, eval_spec, training_receipt, schedule):
    """在一个或多个分区评估已训练的候选模型。

    读取已封存的最优检查点和验证阈值。重复评估无需重新拟合，也不会修改训练标识。
    """
    sid = identity(eval_spec)
    threshold = training_receipt.get("threshold", {}).get("selected_threshold")
    if threshold is None:
        raise ContractError("frozen validation threshold required; Test threshold search is forbidden")
    directory = run / "evaluations" / sid
    if (directory / "receipt.json").exists():
        receipt = read_json(directory / "receipt.json")
        if receipt["spec_id"] != sid or any(
            sha256(directory / p) != h for p, h in receipt["files"].items()
        ):
            raise ContractError("sealed evaluation changed")
        return receipt
    directory.mkdir(parents=True, exist_ok=True)
    write_json(directory / "spec.json", eval_spec, exclusive=True)
    checkpoint.restore(run / "best.pt", model, None, training_receipt["spec_id"], rng=False)
    started = time.monotonic()
    results = {}
    paths = ["spec.json"]
    for label, frame in frames.items():
        prediction = predict(model, frame, forward, schedule["eval_batch_size"])
        rel = "predictions/%s.parquet" % label
        atomic_frame(directory / rel, prediction)
        results[label] = measure(prediction, float(threshold))
        paths.append(rel)
    receipt = {
        "status": "COMPLETE",
        "spec_id": sid,
        "training_run": run.name,
        "training_spec_id": training_receipt["spec_id"],
        "training_receipt_sha256": sha256(run / "receipt.json"),
        "checkpoint_sha256": training_receipt["checkpoint_sha256"],
        "metrics": results,
        "evaluation_contract": eval_spec.get("evaluation_contract"),
        "blocks": list(frames),
        "threshold_used": float(threshold),
        "threshold_source": "sealed training receipt; selected on the checkpoint partition",
        "threshold_selection_partition_identity": training_receipt["threshold"][
            "threshold_selection_partition_identity"
        ],
        "threshold_selection_metric": training_receipt["threshold"]["threshold_selection_metric"],
        "threshold_tie_break_rule": training_receipt["threshold"]["threshold_tie_break_rule"],
        "test_threshold_search_performed": False,
        "runtime_seconds": time.monotonic() - started,
        "files": {p: sha256(directory / p) for p in paths},
        "formal_test_inference": 0 if eval_spec.get("synthetic") else int(
            bool(eval_spec.get("formal_test_access", False))
        ),
        "synthetic": eval_spec.get("synthetic", False),
    }
    write_json(directory / "receipt.json", receipt, exclusive=True)
    return receipt
