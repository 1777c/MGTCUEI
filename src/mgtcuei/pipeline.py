"""按两阶段流程训练一组 MGTCUEI 模型。

1. fit_gcn 在本折训练集上监督拟合 GraphPredictor，结构为 GCN-ORIG 加 512-256-ReLU-2 分类头。检查点用于初始化编码器，在测试集直接评估时即 GNN-only 组。
2. fit_evidence 构建 CUEI，载入 gcn.* 并保持可训练。在冻结 MMed-Llama-3-8B 的读出目标下联合优化 gcn、memory、reader 和 gate。

复现实验需要完成两个阶段，本包未附预训练检查点。
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from . import checkpoint, engine
from .config import (
    CAMPAIGN,
    CHECKPOINT_SECONDARY_METRIC,
    CHECKPOINT_TIE_BREAK,
    FROZEN_DDA52_SOURCE_IDENTITY,
    THRESHOLD_POLICY,
    candidate,
    checkpoint_metric_for_protocol,
    training_config,
)
from .data import Data
from .graph import GCN2, N_DRUGS, N_NODES, GraphPredictor
from .llm import FrozenBoundary, MockBoundary
from .memory import TypedMemory
from .model import EvidenceModel
from .readout import READOUT_ARCHITECTURE, READOUT_CONTRACT_VERSION, READOUT_TYPE
from .utils import ContractError, identity, read_json, sha256, write_json

GCN_MODE = "original"  # GCN-ORIG 不额外添加自环。
GCN_NAME = "GCN-ORIG"
DEV_SELECTION = "Dev-A"


def _frame_identity(frame: pd.DataFrame) -> str:
    return identity(frame.to_dict("list"))


def _seed_membership(frame: pd.DataFrame) -> str:
    return identity(sorted(frame.pair_id.tolist()))


def _coerce_candidate(config: dict) -> dict:
    """将候选配置字典或已保存的配置记录转换为训练配置。"""
    cfg = candidate(
        cuei_span=config.get("cuei_span", "U29-2L"),
        reader=config.get("reader", "cross_attention"),
        gate=config.get("gate", "learned"),
        scale=float(config.get("lambda", 0.01)),
        memory=config.get("memory", "typed_ddi"),
        seed=int(config.get("seed", 2026)),
        prompt=config.get("prompt", "semantic_compact_v1"),
    )
    if "injection_layers" in config and list(config["injection_layers"]) != cfg["injection_layers"]:
        raise ContractError("span/injection_layers mismatch")
    return cfg


class Pipeline:
    def __init__(self, root, *, device=None, llm_root=None, synthetic=False):
        self.root = Path(root).resolve()
        self.data = Data(self.root)
        self.synthetic = bool(synthetic)
        if device is None:
            device = "cpu" if synthetic else ("cuda" if torch.cuda.is_available() else "cpu")
        self.device = torch.device(device)
        self.llm_root = llm_root
        cfg = training_config()
        self.schedule = dict(cfg["schedule"])
        if synthetic:
            self.schedule.update(
                max_epochs=1, patience=1, batch_size=4, microbatch=2,
                eval_batch_size=2, checkpoint_every_steps=1,
            )
        self._boundary = None
        self._prompt_cache: dict[str, dict] = {}

    # 通用方法
    def boundary(self):
        if self._boundary is None:
            if self.synthetic:
                self._boundary = MockBoundary()
            else:
                if not self.llm_root:
                    raise ContractError("--llm-root is required for a real run")
                self._boundary = FrozenBoundary.load(self.llm_root, device=str(self.device))
        return self._boundary

    def schedule_for(self, kind: str):
        schedule = dict(self.schedule)
        if not self.synthetic and kind == "gcn":
            schedule.update(microbatch=schedule["batch_size"], eval_batch_size=256)
        return schedule

    def graph_tensors(self, protocol: str, fold: int):
        x = torch.from_numpy(self.data.features()).float().to(self.device)
        edges, weights = self.data.graph(protocol, fold)
        return (
            x,
            torch.from_numpy(edges).long().to(self.device),
            torch.from_numpy(weights).float().to(self.device),
        )

    def tokens(self, frame: pd.DataFrame, style: str):
        style_cache = self._prompt_cache.setdefault(style, {})
        missing = set(frame.pair_id) - set(style_cache)
        if missing:
            style_cache.update(self.data.prompt_tokens(style, missing))
        rows = [style_cache[pid] for pid in frame.pair_id]
        if self.synthetic:
            rows = [r[:8] for r in rows]
        length = max(map(len, rows))
        ids = torch.zeros(len(rows), length, dtype=torch.long, device=self.device)
        mask = torch.zeros_like(ids)
        for i, row in enumerate(rows):
            ids[i, : len(row)] = torch.as_tensor(row, device=self.device)
            mask[i, : len(row)] = 1
        return ids, mask

    def tune_partition(self, protocol: str, fold: int):
        part = "A" if protocol == "cold" else "val"
        return part, self.data.partition(part, protocol, fold)

    def source_identity(self):
        return identity(
            {
                "release": "MGTCUEI_FINAL_RELEASE_V1",
                "frozen_dda52_source_identity": FROZEN_DDA52_SOURCE_IDENTITY,
                "scientific_assets": self.data.files,
                "benchmarks": self.data.benchmarks.get("benchmark_identity"),
            }
        )

    def base_spec(self, kind: str, protocol: str, fold: int, seed: int, config: dict):
        train = self.data.partition("train", protocol, fold)
        tune_part, tune = self.tune_partition(protocol, fold)
        schedule = self.schedule_for(kind)
        return {
            "campaign": CAMPAIGN,
            "kind": kind,
            "protocol": protocol,
            "fold": int(fold),
            "seed": int(seed),
            "config": config,
            "train_membership": _seed_membership(train),
            "tune_membership": _seed_membership(tune),
            "tune_role": DEV_SELECTION if protocol == "cold" else "fold-local Validation",
            "ordering": "torch.randperm(seed+epoch+701); immutable parquet order",
            "checkpoint_metric": checkpoint_metric_for_protocol(protocol),
            "checkpoint_secondary_metric": CHECKPOINT_SECONDARY_METRIC,
            "checkpoint_selection_rule": list(CHECKPOINT_TIE_BREAK),
            "threshold_policy": dict(THRESHOLD_POLICY),
            "optimizer": {"name": "Adam", "scheduler": "none", **schedule},
            "schedule": schedule,
            "source_identity": self.source_identity(),
            "release_source_identity": FROZEN_DDA52_SOURCE_IDENTITY,
            "synthetic": self.synthetic,
            "formal_test_access": False,
            "graph_to_prediction_paths": 1 if kind == "evidence" else None,
            "gcn_lifecycle": "prefit_then_joint" if kind == "evidence" else None,
            "precision": "FP32 trainable; BF16 frozen LLM" if not self.synthetic else "synthetic FP32",
        }

    # 第一阶段：拟合 GCN
    def fit_gcn(self, protocol: str, fold: int, seed: int, results_root, *, resume=False):
        checkpoint.seed_everything(seed)
        model = GraphPredictor(mode=GCN_MODE, route="gcn", n_drugs=N_DRUGS).to(self.device)
        graph = self.graph_tensors(protocol, fold)
        train = self.data.partition("train", protocol, fold)
        _, tune = self.tune_partition(protocol, fold)
        config = {"name": GCN_NAME, "mode": GCN_MODE, "route": "gcn"}
        spec = self.base_spec("gcn", protocol, fold, seed, config)
        run = Path(results_root) / "jobs" / identity(spec)

        def forward(frame):
            return model(
                graph[0], graph[1], graph[2],
                torch.as_tensor(frame.drug_idx.to_numpy(), device=self.device),
                torch.as_tensor(frame.disease_idx.to_numpy(), device=self.device),
            )

        receipt = engine.train(
            run, model, train, tune, forward, spec, self.schedule_for("gcn"), resume=resume
        )
        return {"run": run, "receipt": receipt, "spec": spec, "model": model,
                "forward": forward, "graph": graph}

    # 第二阶段：训练证据模型
    def gcn_initialization(self, protocol: str, fold: int, seed: int, results_root):
        with torch.random.fork_rng(devices=[]):
            checkpoint.seed_everything(seed)
            component = GCN2(GCN_MODE)
        upstream = self.fit_gcn(protocol, fold, seed, results_root)
        digest = checkpoint.load_component(
            upstream["run"] / "best.pt", component, "gcn.", freeze=False
        )
        component.requires_grad_(True)
        component.train(True)
        component.upstream_checkpoint_sha256 = digest
        component.upstream_run = str(upstream["run"])
        return component, digest, upstream

    def memory_module(self, name: str) -> TypedMemory:
        if name == "typed_ddi":
            return TypedMemory("typed_ddi")
        if name == "fixed_mixed_q1":
            q = torch.from_numpy(np.load(self.data.path("data/q/Q1.npy"), allow_pickle=False))
            return TypedMemory("fixed_mixed", q.float())
        raise ContractError("unregistered memory: " + name)

    def fit_evidence(self, config: dict, protocol: str, fold: int, results_root, *, resume=False):
        config = _coerce_candidate(config)
        seed = int(config["seed"])
        gcn_module, gcn_sha, upstream = self.gcn_initialization(protocol, fold, seed, results_root)
        checkpoint.seed_everything(seed + 401)
        memory = self.memory_module(config["memory"])
        checkpoint.seed_everything(seed)
        readout_authority = {
            "llm_root": str(self.llm_root) if self.llm_root else "synthetic",
            "llm_authority": self.data.files.get("data/llm_contract/MMED_LLAMA_MANIFEST.json"),
        }
        model = EvidenceModel(
            self.boundary(), memory, gcn_module.to(self.device), config,
            {"U30-1L": {"injection_layers": [30]},
             "U29-2L": {"injection_layers": [29, 30]},
             "U28-3L": {"injection_layers": [28, 29, 30]}},
            readout_authority=readout_authority,
        ).to(self.device)
        model.assert_joint_gcn()
        train = self.data.partition("train", protocol, fold)
        _, tune = self.tune_partition(protocol, fold)
        graph = self.graph_tensors(protocol, fold)
        if graph[0].shape[0] != N_NODES:
            raise ContractError("node feature count differs from the graph authority")
        spec = self.base_spec("evidence", protocol, fold, seed, config)
        spec.update(
            upstream={
                "gcn_initialization_sha256": gcn_sha,
                "gcn_initialization_run": str(upstream["run"]),
                "gcn_checkpoint_role": "INITIALIZATION_AUTHORITY",
                "gcn_frozen_during_evidence": False,
                "gcn_in_evidence_optimizer": True,
                "gcn_graph_forward": "live_differentiable_inside_autograd",
                "gcn_lifecycle": "prefit_then_joint",
                "memory_lifecycle": "joint",
                "memory_checkpoint_sha256": None,
            },
            prompt=config["prompt"],
            prompt_sha256=self.data.files[f"data/prompts/{config['prompt']}.parquet"],
            readout_type=READOUT_TYPE,
            readout_contract_version=READOUT_CONTRACT_VERSION,
            trainable_classifier_parameters=0,
            llm_root="data/llm_contract",
        )
        run = Path(results_root) / "jobs" / identity(spec)
        # 拟合前验证提示覆盖情况，缺少配对时在执行 GPU 步骤前报错。
        self.tokens(train, config["prompt"])
        self.tokens(tune, config["prompt"])

        def forward(frame):
            # 每个批次都在自动求导图中重新计算，编码器通过该调用接收损失梯度。
            d, s = model.embed(
                graph[0], graph[1], graph[2],
                torch.as_tensor(frame.drug_idx.to_numpy(), device=self.device),
                torch.as_tensor(frame.disease_idx.to_numpy(), device=self.device),
                N_DRUGS,
            )
            ids, mask = self.tokens(frame, config["prompt"])
            return model(d, s, ids, mask)

        receipt = engine.train(run, model, train, tune, forward, spec, self.schedule, resume=resume)
        return {"run": run, "receipt": receipt, "spec": spec, "model": model,
                "forward": forward, "upstream": upstream, "config": config}

    # 评估
    def eval_spec(self, training_spec: dict, block: str, frame: pd.DataFrame):
        return {
            "kind": "evaluation",
            "block": block,
            "protocol": training_spec["protocol"],
            "fold": training_spec["fold"],
            "training_spec_id": identity(training_spec),
            "eval_membership": _seed_membership(frame),
            "eval_membership_content": _frame_identity(frame),
            "source_identity": self.source_identity(),
            "synthetic": self.synthetic,
            "formal_test_access": block == "test",
            "schedule": self.schedule,
            "evaluation_contract": {
                "block": block,
                "protocol": training_spec["protocol"],
                "fold": int(training_spec["fold"]),
                "rows": int(len(frame)),
                "pair_ids_sha256": _seed_membership(frame),
            },
        }

    def evaluate(self, fitted: dict, block: str, protocol: str, fold: int):
        frame = self.data.evaluation_frame(protocol, fold, block)
        prompt = fitted["config"].get("prompt")
        if prompt:
            self.tokens(frame, prompt)
        spec = self.eval_spec(fitted["spec"], block, frame)
        receipt = engine.evaluate(
            fitted["run"], fitted["model"], fitted["forward"], {block: frame},
            spec, fitted["receipt"], self.schedule,
        )
        return receipt, fitted["run"] / "evaluations" / identity(spec)

    # 高层入口
    def load_fitted(self, run_dir, protocol: str, fold: int):
        """从已封存的训练目录重建模型，无需重新拟合。"""
        run_dir = Path(run_dir)
        spec = read_json(run_dir / "spec.json")
        receipt = read_json(run_dir / "receipt.json")
        if receipt.get("status") != "COMPLETE":
            raise ContractError("run is not complete")
        if receipt["spec_id"] != identity(spec):
            raise ContractError("run spec identity mismatch")
        graph = self.graph_tensors(protocol, fold)
        if spec["kind"] == "gcn":
            model = GraphPredictor(mode=GCN_MODE, route="gcn", n_drugs=N_DRUGS).to(self.device)
            model.eval()
            checkpoint.restore(run_dir / "best.pt", model, None, receipt["spec_id"], rng=False)
            config = {"prompt": None}

            def forward(frame):
                return model(
                    graph[0], graph[1], graph[2],
                    torch.as_tensor(frame.drug_idx.to_numpy(), device=self.device),
                    torch.as_tensor(frame.disease_idx.to_numpy(), device=self.device),
                )
        else:
            config = _coerce_candidate(spec["config"])
            checkpoint.seed_everything(config["seed"] + 401)
            memory = self.memory_module(config["memory"])
            with torch.random.fork_rng(devices=[]):
                checkpoint.seed_everything(config["seed"])
                gcn = GCN2(GCN_MODE)
            readout_authority = {
                "llm_root": str(self.llm_root) if self.llm_root else "synthetic",
                "llm_authority": self.data.files.get(
                    "data/llm_contract/MMED_LLAMA_MANIFEST.json"
                ),
            }
            model = EvidenceModel(
                self.boundary(), memory, gcn.to(self.device), config,
                {"U30-1L": {"injection_layers": [30]},
                 "U29-2L": {"injection_layers": [29, 30]},
                 "U28-3L": {"injection_layers": [28, 29, 30]}},
                readout_authority=readout_authority,
            ).to(self.device)
            model.eval()
            checkpoint.restore(run_dir / "best.pt", model, None, receipt["spec_id"], rng=False)

            def forward(frame):
                d, s = model.embed(
                    graph[0], graph[1], graph[2],
                    torch.as_tensor(frame.drug_idx.to_numpy(), device=self.device),
                    torch.as_tensor(frame.disease_idx.to_numpy(), device=self.device),
                    N_DRUGS,
                )
                ids, mask = self.tokens(frame, config["prompt"])
                return model(d, s, ids, mask)

        return {"run": run_dir, "receipt": receipt, "spec": spec, "model": model,
                "forward": forward, "config": config}

    def run_arm(self, arm: str, protocol: str, fold: int, results_root, *,
                seed: int = 2026, prompt: str | None = None, block: str = "test",
                cuei_span: str = "U29-2L", reader: str = "cross_attention",
                gate: str = "learned", memory: str = "typed_ddi", resume: bool = False):
        """拟合并评估已登记的实验组。

        arm 可为 gnn_only、evidence_basic 或 evidence_semantic。正式协议的 block 为 test，开发诊断可选 Dev-FINAL、Dev-B 或 Dev-C。
        """
        if arm == "gnn_only":
            fitted = self.fit_gcn(protocol, fold, seed, results_root, resume=resume)
            fitted["config"] = {"prompt": None}
            receipt, directory = self.evaluate(fitted, block, protocol, fold)
            return fitted, receipt, directory
        if arm not in ("evidence_basic", "evidence_semantic"):
            raise ContractError("unknown arm: " + arm)
        style = prompt or ("basic_name_only_v1" if arm == "evidence_basic" else "semantic_compact_v1")
        cfg = candidate(
            cuei_span=cuei_span, reader=reader, gate=gate, memory=memory,
            seed=seed, prompt=style,
        )
        fitted = self.fit_evidence(cfg, protocol, fold, results_root, resume=resume)
        receipt, directory = self.evaluate(fitted, block, protocol, fold)
        return fitted, receipt, directory
