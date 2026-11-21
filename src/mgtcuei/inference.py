"""使用训练后的 MGTCUEI 检查点为候选配对评分。

检查点与训练折绑定，图编码器使用该折训练正例拟合，记忆、读取器和门控也在该图上训练。因此推理须指定对应的协议和折。提示是模型输入，候选配对须已存在于训练所用提示版本的冻结缓存。
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import torch
from scipy.special import expit

from . import checkpoint
from .graph import GCN2, N_DRUGS
from .llm import FrozenBoundary
from .model import EvidenceModel
from .pipeline import GCN_MODE, Pipeline, _coerce_candidate
from .utils import ContractError, identity, read_json


class Scorer:
    def __init__(self, root, run_dir, *, protocol, fold, llm_root, device="cuda"):
        self.root = Path(root).resolve()
        self.run_dir = Path(run_dir).resolve()
        self.protocol = protocol
        self.fold = int(fold)
        self.pipeline = Pipeline(self.root, device=device, llm_root=llm_root)
        self.spec = read_json(self.run_dir / "spec.json")
        self.receipt = read_json(self.run_dir / "receipt.json")
        if self.receipt.get("status") != "COMPLETE":
            raise ContractError("run is not complete")
        if self.receipt["spec_id"] != identity(self.spec):
            raise ContractError("run spec identity mismatch")
        self.config = _coerce_candidate(self.spec["config"])
        self.prompt = self.config["prompt"]
        self.threshold = float(self.receipt["threshold"]["selected_threshold"])
        self.model = self._build()
        self.graph = self.pipeline.graph_tensors(protocol, self.fold)

    def _build(self):
        checkpoint.seed_everything(self.config["seed"] + 401)
        memory = self.pipeline.memory_module(self.config["memory"])
        with torch.random.fork_rng(devices=[]):
            checkpoint.seed_everything(self.config["seed"])
            gcn = GCN2(GCN_MODE)
        readout_authority = {
            "llm_root": str(self.pipeline.llm_root),
            "llm_authority": self.pipeline.data.files.get(
                "data/llm_contract/MMED_LLAMA_MANIFEST.json"
            ),
        }
        model = EvidenceModel(
            self.pipeline.boundary(), memory, gcn.to(self.pipeline.device), self.config,
            {"U30-1L": {"injection_layers": [30]},
             "U29-2L": {"injection_layers": [29, 30]},
             "U28-3L": {"injection_layers": [28, 29, 30]}},
            readout_authority=readout_authority,
        ).to(self.pipeline.device)
        checkpoint.restore(
            self.run_dir / "best.pt", model, None, self.receipt["spec_id"], rng=False
        )
        model.eval()
        return model

    def score(self, frame: pd.DataFrame, *, batch_size=32) -> pd.DataFrame:
        required = {"pair_id", "drug_idx", "disease_idx"}
        if not required.issubset(frame.columns):
            raise ContractError("pairs must carry pair_id, drug_idx, disease_idx")
        if frame.pair_id.duplicated().any():
            raise ContractError("duplicate pair_id")
        if not frame.drug_idx.between(0, N_DRUGS - 1).all():
            raise ContractError("drug_idx out of range")
        if not frame.disease_idx.between(0, 4235 - N_DRUGS - 1).all():
            raise ContractError("disease_idx out of range")
        self.pipeline.tokens(frame, self.prompt)  # 检查配对是否具有冻结提示。
        parts = []
        self.model.eval()
        with torch.no_grad():
            for start in range(0, len(frame), batch_size):
                f = frame.iloc[start:start + batch_size]
                out = self._forward(f)
                logits = out.float().cpu().numpy()
                p = f[["pair_id", "drug_idx", "disease_idx"]].copy()
                p["logit_false"] = logits[:, 0]
                p["logit_true"] = logits[:, 1]
                p["score"] = logits[:, 1] - logits[:, 0]
                p["probability"] = expit(p["score"])
                p["predicted_label"] = (p["probability"] >= self.threshold).astype(int)
                p["threshold"] = self.threshold
                parts.append(p)
        return pd.concat(parts, ignore_index=True)

    def _forward(self, frame):
        x, edges, weights = self.graph
        d, s = self.model.embed(
            x, edges, weights,
            torch.as_tensor(frame.drug_idx.to_numpy(), device=self.pipeline.device),
            torch.as_tensor(frame.disease_idx.to_numpy(), device=self.pipeline.device),
            N_DRUGS,
        )
        ids, mask = self.pipeline.tokens(frame, self.prompt)
        return self.model(d, s, ids, mask)
