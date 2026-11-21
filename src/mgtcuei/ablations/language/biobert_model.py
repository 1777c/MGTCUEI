"""BioBERT 对照模型和检查点约定。

EvidenceModel 通过 FrozenBinaryReadout 从因果 LLM 的 lm_head 读取冻结 True/False 行，并要求无可训练读出参数。BioBERT 编码器使用训练后的分类头，因此需要在检查点前缀 gcn.、memory.、reader.、gates. 之外保存 head.。此约定记录为 DDA5_2_LANG_BIOBERT_CHECKPOINT_V1。

其余证据路线与 EvidenceModel 对应，包括 TypedMemory、CrossAttention、ResidualGate、注入和图配对路径，以及优化器参数约束。读取器宽度随 BioBERT 设为 768。
"""

from __future__ import annotations

import math

import torch
from torch import nn

from mgtcuei.readers import READER_REGISTRY  # noqa: E402
from mgtcuei.residual import ResidualGate  # noqa: E402
from mgtcuei.utils import ContractError  # noqa: E402

CHECKPOINT_CONTRACT = "DDA5_2_LANG_BIOBERT_CHECKPOINT_V1"
TRAINABLE_GROUP_PREFIXES = ("gcn.", "memory.", "reader.", "gates.", "head.")
READOUT_ARCHITECTURE = (
    "BioBERTHead: final [CLS] of the frozen BioBERT encoder [B,768] -> "
    "Linear(768,2) trained binary head -> logits[:,0]=False, logits[:,1]=True; "
    "READOUT_ARCHITECTURE_DIFFERENCE=TRUE"
)


class BioBERTHead(nn.Module):
    """从最终 [CLS] 输出二分类 logits。

    engine.predict 将第 0 列解释为 False，第 1 列解释为 True，分数为 logit_true - logit_false。
    """

    def __init__(self, hidden: int = 768):
        super().__init__()
        self.proj = nn.Linear(hidden, 2)
        nn.init.normal_(self.proj.weight, std=0.02)
        nn.init.zeros_(self.proj.bias)

    def forward(self, cls):
        if cls.shape[-1] != self.proj.in_features:
            raise ContractError(
                f"BioBERT head hidden mismatch: {cls.shape[-1]} vs {self.proj.in_features}")
        return self.proj(cls)


class BioBERTEvidenceModel(nn.Module):
    """冻结 BioBERT 编码器上的 CUEI，与 EvidenceModel 的结构对应。"""

    checkpoint_contract = CHECKPOINT_CONTRACT

    def __init__(self, boundary, memory, gcn, config, readout_authority=None):
        super().__init__()
        self.llm = boundary
        self.memory = memory
        self.config = config
        self.gcn = gcn
        if gcn is None:
            raise ContractError("the evidence model requires a graph encoder")

        self.gcn.requires_grad_(True)
        self.gcn.train(True)
        self.gcn_initialization_sha256 = getattr(gcn, "upstream_checkpoint_sha256", None)
        self.gcn_initialization_run = getattr(gcn, "upstream_run", None)
        self.memory.requires_grad_(True)
        self.memory.train(True)

        # 注入层由 12 层编码器边界确定，并与候选配置核对。
        self.injection_layers = list(boundary.injection_layers)
        if list(config["injection_layers"]) != self.injection_layers:
            raise ContractError(
                "BioBERT span/config mismatch: boundary says "
                f"{self.injection_layers}, candidate says {config['injection_layers']}")

        seed = config["seed"]
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed + 211)
            cls = READER_REGISTRY.get(config["reader"])
            if cls is None:
                raise ContractError("reader registry")
            # 复用 Llama 对照组的读取器类型，按 BioBERT 的宽度构造。
            self.reader = cls(boundary.hidden_dim)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed + 307)
            self.reader.v_proj.reset_parameters()
            self.reader.out_proj.reset_parameters()

        self.gates = nn.ModuleDict(
            {str(i): ResidualGate(config["gate"], config["lambda"]) for i in self.injection_layers})

        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed + 401)
            self.head = BioBERTHead(boundary.hidden_dim)

        self.readout_authority = dict(readout_authority or {})
        self.assert_trainable_head()
        self.assert_joint_memory()

    # 模型约定
    def checkpoint_prefixes(self):
        return list(TRAINABLE_GROUP_PREFIXES)

    def trainable_groups(self):
        return {
            "gcn": list(self.gcn.parameters()),
            "memory": list(self.memory.parameters()),
            "reader": list(self.reader.parameters()),
            "gate": list(self.gates.parameters()),
            "head": list(self.head.parameters()),
        }

    def optimizer_parameters(self):
        params = []
        for group in self.trainable_groups().values():
            params.extend(group)
        ids = {id(p) for p in params}
        for p in self.llm.parameters():
            if id(p) in ids:
                raise ContractError("frozen BioBERT parameter inside the evidence optimizer")
        for name, p in self.named_parameters():
            if p.requires_grad and id(p) not in ids:
                raise ContractError("unregistered trainable parameter: " + name)
        return params

    def assert_joint_gcn(self):
        names = [n for n, _ in self.gcn.named_parameters()]
        if not names or any(not p.requires_grad for p in self.gcn.parameters()):
            raise ContractError("the graph encoder must be fully trainable")
        return True

    def assert_joint_memory(self):
        owned = {id(p) for p in self.memory.parameters()}
        if not owned:
            raise ContractError("typed memory has no parameters")
        for p in self.memory.parameters():
            if not p.requires_grad:
                raise ContractError("joint memory must be fully trainable")
        for name, p in self.named_parameters():
            if id(p) in owned and not name.startswith("memory."):
                raise ContractError("memory parameter registered outside memory.")
        return True

    def assert_trainable_head(self):
        """确认 BioBERT 分类头可训练。

        Llama 读出要求可训练参数为 0。BioBERT 对照组要求有可训练分类头，且无类似 lm_head 的缓冲区。
        """
        if not hasattr(self, "head"):
            raise ContractError("BioBERT control requires a classification head")
        n = sum(p.numel() for p in self.head.parameters() if p.requires_grad)
        if n != 2 * self.llm.hidden_dim + 2:
            raise ContractError(f"unexpected BioBERT head parameter count: {n}")
        return True

    # 前向计算
    def embed(self, x, edges, weights, drug_idx, disease_idx, n_drugs):
        z = self.gcn(x, edges, weights)
        if torch.is_grad_enabled() and not z.requires_grad:
            raise ContractError("joint GCN lifecycle requires a differentiable graph path")
        return z[drug_idx], z[n_drugs + disease_idx]

    def reader_at(self, layer):
        if layer not in self.injection_layers:
            raise ContractError("unregistered injection")
        return self.reader

    def forward(self, d, s, ids, mask):
        m = self.memory(d, s)

        def injection(layer):
            def apply(h):
                contribution = self.reader(h, m)
                return self.gates[str(layer)](h, contribution, mask)
            return apply

        h = self.llm(ids, mask, {i: injection(i) for i in self.injection_layers})
        return self.head(self.llm.cls(h))

    def train(self, mode=True):
        super().train(mode)
        self.llm.eval()          # 冻结语言骨干保持评估模式。
        self.head.train(mode)    # 分类头跟随模型的训练状态。
        return self

    def readout_contract(self):
        return {
            "readout_type": "biobert_cls_binary_head",
            "readout_contract_version": CHECKPOINT_CONTRACT,
            "contract_variant": CHECKPOINT_CONTRACT,
            "architecture": READOUT_ARCHITECTURE,
            "READOUT_ARCHITECTURE_DIFFERENCE": True,
            "trainable_classifier_parameters": sum(
                p.numel() for p in self.head.parameters() if p.requires_grad),
            "rows_require_grad": True,
            "logit_order": ["False", "True"],
            "score": "logit_true - logit_false",
            # BioBERT 无 True/False 词元权重行，相关的 Llama 读出字段显式填 None。
            "false_token_id": None,
            "true_token_id": None,
            "llm_root": self.readout_authority.get("llm_root"),
            "llm_authority": self.readout_authority.get("llm_authority"),
            "backbone_identity": self.readout_authority.get("backbone_identity"),
            "backbone_hidden": self.llm.hidden_dim,
            "injection_layers": list(self.injection_layers),
            "readout_source": ("trained BioBERTHead over the frozen BioBERT [CLS]; "
                               "the BioBERT backbone itself is frozen"),
        }

    def component_state_sha(self):
        import hashlib
        out = {}
        for name, group in self.trainable_groups().items():
            h = hashlib.sha256()
            for p in group:
                h.update(p.detach().to(torch.float32).cpu().numpy().tobytes())
            out[name] = h.hexdigest()
        return out


# 检查点
def save(path, model, optimizer, spec_id, state):
    """用扩展后的参数名前缀保存检查点。

    只写入 requires_grad 参数。检查点包含图编码器、记忆、读取器、门控、分类头、优化器状态和训练循环状态。
    """
    import json
    import os
    import random

    import numpy as np

    from mgtcuei.utils import sha256, write_json

    payload = {
        "spec_id": spec_id,
        "weights": {n: p.detach().cpu().clone()
                    for n, p in model.named_parameters() if p.requires_grad},
        "optimizer": optimizer.state_dict() if optimizer is not None else None,
        "state": state,
        "readout": model.readout_contract(),
        "trainable_group_prefixes": model.checkpoint_prefixes(),
        "checkpoint_contract": CHECKPOINT_CONTRACT,
        "gcn_initialization_sha256": getattr(model, "gcn_initialization_sha256", None),
        "gcn_initialization_run": getattr(model, "gcn_initialization_run", None),
        "rng_python": random.getstate(),
        "rng_numpy": np.random.get_state(),
        "rng_torch": torch.get_rng_state(),
        "rng_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    }
    tmp = str(path) + ".tmp"
    torch.save(payload, tmp)
    os.replace(tmp, path)
    # 将 JSON 记录的路径传为 Path，由 write_json 创建父目录。
    from pathlib import Path

    write_json(Path(str(path).rsplit(".", 1)[0] + ".json"),
               {"sha256": sha256(path), "spec_id": spec_id, "readout": payload["readout"],
                "checkpoint_contract": CHECKPOINT_CONTRACT})
    return payload


def read(path, spec_id=None):
    import json
    import os

    from mgtcuei.utils import ContractError as CE
    from mgtcuei.utils import sha256

    side = str(path).rsplit(".", 1)[0] + ".json"
    if not os.path.isfile(side):
        raise CE("checkpoint sidecar missing: " + side)
    with open(side) as fh:
        meta = json.load(fh)
    if meta.get("sha256") != sha256(path):
        raise CE("checkpoint digest mismatch: " + str(path))
    if meta.get("checkpoint_contract") != CHECKPOINT_CONTRACT:
        raise CE("not a BioBERT-arm checkpoint: " + str(path))
    if spec_id is not None and meta.get("spec_id") != spec_id:
        raise CE("checkpoint spec_id mismatch")
    return torch.load(path, weights_only=False)


def restore(path, model, optimizer, spec_id, rng=True):
    import os
    import random

    import numpy as np

    payload = read(path, spec_id)
    params = {n: p for n, p in model.named_parameters() if p.requires_grad}
    if set(params) != set(payload["weights"]):
        raise ContractError("trainable/freeze map changed")
    with torch.no_grad():
        for n, p in params.items():
            if payload["weights"][n].shape != p.shape:
                raise ContractError("checkpoint parameter shape mismatch: " + n)
            p.copy_(payload["weights"][n].to(p.device))
    if optimizer is not None and payload.get("optimizer") is not None:
        optimizer.load_state_dict(payload["optimizer"])
    if rng:
        random.setstate(payload["rng_python"])
        np.random.set_state(payload["rng_numpy"])
        torch.set_rng_state(payload["rng_torch"])
        if torch.cuda.is_available() and payload["rng_cuda"]:
            torch.cuda.set_rng_state_all(payload["rng_cuda"])
    return payload["state"]
