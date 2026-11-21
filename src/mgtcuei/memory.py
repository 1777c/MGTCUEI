"""类型化的外部键值记忆，以及预先登记的固定坐标混合。

槽位名称和维度由 graph.py 统一定义。槽 0 接收 128 维药物表示 h_d，槽 1 接收 128 维疾病表示 h_s，槽 2 接收 256 维交互特征 h_int。输出形状为 B × 3 × MEMORY_SLOT_DIM，每槽 512 维。

记忆是 EvidenceModel 的可训练子模块，由最终证据任务的交叉熵端到端优化。
"""

import torch
from torch import nn
from .graph import SLOT_SOURCE_DIMS, pair_features
from .utils import ContractError

#: TypedMemory 单个槽位经适配器后的宽度。
MEMORY_SLOT_DIM = 512
SLOT_NAMES = ("drug", "disease", "interaction_feature")


class TypedMemory(nn.Module):
    def __init__(self, mode="typed_ddi", q=None):
        super().__init__()
        self.mode = mode
        if mode not in ("typed_ddi", "fixed_mixed"):
            raise ContractError("memory mode")
        if mode == "fixed_mixed":
            if (
                q is None
                or q.shape != (512, 512)
                or not torch.allclose(q.T @ q, torch.eye(512, dtype=q.dtype), atol=2e-6)
            ):
                raise ContractError("registered orthogonal Q required")
        self.register_buffer("q", torch.eye(512) if q is None else q.float().clone())
        self.adapters = nn.ModuleDict(
            {
                name: nn.Sequential(nn.Linear(n, MEMORY_SLOT_DIM), nn.LayerNorm(MEMORY_SLOT_DIM))
                for name, n in zip(SLOT_NAMES, SLOT_SOURCE_DIMS)
            }
        )

    def forward(self, d, s):
        z = pair_features(d, s)
        if self.mode == "fixed_mixed":
            z = z @ self.q.T
        pieces = torch.split(z, list(SLOT_SOURCE_DIMS), dim=-1)
        return torch.stack([adapter(x) for adapter, x in zip(self.adapters.values(), pieces)], 1)


def memory_diagnostics(m):
    x = m.detach().float()
    norm = torch.nn.functional.normalize(x, dim=-1)
    return {
        "slot_mean": x.mean((0, 2)).tolist(),
        "slot_std": x.std((0, 2), unbiased=False).tolist(),
        "across_sample_variance": x.var(0, unbiased=False).mean(-1).tolist(),
        "pairwise_cosine": (norm @ norm.transpose(1, 2)).mean(0).tolist(),
        "constant_output": bool(x.var(0, unbiased=False).max() < 1e-12),
    }
