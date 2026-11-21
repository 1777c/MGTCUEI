"""CUEI 读取器是 TypedMemory 与冻结 LLM 之间的适配器。

三种读取器使用相同的张量形状，输入 h [B,T,4096] 和 m [B,3,512]，输出 [B,T,4096]。

    cross_attention 以当前 LLM 隐藏状态为查询。
    non_query 只使用记忆，学习各槽位权重，与查询无关。
    mean 只使用记忆，固定均匀权重 1/3。

均值基线固定槽位权重，以保留所考察的消融变量。
"""

import torch
from torch import nn

READER_REGISTRY = {}


def register(name):
    def wrap(cls):
        READER_REGISTRY[name] = cls
        cls.reader_name = name
        return cls

    return wrap


@register("cross_attention")
class CrossAttention(nn.Module):
    def __init__(self, hidden=4096):
        super().__init__()
        self.q_proj = nn.Linear(hidden, 512)
        self.k_proj = nn.Linear(512, 512, bias=False)
        self.v_proj = nn.Linear(512, 512)
        self.out_proj = nn.Linear(512, hidden, bias=False)

    def forward(self, h, m):
        with torch.autocast(device_type=h.device.type, enabled=False):
            b, t, _ = h.shape
            q = self.q_proj(h.float()).reshape(b, t, 8, 64).transpose(1, 2)
            k = self.k_proj(m.float()).reshape(b, 3, 8, 64).transpose(1, 2)
            v = self.v_proj(m.float()).reshape(b, 3, 8, 64).transpose(1, 2)
            a = torch.softmax(q @ k.transpose(-1, -2) / 8.0, dim=-1)
            self.last = {
                "K_slot_variance": float(k.detach().var(2, unbiased=False).mean()),
                "V_slot_variance": float(v.detach().var(2, unbiased=False).mean()),
                "attention_entropy": float(-(a.detach() * a.detach().clamp_min(1e-30).log()).sum(-1).mean()),
                "addressing": "query-conditioned",
            }
            return self.out_proj((a @ v).transpose(1, 2).reshape(b, t, 512))


@register("non_query")
class NonQuery(nn.Module):
    def __init__(self, hidden=4096):
        super().__init__()
        self.weighting = nn.Sequential(nn.Linear(1536, 1512), nn.GELU(), nn.Linear(1512, 24))
        self.v_proj = nn.Linear(512, 512)
        self.out_proj = nn.Linear(512, hidden, bias=False)

    def addressing(self, m):
        return self.weighting(m.float().flatten(1)).reshape(-1, 8, 3).softmax(-1)

    def forward(self, h, m):
        with torch.autocast(device_type=h.device.type, enabled=False):
            v = self.v_proj(m.float()).reshape(-1, 3, 8, 64).transpose(1, 2)
            a = self.addressing(m)
            self.last = {
                "V_slot_variance": float(v.detach().var(2, unbiased=False).mean()),
                "attention_entropy": float(-(a.detach() * a.detach().clamp_min(1e-30).log()).sum(-1).mean()),
                "addressing": "memory-only, query-independent",
            }
            c = (a.unsqueeze(-1) * v).sum(2).reshape(-1, 512)
            return self.out_proj(c)[:, None, :].expand(-1, h.shape[1], -1)


@register("mean")
class Mean(nn.Module):
    """对三个类型化槽位取无权重均值。

    先求 m_mean = (m_drug + m_disease + m_interaction) / 3，再通过共享线性层投影到隐藏维度并广播到各词元。槽位权重均匀且独立于查询和词元。
    """

    def __init__(self, hidden=4096):
        super().__init__()
        self.v_proj = nn.Linear(512, 512)
        self.out_proj = nn.Linear(512, hidden, bias=False)

    def addressing(self, m):
        # 三个槽位采用固定的均匀注意力权重。
        return m.new_full((m.shape[0], 8, 3), 1.0 / 3.0)

    def forward(self, h, m):
        with torch.autocast(device_type=h.device.type, enabled=False):
            v = self.v_proj(m.float()).reshape(-1, 3, 8, 64).transpose(1, 2)
            a = self.addressing(m)
            self.last = {
                "V_slot_variance": float(v.detach().var(2, unbiased=False).mean()),
                "attention_entropy": float(-(a.detach() * a.detach().clamp_min(1e-30).log()).sum(-1).mean()),
                "addressing": "memory-only, uniform 1/3, not learned",
            }
            # 投影对槽位轴线性，先平均再投影与先投影再平均等价。
            c = (a.unsqueeze(-1) * v).sum(2).reshape(-1, 512)
            return self.out_proj(c)[:, None, :].expand(-1, h.shape[1], -1)


#: 各读取器声明 conditioned_on 以标明消融条件。parameters 来自实际计数。
READER_CONTRACT = {
    "cross_attention": {
        "class": "CrossAttention",
        "input": "h [B,T,4096] (current LLM hidden state), m [B,3,512] (TypedMemory slots)",
        "output": "[B,T,4096]",
        "query_conditioned": True,
        "sample_conditioned": True,
        "token_conditioned": True,
        "layer_conditioned": False,
        "shared_across_injection_layers": True,
        "addressing": "softmax over 3 slots of q(h) . k(m) / 8, 8 heads",
        "learnable_slot_weighting": True,
    },
    "non_query": {
        "class": "NonQuery",
        "input": "m [B,3,512] only (h accepted for the shared call signature and unused)",
        "output": "[B,T,4096] (constant over T)",
        "query_conditioned": False,
        "sample_conditioned": True,
        "token_conditioned": False,
        "layer_conditioned": False,
        "shared_across_injection_layers": True,
        "addressing": "softmax over 3 slots of an MLP(flatten(m)), 8 heads",
        "learnable_slot_weighting": True,
    },
    "mean": {
        "class": "Mean",
        "input": "m [B,3,512] only (h accepted for the shared call signature and unused)",
        "output": "[B,T,4096] (constant over T)",
        "query_conditioned": False,
        "sample_conditioned": True,
        "token_conditioned": False,
        "layer_conditioned": False,
        "shared_across_injection_layers": True,
        "addressing": "fixed uniform 1/3 over 3 slots; no learned weighting",
        "learnable_slot_weighting": False,
    },
}


def reader_contract():
    """返回实际参数量和声明的条件依赖关系。"""
    out = {}
    for name, cls in READER_REGISTRY.items():
        model = cls(4096)
        out[name] = {
            **READER_CONTRACT[name],
            "parameters": sum(p.numel() for p in model.parameters()),
            "parameter_breakdown": {
                n: p.numel() for n, p in model.named_parameters()
            },
        }
    return out
