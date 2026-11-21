"""冻结的 MMed 边界，显式在上层 Transformer 模块注册注入钩子。"""

import torch
from torch import nn
from .readout import FALSE_TOKEN_ID, TRUE_TOKEN_ID
from .utils import ContractError

#: 冻结 True/False 读出访问的最大权重行索引。
MAX_READOUT_TOKEN_ID = max(FALSE_TOKEN_ID, TRUE_TOKEN_ID)


def anchor(hidden, mask):
    length = mask.long().sum(-1)
    expected = torch.arange(mask.shape[1], device=mask.device)[None, :] < length[:, None]
    if not (length > 0).all() or not torch.equal(expected, mask.bool()):
        raise ContractError("contiguous right-padding mask required")
    return hidden[torch.arange(len(hidden), device=hidden.device), length - 1]


def injection_map(inject, layer=None):
    mapping = ({} if inject is None else inject) if layer is None else {layer: inject}
    if not isinstance(mapping, dict) or list(mapping) not in ([], [30], [29, 30], [28, 29, 30]):
        raise ContractError(
            "only the U28-3L, U29-2L and U30-1L injection spans are permitted; block31 forbidden"
        )
    if not all(callable(fn) for fn in mapping.values()):
        raise ContractError("injection callback required")
    return mapping


class _FrozenLMHeadStub(nn.Module):
    """供合成测试使用的 lm_head 替代对象。

    读出访问 4139（False）和 2575（True）两行，因此替代对象仅分配 max(FALSE_TOKEN_ID, TRUE_TOKEN_ID) + 1 行。真实边界仍独立校验 vocab_size == 128256。
    """

    def __init__(self, hidden, rows):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(rows, hidden) * 0.02, requires_grad=False)


class MockBoundary(nn.Module):
    def __init__(self, hidden=32):
        super().__init__()
        self.hidden_dim = hidden
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(2026)
            self.embedding = nn.Embedding(1024, hidden)
            self.layers = nn.ModuleList(nn.Linear(hidden, hidden, bias=False) for _ in range(32))
            self.norm = nn.LayerNorm(hidden)
            self.lm_head = _FrozenLMHeadStub(hidden, MAX_READOUT_TOKEN_ID + 1)
        self.requires_grad_(False)
        self.eval()

    @property
    def lm_head_weight(self):
        """构造读出所用的冻结 lm_head 权重。"""
        return self.lm_head.weight

    def train(self, mode=True):
        return super().train(False)

    def forward(self, ids, mask, inject=None, layer=None):
        injections = injection_map(inject, layer)
        h = self.embedding(ids.remainder(1024)) * mask[:, :, None]
        h = h.cumsum(1) / mask.long().cumsum(1).clamp_min(1)[:, :, None]
        for i, block in enumerate(self.layers):
            h = h + 0.01 * torch.tanh(block(h))
            if i in injections:
                h = injections[i](h)
        return self.norm(h)


class FrozenBoundary(nn.Module):
    def __init__(self, base, strict=True):
        super().__init__()
        self.base = base
        self.hidden_dim = base.config.hidden_size
        if strict and (
            base.config.num_hidden_layers != 32 or self.hidden_dim != 4096 or base.config.vocab_size != 128256
        ):
            raise ContractError("MMed architecture mismatch")
        self.requires_grad_(False)
        self.eval()

    @classmethod
    def load(cls, path, device="cuda"):
        if not torch.cuda.is_available():
            raise ContractError("加载冻结语言模型需要 CUDA GPU")
        from transformers import AutoModelForCausalLM

        return cls(
            AutoModelForCausalLM.from_pretrained(
                path,
                local_files_only=True,
                trust_remote_code=False,
                torch_dtype=torch.bfloat16,
                low_cpu_mem_usage=True,
            ).to(device)
        )

    @property
    def lm_head_weight(self):
        """构造读出所用的冻结 lm_head 权重。

        这里直接读取模型预训练的答案词元权重行。
        """
        return self.base.lm_head.weight

    def train(self, mode=True):
        super().train(False)
        self.base.eval()
        return self

    def forward(self, ids, mask, inject=None, layer=None):
        handles = []
        injections = injection_map(inject, layer)

        def make_hook(callback):
            def hook(module, args, output):
                value = callback(output[0] if isinstance(output, tuple) else output)
                return (value, *output[1:]) if isinstance(output, tuple) else value

            return hook

        try:
            for index, callback in injections.items():
                if index >= len(self.base.model.layers):
                    raise ContractError("layer out of range")
                handles.append(self.base.model.layers[index].register_forward_hook(make_hook(callback)))
            # 冻结的后续层仍须传递输入梯度，注入路径不能放在 no_grad 中。
            return self.base.model(
                input_ids=ids, attention_mask=mask, use_cache=False, return_dict=True
            ).last_hidden_state
        finally:
            for h in handles:
                h.remove()
