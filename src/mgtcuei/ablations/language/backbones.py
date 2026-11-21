"""冻结 Llama 和 BioBERT 的加载器，以及 BioBERT 注入边界。"""

from __future__ import annotations

import torch
from torch import nn

from ...utils import ContractError


def plan_injection_span(num_layers: int, *, top_k: int = 2,
                        exclude_final_block: bool = True) -> list[int]:
    """返回 num_layers 层结构的最高 k 个可注入层的零基索引。

    exclude_final_block=True 对应冻结 Llama 约定。可注入跨度为 {}、{30}、{29,30} 或 {28,29,30}，第 31 层不可注入。32 层模型的两层跨度为 [29, 30]，即 U29-2L。

    exclude_final_block=False 对应 BioBERT 的顶部两层。在 12 层模型上得到 [10, 11]。两种约定均记录在 configs/LANGUAGE_ARMS.json。
    """
    if num_layers < top_k + 1:
        raise ContractError(f"cannot place a {top_k}-block span on {num_layers} layers")
    top = num_layers - 1 if exclude_final_block else num_layers
    return list(range(top - top_k, top))


LLAMA_GEOMETRY = {"num_hidden_layers": 32, "hidden_size": 4096, "vocab_size": 128256}


def load_llama_boundary(path: str, device: str = "cuda"):
    """为 Llama-3-8B 检查点构造 mgtcuei.llm.FrozenBoundary。

    加载本地权重，构造边界时检查几何配置，以免使用错误骨干。
    """
    from mgtcuei.llm import FrozenBoundary
    from transformers import AutoModelForCausalLM

    if not torch.cuda.is_available():
        raise ContractError("加载冻结语言骨干需要 CUDA GPU")
    base = AutoModelForCausalLM.from_pretrained(
        path, local_files_only=True, trust_remote_code=False,
        torch_dtype=torch.bfloat16, low_cpu_mem_usage=True,
    )
    # FrozenBoundary.__init__ 仍会校验几何配置，先显式检查可在错误中标明所用模型。
    for key, want in LLAMA_GEOMETRY.items():
        got = getattr(base.config, key, None)
        if got != want:
            raise ContractError(f"{path}: {key}={got} but the Llama-3-8B contract requires {want}")
    # 直接构造边界时显式调用 .to(device)，使模型权重与批数据位于同一设备。
    return FrozenBoundary(base.to(device), strict=True)


# BioBERT 对照组
BIOBERT_GEOMETRY = {"num_hidden_layers": 12, "hidden_size": 768, "vocab_size": 28996}


class BioBERTBoundary(nn.Module):
    """冻结编码器边界，将 Llama 的注入机制应用于 BioBERT。

    在 base.encoder.layer[i] 注册前向钩子，用回调结果替换层输出。后续层继续传播注入值，损失梯度可传回读取器。finally 中移除钩子，前向过程保留梯度。

    BioBERT 使用 encoder.layer 作为层容器，可注入跨度为 [10, 11]，无 lm_head。
    """

    #: BioBERT 对应的可注入跨度。
    ALLOWED_SPANS = ([], [11], [10, 11])

    def __init__(self, base, strict: bool = True):
        super().__init__()
        self.base = base
        self.hidden_dim = base.config.hidden_size
        self.num_layers = base.config.num_hidden_layers
        if strict:
            for key, want in BIOBERT_GEOMETRY.items():
                got = getattr(base.config, key, None)
                if got != want:
                    raise ContractError(
                        f"BioBERT architecture mismatch: {key}={got}, expected {want}")
        # 对 12 层 BioBERT 使用零基 [10, 11]。此对照允许在最终编码器层注入。
        self.injection_layers = plan_injection_span(
            self.num_layers, top_k=2, exclude_final_block=False)
        self.requires_grad_(False)
        self.eval()

    # 注入约定
    def injection_map(self, inject=None, layer=None) -> dict:
        mapping = ({} if inject is None else inject) if layer is None else {layer: inject}
        if not isinstance(mapping, dict) or list(mapping) not in self.ALLOWED_SPANS:
            raise ContractError(
                "only the {} injection spans are permitted for BioBERT".format(
                    self.ALLOWED_SPANS))
        if not all(callable(fn) for fn in mapping.values()):
            raise ContractError("injection callback required")
        return mapping

    @property
    def has_lm_head(self) -> bool:
        return False

    def train(self, mode: bool = True):
        # 冻结边界始终处于评估模式。
        return super().train(False)

    # 前向计算
    def forward(self, ids, mask, inject=None, layer=None):
        handles = []
        injections = self.injection_map(inject, layer)

        def make_hook(callback):
            def hook(module, args, output):
                value = callback(output[0] if isinstance(output, tuple) else output)
                return (value, *output[1:]) if isinstance(output, tuple) else value
            return hook

        try:
            for index, callback in injections.items():
                if index >= len(self.base.encoder.layer):
                    raise ContractError("layer out of range")
                handles.append(
                    self.base.encoder.layer[index].register_forward_hook(make_hook(callback)))
            out = self.base(input_ids=ids, attention_mask=mask,
                            return_dict=True).last_hidden_state
            return out
        finally:
            for h in handles:
                h.remove()

    @staticmethod
    def cls(hidden):
        """返回最终隐藏状态中的 [CLS]。

        连续右填充使第 0 位恒为实际词元，因此无需计算掩码长度。Llama 对照组使用最后一个非填充词元。
        """
        return hidden[:, 0]


def load_biobert_boundary(path: str, device: str = "cuda"):
    from transformers import BertModel

    if not torch.cuda.is_available():
        raise ContractError("加载冻结语言骨干需要 CUDA GPU")
    base = BertModel.from_pretrained(
        path, local_files_only=True, trust_remote_code=False,
        torch_dtype=torch.float32, low_cpu_mem_usage=True,
    )
    # 显式将 BioBERT 冻结边界移动到批数据所在设备。
    return BioBERTBoundary(base.to(device), strict=True)
