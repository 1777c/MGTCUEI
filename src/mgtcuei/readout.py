"""冻结的 MMed-Llama True/False 读出，分类器可训练参数为 0。

分类分数由预训练 lm_head 中两种答案词元的权重行计算。

    logits[:, 0] = anchor @ lm_head.weight[FALSE_TOKEN_ID]
    logits[:, 1] = anchor @ lm_head.weight[TRUE_TOKEN_ID]

FrozenBinaryReadout 没有 nn.Parameter，优化器和可训练检查点均不包含它。两行权重存入非持久缓冲区，不进入 state_dict()，每次加载时从冻结 LLM 重建。仅提取两行可避免在 .to(device) 时复制完整的 [V, 4096] 权重矩阵。

用分词器验证 False 和 True 对应的固定词元 ID。带前导空格的 " False" 和 " True" 属于另外的词元。
"""

from __future__ import annotations

from pathlib import Path

import torch
from torch import nn

from .utils import ContractError

#: 读出词元 ID 由 data/llm_contract/tokenizer.json 验证。
FALSE_TOKEN_STRING = "False"
TRUE_TOKEN_STRING = "True"
FALSE_TOKEN_ID = 4139
TRUE_TOKEN_ID = 2575

#: 证据模型配置中记录的读出类型及接口版本。
READOUT_TYPE = "frozen_lm_true_false_rows"
READOUT_CONTRACT_VERSION = "DDA5_2_FROZEN_LM_TRUE_FALSE_READOUT_V1"
READOUT_ARCHITECTURE = (
    "FrozenBinaryReadout: anchor(final LLM hidden) [B,4096] -> frozen MMed-Llama "
    "lm_head rows [4139 (False), 2575 (True)] -> logits[:,0]=False, logits[:,1]=True; "
    "0 trainable classifier parameters; lm_head rows never trainable, never in a checkpoint"
)
ROUTED_INPUT_SOURCE = "anchor(llm_output, mask)"

#: engine.predict 按声明顺序读取两列。第 0 列为 False，第 1 列为 True，分数为 logit_true - logit_false。
LOGIT_ORDER = ("False", "True")

#: 仅使用 LLM 的登记路线为提示、冻结 LLM、冻结 True/False 权重行，可训练分类器参数为 0。
LLM_ONLY_ROUTE_NAME = "FROZEN_LM_HEAD_LLM_ONLY"
SUPERSEDED_LLM_ONLY_ROUTE_NAME = "TEXTONLY_HEAD_LLM_ONLY"


def verify_token_authority(tokenizer_path, *, expected_false=FALSE_TOKEN_ID,
                           expected_true=TRUE_TOKEN_ID):
    """用实际分词器重新确定读出词元 ID。

    编码字面字符串后对照 ID，再根据 ID 验证词元字符串。
    """
    path = Path(tokenizer_path)
    if not path.is_file():
        raise ContractError("tokenizer authority missing: " + str(path))
    from tokenizers import Tokenizer

    tokenizer = Tokenizer.from_file(str(path))
    report = {
        "tokenizer": str(path),
        "expected": {
            FALSE_TOKEN_STRING: expected_false,
            TRUE_TOKEN_STRING: expected_true,
        },
        "measured": {},
    }
    for text, expected in ((FALSE_TOKEN_STRING, expected_false),
                           (TRUE_TOKEN_STRING, expected_true)):
        enc = tokenizer.encode(text, add_special_tokens=False)
        back = [tokenizer.id_to_token(i) for i in enc.ids]
        report["measured"][text] = {"ids": list(enc.ids), "tokens": back}
        if list(enc.ids) != [expected]:
            raise ContractError(
                "token authority mismatch for %r: live tokenizer returned %r, authority says %r"
                % (text, list(enc.ids), [expected])
            )
        if back != [text]:
            raise ContractError(
                "token round-trip mismatch for %r: id %d decodes to %r"
                % (text, expected, back[0] if back else None)
            )
    # 确认带前导空格的两种写法对应不同词元，防止误用。
    for spaced in (" " + FALSE_TOKEN_STRING, " " + TRUE_TOKEN_STRING):
        enc = tokenizer.encode(spaced, add_special_tokens=False)
        if list(enc.ids) in ([expected_false], [expected_true]):
            raise ContractError(
                "leading-space token %r collides with the readout authority" % spaced
            )
        report["measured"][spaced] = {"ids": list(enc.ids), "authoritative": False}
    report["TOKEN_AUTHORITY_VERIFIED"] = True
    report["FALSE_TOKEN_ID"] = expected_false
    report["TRUE_TOKEN_ID"] = expected_true
    return report


class FrozenBinaryReadout(nn.Module):
    """由冻结 lm_head 的两行实现读出。

    唯一张量输入是经 CUEI 修改的冻结 LLM 最终锚定隐藏状态。记忆、GCN 和配对特征无直接读出接口。
    """

    def __init__(self, lm_head_weight, *, false_id=FALSE_TOKEN_ID, true_id=TRUE_TOKEN_ID,
                 expected_hidden=None, authority=None):
        super().__init__()
        if not torch.is_tensor(lm_head_weight) or lm_head_weight.ndim != 2:
            raise ContractError("lm_head weight must be a 2-D tensor")
        if lm_head_weight.requires_grad:
            raise ContractError("lm_head rows must come from a frozen LLM")
        if false_id == true_id:
            raise ContractError("False and True rows must differ")
        vocab, width = lm_head_weight.shape
        for name, index in (("False", false_id), ("True", true_id)):
            if not 0 <= index < vocab:
                raise ContractError("%s token id %d outside vocab %d" % (name, index, vocab))
        if expected_hidden is not None and width != expected_hidden:
            raise ContractError("lm_head width %d != hidden %d" % (width, expected_hidden))
        self.false_id = int(false_id)
        self.true_id = int(true_id)
        self.hidden_dim = int(width)
        rows = lm_head_weight[[self.false_id, self.true_id]].detach().clone()
        if rows.requires_grad:
            raise ContractError("frozen rows must not require grad")
        # persistent=False，缓冲区不写入科学检查点。
        self.register_buffer("binary_rows", rows, persistent=False)
        # 在检查点记录权重行来源，以便恢复运行时核对。
        self.authority = dict(authority or {})
        self.eval()

    # 读出约定
    def trainable_parameter_count(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def readout_contract(self):
        return {
            "readout_type": READOUT_TYPE,
            "readout_contract_version": READOUT_CONTRACT_VERSION,
            "architecture": READOUT_ARCHITECTURE,
            "false_token_id": self.false_id,
            "true_token_id": self.true_id,
            "logit_order": list(LOGIT_ORDER),
            "score": "logit_true - logit_false",
            "trainable_classifier_parameters": self.trainable_parameter_count(),
            "rows_require_grad": bool(self.binary_rows.requires_grad),
            "rows_in_state_dict": False,
            "input_source": ROUTED_INPUT_SOURCE,
            "accepts_memory_argument": False,
            "llm_root": self.authority.get("llm_root"),
            "llm_authority": self.authority.get("llm_authority"),
        }

    def source_rows(self, lm_head_weight):
        """从当前 LLM 提取权重行，供读出校验。"""
        return lm_head_weight[[self.false_id, self.true_id]].detach()

    # 前向计算
    def train(self, mode=True):
        # 冻结读出始终处于评估模式。
        return super().train(False)

    def forward(self, h):
        rows = self.binary_rows
        if h.shape[-1] != rows.shape[-1]:
            raise ContractError(
                "readout hidden mismatch: anchor %d vs lm_head rows %d"
                % (h.shape[-1], rows.shape[-1])
            )
        # 先将锚定状态转为 lm_head 的数据类型，使用冻结行完成线性计算，再将结果转回 FP32 以计算损失和指标。
        with torch.autocast(device_type=h.device.type, enabled=False):
            logits = torch.nn.functional.linear(h.to(rows.dtype), rows)
        return logits.float()
