"""仅使用冻结 LLM 的对照，路径为提示、冻结 MMed-Llama、True/False 读出。

复现 S1 的 FROZEN_LM_HEAD_LLM_ONLY 路线。使用与完整模型相同的冻结边界和 lm_head 权重行，无证据注入或优化器更新。
"""

from __future__ import annotations

import pandas as pd
import torch
from scipy.special import expit

from .llm import FrozenBoundary, anchor
from .readout import FrozenBinaryReadout


class LLMOnlyScorer:
    def __init__(self, llm_root, *, prompt_style="semantic_compact_v1", device="cuda"):
        self.boundary = FrozenBoundary.load(llm_root, device=device)
        self.readout = FrozenBinaryReadout(
            self.boundary.lm_head_weight, expected_hidden=self.boundary.hidden_dim
        ).to(device)
        self.prompt_style = prompt_style
        self.device = device

    def score(self, data, frame: pd.DataFrame, *, batch_size=32) -> pd.DataFrame:
        token_map = data.prompt_tokens(self.prompt_style, list(frame.pair_id))
        parts = []
        with torch.no_grad():
            for start in range(0, len(frame), batch_size):
                f = frame.iloc[start:start + batch_size]
                rows = [token_map[pid] for pid in f.pair_id]
                length = max(map(len, rows))
                ids = torch.zeros(len(rows), length, dtype=torch.long, device=self.device)
                mask = torch.zeros_like(ids)
                for i, row in enumerate(rows):
                    ids[i, : len(row)] = torch.as_tensor(row, device=self.device)
                    mask[i, : len(row)] = 1
                h = self.boundary(ids, mask)
                logits = self.readout(anchor(h, mask)).float().cpu().numpy()
                p = f[["pair_id", "drug_idx", "disease_idx"]].copy()
                p["logit_false"] = logits[:, 0]
                p["logit_true"] = logits[:, 1]
                p["score"] = logits[:, 1] - logits[:, 0]
                p["probability"] = expit(p["score"])
                parts.append(p)
        return pd.concat(parts, ignore_index=True)
