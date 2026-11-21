"""BioBERT 对照组使用原实验的 WordPiece 分词和 CLS 读出。"""

import os

import torch

from ...utils import ContractError
from . import backbones, biobert_engine
from .biobert_model import BioBERTEvidenceModel
from .llama_runner import LanguageArmRunner

MAX_TOKENS = 256


class BioBERTArmRunner(LanguageArmRunner):
    def __init__(self, root, arm):
        super().__init__(root, arm)
        self._tok = None
        self._texts = {}
        biobert_engine.install()

    def boundary(self):
        if self._boundary is None:
            self._boundary = backbones.load_biobert_boundary(self.llm_root, str(self.device))
        return self._boundary

    def build_model(self, component, memory):
        model = BioBERTEvidenceModel(
            self.boundary(), memory, component.to(self.device), self.config,
            readout_authority=self.readout_authority,
        ).to(self.device)
        model.assert_trainable_head()
        return model

    def prepare_inputs(self, *frames):
        self.prepare_texts(frames, self.config["prompt"])

    def tokenizer(self):
        if self._tok is None:
            from tokenizers import Tokenizer
            p = os.path.join(self.llm_root, "tokenizer.json")
            if os.path.isfile(p):
                self._tok = Tokenizer.from_file(p)
            else:
                # BioBERT 仓库提供 WordPiece 的 vocab.txt。缺少 tokenizer.json 时，从相同文件加载等效的快速分词器。
                from transformers import AutoTokenizer
                self._tok = AutoTokenizer.from_pretrained(
                    self.llm_root, local_files_only=True).backend_tokenizer
        return self._tok

    def prepare_texts(self, frames, prompt):
        """为所需配对加载 prompt_text，供 BioBERT 重新分词。"""
        import pandas as pd

        required = set().union(*(set(f.pair_id) for f in frames))
        existing = self._texts.get(prompt)
        missing = required if existing is None else required - set(existing.index)
        if missing:
            table = pd.read_parquet(
                self.data.path(f"data/prompts/{prompt}.parquet"),
                columns=["pair_id", "prompt_text"],
                filters=[("pair_id", "in", sorted(missing))],
            ).set_index("pair_id")
            if set(table.index) != missing or table.index.has_duplicates:
                raise ContractError("prompt text coverage mismatch")
            self._texts[prompt] = table if existing is None else pd.concat([existing, table])
        return self._texts[prompt]

    def tokens(self, f, prompt):
        table = self.prepare_texts([f], prompt)
        rows = []
        for text in table.loc[f.pair_id, "prompt_text"].tolist():
            ids = self.tokenizer().encode(text, add_special_tokens=True).ids[:MAX_TOKENS]
            if not ids:
                raise ContractError("empty BioBERT tokenisation")
            rows.append(ids)
        length = max(map(len, rows))
        ids = torch.zeros(len(rows), length, dtype=torch.long, device=self.device)
        mask = torch.zeros_like(ids)
        for i, row in enumerate(rows):
            ids[i, : len(row)] = torch.as_tensor(row, device=self.device)
            mask[i, : len(row)] = 1
        return ids, mask
