"""读取冻结的开发划分、正式实验折、图结构和提示。

各科学资产由 artifacts/manifests/SCIENTIFIC_ASSETS.json 中的 SHA256 绑定，首次读取时校验。数据划分直接来自冻结文件。
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse

from .graph import N_DRUGS, N_NODES
from .utils import ContractError, read_json, sha256

DEV_BLOCKS = ("A", "B", "C", "D")


class Data:
    def __init__(self, root):
        self.root = Path(root).resolve()
        self.files = read_json(self.root / "artifacts/manifests/SCIENTIFIC_ASSETS.json")["files"]
        self.development = read_json(self.root / "artifacts/manifests/DEVELOPMENT.json")
        if self.development.get("status") != "FROZEN":
            raise ContractError("unfrozen development authority")
        raw = (self.root / "artifacts/manifests/BENCHMARKS_V2.json").read_bytes()
        self.benchmarks = json.loads(raw)
        self._verified: set[str] = set()

    # 资产读取
    def path(self, relative: str) -> Path:
        if relative not in self.files:
            raise ContractError("unregistered scientific asset: " + relative)
        path = (self.root / relative).resolve()
        if not path.is_relative_to(self.root):
            raise ContractError("asset path escapes project")
        if relative not in self._verified:
            if sha256(path) != self.files[relative]:
                raise ContractError("scientific asset changed: " + relative)
            self._verified.add(relative)
        return path

    def features(self) -> np.ndarray:
        x = np.load(self.path("data/node_features.npy"), allow_pickle=False)
        if x.shape != (N_NODES, 256) or x.dtype != np.float32 or not np.isfinite(x).all():
            raise ContractError("node feature contract: expected (4235, 256) finite float32")
        return x

    def prompts(self, style: str) -> pd.DataFrame:
        if style not in ("semantic_compact_v1", "basic_name_only_v1"):
            raise ContractError("prompt registry")
        return pd.read_parquet(self.path(f"data/prompts/{style}.parquet"))

    def prompt_tokens(self, style: str, pair_ids) -> dict:
        table = pd.read_parquet(
            self.path(f"data/prompts/{style}.parquet"),
            columns=["pair_id", "input_token_ids"],
            filters=[("pair_id", "in", list(pair_ids))],
        ).set_index("pair_id")
        missing = set(pair_ids) - set(table.index)
        if missing:
            raise ContractError("missing prompt for %d pair(s)" % len(missing))
        return table["input_token_ids"].to_dict()

    def entities(self):
        return {
            "drug": pd.read_parquet(self.path("data/entities/drug_entity_index.parquet")),
            "disease": pd.read_parquet(self.path("data/entities/disease_entity_index.parquet")),
            "formal": pd.read_parquet(self.path("data/entities/formal_entity_index.parquet")),
        }

    # 配对划分
    @staticmethod
    def _check(frame: pd.DataFrame, name: str) -> pd.DataFrame:
        if frame.empty:
            raise ContractError("empty partition: " + name)
        if frame.pair_id.duplicated().any():
            raise ContractError("duplicate pair identity: " + name)
        if not frame.label.isin([0, 1]).all():
            raise ContractError("label must be 0/1: " + name)
        if not frame.drug_idx.between(0, N_DRUGS - 1).all():
            raise ContractError("drug index out of range: " + name)
        if not frame.disease_idx.between(0, N_NODES - N_DRUGS - 1).all():
            raise ContractError("disease index out of range: " + name)
        return frame

    def partition(self, part: str, protocol: str = "cold", fold: int = 0) -> pd.DataFrame:
        """返回一个冻结的数据分区。

        cold 提供 Train 和 Dev-A 至 Dev-D，正式基准协议提供指定折的 Train、Validation、Test。
        """
        if protocol == "cold":
            if part == "train":
                rel = self.development["common_train"]
            elif part in DEV_BLOCKS:
                rel = self.development["blocks"][part]["path"]
            else:
                raise ContractError("unknown development partition: " + part)
        elif protocol in self.benchmarks["protocols"]:
            if part not in ("train", "val", "test"):
                raise ContractError("unknown benchmark partition")
            entry = self.benchmarks["protocols"][protocol][str(fold)][part]
            return self._check(self._read_pairs(entry["path"]), f"{protocol}/{fold}/{part}")
        else:
            raise ContractError("unregistered protocol: " + protocol)
        return self._check(self._read_pairs(rel), f"{protocol}/{fold}/{part}")

    def _read_pairs(self, rel: str) -> pd.DataFrame:
        frame = pd.read_parquet(self.path(rel))
        if "formal_pair_id" in frame.columns:
            frame = frame.rename(columns={"formal_pair_id": "pair_id"})
        return frame

    def final_dev(self) -> pd.DataFrame:
        """返回 Dev-B、Dev-C 和 Dev-D 合并得到的 Dev-FINAL 评估集。"""
        frames = [self.partition(b) for b in ("B", "C", "D")]
        pooled = pd.concat(frames, ignore_index=True)
        if pooled.pair_id.duplicated().any():
            raise ContractError("Dev-FINAL blocks overlap")
        return pooled

    def evaluation_frame(self, protocol: str, fold: int, block: str) -> pd.DataFrame:
        if protocol == "cold" and block == "Dev-FINAL":
            return self.final_dev()
        return self.partition(block, protocol=protocol, fold=fold)

    # 图结构
    def graph(self, protocol: str = "cold", fold: int = 0):
        """返回单折的有向边列表和权重。

        A_ds 与本折训练正例严格一致，并展开成双向边。再加入无向加权疾病相似矩阵 A_ss，其对角线已包含自环。
        """
        if protocol in self.benchmarks["protocols"]:
            entry = self.benchmarks["protocols"][protocol][str(fold)]
            train = self._read_pairs(entry["train"]["path"])
            rel = entry["graph"]
        else:
            train = self.partition("train", protocol, fold)
            rel = "data/cold/A_ds.npz"
        a = sparse.load_npz(self.path(rel)).tocoo()
        expected = set(map(tuple, train[train.label == 1][["drug_idx", "disease_idx"]].values))
        if set(zip(a.row, a.col)) != expected:
            raise ContractError("A_ds is not exactly the Train positives")
        if a.shape != (N_DRUGS, N_NODES - N_DRUGS):
            raise ContractError("A_ds shape")
        s = sparse.load_npz(self.path("data/A_ss_sparse.npz")).tocoo()
        edges = np.stack(
            [
                np.concatenate([a.row, a.col + N_DRUGS, s.row + N_DRUGS]),
                np.concatenate([a.col + N_DRUGS, a.row, s.col + N_DRUGS]),
            ]
        )
        weights = np.concatenate([a.data, a.data, s.data]).astype("float32")
        return edges, weights

    # 数据验证
    def verify(self) -> dict:
        for rel in self.files:
            self.path(rel)
        x = self.features()
        dev = {b: self.partition(b) for b in DEV_BLOCKS}
        train = self.partition("train")
        pooled = pd.concat(dev.values(), ignore_index=True)
        if pooled.pair_id.duplicated().any():
            raise ContractError("development blocks overlap")
        if set(train.drug_idx) & set(pooled.drug_idx):
            raise ContractError("development drug isolation failed")
        protocols = {}
        for protocol, cfg in self.benchmarks["protocols"].items():
            protocols[protocol] = {}
            for fold in sorted(cfg, key=int):
                parts = {p: self.partition(p, protocol, int(fold)) for p in ("train", "val", "test")}
                for p, f in parts.items():
                    if not (f.label == 1).sum() == (f.label == 0).sum():
                        raise ContractError(f"{protocol}/{fold}/{p} is not 1:1")
                self.graph(protocol, int(fold))
                protocols[protocol][fold] = {p: len(f) for p, f in parts.items()}
        return {
            "status": "PASS",
            "assets": len(self.files),
            "nodes": int(x.shape[0]),
            "feature_dim": int(x.shape[1]),
            "drugs": N_DRUGS,
            "diseases": N_NODES - N_DRUGS,
            "development": {b: len(f) for b, f in dev.items()},
            "cold_train": len(train),
            "protocols": protocols,
        }
