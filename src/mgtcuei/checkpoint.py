"""原子保存可训练参数检查点，并恢复随机数发生器状态。"""

import os
import random
import numpy as np
import torch
from .utils import ContractError, sha256, write_json, read_json


def seed_everything(seed):
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False


#: 证据检查点允许的参数名前缀。
#:
#: 联合训练中的图编码器为可训练子模块，gcn. 随当前检查点保存。冻结的 LLM 和读出权重在加载模型时重建。
#:
#: 项目只有一种训练流程。每次保存时，prefixes_for 都从模型重新推导前缀并与此常量核对。
TRAINABLE_GROUP_PREFIXES = ("gcn.", "memory.", "reader.", "gates.")


def prefixes_for(model):
    if not hasattr(model, "checkpoint_prefixes"):
        return list(TRAINABLE_GROUP_PREFIXES)
    derived = list(model.checkpoint_prefixes())
    if derived != list(TRAINABLE_GROUP_PREFIXES):
        raise ContractError("checkpoint prefix set disagrees with the registered lifecycle")
    return derived


def readout_metadata(model):
    """返回每个检查点记录的冻结读出来源。

    读出权重每次加载时从冻结 LLM 重建。记录模型来源，以便恢复训练时核对 logits 所用的词元权重行。
    """
    if not hasattr(model, "readout_contract"):
        return None
    contract = model.readout_contract()
    return {
        "readout_type": contract["readout_type"],
        "readout_contract_version": contract["readout_contract_version"],
        "false_token_id": contract["false_token_id"],
        "true_token_id": contract["true_token_id"],
        "logit_order": contract["logit_order"],
        "score": contract["score"],
        "trainable_classifier_parameters": contract["trainable_classifier_parameters"],
        "rows_require_grad": contract["rows_require_grad"],
        "llm_root": contract.get("llm_root"),
        "llm_authority": contract.get("llm_authority"),
        "readout_source": "frozen MMed-Llama lm_head.weight rows; rebuilt on load",
    }


def save(path, model, optimizer, spec_id, state):
    path.parent.mkdir(parents=True, exist_ok=True)
    metadata = readout_metadata(model)
    payload = {
        "spec_id": spec_id,
        "weights": {n: p.detach().cpu().clone() for n, p in model.named_parameters() if p.requires_grad},
        "optimizer": optimizer.state_dict(),
        "state": state,
        "readout": metadata,
        "trainable_group_prefixes": prefixes_for(model),
        # 上游 GCN 预测器没有证据训练流程，此字段为 None，避免将上游拟合标记为证据训练。
        "gcn_lifecycle": getattr(model, "gcn_lifecycle", None),
        # 记录用于初始化编码器的上游检查点，恢复训练时可核对来源，无需重新加载上游参数。
        "gcn_initialization_sha256": getattr(model, "gcn_initialization_sha256", None),
        "gcn_initialization_run": getattr(model, "gcn_initialization_run", None),
        "rng_python": random.getstate(),
        "rng_numpy": np.random.get_state(),
        "rng_torch": torch.get_rng_state(),
        "rng_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    }
    temp = path.with_suffix(".tmp")
    with temp.open("wb") as f:
        torch.save(payload, f)
        f.flush()
        os.fsync(f.fileno())
    os.replace(temp, path)
    write_json(
        path.with_suffix(".json"),
        {"sha256": sha256(path), "spec_id": spec_id, "readout": metadata},
    )


def read(path, spec_id=None):
    receipt = read_json(path.with_suffix(".json"))
    if sha256(path) != receipt["sha256"] or (spec_id is not None and spec_id != receipt["spec_id"]):
        raise ContractError("checkpoint content/identity changed")
    # 只读取本地生成并与训练记录绑定的检查点。
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload["spec_id"] != receipt["spec_id"]:
        raise ContractError("checkpoint internal identity")
    return payload


def restore(path, model, optimizer, spec_id, rng=True, expect_readout=True):
    payload = read(path, spec_id)
    params = {n: p for n, p in model.named_parameters() if p.requires_grad}
    if set(params) != set(payload["weights"]):
        raise ContractError("trainable/freeze map changed")
    # 恢复训练时，模型与检查点必须采用同一训练流程。上游 GCN 没有证据训练流程，对应检查点也应如此。
    recorded_lifecycle = payload.get("gcn_lifecycle")
    live_lifecycle = getattr(model, "gcn_lifecycle", None)
    if recorded_lifecycle != live_lifecycle:
        raise ContractError(
            "resumed GCN lifecycle differs from the checkpoint metadata: checkpoint=%r model=%r"
            % (recorded_lifecycle, live_lifecycle)
        )
    # 恢复 last.pt 会载入联合微调后的全部可训练参数，包括 gcn.*。上游检查点只用于首次初始化。除非记录显示 GCN 从未更新，否则恢复后的编码器不得等于其初始状态。
    if expect_readout and live_lifecycle is not None:
        if not any(n.startswith("gcn.") for n in payload["weights"]):
            raise ContractError("a joint-lifecycle checkpoint must carry gcn.* state")
    if expect_readout:
        # 恢复时重新从冻结 LLM 构造读出，并核对已记录的来源，避免切换词元权重行。
        live = readout_metadata(model)
        if live is not None:
            recorded = payload.get("readout")
            if recorded is None:
                raise ContractError("checkpoint carries no frozen-readout identity")
            for key in ("readout_type", "false_token_id", "true_token_id", "logit_order"):
                if recorded.get(key) != live.get(key):
                    raise ContractError(
                        "resumed frozen readout differs from the checkpoint: " + key)
            if recorded.get("trainable_classifier_parameters") != 0:
                raise ContractError("冻结读出的检查点要求可训练分类器参数量为 0")
            if any(n.startswith("head.") for n in payload["weights"]):
                raise ContractError("checkpoint still carries task-head state")
    with torch.no_grad():
        for n, p in params.items():
            if payload["weights"][n].shape != p.shape:
                raise ContractError("checkpoint parameter shape mismatch")
            p.copy_(payload["weights"][n].to(p.device))
    if optimizer is not None:
        optimizer.load_state_dict(payload["optimizer"])
    if rng:
        random.setstate(payload["rng_python"])
        np.random.set_state(payload["rng_numpy"])
        torch.set_rng_state(payload["rng_torch"])
        if payload["rng_cuda"]:
            torch.cuda.set_rng_state_all(payload["rng_cuda"])
    return payload["state"]


def load_component(path, module, prefix, *, freeze=True):
    """将上游组件的权重加载到 module。

    freeze 默认为 True，供上游图评估和诊断使用。证据模型传入 freeze=False，从上游权重初始化后，以交叉熵继续微调编码器。
    """
    payload = read(path)
    expected = {n for n, _ in module.named_parameters()}
    weights = {n[len(prefix) :]: p for n, p in payload["weights"].items() if n.startswith(prefix)}
    if set(weights) != expected:
        raise ContractError("upstream component identity")
    with torch.no_grad():
        for n, p in module.named_parameters():
            p.copy_(weights[n])
    module.requires_grad_(not freeze)
    module.train(not freeze)
    return sha256(path)
