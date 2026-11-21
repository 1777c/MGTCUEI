"""读取语言骨干实验组配置并解析本地模型目录。"""

import os
from pathlib import Path

from ...config import candidate
from ...utils import ContractError, read_json


def load_arms(root):
    config = read_json(root / "configs/LANGUAGE_ARMS.json")
    matrix = read_json(root / "configs/LANGUAGE_ABLATION_MATRIX.json")
    if [job["experiment_id"] for job in matrix["jobs"]] != list(config["arms"]):
        raise ContractError("the language ablation matrix and arm declarations differ")
    common = config["common_config"]
    fitted = candidate(
        cuei_span=common["cuei_span"], reader=common["reader"], gate=common["gate"],
        scale=common["lambda"], memory=common["memory"], seed=common["seed"], prompt=common["prompt"],
    )
    schedule = read_json(root / "configs/training.json")["schedule"]
    if fitted["gcn_lifecycle"] != common["gcn_lifecycle"] or any(
        schedule[key] != value for key, value in common["schedule"].items() if key in schedule
    ):
        raise ContractError("the language fit differs from its registered training configuration")
    return {
        arm_id: dict(arm, arm_id=arm_id, candidate=dict(fitted, injection_layers=arm["injection_layers"]))
        for arm_id, arm in config["arms"].items()
    }


def model_path(root, arm):
    local = os.environ.get(arm["root_env"])
    return Path(local).expanduser().resolve() if local else root / "models" / arm["directory"]
