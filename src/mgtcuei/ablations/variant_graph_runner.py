"""在 Cold 开发划分上监督预拟合图编码器变体，再联合训练 CUEI。"""

import torch

from .. import checkpoint, engine
from ..config import candidate
from ..utils import identity
from .common import AblationPipeline
from .graph_variants import ARMS, GraphVariantPredictor


class VariantRunner(AblationPipeline):
    def __init__(self, root, arm_id, llm_root):
        super().__init__(root, root / "results/ablations/graph", arm_id, candidate(), llm_root)

    def encoder(self):
        return ARMS[self.arm_id][1]()

    def prefit(self, *, resume=False):
        seed = self.config["seed"]
        checkpoint.seed_everything(seed)
        model = GraphVariantPredictor(self.encoder()).to(self.device)
        graph = self.graph_tensors("cold", 0)
        train = self.data.partition("train")
        tune = self.data.partition("A")
        config = {
            "name": ARMS[self.arm_id][0],
            "mode": "original",
            "route": "gcn",
            "encoder_code": self.arm_id,
        }
        spec = self.base_spec("gcn", "cold", 0, seed, config)
        run = self.results / "jobs" / identity(spec)

        def forward(frame):
            return model(
                *graph,
                torch.as_tensor(frame.drug_idx.to_numpy(), device=self.device),
                torch.as_tensor(frame.disease_idx.to_numpy(), device=self.device),
            )

        engine.train(run, model, train, tune, forward, spec, self.schedule_for("gcn"), resume=resume)
        return run
