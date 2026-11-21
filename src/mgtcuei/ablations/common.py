"""在冻结的 Cold 开发划分上拟合和评估消融组。"""

from pathlib import Path

import pandas as pd
import torch

from .. import checkpoint, engine
from ..graph import GCN2, N_DRUGS
from ..model import EvidenceModel
from ..pipeline import GCN_MODE, Pipeline
from ..utils import ContractError, identity, read_json, safe_path, sha256, write_json


class AblationPipeline(Pipeline):
    """复用主线的训练引擎、数据加载器和训练计划。"""

    def __init__(self, root, results, arm_id, config, llm_root):
        super().__init__(root, device="cuda", llm_root=llm_root)
        self.results = Path(results)
        self.arm_id = arm_id
        self.config = config
        self.selection = self.results / arm_id / "selection.json"
        self.readout_authority = {
            "llm_root": "data/llm_contract",
            "llm_authority": self.data.files["data/llm_contract/MMED_LLAMA_MANIFEST.json"],
        }

    def prefit(self, *, resume=False):
        return self.fit_gcn("cold", 0, self.config["seed"], self.results, resume=resume)["run"]

    def encoder(self):
        return GCN2(GCN_MODE)

    def load_encoder(self, run, expected_sha=None):
        with torch.random.fork_rng(devices=[]):
            checkpoint.seed_everything(self.config["seed"])
            encoder = self.encoder()
        digest = checkpoint.load_component(run / "best.pt", encoder, "gcn.", freeze=False)
        if expected_sha is not None and digest != expected_sha:
            raise ContractError("the upstream checkpoint has changed")
        encoder.upstream_checkpoint_sha256 = digest
        encoder.upstream_run = run.relative_to(self.results).as_posix()
        return encoder

    def build_model(self, component, memory):
        spans = {self.config["cuei_span"]: {"injection_layers": self.config["injection_layers"]}}
        return EvidenceModel(
            self.boundary(), memory, component.to(self.device), self.config, spans,
            readout_authority=self.readout_authority,
        ).to(self.device)

    def evidence_model(self, component):
        seed = self.config["seed"]
        checkpoint.seed_everything(seed + 401)
        memory = self.memory_module(self.config["memory"])
        checkpoint.seed_everything(seed)
        model = self.build_model(component, memory)
        model.assert_joint_gcn()
        return model

    def forward_fn(self, model, graph):
        def forward(frame):
            d, s = model.embed(
                *graph,
                torch.as_tensor(frame.drug_idx.to_numpy(), device=self.device),
                torch.as_tensor(frame.disease_idx.to_numpy(), device=self.device),
                N_DRUGS,
            )
            return model(d, s, *self.tokens(frame, self.config["prompt"]))

        return forward

    def prepare_inputs(self, *frames):
        style = self.config["prompt"]
        cache = self._prompt_cache.setdefault(style, {})
        missing = set().union(*(frame.pair_id for frame in frames)) - cache.keys()
        if missing:
            cache.update(self.data.prompt_tokens(style, sorted(missing)))

    def fit(self, *, resume=False):
        encoder = self.load_encoder(self.prefit(resume=resume))
        model = self.evidence_model(encoder)
        train = self.data.partition("train")
        tune = self.data.partition("A")
        graph = self.graph_tensors("cold", 0)
        forward = self.forward_fn(model, graph)
        spec = self.base_spec("evidence", "cold", 0, self.config["seed"], self.config)
        spec.update(
            ablation_arm=self.arm_id,
            upstream={
                "gcn_initialization_run": encoder.upstream_run,
                "gcn_initialization_sha256": encoder.upstream_checkpoint_sha256,
                "gcn_checkpoint_role": "INITIALIZATION_AUTHORITY",
                "gcn_frozen_during_evidence": False,
                "gcn_in_evidence_optimizer": True,
                "gcn_lifecycle": "prefit_then_joint",
            },
            readout_authority=self.readout_authority,
            prompt=self.config["prompt"],
            prompt_sha256=self.data.files[f"data/prompts/{self.config['prompt']}.parquet"],
        )
        run = self.results / "jobs" / identity(spec)
        self.prepare_inputs(train, tune)
        receipt = engine.train(run, model, train, tune, forward, spec, self.schedule, resume=resume)
        frame = self.data.partition("B")
        self.evaluate_block(run, model, forward, spec, receipt, "B", frame)
        result = {
            "run": run.relative_to(self.results).as_posix(),
            "checkpoint_sha256": receipt["checkpoint_sha256"],
            "training_receipt_sha256": sha256(run / "receipt.json"),
            "best_epoch": receipt["best_epoch"],
        }
        write_json(self.selection, result)
        return result

    def evaluate_block(self, run, model, forward, spec, receipt, block, frame):
        self.prepare_inputs(frame)
        eval_spec = self.eval_spec(spec, block, frame)
        result = engine.evaluate(
            run, model, forward, {block: frame}, eval_spec, receipt, self.schedule,
        )
        return result, run / "evaluations" / identity(eval_spec)

    def dev_final(self):
        if not self.selection.is_file():
            raise ContractError(f"{self.arm_id}: fit the arm before evaluating Dev-FINAL")
        selection = read_json(self.selection)
        run = safe_path(self.results, selection["run"])
        receipt_path = run / "receipt.json"
        if sha256(receipt_path) != selection["training_receipt_sha256"]:
            raise ContractError("the training receipt has changed")
        receipt = read_json(receipt_path)
        spec = read_json(run / "spec.json")
        if (
            receipt["spec_id"] != identity(spec)
            or receipt["checkpoint_sha256"] != selection["checkpoint_sha256"]
            or spec["ablation_arm"] != self.arm_id
            or spec["config"] != self.config
            or spec["readout_authority"] != self.readout_authority
        ):
            raise ContractError("the selected checkpoint does not match this arm")
        upstream = spec["upstream"]
        encoder = self.load_encoder(
            safe_path(self.results, upstream["gcn_initialization_run"]),
            upstream["gcn_initialization_sha256"],
        )
        model = self.evidence_model(encoder)
        graph = self.graph_tensors("cold", 0)
        frame = pd.concat(
            [self.data.partition(block).assign(dev_final_block=block) for block in ("B", "C", "D")],
            ignore_index=True,
        )
        if not frame.pair_id.is_unique:
            raise ContractError("Dev-FINAL blocks overlap")
        result, directory = self.evaluate_block(
            run, model, self.forward_fn(model, graph), spec, receipt, "FINAL", frame,
        )
        return directory / "predictions/FINAL.parquet", result["threshold_used"]


def require_cuda(*, bf16=False):
    if not torch.cuda.is_available() or (bf16 and not torch.cuda.is_bf16_supported()):
        requirement = "a CUDA GPU with BF16 support" if bf16 else "a CUDA GPU"
        raise ContractError(f"this stage requires {requirement}")
