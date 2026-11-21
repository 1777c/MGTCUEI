"""MGTCUEI 的图证据预测路径与联合训练。

图证据依次经过 TypedMemory、Reader、LayerResidualGate 和冻结的 MMed-Llama 上层，取得锚定后的最终隐藏状态，再用冻结 lm_head 的 False（4139）和 True（2575）两行读出 logits。读出只接收最终隐藏状态张量，分类器可训练参数为 0。

记忆、读取器和门控由证据模型的交叉熵端到端优化。EvidenceModel.optimizer_parameters 约束梯度和优化器的参数范围。

图编码器由监督式上游 GCN-SL 检查点初始化，再与记忆、读取器和门控联合微调。gcn 跟随模型切换训练和评估模式，embed 在训练时检查图路径可微。
"""

import torch
from typing import Callable
from torch import nn
from .llm import anchor
from .readout import (
    LOGIT_ORDER,
    READOUT_ARCHITECTURE,
    READOUT_CONTRACT_VERSION,
    READOUT_TYPE,
    ROUTED_INPUT_SOURCE,
    FrozenBinaryReadout,
)
from .readers import READER_REGISTRY
from .residual import ResidualGate
from .utils import ContractError

#: 读出架构及其输入来源，用于记录实验配置。
HEAD_ARCHITECTURE = READOUT_ARCHITECTURE
HEAD_INPUT_SOURCE = ROUTED_INPUT_SOURCE

#: 记忆模块的联合训练设置。
MEMORY_LIFECYCLE = "joint"
MEMORY_TRAINABLE_DURING_EVIDENCE = True

#: 证据优化器更新图编码器、记忆、读取器及门控四组参数。
TRAINABLE_GROUPS = ("gcn", "memory", "reader", "gate")
FROZEN_GROUPS = ("llm", "lm_head_rows")

#: 监督式 GCN-SL 生成图编码器初始参数，证据模型随后联合微调编码器。
GCN_LIFECYCLE = "prefit_then_joint"
GCN_LIFECYCLES = ("prefit_then_joint",)
CHECKPOINT_PREFIXES = ("gcn.", "memory.", "reader.", "gates.")

#: 记录分类任务头的可训练参数量和实例数。
TRAINABLE_HEAD_PARAMETERS = 0
ACTIVE_TEXTONLYHEAD_COUNT = 0


def component_state_sha(model):
    """计算各子模块当前参数的 SHA256。

    用于比较匹配消融组的组件初始化，以及核对恢复训练时的模型状态。
    """
    import hashlib

    modules = {"gcn": model.gcn, "memory": model.memory, "reader": model.reader, "gate": model.gates}
    out = {}
    for name, module in modules.items():
        digest = hashlib.sha256()
        for parameter_name, parameter in sorted(module.named_parameters()):
            digest.update(parameter_name.encode())
            digest.update(str(tuple(parameter.shape)).encode())
            digest.update(parameter.detach().float().cpu().contiguous().numpy().tobytes())
        out[name] = digest.hexdigest()
    return out


class EvidenceModel(nn.Module):
    """完整 PURE CUEI 模型，记忆与其他可训练组件联合优化。"""

    memory_lifecycle = MEMORY_LIFECYCLE

    def __init__(self, boundary, memory, gcn, config, cuei_spans, readout_authority=None):
        """gcn 必须是可训练的编码器对象。

        编码器从上游拟合获得初始参数，并在证据训练中参与联合优化。
        """
        super().__init__()
        self.llm = boundary
        self.memory = memory
        self.config = config
        mode = config.get("gcn_lifecycle", GCN_LIFECYCLE)
        if mode not in GCN_LIFECYCLES:
            raise ContractError("unregistered GCN lifecycle")
        self.gcn_lifecycle = mode
        if gcn is None:
            raise ContractError("the evidence model requires a graph encoder; joint training is the only lifecycle")
        self.gcn = gcn
        self.gcn.requires_grad_(True)
        self.gcn.train(True)
        # 记录上游初始化检查点的标识，以便核对联合拟合所用初始参数。恢复训练时不重新加载上游权重。
        self.gcn_initialization_sha256 = getattr(gcn, "upstream_checkpoint_sha256", None)
        self.gcn_initialization_run = getattr(gcn, "upstream_run", None)
        # 联合训练使记忆保持可训练。
        self.memory.requires_grad_(True)
        self.memory.train(True)
        # 预测使用冻结的预训练 lm_head 两行，可训练参数为 0。模型不定义 self.head。
        self.readout = FrozenBinaryReadout(
            boundary.lm_head_weight,
            expected_hidden=boundary.hidden_dim,
            authority=readout_authority,
        )
        self.injection_layers = list(config["injection_layers"])
        span = config["cuei_span"]
        if span not in cuei_spans:
            raise ContractError("unregistered CUEI span")
        if self.injection_layers != cuei_spans[span]["injection_layers"]:
            raise ContractError("span/injection_layers mismatch")
        self.gates = nn.ModuleDict()
        self.injection_trace = []
        self.capture_gradients = False
        self.position_gradients = {}
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(config["seed"] + 211)
            cls: Callable[..., nn.Module] | None = READER_REGISTRY.get(config["reader"])
            if cls is None:
                raise ContractError("reader registry")
            self.reader = cls(boundary.hidden_dim)
        # 各读取器对 V 和 Wo 使用相同的初始化。
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(config["seed"] + 307)
            self.reader.v_proj.reset_parameters()
            self.reader.out_proj.reset_parameters()
        self.gates = nn.ModuleDict(
            {str(i): ResidualGate(config["gate"], config["lambda"]) for i in self.injection_layers}
        )
        self.assert_joint_memory()

    # 模型约定
    def assert_joint_memory(self):
        params = list(self.memory.parameters())
        if not params:
            raise ContractError("memory has no parameters")
        if not all(p.requires_grad for p in params):
            raise ContractError("joint lifecycle requires every memory parameter trainable")
        return True

    def assert_frozen_readout(self):
        """确认各代码路径均无可训练的读出参数。"""
        if self.readout.trainable_parameter_count() != 0:
            raise ContractError("readout must have 0 trainable classifier parameters")
        if list(self.readout.parameters()):
            raise ContractError("readout must expose no nn.Parameter at all")
        if hasattr(self, "head"):
            raise ContractError("冻结读出的证据模型不能包含独立的分类任务头")
        return True

    def trainable_groups(self):
        """按登记顺序返回四组可训练参数。"""
        return {
            "gcn": [p for p in self.gcn.parameters() if p.requires_grad],
            "memory": [p for p in self.memory.parameters() if p.requires_grad],
            "reader": [p for p in self.reader.parameters() if p.requires_grad],
            "gate": [p for g in self.gates.values() for p in g.parameters() if p.requires_grad],
        }

    def registered_groups(self):
        return TRAINABLE_GROUPS

    def checkpoint_prefixes(self):
        """返回证据检查点允许的参数名前缀。"""
        return CHECKPOINT_PREFIXES

    def optimizer_parameters(self):
        """返回可加入优化器的登记参数组。

        发现未登记的可训练参数时拒绝继续。冻结读出没有可训练参数。
        """
        params = []
        for group in self.registered_groups():
            params.extend(self.trainable_groups()[group])
        ids = {id(p) for p in params}
        for p in self.llm.parameters():
            if id(p) in ids:
                raise ContractError("frozen LLM parameter inside the evidence optimizer")
        for name, p in self.named_parameters():
            if p.requires_grad and id(p) not in ids:
                raise ContractError("unregistered trainable parameter outside the groups: " + name)
        self.assert_frozen_readout()
        return params

    def assert_joint_gcn(self):
        """确认图编码器保持可训练。"""
        params = list(self.gcn.parameters())
        if not params:
            raise ContractError("the graph encoder has no parameters")
        if not all(p.requires_grad for p in params):
            raise ContractError("the joint GCN lifecycle requires every encoder parameter trainable")
        if not any(n.startswith("gcn.") for n, _ in self.named_parameters()):
            raise ContractError("the encoder is not bound as a submodule")
        return True

    def embed(self, x, edges, weights, drug_idx, disease_idx, n_drugs):
        """从图编码器提取配对表示，作为图信息进入模型的唯一路径。

        开启梯度时要求编码器调用可微，避免分离或缓存的嵌入令编码器停止接收梯度。验证和评估在 torch.no_grad() 下运行，此检查只适用于训练。
        """
        z = self.gcn(x, edges, weights)
        if torch.is_grad_enabled() and not z.requires_grad:
            raise ContractError("joint GCN lifecycle requires a differentiable graph path")
        return z[drug_idx], z[n_drugs + disease_idx]

    def gcn_grad_norm(self):
        total = 0.0
        for p in self.gcn.parameters():
            if p.grad is not None:
                total += float(p.grad.detach().norm()) ** 2
        return total ** 0.5

    def readout_contract(self):
        return self.readout.readout_contract()

    def memory_grad_norm(self):
        total = 0.0
        for p in self.memory.parameters():
            if p.grad is not None:
                total += float(p.grad.detach().norm()) ** 2
        return total ** 0.5

    def reader_at(self, layer):
        if layer not in self.injection_layers:
            raise ContractError("unregistered injection")
        return self.reader

    def train(self, mode=True):
        super().train(mode)
        self.llm.eval()
        self.readout.eval()
        # 记忆随模型切换训练与评估模式。
        self.memory.train(mode)
        # 图编码器也随模型切换模式，并保持 requires_grad 为 True。
        self.gcn.train(mode)
        return self

    def forward(self, d, s, ids, mask):
        # d 和 s 是本折 GCN 嵌入，模块中只有 self.memory 消费这两个来源于图的张量。
        m = self.memory(d, s)
        self.injection_trace = []
        self.position_gradients = {}

        def injection(layer):
            def apply(h):
                reader = self.reader_at(layer)
                contribution = reader(h, m)
                self.injection_trace.append({"layer": layer, "reader_id": id(reader), "memory_id": id(m)})
                if self.capture_gradients and contribution.requires_grad:

                    def record(gradient):
                        # 使用 VJP 单独获取本次读取器调用受到的真实损失梯度。
                        with torch.enable_grad():
                            local = reader(h.detach(), m.detach())
                            gradients = torch.autograd.grad(
                                local, tuple(reader.parameters()), gradient.detach()
                            )
                        self.position_gradients[str(layer)] = {
                            n: float(g.detach().norm())
                            for (n, _), g in zip(reader.named_parameters(), gradients)
                        }
                        return gradient

                    contribution.register_hook(record)
                return self.gates[str(layer)](h, contribution, mask)

            return apply

        h = self.llm(ids, mask, {i: injection(i) for i in self.injection_layers})
        # 冻结读出只接收经 CUEI 修改的 LLM 最终锚定隐藏状态。logits[:,0] 对应 False，logits[:,1] 对应 True。
        return self.readout(anchor(h, mask))
