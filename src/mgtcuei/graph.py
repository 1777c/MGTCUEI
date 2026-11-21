"""兼容基线的两层 GCN，图传播所需量集中准备一次。"""

from dataclasses import dataclass
import torch
from torch import nn
from .utils import ContractError


@dataclass
class Propagation:
    edges: torch.Tensor
    weights: torch.Tensor
    degree_before: torch.Tensor
    mode: str
    nodes: int


def prepare_graph(edges, weights, nodes, mode):
    if isinstance(edges, Propagation):
        if edges.mode != mode or edges.nodes != nodes:
            raise ContractError("prepared graph identity mismatch")
        return edges
    if mode not in ("original", "self_loop", "isolated_only"):
        raise ContractError("unknown GCN mode")
    if edges.shape != (2, weights.numel()) or edges.dtype != torch.long:
        raise ContractError("edge schema")
    if edges.device != weights.device or not torch.isfinite(weights).all() or (weights < 0).any():
        raise ContractError("edge device/weight")
    if edges.numel() and ((edges < 0).any() or (edges >= nodes).any()):
        raise ContractError("edge index")
    # 先合并冻结邻接矩阵中的重复坐标，再补齐缺少的单位自环。
    a = torch.sparse_coo_tensor(edges, weights.float(), (nodes, nodes)).coalesce()
    e, w = a.indices(), a.values()
    degree = torch.zeros(nodes, device=w.device).index_add(0, e[1], w)
    if mode != "original":
        ids = torch.arange(nodes, device=e.device)
        if mode == "self_loop":
            diagonal = e[0] == e[1]
            if not torch.equal(w[diagonal], torch.ones_like(w[diagonal])):
                raise ContractError("在 self_loop 模式中，已有自环的权重必须为 1")
            present = torch.zeros(nodes, dtype=torch.bool, device=e.device)
            present[e[0, diagonal]] = True
            ids = ids[~present]
        if mode == "isolated_only":
            out = torch.zeros_like(degree).index_add(0, e[0], w)
            if not torch.equal(out == 0, degree == 0):
                raise ContractError("isolated-only requires undirected zero-degree topology")
            ids = ids[degree == 0]
        e = torch.cat((e, torch.stack((ids, ids))), 1)
        w = torch.cat((w, torch.ones(len(ids), device=w.device)))
        a = torch.sparse_coo_tensor(e, w, (nodes, nodes)).coalesce()
        e, w = a.indices(), a.values()
    deg = torch.zeros(nodes, device=w.device).index_add(0, e[1], w)
    inv = deg.clamp_min(1e-30).rsqrt()
    inv = torch.where(deg > 0, inv, 0)
    return Propagation(e, inv[e[0]] * w * inv[e[1]], degree, mode, nodes)


class WeightedGCNConv(nn.Module):
    def __init__(self, n, m):
        super().__init__()
        self.lin = nn.Linear(n, m, bias=False)
        self.bias = nn.Parameter(torch.zeros(m))
        nn.init.xavier_uniform_(self.lin.weight)

    def forward(self, x, g):
        if x.device != g.edges.device or len(x) != g.nodes:
            raise ContractError("GCN feature/graph identity")
        z = self.lin(x)
        return (
            torch.zeros_like(z).index_add(0, g.edges[1], z[g.edges[0]] * g.weights[:, None].to(z.dtype))
            + self.bias
        )


class GCN2(nn.Module):
    def __init__(self, mode="self_loop", input_dim=256):
        super().__init__()
        self.mode = mode
        self.conv1 = WeightedGCNConv(input_dim, 128)
        self.conv2 = WeightedGCNConv(128, 128)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x, edges, weights=None):
        if not torch.isfinite(x).all():
            raise ContractError("nonfinite node features")
        g = prepare_graph(edges, weights, len(x), self.mode)
        return self.conv2(self.dropout(torch.relu(self.conv1(x, g))), g)


#: 配对表示及其各分量的维度。
#:
#: h_d 为药物表示，128 维。h_s 为疾病表示，128 维。
#: h_int = [|h_d - h_s| ; h_d (逐元素乘) h_s]，256 维，作为交互槽输入。
#: h_pair_full = [h_d ; h_s ; h_int]，512 维。
#:
#: TypedMemory 的交互槽输入为 256 维 h_int。上游预测头使用完整的 512 维 h_pair_full。
DRUG_DIM = 128
DISEASE_DIM = 128
INTERACTION_DIM = 256
PAIR_FULL_DIM = DRUG_DIM + DISEASE_DIM + INTERACTION_DIM

#: 冻结图中 [0, N_DRUGS) 为药物节点，[N_DRUGS, 4235) 为疾病节点。统一使用该边界索引嵌入。
N_DRUGS = 3084
N_NODES = 4235
HISTORICAL_H_P = "h_pair_full"
SLOT_SOURCE_DIMS = (DRUG_DIM, DISEASE_DIM, INTERACTION_DIM)


def pair_features(d, s):
    """拼接 h_pair_full = [h_d ; h_s ; |h_d - h_s| ; h_d * h_s]，宽度为 PAIR_FULL_DIM。"""
    return torch.cat((d, s, (d - s).abs(), d * s), -1)


class GraphPredictor(nn.Module):
    def __init__(self, mode="self_loop", route="gcn", n_drugs=3084):
        super().__init__()
        self.route = route
        self.n_drugs = n_drugs
        if route == "gcn":
            self.gcn = GCN2(mode)
        width = 256 if route == "disease" else 512
        self.head = nn.Sequential(nn.Linear(width, 256), nn.ReLU(), nn.Linear(256, 2))

    def forward(self, x, e, w, d, s):
        if self.route == "gcn":
            z = self.gcn(x, e, w)
            v = pair_features(z[d], z[self.n_drugs + s])
        elif self.route == "disease":
            v = x[self.n_drugs + s]
        elif self.route == "raw":
            v = torch.cat((x[d], x[self.n_drugs + s]), -1)
        else:
            raise ContractError("unknown graph predictor")
        return self.head(v)
