#!/usr/bin/env python3
"""S2：T2 图编码器消融。

GCN 深度组复用主线 prepare_graph 和 WeightedGCNConv，保留对称归一化、偏置和初始化。GCN-2L 匹配重放直接使用 GCN2(mode="original")。GAT 和 GraphSAGE 通过 coalesce_original 合并重复边，复用主线 ORIG 拓扑的原始边。

所有组均遵循 ORIG 拓扑，不显式添加自环。GAT-2L 设置 add_self_loops=False。GAT 注意力和 GraphSAGE 均值聚合均在此实现。

GCN 隐藏层先用 ReLU 再用 dropout，最后一层无激活。GAT 的注意力 logits 使用 LeakyReLU(0.2)，隐藏层输出使用 ReLU 类激活，最终层无激活。GraphSAGE 隐藏层也使用 ReLU 类激活，最终层无激活。各组按实际参数量报告。
"""

from __future__ import annotations

import torch
from torch import nn

from ..graph import (
    GCN2,
    WeightedGCNConv,
    prepare_graph,
)

HIDDEN = 128
DROPOUT = 0.2
INPUT_DIM = 256
N_NODES = 4235
N_DRUGS = 3084


# 边处理
def coalesce_original(edges: torch.Tensor, weights: torch.Tensor, nodes: int):
    """合并冻结邻接矩阵的重复坐标，不添加自环。

    复用 prepare_graph(..., "original") 的第一步。GAT 和 GraphSAGE 直接使用原始边权以便各自加权，GCN 的对称归一化 D^-1/2 A D^-1/2 不在此预先应用。
    """
    a = torch.sparse_coo_tensor(edges, weights.float(), (nodes, nodes)).coalesce()
    return a.indices(), a.values()


def assert_parity_with_mainline(edges, weights, nodes):
    """验证合并结果与主线一致。

    用合并后的边重新计算归一化权重，再用 torch.equal 与 prepare_graph 的结果做精确比较。
    """
    e0, w0 = coalesce_original(edges, weights, nodes)
    g = prepare_graph(edges.clone(), weights.clone(), nodes, "original")
    if not torch.equal(e0, g.edges):
        raise AssertionError("edge index diverged from prepare_graph")
    deg = torch.zeros(nodes, device=w0.device).index_add(0, e0[1], w0)
    inv = deg.clamp_min(1e-30).rsqrt()
    inv = torch.where(deg > 0, inv, 0)
    if not torch.equal(inv[e0[0]] * w0 * inv[e0[1]], g.weights):
        raise AssertionError("coarsened weights do not normalise to prepare_graph's")
    if not torch.equal(deg, g.degree_before):
        raise AssertionError("degree diverged from prepare_graph")
    return True


# GCN
class GCNOrig(nn.Module):
    """构造任意深度的 GCN-ORIG，宽度为 256 → 128 → 128 → … → 128。

    两层版本与主线 GCN2(mode="original") 按位一致，匹配重放直接使用主线类。
    """

    def __init__(self, depth: int = 2, input_dim: int = INPUT_DIM):
        super().__init__()
        if depth < 1:
            raise ValueError("depth must be >= 1")
        self.depth = depth
        self.mode = "original"
        dims = [input_dim] + [HIDDEN] * depth
        self.convs = nn.ModuleList(
            WeightedGCNConv(dims[i], dims[i + 1]) for i in range(depth))
        self.dropout = nn.Dropout(DROPOUT)

    def forward(self, x, edges, weights=None):
        g = prepare_graph(edges, weights, len(x), self.mode)
        h = x
        for i, conv in enumerate(self.convs):
            h = conv(h, g)
            if i < self.depth - 1:          # 最后一层不加激活。
                h = self.dropout(torch.relu(h))
        return h


def gcn1_orig():
    return GCNOrig(depth=1)


def gcn2_orig():
    """GCN-2L 参考组使用主线类。"""
    return GCN2(mode="original")


def gcn3_orig():
    return GCNOrig(depth=3)


def gcn4_orig():
    return GCNOrig(depth=4)


# GAT
class GATLayer(nn.Module):
    """显式实现标准四头 GAT 层。

    add_self_loops=False。冻结邻接矩阵缺少自环时，节点不会收到自身的注意力消息。
    """

    def __init__(self, n_in, out_per_head=32, heads=4, concat=True, negative_slope=0.2):
        super().__init__()
        self.heads = heads
        self.out_per_head = out_per_head
        self.concat = concat
        self.add_self_loops = False
        self.negative_slope = negative_slope
        self.out_dim = out_per_head * heads if concat else out_per_head
        self.lin = nn.Linear(n_in, self.out_dim, bias=False)
        self.att_src = nn.Parameter(torch.zeros(heads, out_per_head))
        self.att_dst = nn.Parameter(torch.zeros(heads, out_per_head))
        self.bias = nn.Parameter(torch.zeros(self.out_dim))
        nn.init.xavier_uniform_(self.lin.weight)
        nn.init.xavier_uniform_(self.att_src)
        nn.init.xavier_uniform_(self.att_dst)

    def forward(self, x, edge_index, edge_weight=None):
        n = len(x)
        z = self.lin(x).view(n, self.heads, self.out_per_head)
        src, dst = edge_index[0], edge_index[1]
        # 计算每条边、每个头的注意力 logits。
        logits = (z[src] * self.att_src).sum(-1) + (z[dst] * self.att_dst).sum(-1)
        logits = torch.nn.functional.leaky_relu(logits, self.negative_slope)
        # 对每个目标节点的入边做 softmax。
        alpha = _segment_softmax(logits, dst, n)
        msg = z[src] * alpha[:, :, None]
        out = torch.zeros(n, self.heads, self.out_per_head,
                          dtype=msg.dtype, device=msg.device)
        out = out.index_add(0, dst, msg)
        out = out.reshape(n, self.out_dim) if self.concat else out.mean(1)
        return out + self.bias


def _segment_softmax(logits, index, n):
    """按目标节点 index[e] 分组，对 logits[e] 做 softmax。

    显式处理无边情况，保证无边就无消息。
    """
    if logits.numel() == 0:
        return logits
    m = logits.max(dim=0, keepdim=True).values
    e = torch.exp(logits - m)
    denom = torch.zeros(n, logits.shape[1], dtype=e.dtype, device=e.device)
    denom = denom.index_add(0, index, e).clamp_min(1e-30)
    return e / denom[index]


class GAT2(nn.Module):
    """GAT-2L，宽度 256 → 128 → 128。拼接 4 个 32 维注意力头，add_self_loops=False。"""

    def __init__(self, input_dim: int = INPUT_DIM):
        super().__init__()
        self.add_self_loops = False
        self.heads = 4
        self.per_head = 32
        self.concat = True
        self.layer1 = GATLayer(input_dim, 32, 4, True)
        self.layer2 = GATLayer(HIDDEN, 32, 4, True)
        self.dropout = nn.Dropout(DROPOUT)

    def forward(self, x, edges, weights=None):
        edge_index, _ = coalesce_original(edges, weights, len(x))
        # add_self_loops 为 False，此处保持原有边集合。
        h = torch.relu(self.layer1(x, edge_index))
        h = self.dropout(h)
        return self.layer2(h, edge_index)


# GraphSAGE
class SAGELayer(nn.Module):
    """GraphSAGE 均值聚合器。

    节点自身特征通过 lin_self 变换，邻居特征均值通过 lin_neigh 变换。自身项是 GraphSAGE 的独立变换。
    """

    def __init__(self, n_in, n_out, aggregator="mean"):
        super().__init__()
        if aggregator != "mean":
            raise ValueError("only the mean aggregator is frozen for this arm")
        self.aggregator = aggregator
        self.root_weight = True
        self.lin_self = nn.Linear(n_in, n_out, bias=False)
        self.lin_neigh = nn.Linear(n_in, n_out, bias=False)
        self.bias = nn.Parameter(torch.zeros(n_out))
        nn.init.xavier_uniform_(self.lin_self.weight)
        nn.init.xavier_uniform_(self.lin_neigh.weight)

    def forward(self, x, edge_index, edge_weight=None):
        n = len(x)
        src, dst = edge_index[0], edge_index[1]
        deg = torch.zeros(n, dtype=x.dtype, device=x.device)
        deg = deg.index_add(0, dst, torch.ones(len(dst), dtype=x.dtype, device=x.device))
        acc = torch.zeros(n, x.shape[1], dtype=x.dtype, device=x.device)
        acc = acc.index_add(0, dst, x[src])
        neigh = acc / deg.clamp_min(1.0)[:, None]
        return self.lin_self(x) + self.lin_neigh(neigh) + self.bias


class GraphSAGE2(nn.Module):
    """GraphSAGE-2L，宽度 256 → 128 → 128，使用均值聚合。"""

    def __init__(self, input_dim: int = INPUT_DIM):
        super().__init__()
        self.layer1 = SAGELayer(input_dim, HIDDEN)
        self.layer2 = SAGELayer(HIDDEN, HIDDEN)
        self.dropout = nn.Dropout(DROPOUT)

    def forward(self, x, edges, weights=None):
        edge_index, _ = coalesce_original(edges, weights, len(x))
        h = torch.relu(self.layer1(x, edge_index))
        h = self.dropout(h)
        return self.layer2(h, edge_index)


# 实验组登记。GE02R 是图 5(e) 和表 S12 的同环境匹配重放。
ARMS = {
    "GE01": ("GCN-1L-ORIG", gcn1_orig),
    "GE02R": ("GCN-2L-ORIG", gcn2_orig),
    "GE03": ("GCN-3L-ORIG", gcn3_orig),
    "GE04": ("GCN-4L-ORIG", gcn4_orig),
    "GE05": ("GAT-2L", GAT2),
    "GE06": ("GraphSAGE-2L", GraphSAGE2),
}


# 上游预拟合预测器
class GraphVariantPredictor(nn.Module):
    """第一阶段预拟合包装器。

    沿用 mgtcuei.graph.GraphPredictor(route="gcn")。将 512 维配对特征 [h_d ; h_s ; |h_d-h_s| ; h_d*h_s] 输入 Linear(512,256) → ReLU → Linear(256,2)。预测头与文本提示无关，因此 T1 和 T2 可共享上游 GCN。

    编码器子模块命名为 gcn。检查点保存 gcn.*，供 checkpoint.load_component(path, encoder, "gcn.") 加载初始参数。
    """

    def __init__(self, encoder, n_drugs=N_DRUGS):
        super().__init__()
        self.gcn = encoder
        self.n_drugs = n_drugs
        self.head = nn.Sequential(nn.Linear(512, 256), nn.ReLU(), nn.Linear(256, 2))

    def forward(self, x, e, w, d, s):
        z = self.gcn(x, e, w)
        d_rep, s_rep = z[d], z[self.n_drugs + s]
        v = torch.cat((d_rep, s_rep, (d_rep - s_rep).abs(), d_rep * s_rep), -1)
        return self.head(v)

