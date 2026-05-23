# -*- coding: utf-8 -*-
# D:\pythonpro\SG-MTF-main\models\clinical_graph.py

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.data import Data, Batch
from torch_geometric.nn import GCNConv, global_mean_pool


class ClinicalVariableGraphEncoder(nn.Module):
    """
    PyG-based prior-guided clinical graph encoder.

    输入:
        c_obs: [B, clinical_dim]

    图构建:
        每个病人构建一个 12 节点临床图；
        每个节点复制该病人的完整 clinical vector；
        节点之间按照医学先验边连接；
        可选加入反向边和自环。

    输出:
        clinical_global: [B, graph_dim]
        clinical_nodes:  [B, 12, graph_dim]
        clinical_node_attn: [B, 12]

    这个版本使用:
        torch_geometric.data.Data
        torch_geometric.data.Batch
        torch_geometric.nn.GCNConv
        torch_geometric.nn.global_mean_pool
    """

    DEFAULT_KMNET_NODE_NAMES = [
        "Age",
        "乳房手术",
        "腋窝手术",
        "病理学类型",
        "size（cm）",
        "diff",
        "乳头或皮肤受累",
        "LVI",
        "ER",
        "PR",
        "Ki-67%",
        "EGFR",
    ]

    DEFAULT_KMNET_CAUSAL_EDGES = [
        ("病理学类型", "Ki-67%"),
        ("size（cm）", "Ki-67%"),
        ("diff", "Ki-67%"),
        ("Ki-67%", "LVI"),
        ("LVI", "乳房手术"),
        ("LVI", "腋窝手术"),
        ("乳头或皮肤受累", "乳房手术"),
        ("乳头或皮肤受累", "腋窝手术"),
        ("size（cm）", "乳房手术"),
        ("size（cm）", "腋窝手术"),
    ]

    def __init__(
        self,
        clinical_dim: int,
        numeric_slice=None,
        onehot_slices_dict=None,
        graph_dim: int = 256,
        hidden_dim: int = 128,
        dropout: float = 0.1,
        use_value_norm: bool = True,

        # Compatibility arguments, kept for old sgmtf.py / run_cv.py.
        enable_clinical_self_attention: bool = False,
        num_heads: int = 4,
        num_layers: int = 1,

        # Prior graph options.
        numeric_feature_names=None,
        use_causal_graph: bool = True,
        causal_edges_prior=None,
        add_reverse_edges: bool = True,
        add_self_loops: bool = True,
        causal_graph_layers: int = 3,

        # Pooling option.
        pooling: str = "mean",
    ):
        super().__init__()

        if clinical_dim <= 0:
            raise ValueError("clinical_dim must be > 0")

        if pooling not in ["mean", "attention"]:
            raise ValueError("pooling must be 'mean' or 'attention'")

        self.clinical_dim = int(clinical_dim)
        self.graph_dim = int(graph_dim)
        self.hidden_dim = int(hidden_dim)
        self.dropout_p = float(dropout)
        self.use_value_norm = bool(use_value_norm)

        # Compatibility flags.
        self.enable_clinical_self_attention = False
        self.num_heads = num_heads
        self.num_layers = num_layers

        self.use_causal_graph = bool(use_causal_graph)
        self.add_reverse_edges = bool(add_reverse_edges)
        self.add_self_loops = bool(add_self_loops)
        self.causal_graph_layers = int(causal_graph_layers)
        self.pooling = pooling

        self.node_names = list(self.DEFAULT_KMNET_NODE_NAMES)
        self.num_nodes = len(self.node_names)

        if causal_edges_prior is None:
            causal_edges_prior = list(self.DEFAULT_KMNET_CAUSAL_EDGES)

        self.causal_edges_prior = [(str(a), str(b)) for a, b in causal_edges_prior]

        # ---------------------------------------------------------
        # Build fixed prior edge_index.
        # ---------------------------------------------------------
        edge_index = self._build_edge_index()
        self.register_buffer("edge_index_base", edge_index, persistent=False)

        # Dense adjacency only for debug / visualization.
        dense_adj = self._edge_index_to_dense_adj(edge_index, self.num_nodes)
        self.register_buffer("causal_adj", dense_adj, persistent=False)

        # ---------------------------------------------------------
        # PyG GCN layers.
        # ---------------------------------------------------------
        # 与你 clinical graph only 训练脚本保持一致:
        # c_obs [B, D] -> 每个节点复制完整 D 维特征
        # 所以第一层 GCN 输入维度就是 clinical_dim
        self.gcn1 = GCNConv(self.clinical_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)

        if self.causal_graph_layers <= 1:
            self.gcn2 = None
            self.bn2 = None
            self.gcn3 = None
            self.bn3 = None
            final_dim = hidden_dim
        elif self.causal_graph_layers == 2:
            self.gcn2 = GCNConv(hidden_dim, graph_dim)
            self.bn2 = nn.BatchNorm1d(graph_dim)
            self.gcn3 = None
            self.bn3 = None
            final_dim = graph_dim
        else:
            self.gcn2 = GCNConv(hidden_dim, hidden_dim)
            self.bn2 = nn.BatchNorm1d(hidden_dim)
            self.gcn3 = GCNConv(hidden_dim, graph_dim)
            self.bn3 = nn.BatchNorm1d(graph_dim)
            final_dim = graph_dim

        # 如果只设 1 层，则需要投影到 graph_dim，保证输出维度稳定
        if final_dim != graph_dim:
            self.out_proj = nn.Linear(final_dim, graph_dim)
        else:
            self.out_proj = nn.Identity()

        self.dropout = nn.Dropout(dropout)
        self.out_norm = nn.LayerNorm(graph_dim)

        # 可选 attention pooling
        self.readout_score = nn.Sequential(
            nn.LayerNorm(graph_dim),
            nn.Linear(graph_dim, 1),
        )

        self._init_weights()

    # ---------------------------------------------------------
    # Initialization
    # ---------------------------------------------------------
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    # ---------------------------------------------------------
    # Name normalization
    # ---------------------------------------------------------
    @staticmethod
    def _canon_name(x: str) -> str:
        s = str(x).strip()
        s = s.replace("（", "(").replace("）", ")")
        s = s.replace("％", "%")
        s = s.replace(" ", "")
        s = s.lower()
        return s

    # ---------------------------------------------------------
    # Build PyG edge_index
    # ---------------------------------------------------------
    def _build_edge_index(self) -> torch.Tensor:
        name2idx = {
            self._canon_name(name): idx
            for idx, name in enumerate(self.node_names)
        }

        edges = []
        matched_edges = []
        skipped_edges = []

        if self.use_causal_graph:
            for src, tgt in self.causal_edges_prior:
                src_key = self._canon_name(src)
                tgt_key = self._canon_name(tgt)

                if src_key not in name2idx or tgt_key not in name2idx:
                    skipped_edges.append((src, tgt))
                    continue

                s = name2idx[src_key]
                t = name2idx[tgt_key]

                # PyG edge_index convention:
                # edge_index[0] = source
                # edge_index[1] = target
                edges.append((s, t))
                matched_edges.append((src, tgt))

                if self.add_reverse_edges:
                    edges.append((t, s))
        else:
            # 不使用先验图时，退化为只有自环图
            matched_edges = []

        if self.add_self_loops:
            for i in range(self.num_nodes):
                edges.append((i, i))

        if len(edges) == 0:
            raise ValueError("No graph edges were constructed.")

        self.matched_causal_edges = matched_edges
        self.skipped_causal_edges = skipped_edges

        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
        return edge_index

    @staticmethod
    def _edge_index_to_dense_adj(edge_index: torch.Tensor, num_nodes: int) -> torch.Tensor:
        adj = torch.zeros(num_nodes, num_nodes, dtype=torch.float32)
        src = edge_index[0]
        tgt = edge_index[1]
        adj[tgt, src] = 1.0
        return adj

    # ---------------------------------------------------------
    # Build batched PyG graph
    # ---------------------------------------------------------
    def _build_pyg_batch(self, c_obs: torch.Tensor) -> Batch:
        """
        Args:
            c_obs: [B, clinical_dim]

        Returns:
            PyG Batch
                batch.x: [B * 12, clinical_dim]
                batch.edge_index: [2, B * E]
                batch.batch: [B * 12]
        """
        if c_obs.dim() != 2:
            raise ValueError("c_obs must be [B, D].")

        if c_obs.size(1) != self.clinical_dim:
            raise ValueError(
                f"Expected clinical_dim={self.clinical_dim}, got {c_obs.size(1)}."
            )

        B = c_obs.size(0)
        data_list = []

        edge_index = self.edge_index_base.to(c_obs.device)

        for i in range(B):
            # 每个病人 12 个临床节点，每个节点复制完整临床向量
            node_x = c_obs[i].unsqueeze(0).repeat(self.num_nodes, 1)

            data = Data(
                x=node_x,
                edge_index=edge_index,
            )
            data_list.append(data)

        batch = Batch.from_data_list(data_list)
        return batch

    # ---------------------------------------------------------
    # GCN encoding
    # ---------------------------------------------------------
    def _encode_graph(self, batch: Batch):
        x = batch.x
        edge_index = batch.edge_index
        batch_idx = batch.batch

        x = self.gcn1(x, edge_index)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.dropout(x)

        if self.gcn2 is not None:
            x = self.gcn2(x, edge_index)
            x = self.bn2(x)
            x = F.relu(x)
            x = self.dropout(x)

        if self.gcn3 is not None:
            x = self.gcn3(x, edge_index)
            x = self.bn3(x)
            x = F.relu(x)

        x = self.out_proj(x)

        # nodes: [B, 12, graph_dim]
        B = int(batch_idx.max().item()) + 1
        nodes = x.view(B, self.num_nodes, self.graph_dim)

        return nodes, batch_idx

    # ---------------------------------------------------------
    # Pooling
    # ---------------------------------------------------------
    def _pool_nodes(self, nodes: torch.Tensor):
        """
        Args:
            nodes: [B, N, graph_dim]

        Returns:
            clinical_global: [B, graph_dim]
            attn: [B, N]
        """
        B, N, _ = nodes.shape

        if self.pooling == "mean":
            attn = torch.full(
                (B, N),
                fill_value=1.0 / float(N),
                device=nodes.device,
                dtype=nodes.dtype,
            )
            clinical_global = nodes.mean(dim=1)
        else:
            logits = self.readout_score(nodes).squeeze(-1)
            attn = torch.softmax(logits, dim=1)
            clinical_global = torch.sum(nodes * attn.unsqueeze(-1), dim=1)

        clinical_global = self.out_norm(clinical_global)

        return clinical_global, attn

    # ---------------------------------------------------------
    # Forward
    # ---------------------------------------------------------
    def forward(self, c_obs, m=None):
        """
        Args:
            c_obs: [B, clinical_dim]
            m: ignored, kept for compatibility.

        Returns:
            dict:
                clinical_nodes: [B, 12, graph_dim]
                clinical_node_mask: [B, 12]
                clinical_global: [B, graph_dim]
                clinical_node_attn: [B, 12]
        """
        batch = self._build_pyg_batch(c_obs)
        batch = batch.to(c_obs.device)

        nodes, _ = self._encode_graph(batch)

        clinical_global, attn = self._pool_nodes(nodes)

        node_mask = torch.ones(
            c_obs.size(0),
            self.num_nodes,
            device=c_obs.device,
            dtype=c_obs.dtype,
        )

        return {
            "clinical_nodes": nodes,
            "clinical_node_mask": node_mask,
            "clinical_global": clinical_global,
            "clinical_node_attn": attn,

            # Debug / visualization.
            "clinical_node_names": list(self.node_names),
            "clinical_node_slices": None,
            "clinical_causal_adj": self.causal_adj.detach().cpu(),
            "clinical_matched_causal_edges": list(getattr(self, "matched_causal_edges", [])),
            "clinical_skipped_causal_edges": list(getattr(self, "skipped_causal_edges", [])),
            "clinical_edge_index": self.edge_index_base.detach().cpu(),
        }