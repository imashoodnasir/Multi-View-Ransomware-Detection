
"""
models/graph_encoder.py
-----------------------
Graph encoder using torch_geometric if available; otherwise falls back to a simple MLP aggregator.
"""
from __future__ import annotations
import torch
import torch.nn as nn

HAVE_TORCH_GEOMETRIC = True
try:
    from torch_geometric.nn import GCNConv, global_mean_pool
    from torch_geometric.data import Batch, Data
except Exception:
    HAVE_TORCH_GEOMETRIC = False


class GraphEncoder(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int = 128, num_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        self.use_geometric = HAVE_TORCH_GEOMETRIC
        self.dropout = nn.Dropout(dropout)
        if self.use_geometric:
            self.convs = nn.ModuleList()
            dims = [in_dim] + [hidden_dim] * num_layers
            for a, b in zip(dims[:-1], dims[1:]):
                self.convs.append(GCNConv(a, b))
            self.out_dim = hidden_dim
        else:
            # Fallback: treat graph as a bag of nodes -> MLP + mean
            self.mlp = nn.Sequential(
                nn.Linear(in_dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, hidden_dim),
                nn.GELU(),
            )
            self.out_dim = hidden_dim

    def forward(self, graphs: list | None) -> torch.Tensor:
        """
        graphs: list of dicts or None (length B)
        Returns node-level or graph-level embeddings [B, Dg]
        """
        B = len(graphs)
        if graphs is None or all(g is None for g in graphs):
            # no graphs provided -> return zeros
            return torch.zeros((B, self.out_dim), dtype=torch.float32, device=next(self.parameters()).device)

        if self.use_geometric:
            data_list = []
            for g in graphs:
                x = torch.tensor(g["x"], dtype=torch.float32)
                edge_index = torch.tensor(g["edge_index"], dtype=torch.long)
                data_list.append(Data(x=x, edge_index=edge_index))
            batch = Batch.from_data_list(data_list).to(next(self.parameters()).device)
            h = batch.x
            for conv in self.convs:
                h = conv(h, batch.edge_index)
                h = torch.relu(h)
                h = self.dropout(h)
            pooled = global_mean_pool(h, batch.batch)  # [B, D]
            return pooled
        else:
            # Fallback: compute MLP on nodes then average per-graph
            outs = []
            device = next(self.parameters()).device
            for g in graphs:
                x = torch.tensor(g["x"], dtype=torch.float32, device=device)  # [N, F]
                node_emb = self.mlp(x)                                       # [N, D]
                pooled = node_emb.mean(dim=0, keepdim=True)                  # [1, D]
                outs.append(pooled)
            return torch.cat(outs, dim=0)                                    # [B, D]
