
"""
models/cross_view_fusion.py
---------------------------
Bidirectional cross-attention fusion between Transformer (sequence) and Graph embeddings.
"""
from __future__ import annotations
import torch
import torch.nn as nn


class CrossViewFusion(nn.Module):
    def __init__(self, seq_dim: int, graph_dim: int, proj_dim: int = 256, dropout: float = 0.1):
        super().__init__()
        self.seq_proj = nn.Linear(seq_dim, proj_dim)
        self.graph_proj = nn.Linear(graph_dim, proj_dim)
        self.dropout = nn.Dropout(dropout)

        self.attn_seq_to_graph = nn.MultiheadAttention(embed_dim=proj_dim, num_heads=4, dropout=dropout, batch_first=True)
        self.attn_graph_to_seq = nn.MultiheadAttention(embed_dim=proj_dim, num_heads=4, dropout=dropout, batch_first=True)

        self.ff = nn.Sequential(
            nn.Linear(proj_dim * 2, proj_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(proj_dim, proj_dim),
            nn.GELU(),
        )
        self.out_dim = proj_dim

    def forward(self, h_seq_tokens: torch.Tensor, h_graph: torch.Tensor, key_padding_mask: torch.Tensor | None = None) -> torch.Tensor:
        """
        h_seq_tokens: [B, L, Ds] (token-level Transformer features)
        h_graph:      [B, Dg]    (graph-level pooled features)
        key_padding_mask: [B, L] True where PAD tokens (for attention)
        Returns fused sequence-level representation pooled to [B, D]
        """
        B, L, _ = h_seq_tokens.shape

        zT = self.seq_proj(h_seq_tokens)          # [B, L, d]
        zG = self.graph_proj(h_graph)             # [B, d]
        zG_tokens = zG.unsqueeze(1)               # [B, 1, d]

        # Seq attends to Graph (graph as key/value, seq tokens as query)
        seq2graph, _ = self.attn_seq_to_graph(query=zT, key=zG_tokens, value=zG_tokens,
                                              key_padding_mask=None)  # [B, L, d]

        # Graph attends to Seq (seq as key/value, single graph token as query)
        graph2seq, _ = self.attn_graph_to_seq(query=zG_tokens, key=zT, value=zT,
                                              key_padding_mask=key_padding_mask)  # [B, 1, d]

        # Aggregate: concat per-token enrichment with repeated graph2seq token
        graph2seq_repeat = graph2seq.repeat(1, L, 1)  # [B, L, d]
        fused_tokens = torch.cat([seq2graph, graph2seq_repeat], dim=-1)  # [B, L, 2d]
        fused_tokens = self.ff(fused_tokens)  # [B, L, d]

        # Pool to a single vector (attentive mean)
        attn_scores = torch.softmax((fused_tokens.mean(dim=-1, keepdim=True)), dim=1)  # [B, L, 1]
        if key_padding_mask is not None:
            # set PAD positions to zero contribution
            mask = (~key_padding_mask).float().unsqueeze(-1)  # invert: 1 for valid
            attn_scores = attn_scores * mask
            attn_scores = attn_scores / (attn_scores.sum(dim=1, keepdim=True) + 1e-8)
        pooled = (fused_tokens * attn_scores).sum(dim=1)  # [B, d]
        return pooled
