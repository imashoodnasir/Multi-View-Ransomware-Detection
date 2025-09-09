
"""
models/joint_model.py
---------------------
Full model wiring: Sequence -> Transformer -> Graph -> Cross-View Fusion -> Dual Heads.
"""
from __future__ import annotations
import torch
import torch.nn as nn

from .sequence_encoder import SequenceEncoder
from .transformer_encoder import TransformerRefiner
from .graph_encoder import GraphEncoder
from .cross_view_fusion import CrossViewFusion
from .dual_heads import ClassificationHead, HazardHead


class JointMVTransformerGNN(nn.Module):
    def __init__(self,
                 seq_input_dim: int,
                 rnn_hidden: int,
                 rnn_layers: int,
                 transformer_hidden: int,
                 transformer_heads: int,
                 transformer_layers: int,
                 graph_in_dim: int,
                 graph_hidden: int,
                 graph_layers: int,
                 fusion_dim: int,
                 num_classes: int,
                 dropout: float = 0.1):
        super().__init__()

        self.seq_enc = SequenceEncoder(seq_input_dim, rnn_hidden, num_layers=rnn_layers, dropout=dropout, bidirectional=True)
        self.tr_ref = TransformerRefiner(d_model=self.seq_enc.out_dim, nhead=transformer_heads, num_layers=transformer_layers, dropout=dropout)
        self.graph_enc = GraphEncoder(in_dim=graph_in_dim, hidden_dim=graph_hidden, num_layers=graph_layers, dropout=dropout)

        self.fusion = CrossViewFusion(seq_dim=self.seq_enc.out_dim, graph_dim=self.graph_enc.out_dim, proj_dim=fusion_dim, dropout=dropout)
        self.classifier = ClassificationHead(in_dim=self.fusion.out_dim, num_classes=num_classes, dropout=dropout)
        self.hazard = HazardHead(in_dim=self.fusion.out_dim, dropout=dropout)

    def forward(self, seq: torch.Tensor, mask: torch.Tensor | None, graphs: list | None):
        """
        seq: [B, L, F], mask: [B, L] (1 = valid, 0 = pad) or None
        graphs: list of graph dicts (length B) or None
        """
        key_padding_mask = None
        if mask is not None:
            key_padding_mask = (mask == 0)  # True at PAD

        h_seq = self.seq_enc(seq, mask)                    # [B, L, H]
        h_tr = self.tr_ref(h_seq, src_key_padding_mask=key_padding_mask)  # [B, L, H]
        h_graph = self.graph_enc(graphs)                   # [B, Dg]
        fused = self.fusion(h_tr, h_graph, key_padding_mask=key_padding_mask)  # [B, D]

        logits = self.classifier(fused)                    # [B, C]
        hazard = self.hazard(fused)                        # [B]
        return logits, hazard
