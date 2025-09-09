
"""
models/sequence_encoder.py
--------------------------
Sequence encoder built with GRU (can be swapped for LSTM).
"""
from __future__ import annotations
import torch
import torch.nn as nn


class SequenceEncoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int = 1, dropout: float = 0.1, bidirectional: bool = True):
        super().__init__()
        self.rnn = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional
        )
        self.out_dim = hidden_dim * (2 if bidirectional else 1)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        """
        x: [B, L, F], mask: [B, L] where 1 = valid, 0 = pad (optional)
        Returns token-level embeddings: [B, L, H*D]
        """
        out, _ = self.rnn(x)  # [B, L, H*D]
        return out
