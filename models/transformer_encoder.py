
"""
models/transformer_encoder.py
-----------------------------
Thin wrapper over PyTorch TransformerEncoder to refine sequence features.
"""
from __future__ import annotations
import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 4096):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))  # [1, L, D]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        L = x.size(1)
        return x + self.pe[:, :L]


class TransformerRefiner(nn.Module):
    def __init__(self, d_model: int, nhead: int = 4, num_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        enc_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=d_model*4,
                                               dropout=dropout, batch_first=True, activation="gelu")
        self.encoder = nn.TransformerEncoder(encoder_layer=enc_layer, num_layers=num_layers)
        self.pos = PositionalEncoding(d_model)

    def forward(self, x: torch.Tensor, src_key_padding_mask: torch.Tensor | None = None) -> torch.Tensor:
        """
        x: [B, L, D]
        src_key_padding_mask: [B, L] True for PAD positions
        """
        x = self.pos(x)
        out = self.encoder(x, src_key_padding_mask=src_key_padding_mask)
        return out
