
"""
models/dual_heads.py
--------------------
Classification and Hazard prediction heads (+ joint loss wrapper).
"""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F


class ClassificationHead(nn.Module):
    def __init__(self, in_dim: int, num_classes: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, in_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(in_dim, num_classes),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)  # logits


class HazardHead(nn.Module):
    def __init__(self, in_dim: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, in_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(in_dim, 1),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z).squeeze(-1)  # scalar regression
