
"""
utils.py
--------
Utility helpers for seeding, device selection, progress logging, and simple profiling.
"""
from __future__ import annotations
import os, random, time
import numpy as np
import torch
from typing import Dict, Any


def set_seed(seed: int, deterministic: bool = True):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_device(preferred: str = "cuda") -> torch.device:
    if preferred == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


class AverageMeter:
    """Keeps running average of a scalar metric."""
    def __init__(self):
        self.reset()

    def reset(self):
        self.sum = 0.0
        self.cnt = 0

    def update(self, val: float, n: int = 1):
        self.sum += float(val) * n
        self.cnt += n

    @property
    def avg(self) -> float:
        return self.sum / max(1, self.cnt)


def log_dict(prefix: str, stats: Dict[str, Any]) -> str:
    kv = " | ".join(f"{k}={v:.4f}" if isinstance(v, (float, int)) else f"{k}={v}" for k, v in stats.items())
    line = f"{prefix}: {kv}"
    print(line)
    return line


class Timer:
    def __init__(self, name: str = "timer"):
        self.name = name
        self._t0 = None

    def __enter__(self):
        self._t0 = time.time()
        return self

    def __exit__(self, exc_type, exc, tb):
        dt = time.time() - self._t0
        print(f"[{self.name}] took {dt:.2f}s")
