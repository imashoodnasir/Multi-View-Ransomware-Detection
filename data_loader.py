
"""
data_loader.py
--------------
Dataset classes and collate functions for EMBER and MOTIF.
Includes:
- Sequence tensor preparation (pad/truncate to max_seq_len)
- Optional graph construction placeholders (loaded from graph_builder cache)
- K-Fold split helpers
NOTE: This is a template – plug in actual EMBER/MOTIF parsing.
"""
from __future__ import annotations
import os, json, math, random
from typing import List, Tuple, Dict, Any, Optional
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Subset, random_split
from dataclasses import dataclass

from config import DEFAULT_CONFIG as C

GraphType = Dict[str, Any]  # {"x": node_features[T_nodes, F], "edge_index": [2, E]}


@dataclass
class Sample:
    seq: np.ndarray           # [T, F] sequential features
    label: int                # class index
    graph_path: Optional[str] = None
    hazard_time: Optional[float] = None   # synthetic or provided


class MalwareDataset(Dataset):
    def __init__(self, split: str = "train", dataset_name: str = "EMBER", size:int=2048):
        self.split = split
        self.dataset_name = dataset_name
        self.size = size
        self.max_seq_len = C.max_seq_len
        self.feature_dim = C.feature_dim
        self.graph_node_dim = C.graph_node_dim
        self.num_classes = 2 if dataset_name == "EMBER" else 10  # example: 10 families

        # In real code: load actual data. Here we synthesize for template completeness.
        random.seed(123)
        np.random.seed(123)
        self.samples: List[Sample] = []
        for i in range(size):
            T = random.randint(100, self.max_seq_len)
            seq = np.random.randn(T, self.feature_dim).astype(np.float32)
            label = np.random.randint(0, self.num_classes)
            # emulate graph cache path (would be created by graph_builder.py)
            graph_path = None
            hazard_time = float(np.clip(np.random.gamma(shape=2.0, scale=10.0), 0.1, 100.0))
            self.samples.append(Sample(seq=seq, label=label, graph_path=graph_path, hazard_time=hazard_time))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        s = self.samples[idx]
        seq = s.seq
        T = seq.shape[0]
        # pad/truncate
        if T >= self.max_seq_len:
            seq = seq[: self.max_seq_len]
            mask = np.ones(self.max_seq_len, dtype=np.float32)
        else:
            pad = np.zeros((self.max_seq_len - T, self.feature_dim), dtype=np.float32)
            seq = np.concatenate([seq, pad], axis=0)
            mask = np.zeros(self.max_seq_len, dtype=np.float32)
            mask[:T] = 1.0
        out = {
            "seq": torch.from_numpy(seq),            # [L, F]
            "mask": torch.from_numpy(mask),          # [L]
            "label": torch.tensor(s.label, dtype=torch.long),
            "hazard_time": torch.tensor(s.hazard_time, dtype=torch.float32),
            "graph": None,                           # will be loaded by collate if graph available
        }
        return out


def collate_batch(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    seq = torch.stack([b["seq"] for b in batch], dim=0)          # [B, L, F]
    mask = torch.stack([b["mask"] for b in batch], dim=0)        # [B, L]
    label = torch.stack([b["label"] for b in batch], dim=0)      # [B]
    hazard_time = torch.stack([b["hazard_time"] for b in batch], dim=0)  # [B]

    # Graphs: placeholder None -> downstream handles missing graphs
    graphs = [b.get("graph", None) for b in batch]
    return {"seq": seq, "mask": mask, "label": label, "hazard_time": hazard_time, "graphs": graphs}


def make_loaders(dataset_name: str, batch_size: int, num_workers: int = 0, pin_memory: bool = False) -> Tuple[DataLoader, DataLoader]:
    ds = MalwareDataset(split="train", dataset_name=dataset_name, size=4096)
    n_val = int(0.1 * len(ds))
    n_train = len(ds) - n_val
    train_ds, val_ds = random_split(ds, [n_train, n_val], generator=torch.Generator().manual_seed(123))

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers,
                              pin_memory=pin_memory, collate_fn=collate_batch, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers,
                            pin_memory=pin_memory, collate_fn=collate_batch, drop_last=False)
    return train_loader, val_loader
