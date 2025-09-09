
"""
config.py
---------
Centralized hyperparameters and experiment settings.
Modify these values (or override via CLI args in train.py) to reproduce experiments.
"""

from dataclasses import dataclass, asdict
from typing import Optional, List


@dataclass
class Config:
    # Reproducibility
    seed: int = 42
    deterministic: bool = True

    # Data
    dataset_name: str = "EMBER"            # "EMBER" or "MOTIF"
    data_root: str = "./data"
    use_cached_graphs: bool = True
    max_seq_len: int = 512                 # truncate/pad event sequences
    feature_dim: int = 256                 # per-timestep feature dimension (sequence side)
    graph_node_dim: int = 128              # per-node feature dimension (graph side)
    num_classes: int = 2                   # will be overridden for MOTIF
    use_class_weights: bool = False

    # Model (encoders)
    rnn_hidden: int = 256                  # GRU hidden size
    rnn_layers: int = 1
    transformer_hidden: int = 256
    transformer_heads: int = 4
    transformer_layers: int = 2
    graph_hidden: int = 128
    graph_layers: int = 2
    fusion_dim: int = 256                  # common projection dim for cross-attention
    dropout: float = 0.1

    # Training
    batch_size: int = 128
    lr: float = 1e-4
    weight_decay: float = 1e-5
    max_epochs: int = 100
    early_stopping_patience: int = 10
    grad_clip_norm: float = 1.0

    # Multi-task loss weights
    lambda_cls: float = 1.0
    lambda_haz: float = 1.0

    # Cross-validation
    k_folds: int = 10
    shuffle_cv: bool = True

    # Logging / checkpoints
    log_dir: str = "./logs"
    ckpt_dir: str = "./checkpoints"
    save_best_only: bool = True

    # Misc
    device: str = "cuda"                   # "cuda" or "cpu"
    num_workers: int = 4                   # DataLoader workers
    pin_memory: bool = True

    def to_dict(self):
        return asdict(self)


DEFAULT_CONFIG = Config()
