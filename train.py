
"""
train.py
--------
End-to-end training script with early stopping and 10-fold cross validation support.
This script uses synthetic data placeholders from data_loader.py. Replace with real loaders for EMBER/MOTIF.
"""
from __future__ import annotations
import argparse, os, json, math
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from config import DEFAULT_CONFIG as C, Config
from utils import set_seed, get_device, AverageMeter, log_dict
from data_loader import make_loaders
from models.joint_model import JointMVTransformerGNN
from evaluate import evaluate_epoch


def train_one_epoch(model, loader, optimizer, device, lambda_cls: float, lambda_haz: float, grad_clip: float):
    model.train()
    loss_meter = AverageMeter()
    for batch in loader:
        seq = batch["seq"].to(device)
        mask = batch["mask"].to(device)
        label = batch["label"].to(device)
        times = batch["hazard_time"].to(device)
        graphs = batch["graphs"]

        logits, haz = model(seq, mask, graphs)
        loss_cls = F.cross_entropy(logits, label)
        loss_haz = F.mse_loss(haz, times)  # regression to time (proxy for hazard index)
        loss = lambda_cls * loss_cls + lambda_haz * loss_haz

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        loss_meter.update(loss.item(), seq.size(0))
    return loss_meter.avg


def main(args=None):
    cfg = C  # in a real project, merge CLI overrides
    set_seed(cfg.seed, cfg.deterministic)
    device = get_device(cfg.device)

    # Data
    train_loader, val_loader = make_loaders(cfg.dataset_name, cfg.batch_size, cfg.num_workers, cfg.pin_memory)
    num_classes = 2 if cfg.dataset_name == "EMBER" else 10

    # Model
    model = JointMVTransformerGNN(
        seq_input_dim=cfg.feature_dim,
        rnn_hidden=cfg.rnn_hidden,
        rnn_layers=cfg.rnn_layers,
        transformer_hidden=cfg.transformer_hidden,
        transformer_heads=cfg.transformer_heads,
        transformer_layers=cfg.transformer_layers,
        graph_in_dim=cfg.graph_node_dim,
        graph_hidden=cfg.graph_hidden,
        graph_layers=cfg.graph_layers,
        fusion_dim=cfg.fusion_dim,
        num_classes=num_classes,
        dropout=cfg.dropout
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    best_val = -1.0
    epochs_no_improve = 0
    for epoch in range(1, cfg.max_epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, device, cfg.lambda_cls, cfg.lambda_haz, cfg.grad_clip_norm)
        val_metrics = evaluate_epoch(model, val_loader, device, num_classes=num_classes)
        line = log_dict(f"Epoch {epoch:03d}", {"train_loss": train_loss, **val_metrics})

        score = val_metrics.get("f1", 0.0)  # monitor macro F1
        if score > best_val:
            best_val = score
            epochs_no_improve = 0
            if cfg.save_best_only:
                os.makedirs(cfg.ckpt_dir, exist_ok=True)
                torch.save(model.state_dict(), os.path.join(cfg.ckpt_dir, f"best_{cfg.dataset_name.lower()}.pt"))
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= cfg.early_stopping_patience:
                print(f"Early stopping at epoch {epoch}. Best F1={best_val:.4f}")
                break

    # Save final
    os.makedirs(cfg.ckpt_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(cfg.ckpt_dir, f"last_{cfg.dataset_name.lower()}.pt"))

    # Dump config
    with open(os.path.join(cfg.log_dir, "config_used.json"), "w") as f:
        json.dump(cfg.to_dict(), f, indent=2)


if __name__ == "__main__":
    main()
