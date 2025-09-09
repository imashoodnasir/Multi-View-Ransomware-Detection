
"""
evaluate.py
-----------
Validation / testing loops for classification & hazard metrics.
"""
from __future__ import annotations
from typing import Dict, Any
import torch
import numpy as np

from metrics import classification_metrics, concordance_index, brier_score, mean_absolute_calibration_error


@torch.no_grad()
def evaluate_epoch(model, loader, device, num_classes: int) -> Dict[str, Any]:
    model.eval()
    all_logits, all_labels = [], []
    all_haz, all_times = [], []

    for batch in loader:
        seq = batch["seq"].to(device)
        mask = batch["mask"].to(device)
        label = batch["label"].to(device)
        times = batch["hazard_time"].to(device)
        graphs = batch["graphs"]  # handled inside model

        logits, haz = model(seq, mask, graphs)
        all_logits.append(logits.cpu())
        all_labels.append(label.cpu())
        all_haz.append(haz.cpu())
        all_times.append(times.cpu())

    logits = torch.cat(all_logits, dim=0)
    labels = torch.cat(all_labels, dim=0)
    haz = torch.cat(all_haz, dim=0).numpy()
    times = torch.cat(all_times, dim=0).numpy()

    probs = torch.softmax(logits, dim=-1).numpy()
    preds = logits.argmax(dim=-1).numpy()
    cls_metrics = classification_metrics(labels.numpy(), preds, probs, average="macro", num_classes=num_classes)

    # Hazard metrics (using simplified placeholders, assumes all events observed)
    c_idx = concordance_index(times, haz)
    # Use normalized times as "true prob" and normalized haz as "pred prob" for a coarse Brier proxy
    bs = brier_score((times - times.min()) / (times.max() - times.min() + 1e-8),
                     (haz - haz.min()) / (haz.max() - haz.min() + 1e-8))
    cal = mean_absolute_calibration_error(times, haz)

    out = {**cls_metrics, "c_index": c_idx, "brier": bs, "cal_error": cal}
    return out
