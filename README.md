
# Multi-View Ransomware Detection (Transformer + GNN + Cross-View Fusion)

This repository contains an end-to-end **PyTorch** implementation of the paper’s framework:
**Early and Robust Ransomware Detection via Multi-View Sequential Modeling with Transformers and Graph Fusion**.

The pipeline integrates:
- **Sequence Encoder** (GRU) for temporal patterns
- **Transformer Refiner** for long-range context
- **Graph Encoder** (GCN via *torch-geometric*; graceful fallback to MLP) for structural signals
- **Cross-View Fusion** (bi-directional multihead cross-attention)
- **Dual Heads**: classification (softmax) and hazard prediction (regression)

> ⚠️ The current `data_loader.py` uses **synthetic** samples to make the project runnable out-of-the-box. Replace with actual EMBER/MOTIF parsing to reproduce paper-level results.



---

## Features
- **Multi-view learning**: temporal + structural information fused via cross-attention
- **Robust modeling**: dual-task objective encourages discriminative and risk-sensitive embeddings
- **Metric-rich evaluation**: Accuracy, Precision, Recall, F1, ROC-AUC, MCC; plus hazard metrics (C-index, Brier, calibration error)
- **Reproducible**: unified `config.py`; early stopping, grad clipping, AdamW

---

## Project Structure
```
ransomware_mv_transformer_gnn/
├── config.py
├── utils.py
├── metrics.py
├── stats_tests.py
├── data_loader.py
├── graph_builder.py
├── evaluate.py
├── train.py
├── models/
│   ├── sequence_encoder.py
│   ├── transformer_encoder.py
│   ├── graph_encoder.py
│   ├── cross_view_fusion.py
│   └── dual_heads.py
│   └── joint_model.py
```
> You only need to modify **`data_loader.py`** (and optionally `graph_builder.py`) to switch from synthetic to real datasets.

---

## Quick Start

1. **Create environment & install dependencies**
   ```bash
   # (Recommended) Python 3.10+
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
   pip install scikit-learn scipy numpy pandas
   pip install torch-geometric torch-scatter torch-sparse torch-cluster torch-spline-conv -f https://data.pyg.org/whl/torch-2.3.0+cu121.html
   ```
   > If you cannot install *torch-geometric*, the code will automatically **fallback** to an MLP-based graph encoder.

2. **Run a sanity training (synthetic data)**
   ```bash
   python train.py
   ```
   Checkpoints are saved to `./checkpoints/` and config snapshot to `./logs/config_used.json`.

---

## Requirements
- Python ≥ 3.9
- PyTorch ≥ 2.1 (CUDA recommended)
- (Optional) torch-geometric ≥ 2.4 for true GCN/GAT encoders
- SciPy / scikit-learn for metrics and statistical tests

---

## Configuration
All important knobs are centralized in **`config.py`**. Highlights:
- `dataset_name` = `"EMBER"` or `"MOTIF"`
- `max_seq_len`, `feature_dim` (sequence side)
- `graph_node_dim` (graph side)
- Encoder sizes: `rnn_hidden`, `transformer_heads/layers`, `graph_hidden/layers`
- Fusion projection: `fusion_dim`
- Training: `batch_size`, `lr`, `max_epochs`, `early_stopping_patience`, `grad_clip_norm`
- Multi-task weights: `lambda_cls`, `lambda_haz`

Use `config.py` directly or extend `train.py` to parse CLI overrides.

---

## Training
Default run (synthetic):
```bash
python train.py
```
What happens:
- Builds **JointMVTransformerGNN** (GRU → Transformer → Graph Encoder → Cross-View Fusion → Dual Heads)
- Trains with **joint loss**: `L = λ_cls * CrossEntropy + λ_haz * MSE`
- Early-stops on **macro-F1** from validation
- Logs and saves best/last checkpoints

---

## Evaluation
Validation is executed every epoch via **`evaluate_epoch`** in `evaluate.py`:
- **Classification**: Accuracy, Precision, Recall, F1 (macro), ROC-AUC, MCC
- **Hazard**: C-index, Brier score, calibration error (simplified proxies for survival analysis)

> Replace hazard targets with real labels if your pipeline produces time-to-event annotations.

---

## Datasets (Replace Synthetic)
The paper uses **EMBER** (binary malware vs. benign) and **MOTIF** (multi-class malware families).
- Implement dataset-specific parsing in `data_loader.py`:
  - Produce `seq` tensor `[T, feature_dim]` per sample (pad/trim to `max_seq_len`).
  - Produce `label` as `int` class index.
  - (Optional) Add `hazard_time` if you have time-to-event labels.
  - Load or construct a graph via `graph_builder.py` and put it into the batch element.

**Graph building** (`graph_builder.py`):
- Map binaries/logs into a rooted tree or general graph:
  - **Nodes**: sections, imported APIs, functions, opcode groups, etc.
  - **Edges**: call relations, containment, dependency
- Cache to `./data/graph_cache/` to avoid recomputation.

---

## Ablations & Statistics
- Swap fusion strategy by editing `models/cross_view_fusion.py` (e.g., concat/sum vs. cross-attention).
- Compare single-view baselines: use only `sequence_encoder + transformer` **or** only `graph_encoder`.
- Statistical tests (`stats_tests.py`):
  - **McNemar** for paired decisions
  - **Paired t-test** & **Wilcoxon** for fold-wise scores

---

