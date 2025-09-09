
"""
graph_builder.py
----------------
Utilities to build rooted tree graphs for each sample and cache to disk.
This is a stub with a synthetic generator; replace with real PE parsing for EMBER/MOTIF.
"""
from __future__ import annotations
import os, json, random
from pathlib import Path
from typing import Dict, Any
import numpy as np

CACHE_DIR = Path("./data/graph_cache")
CACHE_DIR.mkdir(parents=True, exist_ok=True)


def build_synthetic_tree(sample_id: str, num_nodes: int = 120, node_dim: int = 128, seed: int = 0) -> Dict[str, Any]:
    rng = np.random.default_rng(seed if seed else None)
    x = rng.normal(size=(num_nodes, node_dim)).astype(np.float32)
    # Make a simple tree edge list (star + chain)
    edges = []
    for i in range(1, num_nodes):
        parent = rng.integers(low=max(0, i-5), high=i)  # connect to a previous node
        edges.append([parent, i])
    edge_index = np.array(edges, dtype=np.int64).T  # [2, E]
    return {"x": x.tolist(), "edge_index": edge_index.tolist()}


def cache_graph(graph: Dict[str, Any], sample_id: str) -> str:
    path = CACHE_DIR / f"{sample_id}.json"
    with open(path, "w") as f:
        json.dump(graph, f)
    return str(path)


def load_graph(path: str) -> Dict[str, Any]:
    with open(path, "r") as f:
        return json.load(f)
