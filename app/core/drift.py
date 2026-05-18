"""Embedding drift — baseline computation from numpy cache and cosine distance scoring."""

from pathlib import Path

import numpy as np
from qdrant_client import QdrantClient

from app.embedding.config import CLIPConfig

# Embeddings exported by scripts/data/export_embeddings.py — COCO images only.
# To update the baseline: re-run export_embeddings.py then restart the API.
_EMBEDDINGS_PATH = Path("data/embeddings/embeddings.npy")


def compute_baseline(client: QdrantClient, config: CLIPConfig) -> np.ndarray:
    """
    Return the mean CLIP embedding across all COCO-indexed images.

    Loads from the pre-exported numpy file at data/embeddings/embeddings.npy — fast,
    Qdrant-independent, and COCO-only by construction (uploads are never exported).
    The baseline is intentionally stable: it represents the reference distribution
    that new uploads are measured against. To update it, re-run export_embeddings.py.

    Args:
        client (QdrantClient): Unused — kept for interface compatibility.
        config (CLIPConfig): Unused — kept for interface compatibility.

    Returns:
        np.ndarray: Mean embedding vector, shape (embedding_dim,).
    """
    embeddings = np.load(_EMBEDDINGS_PATH)
    return embeddings.mean(axis=0).astype(np.float32)


def compute_drift(batch_embeddings: list[np.ndarray], baseline: np.ndarray) -> float:
    """
    Compute cosine distance between the mean of a new batch and the index baseline.

    Both the batch mean and baseline are L2-normalized before the dot product because
    the mean of unit vectors is not itself a unit vector.

    Args:
        batch_embeddings (list[np.ndarray]): CLIP embeddings for the new batch,
            each shape (embedding_dim,).
        baseline (np.ndarray): Baseline mean embedding from compute_baseline(),
            shape (embedding_dim,).

    Returns:
        float: Cosine distance in [0, 2]. 0 = identical direction, higher = more drift.
    """
    batch_mean = np.stack(batch_embeddings).mean(axis=0)

    batch_norm = batch_mean / np.linalg.norm(batch_mean)
    baseline_norm = baseline / np.linalg.norm(baseline)

    cosine_similarity = float(np.dot(batch_norm, baseline_norm))
    return 1.0 - cosine_similarity
