import numpy as np
from qdrant_client import QdrantClient

from app.embedding.config import CLIPConfig


def compute_baseline(client: QdrantClient, config: CLIPConfig) -> np.ndarray:
    """
    Compute the mean CLIP embedding across all indexed images in a Qdrant collection.

    Scrolls the collection in pages of 256 points, collecting every stored vector,
    then returns their unweighted mean. The result is NOT L2-normalized — the mean
    of unit vectors is generally not a unit vector.

    Args:
        client (QdrantClient): Connected Qdrant client.
        config (CLIPConfig): Collection config — collection_name is used for the query.

    Returns:
        np.ndarray: Mean embedding vector, shape (embedding_dim,).
    """
    vectors: list[list[float]] = []
    offset = None

    while True:
        results, next_offset = client.scroll(
            collection_name=config.collection_name,
            with_vectors=True,
            limit=256,
            offset=offset,
        )
        for point in results:
            vectors.append(point.vector)
        if next_offset is None:
            break
        offset = next_offset

    return np.array(vectors, dtype=np.float32).mean(axis=0)


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
