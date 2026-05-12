import os
import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

from app.embedding.config import CLIPConfig


HOST = os.getenv("QDRANT_HOST", "localhost")
PORT = int(os.getenv("QDRANT_PORT", "6333"))


def get_client() -> QdrantClient:
    """
    Return a connected Qdrant client.

    Host and port are read from environment variables so the same code works
    locally (localhost:6333) and inside Docker Compose (qdrant:6333) without changes.
    """
    return QdrantClient(host=HOST, port=PORT)


def create_collection(client: QdrantClient, config:CLIPConfig) -> None:
    """
    Create the visualvault collection if it does not already exist.

    Configured for 512-dim vectors with cosine distance — matching the CLIP
    embedding space. Cosine distance measures the angle between vectors, which
    is the right metric for semantic similarity (magnitude doesn't matter, direction does).
    """
    existing = [c.name for c in client.get_collections().collections]
    if config.collection_name in existing:
        print(f"[Qdrant] Collection '{config.collection_name}' already exists, skipping creation.")
        return

    client.create_collection(
        collection_name=config.collection_name,
        vectors_config=VectorParams(size=config.embedding_dim, distance=Distance.COSINE),
    )
    print(f"[Qdrant] Created collection '{config.collection_name}' ({config.embedding_dim}-dim, cosine).")


def upsert(client: QdrantClient, config: CLIPConfig, image_id: int, embedding: np.ndarray, payload: dict) -> None:
    """
    Insert or update a single image embedding in Qdrant.

    Each point in Qdrant has three parts:
    - id: a unique integer identifying this image
    - vector: the 512-dim CLIP embedding
    - payload: arbitrary metadata (filename, path, etc.) stored alongside the vector

    The payload is what gets returned in search results — it's how we map a
    matching vector back to the actual image file on disk.
    """
    point = PointStruct(
        id=image_id,
        vector=embedding.tolist(),
        payload=payload,
    )
    client.upsert(collection_name=config.collection_name, points=[point])


def search(client: QdrantClient, config: CLIPConfig, query_vector: np.ndarray, top_k: int = 10) -> list[dict]:
    """
    Find the top-k most similar images to a query vector.

    Qdrant computes cosine similarity between the query vector and every stored
    vector, then returns the top_k closest matches. Each result includes the
    payload (filename, path) and a score between 0 and 1 — higher is more similar.

    Args:
        query_vector: 512-dim normalized CLIP embedding of a text query or image.
        top_k: Number of results to return.

    Returns:
        List of dicts with keys: id, score, filename, image_path.
    """
    results = client.query_points(
        collection_name=config.collection_name,
        query=query_vector.tolist(),
        limit=top_k,
    ).points

    return [
        {
            "id": r.id,
            "score": r.score,
            **r.payload,
        }
        for r in results
    ]
