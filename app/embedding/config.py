from dataclasses import dataclass


@dataclass(frozen=True) # makes instances immutable
class CLIPConfig:
    """
    Complete description of a CLIP model variant and its associated storage contract.

    All fields that are coupled — model identity, vector dimension, normalization,
    distance metric, and collection name — live here together so they cannot drift
    out of sync. Every component that needs any of these values reads from one
    CLIPConfig instance, never from separate constants.

    Fields:
        model_id (str): HuggingFace model identifier.
        alias (str): Short name used in logs, MLflow runs, and Prometheus labels.
        embedding_dim (int): Output vector size produced by this model.
        input_size (int): Expected image resolution in pixels (square).
        precision (str): Floating point precision — "fp16" or "fp32".
        backend (str): Inference backend — "pytorch" or "tensorrt".
        engine_path (str | None): Path to compiled .engine file. None for pytorch backend.
        collection_name (str): Qdrant collection that stores this model's embeddings.
        distance_metric (str): Vector similarity metric — "cosine", "dot", or "l2".
        normalize_embeddings (bool): Whether to L2-normalize embeddings before storage.
                                     Must be True when using cosine distance.
    """

    model_id: str
    alias: str
    embedding_dim: int
    input_size: int
    precision: str
    backend: str
    engine_path: str | None
    collection_name: str
    distance_metric: str
    normalize_embeddings: bool


# ── Pre-built configs ──────────────────────────────────────────────────────────

CLIP_BASE = CLIPConfig(
    model_id="openai/clip-vit-base-patch32",
    alias="clip-b32",
    embedding_dim=512,
    input_size=224,
    precision="fp32",
    backend="pytorch",
    engine_path=None,
    collection_name="visualvault",
    distance_metric="cosine",
    normalize_embeddings=True,
)

CLIP_LARGE = CLIPConfig(
    model_id="openai/clip-vit-large-patch14",
    alias="clip-l14",
    embedding_dim=768,
    input_size=224,
    precision="fp32",
    backend="pytorch",
    engine_path=None,
    collection_name="visualvault_large",
    distance_metric="cosine",
    normalize_embeddings=True,
)

CLIP_BASE_TRT = CLIPConfig(
    model_id="openai/clip-vit-base-patch32",
    alias="clip-b32-trt",
    embedding_dim=512,
    input_size=224,
    precision="fp16",
    backend="tensorrt",
    engine_path="artifacts/tensorrt/clip_image_encoder.engine",
    collection_name="visualvault",
    distance_metric="cosine",
    normalize_embeddings=True,
)
