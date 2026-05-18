"""POST /drift/detect — embeds a folder with CLIP and reports cosine drift vs index baseline."""
from pathlib import Path

from fastapi import APIRouter, HTTPException, Request
from PIL import Image

from app.core.drift import compute_drift
from app.core.metrics import EMBEDDING_DRIFT
from app.embedding.config import CLIP_BASE
from app.embedding.registry import registry

router = APIRouter()

ALLOWED_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}


@router.post("/drift/detect")
async def detect_drift(request: Request, folder: str) -> dict:
    """
    Compute embedding drift between a batch of images and the current index baseline.

    Embeds every image in the given folder using CLIP, computes the mean embedding,
    then measures cosine distance against the mean of all indexed embeddings. Updates
    the Prometheus gauge so the score is available on the next /metrics scrape.

    Args:
        request (Request): FastAPI request — used to access app.state.qdrant_client.
        folder (str): Path to a directory of images. Accepts .jpg, .jpeg, .png, .webp.

    Returns:
        dict: drift_score (float), batch_size (int), baseline_size (int).
    """
    folder_path = Path(folder)
    if not folder_path.is_dir():
        raise HTTPException(status_code=422, detail=f"Folder not found: {folder}")

    image_paths = sorted(
        p for p in folder_path.iterdir()
        if p.suffix.lower() in ALLOWED_IMAGE_SUFFIXES
    )
    if not image_paths:
        raise HTTPException(status_code=422, detail=f"No images found in: {folder}")

    provider = registry.get_active()
    batch_embeddings = [
        provider.embed_image(Image.open(p).convert("RGB"))
        for p in image_paths
    ]

    qdrant_client = request.app.state.qdrant_client
    drift_score = compute_drift(batch_embeddings, request.app.state.baseline)

    EMBEDDING_DRIFT.set(drift_score)

    baseline_size = qdrant_client.count(
        collection_name=CLIP_BASE.collection_name
    ).count

    return {
        "drift_score": round(drift_score, 6),
        "batch_size": len(batch_embeddings),
        "baseline_size": baseline_size,
    }
