from io import BytesIO
from pathlib import Path

from PIL import Image

from app.embedding.config import CLIP_BASE
from app.embedding.provider import CLIPPyTorchProvider
from app.embedding.registry import registry
from app.vision import blip, yolo
from app.vector_store import qdrant as qdrant_pipeline
from app.worker.celery_app import celery_app

# Module-level cache — loaded once per worker process, reused for every task
_models = None


def _get_models() -> dict:
    """
    Load all pipeline models and Qdrant client, caching them for the worker process.

    On first call, registers the CLIP provider, loads YOLO, BLIP-2, and connects
    to Qdrant. On every subsequent call, returns the already-loaded cache.

    Args:
        None

    Returns:
        dict: Keys — yolo_model, blip_model, blip_processor, qdrant_client.
              CLIP is accessed via registry.get_active().
    """
    global _models
    if _models is None:
        yolo_model = yolo.load_model()
        blip_model, blip_processor = blip.load_model()
        qdrant_client = qdrant_pipeline.get_client()
        qdrant_pipeline.create_collection(qdrant_client, CLIP_BASE)

        _models = {
            "clip_provider": CLIPPyTorchProvider(CLIP_BASE),
            "yolo_model": yolo_model,
            "blip_model": blip_model,
            "blip_processor": blip_processor,
            "qdrant_client": qdrant_client,
        }

    return _models


@celery_app.task
def ingest_image(filename: str, image_bytes: bytes) -> dict:
    """
    Run the full pipeline on an uploaded image and index it in Qdrant.

    Decodes the raw image bytes, runs YOLO detection, BLIP-2 captioning, and
    CLIP embedding in sequence, then upserts the result into Qdrant. Returns
    tags and caption so the caller can retrieve them by polling the job ID.

    Args:
        filename (str): Original filename — used to derive the Qdrant point ID.
        image_bytes (bytes): Raw image file contents, decoded into a PIL Image internally.

    Returns:
        dict: Keys — filename (str), tags (list[dict] with label and confidence),
              caption (str).
    """
    models = _get_models()
    image = Image.open(BytesIO(image_bytes)).convert("RGB")

    tags_raw = yolo.detect(image, models["yolo_model"])
    caption = blip.caption(image, models["blip_model"], models["blip_processor"])
    embedding = models["clip_provider"].embed_image(image)

    stem = Path(filename).stem
    image_id = int(stem) if stem.isdigit() else abs(hash(filename)) % (2**63)
    qdrant_pipeline.upsert(
        client=models["qdrant_client"],
        config=CLIP_BASE,
        image_id=image_id,
        embedding=embedding,
        payload={"filename": filename, "image_path": f"uploads/{filename}"},
    )

    return {
        "filename": filename,
        "tags": [
            {"label": label, "confidence": round(conf, 4)} for label, conf in tags_raw
        ],
        "caption": caption,
    }
