import time
from io import BytesIO

from fastapi import APIRouter, HTTPException, Request, UploadFile

from app.api.models import TagResult, UploadResponse
from app.pipeline import blip, clip, qdrant as qdrant_pipeline, yolo

router = APIRouter()

ALLOWED_CONTENT_TYPES = {"image/jpeg", "image/png", "image/webp"}


def _image_id_from_filename(filename: str) -> int:
    """
    Derive a stable integer Qdrant ID from an uploaded filename.

    Uses Python's built-in hash truncated to a positive 63-bit integer.
    Uploading the same filename twice produces the same ID, so Qdrant
    upserts (overwrites) instead of creating a duplicate point.

    Args:
        filename (str): Original filename of the uploaded file.

    Returns:
        int: Positive integer suitable for use as a Qdrant point ID.
    """
    return abs(hash(filename)) % (2**63)


@router.post("/upload", response_model=UploadResponse)
async def upload_image(request: Request, file: UploadFile) -> UploadResponse:
    """
    Accept an image upload, run it through the full pipeline, and index it.

    Reads the uploaded file into memory, validates the content type, runs
    YOLO detection, BLIP-2 captioning, and CLIP embedding in sequence, then
    upserts the embedding into Qdrant. Returns tags, caption, and total
    processing time.

    Args:
        request (Request): FastAPI request object — used to access app.state models.
        file (UploadFile): Multipart image file sent by the client.

    Returns:
        UploadResponse: Filename, detected tags, caption, and processing time in ms.
    """
    if file.content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(
            status_code=422,
            detail=f"Unsupported file type '{file.content_type}'. Use JPEG, PNG, or WebP.",
        )
    # these are raw bytes from the uploaded file; we will use PIL to decode them into an image
    contents = await file.read()
    try:
        from PIL import Image
        image = Image.open(BytesIO(contents)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=422, detail="Could not decode image file.")

    # Retrieve models loaded once at startup via lifespan
    state = request.app.state
    t0 = time.perf_counter()

    tags_raw = yolo.detect(image, state.yolo_model)
    caption_text = blip.caption(image, state.blip_model, state.blip_processor)
    embedding = clip.embed_image(image, state.clip_model, state.clip_processor)

    processing_ms = (time.perf_counter() - t0) * 1000

    image_id = _image_id_from_filename(file.filename)
    qdrant_pipeline.upsert(
        client=state.qdrant_client,
        image_id=image_id,
        embedding=embedding,
        payload={"filename": file.filename, "image_path": f"uploads/{file.filename}"},
    )

    return UploadResponse(
        filename=file.filename,
        tags=[TagResult(label=label, confidence=round(conf, 4)) for label, conf in tags_raw],
        caption=caption_text,
        processing_ms=round(processing_ms, 1),
    )
