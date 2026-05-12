from fastapi import APIRouter, HTTPException, UploadFile

from app.worker.tasks import ingest_image

router = APIRouter()

ALLOWED_CONTENT_TYPES = {"image/jpeg", "image/png", "image/webp"}


@router.post("/upload")
async def upload_image(file: UploadFile) -> dict:
    """
    Accept an image upload and dispatch it to a Celery worker for processing.

    Validates the file type, reads the bytes, and pushes an ingest_image task
    to the Celery queue. Returns immediately with a job ID the client can use
    to poll for the result.

    Args:
        file (UploadFile): Multipart image file sent by the client.

    Returns:
        dict: Single key — job_id (str), a UUID the client uses to poll GET /jobs/{job_id}.
    """
    if file.content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(
            status_code=422,
            detail=f"Unsupported file type '{file.content_type}'. Use JPEG, PNG, or WebP.",
        )
    contents = await file.read()
    result = ingest_image.delay(file.filename, contents)
    return {"job_id": result.id}
