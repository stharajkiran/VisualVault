"""POST /upload/video — validates MIME type and dispatches ingest_video to the Celery worker."""
from fastapi import APIRouter, HTTPException, UploadFile

from app.worker.celery_app import celery_app

router = APIRouter()

ALLOWED_VIDEO_TYPES = {"video/mp4", "video/quicktime", "video/x-msvideo", "video/webm"}


@router.post("/upload/video")
async def upload_video(file: UploadFile) -> dict:
    """
    Accept a video upload and dispatch it to the Celery worker for processing.

    The worker handles FFmpeg frame extraction and runs the full pipeline on
    each frame. The API never touches FFmpeg or any ML models.

    Args:
        file (UploadFile): Multipart video file — MP4, MOV, AVI, or WebM.

    Returns:
        dict: Keys — video_filename (str), job_id (str).
    """
    if file.content_type not in ALLOWED_VIDEO_TYPES:
        raise HTTPException(
            status_code=422,
            detail=f"Unsupported file type '{file.content_type}'. Use MP4, MOV, AVI, or WebM.",
        )

    contents = await file.read()
    result = celery_app.send_task(
        "app.worker.tasks.ingest_video",
        args=[file.filename, contents],
    )
    return {"video_filename": file.filename, "job_id": result.id}
