"""Live preview endpoints — POST /live/frame (HTTP) and WebSocket /ws/live."""
import asyncio
from functools import partial

from fastapi import APIRouter, HTTPException, UploadFile, WebSocket, WebSocketDisconnect
from pydantic import BaseModel

from app.worker.tasks import preview_frame

router = APIRouter()


class TagResult(BaseModel):
    label: str
    confidence: float


class PreviewResponse(BaseModel):
    tags: list[TagResult]


@router.post("/live/frame", response_model=PreviewResponse)
async def live_frame(file: UploadFile) -> PreviewResponse:
    """
    Accept a single image frame and return YOLO tags within one second.

    Dispatches preview_frame to the Celery worker and waits synchronously.
    No indexing, captioning, or Qdrant upsert — pure preview.

    Args:
        file (UploadFile): JPEG or PNG frame from the client.

    Returns:
        PreviewResponse: YOLO tags with confidence scores.
    """
    image_bytes = await file.read()
    loop = asyncio.get_event_loop()
    try:
        task = preview_frame.delay(image_bytes)
        result = await loop.run_in_executor(None, partial(task.get, timeout=5))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Preview failed: {exc}")
    return PreviewResponse(tags=result["tags"])


@router.websocket("/ws/live")
async def ws_live(websocket: WebSocket) -> None:
    """
    WebSocket endpoint for real-time frame tagging.

    Client sends raw image bytes; server responds with JSON tags for each frame.
    Runs until the client disconnects. Each frame is dispatched to the Celery
    worker and awaited in a thread executor to avoid blocking the event loop.

    Args:
        websocket (WebSocket): Active WebSocket connection.

    Returns:
        None
    """
    await websocket.accept()
    loop = asyncio.get_event_loop()
    try:
        while True:
            image_bytes = await websocket.receive_bytes()
            task = preview_frame.delay(image_bytes)
            result = await loop.run_in_executor(None, partial(task.get, timeout=5))
            await websocket.send_json(result)
    except WebSocketDisconnect:
        pass
