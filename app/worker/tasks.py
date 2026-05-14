import tempfile
import time
from io import BytesIO
from pathlib import Path

import ffmpeg
from PIL import Image

from app.core.metrics import INFERENCE_LATENCY, PIPELINE_LATENCY
from app.embedding.config import CLIP_BASE
from app.embedding.provider import CLIPPyTorchProvider
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
def ingest_image(
    filename: str, image_bytes: bytes, timestamp_s: float | None = None
) -> dict:
    """
    Run the full pipeline on an uploaded image and index it in Qdrant.

    Decodes the raw image bytes, runs YOLO detection, BLIP-2 captioning, and
    CLIP embedding in sequence, then upserts the result into Qdrant. Returns
    tags and caption so the caller can retrieve them by polling the job ID.

    Args:
        filename (str): Original filename — used to derive the Qdrant point ID.
        image_bytes (bytes): Raw image file contents, decoded into a PIL Image internally.
        timestamp_s (float | None): Seconds from start of video for keyframes; None for images.

    Returns:
        dict: Keys — filename (str), tags (list[dict] with label and confidence),
              caption (str).
    """
    task_start = time.perf_counter()
    try:
        models = _get_models()
        image = Image.open(BytesIO(image_bytes)).convert("RGB")

        t0 = time.perf_counter()
        tags_raw = yolo.detect(image, models["yolo_model"])
        INFERENCE_LATENCY.labels(model="yolo").observe(time.perf_counter() - t0)

        t0 = time.perf_counter()
        caption = blip.caption(image, models["blip_model"], models["blip_processor"])
        INFERENCE_LATENCY.labels(model="blip").observe(time.perf_counter() - t0)

        t0 = time.perf_counter()
        embedding = models["clip_provider"].embed_image(image)
        INFERENCE_LATENCY.labels(model=CLIP_BASE.alias).observe(
            time.perf_counter() - t0
        )

        stem = Path(filename).stem
        image_id = int(stem) if stem.isdigit() else abs(hash(filename)) % (2**63)
        payload: dict = {"filename": filename, "image_path": f"uploads/{filename}"}
        if timestamp_s is not None:
            payload["timestamp_s"] = timestamp_s
        qdrant_pipeline.upsert(
            client=models["qdrant_client"],
            config=CLIP_BASE,
            image_id=image_id,
            embedding=embedding,
            payload=payload,
        )

        return {
            "filename": filename,
            "tags": [
                {"label": label, "confidence": round(conf, 4)}
                for label, conf in tags_raw
            ],
            "caption": caption,
        }
    finally:
        PIPELINE_LATENCY.observe(time.perf_counter() - task_start)


@celery_app.task
def ingest_video(filename: str, video_bytes: bytes) -> dict:
    """
    Extract frames from a video and run the full pipeline on each frame.

    Writes video bytes to a temp file, uses FFmpeg to extract one frame every
    2 seconds, then runs YOLO + BLIP-2 + CLIP + Qdrant upsert on each frame.
    All processing happens in the worker — the API never touches FFmpeg.

    Args:
        filename (str): Original video filename.
        video_bytes (bytes): Raw video file contents.

    Returns:
        dict: Keys — video_filename (str), frame_count (int),
              frames (list[dict] with filename, tags, caption per frame).
    """
    video_stem = Path(filename).stem
    results = []

    with tempfile.TemporaryDirectory() as tmpdir:
        video_path = Path(tmpdir) / filename
        video_path.write_bytes(video_bytes)

        frame_pattern = str(Path(tmpdir) / "frame_%04d.jpg")
        try:
            (
                ffmpeg.input(str(video_path))
                .filter("fps", fps=0.5)
                .output(frame_pattern, vcodec="mjpeg", qscale=2)
                .run(quiet=True, overwrite_output=True)
            )
        except ffmpeg.Error as exc:
            stderr = exc.stderr.decode(errors="replace") if exc.stderr else "unknown error"
            raise RuntimeError(f"FFmpeg extraction failed: {stderr}")

        frame_files = sorted(Path(tmpdir).glob("frame_*.jpg"))

        for idx, frame_path in enumerate(frame_files):
            timestamp_s = idx * 2
            frame_filename = f"{video_stem}_t{timestamp_s}s.jpg"
            frame_bytes = frame_path.read_bytes()
            result = ingest_image(frame_filename, frame_bytes, float(timestamp_s))
            results.append(result)

    return {
        "video_filename": filename,
        "frame_count": len(results),
        "frames": results,
    }
