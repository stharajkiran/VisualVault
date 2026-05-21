"""Celery tasks: ingest_image, ingest_video, preview_frame, check_and_retrain (daily retraining beat)."""
import hashlib
import os
import shutil
import subprocess
import tempfile
import time
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path

import ffmpeg
from PIL import Image

from app.core.label_studio import LOW_CONFIDENCE_THRESHOLD, push_to_label_studio
from app.db.database import append_ai_action, write_governance_record
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
    # PIPELINE_LATENCY covers the full task wall time including any exception path,
    # so it goes in finally rather than after the return
    task_start = time.perf_counter()
    try:
        # Models are loaded once per worker process and reused — _get_models() is
        # a no-op on every call after the first
        models = _get_models()
        image = Image.open(BytesIO(image_bytes)).convert("RGB")

        # YOLO runs first — its output drives the Label Studio flagging decision
        t0 = time.perf_counter()
        tags_raw = yolo.detect(image, models["yolo_model"])
        INFERENCE_LATENCY.labels(model="yolo").observe(time.perf_counter() - t0)

        # Any detection below the confidence threshold gets flagged for human review.
        # push_to_label_studio fails silently — a Label Studio outage never blocks ingestion
        low_conf = [(label, conf) for label, conf in tags_raw if conf < LOW_CONFIDENCE_THRESHOLD]
        if low_conf:
            push_to_label_studio(filename, image_bytes, low_conf)

        # BLIP-2 generates a free-text caption — runs independently of YOLO tags
        t0 = time.perf_counter()
        caption = blip.caption(image, models["blip_model"], models["blip_processor"])
        INFERENCE_LATENCY.labels(model="blip").observe(time.perf_counter() - t0)

        # CLIP produces the embedding used for semantic search — must run last
        # because its output goes directly into Qdrant
        t0 = time.perf_counter()
        embedding = models["clip_provider"].embed_image(image)
        INFERENCE_LATENCY.labels(model=CLIP_BASE.alias).observe(
            time.perf_counter() - t0
        )

        # COCO images have numeric filenames (e.g. "000000000009.jpg") — use the
        # integer directly so Qdrant IDs correlate with COCO annotation IDs.
        # Uploaded images get a hash bounded to Qdrant's uint64 range (2**63)
        stem = Path(filename).stem
        image_id = int(stem) if stem.isdigit() else int(hashlib.sha256(filename.encode()).hexdigest(), 16) % (2**63)

        uploads_dir = Path("data/uploads")
        uploads_dir.mkdir(parents=True, exist_ok=True)
        (uploads_dir / filename).write_bytes(image_bytes)

        payload: dict = {"filename": filename, "image_path": f"data/uploads/{filename}"}
        # timestamp_s is only present for video keyframes — None for static images
        if timestamp_s is not None:
            payload["timestamp_s"] = timestamp_s

        qdrant_pipeline.upsert(
            client=models["qdrant_client"],
            config=CLIP_BASE,
            image_id=image_id,
            embedding=embedding,
            payload=payload,
        )

        write_governance_record(image_id)
        append_ai_action(image_id, {
            "action": "ingest_image",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "tags": [{"label": l, "confidence": round(c, 4)} for l, c in tags_raw],
            "caption": caption,
            "model": CLIP_BASE.alias,
        })

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

    # TemporaryDirectory auto-deletes on exit — FFmpeg requires a real file path,
    # it cannot read from an in-memory buffer
    with tempfile.TemporaryDirectory() as tmpdir:
        video_path = Path(tmpdir) / filename
        video_path.write_bytes(video_bytes)

        # fps=0.5 → one frame every 2 seconds; %04d gives FFmpeg its zero-padded naming
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

        # sorted() ensures frames are processed in chronological order
        frame_files = sorted(Path(tmpdir).glob("frame_*.jpg"))

        for idx, frame_path in enumerate(frame_files):
            # FFmpeg names frames 1-indexed; idx is 0-indexed, so frame 0 → 0s, frame 1 → 2s
            timestamp_s = idx * 2
            frame_filename = f"{video_stem}_t{timestamp_s}s.jpg"
            frame_bytes = frame_path.read_bytes()
            # Call ingest_image directly (not .delay()) — we're already inside a
            # Celery task; dispatching a sub-task would require separate result polling
            result = ingest_image(frame_filename, frame_bytes, float(timestamp_s))
            results.append(result)

    return {
        "video_filename": filename,
        "frame_count": len(results),
        "frames": results,
    }


@celery_app.task
def preview_frame(image_bytes: bytes) -> dict:
    """
    Run YOLO on a single frame and return tags — no indexing, no captioning.

    Used by the live preview endpoint to provide sub-100ms tag feedback on
    webcam frames. BLIP-2, CLIP, and Qdrant are intentionally excluded to
    keep latency low. Nothing is written to Qdrant, PostgreSQL, or disk.

    Args:
        image_bytes (bytes): Raw image file contents.

    Returns:
        dict: tags (list[dict] with label and confidence).
    """
    models = _get_models()
    image = Image.open(BytesIO(image_bytes)).convert("RGB")
    t0 = time.perf_counter()
    tags_raw = yolo.detect(image, models["yolo_model"])
    INFERENCE_LATENCY.labels(model="yolo_preview").observe(time.perf_counter() - t0)
    return {
        "tags": [
            {"label": label, "confidence": round(conf, 4)}
            for label, conf in tags_raw
        ]
    }


@celery_app.task
def find_similar_by_image(image_bytes: bytes, top_k: int = 10) -> list[dict]:
    """
    Embed an image with CLIP and return the top-K most similar assets from Qdrant.

    Used for reverse image search — the image is not indexed. Nothing is written
    to Qdrant, PostgreSQL, or disk. Returns results in the same format as
    qdrant_pipeline.search() so the API can treat both similar endpoints identically.

    Args:
        image_bytes (bytes): Raw image file contents.
        top_k (int): Number of similar assets to return. Defaults to 10.

    Returns:
        list[dict]: Each dict has id, score, filename, image_path keys.
    """
    models = _get_models()
    image = Image.open(BytesIO(image_bytes)).convert("RGB")
    t0 = time.perf_counter()
    embedding = models["clip_provider"].embed_image(image)
    INFERENCE_LATENCY.labels(model="clip_similar").observe(time.perf_counter() - t0)
    return qdrant_pipeline.search(
        models["qdrant_client"], CLIP_BASE, embedding, top_k=top_k
    )


@celery_app.task
def check_and_retrain() -> dict:
    """
    Daily Celery beat task — trigger YOLO fine-tuning when enough corrections exist.

    Queries PostgreSQL for the total correction count. If fewer than 100 corrections
    exist, returns early without training. Otherwise runs YOLO fine-tuning, evaluates
    the new model on holdout data, promotes it only if both precision and recall
    improve, and logs every run to MLflow.

    Args:
        None

    Returns:
        dict: status (str), corrections (int), and per-run metrics when training runs.
    """
    # Lazy imports — mlflow and retrain_yolo are heavy (loads YOLO weights).
    # Importing at module level would slow down every worker startup even when
    # retraining never runs
    import mlflow
    from app.db.database import count_corrections
    from scripts.model.retrain_yolo import retrain_yolo

    n = count_corrections()
    # 100 corrections is the minimum viable fine-tuning dataset — below this,
    # YOLO would overfit to the small correction set
    if n < 100:
        return {"status": "skipped", "corrections": n, "reason": f"need 100+, have {n}"}

    try:
        result = retrain_yolo()

        # Require both metrics to improve — OR would be too lenient. A model can
        # gain precision by being more conservative (fewer detections) at the cost
        # of recall; both must improve to confirm genuine quality gain
        promoted = (
            result["precision"] > result["baseline_precision"]
            and result["recall"] > result["baseline_recall"]
        )

        # MLFLOW_TRACKING_URI differs between local (localhost:5000) and Docker
        # (mlflow:5000) — env var lets the same code work in both environments
        mlflow.set_tracking_uri(
            os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000")
        )
        mlflow.set_experiment("visualvault/yolo-retraining")
        with mlflow.start_run():
            mlflow.log_params({"correction_count": n, "promotion_threshold": 100})
            mlflow.log_metrics({
                "precision": result["precision"],
                "recall": result["recall"],
                "baseline_precision": result["baseline_precision"],
                "baseline_recall": result["baseline_recall"],
            })
            mlflow.set_tag("promoted", str(promoted))
            # Only log the artifact when promoted — avoids storing every failed
            # run's model weights in the MLflow artifact store
            if promoted and result.get("model_path"):
                mlflow.log_artifact(result["model_path"])

        # Update Prometheus gauge so Grafana shows the improvement trend over runs.
        # Set regardless of promotion — even a non-promoted run's score is useful signal
        from app.core.metrics import YOLO_MEAN_CONFIDENCE
        YOLO_MEAN_CONFIDENCE.set(result["precision"])

        if promoted:
            # Overwrite the current fine-tuned model — the previous version is
            # recoverable via DVC (dvc checkout on the prior Git commit)
            shutil.copy(result["model_path"], "artifacts/models/yolo11n_finetuned.pt")
            # Version the promoted model with DVC — requires dvc to be installed
            # in the worker container (add dvc to ml-worker extras in pyproject.toml)
            subprocess.run(
                ["dvc", "add", "artifacts/models/yolo11n_finetuned.pt"],
                check=False,  # DVC failure must not crash the task
            )

        return {
            "status": "retrained",
            "corrections": n,
            "precision": result["precision"],
            "recall": result["recall"],
            "promoted": promoted,
        }

    except Exception as exc:
        # Return failure dict rather than raising — a training crash must not
        # stop the beat schedule from firing again tomorrow
        return {"status": "failed", "error": str(exc)}
