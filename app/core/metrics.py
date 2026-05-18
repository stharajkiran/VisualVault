"""Prometheus metric definitions for the API and Celery worker."""
import os

import redis
from prometheus_client import Counter, Gauge, Histogram

REDIS_URL = os.environ.get("REDIS_URL", "redis://localhost:6379/0")

# Inference latency per embed_image() call, labeled by model alias (e.g. "clip-b32")
INFERENCE_LATENCY = Histogram(
    "visualvault_inference_seconds",
    "Time spent running embed_image() in the Celery worker",
    labelnames=["model"],
    buckets=[0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0],
)

# Number of tasks currently waiting in the Celery queue — read directly from Redis
QUEUE_DEPTH = Gauge(
    "visualvault_queue_depth",
    "Number of ingest_image tasks waiting in the Redis queue",
)

PIPELINE_LATENCY = Histogram(
    "visualvault_pipeline_seconds",
    "Time spent running the full ingest_image Celery task",
    buckets=[0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0],
)

EMBEDDING_DRIFT = Gauge(
    "visualvault_embedding_drift",
    "Cosine distance between the latest batch mean embedding and the index baseline",
)

CORRECTIONS_TOTAL = Counter(
    "visualvault_corrections_total",
    "Total number of human corrections submitted via Label Studio webhook",
)

YOLO_MEAN_CONFIDENCE = Gauge(
    "visualvault_yolo_mean_confidence",
    "Mean detection confidence of the active YOLO model on holdout images — updated after each retraining run",
)


def update_queue_depth() -> None:
    """
    Query Redis for the real Celery queue length and update the gauge.

    Called by the Prometheus instrumentator before each scrape so the metric
    reflects the actual queue state, not an approximation tracked in memory.
    The Celery default queue is a Redis list named 'celery'.

    Args:
        None

    Returns:
        None
    """
    try:
        client = redis.from_url(REDIS_URL)
        depth = client.llen("celery")
        QUEUE_DEPTH.set(depth)
    except Exception:
        pass
