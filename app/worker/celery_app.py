"""Celery app — Redis broker, beat schedule, worker Prometheus metrics."""
import os
from datetime import timedelta

from celery import Celery
from celery.signals import worker_ready
from prometheus_client import start_http_server

REDIS_URL = os.environ.get("REDIS_URL", "redis://localhost:6379/0")
WORKER_METRICS_PORT = int(os.environ.get("WORKER_METRICS_PORT", "9101"))
_metrics_server_started = False

celery_app = Celery(
    "visualvault",
    broker=REDIS_URL,
    backend=REDIS_URL,
    include=["app.worker.tasks"],
)

celery_app.conf.beat_schedule = {
    "check-and-retrain-daily": {
        "task": "app.worker.tasks.check_and_retrain",
        "schedule": timedelta(hours=24),
    },
}


@worker_ready.connect
def start_worker_metrics_server(**_: object) -> None:
    """
    Expose worker-local Prometheus metrics.

    The compose worker uses a single-process pool so task observations and the
    HTTP metrics endpoint share the same in-memory prometheus_client registry.
    """
    global _metrics_server_started
    if not _metrics_server_started:
        start_http_server(WORKER_METRICS_PORT)
        _metrics_server_started = True
