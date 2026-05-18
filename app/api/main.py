from contextlib import asynccontextmanager

from fastapi import FastAPI
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from prometheus_fastapi_instrumentator import Instrumentator
from starlette.responses import Response

from app.api.routes import corrections, drift, governance, jobs, live, search, upload, video
from app.core.drift import compute_baseline
from app.core.metrics import update_queue_depth
from app.embedding.config import CLIP_BASE
from app.embedding.provider import CLIPPyTorchProvider
from app.embedding.registry import registry
from app.vector_store import qdrant as qdrant_pipeline


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Load the CLIP text encoder and connect to Qdrant on startup.

    The API only needs the text encoder — for embedding search queries.
    YOLO, BLIP-2, and the CLIP image encoder belong exclusively in the
    Celery worker where all image inference happens.
    """
    print("[Startup] Registering embedding provider ...")
    provider = CLIPPyTorchProvider(CLIP_BASE)
    provider.warmup()
    registry.register(provider)

    print("[Startup] Connecting to Qdrant ...")
    app.state.qdrant_client = qdrant_pipeline.get_client()
    qdrant_pipeline.create_collection(app.state.qdrant_client, CLIP_BASE)

    print("[Startup] Computing embedding baseline ...")
    try:
        app.state.baseline = compute_baseline(app.state.qdrant_client, CLIP_BASE)
    except Exception as exc:
        print(f"[Startup] Warning: baseline computation failed ({exc}) — drift detection disabled")
        app.state.baseline = None

    print("[Startup] Ready.")
    yield

    app.state.qdrant_client.close()
    print("[Shutdown] Done.")


app = FastAPI(
    title="VisualVault",
    description="Semantic image search — upload images, search by description.",
    version="0.1.0",
    lifespan=lifespan,
)

app.include_router(upload.router)
app.include_router(search.router)
app.include_router(jobs.router)
app.include_router(video.router)
app.include_router(drift.router)
app.include_router(corrections.router)
app.include_router(governance.router)
app.include_router(live.router)

Instrumentator().instrument(app)


@app.get("/metrics", include_in_schema=False)
def metrics() -> Response:
    """Expose Prometheus metrics, updating queue depth from Redis before each scrape."""
    update_queue_depth()
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
