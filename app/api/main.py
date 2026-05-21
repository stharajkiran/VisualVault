import logging
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

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s — %(message)s")
logging.getLogger("transformers").setLevel(logging.WARNING)
logging.getLogger("ultralytics").setLevel(logging.WARNING)
logging.getLogger("torch").setLevel(logging.WARNING)
logging.getLogger("PIL").setLevel(logging.WARNING)

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Load the CLIP text encoder and connect to Qdrant on startup.

    The API only needs the text encoder — for embedding search queries.
    YOLO, BLIP-2, and the CLIP image encoder belong exclusively in the
    Celery worker where all image inference happens.
    """
    logger.info("Registering embedding provider ...")
    provider = CLIPPyTorchProvider(CLIP_BASE)
    provider.embed_text("warmup")  # warm text encoder — API never calls embed_image
    registry.register(provider)

    logger.info("Connecting to Qdrant ...")
    app.state.qdrant_client = qdrant_pipeline.get_client()
    qdrant_pipeline.create_collection(app.state.qdrant_client, CLIP_BASE)

    logger.info("Computing embedding baseline ...")
    try:
        app.state.baseline = compute_baseline(app.state.qdrant_client, CLIP_BASE)
    except Exception as exc:
        logger.warning("Baseline computation failed (%s) — drift detection disabled", exc)
        app.state.baseline = None

    logger.info("Ready.")
    yield

    app.state.qdrant_client.close()
    logger.info("Shutdown complete.")


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
