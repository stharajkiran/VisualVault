from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.api.routes import jobs, search, upload
from app.embedding.config import CLIP_BASE
from app.embedding.provider import CLIPPyTorchProvider
from app.embedding.registry import registry
from app.vision import blip, yolo
from app.vector_store import qdrant as qdrant_pipeline


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Load all models, register the embedding provider, and connect to Qdrant on startup.

    FastAPI calls the code before `yield` when the server starts and the code
    after `yield` when the server stops. CLIP is loaded via the registry so
    the active model can be swapped by changing CLIP_BASE to another config.
    YOLO and BLIP are loaded directly onto app.state — no variants to swap.
    """    
    print("[Startup] Registering embedding provider ...")
    provider = CLIPPyTorchProvider(CLIP_BASE)
    provider.warmup()
    registry.register(provider)

    print("[Startup] Loading vision models ...")
    app.state.yolo_model = yolo.load_model()
    app.state.blip_model, app.state.blip_processor = blip.load_model()

    print("[Startup] Connecting to Qdrant ...")
    app.state.qdrant_client = qdrant_pipeline.get_client()
    qdrant_pipeline.create_collection(app.state.qdrant_client, CLIP_BASE)

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
