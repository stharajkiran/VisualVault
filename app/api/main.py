from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.api.routes import search, upload
from app.pipeline import blip, clip, yolo
from app.pipeline import qdrant as qdrant_pipeline


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Load all models and connect to Qdrant on startup; clean up on shutdown.

    FastAPI calls the code before `yield` when the server starts and the code
    after `yield` when the server stops. Everything stored on `app.state` here
    is available to every route via `request.app.state`.
    """
    print("[Startup] Loading models ...")
    app.state.clip_model, app.state.clip_processor = clip.load_model()
    app.state.yolo_model = yolo.load_model()
    app.state.blip_model, app.state.blip_processor = blip.load_model()

    print("[Startup] Connecting to Qdrant ...")
    app.state.qdrant_client = qdrant_pipeline.get_client()
    qdrant_pipeline.create_collection(app.state.qdrant_client)

    print("[Startup] Ready.")
    yield

    # Shutdown — release Qdrant connection
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
