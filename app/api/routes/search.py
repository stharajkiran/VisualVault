import asyncio
from functools import partial

import numpy as np
from fastapi import APIRouter, HTTPException, Request, UploadFile

from app.api.models import SearchResponse, SearchResult
from app.embedding.config import CLIP_BASE
from app.embedding.registry import registry
from app.vector_store import qdrant as qdrant_pipeline
from app.worker.tasks import find_similar_by_image

router = APIRouter()


@router.get("/search", response_model=SearchResponse)
async def search_images(
    request: Request, query: str, top_k: int = 10
) -> SearchResponse:
    """
    Search the image index using a natural language query.

    Embeds the query text with CLIP, retrieves the top-K most similar images
    from Qdrant by cosine similarity, and returns them ordered by score descending.

    Args:
        request (Request): FastAPI request object — used to access app.state models.
        query (str): Natural language search string, e.g. "a dog on a beach".
        top_k (int): Number of results to return. Defaults to 10, max 50.

    Returns:
        SearchResponse: The original query, list of matching images with scores, and total count.
    """
    if not query.strip():
        raise HTTPException(status_code=422, detail="Query cannot be empty.")

    top_k = min(top_k, 50)  # cap to prevent runaway requests

    state = request.app.state
    query_vector = registry.get_active().embed_text(query)
    raw_results = qdrant_pipeline.search(
        state.qdrant_client, CLIP_BASE, query_vector, top_k=top_k
    )

    results = [
        SearchResult(
            id=r["id"],
            filename=r["filename"],
            image_path=r["image_path"],
            score=round(r["score"], 4),
            timestamp_s=r.get("timestamp_s"),
        )
        for r in raw_results
    ]

    return SearchResponse(query=query, results=results, total=len(results))

@router.get("/similar/{asset_id}", response_model=SearchResponse)
async def similar_images(
    request: Request, asset_id: int, top_k: int = 10
) -> SearchResponse:
    """
    Return the top-K most visually similar images to a given asset.

    Retrieves the asset's stored CLIP vector from Qdrant and uses it as the
    search query — no re-embedding needed. The asset itself is excluded from results.

    Args:
        request (Request): FastAPI request object — used to access app.state.qdrant_client.
        asset_id (int): Qdrant point ID of the reference asset.
        top_k (int): Number of similar images to return. Defaults to 10, max 50.

    Returns:
        SearchResponse: Similar images ordered by cosine similarity score descending.

    Raises:
        HTTPException: 404 if asset_id does not exist in the collection.
    """
    top_k = min(top_k, 50)
    state = request.app.state

    points = state.qdrant_client.retrieve(
        collection_name=CLIP_BASE.collection_name,
        ids=[asset_id],
        with_vectors=True,
    )
    if not points:
        raise HTTPException(status_code=404, detail=f"Asset {asset_id} not found.")

    query_vector = np.array(points[0].vector, dtype=np.float32)
    raw_results = qdrant_pipeline.search(
        state.qdrant_client, CLIP_BASE, query_vector, top_k=top_k + 1
    )

    results = [
        SearchResult(
            id=r["id"],
            filename=r["filename"],
            image_path=r["image_path"],
            score=round(r["score"], 4),
            timestamp_s=r.get("timestamp_s"),
        )
        for r in raw_results
        if r["id"] != asset_id
    ][:top_k]

    return SearchResponse(query=f"similar:{asset_id}", results=results, total=len(results))


@router.post("/similar/upload", response_model=SearchResponse)
async def similar_by_upload(file: UploadFile, top_k: int = 10) -> SearchResponse:
    """
    Find images similar to an uploaded image using CLIP embedding.

    Dispatches find_similar_by_image to the Celery worker and blocks until the
    result is ready (timeout 10s). Blocking is intentional — the caller is waiting
    for results and a job-ID-and-poll pattern would add round trips with no benefit.
    The uploaded image is never indexed.

    Args:
        file (UploadFile): Image file to use as the similarity query.
        top_k (int): Number of similar assets to return. Defaults to 10, max 50.

    Returns:
        SearchResponse: Top-K similar assets ordered by cosine similarity.
    """
    top_k = min(top_k, 50)
    image_bytes = await file.read()
    loop = asyncio.get_event_loop()
    try:
        task = find_similar_by_image.delay(image_bytes, top_k)
        raw_results = await loop.run_in_executor(None, partial(task.get, timeout=10))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Similar search failed: {exc}")

    results = [
        SearchResult(
            id=r["id"],
            filename=r["filename"],
            image_path=r["image_path"],
            score=round(r["score"], 4),
            timestamp_s=r.get("timestamp_s"),
        )
        for r in raw_results
    ]
    return SearchResponse(query="similar:upload", results=results, total=len(results))
