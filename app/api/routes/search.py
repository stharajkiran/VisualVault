from fastapi import APIRouter, HTTPException, Request

from app.api.models import SearchResponse, SearchResult
from app.pipeline import clip, qdrant as qdrant_pipeline

router = APIRouter()


@router.get("/search", response_model=SearchResponse)
async def search_images(request: Request, query: str, top_k: int = 10) -> SearchResponse:
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
    query_vector = clip.embed_text(query, state.clip_model, state.clip_processor)
    raw_results = qdrant_pipeline.search(state.qdrant_client, query_vector, top_k=top_k)

    results = [
        SearchResult(
            filename=r["filename"],
            image_path=r["image_path"],
            score=round(r["score"], 4),
        )
        for r in raw_results
    ]

    return SearchResponse(query=query, results=results, total=len(results))
