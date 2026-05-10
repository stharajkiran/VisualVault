from pydantic import BaseModel


class TagResult(BaseModel):
    """A single object detected by YOLO with its confidence score."""

    label: str
    confidence: float


class UploadResponse(BaseModel):
    """
    Response returned by POST /upload after an image is processed.

    Args:
        filename (str): Original name of the uploaded file.
        tags (list[TagResult]): YOLO detections sorted by confidence descending.
        caption (str): BLIP-2 natural language caption.
        processing_ms (float): Total pipeline time in milliseconds.
    """

    filename: str
    tags: list[TagResult]
    caption: str
    processing_ms: float


class SearchResult(BaseModel):
    """
    A single result returned by GET /search.

    Args:
        filename (str): Image filename in the index.
        image_path (str): Relative path to the image file on disk.
        score (float): Cosine similarity score between query and image (0–1).
    """

    filename: str
    image_path: str
    score: float


class SearchResponse(BaseModel):
    """
    Full response returned by GET /search.

    Args:
        query (str): The original search string submitted by the client.
        results (list[SearchResult]): Top-K matching images sorted by score descending.
        total (int): Number of results returned.
    """

    query: str
    results: list[SearchResult]
    total: int
