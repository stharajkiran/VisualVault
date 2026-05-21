"""
Tests for Block 4E — /similar/{asset_id}, /similar/upload, and /search id field.

All Qdrant and Celery calls are mocked — no live services required.
"""

from io import BytesIO
from unittest.mock import MagicMock, patch

import numpy as np
from fastapi import FastAPI
from fastapi.testclient import TestClient
from PIL import Image

from app.api.routes.search import router

_app = FastAPI()
_app.include_router(router)
client = TestClient(_app)

_FAKE_RESULTS = [
    {
        "id": 1,
        "score": 0.95,
        "filename": "000000000001.jpg",
        "image_path": "data/index/000000000001.jpg",
    },
    {
        "id": 2,
        "score": 0.90,
        "filename": "000000000002.jpg",
        "image_path": "data/index/000000000002.jpg",
    },
]


def _jpeg_bytes() -> bytes:
    buf = BytesIO()
    Image.new("RGB", (64, 64)).save(buf, format="JPEG")
    return buf.getvalue()


def _mock_qdrant_point(asset_id: int) -> MagicMock:
    point = MagicMock()
    point.id = asset_id
    point.vector = np.random.rand(512).tolist()
    return point


def _mock_app_state(asset_id: int | None = 9) -> MagicMock:
    state = MagicMock()
    if asset_id is not None:
        state.qdrant_client.retrieve.return_value = [_mock_qdrant_point(asset_id)]
    else:
        state.qdrant_client.retrieve.return_value = []
    return state


# ── GET /search — id field regression ─────────────────────────────────────────


def test_search_includes_id_in_results():
    """Search results include id field after Block 4E model change."""
    with patch("app.api.routes.search.registry") as mock_reg:
        mock_reg.get_active.return_value.embed_text.return_value = np.zeros(512)
        with patch("app.api.routes.search.qdrant_pipeline.search", return_value=_FAKE_RESULTS):
            _app.state.qdrant_client = MagicMock()
            response = client.get("/search", params={"query": "dog"})

    assert response.status_code == 200
    assert "id" in response.json()["results"][0]


# ── GET /similar/{asset_id} ────────────────────────────────────────────────────


def test_similar_returns_results():
    """Returns similar images excluding the queried asset."""
    results_with_self = [
        {"id": 9, "score": 1.0, "filename": "self.jpg", "image_path": "x"}
    ] + _FAKE_RESULTS
    _app.state.qdrant_client = _mock_app_state(asset_id=9).qdrant_client

    with patch("app.api.routes.search.qdrant_pipeline.search", return_value=results_with_self):
        response = client.get("/similar/9")

    assert response.status_code == 200
    data = response.json()
    ids = [r["id"] for r in data["results"]]
    assert 9 not in ids
    assert data["query"] == "similar:9"


def test_similar_returns_404_when_not_found():
    """Returns 404 when asset does not exist in Qdrant."""
    _app.state.qdrant_client = _mock_app_state(asset_id=None).qdrant_client
    response = client.get("/similar/9999")
    assert response.status_code == 404


def test_similar_excludes_self():
    """The queried asset is never in its own similar results."""
    all_results = [
        {"id": 5, "score": 1.0, "filename": "self.jpg", "image_path": "x"}
    ] + _FAKE_RESULTS
    _app.state.qdrant_client = _mock_app_state(asset_id=5).qdrant_client

    with patch("app.api.routes.search.qdrant_pipeline.search", return_value=all_results):
        response = client.get("/similar/5")

    ids = [r["id"] for r in response.json()["results"]]
    assert 5 not in ids


# ── POST /similar/upload ───────────────────────────────────────────────────────


def test_similar_upload_returns_results():
    """Returns similar images when worker processes the uploaded image."""
    mock_task = MagicMock()
    mock_task.get.return_value = _FAKE_RESULTS

    with patch("app.api.routes.search.find_similar_by_image") as mock_fn:
        mock_fn.delay.return_value = mock_task
        response = client.post(
            "/similar/upload",
            files={"file": ("img.jpg", _jpeg_bytes(), "image/jpeg")},
        )

    assert response.status_code == 200
    data = response.json()
    assert data["query"] == "similar:upload"
    assert len(data["results"]) == 2
    assert data["results"][0]["id"] == 1


def test_similar_upload_returns_500_on_timeout():
    """Returns 500 when the Celery task times out."""
    mock_task = MagicMock()
    mock_task.get.side_effect = Exception("timed out")

    with patch("app.api.routes.search.find_similar_by_image") as mock_fn:
        mock_fn.delay.return_value = mock_task
        response = client.post(
            "/similar/upload",
            files={"file": ("img.jpg", _jpeg_bytes(), "image/jpeg")},
        )

    assert response.status_code == 500
