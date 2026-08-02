"""
Tests for Block 4B — live preview HTTP and WebSocket endpoints, and preview_frame task.

HTTP and WebSocket tests use a minimal FastAPI app with only the live router.
Celery task dispatch is mocked — no real worker required.
"""

from io import BytesIO
from unittest.mock import MagicMock, patch

from fastapi import FastAPI
from fastapi.testclient import TestClient
from PIL import Image

from app.api.routes.live import router

_app = FastAPI()
_app.include_router(router)
client = TestClient(_app)

_FAKE_TAGS = [
    {"label": "person", "confidence": 0.91},
    {"label": "laptop", "confidence": 0.87},
]


def _jpeg_bytes() -> bytes:
    """Return minimal valid JPEG bytes for upload tests."""
    buf = BytesIO()
    Image.new("RGB", (64, 64), color=(128, 128, 128)).save(buf, format="JPEG")
    return buf.getvalue()


def _mock_task(result: dict) -> MagicMock:
    """Build a mock Celery task whose .get() returns result."""
    task = MagicMock()
    task.get.return_value = result
    return task


# ── POST /live/frame ───────────────────────────────────────────────────────────


def test_live_frame_returns_tags():
    """Returns YOLO tags when the worker detects objects."""
    mock_task = _mock_task({"tags": _FAKE_TAGS})
    with patch("app.api.routes.live.celery_app") as mock_celery:
        mock_celery.send_task.return_value = mock_task
        response = client.post(
            "/live/frame",
            files={"file": ("frame.jpg", _jpeg_bytes(), "image/jpeg")},
        )
    assert response.status_code == 200
    mock_celery.send_task.assert_called_once_with(
        "app.worker.tasks.preview_frame",
        args=[_jpeg_bytes()],
    )
    tags = response.json()["tags"]
    assert len(tags) == 2
    assert tags[0]["label"] == "person"
    assert tags[1]["label"] == "laptop"


def test_live_frame_returns_empty_when_no_detections():
    """Returns empty tags list when YOLO finds nothing in the frame."""
    mock_task = _mock_task({"tags": []})
    with patch("app.api.routes.live.celery_app") as mock_celery:
        mock_celery.send_task.return_value = mock_task
        response = client.post(
            "/live/frame",
            files={"file": ("frame.jpg", _jpeg_bytes(), "image/jpeg")},
        )
    assert response.status_code == 200
    assert response.json()["tags"] == []


def test_live_frame_returns_500_on_timeout():
    """Returns 500 when the Celery task times out."""
    mock_task = MagicMock()
    mock_task.get.side_effect = Exception("Task timed out")
    with patch("app.api.routes.live.celery_app") as mock_celery:
        mock_celery.send_task.return_value = mock_task
        response = client.post(
            "/live/frame",
            files={"file": ("frame.jpg", _jpeg_bytes(), "image/jpeg")},
        )
    assert response.status_code == 500


# ── WebSocket /ws/live ─────────────────────────────────────────────────────────


def test_ws_live_returns_tags_for_frame():
    """WebSocket accepts bytes, dispatches task, sends back JSON tags."""
    mock_task = _mock_task({"tags": _FAKE_TAGS})
    with patch("app.api.routes.live.celery_app") as mock_celery:
        mock_celery.send_task.return_value = mock_task
        with client.websocket_connect("/ws/live") as ws:
            ws.send_bytes(_jpeg_bytes())
            data = ws.receive_json()
    assert "tags" in data
    assert data["tags"][0]["label"] == "person"


# ── preview_frame task ─────────────────────────────────────────────────────────


def test_preview_frame_returns_tag_structure():
    """preview_frame returns correct dict structure when YOLO detects objects."""
    fake_image = Image.new("RGB", (64, 64))
    buf = BytesIO()
    fake_image.save(buf, format="JPEG")

    with patch("app.worker.tasks.yolo.detect", return_value=[("person", 0.91), ("laptop", 0.87)]):
        with patch("app.worker.tasks._get_models", return_value={"yolo_model": MagicMock()}):
            with patch("app.worker.tasks.Image.open", return_value=fake_image):
                with patch("app.worker.tasks.INFERENCE_LATENCY") as mock_lat:
                    mock_lat.labels.return_value.observe = MagicMock()
                    from app.worker.tasks import preview_frame

                    result = preview_frame(buf.getvalue())

    assert "tags" in result
    assert result["tags"][0]["label"] == "person"
    assert result["tags"][0]["confidence"] == 0.91


def test_preview_frame_returns_empty_when_no_detections():
    """preview_frame returns empty tags list when YOLO finds nothing."""
    fake_image = Image.new("RGB", (64, 64))
    buf = BytesIO()
    fake_image.save(buf, format="JPEG")

    with patch("app.worker.tasks.yolo.detect", return_value=[]):
        with patch("app.worker.tasks._get_models", return_value={"yolo_model": MagicMock()}):
            with patch("app.worker.tasks.Image.open", return_value=fake_image):
                with patch("app.worker.tasks.INFERENCE_LATENCY") as mock_lat:
                    mock_lat.labels.return_value.observe = MagicMock()
                    from app.worker.tasks import preview_frame

                    result = preview_frame(buf.getvalue())

    assert result["tags"] == []
