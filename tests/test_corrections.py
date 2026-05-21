"""
Tests for Block 3D — corrections webhook and Label Studio helper.

Webhook tests use a minimal FastAPI app with only the corrections router.
Label Studio tests mock httpx to avoid requiring a real server.
File-write tests use tmp_path to avoid touching data/corrections/.
"""

from unittest.mock import MagicMock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.api.routes.corrections import router

_app = FastAPI()
_app.include_router(router)
client = TestClient(_app)


def _label_studio_payload(labels: list[str], filename: str = "test.jpg") -> dict:
    """Build a minimal Label Studio annotation webhook payload."""
    return {
        "task": {"data": {"image": f"/data/upload/{filename}"}},
        "annotation": {
            "result": [
                {
                    "type": "rectanglelabels",
                    "value": {
                        "x": 10.0, "y": 20.0,
                        "width": 30.0, "height": 40.0,
                        "rectanglelabels": [label],
                    },
                }
                for label in labels
            ]
        },
    }


# ── Webhook endpoint tests ─────────────────────────────────────────────────────

def test_submit_correction_writes_yolo_file(tmp_path):
    """Valid payload writes a YOLO11-format .txt file to corrections dir."""
    with patch("app.api.routes.corrections.CORRECTIONS_DIR", tmp_path):
        response = client.post(
            "/corrections/submit",
            json=_label_studio_payload(["person"], filename="frame_001.jpg"),
        )

    assert response.status_code == 200
    body = response.json()
    assert body["corrections"] == 1
    assert body["filename"] == "frame_001"

    txt = (tmp_path / "frame_001.txt").read_text()
    parts = txt.strip().split()
    assert len(parts) == 5
    assert parts[0] == "0"  # person = class 0
    assert float(parts[1]) == pytest.approx(0.25, abs=1e-4)  # cx = (10 + 15) / 100
    assert float(parts[2]) == pytest.approx(0.40, abs=1e-4)  # cy = (20 + 20) / 100
    assert float(parts[3]) == pytest.approx(0.30, abs=1e-4)  # w = 30 / 100
    assert float(parts[4]) == pytest.approx(0.40, abs=1e-4)  # h = 40 / 100


def test_submit_correction_increments_counter(tmp_path):
    """Successful correction increments the Prometheus counter."""
    with patch("app.api.routes.corrections.CORRECTIONS_DIR", tmp_path):
        with patch("app.api.routes.corrections.CORRECTIONS_TOTAL") as mock_counter:
            client.post(
                "/corrections/submit",
                json=_label_studio_payload(["dog"], filename="img.jpg"),
            )
    mock_counter.inc.assert_called_once()


def test_submit_correction_skips_unknown_label(tmp_path):
    """Label not in COCO 80 is skipped — no file written, corrections=0."""
    with patch("app.api.routes.corrections.CORRECTIONS_DIR", tmp_path):
        response = client.post(
            "/corrections/submit",
            json=_label_studio_payload(["spaceship"], filename="ufo.jpg"),
        )

    assert response.status_code == 200
    assert response.json()["corrections"] == 0
    assert not (tmp_path / "ufo.txt").exists()


def test_submit_correction_empty_results(tmp_path):
    """Payload with no annotation results returns corrections=0."""
    payload = {
        "task": {"data": {"image": "/data/upload/empty.jpg"}},
        "annotation": {"result": []},
    }
    with patch("app.api.routes.corrections.CORRECTIONS_DIR", tmp_path):
        response = client.post("/corrections/submit", json=payload)

    assert response.status_code == 200
    assert response.json()["corrections"] == 0


# ── Label Studio helper tests ──────────────────────────────────────────────────

def test_push_to_label_studio_no_ops_without_config():
    """Returns False immediately when env vars are not set."""
    with patch("app.core.label_studio.LABEL_STUDIO_URL", ""):
        with patch("app.core.label_studio.LABEL_STUDIO_API_KEY", ""):
            from app.core.label_studio import push_to_label_studio
            result = push_to_label_studio("test.jpg", b"bytes", [("person", 0.3)])
    assert result is False


def test_push_to_label_studio_returns_true_on_success():
    """Returns True when the Label Studio SDK call succeeds."""
    with patch("app.core.label_studio.LABEL_STUDIO_URL", "http://localhost:8080"):
        with patch("app.core.label_studio.LABEL_STUDIO_API_KEY", "test-token"):
            with patch("app.core.label_studio.LabelStudio") as mock_ls_cls:
                mock_ls_cls.return_value.projects.import_tasks.return_value = None
                from app.core.label_studio import push_to_label_studio
                result = push_to_label_studio("test.jpg", b"bytes", [("person", 0.3)])
    assert result is True


def test_push_to_label_studio_passes_credentials_to_sdk():
    """SDK is initialized with the configured URL and API key."""
    with patch("app.core.label_studio.LABEL_STUDIO_URL", "http://localhost:8080"):
        with patch("app.core.label_studio.LABEL_STUDIO_API_KEY", "test-token"):
            with patch("app.core.label_studio.LabelStudio") as mock_ls_cls:
                mock_ls_cls.return_value.projects.import_tasks.return_value = None
                from app.core.label_studio import push_to_label_studio
                push_to_label_studio("test.jpg", b"bytes", [("person", 0.3)])
    mock_ls_cls.assert_called_once_with(base_url="http://localhost:8080", api_key="test-token")


def test_push_to_label_studio_sends_task_payload():
    """import_tasks is called with the correct project ID and task structure."""
    with patch("app.core.label_studio.LABEL_STUDIO_URL", "http://localhost:8080"):
        with patch("app.core.label_studio.LABEL_STUDIO_API_KEY", "test-token"):
            with patch("app.core.label_studio.LABEL_STUDIO_PROJECT_ID", 42):
                with patch("app.core.label_studio.LabelStudio") as mock_ls_cls:
                    mock_import = mock_ls_cls.return_value.projects.import_tasks
                    mock_import.return_value = None
                    from app.core.label_studio import push_to_label_studio
                    push_to_label_studio("img.jpg", b"bytes", [("dog", 0.4)])
    call_kwargs = mock_import.call_args
    assert call_kwargs.kwargs["id"] == 42
    task = call_kwargs.kwargs["request"][0]
    assert task["data"]["filename"] == "img.jpg"
    assert task["predictions"][0]["result"][0]["value"]["rectanglelabels"] == ["dog"]


def test_push_to_label_studio_returns_false_on_exception():
    """Returns False without raising when the SDK raises an exception."""
    with patch("app.core.label_studio.LABEL_STUDIO_URL", "http://localhost:8080"):
        with patch("app.core.label_studio.LABEL_STUDIO_API_KEY", "test-token"):
            with patch("app.core.label_studio.LabelStudio") as mock_ls_cls:
                mock_ls_cls.return_value.projects.import_tasks.side_effect = Exception("timeout")
                from app.core.label_studio import push_to_label_studio
                result = push_to_label_studio("test.jpg", b"bytes", [("person", 0.3)])
    assert result is False
