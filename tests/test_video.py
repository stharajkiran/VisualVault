"""
Tests for video keyframe extraction — Block 3A.

Pure-logic tests cover timestamp formula and filename format with no external deps.
Endpoint tests use a minimal FastAPI app with only the video router to avoid
triggering the full lifespan (model loading, Qdrant connection).
Task tests cover ingest_video worker logic with ffmpeg and ingest_image mocked out.
"""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, call, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.api.routes.video import ALLOWED_VIDEO_TYPES, router
from app.worker.tasks import ingest_video

# Minimal app — video router only, no lifespan, no model loading
_app = FastAPI()
_app.include_router(router)
client = TestClient(_app)


# ── Pure-logic tests ───────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "idx,expected_ts",
    [
        (0, 0),
        (1, 2),
        (5, 10),
        (14, 28),
    ],
)
def test_timestamp_formula(idx: int, expected_ts: int):
    """Frame index maps to correct timestamp: idx * 2 seconds."""
    assert idx * 2 == expected_ts


@pytest.mark.parametrize(
    "stem,idx,expected",
    [
        ("myvideo", 0, "myvideo_t0s.jpg"),
        ("myvideo", 3, "myvideo_t6s.jpg"),
        ("clip-demo", 1, "clip-demo_t2s.jpg"),
    ],
)
def test_frame_filename_format(stem: str, idx: int, expected: str):
    """Frame filename encodes video stem and timestamp correctly."""
    timestamp_s = idx * 2
    assert f"{stem}_t{timestamp_s}s.jpg" == expected


def test_frame_glob_sorted():
    """Frames globbed from tmpdir are in ascending timestamp order."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmppath = Path(tmpdir)
        for n in [3, 1, 2]:
            (tmppath / f"frame_{n:04d}.jpg").write_bytes(b"fake")

        frame_files = sorted(tmppath.glob("frame_*.jpg"))
        assert [f.name for f in frame_files] == [
            "frame_0001.jpg",
            "frame_0002.jpg",
            "frame_0003.jpg",
        ]


def test_allowed_video_types_contains_common_formats():
    """ALLOWED_VIDEO_TYPES covers MP4, MOV, AVI, and WebM."""
    assert ALLOWED_VIDEO_TYPES == {
        "video/mp4",
        "video/quicktime",
        "video/x-msvideo",
        "video/webm",
    }


# ── Endpoint tests ─────────────────────────────────────────────────────────────


def test_upload_video_rejects_image_content_type():
    """POST /upload/video with image/jpeg returns 422."""
    response = client.post(
        "/upload/video",
        files={"file": ("photo.jpg", b"fake-image-bytes", "image/jpeg")},
    )
    assert response.status_code == 422
    assert "Unsupported file type" in response.json()["detail"]


def test_upload_video_rejects_missing_file():
    """POST /upload/video with no file returns 422."""
    response = client.post("/upload/video")
    assert response.status_code == 422


def test_upload_video_dispatches_single_task():
    """POST /upload/video dispatches one ingest_video task and returns a single job_id."""
    mock_result = MagicMock()
    mock_result.id = "test-job-id"

    with patch("app.api.routes.video.celery_app") as mock_celery:
        mock_celery.send_task.return_value = mock_result

        response = client.post(
            "/upload/video",
            files={"file": ("myvideo.mp4", b"fake-video-bytes", "video/mp4")},
        )

    assert response.status_code == 200
    body = response.json()
    assert body["video_filename"] == "myvideo.mp4"
    assert body["job_id"] == "test-job-id"

    mock_celery.send_task.assert_called_once_with(
        "app.worker.tasks.ingest_video",
        args=["myvideo.mp4", b"fake-video-bytes"],
    )


# ── Task tests ─────────────────────────────────────────────────────────────────


def test_ingest_video_returns_all_frames(tmp_path):
    """ingest_video processes every extracted frame and returns results for each."""
    for i in range(1, 4):
        (tmp_path / f"frame_{i:04d}.jpg").write_bytes(b"fake-jpeg")

    fake_result = {"filename": "x", "tags": [], "caption": "test"}

    with (
        patch("app.worker.tasks.ffmpeg") as mock_ffmpeg,
        patch("app.worker.tasks.tempfile.TemporaryDirectory") as mock_tmpdir,
        patch("app.worker.tasks.ingest_image") as mock_ingest,
    ):
        mock_tmpdir.return_value.__enter__.return_value = str(tmp_path)
        mock_ffmpeg.input.return_value.filter.return_value.output.return_value.run.return_value = (
            None
        )
        mock_ingest.return_value = fake_result

        result = ingest_video("myvideo.mp4", b"fake-video-bytes")

    assert result["video_filename"] == "myvideo.mp4"
    assert result["frame_count"] == 3
    assert len(result["frames"]) == 3


def test_ingest_video_passes_correct_timestamps(tmp_path):
    """ingest_video calls ingest_image with the correct filename and timestamp_s per frame."""
    for i in range(1, 4):
        (tmp_path / f"frame_{i:04d}.jpg").write_bytes(b"fake-jpeg")

    fake_result = {"filename": "x", "tags": [], "caption": "test"}

    with (
        patch("app.worker.tasks.ffmpeg") as mock_ffmpeg,
        patch("app.worker.tasks.tempfile.TemporaryDirectory") as mock_tmpdir,
        patch("app.worker.tasks.ingest_image") as mock_ingest,
    ):
        mock_tmpdir.return_value.__enter__.return_value = str(tmp_path)
        mock_ffmpeg.input.return_value.filter.return_value.output.return_value.run.return_value = (
            None
        )
        mock_ingest.return_value = fake_result

        ingest_video("myvideo.mp4", b"fake-video-bytes")

    assert mock_ingest.call_args_list == [
        call("myvideo_t0s.jpg", b"fake-jpeg", 0.0),
        call("myvideo_t2s.jpg", b"fake-jpeg", 2.0),
        call("myvideo_t4s.jpg", b"fake-jpeg", 4.0),
    ]
