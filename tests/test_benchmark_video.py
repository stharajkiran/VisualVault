"""
Tests for benchmark_video helpers — generate_test_video and extract_frames.

These are integration tests: they invoke real FFmpeg. They use a 4-second
synthetic video so each test completes in well under a second.
"""

import shutil
from pathlib import Path

import pytest

from scripts.eval.benchmark_video import extract_frames, generate_test_video

pytestmark = pytest.mark.skipif(
    shutil.which("ffmpeg") is None,
    reason="FFmpeg not on PATH — skipping video integration tests",
)


@pytest.fixture(scope="module")
def synthetic_video(tmp_path_factory) -> Path:
    """4-second 320x240 test video shared across all tests in this module."""
    tmpdir = tmp_path_factory.mktemp("video")
    video_path = tmpdir / "test.mp4"
    generate_test_video(video_path, duration_s=4, resolution="320x240")
    return video_path


def test_generate_test_video_creates_file(synthetic_video):
    """generate_test_video writes a non-empty .mp4 file."""
    assert synthetic_video.exists()
    assert synthetic_video.stat().st_size > 0


def test_extract_frames_count(synthetic_video, tmp_path):
    """4s video at 0.5fps should produce exactly 2 frames."""
    frames = extract_frames(synthetic_video, tmp_path)
    assert len(frames) == 2


def test_extract_frames_naming(synthetic_video, tmp_path):
    """Extracted frames follow FFmpeg's frame_NNNN.jpg naming convention."""
    frames = extract_frames(synthetic_video, tmp_path)
    assert frames[0].name == "frame_0001.jpg"
    assert frames[1].name == "frame_0002.jpg"


def test_extract_frames_sorted(synthetic_video, tmp_path):
    """Returned frame list is in ascending order."""
    frames = extract_frames(synthetic_video, tmp_path)
    names = [f.name for f in frames]
    assert names == sorted(names)
