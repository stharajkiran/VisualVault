"""
Tests for app/core/drift.py — compute_baseline and compute_drift.

All tests use synthetic numpy vectors. No Qdrant connection, no CLIP model,
no real images required.
"""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from app.core.drift import compute_baseline, compute_drift

# ── compute_baseline ───────────────────────────────────────────────────────────


def test_compute_baseline_shape(tmp_path):
    """Baseline has same dimensionality as the input vectors."""
    vecs = np.random.rand(5, 512).astype(np.float32)
    npy = tmp_path / "embeddings.npy"
    np.save(npy, vecs)
    with patch("app.core.drift._EMBEDDINGS_PATH", npy):
        baseline = compute_baseline(MagicMock(), MagicMock())
    assert baseline.shape == (512,)


def test_compute_baseline_value(tmp_path):
    """Baseline equals the unweighted mean of all indexed vectors."""
    vecs = np.array(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0],
        ],
        dtype=np.float32,
    )
    npy = tmp_path / "embeddings.npy"
    np.save(npy, vecs)
    with patch("app.core.drift._EMBEDDINGS_PATH", npy):
        baseline = compute_baseline(MagicMock(), MagicMock())
    expected = np.array([2 / 3, 2 / 3], dtype=np.float32)
    np.testing.assert_allclose(baseline, expected, atol=1e-6)


def test_compute_baseline_reads_file_not_qdrant(tmp_path):
    """compute_baseline loads from the numpy file — Qdrant client is never called."""
    vecs = np.random.rand(10, 512).astype(np.float32)
    npy = tmp_path / "embeddings.npy"
    np.save(npy, vecs)
    client = MagicMock()
    with patch("app.core.drift._EMBEDDINGS_PATH", npy):
        baseline = compute_baseline(client, MagicMock())
    client.scroll.assert_not_called()
    np.testing.assert_allclose(baseline, vecs.mean(axis=0), atol=1e-6)


# ── compute_drift ──────────────────────────────────────────────────────────────


def test_compute_drift_identical_returns_zero():
    """Batch with the same direction as the baseline has drift score 0."""
    baseline = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    batch = [np.array([1.0, 0.0, 0.0], dtype=np.float32)]
    assert compute_drift(batch, baseline) == pytest.approx(0.0, abs=1e-6)


def test_compute_drift_orthogonal_returns_one():
    """Batch orthogonal to baseline has cosine distance 1.0."""
    baseline = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    batch = [np.array([0.0, 1.0, 0.0], dtype=np.float32)]
    assert compute_drift(batch, baseline) == pytest.approx(1.0, abs=1e-6)


def test_compute_drift_uses_batch_mean():
    """Drift is computed against the mean of the batch, not individual vectors."""
    baseline = np.array([1.0, 0.0], dtype=np.float32)
    # Two opposite vectors — their mean points in the baseline direction
    batch = [
        np.array([1.0, 1.0], dtype=np.float32),
        np.array([1.0, -1.0], dtype=np.float32),
    ]
    # Mean is [1.0, 0.0] — same direction as baseline → drift ≈ 0
    assert compute_drift(batch, baseline) == pytest.approx(0.0, abs=1e-6)


def test_compute_drift_returns_float():
    """Return type is a plain Python float, not a numpy scalar."""
    baseline = np.array([1.0, 0.0], dtype=np.float32)
    batch = [np.array([0.5, 0.5], dtype=np.float32)]
    result = compute_drift(batch, baseline)
    assert isinstance(result, float)
