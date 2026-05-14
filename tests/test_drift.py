"""
Tests for app/core/drift.py — compute_baseline and compute_drift.

All tests use synthetic numpy vectors. No Qdrant connection, no CLIP model,
no real images required.
"""

from unittest.mock import MagicMock

import numpy as np
import pytest

from app.core.drift import compute_baseline, compute_drift

# ── compute_baseline ───────────────────────────────────────────────────────────


def _make_qdrant_client(vectors: list[np.ndarray]) -> MagicMock:
    """Return a mock Qdrant client whose scroll() yields all vectors in one page."""
    points = [MagicMock(vector=v.tolist()) for v in vectors]
    client = MagicMock()
    client.scroll.return_value = (points, None)  # None offset = last page
    return client


def test_compute_baseline_shape():
    """Baseline has same dimensionality as the input vectors."""
    vecs = [np.random.rand(512).astype(np.float32) for _ in range(5)]
    client = _make_qdrant_client(vecs)
    baseline = compute_baseline(client, MagicMock())
    assert baseline.shape == (512,)


def test_compute_baseline_value():
    """Baseline equals the unweighted mean of all indexed vectors."""
    vecs = [
        np.array([1.0, 0.0], dtype=np.float32),
        np.array([0.0, 1.0], dtype=np.float32),
        np.array([1.0, 1.0], dtype=np.float32),
    ]
    client = _make_qdrant_client(vecs)
    baseline = compute_baseline(client, MagicMock())
    expected = np.array([2 / 3, 2 / 3], dtype=np.float32)
    np.testing.assert_allclose(baseline, expected, atol=1e-6)


def test_compute_baseline_paginates():
    """compute_baseline keeps scrolling until next_offset is None."""
    page1 = [MagicMock(vector=[1.0, 0.0])]
    page2 = [MagicMock(vector=[0.0, 1.0])]
    client = MagicMock()
    client.scroll.side_effect = [
        (page1, "cursor-1"),
        (page2, None),
    ]
    baseline = compute_baseline(client, MagicMock())
    assert client.scroll.call_count == 2
    np.testing.assert_allclose(baseline, np.array([0.5, 0.5]), atol=1e-6)


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
