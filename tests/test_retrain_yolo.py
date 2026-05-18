"""Tests for scripts/model/retrain_yolo.py — _collect_pairs and _write_dataset_yaml."""

import tempfile
from pathlib import Path

import yaml

from scripts.model.retrain_yolo import _collect_pairs, _write_dataset_yaml


# ── _collect_pairs ─────────────────────────────────────────────────────────────

def test_collect_pairs_returns_matched_files(tmp_path, monkeypatch):
    """Returns one pair when both .txt and .jpg exist with matching stem."""
    corrections_dir = tmp_path / "corrections"
    images_dir = corrections_dir / "images"
    corrections_dir.mkdir()
    images_dir.mkdir()

    (corrections_dir / "frame_001.txt").write_text("0 0.5 0.5 0.3 0.3")
    (images_dir / "frame_001.jpg").write_bytes(b"fake-image")

    # _collect_pairs reads module-level CORRECTIONS_DIR and IMAGES_DIR constants.
    # monkeypatch redirects them to tmp_path so the test never touches data/corrections/.
    monkeypatch.setattr("scripts.model.retrain_yolo.CORRECTIONS_DIR", corrections_dir)
    monkeypatch.setattr("scripts.model.retrain_yolo.IMAGES_DIR", images_dir)

    pairs = _collect_pairs()
    assert len(pairs) == 1
    assert pairs[0][0].name == "frame_001.jpg"
    assert pairs[0][1].name == "frame_001.txt"


def test_collect_pairs_skips_missing_image(tmp_path, monkeypatch):
    """Skips a .txt file when no matching image exists in images/."""
    corrections_dir = tmp_path / "corrections"
    images_dir = corrections_dir / "images"
    corrections_dir.mkdir()
    images_dir.mkdir()

    (corrections_dir / "frame_002.txt").write_text("1 0.4 0.4 0.2 0.2")

    monkeypatch.setattr("scripts.model.retrain_yolo.CORRECTIONS_DIR", corrections_dir)
    monkeypatch.setattr("scripts.model.retrain_yolo.IMAGES_DIR", images_dir)

    pairs = _collect_pairs()
    assert pairs == []


def test_collect_pairs_returns_multiple(tmp_path, monkeypatch):
    """Returns all pairs when multiple matched files exist."""
    corrections_dir = tmp_path / "corrections"
    images_dir = corrections_dir / "images"
    corrections_dir.mkdir()
    images_dir.mkdir()

    for i in range(3):
        (corrections_dir / f"frame_{i:03d}.txt").write_text(f"{i} 0.5 0.5 0.3 0.3")
        (images_dir / f"frame_{i:03d}.jpg").write_bytes(b"fake")

    monkeypatch.setattr("scripts.model.retrain_yolo.CORRECTIONS_DIR", corrections_dir)
    monkeypatch.setattr("scripts.model.retrain_yolo.IMAGES_DIR", images_dir)

    pairs = _collect_pairs()
    assert len(pairs) == 3


# ── _write_dataset_yaml ────────────────────────────────────────────────────────
# _mean_confidence and retrain_yolo() are not tested here — both load real YOLO
# weights and run GPU inference. Mocking them would test the mock, not the logic.
# Verified manually by running: uv run python scripts/model/retrain_yolo.py

def _make_pairs(tmp_path: Path, n: int) -> list[tuple[Path, Path]]:
    """
    Create n fake image/label pairs in tmp_path for use in dataset yaml tests.

    Plain helper rather than a fixture because each test needs a different n —
    a parametrized fixture would couple unrelated tests. Uses tmp_path (which
    already exists as a pytest-managed directory) so mkdir() never fails on a
    missing parent.
    """
    images_dir = tmp_path / "images"
    labels_dir = tmp_path / "labels"
    images_dir.mkdir()
    labels_dir.mkdir()
    pairs = []
    for i in range(n):
        img = images_dir / f"img_{i}.jpg"
        lbl = labels_dir / f"img_{i}.txt"
        img.write_bytes(b"fake")
        lbl.write_text(f"{i % 80} 0.5 0.5 0.3 0.3")
        pairs.append((img, lbl))
    return pairs


def test_write_dataset_yaml_creates_file(tmp_path):
    """dataset.yaml is written to the tmpdir."""
    pairs = _make_pairs(tmp_path, 2)
    # _write_dataset_yaml takes a tmpdir string — same interface as tempfile at runtime.
    # The with-block auto-deletes the directory after assertions.
    with tempfile.TemporaryDirectory() as tmpdir:
        yaml_path = _write_dataset_yaml(tmpdir, pairs)
        assert yaml_path.exists()


def test_write_dataset_yaml_has_required_keys(tmp_path):
    """dataset.yaml contains train, nc, and names keys."""
    pairs = _make_pairs(tmp_path, 2)
    with tempfile.TemporaryDirectory() as tmpdir:
        yaml_path = _write_dataset_yaml(tmpdir, pairs)
        config = yaml.safe_load(yaml_path.read_text())

    assert "train" in config
    assert "nc" in config
    assert "names" in config


def test_write_dataset_yaml_nc_is_80(tmp_path):
    """nc equals 80 — YOLO11n was trained on COCO 80 classes; fine-tuning must match."""
    pairs = _make_pairs(tmp_path, 1)
    with tempfile.TemporaryDirectory() as tmpdir:
        yaml_path = _write_dataset_yaml(tmpdir, pairs)
        config = yaml.safe_load(yaml_path.read_text())

    assert config["nc"] == 80


def test_write_dataset_yaml_train_txt_lists_images(tmp_path):
    """train.txt contains one path per image pair."""
    pairs = _make_pairs(tmp_path, 3)
    with tempfile.TemporaryDirectory() as tmpdir:
        yaml_path = _write_dataset_yaml(tmpdir, pairs)
        train_txt = Path(yaml_path.parent / "train.txt")
        lines = [l for l in train_txt.read_text().splitlines() if l]

    assert len(lines) == 3
