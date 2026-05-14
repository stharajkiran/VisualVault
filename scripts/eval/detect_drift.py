"""
Standalone drift detection script — Block 3B.

Connects to Qdrant, computes the index baseline, embeds a folder of images
with CLIP, and reports the cosine drift score. Updates the Prometheus gauge
so the metric is visible on the next /metrics scrape.

Run with:
    uv run python scripts/eval/detect_drift.py
    uv run python scripts/eval/detect_drift.py --folder data/ood_test/
    uv run python scripts/eval/detect_drift.py --folder data/index/ --limit 50
"""

import argparse
from pathlib import Path

from PIL import Image

from app.core.drift import compute_baseline, compute_drift
from app.core.metrics import EMBEDDING_DRIFT
from app.embedding.config import CLIP_BASE
from app.embedding.provider import CLIPPyTorchProvider
from app.vector_store import qdrant as qdrant_pipeline

ALLOWED_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute embedding drift against the Qdrant index baseline.")
    parser.add_argument("--folder", type=str, default="data/ood_test/", help="Folder of images to test (default: data/ood_test/)")
    parser.add_argument("--limit", type=int, default=None, help="Max images to embed (default: all)")
    args = parser.parse_args()

    folder = Path(args.folder)
    if not folder.is_dir():
        raise SystemExit(f"Folder not found: {folder}")

    image_paths = sorted(p for p in folder.iterdir() if p.suffix.lower() in ALLOWED_SUFFIXES)
    if args.limit:
        image_paths = image_paths[: args.limit]
    if not image_paths:
        raise SystemExit(f"No images found in {folder}")

    print(f"Loading CLIP ({CLIP_BASE.alias}) ...")
    provider = CLIPPyTorchProvider(CLIP_BASE)

    print(f"Embedding {len(image_paths)} images ...")
    batch_embeddings = [
        provider.embed_image(Image.open(p).convert("RGB"))
        for p in image_paths
    ]

    print("Computing baseline from Qdrant ...")
    client = qdrant_pipeline.get_client()
    baseline = compute_baseline(client, CLIP_BASE)
    baseline_size = client.count(collection_name=CLIP_BASE.collection_name).count

    drift_score = compute_drift(batch_embeddings, baseline)
    EMBEDDING_DRIFT.set(drift_score)

    print("\n── Drift Results ────────────────────────────")
    print(f"  Baseline size    {baseline_size} images")
    print(f"  Batch size       {len(batch_embeddings)} images")
    print(f"  Drift score      {drift_score:.6f}  (cosine distance)")
    print("  Alert threshold  0.15")
    print(f"  Status           {'⚠ DRIFT DETECTED' if drift_score > 0.15 else '✓ within threshold'}")


if __name__ == "__main__":
    main()
