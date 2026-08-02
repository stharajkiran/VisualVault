"""
Block 1A acceptance criterion — run all three models on a single image.

Usage:
    uv run python scripts/run_pipeline.py assets/test_img.jpg
"""

import sys
import time
from pathlib import Path
from PIL import Image

from app.embedding import clip_pytorch as clip
from app.vision import yolo, blip


def main(image_path: str) -> None:
    path = Path(image_path)
    if not path.exists():
        print(f"ERROR: file not found — {image_path}")
        sys.exit(1)

    print(f"\n{'='*50}")
    print(f"Image: {path.name}")
    print(f"{'='*50}\n")

    # --- Load all three models once ---
    clip_model, clip_processor = clip.load_model()
    yolo_model = yolo.load_model()
    blip_model, blip_processor = blip.load_model()

    print(f"\n{'='*50}")
    print("Running pipeline ...")
    print(f"{'='*50}\n")

    img = Image.open(path).convert("RGB")
    t_total = time.perf_counter()

    # YOLO — object tags
    tags = yolo.detect(img, yolo_model)

    # BLIP-2 — natural language caption
    caption_text = blip.caption(img, blip_model, blip_processor)

    # CLIP — image embedding
    embedding = clip.embed_image(img, clip_model, clip_processor)

    total_ms = (time.perf_counter() - t_total) * 1000

    # --- Results summary ---
    print(f"\n{'='*50}")
    print("RESULTS")
    print(f"{'='*50}")
    print(f"Tags       : {[t[0] for t in tags]}")
    print(f"Confidence : {[(t[0], round(t[1], 2)) for t in tags]}")
    print(f"Caption    : {caption_text}")
    print(f"Embedding  : shape={embedding.shape}, norm={embedding @ embedding:.4f}")
    print(f"Total time : {total_ms:.0f}ms")
    print(f"{'='*50}\n")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python scripts/run_pipeline.py <image_path>")
        sys.exit(1)
    main(sys.argv[1])
