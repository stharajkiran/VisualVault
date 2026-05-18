"""
Seed 100 fake corrections for Block 3E testing.

Inserts rows into PostgreSQL and writes YOLO11-format .txt + .jpg files to
data/corrections/ so check_and_retrain() can be triggered without real annotations.

Run with:
    uv run python scripts/dev/seed_corrections.py
"""

import random
import shutil
from pathlib import Path

from app.db.database import write_correction

CORRECTIONS_DIR = Path("data/corrections")
IMAGES_DIR = CORRECTIONS_DIR / "images"
INDEX_DIR = Path("data/index")
N = 100

# COCO class IDs to use in fake labels
SAMPLE_CLASS_IDS = [0, 1, 2, 3, 14, 15, 16, 56, 57, 62]

# Pick a real image to copy as placeholder — any JPEG from the index works
source_images = sorted(INDEX_DIR.glob("*.jpg"))
if not source_images:
    raise SystemExit("No images found in data/index/ — index must be populated first")
source_image = source_images[0]


def main() -> None:
    CORRECTIONS_DIR.mkdir(parents=True, exist_ok=True)
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)

    for i in range(N):
        filename = f"seed_{i:04d}.jpg"
        stem = f"seed_{i:04d}"

        # Random valid YOLO11 bounding box: class cx cy w h, all normalized 0-1
        class_id = random.choice(SAMPLE_CLASS_IDS)
        cx = round(random.uniform(0.1, 0.9), 6)
        cy = round(random.uniform(0.1, 0.9), 6)
        w = round(random.uniform(0.05, 0.4), 6)
        h = round(random.uniform(0.05, 0.4), 6)

        # Write YOLO11 label file
        (CORRECTIONS_DIR / f"{stem}.txt").write_text(f"{class_id} {cx} {cy} {w} {h}")

        # Copy a real JPEG so retrain_yolo finds a valid image
        shutil.copy(source_image, IMAGES_DIR / filename)

        # Insert into PostgreSQL
        write_correction(filename, class_id, cx, cy, w, h)

    print(f"Seeded {N} corrections:")
    print(f"  .txt files → {CORRECTIONS_DIR}")
    print(f"  .jpg files → {IMAGES_DIR}")
    print(f"  PostgreSQL rows → corrections table")
    print(f"\nReady to trigger: celery -A app.worker.celery_app.celery_app call app.worker.tasks.check_and_retrain")


if __name__ == "__main__":
    main()
