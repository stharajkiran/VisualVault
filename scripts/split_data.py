"""
One-time data split: 4,000 images → data/index/, 1,000 images → data/holdout/.

Run once before indexing. Never run again — the split is the measurement anchor
for every accuracy claim in the project.

Usage:
    python scripts/split_data.py
"""

import random
import shutil
import sys
from pathlib import Path

SEED = 42

INDEX_SIZE = 4000
HOLDOUT_SIZE = 1000

RAW_DIR = Path(__file__).parent.parent / "data" / "raw" / "val2017"
INDEX_DIR = Path(__file__).parent.parent / "data" / "index"
HOLDOUT_DIR = Path(__file__).parent.parent / "data" / "holdout"


def main() -> None:
    images = sorted(RAW_DIR.glob("*.jpg"))
    if len(images) != 5000:
        print(f"ERROR: expected 5000 images in {RAW_DIR}, found {len(images)}")
        sys.exit(1)

    # Guard — refuse to overwrite an existing split
    if any(INDEX_DIR.iterdir()) or any(HOLDOUT_DIR.iterdir()):
        print("ERROR: data/index/ or data/holdout/ already contains files.")
        print("The split must only be created once. Delete the folders manually if you need to reset.")
        sys.exit(1)

    random.seed(SEED)
    shuffled = images[:]
    random.shuffle(shuffled)

    index_images = shuffled[:INDEX_SIZE]
    holdout_images = shuffled[INDEX_SIZE:INDEX_SIZE + HOLDOUT_SIZE]

    print(f"Copying {INDEX_SIZE} images to data/index/ ...")
    for img in index_images:
        shutil.copy2(img, INDEX_DIR / img.name)

    print(f"Copying {HOLDOUT_SIZE} images to data/holdout/ ...")
    for img in holdout_images:
        shutil.copy2(img, HOLDOUT_DIR / img.name)

    print(f"\nDone.")
    print(f"  data/index/   : {len(list(INDEX_DIR.glob('*.jpg')))} images")
    print(f"  data/holdout/ : {len(list(HOLDOUT_DIR.glob('*.jpg')))} images")
    print(f"\nCommit this split with git before indexing anything.")


if __name__ == "__main__":
    main()
