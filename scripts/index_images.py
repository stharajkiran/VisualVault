"""
Index all images in data/index/ into Qdrant.

Loads CLIP once, embeds each image, and upserts the vector + metadata into
the visualvault collection. Run after split_data.py and before any searches.

Usage:
    python scripts/index_images.py
"""

import sys
import time
from PIL import Image
from tqdm import tqdm

from app.pipeline import clip, qdrant as qdrant_pipeline
from utils.find_root import find_project_root
import logging

PROJECT_ROOT = find_project_root()
INDEX_DIR = PROJECT_ROOT / "data" / "index"

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def main() -> None:
    images = sorted(INDEX_DIR.glob("*.jpg"))
    if not images:
        logger.error(f"No images found in {INDEX_DIR}. Run split_data.py first.")
        sys.exit(1)

    logger.info(f"Found {len(images)} images to index.\n")

    # Connect to Qdrant and create collection if needed
    client = qdrant_pipeline.get_client()
    qdrant_pipeline.create_collection(client)

    # Load CLIP once — reused for all 4,000 images
    clip_model, clip_processor = clip.load_model()

    logger.info(f"\nIndexing {len(images)} images ...\n")
    t_start = time.perf_counter()
    errors = 0

    for image_path in tqdm(images, unit="img"):
        # COCO filename is zero-padded: 000000012345.jpg → ID = 12345
        image_id = int(image_path.stem)

        try:
            img = Image.open(image_path).convert("RGB")
            embedding = clip.embed_image(img, clip_model, clip_processor)
            qdrant_pipeline.upsert(
                client=client,
                image_id=image_id,
                embedding=embedding,
                payload={
                    "filename": image_path.name,
                    "image_path": str(image_path),
                },
            )
        except Exception as e:
            logger.warning(f"Failed on {image_path.name}: {e}")
            errors += 1

    total_sec = time.perf_counter() - t_start
    indexed = len(images) - errors
    logger.info(f"\nDone in {total_sec:.1f}s — {indexed}/{len(images)} images indexed "
                f"({indexed / total_sec:.1f} img/s)")

    # Verify the count in Qdrant matches what we expect
    count = client.count(collection_name=qdrant_pipeline.COLLECTION_NAME).count
    logger.info(f"Qdrant collection count: {count}")


if __name__ == "__main__":
    main()
