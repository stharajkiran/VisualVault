"""
Index all images in data/index/ into Qdrant.

Supports multiple CLIP model variants via --model flag. Each model writes
to its own Qdrant collection as defined in its CLIPConfig.

Usage:
    python scripts/data/index_images.py                  # ViT-B/32 (default)
    python scripts/data/index_images.py --model large    # ViT-L/14
"""

import argparse
import logging
import sys
import time
from PIL import Image
from tqdm import tqdm

from app.embedding.config import CLIP_BASE, CLIP_LARGE, CLIPConfig
from app.embedding.provider import CLIPPyTorchProvider
from app.vector_store import qdrant as qdrant_pipeline
from utils.find_root import find_project_root

PROJECT_ROOT = find_project_root()
INDEX_DIR = PROJECT_ROOT / "data" / "index"

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logging.getLogger("httpx").setLevel(logging.WARNING)

MODEL_CHOICES: dict[str, CLIPConfig] = {
    "base": CLIP_BASE,
    "large": CLIP_LARGE,
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Index images into Qdrant.")
    parser.add_argument(
        "--model",
        choices=list(MODEL_CHOICES.keys()),
        default="base",
        help="CLIP model variant to use (default: base)",
    )
    args = parser.parse_args()
    config = MODEL_CHOICES[args.model]

    images = sorted(INDEX_DIR.glob("*.jpg"))
    if not images:
        logger.error(f"No images found in {INDEX_DIR}. Run split_data.py first.")
        sys.exit(1)

    logger.info(f"Model    : {config.alias} ({config.model_id})")
    logger.info(f"Collection: {config.collection_name} ({config.embedding_dim}-dim)")
    logger.info(f"Images   : {len(images)}\n")

    client = qdrant_pipeline.get_client()
    qdrant_pipeline.create_collection(client, config)

    provider = CLIPPyTorchProvider(config)
    provider.warmup()

    t_start = time.perf_counter()
    errors = 0

    for image_path in tqdm(images, unit="img"):
        image_id = int(image_path.stem)
        try:
            img = Image.open(image_path).convert("RGB")
            embedding = provider.embed_image(img)
            qdrant_pipeline.upsert(
                client=client,
                config=config,
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
    logger.info(
        f"\nDone in {total_sec:.1f}s — {indexed}/{len(images)} images indexed "
        f"({indexed / total_sec:.1f} img/s)"
    )
    count = client.count(collection_name=config.collection_name).count
    logger.info(f"Qdrant collection '{config.collection_name}' count: {count}")


if __name__ == "__main__":
    main()
