"""
Export all CLIP embeddings from Qdrant to numpy files for HF Spaces demo.

Produces two files in spaces/:
  - embeddings.npy  : shape (N, 512), float32 — one CLIP vector per image
  - image_ids.npy   : shape (N,), int64  — COCO image ID for each row

The HF Spaces app loads these files and does nearest-neighbour search
with numpy dot product instead of Qdrant.

Usage:
    python scripts/export_embeddings.py
"""

import sys
from pathlib import Path

import numpy as np
from qdrant_client import QdrantClient
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.vector_store.qdrant import COLLECTION_NAME

OUTPUT_DIR = Path(__file__).parent.parent / "data" / "embeddings"
BATCH_SIZE = 256


def export(client: QdrantClient) -> None:
    """
    Scroll through all Qdrant points and export vectors + IDs to numpy files.

    Qdrant's scroll API fetches points in batches using a cursor — it returns
    a page of results and a cursor to the next page. We keep fetching until
    the cursor is None, meaning we've reached the end.

    Args:
        client (QdrantClient): Connected Qdrant client instance.

    Returns:
        None. Writes embeddings.npy and image_ids.npy to the spaces/ directory.
    """
    OUTPUT_DIR.mkdir(exist_ok=True)

    embeddings = []
    image_ids = []
    offset = None

    total = client.count(collection_name=COLLECTION_NAME).count
    print(f"Exporting {total} vectors from Qdrant ...")

    with tqdm(total=total, unit="vec") as pbar:
        while True:
            results, next_offset = client.scroll(
                collection_name=COLLECTION_NAME,
                limit=BATCH_SIZE,
                offset=offset,
                with_vectors=True,
                with_payload=False,
            )

            if not results:
                break

            for point in results:
                embeddings.append(point.vector)
                image_ids.append(point.id)

            pbar.update(len(results))
            offset = next_offset

            if next_offset is None:
                break

    embeddings_arr = np.array(embeddings, dtype=np.float32)
    image_ids_arr = np.array(image_ids, dtype=np.int64)

    np.save(OUTPUT_DIR / "embeddings.npy", embeddings_arr)
    np.save(OUTPUT_DIR / "image_ids.npy", image_ids_arr)

    print(f"\nSaved to {OUTPUT_DIR}/")
    print(f"  embeddings.npy : {embeddings_arr.shape}  ({embeddings_arr.nbytes / 1e6:.1f} MB)")
    print(f"  image_ids.npy  : {image_ids_arr.shape}")


def main() -> None:
    client = QdrantClient(host="localhost", port=6333)
    export(client)


if __name__ == "__main__":
    main()
