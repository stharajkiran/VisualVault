"""
Delete hash-based duplicate points from Qdrant.

Block 2A batch_ingest tests indexed COCO images with hash-based IDs instead of
COCO integer IDs, creating duplicates. COCO IDs are small integers (max ~581929).
Any point with ID > 1_000_000 is a hash-based duplicate — safe to delete.

Run with:
    python scripts/dev/cleanup_qdrant.py
"""

from qdrant_client.models import PointIdsList
from app.vector_store import qdrant as q

client = q.get_client()

result = client.scroll("visualvault", limit=10000, with_payload=False)
all_points = result[0]

hash_ids = [p.id for p in all_points if p.id > 1_000_000]
print(f"Total points in collection : {len(all_points)}")
print(f"Hash-based duplicates found: {len(hash_ids)}")

if hash_ids:
    client.delete(
        collection_name="visualvault",
        points_selector=PointIdsList(points=hash_ids),
    )
    print("Deleted.")

count_after = client.count("visualvault").count
print(f"Collection count after     : {count_after}")
