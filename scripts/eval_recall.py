"""
Recall@10 evaluation against the frozen eval_pairs.json.

For each query in eval_pairs.json, embeds the query with CLIP, searches Qdrant
for top-10 results, and checks whether the correct image appears in those results.
Recall@10 = hits / total pairs.

Usage:
    python scripts/eval_recall.py

Never modify eval_pairs.json after running this for the first time.
The baseline number recorded here is the measurement anchor for the entire project.
"""

import json
import sys
from pathlib import Path

from tqdm import tqdm

from app.pipeline import clip, qdrant as qdrant_pipeline

EVAL_PAIRS_FILE = Path(__file__).parent.parent / "data" / "eval_pairs.json"
TOP_K = 10


def load_eval_pairs(path: Path) -> list[dict]:
    """
    Load and validate the evaluation pairs from disk.

    Args:
        path (Path): Path to eval_pairs.json.

    Returns:
        list[dict]: List of eval pair dicts, each with keys: query, image_id, image_path, source.
    """
    if not path.exists():
        print(f"ERROR: {path} not found. Run generate_eval_pairs.py first.")
        sys.exit(1)
    with open(path) as f:
        pairs = json.load(f)
    print(f"Loaded {len(pairs)} eval pairs.")
    return pairs


def run_recall_at_k(
    pairs: list[dict],
    clip_model,
    clip_processor,
    qdrant_client,
    k: int = TOP_K,
) -> tuple[float, list[dict]]:
    """
    Compute Recall@K across all eval pairs.

    For each pair, embeds the query text with CLIP, retrieves the top-K results
    from Qdrant, and checks whether the correct image_id appears in those results.

    Args:
        pairs (list[dict]): Eval pairs loaded from eval_pairs.json.
        clip_model: Loaded CLIPModel instance.
        clip_processor: Loaded CLIPProcessor instance.
        qdrant_client: Connected Qdrant client.
        k (int): Number of top results to check. Default is 10.

    Returns:
        tuple[float, list[dict]]: (recall_score, misses) where recall_score is between
        0.0 and 1.0, and misses is the list of pairs where the correct image was not found.
    """
    hits = 0
    misses = []

    for pair in tqdm(pairs, unit="query"):
        query_vec = clip.embed_text(pair["query"], clip_model, clip_processor)
        results = qdrant_pipeline.search(qdrant_client, query_vec, top_k=k)

        returned_ids = {str(r["id"]) for r in results}
        correct_id = str(int(pair["image_id"]))  # strip leading zeros for comparison

        if correct_id in returned_ids:
            hits += 1
        else:
            misses.append(pair)

    recall = hits / len(pairs)
    return recall, misses


def main() -> None:
    pairs = load_eval_pairs(EVAL_PAIRS_FILE)

    clip_model, clip_processor = clip.load_model()
    qdrant_client = qdrant_pipeline.get_client()

    print(f"\nRunning Recall@{TOP_K} over {len(pairs)} queries ...\n")
    recall, misses = run_recall_at_k(pairs, clip_model, clip_processor, qdrant_client)

    print(f"\n{'='*45}")
    print(f"  Recall@{TOP_K}  :  {recall:.4f}  ({recall*100:.1f}%)")
    print(f"  Hits      :  {len(pairs) - len(misses)} / {len(pairs)}")
    print(f"  Misses    :  {len(misses)}")
    print(f"{'='*45}")
    print(f"\nExpected range: 0.70 – 0.78 for frozen CLIP ViT-B/32.")

    if misses:
        print(f"\nSample misses (first 5):")
        for m in misses[:5]:
            print(f"  [{m['image_id']}] {m['query']}")

    print(f"\nRecord this number in README.md as the Phase 1 Recall@10 baseline.")


if __name__ == "__main__":
    main()
