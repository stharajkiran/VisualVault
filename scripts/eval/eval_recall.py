"""
Recall@10 evaluation against the frozen eval_pairs.json.

Supports multiple CLIP model variants via --model flag. Each model is
evaluated against its own Qdrant collection as defined in its CLIPConfig.

Usage:
    python scripts/eval/eval_recall.py                  # ViT-B/32 (default)
    python scripts/eval/eval_recall.py --model large    # ViT-L/14

Never modify eval_pairs.json. It is the frozen evaluation contract for the project.
"""

import argparse
import json
import sys
from pathlib import Path

from tqdm import tqdm

from app.embedding.config import CLIP_BASE, CLIP_LARGE, CLIPConfig
from app.embedding.provider import CLIPPyTorchProvider
from app.vector_store import qdrant as qdrant_pipeline

EVAL_PAIRS_FILE = Path(__file__).parent.parent.parent / "data" / "eval_pairs.json"
TOP_K = 10

MODEL_CHOICES: dict[str, CLIPConfig] = {
    "base": CLIP_BASE,
    "large": CLIP_LARGE,
}


def load_eval_pairs(path: Path) -> list[dict]:
    """
    Load and validate the evaluation pairs from disk.

    Args:
        path (Path): Path to eval_pairs.json.

    Returns:
        list[dict]: List of eval pair dicts with keys: query, image_id, image_path, source.
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
    provider: CLIPPyTorchProvider,
    qdrant_client,
    k: int = TOP_K,
) -> tuple[float, list[dict]]:
    """
    Compute Recall@K across all eval pairs using the given provider and collection.

    For each pair, embeds the query text with the provider, retrieves the top-K
    results from the provider's Qdrant collection, and checks whether the correct
    image_id appears in those results.

    Args:
        pairs (list[dict]): Eval pairs loaded from eval_pairs.json.
        provider (CLIPPyTorchProvider): Loaded embedding provider — determines
                                        which model and collection are used.
        qdrant_client: Connected Qdrant client.
        k (int): Number of top results to check. Default is 10.

    Returns:
        tuple[float, list[dict]]: (recall_score, misses) where recall_score is between
        0.0 and 1.0, and misses is the list of pairs where the correct image was not found.
    """
    hits = 0
    misses = []

    for pair in tqdm(pairs, unit="query"):
        query_vec = provider.embed_text(pair["query"])
        results = qdrant_pipeline.search(qdrant_client, provider.config, query_vec, top_k=k)

        returned_ids = {str(r["id"]) for r in results}
        correct_id = str(int(pair["image_id"]))

        if correct_id in returned_ids:
            hits += 1
        else:
            misses.append(pair)

    recall = hits / len(pairs)
    return recall, misses


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate Recall@10 on eval_pairs.json.")
    parser.add_argument(
        "--model",
        choices=list(MODEL_CHOICES.keys()),
        default="base",
        help="CLIP model variant to evaluate (default: base)",
    )
    args = parser.parse_args()
    config = MODEL_CHOICES[args.model]

    pairs = load_eval_pairs(EVAL_PAIRS_FILE)

    print(f"\nModel     : {config.alias} ({config.model_id})")
    print(f"Collection: {config.collection_name}")

    provider = CLIPPyTorchProvider(config)
    provider.warmup()
    qdrant_client = qdrant_pipeline.get_client()

    print(f"\nRunning Recall@{TOP_K} over {len(pairs)} queries ...\n")
    recall, misses = run_recall_at_k(pairs, provider, qdrant_client)

    print(f"\n{'='*45}")
    print(f"  Model     :  {config.alias}")
    print(f"  Recall@{TOP_K}  :  {recall:.4f}  ({recall*100:.1f}%)")
    print(f"  Hits      :  {len(pairs) - len(misses)} / {len(pairs)}")
    print(f"  Misses    :  {len(misses)}")
    print(f"{'='*45}")

    if misses:
        print("\nSample misses (first 5):")
        for m in misses[:5]:
            print(f"  [{m['image_id']}] {m['query']}")


if __name__ == "__main__":
    main()
