"""
YOLO class-presence evaluation on held-out COCO images.

Measures how well YOLO11n detects the right object categories on images it
has never been indexed against. Uses class-presence precision/recall rather
than IoU-based mAP because VisualVault uses YOLO for semantic tagging, not
localization — a correct tag at the wrong location is still a correct tag.

Usage:
    python scripts/eval/eval_yolo_holdout.py
    python scripts/eval/eval_yolo_holdout.py --limit 100 --threshold 0.3
"""

import argparse
import json
import sys
from pathlib import Path

from PIL import Image
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from app.vision import yolo

ANNOTATIONS_FILE = PROJECT_ROOT / "data" / "raw" / "annotations" / "instances_val2017.json"
HOLDOUT_DIR = PROJECT_ROOT / "data" / "holdout"
DEFAULT_LIMIT = 200
DEFAULT_THRESHOLD = 0.25


def load_ground_truth(annotations_path: Path) -> dict[str, set[str]]:
    """
    Build a mapping from image filename to set of ground truth class names.

    Reads COCO instances_val2017.json and maps each image filename to the
    set of category names that appear in its annotations.

    Args:
        annotations_path (Path): Path to instances_val2017.json.

    Returns:
        dict[str, set[str]]: filename → set of ground truth class names.
    """
    with open(annotations_path) as f:
        coco = json.load(f)

    category_map: dict[int, str] = {c["id"]: c["name"] for c in coco["categories"]}
    image_id_to_filename: dict[int, str] = {
        img["id"]: img["file_name"] for img in coco["images"]
    }

    gt: dict[str, set[str]] = {}
    for ann in coco["annotations"]:
        filename = image_id_to_filename[ann["image_id"]]
        gt.setdefault(filename, set()).add(category_map[ann["category_id"]])

    return gt


def evaluate(limit: int, threshold: float) -> None:
    """
    Run YOLO on held-out images and print class-presence precision, recall, F1.

    Args:
        limit (int): Maximum number of holdout images to evaluate.
        threshold (float): Minimum YOLO confidence to count as a detection.

    Returns:
        None
    """
    print(f"Loading COCO ground truth from {ANNOTATIONS_FILE} ...")
    gt = load_ground_truth(ANNOTATIONS_FILE)

    holdout_images = sorted(HOLDOUT_DIR.glob("*.jpg"))[:limit]
    if not holdout_images:
        print(f"No images found in {HOLDOUT_DIR}")
        sys.exit(1)

    print(f"Loading YOLO model ...")
    model = yolo.load_model()

    precisions, recalls = [], []
    zero_gt, zero_pred = 0, 0

    for image_path in tqdm(holdout_images, desc="Evaluating", unit="img"):
        gt_labels = gt.get(image_path.name, set())
        if not gt_labels:
            zero_gt += 1
            continue

        image = Image.open(image_path).convert("RGB")
        detections = yolo.detect(image, model)
        pred_labels = {label for label, conf in detections if conf >= threshold}

        if not pred_labels:
            zero_pred += 1
            precisions.append(0.0)
            recalls.append(0.0)
            continue

        tp = len(pred_labels & gt_labels)
        precision = tp / len(pred_labels)
        recall = tp / len(gt_labels)
        precisions.append(precision)
        recalls.append(recall)

    mean_p = sum(precisions) / len(precisions) if precisions else 0.0
    mean_r = sum(recalls) / len(recalls) if recalls else 0.0
    f1 = 2 * mean_p * mean_r / (mean_p + mean_r) if (mean_p + mean_r) > 0 else 0.0

    print(f"\n{'─' * 50}")
    print(f"YOLO Holdout Evaluation — class-presence metric")
    print(f"{'─' * 50}")
    print(f"Images evaluated   : {len(holdout_images)}")
    print(f"Images with GT     : {len(holdout_images) - zero_gt}")
    print(f"Images no GT       : {zero_gt} (no COCO annotations)")
    print(f"Images no detection: {zero_pred}")
    print(f"Confidence threshold: {threshold}")
    print(f"{'─' * 50}")
    print(f"Precision          : {mean_p:.4f} ({mean_p:.1%})")
    print(f"Recall             : {mean_r:.4f} ({mean_r:.1%})")
    print(f"F1                 : {f1:.4f} ({f1:.1%})")
    print(f"{'─' * 50}")
    print("Metric: class-presence (not IoU-based mAP). Correct tag")
    print("at any location counts — appropriate for search tagging.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate YOLO on holdout images.")
    parser.add_argument("--limit", type=int, default=DEFAULT_LIMIT,
                        help=f"Number of holdout images to evaluate (default: {DEFAULT_LIMIT})")
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD,
                        help=f"YOLO confidence threshold (default: {DEFAULT_THRESHOLD})")
    args = parser.parse_args()
    evaluate(args.limit, args.threshold)


if __name__ == "__main__":
    main()
