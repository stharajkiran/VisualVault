"""
YOLO11n fine-tuning on human-verified corrections — Block 3E.

Pairs label files from data/corrections/ with images from data/corrections/images/,
runs a short fine-tuning run on top of the base YOLO11n weights, and evaluates
both the new and baseline models on data/holdout/ using mean detection confidence
as a proxy metric (true precision/recall requires COCO ground-truth conversion).

Run manually with:
    uv run python scripts/model/retrain_yolo.py
"""

import tempfile
from pathlib import Path

import yaml
from ultralytics import YOLO

CORRECTIONS_DIR = Path("data/corrections")
IMAGES_DIR = CORRECTIONS_DIR / "images"
HOLDOUT_DIR = Path("data/holdout")
BASE_MODEL = Path("artifacts/models/yolo11n.pt")
FINETUNED_MODEL = Path("artifacts/models/yolo11n_finetuned.pt")

EPOCHS = 10
CONFIDENCE_THRESHOLD = 0.25


def _collect_pairs() -> list[tuple[Path, Path]]:
    """
    Return (image_path, label_path) pairs where both files exist.

    Args:
        None

    Returns:
        list[tuple[Path, Path]]: Valid image/label pairs for training.
    """
    pairs = []
    for label_path in sorted(CORRECTIONS_DIR.glob("*.txt")):
        image_path = IMAGES_DIR / (label_path.stem + ".jpg")
        if image_path.exists():
            pairs.append((image_path, label_path))
    return pairs


def _write_dataset_yaml(tmpdir: str, pairs: list[tuple[Path, Path]]) -> Path:
    """
    Write a YOLO dataset YAML config to a temp directory.

    Creates train.txt listing all image paths and a dataset.yaml pointing at it.
    YOLO's training API requires this file structure.

    Args:
        tmpdir (str): Temporary directory path.
        pairs (list[tuple[Path, Path]]): Image/label pairs to include.

    Returns:
        Path: Path to the written dataset.yaml file.
    """
    tmp = Path(tmpdir)

    train_txt = tmp / "train.txt"
    train_txt.write_text("\n".join(str(p.resolve()) for p, _ in pairs))

    # Copy label files next to images so YOLO can find them
    labels_dir = tmp / "labels"
    labels_dir.mkdir()
    for _, label_path in pairs:
        (labels_dir / label_path.name).write_text(label_path.read_text())

    dataset_yaml = tmp / "dataset.yaml"
    config = {
        "train": str(train_txt),
        "val": str(train_txt),
        "nc": 80,
        "names": [
            "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train",
            "truck", "boat", "traffic light", "fire hydrant", "stop sign",
            "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep",
            "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
            "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard",
            "sports ball", "kite", "baseball bat", "baseball glove", "skateboard",
            "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork",
            "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
            "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
            "couch", "potted plant", "bed", "dining table", "toilet", "tv",
            "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave",
            "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase",
            "scissors", "teddy bear", "hair drier", "toothbrush",
        ],
    }
    dataset_yaml.write_text(yaml.dump(config))
    return dataset_yaml


def _mean_confidence(model: YOLO, image_dir: Path, limit: int = 100) -> float:
    """
    Compute mean detection confidence on a sample of images.

    Used as a proxy metric for model quality on holdout images where
    YOLO-format ground truth annotations are not available.

    Args:
        model (YOLO): Loaded YOLO model to evaluate.
        image_dir (Path): Directory of images to run inference on.
        limit (int): Max images to evaluate (default 100 for speed).

    Returns:
        float: Mean confidence across all detections, or 0.0 if none found.
    """
    images = sorted(image_dir.glob("*.jpg"))[:limit]
    if not images:
        return 0.0

    confidences = []
    for img_path in images:
        results = model(str(img_path), conf=CONFIDENCE_THRESHOLD, verbose=False)
        for result in results:
            if result.boxes is not None:
                confidences.extend(result.boxes.conf.cpu().tolist())

    return sum(confidences) / len(confidences) if confidences else 0.0


def retrain_yolo() -> dict:
    """
    Fine-tune YOLO11n on corrections and evaluate against holdout images.

    Collects image/label pairs from data/corrections/, runs a short training
    run on top of the base weights, then measures mean detection confidence
    on data/holdout/ for both the new and baseline models.

    Args:
        None

    Returns:
        dict: precision (float), recall (float), baseline_precision (float),
              baseline_recall (float), model_path (str).
              Note: precision/recall here are mean confidence proxies —
              not true P/R, which requires ground-truth COCO annotations.
    """
    pairs = _collect_pairs()
    if not pairs:
        raise RuntimeError("No valid image/label pairs found in data/corrections/")

    print(f"Fine-tuning on {len(pairs)} correction pairs ...")

    with tempfile.TemporaryDirectory() as tmpdir:
        dataset_yaml = _write_dataset_yaml(tmpdir, pairs)

        model = YOLO(str(BASE_MODEL))
        model.train(
            data=str(dataset_yaml),
            epochs=EPOCHS,
            imgsz=640,
            batch=8,
            workers=0,  # disable DataLoader multiprocessing — avoids /dev/shm exhaustion in Docker
            verbose=False,
        )

        new_model_path = Path(model.trainer.best)

    new_model = YOLO(str(new_model_path))
    baseline_path = FINETUNED_MODEL if FINETUNED_MODEL.exists() else BASE_MODEL
    baseline_model = YOLO(str(baseline_path))

    print("Evaluating on holdout ...")
    new_conf = _mean_confidence(new_model, HOLDOUT_DIR)
    baseline_conf = _mean_confidence(baseline_model, HOLDOUT_DIR)

    print(f"New model confidence:      {new_conf:.4f}")
    print(f"Baseline model confidence: {baseline_conf:.4f}")

    return {
        "precision": new_conf,
        "recall": new_conf,
        "baseline_precision": baseline_conf,
        "baseline_recall": baseline_conf,
        "model_path": str(new_model_path),
    }


if __name__ == "__main__":
    result = retrain_yolo()
    print(result)
