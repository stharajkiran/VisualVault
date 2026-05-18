"""Label Studio integration — saves flagged images to disk and pushes review tasks."""

import base64
import logging
import os
from pathlib import Path

from label_studio_sdk import LabelStudio

CORRECTIONS_IMAGES_DIR = Path("data/corrections/images")
logger = logging.getLogger(__name__)

LABEL_STUDIO_URL = os.environ.get("LABEL_STUDIO_URL", "").rstrip("/")
LABEL_STUDIO_API_KEY = os.environ.get("LABEL_STUDIO_API_KEY", "").strip().strip("\"'")
LABEL_STUDIO_PROJECT_ID = int(os.environ.get("LABEL_STUDIO_PROJECT_ID", "1"))

# Confidence below this threshold triggers a Label Studio review task
LOW_CONFIDENCE_THRESHOLD = 0.5


def push_to_label_studio(
    filename: str,
    image_bytes: bytes,
    detections: list[tuple[str, float]],
) -> bool:
    """
    Push a low-confidence image to Label Studio as a review task.

    Saves the image to data/corrections/images/ for retraining, then creates
    a Label Studio task with YOLO detections as pre-annotations. Uses the
    label-studio-sdk which handles auth internally — avoids manual Bearer/Token
    header construction that broke with Label Studio 1.23 JWT tokens.

    No-ops silently if LABEL_STUDIO_URL or LABEL_STUDIO_API_KEY are not set
    so a missing Label Studio config never blocks image ingestion.

    Args:
        filename (str): Original image filename — used as the task reference.
        image_bytes (bytes): Raw image file contents.
        detections (list[tuple[str, float]]): YOLO detections as (label, confidence) pairs.

    Returns:
        bool: True if the task was created successfully, False otherwise.
    """
    # Always save the image to disk — retraining needs it regardless of Label Studio
    CORRECTIONS_IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    (CORRECTIONS_IMAGES_DIR / filename).write_bytes(image_bytes)

    if not LABEL_STUDIO_URL or not LABEL_STUDIO_API_KEY:
        return False

    encoded = base64.b64encode(image_bytes).decode("utf-8")
    data_uri = f"data:image/jpeg;base64,{encoded}"

    # Placeholder bounding boxes — yolo.detect() returns class+confidence only,
    # not pixel coordinates. Reviewer sees the correct label guess without a precise box.
    pre_annotations = [
        {
            "type": "rectanglelabels",
            "value": {
                "rectanglelabels": [label],
                "x": 10.0,
                "y": 10.0,
                "width": 80.0,
                "height": 80.0,
            },
            "score": round(conf, 4),
        }
        for label, conf in detections
    ]

    task = {
        "data": {"image": data_uri, "filename": filename},
        "predictions": [{"result": pre_annotations}] if pre_annotations else [],
    }

    try:
        ls = LabelStudio(base_url=LABEL_STUDIO_URL, api_key=LABEL_STUDIO_API_KEY)
        ls.projects.import_tasks(id=LABEL_STUDIO_PROJECT_ID, request=[task])
        return True
    except Exception:
        logger.exception("Label Studio import failed: url=%s project=%s", LABEL_STUDIO_URL, LABEL_STUDIO_PROJECT_ID)
        return False
