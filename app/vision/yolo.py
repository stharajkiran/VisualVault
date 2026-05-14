import time
import torch
from PIL import Image
from ultralytics import YOLO

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_NAME = "artifacts/models/yolo11n.pt"
CONFIDENCE_THRESHOLD = 0.25


def load_model() -> YOLO:
    """
    Load YOLO11 nano onto the GPU.

    Ultralytics downloads the weights automatically on first use (~5MB).
    The nano variant (yolo11n) is the fastest in the YOLO11 family —
    accurate enough for tagging, fast enough to run alongside CLIP and BLIP-2
    on the same GPU without becoming the bottleneck.

    Returns:
        YOLO model instance ready for inference.
    """
    print(f"[YOLO] Loading {MODEL_NAME} on {DEVICE} ...")
    t0 = time.perf_counter()
    model = YOLO(MODEL_NAME)
    model.to(DEVICE)
    elapsed = (time.perf_counter() - t0) * 1000
    print(f"[YOLO] Loaded in {elapsed:.0f}ms")
    return model


def detect(image: Image.Image, model: YOLO) -> list[tuple[str, float]]:
    """
    Run object detection on a PIL image and return filtered tags.

    YOLO scans the entire image in one forward pass and returns bounding boxes,
    class labels, and confidence scores. We discard boxes below
    CONFIDENCE_THRESHOLD to avoid surfacing low-quality detections as tags.
    Duplicate labels are deduplicated — if two dogs are detected, "dog" appears
    once with the higher of the two confidence scores.

    Args:
        image: PIL Image in any mode (converted to RGB internally).
        model: Loaded YOLO instance.

    Returns:
        List of (label, confidence) tuples sorted by confidence descending.
        e.g. [("person", 0.91), ("bicycle", 0.78), ("car", 0.52)]
    """
    t0 = time.perf_counter()
    results = model(image, verbose=False)

    # results[0].boxes holds all detections for the first (only) image.
    # Each box has .cls (class index) and .conf (confidence score).
    seen: dict[str, float] = {}
    for box in results[0].boxes:
        label = model.names[int(box.cls)]
        conf = float(box.conf)
        if conf >= CONFIDENCE_THRESHOLD:
            # Keep only the highest-confidence detection per label
            if label not in seen or conf > seen[label]:
                seen[label] = conf

    tags = sorted(seen.items(), key=lambda x: x[1], reverse=True)
    elapsed = (time.perf_counter() - t0) * 1000
    print(f"[YOLO] detect: {elapsed:.1f}ms, tags={[t[0] for t in tags]}")
    return tags



if __name__ == "__main__":
    from PIL import Image
    from app.vision.yolo import load_model, detect
    model = load_model()
    # img = Image.new('RGB', (640, 640), color=(200, 200, 200))
    # load image from assets
    img = Image.open("assets/img.jpg").convert("RGB")
    tags = detect(img, model)
    print('tags:', tags)