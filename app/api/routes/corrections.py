"""POST /corrections/submit — Label Studio webhook, writes YOLO11 labels and PostgreSQL rows."""
from pathlib import Path

from fastapi import APIRouter, Request

from app.db.database import write_correction
from app.core.metrics import CORRECTIONS_TOTAL

router = APIRouter()

CORRECTIONS_DIR = Path("data/corrections")

# COCO 80-class label → YOLO class index mapping
COCO_CLASS_IDS: dict[str, int] = {
    "person": 0, "bicycle": 1, "car": 2, "motorcycle": 3, "airplane": 4,
    "bus": 5, "train": 6, "truck": 7, "boat": 8, "traffic light": 9,
    "fire hydrant": 10, "stop sign": 11, "parking meter": 12, "bench": 13,
    "bird": 14, "cat": 15, "dog": 16, "horse": 17, "sheep": 18, "cow": 19,
    "elephant": 20, "bear": 21, "zebra": 22, "giraffe": 23, "backpack": 24,
    "umbrella": 25, "handbag": 26, "tie": 27, "suitcase": 28, "frisbee": 29,
    "skis": 30, "snowboard": 31, "sports ball": 32, "kite": 33,
    "baseball bat": 34, "baseball glove": 35, "skateboard": 36,
    "surfboard": 37, "tennis racket": 38, "bottle": 39, "wine glass": 40,
    "cup": 41, "fork": 42, "knife": 43, "spoon": 44, "bowl": 45,
    "banana": 46, "apple": 47, "sandwich": 48, "orange": 49, "broccoli": 50,
    "carrot": 51, "hot dog": 52, "pizza": 53, "donut": 54, "cake": 55,
    "chair": 56, "couch": 57, "potted plant": 58, "bed": 59,
    "dining table": 60, "toilet": 61, "tv": 62, "laptop": 63, "mouse": 64,
    "remote": 65, "keyboard": 66, "cell phone": 67, "microwave": 68,
    "oven": 69, "toaster": 70, "sink": 71, "refrigerator": 72, "book": 73,
    "clock": 74, "vase": 75, "scissors": 76, "teddy bear": 77,
    "hair drier": 78, "toothbrush": 79,
}


@router.post("/corrections/submit")
async def submit_correction(request: Request) -> dict:
    """
    Receive a Label Studio annotation webhook and write corrections in YOLO11 format.

    Parses bounding box results from the Label Studio payload, converts coordinates
    from Label Studio's percentage-based system to YOLO11's normalized center format,
    and writes one .txt file per image to data/corrections/. Increments the
    Prometheus corrections counter on each successful write.

    Args:
        request (Request): Raw FastAPI request — Label Studio sends arbitrary JSON.

    Returns:
        dict: filename (str), corrections (int) — number of boxes written.
    """
    payload = await request.json()

    image_path = payload.get("task", {}).get("data", {}).get("image", "")
    filename = Path(image_path).stem

    results = payload.get("annotation", {}).get("result", [])
    lines: list[str] = []
    parsed: list[tuple[int, float, float, float, float]] = []

    for result in results:
        if result.get("type") != "rectanglelabels":
            continue
        value = result.get("value", {})
        labels = value.get("rectanglelabels", [])
        if not labels:
            continue

        class_id = COCO_CLASS_IDS.get(labels[0])
        if class_id is None:
            continue

        # Label Studio: x, y = top-left corner as % of image dimensions
        # YOLO11: cx, cy = center as fraction of image dimensions
        x, y, w, h = value["x"], value["y"], value["width"], value["height"]
        cx = (x + w / 2) / 100
        cy = (y + h / 2) / 100
        nw = w / 100
        nh = h / 100
        lines.append(f"{class_id} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")
        parsed.append((class_id, cx, cy, nw, nh))

    if lines:
        CORRECTIONS_DIR.mkdir(parents=True, exist_ok=True)
        (CORRECTIONS_DIR / f"{filename}.txt").write_text("\n".join(lines))
        for class_id, cx, cy, nw, nh in parsed:
            write_correction(filename, class_id, cx, cy, nw, nh)
        CORRECTIONS_TOTAL.inc()

    return {"filename": filename, "corrections": len(lines)}
