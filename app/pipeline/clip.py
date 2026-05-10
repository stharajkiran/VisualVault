import time
import numpy as np
import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_NAME = "openai/clip-vit-base-patch32"


def load_model() -> tuple[CLIPModel, CLIPProcessor]:

    print(f"[CLIP] Loading {MODEL_NAME} on {DEVICE} ...")
    t0 = time.perf_counter()
    model = CLIPModel.from_pretrained(MODEL_NAME).to(DEVICE)
    processor = CLIPProcessor.from_pretrained(MODEL_NAME)
    model.eval()
    elapsed = (time.perf_counter() - t0) * 1000
    print(f"[CLIP] Loaded in {elapsed:.0f}ms")
    return model, processor


def embed_image(image: Image.Image, model: CLIPModel, processor: CLIPProcessor) -> np.ndarray:
    """
    Convert a PIL image into a normalized 512-dim CLIP embedding.

    The processor resizes and normalizes the image into the format CLIP expects,
    then the model encodes it into a vector. We normalize the output so its length
    is exactly 1.0 — this makes similarity comparisons (via dot product) fast and
    consistent, which is what Qdrant uses internally for cosine distance.

    Args:
        image: Any PIL Image (mode will be handled by the processor).
        model: Loaded CLIPModel on DEVICE.
        processor: Matching CLIPProcessor for the same model.

    Returns:
        np.ndarray of shape (512,), dtype float32, L2-norm = 1.0.
    """
    t0 = time.perf_counter()
    # The processor handles resizing, cropping, and normalization. It returns a dict of tensors,
    inputs = processor(images=image, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        features = model.get_image_features(**inputs)
    vec = features.pooler_output.squeeze().cpu().numpy().astype(np.float32)
    vec = vec / np.linalg.norm(vec)
    elapsed = (time.perf_counter() - t0) * 1000
    print(f"[CLIP] embed_image: {elapsed:.1f}ms, shape={vec.shape}, norm={np.linalg.norm(vec):.4f}")
    return vec


def embed_text(query: str, model: CLIPModel, processor: CLIPProcessor) -> np.ndarray:
    """
    Convert a text query into a normalized 512-dim CLIP embedding.

    Uses the text encoder side of CLIP. Because CLIP was trained to align image
    and text embeddings in the same vector space, the resulting vector can be
    compared directly against image embeddings via dot product — that's what
    makes "search by description" work without any manual tags.

    Args:
        query: Natural language search string (e.g. "a dog on a beach").
        model: Loaded CLIPModel on DEVICE.
        processor: Matching CLIPProcessor for the same model.

    Returns:
        np.ndarray of shape (512,), dtype float32, L2-norm = 1.0.
    """
    t0 = time.perf_counter()
    inputs = processor(text=query, return_tensors="pt", padding=True, truncation=True).to(DEVICE)
    with torch.no_grad():
        features = model.get_text_features(**inputs)
    vec = features.pooler_output.squeeze().cpu().numpy().astype(np.float32)
    vec = vec / np.linalg.norm(vec)
    elapsed = (time.perf_counter() - t0) * 1000
    print(f"[CLIP] embed_text: {elapsed:.1f}ms, shape={vec.shape}, norm={np.linalg.norm(vec):.4f}")
    return vec


if __name__ == "__main__":
    from PIL import Image
    from app.pipeline.clip import load_model, embed_image, embed_text
    model, processor = load_model()
    img = Image.new('RGB', (224, 224), color=(128, 128, 128))
    img_vec = embed_image(img, model, processor)
    txt_vec = embed_text('a grey square', model, processor)
    print('similarity:', img_vec @ txt_vec)


