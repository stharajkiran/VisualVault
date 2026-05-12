"""
VisualVault — HuggingFace Spaces demo

Lightweight semantic image search using CLIP + pre-computed embeddings.
No Qdrant, no YOLO, no BLIP-2 — search only.

Images are displayed via their public COCO dataset URLs.
"""

import numpy as np
import streamlit as st
from pathlib import Path
from transformers import CLIPModel, CLIPProcessor
import torch

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_NAME = "openai/clip-vit-base-patch32"
# 12 digits in COCO image IDs
COCO_URL = "http://images.cocodataset.org/val2017/{image_id:012d}.jpg"

EMBEDDINGS_PATH = Path(__file__).parent / "embeddings.npy"
IMAGE_IDS_PATH = Path(__file__).parent / "image_ids.npy"


@st.cache_resource
def load_model() -> tuple[CLIPModel, CLIPProcessor]:
    """
    Load CLIP model and processor, cached for the lifetime of the server process.

    Args: None

    Returns:
        tuple[CLIPModel, CLIPProcessor]: Loaded model and processor on DEVICE.
    """
    model = CLIPModel.from_pretrained(MODEL_NAME).to(DEVICE)
    processor = CLIPProcessor.from_pretrained(MODEL_NAME)
    model.eval()
    return model, processor


@st.cache_resource
def load_embeddings() -> tuple[np.ndarray, np.ndarray]:
    """
    Load pre-computed CLIP embeddings and their COCO image IDs from disk.

    Args: None

    Returns:
        tuple[np.ndarray, np.ndarray]: embeddings of shape (N, 512) and image_ids of shape (N,).
    """
    embeddings = np.load(EMBEDDINGS_PATH)
    image_ids = np.load(IMAGE_IDS_PATH)
    return embeddings, image_ids


def search(query: str, model: CLIPModel, processor: CLIPProcessor,
           embeddings: np.ndarray, image_ids: np.ndarray, top_k: int) -> list[tuple[int, float]]:
    """
    Embed a text query with CLIP and return the top-K most similar image IDs.

    Similarity is computed as dot product between the normalized query vector
    and all pre-computed image embeddings — equivalent to cosine similarity
    since all vectors are L2-normalized.

    Args:
        query (str): Natural language search string.
        model (CLIPModel): Loaded CLIP model.
        processor (CLIPProcessor): Matching CLIP processor.
        embeddings (np.ndarray): Pre-computed image embeddings, shape (N, 512).
        image_ids (np.ndarray): COCO image IDs corresponding to each embedding row.
        top_k (int): Number of top results to return.

    Returns:
        list[tuple[int, float]]: List of (image_id, score) sorted by score descending.
    """
    inputs = processor(text=query, return_tensors="pt", padding=True, truncation=True).to(DEVICE)
    with torch.no_grad():
        features = model.get_text_features(**inputs)
    query_vec = features.pooler_output.squeeze().cpu().numpy().astype(np.float32)
    query_vec = query_vec / np.linalg.norm(query_vec)

    # Dot product against all embeddings — fast numpy matrix multiplication
    scores = embeddings @ query_vec
    top_indices = np.argsort(scores)[::-1][:top_k]

    return [(int(image_ids[i]), float(scores[i])) for i in top_indices]


# ── UI ─────────────────────────────────────────────────────────────────────────

st.set_page_config(page_title="VisualVault Demo", layout="wide")
st.title("VisualVault")
st.caption("Semantic image search — type a description, find matching images.")
st.info("Demo version: search only. Upload and full pipeline available in the full stack.")

with st.spinner("Loading CLIP model ..."):
    model, processor = load_model()
    embeddings, image_ids = load_embeddings()

st.success(f"Ready — {len(embeddings):,} images indexed.")

query = st.text_input("Search query", placeholder="e.g. a dog running on the beach")
top_k = st.slider("Number of results", min_value=1, max_value=20, value=9)

if st.button("Search") and query.strip():
    with st.spinner("Searching ..."):
        results = search(query, model, processor, embeddings, image_ids, top_k)

    st.write(f"**Top {len(results)} results** for: *{query}*")
    cols = st.columns(3)

    shown = 0
    for image_id, score in results:
        # Skip non-COCO images (test uploads with hashed IDs)
        if image_id > 999999999999:
            continue
        url = COCO_URL.format(image_id=image_id)
        with cols[shown % 3]:
            st.image(url, use_column_width=True)
            st.caption(f"Score: {score:.3f}")
        shown += 1
