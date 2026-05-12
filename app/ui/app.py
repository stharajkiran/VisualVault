"""
VisualVault — Streamlit UI

Two pages:
  - Upload: send an image to the FastAPI backend, display tags and caption
  - Search: query the library by description, display matching images in a grid

Run with:
    streamlit run app/ui/app.py

Requires the FastAPI backend to be running:
    uvicorn app.api.main:app --reload
"""

import os
import time
import httpx
import streamlit as st
from pathlib import Path

# Reads from environment variable so it works both locally and inside Docker.
# Locally: API_BASE defaults to localhost. In Docker: set to http://api:8000.
API_BASE = os.getenv("API_BASE", "http://localhost:8000")
DATA_INDEX_DIR = Path(os.getenv("DATA_INDEX_DIR", str(Path(__file__).parent.parent.parent / "data" / "index")))

st.set_page_config(page_title="VisualVault", layout="wide")
st.title("VisualVault")
st.caption("Semantic image search — upload images, search by description.")

page = st.sidebar.radio("Navigate", ["Upload", "Search"])

# ── Upload page ────────────────────────────────────────────────────────────────
if page == "Upload":
    st.header("Upload an Image")
    st.write("Upload an image to automatically generate tags and a caption.")

    uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png", "webp"])

    if uploaded_file is not None:
        col1, col2 = st.columns(2)

        with col1:
            st.image(uploaded_file, caption="Uploaded image", use_container_width=True)

        with col2:
            with st.spinner("Uploading and processing..."):
                try:
                    # POST /upload returns immediately with a job_id
                    response = httpx.post(
                        f"{API_BASE}/upload",
                        files={"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)},
                        timeout=30,
                    )
                    response.raise_for_status()
                    job_id = response.json()["job_id"]

                    # Poll GET /jobs/{job_id} until pipeline finishes
                    result = None
                    for _ in range(60):
                        job = httpx.get(f"{API_BASE}/jobs/{job_id}", timeout=10).json()
                        if job["status"] == "SUCCESS":
                            result = job["result"]
                            break
                        elif job["status"] == "FAILURE":
                            st.error(f"Pipeline failed: {job.get('error')}")
                            result = None
                            break
                        time.sleep(1)
                    else: # this is for-else on the for loop: runs if we exit the loop without hitting break
                        st.error("Pipeline timed out after 60 seconds.")
                        result = None

                    if result:
                        st.subheader("Results")
                        st.write(f"**Caption:** {result['caption']}")
                        st.write("**Tags:**")
                        for tag in result["tags"]:
                            st.write(f"- {tag['label']} ({tag['confidence']:.0%} confidence)")

                except httpx.ConnectError:
                    st.error("Cannot connect to the API. Make sure `uvicorn app.api.main:app --reload` is running.")
                except Exception as e:
                    st.error(f"Upload failed: {e}")

# ── Search page ────────────────────────────────────────────────────────────────
elif page == "Search":
    st.header("Search the Library")
    st.write("Describe what you're looking for in plain English.")

    query = st.text_input("Search query", placeholder="e.g. a dog running on the beach")
    top_k = st.slider("Number of results", min_value=1, max_value=20, value=10)

    if st.button("Search") and query.strip():
        with st.spinner("Searching..."):
            try:
                response = httpx.get(
                    f"{API_BASE}/search",
                    params={"query": query, "top_k": top_k},
                    timeout=30,
                )
                response.raise_for_status()
                result = response.json()

                st.write(f"**{result['total']} results** for: *{result['query']}*")

                # Display results in a 4-column grid
                cols = st.columns(4)
                for i, item in enumerate(result["results"]):
                    image_path = DATA_INDEX_DIR / item["filename"]
                    with cols[i % 4]:
                        if image_path.exists():
                            st.image(str(image_path), use_container_width=True)
                        else:
                            st.write(f"[{item['filename']}]")
                        st.caption(f"Score: {item['score']:.3f}")

            except httpx.ConnectError:
                st.error("Cannot connect to the API. Make sure `uvicorn app.api.main:app --reload` is running.")
            except Exception as e:
                st.error(f"Search failed: {e}")
