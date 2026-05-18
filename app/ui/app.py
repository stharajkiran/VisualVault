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
API_BASE = os.getenv("API_BASE", "http://127.0.0.1:8000")
DATA_INDEX_DIR = Path(os.getenv("DATA_INDEX_DIR", str(Path(__file__).parent.parent.parent / "data" / "index")))

st.set_page_config(page_title="VisualVault", layout="wide")
st.title("VisualVault")
st.caption("Semantic image search — upload images, search by description.")

page = st.sidebar.radio("Navigate", ["Upload", "Search", "Governance", "Live"])

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

    if "search_results" not in st.session_state:
        st.session_state.search_results = None
    if "similar_asset_id" not in st.session_state:
        st.session_state.similar_asset_id = None

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
                st.session_state.search_results = response.json()
                st.session_state.similar_asset_id = None
            except httpx.ConnectError:
                st.error("Cannot connect to the API. Make sure `uvicorn app.api.main:app --reload` is running.")
            except Exception as e:
                st.error(f"Search failed: {e}")

    if st.session_state.search_results:
        result = st.session_state.search_results
        st.write(f"**{result['total']} results** for: *{result['query']}*")

        cols = st.columns(4)
        for i, item in enumerate(result["results"]):
            image_path = DATA_INDEX_DIR / item["filename"]
            with cols[i % 4]:
                if image_path.exists():
                    st.image(str(image_path), use_container_width=True)
                else:
                    st.write(f"[{item['filename']}]")
                st.caption(f"Score: {item['score']:.3f}")
                if st.button("Find similar", key=f"similar_{item['id']}"):
                    st.session_state.similar_asset_id = item["id"]
                    st.rerun()

    if st.session_state.similar_asset_id is not None:
        st.divider()
        st.subheader(f"Similar to asset {st.session_state.similar_asset_id}")
        if st.button("Clear"):
            st.session_state.similar_asset_id = None
            st.rerun()
        try:
            resp = httpx.get(
                f"{API_BASE}/similar/{st.session_state.similar_asset_id}",
                timeout=10,
            )
            resp.raise_for_status()
            similar = resp.json()
            cols = st.columns(4)
            for i, item in enumerate(similar["results"]):
                image_path = DATA_INDEX_DIR / item["filename"]
                with cols[i % 4]:
                    if image_path.exists():
                        st.image(str(image_path), use_container_width=True)
                    else:
                        st.write(f"[{item['filename']}]")
                    st.caption(f"Score: {item['score']:.3f}")
        except httpx.ConnectError:
            st.error("Cannot connect to the API.")
        except Exception as e:
            st.error(f"Similar search failed: {e}")

    st.divider()
    st.subheader("Reverse Image Search")
    st.caption("Upload any image to find visually similar ones in the library. The image is not indexed.")

    upload_for_similar = st.file_uploader(
        "Upload image", type=["jpg", "jpeg", "png", "webp"], key="reverse_search"
    )
    if upload_for_similar and st.button("Find similar images"):
        with st.spinner("Searching..."):
            try:
                resp = httpx.post(
                    f"{API_BASE}/similar/upload",
                    files={"file": (upload_for_similar.name, upload_for_similar.getvalue(), upload_for_similar.type)},
                    timeout=15,
                )
                resp.raise_for_status()
                similar = resp.json()
                st.write(f"**{similar['total']} similar images found**")
                cols = st.columns(4)
                for i, item in enumerate(similar["results"]):
                    image_path = DATA_INDEX_DIR / item["filename"]
                    with cols[i % 4]:
                        if image_path.exists():
                            st.image(str(image_path), use_container_width=True)
                        else:
                            st.write(f"[{item['filename']}]")
                        st.caption(f"Score: {item['score']:.3f}")
            except httpx.ConnectError:
                st.error("Cannot connect to the API.")
            except Exception as e:
                st.error(f"Reverse search failed: {e}")

# ── Governance page ────────────────────────────────────────────────────────────
elif page == "Governance":
    st.header("Governance")
    st.write("Review asset rights, expiry status, and AI action audit logs.")

    st.subheader("Expired Assets")
    try:
        resp = httpx.get(f"{API_BASE}/governance/expired", timeout=10)
        resp.raise_for_status()
        expired = resp.json()
        if expired:
            st.dataframe(expired, use_container_width=True)
        else:
            st.success("No expired assets.")
    except httpx.ConnectError:
        st.error("Cannot connect to the API.")
    except Exception as e:
        st.error(f"Failed to load expired assets: {e}")

    st.divider()

    st.subheader("Rights Missing")
    try:
        resp = httpx.get(f"{API_BASE}/governance/rights-missing", timeout=10)
        resp.raise_for_status()
        missing = resp.json()
        if missing:
            st.dataframe(missing, use_container_width=True)
        else:
            st.success("All assets have usage rights set.")
    except httpx.ConnectError:
        st.error("Cannot connect to the API.")
    except Exception as e:
        st.error(f"Failed to load rights-missing assets: {e}")

    st.divider()

    st.subheader("Asset Audit Log")
    asset_id = st.number_input("Asset ID", min_value=0, step=1)
    if st.button("Fetch Audit"):
        try:
            resp = httpx.get(f"{API_BASE}/governance/audit/{int(asset_id)}", timeout=10)
            if resp.status_code == 404:
                st.warning(f"No governance record found for asset {int(asset_id)}.")
            else:
                resp.raise_for_status()
                record = resp.json()
                st.json(record)
        except httpx.ConnectError:
            st.error("Cannot connect to the API.")
        except Exception as e:
            st.error(f"Audit fetch failed: {e}")

# ── Live page ──────────────────────────────────────────────────────────────────
elif page == "Live":
    st.header("Live Preview")
    st.caption("Point your webcam at any scene — capture a frame to see YOLO tags. Frames are not indexed.")

    live_mode = st.checkbox("Start Live Preview")
    frame = st.camera_input("Webcam", label_visibility="collapsed")

    if live_mode and frame is not None:
        try:
            resp = httpx.post(
                f"{API_BASE}/live/frame",
                files={"file": ("frame.jpg", frame.getvalue(), "image/jpeg")},
                timeout=10,
            )
            resp.raise_for_status()
            tags = resp.json().get("tags", [])
            if tags:
                st.subheader("Detected")
                for tag in tags:
                    st.write(f"- **{tag['label']}** ({tag['confidence']:.0%})")
            else:
                st.info("Nothing detected.")
        except httpx.ConnectError:
            st.error("Cannot connect to the API.")
        except Exception as e:
            st.error(f"Preview failed: {e}")
