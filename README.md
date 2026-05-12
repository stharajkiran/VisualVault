# VisualVault

**Semantic image and video search platform for marketing teams.**  
Upload images, get automatic tags and captions. Search your entire library in plain English.

[![HF Spaces](https://img.shields.io/badge/🤗%20HuggingFace-Live%20Demo-blue)](https://huggingface.co/spaces/kstha/visualvault-demo)

---

## What it does

VisualVault lets you search a large image library by describing what you want — "happy family at the beach", "cyclist in traffic at night" — and returns visually matching results even if those words were never typed as tags. Three computer vision models run on every uploaded image automatically: object detection, natural language captioning, and semantic embedding for search.

---

## Metrics

| Metric | Value | Phase |
|---|---|---|
| Recall@10 (CLIP ViT-B/32) | 0.94 (GPT-4o-mini queries, indexed images) | 1 |
| Index size | 4,000 images | 1 |
| Search latency p95 | 15ms (Qdrant, top-10, 4k images) | 1 |
| Indexing throughput | 55 img/s | 1 |
| TensorRT latency reduction | _TBD_ | 2 |
| Batch throughput | _TBD_ | 2 |
| Tagging error reduction | _TBD_ | 3 |

---

## Architecture

```
[Browser] → [FastAPI] → [Celery Worker]
                              ↓
                    CLIP + YOLO11 + BLIP-2 (RTX 5090)
                              ↓
               [Qdrant]  [PostgreSQL]  [MinIO]
                              ↓
                    [Prometheus + Grafana]
```

_Full architecture diagram added in Block 4C._

---

## Tech stack

| Layer | Tools |
|---|---|
| ML models | CLIP ViT-B/32, YOLO11n, BLIP-2 OPT-2.7B, TensorRT |
| Search | Qdrant (vector), PostgreSQL (metadata) |
| Storage | MinIO (S3-compatible) |
| Backend | FastAPI, Celery, Redis |
| Frontend | Streamlit (main app + HF Spaces demo) |
| MLOps | MLflow, DVC, Evidently AI, Label Studio |
| Monitoring | Prometheus, Grafana |
| Infrastructure | Docker, Docker Compose, GitHub Actions |

---

## Quick start

_Docker Compose setup added in Block 1E._

```bash
git clone https://github.com/stharajkiran/visualvault
cd visualvault
cp .env.example .env   # fill in your values
docker-compose up
```

---

## Roadmap

| Phase | What it delivers |
|---|---|
| 1 — Semantic Search Engine ✅ | CLIP + Qdrant semantic search, live demo on HF Spaces |
| 2 — Production Pipeline 🔄 | 500 img/min throughput, 4× TensorRT speedup, Prometheus/Grafana observability |
| 3 — Self-Healing System | Evidently AI drift detection, Label Studio active learning, automated YOLO retraining |
| 4 — Enterprise Platform | Consent governance, real-time ingest, usage analytics, 34% tagging error reduction |
