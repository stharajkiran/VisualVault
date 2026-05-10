# VisualVault — Project Knowledge Base

This is the single source of truth for the VisualVault project. Paste it at the start of a new Claude chat or give it to Claude Code as a reference. After reading it, any assistant should know what the project is, why it exists, every layer of its architecture, every phase of the build, every block inside every phase, what the success criteria are, and how Claude Code is supposed to be used at each step.

---

## Section 1 — What VisualVault is and why it exists

VisualVault is a web app for managing a large library of images and videos. Think of it as Google Photos for a company's marketing team, except it understands what is inside every image automatically and lets you search using normal English instead of keywords.

The core workflow is simple:

1. A user uploads images or videos.
2. The system runs each one through computer vision models and figures out what is in it — no manual tagging.
3. The user later searches the library by describing what they want ("happy family at the beach") and gets back the right images even if no one ever typed those words as tags.

Everything else in this project is built on top of that one idea.

### The problem

Companies like marketing teams, news agencies, and creative studios sit on millions of images and videos they cannot find when they need them. Manual tagging is slow, expensive, inconsistent, and never finished. Commercial tools that solve this exist — PhotoShelter, Bynder, Adobe Experience Manager, Canto — and they are full enterprise products. VisualVault is a portfolio version of one of those platforms, built end to end and matched feature for feature against what those companies actually ship.

### Why this project exists

There are three reasons. All three matter equally.

1. **Demonstrate full-stack production ML engineering.** Not just training a model — everything around it: serving, optimizing, monitoring, retraining, and governing.
2. **Be immediately demonstrable.** A hiring manager clicks one link, tries the live demo, understands what it does in 30 seconds, and sees real performance numbers in the README without reading any code.
3. **Build incrementally.** Each completed phase is a deployable, resume-worthy project on its own. Job applications can start after Phase 1. Nothing gets thrown away — every later phase stacks on top.

### Why this stands out from other portfolio projects

Most CV portfolio projects stop at "I trained a model and put it in a Streamlit app." VisualVault goes several layers deeper in ways that are genuinely rare in portfolios:

- Three models running concurrently on one GPU through a single async pipeline.
- Self-monitoring of input data distribution with alerts when things drift.
- A complete active learning loop where uncertainty triggers human review and corrections feed retraining automatically.
- Documented model tradeoff decisions backed by measured A/B numbers, not intuition or defaults.
- One pipeline handling both images and video.
- An enterprise governance layer covering consent, usage rights, expiry, and audit logs — the exact gap the 2026 DAM market reports flag as the biggest unsolved problem in the space.
- Fully containerized and live on the public internet for anyone to try.

The point is to look like a production ML engineer, not a researcher or hobbyist.

---

## Section 2 — User-facing features

| Feature | What the user does | What the system does |
|---|---|---|
| Upload and auto-tag | Drags an image into the browser | Within seconds: object tags (person, bicycle, city street) and a natural language caption (a cyclist navigating downtown traffic) |
| Semantic search | Types "happy family at the beach" | Returns visually matching images even if those exact words were never typed as tags |
| Batch upload | Drops a zip file of 500 images | Queues all in the background, shows a progress bar, processes in parallel |
| Video search | Uploads a video file | Breaks it into frames, indexes each one, lets the user search and jump to the exact timestamp where something appears |
| Find similar | Clicks any image | Returns the 10 most visually similar images from the library |
| Real-time tagging | Points a webcam at something | Tags appear within one second, live |
| Usage analytics | Opens analytics tab | Sees most-retrieved assets and search queries that consistently return zero useful results |
| Governance | Opens governance tab | Sees images needing consent, images past their expiry date, and the full audit log per asset |
| Self-monitoring | Nothing — runs automatically | When uploads drift from the training distribution, the system alerts |
| Human review loop | Reviews flagged low-confidence tags in Label Studio | Corrections feed back into automated retraining and improve the model over time |

---

## Section 3 — The full tech stack

Every tool has a specific job. Nothing is here because it is trendy. Each one is here because a comparable commercial product would use it or something functionally identical.

### Layer 1 — Machine learning models

| Tool | Job |
|---|---|
| CLIP ViT-B/32 | Converts images and text into embeddings (numerical vectors). This is what makes semantic search possible — an image and a sentence can be compared if they are both in the same embedding space. |
| YOLO11 | Detects and labels objects inside images (person, car, dog, bicycle, etc). |
| BLIP-2 | Generates a natural language caption for an image (a cyclist navigating downtown traffic). |
| FFmpeg | Extracts frames from video files at a fixed interval so each frame can go through the same pipeline as a regular image. |
| TensorRT | Compresses and optimizes CLIP, YOLO11, and BLIP-2 to run roughly 3×–5× faster on the RTX 5090. |

### Layer 2 — Data and search

| Tool | Job |
|---|---|
| Qdrant | Vector database. Stores CLIP embeddings and retrieves the most similar ones when a user searches. |
| PostgreSQL | Relational database. Stores metadata, tags, captions, governance records, consent flags, audit logs, and the human-correction queue. |
| MinIO | Self-hosted file storage that behaves like Amazon S3. Stores the actual image and video files. |

### Layer 3 — Backend and processing

| Tool | Job |
|---|---|
| FastAPI | The REST API. Handles uploads, search queries, status checks, and governance endpoints. |
| Celery | Async task queue. Runs all heavy model inference in the background so the API never blocks. |
| Redis | Message broker. Carries jobs between FastAPI and the Celery workers. |

### Layer 4 — Frontend

| Tool | Job |
|---|---|
| React + Vite | The main web interface where users upload, search, and view results. |
| Gradio | A lightweight public demo deployed to Hugging Face Spaces for the live demo badge. |

### Layer 5 — MLOps and monitoring

| Tool | Job |
|---|---|
| MLflow | Tracks every experiment, model version, and metric. Makes rollback to a previous model a one-line command. |
| DVC | Versions datasets alongside the code so a model and its training data are always paired. Set up during Block 3C alongside MLflow. |
| Evidently AI | Detects when incoming images drift away from the training distribution. |
| Label Studio | Human review interface. When the system is unsure about a tag, the asset goes here for a person to correct. |
| Prometheus | Collects metrics in real time — request latency, queue depth, GPU utilization, throughput, model inference time. |
| Grafana | Dashboard layer on top of Prometheus. This is what gets screenshotted for the README. |

### Layer 6 — Infrastructure

| Tool | Job |
|---|---|
| Docker | Every service runs in its own container. |
| Docker Compose | One command starts the entire stack — API, workers, databases, monitoring. |
| GitHub Actions | Runs linting, tests, model smoke tests, and a Docker build check on every pull request. |
| Hugging Face Spaces | Hosts the live public demo so anyone can try the project from a browser. |

---

## Section 4 — The data layer

**Read this before starting Block 1B.** The evaluation methodology established here is the measurement anchor for every accuracy claim in the project. Get it wrong and the headline numbers in your resume and README are meaningless.

### The foundational distinction

CLIP, YOLO11, and BLIP-2 are pretrained models. In Phases 1 and 2, you do not train anything — you run inference. The data layer in those phases is about what you index, what you evaluate against, and how you keep those two things honest and separate.

Training data in the traditional sense only enters in Phase 3, when human corrections feed back into YOLO fine-tuning. But the evaluation set you build in Phase 1 is what makes that training meaningful. Without a fixed held-out set established early, you have no way to know whether fine-tuning actually helped. Get the split wrong in Phase 1 and the "34% tagging error reduction" target in Block 3E means nothing.

### Why COCO

COCO (Common Objects in Context) is a standard computer vision benchmark dataset from Microsoft. It contains over 100,000 images across 80 object categories — people, animals, vehicles, food, furniture, and more — with ground-truth bounding boxes, category labels, and captions for every image. It is free to download.

The reason we use COCO instead of random images: with random images there is no way to know objectively whether search results are correct. With COCO, every image has ground-truth labels. If you search "dog" and the wrong images come back, you have a measurable problem, not just a feeling. Every accuracy claim in the README is verifiable against this ground truth.

Do not use a single-domain dataset. A corpus of only nature photos makes semantic search look trivial and makes drift detection uninteresting. COCO has genuine visual diversity — that is required for both search quality and for the drift injection test in Block 3B to be meaningful.

### Download

- Source: `images.cocodataset.org`, 2017 validation split
- Size: approximately 1 GB, 5,000 images
- Also download: the 2017 val annotations JSON — ground-truth labels and captions
- Do not download the full training split (118,000 images, ~18 GB) until the pipeline is confirmed working on the val split
- Before Block 1C (no MinIO yet): store images locally in `assets/`
- From Block 1C onward: store images in MinIO

### The data split — establish this before Block 1B

Partition the 5,000 images before indexing a single one. This directory layout is the foundation of every measurement in the project:

```
data/
  raw/            # all 5,000 original COCO images, never modified
  index/          # 4,000 images — what goes into Qdrant
  holdout/        # 1,000 images — never indexed, never shown in demos
  ood_test/       # out-of-distribution images for drift testing (Phase 3)
  corrections/    # human-verified YOLO labels from Label Studio (Phase 3)
  eval_pairs.json # built during Block 1B, committed and frozen forever
```

**The holdout set never touches Qdrant.** Not during development, not during demos, not during drift testing. If evaluation images are in the search index, Recall@10 is inflated by exact-match retrieval rather than measuring semantic generalization. This is the single most common data leakage mistake in retrieval portfolio projects.

### eval_pairs.json — build during Block 1B, freeze permanently

From the 1,000 holdout images, hand-select 200 and write a natural language query for each one. This becomes `eval_pairs.json`. It is the ground truth for every CLIP-side model evaluation in the project from Phase 1 through Phase 4.

Each entry looks like this:

```json
{
  "query": "a person riding a bicycle through a busy city street",
  "image_id": "000000012345",
  "image_path": "holdout/000000012345.jpg",
  "source": "coco_val2017"
}
```

Query quality rules:

- **Bad:** `"dog"` — too ambiguous, too many matches, measures nothing
- **Good:** `"golden retriever sitting next to a red fire hydrant"` — specific enough that there is one clearly correct answer among the 4,000 indexed images, written the way a real user would type it

Write all 200 queries yourself. This takes about three hours and cannot be automated — the point is that these are human-verified ground truth pairs. Commit the file to the repo and tag the commit. Never modify it again. **This file is the evaluation contract for the entire project.**

### Recall@10 — the primary CLIP metric

For each pair in `eval_pairs.json`, run the query through the search pipeline and check whether the correct image appears in the top 10 results. Average across all 200 pairs. That is Recall@10.

A frozen CLIP ViT-B/32 on well-written queries against a diverse 4,000-image corpus should land around 0.70–0.78. Record this number in Block 1B. It is the Phase 1 baseline. Every CLIP-side experiment in the project is compared against it.

**Important scope:** Recall@10 measures CLIP-based search quality. CLIP is not fine-tuned at any point in this project. Do not use Recall@10 to evaluate YOLO fine-tuning — it will not move and will only create confusion. YOLO is evaluated separately on precision and recall against COCO ground-truth bounding boxes (see Phase 3 data section below).

### The single most important rule

Commit `eval_pairs.json` during Block 1B and never touch it again. Both measurement anchors must be in place before Phase 3 starts:

- `eval_pairs.json` — fixed reference for every CLIP-side change
- COCO holdout bounding boxes — fixed reference for every YOLO-side change

If you modify either after Phase 1, none of the downstream numbers are comparable and the headline accuracy claims collapse.

---

## Section 5 — Phase overview

Each phase ends with a tagged GitHub release (v1.0 through v4.0), a real deployable artifact, and a resume line usable immediately.

| Phase | Name | Headline outcome | Resume line |
|---|---|---|---|
| 1 | Semantic Search Engine | Working image search on 4,000 COCO images, live on Hugging Face Spaces | Built a semantic image search engine using CLIP and Qdrant, sub-50ms p95 latency on 4,000 indexed images with architecture that scales to 10,000+, deployed live on HF Spaces |
| 2 | Production Pipeline | Async, optimized, monitored, CI/CD green | Scaled to 500 images/min, 4× latency reduction via TensorRT, full Prometheus/Grafana observability, automated PR validation |
| 3 | Self-Healing System | Drift detection, active learning, automated retraining, video support | Built a self-healing pipeline with Evidently AI drift detection and a Label Studio active learning loop, targeting 34% tagging error reduction over four weeks of automated retraining |
| 4 | Enterprise Platform | Governance, real-time streaming, polished public presence | Built an enterprise-grade AI media intelligence platform with governance, real-time ingest, drift monitoring, and active learning — directly comparable to commercial DAM platforms |

---

## Section 6 — Phase 1: The Semantic Search Engine

### Goal

A working image search engine, indexed on real data with verified accuracy, deployed publicly. This is the core value of the entire project. Every later phase is built on top of this.

### Targets

- Sub-50ms p95 search latency on the 4,000-image index
- Architecture load-tested to 10,000+ images (even if the daily demo runs 4,000)
- Recall@10 baseline of 0.70–0.78 on the frozen eval set
- Live on Hugging Face Spaces with a demo badge in README

### Skills demonstrated

Multimodal ML, vector databases, embedding-based retrieval, REST API design, containerization, live deployment.

### Blocks

| Block | What it builds | Level | Done when |
|---|---|---|---|
| 1A | Local pipeline — one script, one image in, YOLO tags + BLIP-2 caption + CLIP 512-dim embedding out. GPU confirmed active. Per-model latency logged. | 2 — Guided | `python scripts/run_pipeline.py assets/test.jpg` prints tags, caption, embedding shape, and ms per model |
| 1B | Qdrant integration and the evaluation contract — establish data split (4,000 index / 1,000 holdout), index the 4,000 images, hand-write 200 query pairs from holdout images into `eval_pairs.json`, run Recall@10 and record the baseline. Plan for two sessions: one for the split and indexing, one for writing the eval pairs. | 2 — Guided | Data split in place, 4,000 COCO images indexed, `eval_pairs.json` committed and tagged, Recall@10 baseline recorded (expect 0.70–0.78), search under 100ms locally |
| 1C | FastAPI backend — `POST /upload` and `GET /search`, Swagger UI, input validation, error handling | 3 — Block sprint | Both endpoints work, non-image upload returns a clean 422, response times logged |
| 1D | React frontend — drag-and-drop upload with progress, search bar, results grid | 3 — Block sprint | Someone unfamiliar can upload and search without instructions |
| 1E | Docker Compose + HF Spaces — one-command full stack; quantized Gradio demo deployed to Hugging Face Spaces | 3 — Block sprint | `docker-compose up` starts everything cleanly, HF demo live, badge in README |

---

## Section 7 — Phase 2: The Production Pipeline

### Goal

Same search engine, engineered for real load. Async background processing, models optimized for the RTX 5090, full observability, automated CI on every pull request.

### Targets

- 500 images per minute throughput
- 4× latency reduction via TensorRT (the single most cited resume metric — it must be real and you must be able to explain it)
- p95 search latency under 50ms
- 100% of PRs auto-tested

### Skills demonstrated

Async task queues, GPU optimization, ONNX export, performance benchmarking, A/B model testing, observability, CI/CD.

### Blocks

| Block | What it builds | Level | Done when |
|---|---|---|---|
| 2A | Async with Celery + Redis — `POST /upload` returns HTTP 202 + job ID immediately; `GET /jobs/{id}` polls for results; processing happens in workers | 2 — Guided | 10 simultaneous uploads all complete successfully |
| 2B | Batch ingestion — `POST /batch` accepts a zip, queues each image, React shows progress, throughput benchmarked at 100/250/500 images | 3 — Block sprint | Throughput numbers per batch size committed to README, bottleneck identified |
| 2C | TensorRT optimization — export CLIP, YOLO11, BLIP-2 to ONNX, compile to TensorRT engines, replace PyTorch calls. **Highest-risk block in the project.** | 2 — Guided | 3×–5× speedup measured per model, no accuracy regression on the eval set |
| 2D | A/B model comparison — benchmark ViT-B/32 vs ViT-L/14 on the same `eval_pairs.json` under identical conditions; write `DECISIONS.md` entry | 3 — Block sprint | Recall@10 and latency documented for both models, decision written with measured data |
| 2E | Prometheus + Grafana — metrics for HTTP latency, queue depth, GPU usage, per-model inference time; alerts at queue depth >100 and p95 >200ms | 3 — Block sprint | Dashboard live, alerts configured, screenshot in README |
| 2F | GitHub Actions CI/CD — ruff lint, pytest, model smoke test, Docker build check, all blocking on PR merge | 3 — Block sprint | Workflow runs on every PR, deliberately broken test confirms blocking works |

### Data notes for Phase 2

- No new data added to the index.
- Block 2D A/B comparison: run `eval_pairs.json` against both ViT-B/32 and ViT-L/14 under identical conditions — same 200 queries, same 4,000 indexed images, same Recall@10 metric. Expected outcome: ViT-L/14 gains roughly 3–5 points of Recall@10 and costs roughly 6× the inference latency. That tradeoff, with real measured numbers, is your `DECISIONS.md` entry.
- Block 2B throughput test: use images from `data/index/` (already indexed). This is a pipeline performance test, not a retrieval quality test. Using already-indexed images is fine here.

---

## Section 8 — Phase 3: The Self-Healing System

### Goal

The pipeline now monitors itself, detects when it might be wrong, sends uncertain predictions to humans, learns from the corrections, and retrains automatically. This is the phase that separates a portfolio project from a production system.

### Targets

- **Tagging error rate target: 34% reduction over four weeks.** This number is a target measured in Block 4F, not a figure available at the end of Phase 3. The automated retraining loop is what makes this achievable. Track YOLO precision and recall per retraining run in MLflow so the trend line exists when you need it.
- Drift detection alerts within one batch cycle of a distribution shift
- Video pipeline processes at ~500ms per minute of footage

### Skills demonstrated

Drift detection, active learning, automated retraining, MLOps maturity, human-in-the-loop systems, video processing, data flywheel architecture.

### Blocks

| Block | What it builds | Level | Done when |
|---|---|---|---|
| 3A | Video keyframe extraction — `POST /upload/video`, FFmpeg pulls one frame every 2 sec, each frame becomes a regular Celery task, search returns timestamped video frames | 3 — Block sprint | Search returns frames with timestamps, throughput measured in seconds per minute of footage |
| 3B | Evidently AI drift detection — after each batch, compare mean CLIP embedding vs baseline; alert if cosine drift exceeds threshold; write to Prometheus; fire Slack/email | 2 — Guided | Drift metric in Grafana, intentional OOD batch fires alert, threshold choice documented |
| 3C | MLflow experiment tracking + DVC — log every experiment run (parameters, metrics, model artifact); set up DVC to version `data/index/`, `data/holdout/`, and `eval_pairs.json` alongside model artifacts; MLflow tracking server in Docker Compose | 3 — Block sprint | MLflow UI shows 5+ runs, DVC tracking confirmed, model rollback tested in one command |
| 3D | Label Studio active learning queue — YOLO confidence under threshold pushes asset to Label Studio; webhook writes corrections to PostgreSQL; correction rate tracked | 2 — Guided | Low-confidence assets appear in Label Studio automatically, corrections write back, correction rate logged |
| 3E | Automated retraining + data flywheel — Celery beat checks correction queue daily; when 100+ verified examples exist, kicks off YOLO fine-tuning; evaluates on holdout COCO bounding boxes; promotes only if precision and recall both improve | 3 — Block sprint | MLflow shows before/after YOLO precision and recall, Grafana shows improvement trend |

### Data notes for Phase 3

**Drift detection (Block 3B):**
- Baseline: compute the mean CLIP embedding across all 4,000 indexed images and store it as the reference point.
- OOD test batch: 200–300 images from a genuinely different visual domain — technical diagrams, medical imaging, satellite imagery, or abstract art. Store in `data/ood_test/`. Never add to the index.
- Threshold: do not just accept 0.15. Run the drift score on several in-distribution batches first, observe the natural variance, and set the threshold above the highest in-distribution drift score you observe. Plot the histogram of in-distribution drift scores. Commit it. Justify the chosen threshold in `DECISIONS.md`.

**YOLO confidence threshold (Block 3D):**
- Do not just accept 0.6. Run YOLO against the full index dataset first and plot the confidence distribution. If 80% of predictions are above 0.85 and 15% fall between 0.4 and 0.6, then 0.6 is reasonable. If the distribution is bimodal with a cliff at 0.5, set the threshold lower. The histogram and justification go in `DECISIONS.md`.

**Label Studio output format (Block 3D):**
- Corrections must be written in YOLO11 format — a directory with images and corresponding `.txt` label files — directly into `data/corrections/` so the fine-tuning script can point at it without preprocessing.

**Retraining gate (Block 3E):**
- YOLO fine-tuning is gated on YOLO **precision and recall** against COCO ground-truth bounding boxes on holdout images — not on Recall@10. Recall@10 measures CLIP-based search quality, and CLIP is not being fine-tuned. Fine-tuning YOLO will not move Recall@10. The new YOLO model is promoted only if it beats the current model on both precision and recall. If it does not improve, keep the old model and log the failure to MLflow.

---

## Section 9 — Phase 4: The Enterprise Platform

### Goal

The complete system. Governance, real-time streaming, full documentation, and public visibility. Looks and behaves like commercial DAM products.

### Targets

- 100% governance coverage of assets with face detections
- Real-time ingest latency under one second
- Architecture documented end to end
- Live demo accessible globally

### Skills demonstrated

Enterprise system design, governance and compliance, streaming vs batch architecture, technical writing, cross-functional communication, full platform engineering.

### Blocks

| Block | What it builds | Level | Done when |
|---|---|---|---|
| 4A | Governance layer — four new asset fields in PostgreSQL: `face_consent`, `usage_rights`, `expiry_date`, `ai_actions_log` (JSONB); three new endpoints: `GET /governance/expired`, `GET /governance/consent-missing`, `GET /governance/audit/{asset_id}`; governance tab in React | 3 — Block sprint | Endpoints return correct data, React tab shows expired and consent-missing lists |
| 4B | Real-time ingest — webcam or RTSP stream, sample 1 frame/sec, push each frame through the existing Celery pipeline, return tags to React via WebSocket within one second; mode toggle in UI | 2 — Guided | Tags appear within one second, both webcam and RTSP modes work, toggle works |
| 4C | Architecture diagram + DECISIONS.md — full system diagram covering all components and both the batch and streaming data paths, the drift loop, and the active learning loop; major architectural decisions documented with the measured data behind them | 3 — Block sprint | Diagram embedded in README, DECISIONS.md committed with data for every major choice |
| 4D | Usage analytics dashboard — log every search result retrieval; show top-retrieved assets and recurring zero-result queries in a new React tab | 3 — Block sprint | Analytics tab shows both lists with counts, numbers available for the README metrics table |
| 4E | "Find similar" endpoint — `GET /similar/{asset_id}` returns top-10 most visually similar assets using the asset's existing CLIP embedding; "Find similar" button on each image card in React | 3 — Block sprint | Endpoint returns visually sensible results, UI button works |
| 4F | Impact metrics + README polish — full measured metrics table; 30-second summary at top; quick start (Docker, under 5 min); badges; manual YOLO baseline on 200 held-out images to produce the 34% tagging error reduction figure | 3 — Block sprint | All numbers measured, 30-second summary tested on an outsider, quick start verified on a clean machine |
| 4G | Public presence — 2-minute Loom walkthrough; Medium or TDS article on one engineering decision; LinkedIn post with architecture diagram; v4.0 GitHub release with full changelog | 3 — Block sprint | Loom embedded in README, article published and linked, LinkedIn posted, v4.0 tagged |

### Data notes for Phase 4

**Manual YOLO baseline (Block 4F):**
- Take 200 images from `data/index/` (not holdout). For each image, write down the objects visually present. Compare against YOLO predictions. Compute precision and recall for the pre-fine-tuning model and the post-fine-tuning model. The delta is the "34% error reduction" figure.
- **Leakage rule:** do not sample from images that were sent to Label Studio for correction. Those images were effectively training data for the fine-tuned YOLO. Including them in the "after" evaluation inflates the number. Filter them out of `data/index/` before sampling. The 200 images must be ones the fine-tuned model has never seen with corrected labels.
- This takes about four hours of manual work. It is not optional.

**Time-saved estimate (Block 4F):**
- Time yourself manually tagging 20 images — writing tags and a caption for each. Get a per-image time. Extrapolate to a realistic weekly ingest volume (500 images per week for a mid-size marketing team). Compare against pipeline throughput. Multiply the difference by an assumed hourly rate. Document the assumptions explicitly — that is not a weakness, it is what a real business case looks like.

---

## Section 10 — How Claude Code is used

Vary how much control you give Claude Code based on how new the work is to you.

### Level 2 — Guided implementation

Use when the library, tool, or concept is new. One function or one class at a time. Read every line, understand it, ask questions before moving on.

Use Level 2 for:
- First time using a library (Qdrant, Celery, TensorRT, Evidently AI, Label Studio, WebSockets)
- Anything where you need to defend the design in an interview

Prompt shape:
```
Write only the embed_image function in app/pipeline/clip.py.
Input: a PIL Image object
Output: normalized numpy array of shape (512,)
Use the model that's already loaded as a module-level variable.
Do not write anything else in the file yet.
```

After: read it line by line, ask Claude Code to explain anything unclear, then ask for the next function.

### Level 3 — Block sprint

Use when the concept is familiar and you just need the boilerplate written fast. Provide a complete scoped prompt for the entire block — file structure, function signatures, acceptance criteria, constraints. Claude Code drafts everything. You run it, paste failures back, it fixes. You review the final result, not every line.

Use Level 3 for:
- Concepts you already understand (FastAPI endpoints, React components, Docker Compose, GitHub Actions YAML)
- Most blocks in this project

Prompt shape:
```
Create app/pipeline/clip.py

Requirements:
- load_model() loads CLIP ViT-B/32 onto CUDA, returns (model, preprocess)
- embed_image(pil_image, model, preprocess) -> np.ndarray shape (512,) normalized
- embed_text(query, model, preprocess) -> np.ndarray shape (512,) normalized
- Both embed functions log inference time in milliseconds to stdout
- Use torch.no_grad() and half precision throughout

Constraints:
- No classes, just functions
- No FastAPI imports — this is pure pipeline code
- Do not create any other files

Acceptance criteria:
- python -m app.pipeline.clip with one test image prints inference time
- Output vector has norm of 1.0
```

After: run the acceptance criteria. Pass = move on. Fail = paste the full traceback back (not a description of it).

### Level 4 — Phase sprint with checkpoints

Use only when the entire phase feels like familiar territory — typically Phase 3 or 4 after Phase 1 and 2 are done. One prompt covers the whole phase, broken into sequential tasks with explicit checkpoints. Claude Code stops after each task and waits for your verification before continuing.

You are not involved inside tasks — only at the boundaries between them.

### When to drop a level

If something is broken and you do not know why, drop to Level 2 for that piece. The cost of going slow on one debugging session is small. The cost of pretending to understand code you do not is large. Timebox debugging sessions to two hours; after that, move forward with a working alternative and return to the problem later.

### Block-to-level summary

| Block | Title | Level |
|---|---|---|
| 1A | Local pipeline proof of concept | 2 |
| 1B | Qdrant integration + eval contract | 2 |
| 1C | FastAPI backend | 3 |
| 1D | React frontend | 3 |
| 1E | Docker Compose + HF Spaces | 3 |
| 2A | Async with Celery + Redis | 2 |
| 2B | Batch ingestion endpoint | 3 |
| 2C | TensorRT optimization ⚠️ highest risk | 2 |
| 2D | A/B model comparison | 3 |
| 2E | Prometheus + Grafana | 3 |
| 2F | GitHub Actions CI/CD | 3 |
| 3A | Video keyframe extraction | 3 |
| 3B | Evidently AI drift detection | 2 |
| 3C | MLflow tracking + DVC | 3 |
| 3D | Label Studio active learning queue | 2 |
| 3E | Automated retraining + data flywheel | 3 |
| 4A | Governance layer | 3 |
| 4B | Real-time ingest | 2 |
| 4C | Architecture diagram + DECISIONS.md | 3 |
| 4D | Usage analytics dashboard | 3 |
| 4E | "Find similar" endpoint | 3 |
| 4F | Impact metrics + README polish | 3 |
| 4G | Public presence | 3 |

**Level 2 blocks — go slow, understand every line:** 1A, 1B, 2A, 2C, 3B, 3D, 4B.

---

## Section 11 — CLAUDE.md template

CLAUDE.md sits at the repo root. Claude Code reads it on every prompt. Keep it short and update the current block line as you move through the project.

```markdown
# VisualVault — Claude Code Reference

## Project structure
[paste current directory layout]

## Constraints that apply to every task
- Package manager is uv, not pip
- Python 3.11
- Windows with WSL2 for Docker
- GPU is RTX 5090, CUDA 12.x
- No synchronous model calls inside FastAPI — always Celery tasks
- All new files need a corresponding test in tests/

## Current phase and block
Phase 1, Block 1A — local pipeline proof of concept
No database, no API, no Docker yet.

## Acceptance criteria format
Every task must end with a verification command I can run locally.
```

---

## Section 12 — Industry standards this project follows

Every block should leave the codebase still meeting all of these.

- Every service runs in a Docker container.
- The entire system starts with one command (`docker-compose up`).
- All model experiments are tracked and versioned with MLflow. Datasets are versioned with DVC.
- No model goes to production without passing automated validation on a held-out set.
- Monitoring covers both infrastructure metrics (latency, queue depth, GPU) and ML-specific metrics (drift scores, confidence distributions).
- Human review is integrated into the pipeline, not bolted on after the fact.
- The README is written as a product document, not a code explanation.
- Every quantified impact number in the README is measured, not estimated.
- The holdout set is never indexed in Qdrant — no exceptions, no debugging shortcuts.
- `eval_pairs.json` is frozen after Block 1B and never edited.
- Every model promotion is gated on a metric measured against ground truth, not on inspection or intuition.

---

## Section 13 — How to talk about VisualVault in interviews

| After phase | What to say |
|---|---|
| Phase 1 | I built a semantic image search engine using CLIP and Qdrant. Sub-50ms search latency on 4,000 images. Here is the live demo — try searching for anything. |
| Phase 2 | I scaled it to 500 images per minute, cut inference latency 4× with TensorRT, and added full Prometheus/Grafana observability. I ran an A/B between two CLIP model sizes and chose ViT-B/32 because at our query volume the 6× latency improvement outweighed the 4% recall drop — I have the numbers. |
| Phase 3 | I added drift detection so the system knows when incoming images stop matching what it was trained on, and an active learning loop where uncertain predictions go to human review and corrections flow back into automated retraining. The loop runs on a 100-example trigger with a held-out validation gate — new model only promotes if precision and recall both improve. |
| Phase 4 | I added enterprise features — consent tracking, usage rights, audit logs, real-time streaming alongside batch — directly comparable to PhotoShelter and Bynder. Tagging accuracy improved 34% over four weeks. It is live right now if you want to try it. |

---

## Section 14 — Non-negotiables for the senior-manager voice

When Claude is acting as the senior ML engineering manager for this project:

- **Concept first, then code.** Always explain the why before the how.
- **Do not auto-complete entire files.** Guide component by component at Level 2. Full drafts at Level 3.
- **Skip basics.** Do not over-explain things the user already knows.
- **Push back** when the user is doing something that would not be acceptable in production.
- **Hold the user to the milestone definitions.** A block is not "done" until its acceptance criterion verifies clean.
- **Do not let the user skip documentation, benchmarking, and presentation work** at the end of each phase. Those steps are as important as the code.
- **Timebox debugging sessions to two hours.** After that, move forward with a working alternative and return later.
- **Every quantified claim in the README must come from a measured number, not an estimate.**

---

## Section 15 — How to use this document

**Starting a new Claude chat:**

1. Paste this entire file as the first message, or attach it to the project.
2. Add: `"I am the user described in this document. Today I am working on [Phase X, Block Y]."`
3. Add the operating level rule: `"Treat me as a senior ML engineer treats a competent junior. Concept first, then code. Do not auto-complete entire files unless the block is at Level 3 or higher. Push back when I am cutting corners."`

**Using Claude Code:**

1. Save this file alongside `CLAUDE.md` at the repo root, or reference it from within `CLAUDE.md`.
2. In `CLAUDE.md` itself, keep only the constraints, current phase/block pointer, and acceptance-criteria format. The full project context lives in this knowledge base.
3. Every prompt to Claude Code should specify the level (2, 3, or 4) and end with an acceptance criterion that is a runnable verification command.
4. If a block is failing, drop to Level 2 for the failing piece and walk through it function by function. Do not describe errors — paste the full traceback.
