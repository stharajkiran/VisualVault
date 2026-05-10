# VisualVault — Claude Code Reference

## Project structure
```
visualvault/
  app/
    pipeline/     # ML model wrappers (CLIP, YOLO, BLIP-2)
    api/          # FastAPI endpoints
    db/           # SQLAlchemy models and session
  scripts/        # run_pipeline.py and data utilities
  data/
    raw/          # all 5,000 COCO val2017 images, never modified
    index/        # 4,000 images — what goes into Qdrant
    holdout/      # 1,000 images — NEVER indexed, NEVER shown in demos
    ood_test/     # out-of-distribution images for drift testing (Phase 3)
    corrections/  # human-verified YOLO labels from Label Studio (Phase 3)
  assets/         # test images for local development (before MinIO)
  tests/          # one test file per module
  .github/
    workflows/    # CI/CD (Phase 2)
```

## Constraints that apply to every task
- Package manager is uv, not pip
- Python 3.11
- Windows 11 with WSL2 for Docker
- GPU is RTX 5090, CUDA 12.x (cu128)
- No synchronous model calls inside FastAPI — always Celery tasks (Phase 2+)
- All new files need a corresponding test in tests/
- data/holdout/ is NEVER indexed in Qdrant — no exceptions
- eval_pairs.json is frozen after Block 1B — never edited again

## Current phase and block
Phase 1, Block 1B — Qdrant integration and the evaluation contract.
Establish data split (4,000 index / 1,000 holdout), index the 4,000 images,
hand-write 200 query pairs from holdout images into eval_pairs.json,
run Recall@10 and record the baseline.

## Acceptance criteria format
Every task must end with a verification command I can run locally.

## Block 1B acceptance criteria
- data/index/ has 4,000 images, data/holdout/ has 1,000 images
- All 4,000 images are indexed in Qdrant (collection: visualvault)
- eval_pairs.json committed with 200 hand-written query pairs
- Recall@10 baseline measured and recorded (expect 0.70–0.78)
- Search query returns results in under 100ms locally
