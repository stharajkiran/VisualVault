"""
Log all historical VisualVault eval measurements into MLflow.

Run once after `docker-compose up mlflow` to populate the UI with
all measured runs from Blocks 1B, 2C, 2D, and 3B.

Run with:
    uv run python scripts/eval/log_experiments.py
    uv run python scripts/eval/log_experiments.py --tracking-uri http://localhost:5000
"""

import argparse

import mlflow


def log_recall_runs(tracking_uri: str) -> None:
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment("visualvault/recall-evaluation")

    with mlflow.start_run(run_name="clip-b32-coco-val"):
        mlflow.log_params({
            "model": "clip-b32",
            "model_id": "openai/clip-vit-base-patch32",
            "embedding_dim": 512,
            "dataset": "coco-val2017",
            "index_size": 4000,
            "n_queries": 200,
            "top_k": 10,
            "query_source": "gpt-4o-mini",
        })
        mlflow.log_metrics({
            "recall_at_10": 0.94,
            "search_latency_avg_ms": 13.7,
            "search_latency_p95_ms": 15.0,
            "indexing_throughput_img_s": 55.0,
        })

    with mlflow.start_run(run_name="clip-l14-coco-val"):
        mlflow.log_params({
            "model": "clip-l14",
            "model_id": "openai/clip-vit-large-patch14",
            "embedding_dim": 768,
            "dataset": "coco-val2017",
            "index_size": 4000,
            "n_queries": 200,
            "top_k": 10,
            "query_source": "gpt-4o-mini",
        })
        mlflow.log_metrics({
            "recall_at_10": 0.955,
            "indexing_throughput_img_s": 38.4,
        })


def log_inference_benchmark_runs(tracking_uri: str) -> None:
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment("visualvault/inference-benchmark")

    with mlflow.start_run(run_name="clip-b32-pytorch"):
        mlflow.log_params({
            "model": "clip-b32",
            "backend": "pytorch",
            "device": "cuda",
            "gpu": "RTX 5090",
            "n_runs": 2000,
        })
        mlflow.log_metrics({
            "preprocessing_p50_ms": 56.51,
            "preprocessing_p95_ms": 62.04,
            "inference_p50_ms": 3.58,
            "inference_p95_ms": 4.08,
            "speedup_vs_pytorch": 1.0,
        })

    with mlflow.start_run(run_name="clip-b32-tensorrt"):
        mlflow.log_params({
            "model": "clip-b32",
            "backend": "tensorrt",
            "precision": "fp16",
            "device": "cuda",
            "gpu": "RTX 5090",
            "n_runs": 2000,
        })
        mlflow.log_metrics({
            "preprocessing_p50_ms": 56.51,
            "preprocessing_p95_ms": 62.04,
            "inference_p50_ms": 0.94,
            "inference_p95_ms": 1.34,
            "speedup_vs_pytorch": 3.8,
        })

    with mlflow.start_run(run_name="clip-l14-pytorch"):
        mlflow.log_params({
            "model": "clip-l14",
            "backend": "pytorch",
            "device": "cuda",
            "gpu": "RTX 5090",
            "n_runs": 2000,
        })
        mlflow.log_metrics({
            "preprocessing_p50_ms": 54.34,
            "inference_p50_ms": 11.96,
            "inference_p95_ms": 12.91,
            "speedup_vs_pytorch": 1.0,
        })


def log_drift_runs(tracking_uri: str) -> None:
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment("visualvault/drift-detection")

    with mlflow.start_run(run_name="ood-test-batch"):
        mlflow.log_params({
            "baseline_model": "clip-b32",
            "baseline_size": 4003,
            "batch_size": 10,
            "batch_source": "data/ood_test/",
            "distance_metric": "cosine",
            "alert_threshold": 0.15,
        })
        mlflow.log_metrics({
            "drift_score": 0.339487,
            "threshold": 0.15,
        })
        mlflow.set_tag("status", "DRIFT DETECTED")


def main() -> None:
    parser = argparse.ArgumentParser(description="Log historical VisualVault eval runs to MLflow.")
    parser.add_argument("--tracking-uri", type=str, default="http://localhost:5000", help="MLflow tracking server URI")
    args = parser.parse_args()

    print(f"Logging to MLflow at {args.tracking_uri} ...")

    log_recall_runs(args.tracking_uri)
    print("  ✓ recall-evaluation  (2 runs)")

    log_inference_benchmark_runs(args.tracking_uri)
    print("  ✓ inference-benchmark  (3 runs)")

    log_drift_runs(args.tracking_uri)
    print("  ✓ drift-detection  (1 run)")

    print(f"\nDone. Open {args.tracking_uri} to view all runs.")


if __name__ == "__main__":
    main()
