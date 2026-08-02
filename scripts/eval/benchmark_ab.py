"""
Latency benchmark for A/B model comparison: ViT-B/32 vs ViT-L/14.

Measures preprocessing and inference separately for a fair comparison.
Both models use the PyTorch backend under identical conditions.

Run with:
    python scripts/eval/benchmark_ab.py
"""

import time
import numpy as np
import torch
from PIL import Image

from app.embedding.config import CLIP_BASE, CLIP_LARGE
from app.embedding.provider import CLIPPyTorchProvider

IMG_PATH = "assets/test_img.jpg"
WARMUP = 20
RUNS = 200

img = Image.open(IMG_PATH).convert("RGB")


def benchmark(provider: CLIPPyTorchProvider) -> dict:
    """
    Measure preprocessing and inference latency for a provider.

    Args:
        provider (CLIPPyTorchProvider): Loaded and warmed-up provider.

    Returns:
        dict: Keys — pre_p50, pre_p95, inf_p50, inf_p95 (all in ms).
    """
    # Preprocessing benchmark
    pre_latencies = []
    for _ in range(WARMUP):
        provider._processor(images=img, return_tensors="pt")
    for _ in range(RUNS):
        t0 = time.perf_counter()
        provider._processor(images=img, return_tensors="pt")
        pre_latencies.append((time.perf_counter() - t0) * 1000)

    # Pre-process once for inference benchmark
    inputs = provider._processor(images=img, return_tensors="pt")
    pixel_values = inputs["pixel_values"].to(dtype=torch.float32, device="cuda")

    # Inference benchmark
    inf_latencies = []
    for _ in range(WARMUP):
        with torch.no_grad():
            provider._model.get_image_features(pixel_values=pixel_values)
    torch.cuda.synchronize()
    for _ in range(RUNS):
        t0 = time.perf_counter()
        with torch.no_grad():
            provider._model.get_image_features(pixel_values=pixel_values)
        torch.cuda.synchronize()
        inf_latencies.append((time.perf_counter() - t0) * 1000)

    return {
        "pre_p50": np.percentile(pre_latencies, 50),
        "pre_p95": np.percentile(pre_latencies, 95),
        "inf_p50": np.percentile(inf_latencies, 50),
        "inf_p95": np.percentile(inf_latencies, 95),
    }


print("Loading ViT-B/32 ...")
base_provider = CLIPPyTorchProvider(CLIP_BASE)
base_provider.warmup()

print("Loading ViT-L/14 ...")
large_provider = CLIPPyTorchProvider(CLIP_LARGE)
large_provider.warmup()

print("\nBenchmarking ViT-B/32 ...")
base_results = benchmark(base_provider)

print("Benchmarking ViT-L/14 ...")
large_results = benchmark(large_provider)

print(f"\n{'─'*55}")
print(f"{'Metric':<30} {'ViT-B/32':>10} {'ViT-L/14':>10}")
print(f"{'─'*55}")
print(
    f"{'Preprocessing p50 (ms)':<30} {base_results['pre_p50']:>10.2f} {large_results['pre_p50']:>10.2f}"
)
print(
    f"{'Preprocessing p95 (ms)':<30} {base_results['pre_p95']:>10.2f} {large_results['pre_p95']:>10.2f}"
)
print(
    f"{'Inference p50 (ms)':<30} {base_results['inf_p50']:>10.2f} {large_results['inf_p50']:>10.2f}"
)
print(
    f"{'Inference p95 (ms)':<30} {base_results['inf_p95']:>10.2f} {large_results['inf_p95']:>10.2f}"
)
print(f"{'─'*55}")
print(
    f"{'Inference slowdown':<30} {'1.0×':>10} {large_results['inf_p50']/base_results['inf_p50']:>9.1f}×"
)
