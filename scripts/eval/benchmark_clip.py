"""
Benchmark CLIP inference: PyTorch vs TensorRT.

Preprocessing is measured separately and excluded from inference timing.
Inference timing covers: GPU transfer, model forward pass, CPU transfer, normalization.

Run with:
    python scripts/eval/benchmark_clip.py
"""

import time
import numpy as np
import torch
from PIL import Image
import tensorrt as trt
from app.embedding.clip_pytorch import load_model
from app.embedding.clip_tensorrt import load_engine

TRT_LOGGER = trt.Logger(trt.Logger.ERROR)
img = Image.open("assets/test_img.jpg").convert("RGB")
WARMUP = 20
RUNS = 2000

# ── Load models ──
print("Loading PyTorch CLIP ...")
pt_model, pt_processor = load_model()

print("Loading TRT engine ...")
engine, context, trt_processor = load_engine()

stream = torch.cuda.current_stream()

###################################################################################################
# ── Preprocessing benchmark ──
latencies = []
for _ in range(WARMUP):
    pt_processor(images=img, return_tensors="pt")
for _ in range(RUNS):
    t0 = time.perf_counter()
    pt_processor(images=img, return_tensors="pt")
    latencies.append((time.perf_counter() - t0) * 1000)

pre_p50 = np.percentile(latencies, 50)
pre_p95 = np.percentile(latencies, 95)
print("\nPreprocessing (CPU, excluded from inference timing below)")
print(f"  p50: {pre_p50:.2f}ms  p95: {pre_p95:.2f}ms")

# Pre-process once — used as input for both inference benchmarks
pixel_values = pt_processor(images=img, return_tensors="pt")["pixel_values"].to(dtype=torch.float32)
pixel_values_gpu = pixel_values.to(device="cuda")

###################################################################################################
# ── PyTorch inference benchmark ──
# Timing covers: GPU transfer → forward pass → CPU transfer → normalize
for _ in range(WARMUP):
    with torch.no_grad():
        feat = pt_model.get_image_features(pixel_values=pixel_values_gpu)
    vec = feat.pooler_output.squeeze().cpu().numpy()
    vec = vec / np.linalg.norm(vec)
torch.cuda.synchronize()

latencies = []
for _ in range(RUNS):
    t0 = time.perf_counter()
    with torch.no_grad():
        feat = pt_model.get_image_features(pixel_values=pixel_values_gpu)
    vec = feat.pooler_output.squeeze().cpu().numpy()
    vec = vec / np.linalg.norm(vec)
    torch.cuda.synchronize()
    latencies.append((time.perf_counter() - t0) * 1000)

pt_p50 = np.percentile(latencies, 50)
pt_p95 = np.percentile(latencies, 95)
print("\nPyTorch inference (forward + CPU transfer + normalize)")
print(f"  p50: {pt_p50:.2f}ms  p95: {pt_p95:.2f}ms")

###################################################################################################
# ── TRT inference benchmark ──
# Timing covers: GPU transfer → TRT execute → CPU transfer → normalize
output_tensor = torch.empty((1, 512), dtype=torch.float32, device="cuda")

output_tensor = torch.empty((1, 512), dtype=torch.float32, device="cuda")
context.set_input_shape("pixel_values", tuple(pixel_values_gpu.shape))
context.set_tensor_address("pixel_values", pixel_values_gpu.data_ptr())
context.set_tensor_address("embeddings", output_tensor.data_ptr())

for _ in range(WARMUP):
    context.execute_async_v3(stream_handle=stream.cuda_stream)
    stream.synchronize()
    vec = output_tensor.cpu().numpy().squeeze()
    vec = vec / np.linalg.norm(vec)
stream.synchronize()

latencies = []
for _ in range(RUNS):
    t0 = time.perf_counter()
    context.execute_async_v3(stream_handle=stream.cuda_stream)
    stream.synchronize()
    vec = output_tensor.cpu().numpy().squeeze()
    vec = vec / np.linalg.norm(vec)
    latencies.append((time.perf_counter() - t0) * 1000)

trt_p50 = np.percentile(latencies, 50)
trt_p95 = np.percentile(latencies, 95)
print("\nTensorRT inference (execute + CPU transfer + normalize)")
print(f"  p50: {trt_p50:.2f}ms  p95: {trt_p95:.2f}ms")

# ── Summary ──
print(f"\n{'─'*50}")
print(f"  Preprocessing (shared, excluded) : {pre_p50:.2f}ms p50")
print(f"  PyTorch inference                : {pt_p50:.2f}ms p50")
print(f"  TensorRT inference               : {trt_p50:.2f}ms p50")
print(f"  Inference speedup                : {pt_p50/trt_p50:.1f}x")
