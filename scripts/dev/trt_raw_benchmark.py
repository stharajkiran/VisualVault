"""
Measures pure TRT engine execution time with no preprocessing or allocation overhead.
Used to isolate whether the bottleneck is in the engine itself or in the Python wrapper.

Run with:
    python scripts/dev/trt_raw_benchmark.py
"""

import time
import numpy as np
import torch
import tensorrt as trt
from app.embedding.clip_tensorrt import load_engine

WARMUP = 20
RUNS = 200

engine, context, _ = load_engine()

# Pre-allocate input and output buffers once — reused across all runs
pixel_tensor = torch.randn(1, 3, 224, 224, dtype=torch.float32, device="cuda")
output_tensor = torch.empty((1, 512), dtype=torch.float32, device="cuda")

context.set_input_shape("pixel_values", (1, 3, 224, 224))
context.set_tensor_address("pixel_values", pixel_tensor.data_ptr())
context.set_tensor_address("embeddings", output_tensor.data_ptr())

stream = torch.cuda.current_stream()

# Warmup
for _ in range(WARMUP):
    context.execute_async_v3(stream_handle=stream.cuda_stream)
stream.synchronize()

# Benchmark
latencies = []
for _ in range(RUNS):
    t0 = time.perf_counter()
    context.execute_async_v3(stream_handle=stream.cuda_stream)
    stream.synchronize()
    latencies.append((time.perf_counter() - t0) * 1000)

p50 = np.percentile(latencies, 50)
p95 = np.percentile(latencies, 95)
print(f"TRT raw engine — p50: {p50:.2f}ms  p95: {p95:.2f}ms")
print("(No preprocessing, no buffer allocation — pure engine execution only)")
