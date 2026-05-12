# Pre-allocate output buffer once, reuse across calls
output_tensor = torch.empty((1, 512), dtype=torch.float32, device="cuda")
pixel_tensor = torch.randn(1, 3, 224, 224, dtype=torch.float32, device="cuda")

context.set_input_shape("pixel_values", (1, 3, 224, 224))
context.set_tensor_address("pixel_values", pixel_tensor.data_ptr())
context.set_tensor_address("embeddings", output_tensor.data_ptr())

stream = torch.cuda.current_stream()
latencies = []
for _ in range(200):
    t0 = time.perf_counter()
    context.execute_async_v3(stream_handle=stream.cuda_stream)
    stream.synchronize()
    latencies.append((time.perf_counter() - t0) * 1000)

print(f"TRT raw engine p50: {np.percentile(latencies, 50):.2f}ms")
