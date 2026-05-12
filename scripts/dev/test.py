import time
from app.embedding import clip_pytorch as clip
from app.vector_store import qdrant as q

client = q.get_client()
model, processor = clip.load_model()

# Embed once (not part of the latency we're measuring)
vec = clip.embed_text('a person on a bicycle', model, processor)

# Measure Qdrant search latency across 10 queries
times = []
for _ in range(10):
    t0 = time.perf_counter()
    q.search(client, vec, top_k=10)
    times.append((time.perf_counter() - t0) * 1000)

print(f'Qdrant search — avg: {sum(times)/len(times):.1f}ms, p95: {sorted(times)[9]:.1f}ms')
