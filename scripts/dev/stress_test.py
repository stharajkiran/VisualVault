"""
Stress test: sends N concurrent upload requests and shows how they pile up.

Run with:
    python scripts/stress_test.py

Requires the FastAPI server to be running:
    uvicorn app.api.main:app --reload
"""

import time
import threading
import requests

URL = "http://localhost:8000/upload"
IMAGE_PATH = "assets/img.jpg"
NUM_REQUESTS = 20


def upload(index: int, results: list) -> None:
    """
    Send one upload request and record timing.

    Args:
        index (int): Request number, used for display ordering.
        results (list): Shared list where this thread appends its timing dict.

    Returns:
        None
    """
    with open(IMAGE_PATH, "rb") as f:
        image_bytes = f.read()

    sent_at = time.perf_counter()
    print(f"  [req {index}] sent at t=0.000s")

    response = requests.post(
        URL,
        files={"file": ("img.jpg", image_bytes, "image/jpeg")},
    )

    elapsed = time.perf_counter() - sent_at
    status = response.status_code
    results.append({"index": index, "elapsed": elapsed, "status": status})
    print(f"  [req {index}] done  at t={elapsed:.3f}s  (HTTP {status})")


def main():
    print(f"\nFiring {NUM_REQUESTS} concurrent uploads at {URL}")
    print(f"Image: {IMAGE_PATH}")
    print("-" * 55)
    print("If requests ran in true parallel, all would finish at ~same time.")
    print("If they serialize, each one adds ~244ms on top of the last.")
    print("-" * 55)

    results = []
    threads = [
        threading.Thread(target=upload, args=(i + 1, results))
        for i in range(NUM_REQUESTS)
    ]

    wall_start = time.perf_counter()
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    wall_total = time.perf_counter() - wall_start

    print("-" * 55)
    results.sort(key=lambda r: r["index"])
    print("\nSummary:")
    for r in results:
        print(f"  req {r['index']}: {r['elapsed']:.3f}s")

    fastest = min(r["elapsed"] for r in results)
    slowest = max(r["elapsed"] for r in results)
    print(f"\n  Fastest response : {fastest:.3f}s")
    print(f"  Slowest response : {slowest:.3f}s")
    print(f"  Total wall time  : {wall_total:.3f}s")
    print(f"\n  Ideal (parallel) : all finish near {fastest:.3f}s")
    print(f"  Actual penalty   : slowest req waited {slowest - fastest:.3f}s extra")

    if slowest > fastest * 2:
        print("\n  RESULT: requests SERIALIZED — each upload blocked the next.")
    else:
        print("\n  RESULT: requests ran roughly in parallel.")


if __name__ == "__main__":
    main()
