"""
Batch ingestion script — dispatches a folder of images to the Celery worker
and reports throughput on completion.

Run with:
    python scripts/data/batch_ingest.py --folder data/index --limit 100
"""

import argparse
import time
from pathlib import Path

from tqdm import tqdm

from app.worker.tasks import ingest_image

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}


def collect_image_paths(folder: str, limit: int | None) -> list[Path]:
    """
    Collect image file paths from a folder, optionally capped at a limit.

    Args:
        folder (str): Path to the folder to scan for images.
        limit (int | None): Maximum number of images to return. None means all.

    Returns:
        list[Path]: Sorted list of image file paths.
    """
    paths = sorted(
        p for p in Path(folder).iterdir() if p.suffix.lower() in SUPPORTED_EXTENSIONS
    )
    return paths[:limit] if limit else paths


def dispatch_tasks(image_paths: list[Path]) -> list[tuple[Path, object]]:
    """
    Dispatch one Celery ingest_image task per image without waiting for results.

    Reads each image file and calls ingest_image.delay() to push it onto the
    Celery queue. Returns immediately after all tasks are dispatched — does not
    wait for any pipeline to finish.

    Args:
        image_paths (list[Path]): List of image file paths to ingest.

    Returns:
        list[tuple[Path, object]]: Each entry is (image_path, AsyncResult).
    """
    jobs = []
    print(f"Dispatching {len(image_paths)} tasks ...")
    for path in tqdm(image_paths):
        image_bytes = path.read_bytes()
        result = ingest_image.delay(path.name, image_bytes)
        jobs.append((path, result))
    return jobs


def poll_until_done(jobs: list[tuple[Path, object]]) -> dict:
    """
    Poll all dispatched jobs until every one reaches SUCCESS or FAILURE.

    Repeatedly checks job states in a loop, sleeping briefly between rounds
    to avoid hammering Redis. Tracks and prints a live progress bar.

    Args:
        jobs (list[tuple[Path, object]]): Output from dispatch_tasks().

    Returns:
        dict: Keys — succeeded (int), failed (int), total (int).
    """
    total = len(jobs)
    print(f"\nWaiting for {total} tasks to complete ...")

    done = set()
    failed = set()
    pbar = tqdm(total=total, unit="img")

    while len(done) + len(failed) < total:
        for path, result in jobs:
            key = str(path)
            if key in done or key in failed:
                continue
            if result.state == "SUCCESS":
                done.add(key)
                pbar.update(1)
            elif result.state == "FAILURE":
                failed.add(key)
                pbar.update(1)
                tqdm.write(f"  FAILED: {path.name} — {result.result}")
        time.sleep(0.5)

    pbar.close()
    return {"succeeded": len(done), "failed": len(failed), "total": total}


def main():
    parser = argparse.ArgumentParser(description="Batch ingest images via Celery.")
    parser.add_argument(
        "--folder", default="data/index", help="Folder of images to ingest."
    )
    parser.add_argument(
        "--limit", type=int, default=None, help="Max images to process (default: all)."
    )
    args = parser.parse_args()

    image_paths = collect_image_paths(args.folder, args.limit)
    if not image_paths:
        print(f"No images found in {args.folder}")
        return

    print(f"\nBatch ingestion — {len(image_paths)} images from {args.folder}")
    print("-" * 55)

    wall_start = time.perf_counter()
    # Dispatch one Celery ingest_image task per image without waiting for results
    jobs = dispatch_tasks(image_paths)
    # Poll all dispatched jobs until every one reaches SUCCESS or FAILURE.
    summary = poll_until_done(jobs)
    wall_total = time.perf_counter() - wall_start

    imgs_per_sec = summary["succeeded"] / wall_total
    imgs_per_min = imgs_per_sec * 60

    print("\n" + "-" * 55)
    print(f"  Succeeded   : {summary['succeeded']} / {summary['total']}")
    print(f"  Failed      : {summary['failed']}")
    print(f"  Wall time   : {wall_total:.1f}s")
    print(f"  Throughput  : {imgs_per_sec:.1f} img/s  ({imgs_per_min:.0f} img/min)")
    print(f"\n  Target      : 500 img/min (requires Docker worker with prefork)")

    if imgs_per_min >= 500:
        print("  RESULT: target met ✓")
    else:
        print(
            f"  RESULT: {imgs_per_min:.0f} img/min — below target (expected on Windows solo worker)"
        )


if __name__ == "__main__":
    main()
