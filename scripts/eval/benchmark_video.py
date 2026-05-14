"""
Benchmark video keyframe extraction throughput.

Generates a synthetic test video using FFmpeg's testsrc filter, runs frame
extraction at 0.5 fps, and reports processing time per minute of footage.

Run with:
    uv run python scripts/eval/benchmark_video.py
    uv run python scripts/eval/benchmark_video.py --duration 120
"""

import argparse
import tempfile
import time
from pathlib import Path

import ffmpeg


def generate_test_video(output_path: Path, duration_s: int, resolution: str = "1280x720") -> None:
    """
    Generate a synthetic test video using FFmpeg's testsrc lavfi filter.

    Args:
        output_path (Path): Destination .mp4 file path.
        duration_s (int): Duration in seconds.
        resolution (str): Width x height string, e.g. "1280x720".
    """
    (
        ffmpeg
        .input(f"testsrc=duration={duration_s}:size={resolution}:rate=30", format="lavfi")
        .output(str(output_path), vcodec="libx264", crf=23, preset="ultrafast")
        .run(quiet=True, overwrite_output=True)
    )


def extract_frames(video_path: Path, output_dir: Path) -> list[Path]:
    """
    Extract one frame every 2 seconds from a video into output_dir.

    Args:
        video_path (Path): Source video file.
        output_dir (Path): Directory to write frame_NNNN.jpg files into.

    Returns:
        list[Path]: Sorted list of extracted frame paths.
    """
    frame_pattern = str(output_dir / "frame_%04d.jpg")
    (
        ffmpeg.input(str(video_path))
        .filter("fps", fps=0.5)
        .output(frame_pattern, vcodec="mjpeg", qscale=2)
        .run(quiet=True, overwrite_output=True)
    )
    return sorted(output_dir.glob("frame_*.jpg"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark video frame extraction throughput.")
    parser.add_argument("--duration", type=int, default=60, help="Synthetic video duration in seconds (default: 60)")
    parser.add_argument("--resolution", type=str, default="1280x720", help="Video resolution (default: 1280x720)")
    args = parser.parse_args()

    print(f"Generating {args.duration}s synthetic video at {args.resolution} ...")

    with tempfile.TemporaryDirectory() as tmpdir:
        tmppath = Path(tmpdir)
        video_path = tmppath / "test_video.mp4"
        frames_dir = tmppath / "frames"
        frames_dir.mkdir()

        t0 = time.perf_counter()
        generate_test_video(video_path, args.duration, args.resolution)
        gen_time = time.perf_counter() - t0

        video_mb = video_path.stat().st_size / 1024 / 1024
        print(f"Video ready: {video_mb:.1f} MB, generated in {gen_time:.1f}s")

        print("Extracting frames ...")
        t0 = time.perf_counter()
        frames = extract_frames(video_path, frames_dir)
        extract_time = time.perf_counter() - t0

    expected = args.duration // 2
    footage_minutes = args.duration / 60
    processing_per_minute = extract_time / footage_minutes

    print("\n── Results ──────────────────────────────────")
    print(f"  Footage duration    {args.duration}s ({footage_minutes:.1f} min)")
    print(f"  Frames extracted    {len(frames)}  (expected {expected})")
    print(f"  Extraction time     {extract_time:.2f}s")
    print(f"  Throughput          {processing_per_minute:.2f}s processing / min of footage")
    print(f"  Realtime ratio      {extract_time / args.duration:.3f}×  (1.0 = real-time)")


if __name__ == "__main__":
    main()
