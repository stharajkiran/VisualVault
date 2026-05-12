"""
Compile CLIP's ONNX image encoder into a TensorRT engine.

Reads artifacts/clip_image_encoder.onnx and produces
artifacts/clip_image_encoder.engine optimized for the local GPU.

WARNING: The engine file is hardware-specific — it will only run on the
GPU it was compiled on. Do not copy .engine files between machines.

Compilation takes 5–15 minutes. The engine is saved to disk so this
only needs to run once.

Run with:
    python scripts/model/compile_clip_trt.py
"""

import tensorrt as trt
from pathlib import Path

ONNX_PATH = Path("artifacts/onnx/clip_image_encoder.onnx")
ENGINE_PATH = Path("artifacts/tensorrt/clip_image_encoder.engine")

# Batch size range for the optimization profile
BATCH_MIN = 1
BATCH_OPT = 1  # optimize for single-image inference (most common case)
BATCH_MAX = 8


def build_engine(onnx_path: Path, engine_path: Path) -> None:
    """
    Parse an ONNX file and compile it into a TensorRT FP16 engine.

    Creates a TensorRT builder, enables FP16 precision, sets a dynamic
    batch optimization profile, and serializes the compiled engine to disk.
    Compilation benchmarks multiple CUDA kernel implementations per layer
    and picks the fastest — this is what makes it slow but the result fast.

    Args:
        onnx_path (Path): Path to the input .onnx file.
        engine_path (Path): Path where the compiled .engine file is written.

    Returns:
        None
    """
    logger = trt.Logger(trt.Logger.INFO)
    builder = trt.Builder(logger)
    network = builder.create_network(
        # EXPLICIT_BATCH flag is required for dynamic batch support
        1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    )
    # Reads the .onnx file and populates the empty network with layers and tensors.
    parser = trt.OnnxParser(network, logger)

    print(f"[TRT] Parsing ONNX from {onnx_path} ...")
    # parse_from_file lets TRT resolve the external .data file relative to
    # the .onnx file's own directory — fixes "Failed to open .onnx.data" error
    if not parser.parse_from_file(str(onnx_path.absolute())):
        for i in range(parser.num_errors):
            print(f"  ERROR: {parser.get_error(i)}")
        raise RuntimeError("ONNX parsing failed — see errors above.")
    print("[TRT] ONNX parsed successfully.")

    config = builder.create_builder_config()

    # FP16 mode — halves memory bandwidth, ~2x speedup on tensor cores
    if builder.platform_has_fast_fp16:
        config.set_flag(trt.BuilderFlag.FP16)
        print("[TRT] FP16 mode enabled.")
    else:
        print("[TRT] WARNING: FP16 not supported on this GPU, using FP32.")

    # Optimization profile — tells TRT the batch size range to optimize for
    profile = builder.create_optimization_profile()
    profile.set_shape(
        "pixel_values",
        min=(BATCH_MIN, 3, 224, 224),
        opt=(BATCH_OPT, 3, 224, 224),
        max=(BATCH_MAX, 3, 224, 224),
    )
    config.add_optimization_profile(profile)

    print("[TRT] Building engine — this will take 5–15 minutes ...")
    # compilation step where TensorRT benchmarks different implementations of each layer
    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        raise RuntimeError("Engine build failed — check GPU memory and TRT logs above.")

    engine_path.parent.mkdir(parents=True, exist_ok=True)
    with open(engine_path, "wb") as f:
        f.write(serialized)

    size_mb = engine_path.stat().st_size / (1024 ** 2)
    print(f"[TRT] Engine saved to {engine_path} ({size_mb:.1f} MB)")


if __name__ == "__main__":
    if not ONNX_PATH.exists():
        raise FileNotFoundError(
            f"{ONNX_PATH} not found. Run scripts/model/export_clip_onnx.py first."
        )
    build_engine(ONNX_PATH, ENGINE_PATH)
