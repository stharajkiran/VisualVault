import numpy as np
import tensorrt as trt
import torch
from pathlib import Path
from PIL import Image
from transformers import CLIPProcessor

ENGINE_PATH = Path("artifacts/tensorrt/clip_image_encoder.engine")
PROCESSOR_NAME = "openai/clip-vit-base-patch32"


def load_engine() -> tuple[trt.ICudaEngine, trt.IExecutionContext, CLIPProcessor]:
    """
    Load the TensorRT engine from disk and create an execution context.

    Deserializes the compiled .engine file into a TRT runtime engine and
    creates an execution context — the stateful object that holds GPU
    memory allocations and runs inference. Also loads the CLIP processor
    for image preprocessing (resize, normalize) which stays in PyTorch.

    Args:
        None

    Returns:
        tuple: (engine, context, processor) where engine is the TRT ICudaEngine,
               context is the IExecutionContext, and processor is the CLIPProcessor.
    """
    logger = trt.Logger(trt.Logger.WARNING)
    runtime = trt.Runtime(logger)

    print(f"[CLIP-TRT] Loading engine from {ENGINE_PATH} ...")
    with open(ENGINE_PATH, "rb") as f:
        engine = runtime.deserialize_cuda_engine(f.read())

    if engine is None:
        raise RuntimeError(f"Failed to deserialize engine from {ENGINE_PATH}")

    context = engine.create_execution_context()
    processor = CLIPProcessor.from_pretrained(PROCESSOR_NAME)
    print("[CLIP-TRT] Engine loaded.")
    return engine, context, processor


def embed_image_trt(
    image: Image.Image,
    context: trt.IExecutionContext,
    processor: CLIPProcessor,
) -> np.ndarray:
    """
    Embed a PIL image using the TensorRT CLIP encoder.

    Drop-in replacement for clip.embed_image(). Returns a normalized
    512-dim float32 embedding.

    Args:
        image (Image.Image): Input PIL image in any mode — converted to RGB internally.
        context (trt.IExecutionContext): Execution context from load_engine().
        processor (CLIPProcessor): CLIP processor for preprocessing.

    Returns:
        np.ndarray: Shape (512,), dtype float32, L2-norm = 1.0.
    """
    # Preprocess on CPU, move to GPU as float32
    # processor returns a (1, 3, 224, 224) tensor — fixed shape, matches engine
    inputs = processor(images=image.convert("RGB"), return_tensors="pt")
    pixel_tensor = inputs["pixel_values"].to(dtype=torch.float32, device="cuda")

    # Allocate output buffer on GPU
    output_tensor = torch.empty((1, 512), dtype=torch.float32, device="cuda")

    # Bind tensor addresses — TRT writes directly into output_tensor's GPU memory
    context.set_input_shape("pixel_values", tuple(pixel_tensor.shape))
    context.set_tensor_address("pixel_values", pixel_tensor.data_ptr())
    context.set_tensor_address("embeddings", output_tensor.data_ptr())

    # Run inference on the current PyTorch CUDA stream
    # Using the same stream as PyTorch means no cross-stream race conditions
    current_stream = torch.cuda.current_stream()
    context.execute_async_v3(stream_handle=current_stream.cuda_stream)

    # Synchronize this stream before reading output — ensures GPU write is complete
    current_stream.synchronize()

    # Move to CPU, squeeze batch dim, normalize to unit vector
    vec = output_tensor.cpu().numpy().squeeze()
    return vec / np.linalg.norm(vec)
