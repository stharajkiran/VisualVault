import logging
import time

import torch
from PIL import Image
from transformers import Blip2ForConditionalGeneration, Blip2Processor

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_NAME = "Salesforce/blip2-opt-2.7b"
MAX_NEW_TOKENS = 50
logger = logging.getLogger(__name__)


def load_model() -> tuple[Blip2ForConditionalGeneration, Blip2Processor]:
    """
    Load BLIP-2 onto the GPU in float16.

    BLIP-2 combines a vision encoder (ViT) with a large language model (OPT-2.7B)
    connected by a lightweight bridge called Q-Former. The LLM side is what makes
    it generative — it predicts caption tokens one at a time conditioned on the image.

    We load in float16 (half precision) to cut VRAM usage roughly in half (~3GB vs ~6GB).
    This is safe for inference — float16 has enough precision for generation tasks.

    Returns:
        Tuple of (model, processor) ready for inference.
    """
    logger.info("Loading %s on %s ...", MODEL_NAME, DEVICE)
    t0 = time.perf_counter()
    processor = Blip2Processor.from_pretrained(MODEL_NAME)
    model = Blip2ForConditionalGeneration.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16,
    ).to(DEVICE)
    model.eval()
    elapsed = (time.perf_counter() - t0) * 1000
    logger.info("BLIP-2 loaded in %.0fms", elapsed)
    return model, processor


def caption(
    image: Image.Image, model: Blip2ForConditionalGeneration, processor: Blip2Processor
) -> str:
    """
    Generate a natural language caption for a PIL image.

    The processor converts the image into pixel tensors and casts them to float16
    to match the model weights. The model then generates caption tokens autoregressively —
    each token is predicted from the image features plus all previously generated tokens.
    MAX_NEW_TOKENS caps the generation length so the model doesn't run indefinitely.

    Args:
        image: PIL Image in any mode.
        model: Loaded Blip2ForConditionalGeneration on DEVICE.
        processor: Matching Blip2Processor.

    Returns:
        Caption string, e.g. "a person standing next to a potted plant".
    """
    t0 = time.perf_counter()
    inputs = processor(images=image, return_tensors="pt").to(DEVICE, torch.float16)
    with torch.no_grad():
        output_ids = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS)
    caption_text = processor.decode(output_ids[0], skip_special_tokens=True).strip()
    elapsed = (time.perf_counter() - t0) * 1000
    logger.debug("BLIP-2 caption: %.1fms | '%s'", elapsed, caption_text)
    return caption_text


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    model, processor = load_model()
    img = Image.open("assets/test_img.jpg").convert("RGB")
    print(caption(img, model, processor))
