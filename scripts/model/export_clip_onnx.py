"""
Export CLIP's image encoder from PyTorch to ONNX format.

The ONNX file is the intermediate step before TensorRT compilation.
Output: artifacts/clip_image_encoder.onnx

Run with:
    python scripts/model/export_clip_onnx.py
"""

import torch
from pathlib import Path

from transformers import CLIPModel
from app.embedding.clip_pytorch import load_model

OUTPUT_PATH = Path("artifacts/onnx/clip_image_encoder.onnx")


class CLIPImageEncoder(torch.nn.Module):
    """
    Thin wrapper around CLIPModel that exposes only the image encoder.

    torch.onnx.export traces a single forward() call. CLIPModel's forward()
    expects both image and text inputs. This wrapper isolates get_image_features()
    so the ONNX graph contains only the image encoder path.

    Args:
        clip_model: A loaded CLIPModel instance.

    Returns:
        torch.Tensor of shape (1, 512) — raw (unnormalized) image embedding.
    """

    def __init__(self, clip_model: CLIPModel):
        super().__init__()
        self.model = clip_model

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """
        Run the image encoder on preprocessed pixel values.

        Args:
            pixel_values (torch.Tensor): Shape (1, 3, 224, 224), float32, on CUDA.

        Returns:
            torch.Tensor: Shape (1, 512), float32 — raw embedding before normalization.
        """
        features = self.model.get_image_features(pixel_values=pixel_values)
        return features.pooler_output


def export(output_path: Path) -> None:
    """
    Load CLIP, trace the image encoder with a dummy input, and save as ONNX.

    Uses a dummy tensor of the exact shape the CLIP processor produces:
    (1, 3, 224, 224). Dynamic batch axis is enabled so the TensorRT engine
    can handle batch sizes other than 1 at runtime.

    Args:
        output_path (Path): Where to write the .onnx file.

    Returns:
        None
    """
    print("[Export] Loading CLIP model ...")
    clip_model, _ = load_model()
    encoder = CLIPImageEncoder(clip_model).eval()

    # Dummy input — same shape the processor produces for a single image
    device = next(encoder.parameters()).device
    dummy = torch.randn(1, 3, 224, 224, dtype=torch.float32).to(device)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"[Export] Tracing and exporting to {output_path} ...")
    with torch.no_grad():
        torch.onnx.export(
            encoder,
            dummy,
            str(output_path),
            input_names=["pixel_values"],
            output_names=["embeddings"],
            # Dynamic batch axis lets TensorRT handle batch sizes > 1 at runtime
            dynamic_axes={
                "pixel_values": {0: "batch_size"},
                "embeddings": {0: "batch_size"},
            },
            opset_version=18,
        )

    size_mb = output_path.stat().st_size / (1024**2)
    print(f"[Export] Done — {output_path} ({size_mb:.1f} MB)")


def verify(output_path: Path) -> None:
    """
    Load the exported ONNX file and check it is a valid graph.

    Runs onnx.checker.check_model() which validates the graph structure,
    operator types, and shape consistency. Does not run inference.

    Args:
        output_path (Path): Path to the .onnx file to verify.

    Returns:
        None
    """
    import onnx

    print("[Verify] Checking ONNX graph ...")
    model = onnx.load(str(output_path))
    onnx.checker.check_model(model)
    print("[Verify] ONNX graph is valid ✓")


if __name__ == "__main__":
    export(OUTPUT_PATH)
    verify(OUTPUT_PATH)
