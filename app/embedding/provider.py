from abc import ABC, abstractmethod

import numpy as np
import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

from app.embedding.config import CLIPConfig

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class EmbeddingProvider(ABC):
    """
    Abstract interface for any image and text embedding model.

    All components above this layer — Celery tasks, FastAPI routes, Qdrant
    client — only ever call embed_image() and embed_text(). They never import
    CLIPModel or trt.ICudaEngine directly. Swapping backends or model variants
    requires only changing which provider is registered, not the calling code.
    """

    @property
    @abstractmethod
    def config(self) -> CLIPConfig:
        """Return the CLIPConfig that fully describes this provider."""

    @abstractmethod
    def embed_image(self, image: Image.Image) -> np.ndarray:
        """
        Embed a PIL image into a normalized vector.

        Args:
            image (Image.Image): Input image in any mode.

        Returns:
            np.ndarray: Shape (embedding_dim,), dtype float32. L2-norm = 1.0
                        if config.normalize_embeddings is True.
        """

    @abstractmethod
    def embed_text(self, text: str) -> np.ndarray:
        """
        Embed a text query into a normalized vector.

        Args:
            text (str): Natural language query string.

        Returns:
            np.ndarray: Shape (embedding_dim,), dtype float32. L2-norm = 1.0
                        if config.normalize_embeddings is True.
        """

    def warmup(self) -> None:
        """
        Run a dummy forward pass to eliminate JIT overhead on first real request.

        Called once after the model is loaded. Default implementation runs
        embed_image on a blank image. Subclasses may override if needed.

        Args:
            None

        Returns:
            None
        """
        dummy = Image.new("RGB", (self.config.input_size, self.config.input_size))
        self.embed_image(dummy)


class CLIPPyTorchProvider(EmbeddingProvider):
    """
    EmbeddingProvider backed by a HuggingFace CLIPModel running on PyTorch.

    Loads the model specified in config.model_id at construction time.
    Normalization is controlled by config.normalize_embeddings.
    """

    def __init__(self, config: CLIPConfig) -> None:
        """
        Load the CLIP model and processor from HuggingFace onto the GPU.

        Args:
            config (CLIPConfig): Model configuration — model_id, embedding_dim,
                                 normalize_embeddings, and other fields.

        Returns:
            None
        """
        self._config = config
        print(f"[{config.alias}] Loading {config.model_id} on {DEVICE} ...")
        self._model: CLIPModel = CLIPModel.from_pretrained(config.model_id).to(DEVICE)
        self._model.eval()
        self._processor: CLIPProcessor = CLIPProcessor.from_pretrained(config.model_id)
        print(f"[{config.alias}] Ready.")

    @property
    def config(self) -> CLIPConfig:
        return self._config

    def embed_image(self, image: Image.Image) -> np.ndarray:
        """
        Embed a PIL image using the PyTorch CLIP image encoder.

        Args:
            image (Image.Image): Input image in any mode — converted to RGB internally.

        Returns:
            np.ndarray: Shape (embedding_dim,), dtype float32.
        """
        inputs = self._processor(images=image.convert("RGB"), return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            features = self._model.get_image_features(**inputs)
        vec = features.pooler_output.squeeze().cpu().numpy().astype(np.float32)
        if self._config.normalize_embeddings:
            vec = vec / np.linalg.norm(vec)
        return vec

    def embed_text(self, text: str) -> np.ndarray:
        """
        Embed a text query using the PyTorch CLIP text encoder.

        Args:
            text (str): Natural language query string.

        Returns:
            np.ndarray: Shape (embedding_dim,), dtype float32.
        """
        inputs = self._processor(
            text=text, return_tensors="pt", padding=True, truncation=True
        ).to(DEVICE)
        with torch.no_grad():
            features = self._model.get_text_features(**inputs)
        vec = features.pooler_output.squeeze().cpu().numpy().astype(np.float32)
        if self._config.normalize_embeddings:
            vec = vec / np.linalg.norm(vec)
        return vec


class CLIPTensorRTProvider(EmbeddingProvider):
    """
    EmbeddingProvider backed by a compiled TensorRT engine for image embedding.

    Uses the TRT engine (from config.engine_path) for embed_image() and falls
    back to PyTorch for embed_text() — the TRT engine covers only the image
    encoder. The text encoder is called infrequently (search time only) so
    PyTorch latency there is acceptable.
    """

    def __init__(self, config: CLIPConfig) -> None:
        """
        Load the TRT engine and PyTorch text encoder from disk.

        Args:
            config (CLIPConfig): Must have backend="tensorrt" and a valid engine_path.

        Returns:
            None
        """
        import tensorrt as trt

        if config.engine_path is None:
            raise ValueError("CLIPTensorRTProvider requires engine_path in config — got None")

        self._config = config
        self._processor: CLIPProcessor = CLIPProcessor.from_pretrained(config.model_id)

        print(f"[{config.alias}] Loading TRT engine from {config.engine_path} ...")
        logger = trt.Logger(trt.Logger.WARNING)
        runtime = trt.Runtime(logger)
        with open(config.engine_path, "rb") as f:
            self._engine = runtime.deserialize_cuda_engine(f.read())
        if self._engine is None:
            raise RuntimeError(f"Failed to deserialize TRT engine from {config.engine_path}")
        self._context = self._engine.create_execution_context()

        # PyTorch text encoder — loaded once, used for embed_text()
        print(f"[{config.alias}] Loading PyTorch text encoder for embed_text() ...")
        self._text_model: CLIPModel = CLIPModel.from_pretrained(config.model_id).to(DEVICE)
        self._text_model.eval()
        print(f"[{config.alias}] Ready.")

    @property
    def config(self) -> CLIPConfig:
        return self._config

    def embed_image(self, image: Image.Image) -> np.ndarray:
        """
        Embed a PIL image using the TensorRT CLIP image encoder.

        Args:
            image (Image.Image): Input image in any mode — converted to RGB internally.

        Returns:
            np.ndarray: Shape (embedding_dim,), dtype float32.
        """
        inputs = self._processor(images=image.convert("RGB"), return_tensors="pt")
        pixel_tensor = inputs["pixel_values"].to(dtype=torch.float32, device="cuda")
        output_tensor = torch.empty((1, self._config.embedding_dim), dtype=torch.float32, device="cuda")

        stream = torch.cuda.current_stream()
        self._context.set_input_shape("pixel_values", tuple(pixel_tensor.shape))
        self._context.set_tensor_address("pixel_values", pixel_tensor.data_ptr())
        self._context.set_tensor_address("embeddings", output_tensor.data_ptr())
        self._context.execute_async_v3(stream_handle=stream.cuda_stream)
        stream.synchronize()

        vec = output_tensor.cpu().numpy().squeeze().astype(np.float32)
        if self._config.normalize_embeddings:
            vec = vec / np.linalg.norm(vec)
        return vec

    def embed_text(self, text: str) -> np.ndarray:
        """
        Embed a text query using the PyTorch CLIP text encoder.

        TRT engine covers only the image encoder. Text embedding falls back
        to PyTorch — called only at search time, not during batch ingestion.

        Args:
            text (str): Natural language query string.

        Returns:
            np.ndarray: Shape (embedding_dim,), dtype float32.
        """
        inputs = self._processor(
            text=text, return_tensors="pt", padding=True, truncation=True
        ).to(DEVICE)
        with torch.no_grad():
            features = self._text_model.get_text_features(**inputs)
        vec = features.pooler_output.squeeze().cpu().numpy().astype(np.float32)
        if self._config.normalize_embeddings:
            vec = vec / np.linalg.norm(vec)
        return vec
