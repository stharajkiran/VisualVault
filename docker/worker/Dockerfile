FROM python:3.11-slim

WORKDIR /app

# System deps for OpenCV, Pillow, and FFmpeg
# PyTorch CUDA wheels bundle their own CUDA runtime — no CUDA base image needed
# The host NVIDIA driver is passed through by Docker at runtime
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Install uv
RUN pip install uv --no-cache-dir

COPY pyproject.toml uv.lock ./

# Install full ML stack + GPU deps
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --extra ml-worker --extra gpu --no-install-project

COPY app/ ./app/
COPY artifacts/ ./artifacts/
COPY scripts/ ./scripts/

ENV PYTHONPATH=/app
ENV HF_HOME=/cache/huggingface
ENV ULTRALYTICS_DIR=/cache/ultralytics
