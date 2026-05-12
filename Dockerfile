FROM python:3.11-slim

WORKDIR /app

# Install system dependencies needed by Pillow, OpenCV, and ffmpeg
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Install uv
RUN pip install uv --no-cache-dir

# Copy dependency files first — Docker caches this layer separately.
# If only app code changes, this layer is reused and deps are not reinstalled.
COPY pyproject.toml uv.lock ./

# Install dependencies without dev extras
# --no-install-project skips installing the project itself (we COPY it below)
RUN uv sync --frozen --no-dev --no-install-project

# Copy application code
COPY app/ ./app/
COPY scripts/ ./scripts/
COPY data/eval_pairs.json ./data/eval_pairs.json

# PYTHONPATH lets Python find the app package without installing it
ENV PYTHONPATH=/app

# HuggingFace and Ultralytics cache directories
ENV HF_HOME=/cache/huggingface
ENV ULTRALYTICS_DIR=/cache/ultralytics
