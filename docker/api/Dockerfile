FROM python:3.11-slim

WORKDIR /app

# Install uv
RUN pip install uv --no-cache-dir

# Copy dependency files first — Docker caches this layer separately.
# If only app code changes, this layer is reused and deps are not reinstalled.
COPY pyproject.toml uv.lock ./

# Install base deps + text encoder only — API never runs image inference
# torch resolves to pytorch-cpu index via ml-api conflict declaration in pyproject.toml
RUN uv sync --frozen --no-dev --extra ml-api --no-install-project && \
    uv cache clean

# Copy application code
COPY app/ ./app/

ENV PYTHONPATH=/app
ENV HF_HOME=/cache/huggingface
