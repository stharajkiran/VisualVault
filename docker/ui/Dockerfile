FROM python:3.11-slim

WORKDIR /app

RUN pip install uv --no-cache-dir

COPY pyproject.toml uv.lock ./

# Install Streamlit and minimal deps only — UI makes HTTP calls to the API, loads no models
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --extra ui --no-install-project

COPY app/ui/ ./app/ui/

ENV PYTHONPATH=/app
