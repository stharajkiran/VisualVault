FROM python:3.11-slim

WORKDIR /app

RUN pip install uv --no-cache-dir

COPY pyproject.toml uv.lock ./

# Install Streamlit and minimal deps only — UI makes HTTP calls to the API, loads no models
RUN uv sync --frozen --no-dev --extra ui --no-install-project && \
    uv cache clean

COPY app/ui/ ./app/ui/

ENV PYTHONPATH=/app
