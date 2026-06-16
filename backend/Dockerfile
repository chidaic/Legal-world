FROM ghcr.io/astral-sh/uv:0.9.30-python3.11-bookworm-slim

ARG UV_INDEX_URL=https://pypi.org/simple

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_SYSTEM_PYTHON=1 \
    UV_INDEX_URL=${UV_INDEX_URL} \
    PYTHONPATH=/app/backend

WORKDIR /app/backend

COPY requirements.txt /app/requirements.txt

RUN apt-get update \
    && apt-get install -y --no-install-recommends curl gcc python3-dev \
    && rm -rf /var/lib/apt/lists/*

RUN uv pip install --system -r /app/requirements.txt

COPY backend /app/backend

EXPOSE 8000

CMD ["python", "-m", "uvicorn", "ws_server:app", "--host", "0.0.0.0", "--port", "8000"]
