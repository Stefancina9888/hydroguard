# Build from the repository root: docker build -t hydroguard:phase4 .
FROM python:3.11.15-slim-bookworm AS builder
COPY --from=ghcr.io/astral-sh/uv:0.13.4 /uv /bin/uv
WORKDIR /app
ENV UV_PYTHON_DOWNLOADS=never UV_LINK_MODE=copy
COPY serving/pyproject.toml serving/uv.lock /app/serving/
COPY serving/src/hydroguard_serving /app/serving/src/hydroguard_serving
COPY src/hydroguard/__init__.py src/hydroguard/data.py src/hydroguard/features.py src/hydroguard/inference.py /app/src/hydroguard/
RUN uv sync --project /app/serving --locked --no-dev --no-editable --python /usr/local/bin/python

FROM python:3.11.15-slim-bookworm AS runtime
WORKDIR /app
ENV PATH="/app/serving/.venv/bin:$PATH" \
    HYDROGUARD_MODEL_DIR=/app/model \
    PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 \
    OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
COPY --from=builder /app/serving/.venv /app/serving/.venv
COPY artifacts/models/phase3b-v1/pipeline.pkl artifacts/models/phase3b-v1/metadata.json /app/model/
USER 10001:10001
EXPOSE 8000
HEALTHCHECK --interval=15s --timeout=5s --start-period=15s --retries=3 \
    CMD ["python", "-c", "import json, urllib.request; r=json.load(urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=4)); assert r['ready'] is True"]
CMD ["uvicorn", "hydroguard_serving.app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
