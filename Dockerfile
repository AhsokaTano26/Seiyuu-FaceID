# syntax=docker/dockerfile:1
FROM ghcr.io/astral-sh/uv:0.12.15 AS uv
FROM python:3.13-slim-bookworm AS builder

COPY --from=uv /uv /usr/local/bin/uv
ENV UV_PYTHON_DOWNLOADS=never UV_LINK_MODE=copy UV_COMPILE_BYTECODE=1
WORKDIR /app
COPY pyproject.toml uv.lock README.md LICENSE ./
COPY src/ ./src/
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-dev --extra api --no-editable \
    && uv pip check --python /app/.venv/bin/python

FROM python:3.13-slim-bookworm AS runtime
# The locked opencv-python wheel needs these native libraries even without a GUI.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --uid 10001 --user-group --no-create-home --shell /usr/sbin/nologin faceid \
    && mkdir -p /app/models /app/configs
WORKDIR /app
COPY --from=builder /app/.venv /app/.venv
COPY configs/default.yaml /app/configs/default.yaml
ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    SEIYUU_FACEID_CONFIG=/app/configs/default.yaml \
    SEIYUU_FACEID_SERVE_UI=0 \
    OMP_NUM_THREADS=2 \
    OPENBLAS_NUM_THREADS=1
USER 10001:10001
# Fail a CI build before publishing if a platform is missing a native runtime dependency.
RUN python -c "import cv2, numpy, onnxruntime; from seiyuu_faceid.api.app import create_app"
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=120s --retries=3 \
    CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/ready', timeout=3).close()"]
CMD ["uvicorn", "seiyuu_faceid.api.app:create_app", "--factory", "--host", "0.0.0.0", "--port", "8000", "--workers", "1", "--timeout-graceful-shutdown", "60"]
