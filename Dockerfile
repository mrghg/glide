# syntax=docker/dockerfile:1.7
FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 \
    python3.11-venv \
    python3-pip \
    ca-certificates \
    curl \
    tini \
    && rm -rf /var/lib/apt/lists/*

# Keep a predictable python executable path
RUN ln -sf /usr/bin/python3.11 /usr/local/bin/python

# Install Python dependencies + the package. The runtime image installs the
# core deps only (NO `[viz]` extras — cartopy / geoviews are notebook-only and
# bring a C-extension build chain we don't want in this image). The pytorch
# CUDA wheel is selected by `--extra-index-url`. Source is copied first so the
# editable install resolves; layer-cache efficiency is sacrificed in exchange
# for pyproject being the single source of truth for deps.
COPY pyproject.toml README.md /app/
COPY src /app/src
RUN python -m pip install --upgrade pip setuptools wheel && \
    python -m pip install --extra-index-url https://download.pytorch.org/whl/cu124 -e /app

ENV PYTHONPATH=/app/src

# Cloud Run provides PORT; this app is batch-oriented but keeps API-friendly envs.
ENV PORT=8080

ENTRYPOINT ["/usr/bin/tini", "--"]
CMD ["python", "-m", "lpdm.main"]
