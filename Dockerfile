# syntax=docker/dockerfile:1
#
# Multi-stage build for the Food-11 serving API.
#
#   builder  - installs the locked dependency tree into /app/.venv
#   runtime  - a clean slim image that gets only that .venv plus src/
#
# uv itself, the build caches and any compiler toolchain stay in the builder and
# never reach the final image.

# ---------------------------------------------------------------- builder ----
FROM python:3.13-slim AS builder

# uv ships as a static binary in its own image; copying it beats curl|sh.
COPY --from=ghcr.io/astral-sh/uv:0.12.13 /uv /usr/local/bin/uv

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never

WORKDIR /app

# Only the dependency manifests, so this layer's cache survives any source edit.
# --no-install-project: the app runs from src/ and is never imported as the
# installed 'mlops' package, so building it here would only invalidate the cache.
COPY pyproject.toml uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --no-install-project

# ---------------------------------------------------------------- runtime ----
FROM python:3.13-slim AS runtime

WORKDIR /app

# Create the unprivileged user BEFORE copying, so nothing needs chown later.
# A `chown -R` after the COPY would rewrite every file in .venv and Docker would
# store that as a second, full-size layer - it cost 1.76 GB before this change.
# The venv is read-only at runtime and root-owned files are world-readable.
RUN useradd --create-home --uid 1000 appuser

# The prebuilt environment. No uv, no build cache, no compilers.
COPY --from=builder /app/.venv /app/.venv

ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    MLFLOW_TRACKING_URI=http://host.docker.internal:5000

# Source last: the layer that changes most often is the cheapest to rebuild.
COPY src ./src

# Serving does not need root.
USER appuser

EXPOSE 8000

ENTRYPOINT ["uvicorn", "src.food11.serve:app", "--host", "0.0.0.0", "--port", "8000"]
