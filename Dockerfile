# PIT WALL — Hugging Face Space (Docker SDK).
#
# Gradio/Streamlit SDKs don't fit: this is a Next.js frontend against a FastAPI
# backend. Docker runs it.
#
# The frontend is built as a *static export* and served by FastAPI itself. The
# first attempt shipped Next's standalone server into a python:slim runtime and
# died on `node: not found` — a bug the build succeeded through and only running
# the container exposed. Exporting instead means one process, no Node at
# runtime, and a much smaller image. Every page here is client-rendered, so
# nothing is lost.
#
# No GPU needed: race data is precomputed, so only Live Analysis runs a model.

FROM node:20-slim AS frontend
WORKDIR /build
COPY frontend/package*.json ./
RUN npm ci
COPY frontend/ ./
# Same origin in the Space, so the browser uses relative /api paths.
ENV NEXT_PUBLIC_API="" \
    NEXT_OUTPUT=export
RUN npm run build


FROM python:3.12-slim
WORKDIR /app

# libsndfile backs soundfile/librosa; ffmpeg covers the mp3/webm uploads the
# Live screen accepts.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libsndfile1 ffmpeg curl \
    && rm -rf /var/lib/apt/lists/*

# Spaces run as a non-root user; give it a writable HF cache.
RUN useradd -m -u 1000 user
ENV HF_HOME=/home/user/.cache/huggingface \
    PYTHONUNBUFFERED=1 \
    PITWALL_STATIC=/app/static

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY --chown=user backend/ /app/backend/
COPY --from=frontend --chown=user /build/out /app/static

USER user
EXPOSE 7860

HEALTHCHECK --interval=30s --timeout=5s --start-period=240s \
    CMD curl -fsS http://localhost:7860/api/health || exit 1

WORKDIR /app/backend
CMD ["python", "-m", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]
