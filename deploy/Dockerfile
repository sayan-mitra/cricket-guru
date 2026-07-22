# All-in-one image for a single-machine deploy (Railway): Postgres + Streamlit app + embedded Qdrant.
# The Cricsheet dump and the built Qdrant index are baked in, so the container comes up self-contained.
# Build locally (the corpora/index are gitignored, so Railway can't git-build this) and push to a
# registry; point Railway at the pushed image. LLM keys come from Railway env vars, never the image.
#
#   docker build -f deploy/Dockerfile -t ghcr.io/<you>/cricket-guru:latest .
#   docker push ghcr.io/<you>/cricket-guru:latest
FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
      postgresql postgresql-client gosu gzip \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY backend/requirements.txt backend/requirements.txt
# --no-deps: install the exact frozen set without re-resolving (the venv is internally consistent
# at runtime but pip's strict resolver rejects it; --no-deps reproduces it faithfully).
RUN pip install --no-cache-dir --no-deps -r backend/requirements.txt

COPY backend backend
COPY frontend frontend
COPY .streamlit .streamlit
COPY data/gold data/gold
COPY data/results data/results
# Built vector index. COPY the tarball as a single file (ADD auto-extracts, and BuildKit's macOS
# file-sharing chokes on the contents); the entrypoint unpacks it into data/qdrant at boot.
COPY data/qdrant.tar.gz data/qdrant.tar.gz
COPY deploy/cricket_guru.dump.gz deploy/cricket_guru.dump.gz
COPY deploy/entrypoint.sh entrypoint.sh
RUN chmod +x entrypoint.sh

# CG_QDRANT_URL is intentionally unset -> the app uses the embedded on-disk index at data/qdrant.
# LLM keys (OPENAI_API_KEY, ANTHROPIC_API_KEY, TAVILY_API_KEY) are injected by Railway, not baked.
ENV PYTHONPATH=/app/backend \
    CG_HOST=localhost \
    CG_USER=cricket \
    CG_DB=cricket_guru \
    STREAMLIT_SERVER_HEADLESS=true

EXPOSE 8501
CMD ["/app/entrypoint.sh"]
