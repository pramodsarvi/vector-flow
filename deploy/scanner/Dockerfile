# Spatial Scanner — API + web viewer in one image.
#
# Build from the repository root so the build context contains the service:
#   docker build -f deploy/scanner/Dockerfile -t spatial-scanner:latest .
#
# Run:
#   docker run -p 8765:8765 -v scanner-data:/data spatial-scanner:latest

FROM python:3.12-slim AS base

# opencv-python-headless still needs libglib; everything else in the stack ships
# manylinux wheels, so no compiler toolchain is required in the final image.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Dependencies first: this layer is cached until the pinned versions change.
COPY services/scanner/requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Then the service itself, so a code change rebuilds only the layers below.
COPY services/scanner/engine/ engine/
COPY services/scanner/viewer/ viewer/
COPY services/scanner/capture/sample-data/ capture/sample-data/

# Scans are user data and belong on a volume, not in the image.
ENV SCANNER_DATA=/data \
    SCANNER_SAMPLE=/app/capture/sample-data \
    SCANNER_VIEWER=/app/viewer \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# Run as a non-root user; it must own /data because the service writes there.
RUN useradd --system --create-home --uid 10001 scanner \
    && mkdir -p /data \
    && chown -R scanner:scanner /app /data
USER scanner

VOLUME ["/data"]
EXPOSE 8765

# /api/config is the one unauthenticated endpoint, so the probe works whether or
# not SCANNER_TOKEN is set.
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8765/api/config', timeout=4)"

CMD ["python", "-m", "uvicorn", "engine.service.app:app", "--host", "0.0.0.0", "--port", "8765"]
