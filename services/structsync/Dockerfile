# StructuralSync web app — GPU image. Build:  docker compose build   Run:  docker compose up -d
FROM python:3.12-slim

RUN apt-get update && apt-get install -y --no-install-recommends git libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY requirements.txt requirements.lock ./
# exact, verified versions; PyPI torch wheels ship with CUDA (host needs only the NVIDIA driver + toolkit)
RUN pip install --no-cache-dir -r requirements.lock

COPY structsync/ structsync/
COPY web/ web/
COPY catalog/ catalog/
COPY app.py eval.py smoke_test.py prefetch.py runpod_entrypoint.sh README.md OPTIMISATION_ROADMAP.md RUNPOD.md ./

ENV STRUCTSYNC_DATA=/data \
    STRUCTSYNC_CATALOG=/app/catalog \
    HF_HOME=/hf \
    PYTHONUNBUFFERED=1
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=10s --start-period=120s \
  CMD python -c "import urllib.request,os; t=os.environ.get('STRUCTSYNC_TOKEN'); r=urllib.request.Request('http://127.0.0.1:8000/api/config', headers={'Authorization': f'Bearer {t}'} if t else {}); urllib.request.urlopen(r)"
CMD ["python", "-m", "uvicorn", "web.server:app", "--host", "0.0.0.0", "--port", "8000"]
