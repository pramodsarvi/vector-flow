# Spatial Scanner — monorepo

Two independent services for capturing and re-imagining physical rooms.

| Service | What it does | Hardware |
|---|---|---|
| [**scanner**](services/scanner/) | Camera + LiDAR scan → metric 2D CAD floor design (`.dxf`, `.svg`, `.pdf`, `.json`) and an explorable 3D room model (`.glb`, `.obj`) | CPU only |
| [**structsync**](services/structsync/) | One kitchen photo + a material catalogue → a photoreal finish swap | NVIDIA GPU |

They share no code and deploy separately. Each has its own README, test suite
and dependency pins; this repository just keeps them together.

---

## Layout

```text
spatial-scanner-product/
├── services/
│   ├── scanner/        camera + LiDAR → floor design + 3D model  (CPU)
│   │   ├── capture/    how a scan is produced (iOS, rig, synthetic generator)
│   │   ├── engine/     ingest → geometry → export, plus the HTTP service
│   │   ├── viewer/     the web frontend (ES modules, no build step)
│   │   └── tests/
│   └── structsync/     photo + catalogue → finish swap  (GPU, own git repo)
├── deploy/             Dockerfile, compose, deployment guide
├── docs/               architecture and API reference
└── README.md           you are here
```

---

## Quick start

**Scanner** (no GPU needed, runs in seconds):

```bash
cd services/scanner
make setup && make run       # → http://127.0.0.1:8765/viewer/
```

**StructuralSync** (needs an NVIDIA GPU and ~13 GB of model weights):

```bash
cd services/structsync
bash setup_gpu.sh --download-models && ./run_web.sh   # → http://127.0.0.1:8000
```

**Both, in containers** — see [`deploy/`](deploy/):

```bash
docker compose -f deploy/docker-compose.yml up -d --build          # scanner only
docker compose -f deploy/docker-compose.yml --profile gpu up -d    # + structsync
```

---

## Tests

Each service is tested independently, against its own virtualenv:

```bash
cd services/scanner    && make test      # 75 tests, ~20 s, CPU only
cd services/structsync && .venv/bin/python -m pytest   # 118 tests, no model weights loaded
```

---

## Documentation

- [**Features**](docs/FEATURES.md) — everything both services can do, in one place
- [Architecture](docs/architecture.md) — how a scan flows through the engine, and why it is split the way it is
- [HTTP API](docs/api.md) — endpoints, error shape, polling a scan
- [Deployment](deploy/README.md) — images, compose, configuration, exposing the service safely
- [Scanner service](services/scanner/README.md) — quick start, accuracy figures, known limitations
- [StructuralSync service](services/structsync/README.md) — pipeline stages, catalogue, realism notes
