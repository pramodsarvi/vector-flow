# Deployment

Everything needed to run the services in containers. Run the commands from the
**repository root** — the scanner's build context is the root, so one image can
pick up the engine, the viewer and the bundled sample scan.

```text
deploy/
├── docker-compose.yml     both services; structsync is behind the `gpu` profile
├── scanner/Dockerfile     the scanner image (CPU only)
├── .env.example           copy to .env to set ports, bind address and tokens
└── README.md              this file
```

---

## Scanner (CPU)

```bash
docker compose -f deploy/docker-compose.yml up -d --build
```

Then open <http://127.0.0.1:8765/viewer/>. The image is ~1 GB and needs no GPU;
a scan of the bundled sample room takes about two seconds.

Build or run it directly, without compose:

```bash
docker build -f deploy/scanner/Dockerfile -t spatial-scanner:latest .
docker run -d -p 8765:8765 -v scanner-data:/data spatial-scanner:latest
```

## StructuralSync (GPU)

Opt-in, because the image is ~10 GB and it needs an NVIDIA GPU plus the
container toolkit:

```bash
docker compose -f deploy/docker-compose.yml --profile gpu up -d --build
```

The first start downloads roughly 10 GB of model weights into the `structsync-hf`
volume; watch `docker compose logs -f`. Its own `services/structsync/docker-compose.yml`
remains valid if you would rather deploy that service on its own.

---

## Configuration

Copy `deploy/.env.example` to `deploy/.env`; compose reads it automatically.

| Variable | Default | Meaning |
|---|---|---|
| `BIND` | `127.0.0.1` | Interface the ports publish on |
| `SCANNER_PORT` | `8765` | Host port for the scanner |
| `SCANNER_TOKEN` | *(empty)* | Require this token on every scanner API call |
| `SCANNER_LOG` | `INFO` | Log level |
| `SCANNER_MAX_UPLOAD_MB` | `512` | Largest accepted upload |
| `SCANNER_MAX_SCANS` | `50` | Scans kept before the oldest finished ones are pruned |
| `STRUCTSYNC_PORT` | `8000` | Host port for StructuralSync |
| `STRUCTSYNC_TOKEN` | *(empty)* | Require this token on every StructuralSync API call |

### Exposing a service beyond localhost

Both services bind to `127.0.0.1` by default. Setting `BIND=0.0.0.0` publishes
them on your network, and **neither service requires authentication unless you
give it a token**. Set the matching token in the same change:

```bash
BIND=0.0.0.0
SCANNER_TOKEN=$(openssl rand -hex 24)
```

The scanner's own `run.sh --public` refuses to start without `SCANNER_TOKEN`;
compose cannot enforce that, so it is on you here.

---

## Data and upgrades

Scans and projects live in named volumes, not in the images, so rebuilding is
safe:

| Volume | Holds |
|---|---|
| `scanner-data` | scans: inputs, generated artifacts, metadata |
| `structsync-data` | projects, photos, renders |
| `structsync-catalog` | materials (seeded from the image on first start) |
| `structsync-hf` | downloaded model weights |

```bash
docker compose -f deploy/docker-compose.yml pull        # or rebuild
docker compose -f deploy/docker-compose.yml up -d --build
```

To start the scanner over from scratch: `docker compose -f deploy/docker-compose.yml down -v`
(this deletes the scans).

---

## Health

The scanner's container healthcheck polls `/api/config`, the one endpoint that
works whether or not a token is set:

```bash
docker compose -f deploy/docker-compose.yml ps          # shows healthy / unhealthy
curl -s http://127.0.0.1:8765/api/health | python3 -m json.tool
```

`/api/health` reports the background worker's queue depth, which is the useful
signal when scans appear to be stuck.

---

## Notes

- The scanner image runs as a non-root user (`uid 10001`) and owns `/data`.
  A bind mount from the host needs matching ownership, or use the named volume.
- One scan is processed at a time per container, by design: the pipeline seeds
  the global RNG so measurements are reproducible. Scale by running more
  replicas with separate data volumes, not by threading inside one.
- `.dockerignore` at the repository root keeps virtualenvs, caches, scan data
  and the other service out of the build context.
