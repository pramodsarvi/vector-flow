# Deploying on RunPod

**GPU:** 8 GB VRAM works (that's what this was developed on); 16–24 GB (RTX A4000/A5000/4090) is comfortable and
lets you raise `STRUCTSYNC_SCENES`. **Host driver:** the pinned torch 2.15 wheel is built for CUDA 13.2 and needs
driver ≥ 580. On *Deploy → Additional filters → CUDA Versions* select **13.0 or newer**, otherwise torch reports
"no GPU / driver too old".

## Option A — your own image (recommended)
```bash
docker build -t <you>/structuralsync:1.0 .
docker push <you>/structuralsync:1.0        # Docker Hub or GHCR (private is fine; add registry creds in RunPod)
```
RunPod → Pods → Deploy, then:
| Field | Value |
|---|---|
| Container image | `<you>/structuralsync:1.0` |
| Container disk | 30 GB |
| Volume disk | 40 GB, mount path `/workspace` (weights ~13 GB, plus projects/renders) |
| Expose HTTP port | `8000` |
| Container start command | `bash /app/runpod_entrypoint.sh` |
| Env vars | `STRUCTSYNC_TOKEN` = a long random secret (**required**); optional `STRUCTSYNC_SCENES=2`, `STRUCTSYNC_CACHE_MB=4096` |

## Option B — no registry (stock "RunPod PyTorch" template)
Upload `structuralsync-*.tar.gz` (Jupyter upload or `runpodctl send`), then in the pod terminal:
```bash
cd /workspace && tar xzf structuralsync-*.tar.gz && cd structuralsync
export STRUCTSYNC_TOKEN=<secret>
bash runpod_entrypoint.sh      # first start: builds a venv + downloads weights (~10–20 min); later starts: seconds
```
Expose HTTP port `8000` in the pod settings. Use `nohup bash runpod_entrypoint.sh > /workspace/server.log 2>&1 &`
to keep it running after you close the terminal.

## Using it
Open `https://<POD_ID>-8000.proxy.runpod.net`; the UI asks once for the token. First start downloads weights and
then preloads the models — `/api/health` shows `device: cuda`. Wait for it before demoing.

## Notes
- The RunPod proxy URL is **public**; the entrypoint refuses to start without `STRUCTSYNC_TOKEN`.
- The proxy cuts requests at ~100 s. Analysis (~25 s) and renders (~1 s) are well inside that.
- Persistent on `/workspace`: `data/` (projects), `catalog/` (seeded from the image once; imports persist), `hf/` (weights).
  Container disk is wiped on restart; **stopping** a pod keeps the volume, **terminating** it deletes it.
- The SDXL seam harmoniser needs extra weights (set `STRUCTSYNC_ALLOW_DOWNLOAD=1`; several GB more, not fetched by default); without them the
  deterministic projection renderer is used — that is what the demo uses.
- Bill by the hour: stop the pod when idle.
