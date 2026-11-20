# Running both services on one RunPod pod

How the current pod is set up: Scanner (CPU) and StructuralSync (GPU) on one
pod, both reachable through a **single exposed HTTP port, 8080**, via a small
nginx gateway.

```text
https://<POD_ID>-8080.proxy.runpod.net/            → StructuralSync  (127.0.0.1:8000)
https://<POD_ID>-8080.proxy.runpod.net/scanner/    → Scanner viewer  (127.0.0.1:8765)
```

`gateway.nginx.conf` routes by path. The scanner viewer calls absolute `/api/...`
URLs, which would collide with StructuralSync's `/api/...`, so the gateway
rewrites them to `/scanner/api/...` in the JavaScript it serves (`sub_filter`).
No service code is changed.

## Rebuild after a restart (~15 min)

Without a Volume Disk at `/workspace`, any pod edit or restart erases the code,
the venvs, the ~21 GB of weights and the tokens. Redeploy with a volume to avoid
this.

```bash
cd /workspace && git clone https://github.com/pramodsarvi/vector-flow.git && cd vector-flow

# Scanner (CPU) — bound to localhost; the gateway exposes it
cd services/scanner && make setup
echo "SCANNER_TOKEN=$(python3 -c 'import secrets;print(secrets.token_urlsafe(24))')" > .env
nohup ./run.sh > /workspace/scanner.log 2>&1 &

# StructuralSync (GPU) — downloads ~21 GB of weights
cd ../structsync && bash setup_gpu.sh --download-models --with-harmonizer
# Optional but recommended, the URL is public:
# echo "STRUCTSYNC_TOKEN=$(python3 -c 'import secrets;print(secrets.token_urlsafe(24))')" > .env
nohup ./run_web.sh --preload > /workspace/structsync.log 2>&1 &

# Gateway on :8080
mkdir -p /workspace/proxy/logs
cp ../../deploy/runpod/gateway.nginx.conf /workspace/proxy/nginx.conf
nginx -c /workspace/proxy/nginx.conf -p /workspace/proxy/

cat ../scanner/.env .env 2>/dev/null     # show tokens
```

StructuralSync is ready when `/workspace/structsync.log` shows
`Uvicorn running`; with `--preload` the models finish loading shortly after.

## Notes

- **Which ports are exposed.** On this pod only 8080 (and Jupyter on 8888) are
  exposed through the RunPod proxy. Check with
  `curl -so /dev/null -w '%{http_code}' https://$RUNPOD_POD_ID-<port>.proxy.runpod.net/`:
  `404` means the port is not exposed, `502` means exposed but nothing listening.
- **Tokens.** The proxy URL is public. The Scanner requires `SCANNER_TOKEN`
  (its viewer asks once). StructuralSync without `STRUCTSYNC_TOKEN` is open to
  anyone with the URL, including the GPU.
- **Reload the gateway** after editing:
  `nginx -c /workspace/proxy/nginx.conf -p /workspace/proxy/ -s reload`.
- **Proxy timeout.** RunPod cuts requests at ~100 s, so long work must run as a
  background job that the browser polls.
