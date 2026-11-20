# HTTP API

The service is documented live at `/docs` (OpenAPI) once it is running. This
page is the written reference.

## Endpoints

A scan is a resource with a lifecycle. You create it, the background worker processes it, and
you poll until it is `ready`. Reconstruction never runs inside a request handler, so the API
stays responsive while a scan is being processed, and one worker runs at a time because the
pipeline seeds the global RNG for reproducible measurements.

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/api/config` | Public: limits and whether a token is required |
| `GET` | `/api/health` | Service status and worker queue depth |
| `GET` | `/api/scans` | List scans, newest first |
| `POST` | `/api/scans` | Create from an uploaded bundle (`cloud`, optional `poses`, `frames`) |
| `POST` | `/api/scans/sample` | Create from the bundled demo capture |
| `GET` | `/api/scans/{id}` | Status, progress, metrics, available artifacts |
| `POST` | `/api/scans/{id}/reprocess` | Re-run the pipeline on the same input |
| `DELETE` | `/api/scans/{id}` | Delete the scan and its artifacts |
| `GET` | `/api/scans/{id}/floorplan` | The floorplan schema |
| `GET` | `/api/scans/{id}/colliders` | Physics colliders |
| `GET` | `/api/scans/{id}/artifacts/{name}` | SVG · DXF · PDF · JSON · OBJ · GLB · PLY |
| `GET` | `/api/scans/{id}/frames/{file}` | A captured RGB frame |

`/api/scan/*` (singular) still works and resolves to the most recent finished scan, so existing
scripts do not break.

### Configuration

Every setting is an environment variable; copy `.env.example` to `.env` to set them.

| Variable | Default | Meaning |
|---|---|---|
| `SCANNER_TOKEN` | *(empty)* | Require this token on every API call. **Set it before exposing the service beyond localhost.** |
| `SCANNER_DATA` | `./data` | Where scans are stored |
| `SCANNER_CORS_ORIGINS` | *(empty)* | Allowed browser origins; empty means same-origin only |
| `SCANNER_MAX_UPLOAD_MB` | `512` | Largest accepted upload |
| `SCANNER_MAX_SCANS` | `50` | Scans kept before the oldest finished ones are pruned |
| `SCANNER_LOG` | `INFO` | Log level |

`./run.sh --public` refuses to start without `SCANNER_TOKEN`, so the service cannot be exposed
to a network by accident.

---


---

## Error shape

Every non-2xx response has the same body, so a client needs one error path:

```json
{ "error": "not_found", "detail": "No such scan or artifact: 20261117-120000-abc123" }
```

`detail` is written to be shown to a person. Internal exception text, stack
traces and absolute paths stay in the service log and are never returned.

| Status | `error` | When |
|---|---|---|
| 400 | `invalid_upload` / `bad_request` | Malformed point cloud, bad poses file, unusable frame name |
| 401 | `unauthorized` | `SCANNER_TOKEN` is set and the request did not carry it |
| 404 | `not_found` | Unknown scan id, or an artifact that is not on the allowlist |
| 409 | `conflict` | Artifact requested before the scan finished, or reprocess while already running |
| 413 | `too_large` | Upload exceeded `SCANNER_MAX_UPLOAD_MB` |
| 500 | `internal_error` | Anything unhandled; details are in the log |

---

## Polling a scan

`POST /api/scans` and `POST /api/scans/sample` return `202` immediately with a
scan in `queued`. Poll `GET /api/scans/{id}` until `status` is `ready` or
`failed`; `progress` carries `{step, total, message}` for a progress bar.

```text
queued -> processing -> ready
                     -> failed   (error holds a one-line reason)
```

Artifacts only exist once the scan is `ready`; asking earlier returns `409`
rather than a misleading `404`.
