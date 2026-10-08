# Watching a Hyperloom run in Prometheus and Grafana

Every `hyperloom ... optimize` run starts a metrics exporter on
`127.0.0.1:9477` (log: `<session>/runtime/metrics_exporter.log`). It serves:

| Path | What |
|---|---|
| `/metrics` | Session state: phase, budgets, tasks, lanes, GPU leases, results, liveness |
| `/sd/inference` | Prometheus HTTP-SD pointing at the inference server being benchmarked |
| `/healthz` | `ok` while serving |

The exporter keeps serving for 120 s after the run exits, so the final state
(stop reason, final gain) is scraped, then exits by itself.

| Flag | Env | Default |
|---|---|---|
| `--no-metrics-exporter` | `HYPERLOOM_METRICS_EXPORTER=0` | on |
| `--metrics-listen HOST:PORT` | `HYPERLOOM_METRICS_LISTEN` | `127.0.0.1:9477` |
| `--metrics-grace-sec N` | `HYPERLOOM_METRICS_GRACE_SEC` | `120` |

Run it by hand against any session (for example a finished one):

```bash
python -m hyperloom.observability.exporter --session-dir <session> --listen 127.0.0.1:9477
```

## Running work

The dashboard shows running tasks by action (`kind`, such as `baseline`,
`explore`, `roofline`, or `conc_sweep`):

| Metric | Meaning |
|---|---|
| `hyperloom_running_tasks{kind="…"}` | Number of running tasks for that action |
| `hyperloom_running_task_elapsed_seconds{kind="…"}` | Age of the oldest running task for that action |
| `hyperloom_running_task_progress_age_seconds{kind="…"}` | Age of the freshest timestamped progress for running tasks of that action; absent until progress is published |

Task IDs and progress messages are not metric labels. Unknown action kinds are
aggregated as `other`, so repeated tasks do not create new series.

Fresh task progress is positive liveness evidence while a long benchmark leaves
`state.json` unchanged. Merely rereading the database, refreshing a lease, or
keeping the optimizer PID alive does not prove progress. An old state with no
fresh progress remains stale; inspect task elapsed and progress age together.

## Local stack (Docker Compose)

```bash
cd examples/observability/compose
docker compose up -d
# Grafana: http://127.0.0.1:3000  →  Hyperloom / "Hyperloom session"
```

## Existing or Kubernetes Prometheus

1. Start the run with `--metrics-listen 0.0.0.0:9477` (or the env var).
2. Edit `k8s/scrapeconfig.yaml` (host IP, selector label) and `kubectl apply -f` it.
3. Load the dashboard for the Grafana sidecar:

   ```bash
   kubectl -n monitoring create configmap hyperloom-dashboard \
     --from-file=hyperloom.json=dashboards/hyperloom.json
   kubectl -n monitoring label configmap hyperloom-dashboard grafana_dashboard=1
   ```

`/sd/inference` advertises the session's live benchmark server at the address
it binds, usually `127.0.0.1:<port>`, and returns `[]` while none is running.
The port changes with every server launch. Inference-server scraping therefore
works with a Prometheus on the same host (the Compose stack, or host
networking). A remote or Kubernetes Prometheus still gets the exporter's
`/metrics`, but cannot reach a loopback-bound server. `HYPERLOOM_VLLM_URL`
overrides discovery only for a fixed server you started on a reachable
address yourself.

The endpoint has no authentication and reveals the model, framework and
workload shape. Allow port 9477 only from your Prometheus nodes.

## Hyperloom in Docker

The shipped run skills start the ROCm container without host networking or a
published port, so the exporter is reachable only inside the container. To
scrape it from the host, add `--network=host` to that `docker run`, or publish
`-p 127.0.0.1:9477:9477` and set `HYPERLOOM_METRICS_LISTEN=0.0.0.0:9477` in
the container. Only host networking also makes `/sd/inference` useful, since
the benchmark server binds the container's loopback. If the container exits,
the grace period is lost; everything up to that point is already in
Prometheus.

## What is not here

GPU hardware metrics come from the AMD Device Metrics Exporter, and inference
server metrics come from the server's own `/metrics` (found via
`/sd/inference`). The exporter does not duplicate either.
