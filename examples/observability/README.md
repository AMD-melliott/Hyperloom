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

## Running work and resumed clocks

The dashboard shows running tasks by action (`kind`, such as `baseline`,
`explore`, `roofline`, or `conc_sweep`):

| Metric | Meaning |
|---|---|
| `hyperloom_running_tasks{kind="…"}` | Number of running tasks for that action |
| `hyperloom_running_task_elapsed_seconds{kind="…"}` | Age of the oldest running task for that action |
| `hyperloom_running_task_progress_age_seconds{kind="…"}` | Age of the freshest timestamped progress for running tasks of that action; absent until progress is published |

What Hyperloom found is exported per attempt from the optimization journal:

| Metric | Meaning |
|---|---|
| `hyperloom_optimization_info{ordinal,phase,kind,lever,outcome}` | One row per measured KEEP or any REVERT (newest 100), value 1 |
| `hyperloom_optimization_throughput{ordinal}` / `…_gain_percent{ordinal}` | Measured throughput and gain for that attempt, when measured |
| `hyperloom_optimization_accuracy{ordinal}` | Accuracy-eval score of an adopted attempt |
| `hyperloom_accuracy{stage="baseline\|best"}` | Accuracy of the baseline and of the best adopted stack |

Task IDs and progress messages are not metric labels. Unknown action kinds are
aggregated as `other`, so repeated tasks do not create new series.

Fresh task progress is positive liveness evidence while a long benchmark leaves
`state.json` unchanged. Merely rereading the database, refreshing a lease, or
keeping the optimizer PID alive does not prove progress. An old state with no
fresh progress remains stale; inspect task elapsed and progress age together.

Resumed phase and session clocks use the coordinator's charged execution legs,
not the entire wall span from the original start. Previously charged time is
counted once, and gaps while the optimizer was stopped are excluded.

## Local stack (Docker Compose)

```bash
cd examples/observability/compose
docker compose up -d
# Grafana: http://127.0.0.1:3000  →  Hyperloom / "Hyperloom session"
```

## Dashboards

| Dashboard | uid | Use |
|---|---|---|
| `dashboards/hyperloom.json` | `hyperloom-session` | One session over time: phase timeline, throughput, budgets, tasks |
| `dashboards/hyperloom-overview.json` | `hyperloom-overview` | Every run Prometheus still holds: one row per session, plus charts across runs and, for the selected session, what Hyperloom tried |

The overview reads the last value of each series over the dashboard's time
range (default 60 days), so a run's final result stays visible after its
exporter has exited. Click a run's start date to open its timeline. Prometheus
retention bounds how far back it can look. The per-attempt table and the
accuracy columns fill only for runs recorded after those series were added;
older runs show no data there.

## Existing or Kubernetes Prometheus

1. Start the run with `--metrics-listen 0.0.0.0:9477` (or the env var).
2. Edit `k8s/scrapeconfig.yaml` (host IP, selector label) and `kubectl apply -f` it.
3. Load the dashboard for the Grafana sidecar:

   ```bash
   kubectl -n monitoring create configmap hyperloom-dashboard \
     --from-file=hyperloom.json=dashboards/hyperloom.json \
     --from-file=hyperloom-overview.json=dashboards/hyperloom-overview.json
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
