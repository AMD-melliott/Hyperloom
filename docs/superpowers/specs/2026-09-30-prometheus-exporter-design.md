# Hyperloom Prometheus exporter — design

Status: draft for review · Branch: `feat/observability-status-layer` · Date: 2026-09-30

## Goal

Make a running Hyperloom session visible in Prometheus/Grafana, so an operator
(or a demo audience) can see what phase it is in, how much budget remains, what
work is in flight, whether it is alive, and how throughput has moved — without
attaching a terminal.

Success criteria:

- Every optimizer run exposes `/metrics` by default; `--no-metrics-exporter`
  turns it off.
- The exporter can never fail, slow, or alter a run.
- The final state of a run (dead, stop reason, final gain) is still scrapeable
  after the optimizer exits.
- The same exporter works with a local Docker Compose stack and with an existing
  (including Kubernetes-hosted) Prometheus.
- The exporter core can later run as a standalone daemon without redesign.

Non-goals: multiple concurrent sessions per exporter, GPU hardware metrics,
inference-server metrics (both scraped from their native exporters), log/trace
pipelines (Loki, Tempo, Langfuse), a production-grade dashboard.

## Context

The branch already has a read-only observability layer:

- `observability.collector.SessionMonitor` runs per-source collectors
  (session artifacts, GPU probe, server scrape) on independent background
  cadences and caches the latest result with health/age per source.
- `observability.assemble.load_snapshot` builds an immutable `Snapshot`
  (phases, budgets, tasks, lanes, GPU leases, result, current step, liveness,
  source health).
- `observability.render` has two renderers over that snapshot: text and JSON.
- `observability.readonly` guarantees the layer never writes the session dir.

Hyperloom has zero runtime dependencies (`pyproject.toml: dependencies = []`),
so the exporter uses only the standard library.

A supervisor restart ends the optimizer process but not the session; a resume
starts a new optimizer process on the same session dir. "Per run" therefore
means per optimizer process, and consecutive processes must hand the port over
cleanly.

## Architecture

```
optimizer process (cli run)                exporter child process
───────────────────────────                ─────────────────────────────────────
acquire session lock                        SessionMonitor(session_dir,
spawn exporter ──(start_new_session)──────►   gpu=False, server=False)
coordinator.run(...)                        ThreadingHTTPServer
  ...                                         GET /metrics  → render_prometheus(cached snapshot)
finally: release lock (exporter untouched)    GET /healthz  → 200 while serving
exit                                          GET /sd/inference → HTTP-SD JSON
                                            parent watchdog → grace → exit
```

### Units

1. **`observability/render/prometheus.py`** — pure function
   `render_prometheus(snapshot: Snapshot | None, *, exporter: ExporterInfo) -> str`.
   Emits Prometheus text exposition format 0.0.4. No I/O, no clock reads beyond
   the snapshot's own. Exported from `observability.render` beside
   `render_json` / `render_status`.
2. **`observability/exporter.py`** — the process: argument parsing, a
   `SessionMonitor`, a stdlib `ThreadingHTTPServer`, the parent watchdog, and
   the lock-handover check. Entry point
   `python -m hyperloom.observability.exporter`.
3. **Run wiring** in `inference_optimizer/cli/__init__.py` — a small
   `start_metrics_exporter(session_dir, args) -> subprocess.Popen | None`
   helper called right after the session lock is acquired.
4. **Deployment examples** under `examples/observability/` — Compose stack,
   Prometheus configs, Kubernetes `ScrapeConfig`, starter Grafana dashboard.

Each request to `/metrics` renders from the monitor's cached snapshot. A scrape
never triggers session I/O, so scrape latency is independent of a slow or hung
source.

## Metrics

Naming follows Prometheus conventions (`_seconds`, `_total`, `_info`, base
units). Once a snapshot exists, every series carries three constant labels
taken from its `SessionInfo`: `session_id`, `model`, `framework`. The minimal
page served before a session is observed carries none. No other label may take an
unbounded set of values: task IDs, holder IDs, URLs and free text never become
labels.

### Session and liveness

| Metric | Type | Labels | Source |
|---|---|---|---|
| `hyperloom_session_info` | gauge (=1) | `framework_version`, `gpu_type`, `precision`, `tp`, `ep`, `conc`, `isl`, `osl`, `objective_kind` | `SessionInfo` |
| `hyperloom_session_observed` | gauge 0/1 | — | 1 when a snapshot was assembled |
| `hyperloom_liveness` | gauge, one-hot | `state` = live/stale/dead/unknown | `Snapshot.liveness` |
| `hyperloom_state_age_seconds` | gauge | — | `state_age_s` |
| `hyperloom_last_activity_age_seconds` | gauge | — | `last_activity_age_s` |
| `hyperloom_session_start_timestamp_seconds` | gauge | — | `SessionInfo.started_at` |
| `hyperloom_session_elapsed_seconds` / `_remaining_seconds` | gauge | — | snapshot |
| `hyperloom_macro_cycle`, `hyperloom_tick` | gauge | — | snapshot |

### Phases

| Metric | Type | Labels | Source |
|---|---|---|---|
| `hyperloom_phase_current` | gauge, one-hot | `phase` | `PhaseProgress.is_current` |
| `hyperloom_phase_elapsed_seconds` | gauge | `phase` | `elapsed_s` |
| `hyperloom_phase_budget_seconds` | gauge | `phase` | `budget_total_s` |
| `hyperloom_phase_budget_remaining_seconds` | gauge | `phase` | `budget_remaining_s` |
| `hyperloom_phase_cap_seconds` | gauge | `phase` | `cap_s` |

`phase` takes the fixed phase names in `snapshot.phases` (PRELUDE, ENABLEMENT,
FRAMEWORK_AGENT, KERNEL_AGENT, SWEEP, CLOSE). A value that is `None` in the
snapshot is omitted, never exported as 0.

### Work in flight

| Metric | Type | Labels | Source |
|---|---|---|---|
| `hyperloom_tasks` | gauge | `state` = queued/running/succeeded/failed/cancelled | `TaskCounts` |
| `hyperloom_lane_held` / `hyperloom_lane_capacity` | gauge | `lane` | `LaneOccupancy` |
| `hyperloom_gpu_leased` | gauge 0/1 | `gpu_id` | `GpuLease` (expired leases = 0) |
| `hyperloom_current_step_info` | gauge (=1) | `phase`, `step` | `CurrentStep` |
| `hyperloom_current_step_start_timestamp_seconds` | gauge | — | `CurrentStep.started_unix` |
| `hyperloom_current_step_deadline_timestamp_seconds` | gauge | — | `CurrentStep.deadline_unix` |

`step` is a code-defined identifier (e.g. `geak_e2e`), not free text; `detail`
is not exported.

### Results

| Metric | Type | Labels | Source |
|---|---|---|---|
| `hyperloom_throughput_baseline` | gauge | — | `ResultSummary.baseline_tput` |
| `hyperloom_throughput_best` | gauge | — | `best_tput` |
| `hyperloom_gain_percent` | gauge | `kind` = raw/validated | `cumulative_gain_pct`, `cumulative_gain_validated_pct` |
| `hyperloom_target_gap_percent` | gauge | — | `target_gap_pct` |
| `hyperloom_crashes` | gauge | — | `crash_count` |
| `hyperloom_stop_info` | gauge (=1) | `reason` | `stop_reason`, only once set |
| `hyperloom_accuracy` | gauge | `stage` = baseline/best | `baseline_accuracy`, accuracy of the last adopted stack entry |
| `hyperloom_optimization_info` | gauge (=1) | `ordinal`, `phase`, `kind`, `lever`, `outcome` | one row per journal KEEP (measured) or REVERT, newest 100 |
| `hyperloom_optimization_throughput` | gauge | `ordinal` | journal `throughput_after` |
| `hyperloom_optimization_gain_percent` | gauge | `ordinal` | journal `gain_pct` |
| `hyperloom_optimization_accuracy` | gauge | `ordinal` | `optimization_stack[].accuracy`, joined by task |

The optimization rows come from `reports/optimization_journal.json`, the only
artifact that records reverted attempts. `ordinal` is the row's journal position,
so the value gauges join to `hyperloom_optimization_info` on it. Accuracy is only
known for adopted attempts. `lever` is truncated to 160 characters.

Throughput values are exported in the session's own graded unit, unconverted.
`hyperloom_crashes` is a gauge, not a counter, because it is read from persisted
state and can be reconciled downward.

### Exporter self-observation

| Metric | Type | Labels |
|---|---|---|
| `hyperloom_exporter_build_info` | gauge (=1) | `version` |
| `hyperloom_exporter_parent_alive` | gauge 0/1 | — |
| `hyperloom_source_up` | gauge 0/1 | `source` |
| `hyperloom_source_age_seconds` | gauge | `source` |
| `hyperloom_source_collect_duration_seconds` | gauge | `source` |
| `hyperloom_source_consecutive_failures` | gauge | `source` |

When no snapshot exists yet (session dir not populated), `/metrics` still
returns 200 with `hyperloom_exporter_build_info`,
`hyperloom_exporter_parent_alive` and `hyperloom_session_observed 0`.

### Deliberately not exported

- GPU utilisation, memory, power, temperature: use the AMD Device Metrics
  Exporter.
- vLLM/SGLang request, KV-cache and token metrics: Prometheus scrapes the
  server's own `/metrics`, found through `/sd/inference` (below).
- The exporter runs its `SessionMonitor` with `gpu=False, server=False` by
  default, so it does not add a second `amd-smi` poller or scraper during a
  benchmark. `--gpu` / `--server` re-enable them for hosts without those
  exporters.

## Inference-server discovery (`/sd/inference`)

The inference server's port changes between launches and there is no server at
all during some phases. `/sd/inference` returns Prometheus HTTP-SD JSON built
from `sources.server.find_session_server(session_dir)`, with an explicit
`HYPERLOOM_VLLM_URL` override taking precedence. Session discovery reads the
live lifecycle server's `{framework}_{port}.pid` / `.json` records; it never
uses the host-wide default-port probe:

```json
[{"targets": ["127.0.0.1:8000"], "labels": {"session_id": "...", "model": "...", "framework": "vllm"}}]
```

or `[]` when no live server is found for this session. This is served over HTTP instead of
written to a `file_sd` file, so the exporter keeps the observability layer's
no-writes guarantee and a Kubernetes-hosted Prometheus (which cannot read a host
file) can use it through `httpSDConfigs`. Discovery runs on its own collector
cadence (default 15s) and the endpoint serves the cached result.

Remote Prometheus caveat: the discovered address is whatever the server binds.
A server on `127.0.0.1` is only scrapeable by a Prometheus on the same host. The
exporter reports the address as found; it does not rewrite it.

## Lifecycle

### Start

After `_acquire_session_lock_or_exit` succeeds, and unless disabled, the run
calls `start_metrics_exporter`, which spawns:

```
python -m hyperloom.observability.exporter \
    --session-dir SD --parent-pid <optimizer pid> --listen 127.0.0.1:9477
```

using the current interpreter, `start_new_session=True` (so Ctrl-C to the
optimizer's terminal does not kill the exporter before its grace period), and
stdout/stderr redirected to the run's existing log directory. The child's PID
and URL are logged once at `INFO`. Spawn failure logs one warning and the run
continues.

### Configuration

| Flag | Env | Default |
|---|---|---|
| `--no-metrics-exporter` | `HYPERLOOM_METRICS_EXPORTER=0` | enabled |
| `--metrics-listen HOST:PORT` | `HYPERLOOM_METRICS_LISTEN` | `127.0.0.1:9477` |
| `--metrics-grace-sec N` | `HYPERLOOM_METRICS_GRACE_SEC` | `120` |

The flag wins over the env var. The env vars let skills and Docker invocations
configure the exporter without editing command lines.

### Bind and handover

- The server sets `SO_REUSEADDR` and retries a busy port with capped backoff
  while `--parent-pid` is supplied and the watchdog has not requested exit.
  SIGTERM/SIGINT interrupt the wait. Without a parent, it tries once; a busy
  port produces one warning and a clean exit 0. Other bind errors are not
  retried. The run is unaffected.
- A previous exporter still in its grace period gives the port up when a new
  optimizer takes over the session: each watchdog pass reads the session lock
  (through the existing read-only `lockfile` source), and if the lock's owner
  PID is alive and differs from `--parent-pid`, the old exporter exits at once.
  A new session can wait through the previous exporter's remaining grace period
  rather than abandoning its endpoint after a fixed timeout.

### Parent watchdog and grace

- Every 2s the exporter checks `os.getppid() == --parent-pid`. The child is
  spawned by the optimizer itself, so when the optimizer exits the child is
  reparented and the check fails. That avoids the PID-reuse race a bare
  `os.kill(pid, 0)` probe would have.
- When the parent is gone, `hyperloom_exporter_parent_alive` becomes 0 and the
  exporter keeps collecting and serving for `--grace-sec` so Prometheus captures
  the terminal snapshot (`liveness="dead"`, `hyperloom_stop_info`, final gain).
  Then it exits 0.
- The optimizer does not stop the exporter in its `finally` block, since that
  would defeat the grace period. The exporter always ends through its own
  watchdog, so it can outlive the optimizer by at most the grace period plus one
  watchdog interval.
- SIGTERM/SIGINT delivered to the exporter itself stop it immediately.
- In Docker mode the exporter shares the container, so if the container exits
  the grace period is lost. That is accepted: Prometheus still has every sample
  up to the exit.

### Future standalone daemon

The same entry point without `--parent-pid` runs until signalled.
`--follow-newest` (not built now) would re-resolve the session with
`assemble.resolve_session_dir` on each watchdog pass and rebuild the monitor
when it changes. Nothing in the render or HTTP layers depends on having a
parent.

## Deployment

### Docker Compose (`examples/observability/compose/`)

- `docker-compose.yml`: Prometheus, Grafana and the AMD Device Metrics
  Exporter. Prometheus uses `network_mode: host` so it can reach the default
  `127.0.0.1:9477` and loopback-bound inference servers.
- `prometheus.yml`: a static job for `127.0.0.1:9477`, an `http_sd_configs`
  job pointed at `http://127.0.0.1:9477/sd/inference`, and the AMD exporter job.
- Grafana provisioning: datasource plus the starter dashboard.

### Existing or Kubernetes Prometheus (`examples/observability/k8s/`)

- The run must use `--metrics-listen 0.0.0.0:9477`, or a specific interface.
- `scrapeconfig.yaml`: a prometheus-operator `ScrapeConfig` with
  `staticConfigs` for `<HOST_IP>:9477` and `httpSDConfigs` for
  `http://<HOST_IP>:9477/sd/inference`.
- `dashboard-configmap.yaml`: the same dashboard JSON, labelled for the Grafana
  sidecar.
- Exposure: the endpoint is unauthenticated and reveals model, framework and
  workload shape. Binding to a non-loopback address is an explicit operator
  choice. The docs recommend a host firewall rule scoped to the Prometheus
  nodes.

### Hyperloom in Docker mode

The ROCm container must either use `--network=host` (the usual setup for these
containers) or publish `-p 9477:9477` together with
`--metrics-listen 0.0.0.0:9477` inside the container. `hyperloom-setup` and the
run skills are updated to mention this.

### Multi-node

The exporter runs only where the optimizer (coordinator) process runs. Worker
nodes are covered by the AMD Device Metrics Exporter.

## Starter dashboard

One dashboard with a `session_id` variable:

- Top row stats: liveness, current phase, elapsed/remaining, gain (raw and
  validated), crashes.
- A state timeline of `hyperloom_phase_current`.
- Phase budget: elapsed vs budget per phase (bar gauge).
- Throughput: baseline vs best over time.
- Work: tasks by state, lane held/capacity, GPU leases next to AMD GPU
  utilisation.
- Current step with time left to its deadline.
- Exporter health: source up, age and collect duration.

## Error handling

- Rendering is total over any `Snapshot`: `None` fields are omitted and label
  values are escaped per the exposition format. A bug in one metric family is
  caught per family, logged, and counted in
  `hyperloom_exporter_render_errors_total`; the rest of the page is still
  served.
- HTTP handler errors return 500 for that request only; the server keeps
  running.
- Source failures already surface as `SourceHealth`; the exporter maps them to
  `hyperloom_source_*` and never raises on them.
- Every failure of the exporter process leaves the optimizer untouched. The
  only coupling is the spawn call, and that is guarded.

## Testing

- **Renderer contract** (`observability/tests/test_render_prometheus.py`) over
  the existing snapshot fixtures in `conftest.py`:
  - output parses as valid exposition format (a small test-side parser; the
    existing `sources.server.parse_prometheus` covers samples, plus
    HELP/TYPE line checks);
  - the metric-name set matches a pinned list (renames are deliberate);
  - `None` fields are omitted;
  - label values are escaped;
  - the cardinality guard fails if any label outside a fixed allow-list
    appears, or if `phase` / `state` / `lane` take values outside their
    enumerations;
  - an empty or `None` snapshot yields the minimal page.
- **Exporter process** (`observability/tests/test_exporter.py`):
  - `/metrics`, `/healthz` and `/sd/inference` respond against a fixture
    session dir;
  - `/metrics` still answers within 1s while a registered source is hung;
  - parent death leads to `parent_alive 0`, then exit after the (shortened)
    grace period;
  - lock handover makes the old exporter exit early;
  - a busy port retries until it becomes free or the parent watchdog requests
    exit; without a parent it leads to one attempt and a clean exit 0;
  - the exporter writes nothing to the session dir (asserted with the existing
    read-only invariant helpers).
- **Run wiring** (`inference_optimizer/tests/`):
  - the exporter is spawned by default, with the expected argv;
  - `--no-metrics-exporter` and `HYPERLOOM_METRICS_EXPORTER=0` suppress it;
  - a spawn failure does not fail the run.
- **Lint and format** under the repository's ruff configuration, with no new
  `noqa` directives.

## Open questions

None blocking. Port 9477 is not in the Prometheus default-port allocation list;
it can be registered later if the exporter is upstreamed.
