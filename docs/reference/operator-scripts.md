---
myst:
    html_meta:
        "description": "Reference for Hyperloom operator scripts: status, dump_session_breakdown, dump_session_report, and event_counts. Use these utilities to monitor, inspect, export, and report on session data, including live GPU and inference-server metrics."
        "keywords": "Hyperloom, operator scripts, session status, progress, monitoring, live metrics, amd-smi, GPU utilization, vLLM metrics, framework version, model identity, ISL, OSL, session breakdown, session report, event counts, LLM inference, AMD GPU, ROCm, debugging, observability, operator tools"
---
# Hyperloom operator scripts

A short reference for the operator-facing scripts under
`src/hyperloom/inference_optimizer/tools/`. These are not part of the agent loop —
they are utilities you run by hand against a finished or in-progress
session directory.

When no explicit `--session-dir` is given, scripts resolve the active session in
two steps: `INFERENCE_OPTIMIZER_CURRENT_SESSION_DIR` when set, otherwise the
workspace root (`USER_DATA_PATH`, falling back to `/workspace/hyperloom`). This
does **not** auto-discover the latest `$USER_DATA_PATH/<model>/<ts>/` per-session
subdir — under the per-model timestamp layout, pass `--session-dir` explicitly
(or rely on `INFERENCE_OPTIMIZER_CURRENT_SESSION_DIR`, which the CLI sets during
a run). See [Hyperloom authentication and credentials](authentication.md).

---

## `status.py`

Print the current phase, wall-clock budget, lane and GPU occupancy, task
counts, and recent lifecycle events for a session. Read-only, so it is safe to
run against a live optimization.

Also available as a subcommand: `python -m hyperloom.inference_optimizer.cli status`.

Use this when:

* You want to know what step a running optimization is on without reading the
  log firehose.
* You need a machine-readable progress snapshot for a dashboard or a wrapper
  script (`--json`).
* You are checking whether a session is still alive after a suspected crash.

Unlike the other scripts on this page, `status.py` auto-discovers the newest
`$USER_DATA_PATH/<model>/<timestamp>/` session when no `--session-dir` is
given, instead of stopping at the workspace root.

### Usage

```bash
# Active session (auto-discovered)
python -m hyperloom.inference_optimizer.tools.status

# A specific session
python -m hyperloom.inference_optimizer.tools.status --session-dir <SD>

# Machine-readable snapshot
python -m hyperloom.inference_optimizer.tools.status --session-dir <SD> --json

# Live view (Ctrl-C to exit)
python -m hyperloom.inference_optimizer.tools.status --watch

# Live view: repaint every 0.5s, re-read the session every 5s
python -m hyperloom.inference_optimizer.tools.status --watch --interval 0.5 --collect-interval 5
```

Useful flags: `--no-gpu` and `--no-server` skip those probes, `--vllm-url`
pins the metrics endpoint instead of discovering it, and `--show-sources`
keeps the collection-health footer visible even when nothing is wrong.

### Reading the output

| Field | Meaning |
|-------|---------|
| Header line 1 | The model being optimized and the inference stack running it, e.g. `Qwen/Qwen3-14B-FP8 · vllm 0.27.2rc1.dev150+g311b3513a`. The model name is recovered from `model_path` when it points into a Hugging Face cache, because the manifest's `model_name` is then the snapshot commit sha. The framework version comes from the manifest's `stack_fingerprint`. |
| Header line 2 | Topology and workload shape: `mi300x · TP=1 · conc=64 · ISL 1024 · OSL 1024 · bf16`. Folded onto line 1 when the terminal is wide enough. |
| `running` / `working (loop quiet)` / `ended` / `liveness unknown` | State of the optimizer. `ended` means the session recorded a `stop_reason`. `working (loop quiet)` means the phase machine has not ticked recently but the session tree is still being written to — normally a long blocking dispatch such as a GEAK run, not a fault. |
| `last tick ... ago` / `activity ... ago` | Age of the newest phase-machine write, and of the newest write anywhere in the session tree. A large gap between the two is the signature of a blocked phase. |
| `KERNEL_AGENT → geak_e2e` | The long-running step the phase is blocked on, from `runtime/current_step.json`, with its elapsed time, budget, and kill deadline. |
| `ELAPSED` | `HH:MM`, hours not wrapped at 24. Cumulative across **all** macro cycles, not just the current entry — phases repeat, since SWEEP loops back to EXPLORE. |
| `BUDGET` | The charge-back allotment for the running phase: its share of the time *still remaining*, renormalized over itself and the phases ahead, so an overrunning earlier phase shrinks every later one. This is also the figure the phase's own agent is told it has left. |
| `CAP` | The flat wall-clock ceiling, `min(max_minutes × pct, 24h × pct)`. Unlike `BUDGET` it does not move. Blank for PRELUDE and CLOSE, which have no exit check that consults it. |
| `USED` | Elapsed against whichever limit this phase's exit check actually consults, and which one that is. EXPLORE, KERNEL_AGENT and SWEEP exit on the budget **or** the cap, so the smaller binds; FRAMEWORK_AGENT exits on the cap alone. Above 100% is a real overrun and is reported, not clamped. |
| `GPU (host-wide)` | Utilization, VRAM and power from `amd-smi`. **Host-wide**: these come from the node's driver and include every tenant, so on a shared box they are not attributable to this session. |
| `vLLM` | Live counters from the server's `/metrics`. Absent during KERNEL_AGENT, when no server is running. Throughput needs two samples, so it shows `—` on the first frame. |
| `WORK` | What sub-agents are doing: GEAK round/engineer progress, and each live run's heartbeat `note` — a short description the agent writes itself every few minutes. |
| `SOURCES` | Per-source collection health. Shown when a probe is failing, with the age of the value still on screen; always shown under `--show-sources`. |
| `gain ... validated` | Gain re-measured on a fresh server. A provisional per-round figure is shown alongside only when the two disagree. |
| `—` | Not measured. Never a substitute for a real zero. |

### Behaviour under `--watch`

Collection runs on background threads; the paint loop only reads a cache. Two
consequences worth knowing:

* **A slow or hung probe costs one field, not the frame.** The timers keep
  ticking at `--interval` regardless of what the collectors are doing.
* **A failing probe keeps its last good value**, dimmed and stamped with its
  real age, rather than blanking or showing a fabricated zero. The `SOURCES`
  row names what broke.

`--json` and one-shot mode stay fully synchronous, so a single invocation
performs exactly one read and produces exactly one document.

Exit codes: `0` when the session was read (whatever state it is in), `3` when
no session directory could be resolved, `130` on Ctrl-C, `128+N` when killed by
signal N. As with `examine`-style commands, the exit code reports whether the
command **ran**, not what it found — so a script can distinguish "could not
look" from "looked, and the run is over".

### JSON schema

`--json` emits `schema_version: 2`. Version 2 is additive — every version 1 key
is unchanged — and adds three blocks: `activity` (current step, running work,
recent writes, GEAK progress), `metrics` (`gpu`, `server`), and `sources`
(per-source outcome, age, error, consecutive failures). `rendered_at_unix`
alongside `observed_at_unix` tells a consumer how old the underlying reading is.

Each `phases[]` entry carries `limit_s` and `limit_kind` (`"cap"` or
`"budget"`) alongside `budget_total_s` and `cap_s`. `pct_used` is measured
against `limit_s`; `pct_of_budget` is the charge-back ratio on its own. Both are
`null` for PRELUDE and CLOSE, whose caps are computable but unenforced.

`session` also gained `model_display`, `model_path`, `model_revision`, and
`framework_version`. `model_name` still carries the manifest's literal value —
often a Hugging Face snapshot sha — so it remains a valid join key against other
Hyperloom artifacts; `model_display` is the resolved `org/repo` and is what a
human-facing consumer should read.

---

## `dump_session_breakdown.py`

Produce a `session_breakdown.json` from a session directory. Same
builder as the live Coordinator `session_breakdown` action and the
`cli.py` finally-block safety net.

Use this when:

* You want to (re)produce the breakdown for a historical session on a shared
  filesystem.
* A live session crashed before reaching the closing phase and you
  want the partial breakdown anyway.
* You need to bulk-export breakdowns for downstream indexing.

### Usage

Use these commands to produce a session breakdown.

```bash
# Live session in the current sandbox (USER_DATA_PATH or /workspace/hyperloom)
python -m hyperloom.inference_optimizer.tools.dump_session_breakdown

# Historical session on a shared filesystem
python -m hyperloom.inference_optimizer.tools.dump_session_breakdown \
    --session-dir /shared/hyperloom-sessions/<user>/<sid>

# Override output path (don't touch session_dir)
python -m hyperloom.inference_optimizer.tools.dump_session_breakdown \
    --session-dir <SD> --output /tmp/breakdown-<sid>.json

# Bulk historical
for d in /shared/hyperloom-sessions/*/*; do
    [ -d "$d" ] || continue
    python -m hyperloom.inference_optimizer.tools.dump_session_breakdown \
        --session-dir "$d" > /dev/null
done
```

The default output path is `<session_dir>/session_breakdown.json`
(overwrites if present; the file is rebuilt deterministically from raw
artifacts).

### Output

`session_breakdown.json` conforming to
[`session_breakdown.json` integration in Hyperloom](session-breakdown.md).
The script exits 0 on success, prints a one-line summary, and writes
collector warnings to the `warnings[]` field rather than failing.

---

## `dump_session_report.py`

Render a markdown session report from a `session_breakdown.json`.
Deterministic by default; optionally large language model (LLM)-polished when an
OpenAI-compatible endpoint is configured.

Use this when:

* You want a human-readable summary to paste into a PR, Slack, or
  email.
* You want to generate the same report for many sessions in bulk.

### Usage

Use the following commands to render a session report.

```bash
# Deterministic only (no LLM):
python -m hyperloom.inference_optimizer.tools.dump_session_report \
    --input  /shared/hyperloom-sessions/<user>/<sid>/session_breakdown.json \
    --output /shared/hyperloom-sessions/<user>/<sid>/session_report.md

# With LLM-polished prose (OpenAI-compatible endpoint):
HYPERLOOM_REPORT_LLM_BACKEND=openai \
OPENAI_BASE_URL=https://your-openai-compatible-endpoint/v1 \
OPENAI_API_KEY=... \
python -m hyperloom.inference_optimizer.tools.dump_session_report \
    --input  /shared/hyperloom-sessions/<user>/<sid>/session_breakdown.json \
    --output /shared/hyperloom-sessions/<user>/<sid>/session_report.md
```

When `--output` is omitted the report is written to
`<session_dir>/session_report.md` next to the input file. The LLM
user prompt and raw response (when used) are persisted alongside as
`session_report_prompt.json` / `session_report_llm_raw.txt` so
hallucinations can be audited after the fact.

### LLM hardening

The script applies the following safeguards when LLM polishing is enabled.

* The deterministic skeleton (headline numbers, action_path,
  kernel_lifecycle counts) is generated *without* the LLM; the LLM
  only rewrites prose.
* If the LLM call fails (timeout, 5xx, malformed response), the
  script falls back to the deterministic report and exits 0.
* If you do not want any LLM call, leave
  `HYPERLOOM_REPORT_LLM_BACKEND` unset.

---

## `event_counts.py`

Print recent action / proposal / kernel counts from a session's
`coordinator.db`.

Use this when:

* You want a quick "is this session making progress?" check without
  reading logs.
* You are debugging an apparent stall and want to see what kind of
  events are landing.

### Usage

Use the following commands to print event counts for a session.

```bash
python -m hyperloom.inference_optimizer.tools.event_counts            # default session_dir
python -m hyperloom.inference_optimizer.tools.event_counts /path/to/session
```

Reads at most the last 500 events from
`$SESSION_DIR/storage/coordinator.db` and emits a JSON object of
`{category: count}`. Exit code 2 if the database (DB) is missing.

### Example output

The script emits a JSON object of event categories and counts.

```json
{
  "delegated:kernel_optimization:succeeded": 7,
  "delegated:tracelens_analysis:succeeded": 1,
  "kernel_request:kernel_optimization": 7,
  "kernel_request:tracelens_analysis": 1,
  "kernel_response:kernel_optimization:KEEP": 3,
  "kernel_response:kernel_optimization:NEEDS_REVIEW": 4,
  "proposal:explore": 12,
  "proposal:specialist": 4,
  "proposal:kernel_opt": 5
}
```

A long run with healthy progress has roughly proportional
`proposal:*` and `delegated:*:succeeded` counts. A stuck run typically
shows many `kernel_request:*` and few `kernel_response:*`.

---

## Additional operator tools

The same tools package also contains smaller utilities that are useful during
incident response or launch validation:

* `backfill_langfuse.py`: replay one finished session's `reports/trace/` into
  Langfuse after the fact:
  `python -m hyperloom.inference_optimizer.tools.backfill_langfuse --session-dir <SD> [--dry-run]`.
* `preflight_optimizer.py`: launcher-side local preflight for stale serving
  processes, torch/ROCm device visibility, GPU VRAM occupancy (exits non-zero
  when any card exceeds 1% of its total capacity), and model path existence:
  `python src/hyperloom/inference_optimizer/tools/preflight_optimizer.py MODEL_PATH`.
  A non-zero exit must abort the launch.
* `read_optimizer_state.py`: concise `state.json` / lifecycle summary:
  `python src/hyperloom/inference_optimizer/tools/read_optimizer_state.py SESSION_DIR`.
* `robustness_monitor.sh.example`: shell example for polling robustness
  findings around a session; copy/adapt it for local operator workflows.

---

## Related topics

Use these resources for related reference information:

* [`session_breakdown.json` integration in Hyperloom](session-breakdown.md): The schema produced by `dump_session_breakdown.py`.
* [Hyperloom self-hosting and operations guide](operations.md): Retention recommendations, including which scripts' outputs to back up long-term.
* [Troubleshooting Hyperloom](troubleshooting.md): Symptoms vs which script to reach for first.
