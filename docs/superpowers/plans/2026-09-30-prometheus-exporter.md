# Hyperloom Prometheus Exporter Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Every Hyperloom optimizer run serves Prometheus metrics describing its session (phase, budget, work in flight, liveness, results) from a stdlib-only child process that can never affect the run.

**Architecture:** A pure renderer turns the existing `observability.model.Snapshot` into Prometheus text format. A separate exporter process wraps the existing `SessionMonitor` with a stdlib `ThreadingHTTPServer` (`/metrics`, `/healthz`, `/sd/inference`), a parent watchdog with a grace period, and a lock-handover check. The optimizer spawns it just before `coordinator.run(...)` and never stops it. Deployment examples cover Docker Compose and Kubernetes (prometheus-operator).

**Tech Stack:** Python 3.10 standard library only (`http.server`, `subprocess`, `threading`), pytest, ruff; Prometheus text exposition 0.0.4; Grafana dashboard JSON; Docker Compose; prometheus-operator `ScrapeConfig`.

**Spec:** `docs/superpowers/specs/2026-09-30-prometheus-exporter-design.md`

**Implementation status:** The code sketches below record the original plan,
not the final runtime contract. Final behavior is documented in the linked spec:
busy-port retries follow the parent watchdog rather than a fixed 15-second
timeout, inference HTTP-SD is session-scoped, and render-error HELP text counts
failures across scrapes. Unused BLE001 directives are removed while their
isolation reasons remain plain comments.

## Global Constraints

- No new runtime dependencies: `pyproject.toml` keeps `dependencies = []`. No `prometheus_client`.
- The exporter must never fail, slow, or alter a run; every failure path in the spawn helper logs one warning and returns.
- The observability layer never writes the session dir. The exporter's log file is opened by the *optimizer* (parent) at `<session_dir>/runtime/metrics_exporter.log`.
- `import hyperloom.observability` must not import `hyperloom.orchestrator` (enforced by `tests/test_invariants.py::test_package_import_does_not_pull_the_orchestrator`). The exporter module lives in the package, so it is subject to the same rule.
- Defaults: listen `127.0.0.1:9477`, grace `120` s, watchdog interval `2` s, inference discovery cadence `15` s. Busy-port retries continue until binding succeeds or the parent watchdog requests exit; parentless exporters try once.
- Flags / env: `--no-metrics-exporter` / `HYPERLOOM_METRICS_EXPORTER=0`; `--metrics-listen HOST:PORT` / `HYPERLOOM_METRICS_LISTEN`; `--metrics-grace-sec N` / `HYPERLOOM_METRICS_GRACE_SEC`. The flag wins over the env var.
- Metric prefix `hyperloom_`. Constant labels on every series once a snapshot exists: `session_id`, `model`, `framework`. No other label takes unbounded values. `None` values are omitted, never exported as 0.
- Spawn with `start_new_session=True`; the optimizer never kills the exporter.
- Ruff config in `pyproject.toml` (line length 120, rules `E,F,W,BLE,RUF100`). Broad `except Exception` is only allowed with `# noqa: BLE001 - <reason>` where the spec requires isolation.
- Every new Python file starts with the two SPDX header lines used across the repo:
  `# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.` / `# SPDX-License-Identifier: MIT`.
- Tests run with `venv/bin/python -m pytest -q -p no:cacheprovider <paths>` from the repo root.

## Review Focus

1. **Label values with quotes, backslashes, newlines or slashes** (model names such as `Qwen/Qwen3.8-27B`, odd stop reasons) must be escaped rather than corrupting the page. Test: Task 1 `test_escape_label_value` plus Task 2 `test_label_values_are_escaped`.
2. **NaN, infinite or negative values** (an overrun phase has negative remaining budget; a broken throughput reading may be NaN) must render as `NaN`, `+Inf` or a negative number, never crash or be clamped. Test: Task 1 `test_format_value_specials` plus Task 2 `test_negative_and_nan_values_render`.
3. **The exporter starts before `state.json` exists** (a fresh run spawns the exporter immediately after the lock is taken). `/metrics` must return 200 with `hyperloom_session_observed 0`. Test: Task 3 `test_metrics_before_session_exists`.
4. **The optimizer is SIGKILLed or OOM-killed** (no `finally` runs). The exporter must still notice and exit after the grace period. Test: Task 4 `test_exporter_exits_after_parent_is_killed`.
5. **Resume while the previous exporter is in its grace period** (port busy). The new exporter must retry, and the old one must yield on lock takeover. Test: Task 4 `test_bind_retries_until_port_frees` plus `test_watchdog_exits_on_lock_takeover`.

---

## File Structure

| File | Responsibility |
|---|---|
| Create `src/hyperloom/observability/render/prometheus.py` | Exposition primitives (escape, format, family writer) and `render_prometheus(snapshot, *, exporter, on_family_error)` |
| Modify `src/hyperloom/observability/render/__init__.py` | Re-export `render_prometheus`, `ExporterInfo`, `PROMETHEUS_CONTENT_TYPE` |
| Create `src/hyperloom/observability/exporter.py` | Exporter process: state, HTTP handler, SD source, watchdog, bind retry, `main()` |
| Create `src/hyperloom/inference_optimizer/cli/metrics_exporter.py` | Resolve flags/env into a launch config and spawn the exporter |
| Modify `src/hyperloom/inference_optimizer/cli/parser.py` (after `--closing-grace-sec`, ~line 476) | Three new `optimize` flags |
| Modify `src/hyperloom/inference_optimizer/cli/__init__.py` (~line 2436, before `stop_reason = await coordinator.run(`) | One call to `start_metrics_exporter` |
| Create `src/hyperloom/observability/tests/test_render_prometheus.py` | Renderer contract tests (Tasks 1–2) |
| Create `src/hyperloom/observability/tests/test_exporter.py` | HTTP + lifecycle tests (Tasks 3–4) |
| Create `src/hyperloom/inference_optimizer/tests/test_metrics_exporter_cli.py` | Flag/env/spawn/wiring tests (Task 5) |
| Create `src/hyperloom/observability/tests/test_deploy_examples.py` | Dashboard/config reference only real metric names (Task 6) |
| Create `examples/observability/**` | Compose stack, Prometheus config, K8s ScrapeConfigs, dashboard, README |
| Modify `src/hyperloom/skills/hyperloom-setup/SKILL.md` | Docker-mode networking note for the exporter |

---

### Task 1: Exposition primitives

**Files:**
- Create: `src/hyperloom/observability/render/prometheus.py`
- Test: `src/hyperloom/observability/tests/test_render_prometheus.py`

**Interfaces:**
- Produces (used by Task 2 and Task 3):
  - `PROMETHEUS_CONTENT_TYPE: str = "text/plain; version=0.0.4; charset=utf-8"`
  - `class MetricFamily(name: str, kind: str, help: str)` with `.samples: list[tuple[dict[str, str], float]]` and `.add(value: float | int | bool | None, **labels: object) -> None` (a `None` value is dropped)
  - `escape_label_value(value: str) -> str`
  - `format_value(value: float) -> str`
  - `format_families(families: Iterable[MetricFamily], *, const_labels: Mapping[str, str]) -> str`

- [ ] **Step 1: Write the failing tests**

Create `src/hyperloom/observability/tests/test_render_prometheus.py`:

```python
# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Contract tests for the Prometheus renderer."""

from __future__ import annotations

import math

from hyperloom.observability.render.prometheus import (
    MetricFamily,
    escape_label_value,
    format_families,
    format_value,
)


def test_escape_label_value() -> None:
    assert escape_label_value('a"b\\c\nd/e') == 'a\\"b\\\\c\\nd/e'


def test_format_value_specials() -> None:
    assert format_value(1.0) == "1"
    assert format_value(-3600.0) == "-3600"
    assert format_value(0.25) == "0.25"
    assert format_value(math.nan) == "NaN"
    assert format_value(math.inf) == "+Inf"
    assert format_value(-math.inf) == "-Inf"
    assert format_value(1e20) == "1e+20"


def test_none_samples_are_dropped_and_empty_families_skipped() -> None:
    kept = MetricFamily("hyperloom_a", "gauge", "Kept.")
    kept.add(None, phase="X")
    kept.add(2, phase="Y")
    empty = MetricFamily("hyperloom_b", "gauge", "Empty.")
    empty.add(None)

    text = format_families([kept, empty], const_labels={"session_id": "s"})

    assert text == (
        "# HELP hyperloom_a Kept.\n"
        "# TYPE hyperloom_a gauge\n"
        'hyperloom_a{session_id="s",phase="Y"} 2\n'
    )


def test_unlabelled_sample_has_no_braces() -> None:
    family = MetricFamily("hyperloom_c", "counter", "Count.")
    family.add(True)
    assert format_families([family], const_labels={}).splitlines()[-1] == "hyperloom_c 1"
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `venv/bin/python -m pytest -q -p no:cacheprovider src/hyperloom/observability/tests/test_render_prometheus.py`
Expected: FAIL, collection error `ModuleNotFoundError: No module named 'hyperloom.observability.render.prometheus'`

- [ ] **Step 3: Write the implementation**

Create `src/hyperloom/observability/render/prometheus.py`:

```python
# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Prometheus text exposition of a :class:`~..model.Snapshot`.

Metric names are a stable contract, pinned by
``tests/test_render_prometheus.py``: a dashboard or alert keyed on a name must
not break because a model field was renamed. Like :mod:`.json_out`, ``None`` is
never coerced to ``0``. A sample whose value is unknown is omitted, so a panel
shows "no data" rather than a plausible, wrong zero.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field


PROMETHEUS_CONTENT_TYPE = "text/plain; version=0.0.4; charset=utf-8"


@dataclass
class MetricFamily:
    """One ``# HELP`` / ``# TYPE`` block and its samples."""

    name: str
    kind: str
    help: str
    samples: list[tuple[dict[str, str], float]] = field(default_factory=list)

    def add(self, value: float | int | bool | None, **labels: object) -> None:
        """Append a sample; a ``None`` value is dropped rather than zeroed."""
        if value is None:
            return
        self.samples.append(({key: "" if val is None else str(val) for key, val in labels.items()}, float(value)))


def escape_label_value(value: str) -> str:
    """Escape a label value per the exposition format."""
    return value.replace("\\", "\\\\").replace("\n", "\\n").replace('"', '\\"')


def format_value(value: float) -> str:
    """Render a sample value, including the format's NaN/Inf spellings."""
    if math.isnan(value):
        return "NaN"
    if math.isinf(value):
        return "+Inf" if value > 0 else "-Inf"
    if value.is_integer() and abs(value) < 1e15:
        return str(int(value))
    return repr(value)


def format_families(families: Iterable[MetricFamily], *, const_labels: Mapping[str, str]) -> str:
    """Serialise families, prefixing every sample's labels with ``const_labels``."""
    lines: list[str] = []
    for family in families:
        if not family.samples:
            continue
        lines.append(f"# HELP {family.name} {family.help}")
        lines.append(f"# TYPE {family.name} {family.kind}")
        for labels, value in family.samples:
            merged = {**const_labels, **labels}
            if merged:
                body = ",".join(f'{key}="{escape_label_value(val)}"' for key, val in merged.items())
                lines.append(f"{family.name}{{{body}}} {format_value(value)}")
            else:
                lines.append(f"{family.name} {format_value(value)}")
    return "\n".join(lines) + "\n" if lines else ""
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `venv/bin/python -m pytest -q -p no:cacheprovider src/hyperloom/observability/tests/test_render_prometheus.py`
Expected: 4 passed

- [ ] **Step 5: Lint and commit**

```bash
ruff check src/hyperloom/observability/render/prometheus.py src/hyperloom/observability/tests/test_render_prometheus.py
ruff format src/hyperloom/observability/render/prometheus.py src/hyperloom/observability/tests/test_render_prometheus.py
git add src/hyperloom/observability/render/prometheus.py src/hyperloom/observability/tests/test_render_prometheus.py
git commit -m "feat(observability): add Prometheus exposition primitives"
```

---

### Task 2: `render_prometheus` over a Snapshot

**Files:**
- Modify: `src/hyperloom/observability/render/prometheus.py` (append)
- Modify: `src/hyperloom/observability/render/__init__.py`
- Test: `src/hyperloom/observability/tests/test_render_prometheus.py` (append)

**Interfaces:**
- Consumes: Task 1 primitives; `hyperloom.observability.model` types `Snapshot`, `Liveness`, `SourceOutcome`, `CurrentStep`, `SourceHealth`.
- Produces (used by Tasks 3, 6):
  - `@dataclass(frozen=True) class ExporterInfo(version: str, parent_alive: bool | None = None, render_errors_total: int = 0)`
  - `session_labels(snapshot: Snapshot) -> dict[str, str]` returning keys `session_id`, `model`, `framework`
  - `render_prometheus(snapshot: Snapshot | None, *, exporter: ExporterInfo, on_family_error: Callable[[str], None] | None = None) -> str`
  - Test module constant `PINNED_METRIC_NAMES: frozenset[str]` (imported by Task 6's test)

Fixture facts (from `tests/conftest.py`'s `session_dir` fixture with `frozen_clock`): session_id `test-model_20260806T000000Z_deadbeef`, model `test-model`, framework `sglang`; current phase `FRAMEWORK_AGENT` (elapsed 7200, budget ≈15827.59, cap 19440); `PRELUDE` has `budget_total_s=None`; tasks queued=1 running=1 succeeded=1 failed=1 cancelled=0; lane `benchmark_lane` held 1 of 1, `research_lane` 0 of 4; GPU 0 leased and not expired; baseline 100, best 115, gain raw 15 / validated 12, target gap 18, crashes 0; no stop reason; `current_step` is `None`; liveness `unknown`; `last_activity_age_s` is `None`.

- [ ] **Step 1: Write the failing tests**

Append to `src/hyperloom/observability/tests/test_render_prometheus.py` (and merge the new imports into the import block at the top of the file):

```python
import dataclasses
import re
from pathlib import Path

import pytest

from hyperloom.observability import load_snapshot
from hyperloom.observability.model import CurrentStep, Liveness, SourceHealth, SourceOutcome
from hyperloom.observability.render import prometheus as prom
from hyperloom.observability.render.prometheus import ExporterInfo, render_prometheus


PINNED_METRIC_NAMES = frozenset(
    {
        "hyperloom_session_info",
        "hyperloom_session_observed",
        "hyperloom_liveness",
        "hyperloom_state_age_seconds",
        "hyperloom_last_activity_age_seconds",
        "hyperloom_session_start_timestamp_seconds",
        "hyperloom_session_elapsed_seconds",
        "hyperloom_session_remaining_seconds",
        "hyperloom_macro_cycle",
        "hyperloom_tick",
        "hyperloom_phase_current",
        "hyperloom_phase_elapsed_seconds",
        "hyperloom_phase_budget_seconds",
        "hyperloom_phase_budget_remaining_seconds",
        "hyperloom_phase_cap_seconds",
        "hyperloom_tasks",
        "hyperloom_lane_held",
        "hyperloom_lane_capacity",
        "hyperloom_gpu_leased",
        "hyperloom_current_step_info",
        "hyperloom_current_step_start_timestamp_seconds",
        "hyperloom_current_step_deadline_timestamp_seconds",
        "hyperloom_throughput_baseline",
        "hyperloom_throughput_best",
        "hyperloom_gain_percent",
        "hyperloom_target_gap_percent",
        "hyperloom_crashes",
        "hyperloom_stop_info",
        "hyperloom_exporter_build_info",
        "hyperloom_exporter_parent_alive",
        "hyperloom_exporter_render_errors_total",
        "hyperloom_source_up",
        "hyperloom_source_age_seconds",
        "hyperloom_source_collect_duration_seconds",
        "hyperloom_source_consecutive_failures",
    }
)

ALLOWED_LABELS = frozenset(
    {
        "session_id", "model", "framework",
        "framework_version", "gpu_type", "precision", "tp", "ep", "conc", "isl", "osl", "objective_kind",
        "state", "phase", "lane", "gpu_id", "step", "kind", "reason", "source", "version",
    }
)
KNOWN_PHASES = frozenset({"PRELUDE", "ENABLEMENT", "FRAMEWORK_AGENT", "KERNEL_AGENT", "SWEEP", "CLOSE"})

_SAMPLE = re.compile(r"^(?P<name>[a-zA-Z_:][a-zA-Z0-9_:]*)(?:\{(?P<labels>.*)\})? (?P<value>\S+)$")
_LABEL = re.compile(r'(?P<key>[a-zA-Z_][a-zA-Z0-9_]*)="(?P<val>(?:[^"\\]|\\.)*)"')

EXPORTER = ExporterInfo(version="9.9.9", parent_alive=True)


def parse_exposition(text: str) -> tuple[dict[str, str], list[tuple[str, dict[str, str], str]]]:
    """Return ``({family: type}, [(name, labels, raw_value)])`` and assert the format is well-formed."""
    types: dict[str, str] = {}
    samples: list[tuple[str, dict[str, str], str]] = []
    helped: set[str] = set()
    for line in text.splitlines():
        if line.startswith("# HELP "):
            name = line.split()[2]
            assert name not in helped, f"duplicate HELP for {name}"
            helped.add(name)
        elif line.startswith("# TYPE "):
            _, _, name, kind = line.split()
            assert name in helped, f"TYPE before HELP for {name}"
            assert kind in {"gauge", "counter"}
            types[name] = kind
        else:
            match = _SAMPLE.match(line)
            assert match, f"malformed sample line: {line!r}"
            labels = {m["key"]: m["val"] for m in _LABEL.finditer(match["labels"] or "")}
            assert match["name"] in types, f"sample before TYPE: {line!r}"
            samples.append((match["name"], labels, match["value"]))
    return types, samples


def value_of(samples, name: str, **labels: str) -> float:
    hits = [v for n, lab, v in samples if n == name and all(lab.get(k) == w for k, w in labels.items())]
    assert len(hits) == 1, f"{name}{labels}: expected one sample, got {hits}"
    return float(hits[0])


def names_of(samples) -> set[str]:
    return {name for name, _, _ in samples}


@pytest.fixture
def snapshot(session_dir: Path, frozen_clock):
    snap = load_snapshot(session_dir, now_unix=frozen_clock)
    assert snap is not None
    return snap


@pytest.fixture
def full_snapshot(snapshot):
    """The fixture snapshot with every optional field populated."""
    return dataclasses.replace(
        snapshot,
        liveness=Liveness.LIVE,
        last_activity_age_s=5.0,
        current_step=CurrentStep(
            phase="KERNEL_AGENT",
            step="geak_e2e",
            detail="GEAK e2e (from=FRAMEWORK_AGENT)",
            started_unix=1785988000.0,
            deadline_unix=1786019000.0,
        ),
        result=dataclasses.replace(snapshot.result, stop_reason="time_exhausted"),
        source_health=(
            SourceHealth(name="session", outcome=SourceOutcome.OK, age_s=1.5, duration_s=0.02),
            SourceHealth(name="inference_sd", outcome=SourceOutcome.ERROR, age_s=30.0, error="x", consecutive_failures=2),
        ),
    )


def test_every_pinned_metric_is_rendered_for_a_full_snapshot(full_snapshot) -> None:
    types, samples = parse_exposition(render_prometheus(full_snapshot, exporter=EXPORTER))
    assert names_of(samples) == PINNED_METRIC_NAMES
    assert types["hyperloom_exporter_render_errors_total"] == "counter"
    assert {kind for name, kind in types.items() if name != "hyperloom_exporter_render_errors_total"} == {"gauge"}


def test_values_follow_the_snapshot(snapshot) -> None:
    _, samples = parse_exposition(render_prometheus(snapshot, exporter=EXPORTER))
    const = {"session_id": "test-model_20260806T000000Z_deadbeef", "model": "test-model", "framework": "sglang"}

    for _, labels, _ in samples:
        assert {k: labels[k] for k in const} == const

    assert value_of(samples, "hyperloom_phase_current", phase="FRAMEWORK_AGENT") == 1
    assert value_of(samples, "hyperloom_phase_current", phase="PRELUDE") == 0
    assert value_of(samples, "hyperloom_phase_elapsed_seconds", phase="FRAMEWORK_AGENT") == 7200
    assert value_of(samples, "hyperloom_phase_cap_seconds", phase="FRAMEWORK_AGENT") == 19440
    assert value_of(samples, "hyperloom_tasks", state="running") == 1
    assert value_of(samples, "hyperloom_tasks", state="cancelled") == 0
    assert value_of(samples, "hyperloom_lane_held", lane="benchmark_lane") == 1
    assert value_of(samples, "hyperloom_lane_capacity", lane="research_lane") == 4
    assert value_of(samples, "hyperloom_gpu_leased", gpu_id="0") == 1
    assert value_of(samples, "hyperloom_throughput_baseline") == 100
    assert value_of(samples, "hyperloom_throughput_best") == 115
    assert value_of(samples, "hyperloom_gain_percent", kind="raw") == 15
    assert value_of(samples, "hyperloom_gain_percent", kind="validated") == 12
    assert value_of(samples, "hyperloom_liveness", state="unknown") == 1
    assert value_of(samples, "hyperloom_liveness", state="live") == 0
    assert value_of(samples, "hyperloom_session_start_timestamp_seconds") == 1785974400
    assert value_of(samples, "hyperloom_session_info", gpu_type="mi300x", tp="8", precision="fp8") == 1
    assert value_of(samples, "hyperloom_session_observed") == 1
    assert value_of(samples, "hyperloom_exporter_parent_alive") == 1
    assert value_of(samples, "hyperloom_exporter_build_info", version="9.9.9") == 1


def test_none_is_omitted_not_zeroed(snapshot) -> None:
    _, samples = parse_exposition(render_prometheus(snapshot, exporter=ExporterInfo(version="1")))
    budget_phases = {lab["phase"] for n, lab, _ in samples if n == "hyperloom_phase_budget_seconds"}
    assert "PRELUDE" not in budget_phases
    assert "FRAMEWORK_AGENT" in budget_phases
    assert "hyperloom_last_activity_age_seconds" not in names_of(samples)
    assert "hyperloom_current_step_info" not in names_of(samples)
    assert "hyperloom_stop_info" not in names_of(samples)
    assert "hyperloom_exporter_parent_alive" not in names_of(samples)


def test_labels_stay_inside_the_allow_list(full_snapshot) -> None:
    _, samples = parse_exposition(render_prometheus(full_snapshot, exporter=EXPORTER))
    for name, labels, _ in samples:
        assert set(labels) <= ALLOWED_LABELS, f"{name} carries unexpected labels {set(labels) - ALLOWED_LABELS}"
        if "phase" in labels:
            assert labels["phase"] in KNOWN_PHASES
        if name == "hyperloom_tasks":
            assert labels["state"] in {"queued", "running", "succeeded", "failed", "cancelled"}
        if name == "hyperloom_liveness":
            assert labels["state"] in {"live", "stale", "dead", "unknown"}
    assert not any("detail" in labels or "holder" in labels for _, labels, _ in samples)


def test_label_values_are_escaped(snapshot) -> None:
    odd = dataclasses.replace(snapshot, session=dataclasses.replace(snapshot.session, model_display='Qwen/"x"\\y\nz'))
    text = render_prometheus(odd, exporter=EXPORTER)
    assert 'model="Qwen/\\"x\\"\\\\y\\nz"' in text
    parse_exposition(text)


def test_negative_and_nan_values_render(snapshot) -> None:
    overrun = dataclasses.replace(
        snapshot,
        session_remaining_s=-120.0,
        result=dataclasses.replace(snapshot.result, best_tput=float("nan")),
    )
    _, samples = parse_exposition(render_prometheus(overrun, exporter=EXPORTER))
    assert value_of(samples, "hyperloom_session_remaining_seconds") == -120
    assert [v for n, _, v in samples if n == "hyperloom_throughput_best"] == ["NaN"]


def test_source_up_is_zero_only_on_error(full_snapshot) -> None:
    _, samples = parse_exposition(render_prometheus(full_snapshot, exporter=EXPORTER))
    assert value_of(samples, "hyperloom_source_up", source="session") == 1
    assert value_of(samples, "hyperloom_source_up", source="inference_sd") == 0
    assert value_of(samples, "hyperloom_source_consecutive_failures", source="inference_sd") == 2


def test_absent_source_counts_as_up(snapshot) -> None:
    _, samples = parse_exposition(render_prometheus(snapshot, exporter=EXPORTER))
    assert value_of(samples, "hyperloom_source_up", source="lock") == 1


def test_minimal_page_without_a_snapshot() -> None:
    text = render_prometheus(None, exporter=ExporterInfo(version="1.2.3", parent_alive=False))
    _, samples = parse_exposition(text)
    assert names_of(samples) == {
        "hyperloom_exporter_build_info",
        "hyperloom_exporter_parent_alive",
        "hyperloom_exporter_render_errors_total",
        "hyperloom_session_observed",
    }
    assert value_of(samples, "hyperloom_session_observed") == 0
    assert value_of(samples, "hyperloom_exporter_parent_alive") == 0
    assert "session_id=" not in text


def test_a_broken_family_does_not_blank_the_page(snapshot, monkeypatch) -> None:
    def boom(_snapshot):
        raise RuntimeError("bug in one family")

    builders = tuple((name, boom if name == "work" else fn) for name, fn in prom._SNAPSHOT_BUILDERS)
    monkeypatch.setattr(prom, "_SNAPSHOT_BUILDERS", builders)
    failed: list[str] = []

    text = render_prometheus(
        snapshot, exporter=ExporterInfo(version="1", render_errors_total=3), on_family_error=failed.append
    )

    _, samples = parse_exposition(text)
    assert failed == ["work"]
    assert "hyperloom_tasks" not in names_of(samples)
    assert "hyperloom_phase_current" in names_of(samples)
    assert value_of(samples, "hyperloom_exporter_render_errors_total") == 4
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `venv/bin/python -m pytest -q -p no:cacheprovider src/hyperloom/observability/tests/test_render_prometheus.py`
Expected: FAIL with `ImportError: cannot import name 'ExporterInfo'`

- [ ] **Step 3: Write the implementation**

Add the new imports to the top of `src/hyperloom/observability/render/prometheus.py`, merging them into the existing import block:

```python
import logging
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path

from ..model import Liveness, Snapshot, SourceOutcome

log = logging.getLogger(__name__)

TASK_STATES = ("queued", "running", "succeeded", "failed", "cancelled")
```

Append to the end of the same file:

```python
@dataclass(frozen=True)
class ExporterInfo:
    """What the exporter process knows about itself, rendered beside the snapshot."""

    version: str
    parent_alive: bool | None = None
    render_errors_total: int = 0


def session_labels(snapshot: Snapshot) -> dict[str, str]:
    """Constant labels identifying the session on every series."""
    session = snapshot.session
    return {
        "session_id": session.session_id or Path(session.session_dir).name,
        "model": session.model_display or session.model_name or "",
        "framework": session.framework or "",
    }


def _iso_to_unix(ts: str | None) -> float | None:
    if not ts:
        return None
    try:
        parsed = datetime.fromisoformat(ts.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.timestamp()


def _gauge(name: str, help_text: str) -> MetricFamily:
    return MetricFamily(name, "gauge", help_text)


def _session_families(s: Snapshot) -> list[MetricFamily]:
    info = _gauge("hyperloom_session_info", "Static description of the observed session.")
    session = s.session
    info.add(
        1,
        framework_version=session.framework_version,
        gpu_type=session.gpu_type,
        precision=session.precision,
        tp=session.tp,
        ep=session.ep,
        conc=session.conc,
        isl=session.isl,
        osl=session.osl,
        objective_kind=session.objective_kind,
    )
    liveness = _gauge("hyperloom_liveness", "One-hot liveness verdict for the optimizer.")
    for state in Liveness:
        liveness.add(1 if s.liveness is state else 0, state=state.value)
    state_age = _gauge("hyperloom_state_age_seconds", "Seconds since state.json was last written.")
    state_age.add(s.state_age_s)
    activity = _gauge("hyperloom_last_activity_age_seconds", "Seconds since the last observed session activity.")
    activity.add(s.last_activity_age_s)
    start = _gauge("hyperloom_session_start_timestamp_seconds", "Session start time, Unix seconds.")
    start.add(_iso_to_unix(session.started_at))
    elapsed = _gauge("hyperloom_session_elapsed_seconds", "Wall-clock seconds the session has run.")
    elapsed.add(s.session_elapsed_s)
    remaining = _gauge("hyperloom_session_remaining_seconds", "Wall-clock seconds left in the session budget.")
    remaining.add(s.session_remaining_s)
    cycle = _gauge("hyperloom_macro_cycle", "Current macro cycle.")
    cycle.add(s.macro_cycle)
    tick = _gauge("hyperloom_tick", "Current coordinator tick.")
    tick.add(s.tick)
    return [info, liveness, state_age, activity, start, elapsed, remaining, cycle, tick]


def _phase_families(s: Snapshot) -> list[MetricFamily]:
    current = _gauge("hyperloom_phase_current", "1 for the running phase, 0 for the others.")
    elapsed = _gauge("hyperloom_phase_elapsed_seconds", "Seconds spent in each phase.")
    budget = _gauge("hyperloom_phase_budget_seconds", "Budget allotted to each phase, seconds.")
    remaining = _gauge("hyperloom_phase_budget_remaining_seconds", "Budget left in each phase, seconds.")
    cap = _gauge("hyperloom_phase_cap_seconds", "Absolute cap on each phase, seconds.")
    for row in s.phases:
        current.add(1 if row.is_current else 0, phase=row.name)
        elapsed.add(row.elapsed_s, phase=row.name)
        budget.add(row.budget_total_s, phase=row.name)
        remaining.add(row.budget_remaining_s, phase=row.name)
        cap.add(row.cap_s, phase=row.name)
    return [current, elapsed, budget, remaining, cap]


def _work_families(s: Snapshot) -> list[MetricFamily]:
    tasks = _gauge("hyperloom_tasks", "Coordinator tasks by state.")
    for state in TASK_STATES:
        tasks.add(getattr(s.tasks, state), state=state)
    held = _gauge("hyperloom_lane_held", "Leases held on each resource lane.")
    capacity = _gauge("hyperloom_lane_capacity", "Capacity of each resource lane.")
    for lane in s.lanes:
        held.add(lane.held, lane=lane.lane)
        capacity.add(lane.capacity, lane=lane.lane)
    leased = _gauge("hyperloom_gpu_leased", "1 when a GPU holds an unexpired lease.")
    for lease in s.gpu_leases:
        leased.add(0 if lease.expired else 1, gpu_id=lease.gpu_id)
    return [tasks, held, capacity, leased]


def _step_families(s: Snapshot) -> list[MetricFamily]:
    info = _gauge("hyperloom_current_step_info", "The blocking step a phase has published, if any.")
    started = _gauge("hyperloom_current_step_start_timestamp_seconds", "When the current step started, Unix seconds.")
    deadline = _gauge("hyperloom_current_step_deadline_timestamp_seconds", "When the current step will be killed.")
    step = s.current_step
    if step is not None:
        info.add(1, phase=step.phase, step=step.step)
        started.add(step.started_unix)
        deadline.add(step.deadline_unix)
    return [info, started, deadline]


def _result_families(s: Snapshot) -> list[MetricFamily]:
    result = s.result
    baseline = _gauge("hyperloom_throughput_baseline", "Baseline throughput in the session's graded unit.")
    baseline.add(result.baseline_tput)
    best = _gauge("hyperloom_throughput_best", "Best throughput so far in the session's graded unit.")
    best.add(result.best_tput)
    gain = _gauge("hyperloom_gain_percent", "Cumulative gain over baseline, percent.")
    gain.add(result.cumulative_gain_pct, kind="raw")
    gain.add(result.cumulative_gain_validated_pct, kind="validated")
    gap = _gauge("hyperloom_target_gap_percent", "Remaining gap to the target, percent.")
    gap.add(result.target_gap_pct)
    crashes = _gauge("hyperloom_crashes", "Crash count as persisted by the optimizer.")
    crashes.add(result.crash_count)
    stop = _gauge("hyperloom_stop_info", "Why the session stopped; present only once it has.")
    if result.stop_reason:
        stop.add(1, reason=result.stop_reason)
    return [baseline, best, gain, gap, crashes, stop]


def _source_families(s: Snapshot) -> list[MetricFamily]:
    up = _gauge("hyperloom_source_up", "0 when the source's last read failed; absent artifacts count as up.")
    age = _gauge("hyperloom_source_age_seconds", "Age of each source's cached reading.")
    duration = _gauge("hyperloom_source_collect_duration_seconds", "Duration of each source's last read.")
    failures = _gauge("hyperloom_source_consecutive_failures", "Consecutive failed reads per source.")
    for row in s.source_health:
        up.add(0 if row.outcome is SourceOutcome.ERROR else 1, source=row.name)
        age.add(row.age_s, source=row.name)
        duration.add(row.duration_s, source=row.name)
        failures.add(row.consecutive_failures, source=row.name)
    return [up, age, duration, failures]


# Each builder is isolated: one raising must not blank the rest of the page.
_SNAPSHOT_BUILDERS: tuple[tuple[str, Callable[[Snapshot], list[MetricFamily]]], ...] = (
    ("session", _session_families),
    ("phases", _phase_families),
    ("work", _work_families),
    ("step", _step_families),
    ("result", _result_families),
    ("sources", _source_families),
)


def _exporter_families(snapshot: Snapshot | None, exporter: ExporterInfo, *, errors_total: int) -> list[MetricFamily]:
    build = _gauge("hyperloom_exporter_build_info", "Exporter build information.")
    build.add(1, version=exporter.version)
    parent = _gauge("hyperloom_exporter_parent_alive", "1 while the optimizer that spawned the exporter runs.")
    if exporter.parent_alive is not None:
        parent.add(1 if exporter.parent_alive else 0)
    observed = _gauge("hyperloom_session_observed", "1 once the session has been read successfully.")
    observed.add(0 if snapshot is None else 1)
    errors = MetricFamily("hyperloom_exporter_render_errors_total", "counter", "Metric families that failed to render.")
    errors.add(errors_total)
    return [build, parent, observed, errors]


def render_prometheus(
    snapshot: Snapshot | None,
    *,
    exporter: ExporterInfo,
    on_family_error: Callable[[str], None] | None = None,
) -> str:
    """Render a snapshot as a Prometheus exposition page.

    Args:
        snapshot: The observation, or ``None`` before the session was first read.
        exporter: Exporter self-description; ``render_errors_total`` is the count
            accumulated before this render.
        on_family_error: Called with a builder name each time one raises, so the
            caller can accumulate the counter across scrapes.

    Returns:
        The page, ending in a newline.
    """
    families: list[MetricFamily] = []
    const: dict[str, str] = {}
    failed = 0
    if snapshot is not None:
        const = session_labels(snapshot)
        for name, build in _SNAPSHOT_BUILDERS:
            try:
                families.extend(build(snapshot))
            except Exception:  # noqa: BLE001 - one broken family must not blank the whole page
                log.warning("prometheus: rendering the %s family failed", name, exc_info=True)
                failed += 1
                if on_family_error is not None:
                    on_family_error(name)
    families.extend(_exporter_families(snapshot, exporter, errors_total=exporter.render_errors_total + failed))
    return format_families(families, const_labels=const)
```

Update `src/hyperloom/observability/render/__init__.py`:

```python
from .format import Style, detect_style
from .json_out import SCHEMA_VERSION, render_json, to_dict
from .prometheus import PROMETHEUS_CONTENT_TYPE, ExporterInfo, render_prometheus
from .text import render_status


__all__ = [
    "PROMETHEUS_CONTENT_TYPE",
    "SCHEMA_VERSION",
    "ExporterInfo",
    "Style",
    "detect_style",
    "render_json",
    "render_prometheus",
    "render_status",
    "to_dict",
]
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `venv/bin/python -m pytest -q -p no:cacheprovider src/hyperloom/observability/tests/test_render_prometheus.py src/hyperloom/observability/tests/test_invariants.py`
Expected: all pass (15 render tests plus the existing invariants, including the no-orchestrator-import check)

- [ ] **Step 5: Lint and commit**

```bash
ruff check src/hyperloom/observability/render/ src/hyperloom/observability/tests/test_render_prometheus.py
ruff format src/hyperloom/observability/render/ src/hyperloom/observability/tests/test_render_prometheus.py
git add src/hyperloom/observability/render/ src/hyperloom/observability/tests/test_render_prometheus.py
git commit -m "feat(observability): render snapshots as Prometheus metrics"
```

---

### Task 3: Exporter HTTP surface

**Files:**
- Create: `src/hyperloom/observability/exporter.py`
- Test: `src/hyperloom/observability/tests/test_exporter.py`

**Interfaces:**
- Consumes: `render_prometheus`, `ExporterInfo`, `PROMETHEUS_CONTENT_TYPE`, `session_labels` (Task 2); `SessionMonitor` (`collector.py`; `.collector.register(name, fn, interval_s=...)`, `.collector.cache.get(name) -> CachedValue` with `.value`, `.current() -> Snapshot | None`, `.start()` runs one synchronous `collect_once` then starts background threads, `.stop()`); `sources.server.discover_base_url() -> str | None`; `SourceResult.hit(data)`.
- Produces (used by Task 4):
  - `SD_SOURCE = "inference_sd"`
  - `register_inference_sd(monitor: SessionMonitor, *, interval_s: float = SD_INTERVAL_SEC) -> None`
  - `class ExporterState(*, monitor: SessionMonitor, version: str, parent_alive: Callable[[], bool | None] = lambda: None)` with `.metrics_text() -> str` and `.sd_json() -> str`
  - `make_server(state: ExporterState, host: str, port: int) -> ExporterHTTPServer` (raises `OSError` when the bind fails)
  - `exporter_version() -> str`

- [ ] **Step 1: Write the failing tests**

Create `src/hyperloom/observability/tests/test_exporter.py`:

```python
# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Exporter process tests: HTTP surface and lifecycle."""

from __future__ import annotations

import json
import threading
import time
import urllib.error
import urllib.request
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import pytest

from hyperloom.observability import exporter as exp
from hyperloom.observability.collector import SessionMonitor
from hyperloom.observability.sources.base import SourceResult

from .test_invariants import _tree_fingerprint


@contextmanager
def serving(state: exp.ExporterState) -> Iterator[str]:
    server = exp.make_server(state, "127.0.0.1", 0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_address[1]}"
    finally:
        server.shutdown()
        server.server_close()


@contextmanager
def monitored(session_dir: Path) -> Iterator[SessionMonitor]:
    monitor = SessionMonitor(session_dir, gpu=False, server=False)
    exp.register_inference_sd(monitor)
    monitor.start()
    try:
        yield monitor
    finally:
        monitor.stop()


def get(url: str) -> tuple[int, str, str]:
    try:
        with urllib.request.urlopen(url, timeout=5) as response:
            return response.status, response.headers["Content-Type"], response.read().decode()
    except urllib.error.HTTPError as err:
        return err.code, err.headers.get("Content-Type", ""), ""


def test_metrics_endpoint_serves_the_session(session_dir: Path, monkeypatch) -> None:
    monkeypatch.setattr(exp, "discover_base_url", lambda: None)
    with monitored(session_dir) as monitor, serving(exp.ExporterState(monitor=monitor, version="t")) as base:
        status, ctype, body = get(f"{base}/metrics")
    assert status == 200
    assert ctype == "text/plain; version=0.0.4; charset=utf-8"
    assert 'hyperloom_phase_current{session_id="test-model_20260806T000000Z_deadbeef"' in body
    assert "hyperloom_session_observed{" in body
    assert 'hyperloom_source_up{session_id="test-model_20260806T000000Z_deadbeef",model="test-model",framework="sglang",source="inference_sd"} 1' in body


def test_healthz_and_unknown_path(session_dir: Path, monkeypatch) -> None:
    monkeypatch.setattr(exp, "discover_base_url", lambda: None)
    with monitored(session_dir) as monitor, serving(exp.ExporterState(monitor=monitor, version="t")) as base:
        assert get(f"{base}/healthz")[0] == 200
        assert get(f"{base}/nope")[0] == 404


def test_inference_sd_advertises_the_discovered_server(session_dir: Path, monkeypatch) -> None:
    monkeypatch.setattr(exp, "discover_base_url", lambda: "http://127.0.0.1:8000")
    with monitored(session_dir) as monitor, serving(exp.ExporterState(monitor=monitor, version="t")) as base:
        status, ctype, body = get(f"{base}/sd/inference")
    assert status == 200
    assert ctype == "application/json"
    assert json.loads(body) == [
        {
            "targets": ["127.0.0.1:8000"],
            "labels": {
                "session_id": "test-model_20260806T000000Z_deadbeef",
                "model": "test-model",
                "framework": "sglang",
            },
        }
    ]


def test_inference_sd_is_empty_without_a_server(session_dir: Path, monkeypatch) -> None:
    monkeypatch.setattr(exp, "discover_base_url", lambda: None)
    with monitored(session_dir) as monitor, serving(exp.ExporterState(monitor=monitor, version="t")) as base:
        assert json.loads(get(f"{base}/sd/inference")[2]) == []


def test_metrics_before_session_exists(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(exp, "discover_base_url", lambda: None)
    missing = tmp_path / "model" / "20260930T000000Z"
    with monitored(missing) as monitor, serving(exp.ExporterState(monitor=monitor, version="t")) as base:
        status, _, body = get(f"{base}/metrics")
    assert status == 200
    assert "hyperloom_session_observed 0" in body
    assert not missing.exists(), "the exporter must not create the session dir"


def test_metrics_answer_while_a_source_hangs(session_dir: Path, monkeypatch) -> None:
    monkeypatch.setattr(exp, "discover_base_url", lambda: None)
    release = threading.Event()
    calls = {"n": 0}

    def stuck() -> SourceResult:
        calls["n"] += 1
        if calls["n"] > 1:
            release.wait(30)
        return SourceResult.hit(None)

    monitor = SessionMonitor(session_dir, gpu=False, server=False)
    monitor.collector.register("stuck", stuck, interval_s=0.05)
    monitor.start()
    try:
        time.sleep(0.3)
        with serving(exp.ExporterState(monitor=monitor, version="t")) as base:
            started = time.monotonic()
            status, _, _ = get(f"{base}/metrics")
            assert status == 200
            assert time.monotonic() - started < 1.0
    finally:
        release.set()
        monitor.stop()


def test_a_failing_render_returns_500_and_the_server_survives(session_dir: Path, monkeypatch) -> None:
    monkeypatch.setattr(exp, "discover_base_url", lambda: None)
    with monitored(session_dir) as monitor:
        state = exp.ExporterState(monitor=monitor, version="t")
        original = state.metrics_text
        monkeypatch.setattr(state, "metrics_text", lambda: (_ for _ in ()).throw(RuntimeError("x")))
        with serving(state) as base:
            assert get(f"{base}/metrics")[0] == 500
            monkeypatch.setattr(state, "metrics_text", original)
            assert get(f"{base}/metrics")[0] == 200


def test_serving_never_writes_the_session(session_dir: Path, monkeypatch) -> None:
    monkeypatch.setattr(exp, "discover_base_url", lambda: "http://127.0.0.1:8000")
    before = _tree_fingerprint(session_dir)
    with monitored(session_dir) as monitor, serving(exp.ExporterState(monitor=monitor, version="t")) as base:
        for path in ("/metrics", "/sd/inference", "/healthz"):
            get(f"{base}{path}")
    assert _tree_fingerprint(session_dir) == before


def test_render_errors_accumulate_across_scrapes(session_dir: Path, monkeypatch) -> None:
    from hyperloom.observability.render import prometheus as prom

    def boom(_snapshot):
        raise RuntimeError("x")

    monkeypatch.setattr(prom, "_SNAPSHOT_BUILDERS", (("session", boom),))
    monkeypatch.setattr(exp, "discover_base_url", lambda: None)
    with monitored(session_dir) as monitor:
        state = exp.ExporterState(monitor=monitor, version="t")
        state.metrics_text()
        page = state.metrics_text()
    assert "hyperloom_exporter_render_errors_total 2" in page.replace(
        '{session_id="test-model_20260806T000000Z_deadbeef",model="test-model",framework="sglang"}', ""
    )


def test_exporter_version_is_a_string() -> None:
    assert isinstance(exp.exporter_version(), str) and exp.exporter_version()


@pytest.fixture(autouse=True)
def _no_real_discovery_env(monkeypatch):
    monkeypatch.delenv("HYPERLOOM_VLLM_URL", raising=False)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `venv/bin/python -m pytest -q -p no:cacheprovider src/hyperloom/observability/tests/test_exporter.py`
Expected: FAIL with `ImportError: cannot import name 'exporter' from 'hyperloom.observability'`

- [ ] **Step 3: Write the implementation**

Create `src/hyperloom/observability/exporter.py`:

```python
# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Serve a session's snapshot as Prometheus metrics.

Spawned by the optimizer for the life of one run (see
``inference_optimizer/cli/metrics_exporter.py``), or run by hand::

    python -m hyperloom.observability.exporter --session-dir SD --listen 127.0.0.1:9477

Endpoints: ``/metrics`` (exposition format), ``/healthz``, and
``/sd/inference`` (Prometheus HTTP-SD JSON naming the inference server, whose
port changes between launches). Every request renders from the monitor's cache,
so a scrape never waits on session I/O. Like the rest of the package, this
never writes the session directory.
"""

from __future__ import annotations

import json
import logging
import threading
from collections.abc import Callable
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlsplit

from .collector import SessionMonitor
from .render.prometheus import PROMETHEUS_CONTENT_TYPE, ExporterInfo, render_prometheus, session_labels
from .sources.base import SourceResult
from .sources.server import discover_base_url

log = logging.getLogger(__name__)

SD_SOURCE = "inference_sd"
SD_INTERVAL_SEC = 15.0


def exporter_version() -> str:
    """Installed Hyperloom version, or ``"unknown"`` for an uninstalled tree."""
    from importlib.metadata import PackageNotFoundError, version

    try:
        return version("hyperloom")
    except PackageNotFoundError:
        return "unknown"


def _read_inference_target() -> SourceResult:
    # Always a hit, even with no server: an ABSENT result would keep the
    # previous URL cached and keep advertising a server that has gone away.
    return SourceResult.hit({"url": discover_base_url()})


def register_inference_sd(monitor: SessionMonitor, *, interval_s: float = SD_INTERVAL_SEC) -> None:
    """Poll for the inference server on the monitor's collector."""
    monitor.collector.register(SD_SOURCE, _read_inference_target, interval_s=interval_s)


class ExporterState:
    """What the HTTP handler renders from."""

    def __init__(
        self,
        *,
        monitor: SessionMonitor,
        version: str,
        parent_alive: Callable[[], bool | None] = lambda: None,
    ) -> None:
        self.monitor = monitor
        self.version = version
        self._parent_alive = parent_alive
        self._render_errors = 0
        self._lock = threading.Lock()

    def _count_render_error(self, _family: str) -> None:
        with self._lock:
            self._render_errors += 1

    def metrics_text(self) -> str:
        """Render the current snapshot."""
        with self._lock:
            errors = self._render_errors
        info = ExporterInfo(version=self.version, parent_alive=self._parent_alive(), render_errors_total=errors)
        return render_prometheus(self.monitor.current(), exporter=info, on_family_error=self._count_render_error)

    def sd_json(self) -> str:
        """HTTP-SD target list for the inference server; ``[]`` when none is listening."""
        cached = self.monitor.collector.cache.get(SD_SOURCE).value or {}
        url = cached.get("url")
        if not url:
            return "[]"
        netloc = urlsplit(url).netloc
        if not netloc:
            return "[]"
        snapshot = self.monitor.current()
        labels = session_labels(snapshot) if snapshot is not None else {}
        return json.dumps([{"targets": [netloc], "labels": labels}])


class ExporterHTTPServer(ThreadingHTTPServer):
    """Threaded server that can rebind a port left in TIME_WAIT."""

    allow_reuse_address = True
    daemon_threads = True


def make_server(state: ExporterState, host: str, port: int) -> ExporterHTTPServer:
    """Bind the exporter's HTTP server; raises ``OSError`` when the bind fails."""
    routes: dict[str, tuple[Callable[[], str], str]] = {
        # Resolved per request, not bound at build time.
        "/metrics": (lambda: state.metrics_text(), PROMETHEUS_CONTENT_TYPE),
        "/healthz": (lambda: "ok\n", "text/plain; charset=utf-8"),
        "/sd/inference": (lambda: state.sd_json(), "application/json"),
    }

    class Handler(BaseHTTPRequestHandler):
        server_version = "hyperloom-exporter"

        def do_GET(self) -> None:
            path = self.path.split("?", 1)[0]
            route = routes.get(path)
            if route is None:
                self.send_error(404)
                return
            render, content_type = route
            try:
                body = render().encode("utf-8")
            except Exception:  # noqa: BLE001 - a failed request must not take the server down
                log.warning("exporter: rendering %s failed", path, exc_info=True)
                self.send_error(500)
                return
            self.send_response(200)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, format: str, *args: object) -> None:
            log.debug("exporter: " + format, *args)

    return ExporterHTTPServer((host, port), Handler)
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `venv/bin/python -m pytest -q -p no:cacheprovider src/hyperloom/observability/tests/test_exporter.py src/hyperloom/observability/tests/test_invariants.py`
Expected: all pass

- [ ] **Step 5: Lint and commit**

```bash
ruff check src/hyperloom/observability/exporter.py src/hyperloom/observability/tests/test_exporter.py
ruff format src/hyperloom/observability/exporter.py src/hyperloom/observability/tests/test_exporter.py
git add src/hyperloom/observability/exporter.py src/hyperloom/observability/tests/test_exporter.py
git commit -m "feat(observability): serve metrics and inference SD over HTTP"
```

---

### Task 4: Exporter lifecycle — watchdog, bind retry, `main()`

**Files:**
- Modify: `src/hyperloom/observability/exporter.py` (append)
- Test: `src/hyperloom/observability/tests/test_exporter.py` (append)

**Interfaces:**
- Consumes: Task 3 (`ExporterState`, `make_server`, `register_inference_sd`, `exporter_version`); `sources.lockfile.LockFileSource().read(session_dir) -> SourceResult`, where `.data` holds `"pid"` and `"pid_alive"` (`True`, `False` or `None`); `assemble.resolve_session_dir(explicit, *, model) -> Path | None`.
- Produces (used by Task 5):
  - CLI `python -m hyperloom.observability.exporter --session-dir PATH [--parent-pid PID] [--listen HOST:PORT] [--grace-sec N] [--watchdog-interval N] [--bind-retry-sec N] [--gpu] [--server] [--model NAME]`
  - Constants `DEFAULT_LISTEN = "127.0.0.1:9477"`, `DEFAULT_GRACE_SEC = 120.0`, `EXIT_OK = 0`, `EXIT_CONFIG_ERROR = 3`
  - `parse_listen(value: str) -> tuple[str, int]`, `class Watchdog`, `bind_with_retry(...)`, `main(argv: list[str] | None = None) -> int`

- [ ] **Step 1: Write the failing tests**

Append to `src/hyperloom/observability/tests/test_exporter.py`, merging the new imports into the top block:

```python
import errno
import os
import signal
import socket
import subprocess
import sys


class FakeClock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


def lock_result(pid: int | None, alive: bool | None) -> Callable[[Path], SourceResult]:
    return lambda _sd: SourceResult.hit({"pid": pid, "pid_alive": alive})


@pytest.mark.parametrize(
    ("value", "expected"),
    [("127.0.0.1:9477", ("127.0.0.1", 9477)), ("0.0.0.0:0", ("0.0.0.0", 0)), (":9477", ("", 9477))],
)
def test_parse_listen_accepts(value: str, expected: tuple[str, int]) -> None:
    assert exp.parse_listen(value) == expected


@pytest.mark.parametrize("value", ["9477", "host:abc", "host:70000", "[::1]:9477", ""])
def test_parse_listen_rejects(value: str) -> None:
    with pytest.raises(ValueError):
        exp.parse_listen(value)


def test_watchdog_without_a_parent_runs_forever(tmp_path: Path) -> None:
    dog = exp.Watchdog(parent_pid=None, session_dir=tmp_path, grace_s=1.0, read_lock=lock_result(None, None))
    assert dog.parent_alive() is None
    assert dog.poll() == exp.RUN


def test_watchdog_grace_then_exit(tmp_path: Path) -> None:
    clock = FakeClock()
    ppid = {"value": 4242}
    dog = exp.Watchdog(
        parent_pid=4242,
        session_dir=tmp_path,
        grace_s=120.0,
        getppid=lambda: ppid["value"],
        read_lock=lock_result(4242, True),
        clock=clock,
    )
    assert dog.poll() == exp.RUN
    assert dog.parent_alive() is True

    ppid["value"] = 1
    clock.now = 10.0
    assert dog.poll() == exp.GRACE
    assert dog.parent_alive() is False
    clock.now = 129.0
    assert dog.poll() == exp.GRACE
    clock.now = 130.0
    assert dog.poll() == exp.EXIT


def test_watchdog_exits_on_lock_takeover(tmp_path: Path) -> None:
    dog = exp.Watchdog(
        parent_pid=4242,
        session_dir=tmp_path,
        grace_s=120.0,
        getppid=lambda: 1,
        read_lock=lock_result(5555, True),
        clock=FakeClock(),
    )
    assert dog.poll() == exp.EXIT


@pytest.mark.parametrize(("pid", "alive"), [(4242, True), (5555, False), (5555, None), (None, None)])
def test_watchdog_ignores_a_lock_that_is_not_a_live_successor(tmp_path: Path, pid, alive) -> None:
    dog = exp.Watchdog(
        parent_pid=4242,
        session_dir=tmp_path,
        grace_s=120.0,
        getppid=lambda: 4242,
        read_lock=lock_result(pid, alive),
        clock=FakeClock(),
    )
    assert dog.poll() == exp.RUN


def _occupy_port() -> tuple[socket.socket, int]:
    blocker = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    blocker.bind(("127.0.0.1", 0))
    blocker.listen(1)
    return blocker, blocker.getsockname()[1]


def test_bind_gives_up_after_the_retry_window(session_dir: Path) -> None:
    blocker, port = _occupy_port()
    clock = FakeClock()

    def sleep(seconds: float) -> None:
        clock.now += seconds

    try:
        with monitored(session_dir) as monitor:
            server = exp.bind_with_retry(
                exp.ExporterState(monitor=monitor, version="t"), "127.0.0.1", port, retry_s=15.0, sleep=sleep, clock=clock
            )
    finally:
        blocker.close()
    assert server is None
    assert clock.now >= 15.0


def test_bind_retries_until_port_frees(session_dir: Path) -> None:
    blocker, port = _occupy_port()
    clock = FakeClock()

    def sleep(seconds: float) -> None:
        clock.now += seconds
        blocker.close()

    with monitored(session_dir) as monitor:
        server = exp.bind_with_retry(
            exp.ExporterState(monitor=monitor, version="t"), "127.0.0.1", port, retry_s=15.0, sleep=sleep, clock=clock
        )
    assert server is not None
    assert server.server_address[1] == port
    server.server_close()


def test_bind_does_not_retry_other_errors(session_dir: Path, monkeypatch) -> None:
    def refuse(*_args, **_kwargs):
        raise OSError(errno.EADDRNOTAVAIL, "not here")

    monkeypatch.setattr(exp, "make_server", refuse)
    slept: list[float] = []
    with monitored(session_dir) as monitor:
        server = exp.bind_with_retry(
            exp.ExporterState(monitor=monitor, version="t"), "10.255.255.1", 9477, sleep=slept.append
        )
    assert server is None
    assert slept == []


def test_main_rejects_a_bad_listen_address(session_dir: Path) -> None:
    assert exp.main(["--session-dir", str(session_dir), "--listen", "nonsense"]) == exp.EXIT_CONFIG_ERROR


def _free_port() -> int:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


PARENT_SCRIPT = """
import subprocess, sys, time
child = subprocess.Popen(
    [sys.executable, "-m", "hyperloom.observability.exporter", "--session-dir", sys.argv[1],
     "--parent-pid", str(__import__("os").getpid()), "--listen", sys.argv[2],
     "--grace-sec", "1", "--watchdog-interval", "0.2"],
    start_new_session=True,
)
print(child.pid, flush=True)
time.sleep(600)
"""


def _pid_running(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    # A reaped-by-init zombie still answers kill(0); check its state.
    try:
        return Path(f"/proc/{pid}/stat").read_text().split()[2] != "Z"
    except FileNotFoundError:
        return False


def test_exporter_exits_after_parent_is_killed(session_dir: Path) -> None:
    port = _free_port()
    parent = subprocess.Popen(
        [sys.executable, "-c", PARENT_SCRIPT, str(session_dir), f"127.0.0.1:{port}"],
        stdout=subprocess.PIPE,
        text=True,
    )
    child_pid = int(parent.stdout.readline())
    try:
        deadline = time.monotonic() + 20
        while time.monotonic() < deadline:
            try:
                if get(f"http://127.0.0.1:{port}/healthz")[0] == 200:
                    break
            except OSError:
                pass
            time.sleep(0.1)
        else:
            pytest.fail("exporter never started serving")

        status, _, body = get(f"http://127.0.0.1:{port}/metrics")
        assert status == 200
        assert "hyperloom_exporter_parent_alive" in body

        parent.send_signal(signal.SIGKILL)
        parent.wait(10)

        deadline = time.monotonic() + 15
        while time.monotonic() < deadline and _pid_running(child_pid):
            time.sleep(0.2)
        assert not _pid_running(child_pid), "exporter outlived its grace period"
    finally:
        if _pid_running(child_pid):
            os.kill(child_pid, signal.SIGKILL)
        if parent.poll() is None:
            parent.kill()
```

Also add `from collections.abc import Callable, Iterator` to the imports (replacing the existing `Iterator` import).

- [ ] **Step 2: Run the tests to verify they fail**

Run: `venv/bin/python -m pytest -q -p no:cacheprovider src/hyperloom/observability/tests/test_exporter.py`
Expected: FAIL with `AttributeError: module 'hyperloom.observability.exporter' has no attribute 'parse_listen'` (and similar for `Watchdog`, `RUN`, `bind_with_retry`, `main`)

- [ ] **Step 3: Write the implementation**

Add these imports to `src/hyperloom/observability/exporter.py`, merging them into its import block:

```python
import argparse
import errno
import os
import signal
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

from .assemble import resolve_session_dir
from .sources.lockfile import LockFileSource
```

Add these constants after `SD_INTERVAL_SEC`:

```python
DEFAULT_LISTEN = "127.0.0.1:9477"
DEFAULT_GRACE_SEC = 120.0
WATCHDOG_INTERVAL_SEC = 2.0
BIND_RETRY_SEC = 15.0

# Mirrors ``inference_optimizer/tools/status.py``: the exit code reports whether
# the command ran, so an exporter that could not bind still exits 0.
EXIT_OK = 0
EXIT_CONFIG_ERROR = 3

RUN = "run"
GRACE = "grace"
EXIT = "exit"
```

Append to the end of the file:

```python
def parse_listen(value: str) -> tuple[str, int]:
    """Split ``HOST:PORT``; an empty host binds every IPv4 interface."""
    host, sep, port_text = value.rpartition(":")
    if not sep:
        raise ValueError(f"listen address {value!r} is not HOST:PORT")
    if host.startswith("["):
        raise ValueError(f"listen address {value!r}: IPv6 is not supported")
    try:
        port = int(port_text)
    except ValueError:
        raise ValueError(f"listen address {value!r} has a non-numeric port") from None
    if not 0 <= port <= 65535:
        raise ValueError(f"listen address {value!r} has an out-of-range port")
    return host, port


@dataclass
class Watchdog:
    """Decide, once per interval, whether the exporter should keep serving.

    The optimizer spawns the exporter directly, so ``os.getppid()`` stops
    matching the moment the optimizer exits (however it exits) and the child is
    reparented. That avoids the PID-reuse race a bare ``kill(pid, 0)`` has.
    """

    parent_pid: int | None
    session_dir: Path
    grace_s: float
    getppid: Callable[[], int] = os.getppid
    read_lock: Callable[[Path], SourceResult] = field(default_factory=lambda: LockFileSource().read)
    clock: Callable[[], float] = time.monotonic
    _parent_gone_at: float | None = None

    def parent_alive(self) -> bool | None:
        """``None`` when running without a parent."""
        if self.parent_pid is None:
            return None
        return self.getppid() == self.parent_pid

    def poll(self) -> str:
        """Return :data:`RUN`, :data:`GRACE` or :data:`EXIT`."""
        if self.parent_pid is None:
            return RUN
        if self._taken_over():
            return EXIT
        if self.parent_alive():
            return RUN
        now = self.clock()
        if self._parent_gone_at is None:
            self._parent_gone_at = now
        return EXIT if now - self._parent_gone_at >= self.grace_s else GRACE

    def _taken_over(self) -> bool:
        # A resumed run took the session lock: free the port for its exporter.
        result = self.read_lock(self.session_dir)
        if not result.ok or not isinstance(result.data, dict):
            return False
        pid = result.data.get("pid")
        return pid is not None and pid != self.parent_pid and result.data.get("pid_alive") is True


def bind_with_retry(
    state: ExporterState,
    host: str,
    port: int,
    *,
    retry_s: float = BIND_RETRY_SEC,
    sleep: Callable[[float], None] = time.sleep,
    clock: Callable[[], float] = time.monotonic,
) -> ExporterHTTPServer | None:
    """Bind, retrying a busy port for ``retry_s``; ``None`` means give up quietly."""
    deadline = clock() + retry_s
    delay = 0.25
    while True:
        try:
            return make_server(state, host, port)
        except OSError as exc:
            if exc.errno != errno.EADDRINUSE or clock() >= deadline:
                log.warning("exporter: cannot bind %s:%s (%s); not serving metrics", host, port, exc)
                return None
        sleep(delay)
        delay = min(delay * 2, 2.0)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="hyperloom-exporter", description=__doc__.split("\n\n")[0])
    parser.add_argument("--session-dir", default=None, help="Session directory; auto-discovered when omitted.")
    parser.add_argument("--model", default=None, help="Narrow session auto-discovery to one model basename.")
    parser.add_argument("--parent-pid", type=int, default=None, help="Exit a grace period after this process ends.")
    parser.add_argument("--listen", default=DEFAULT_LISTEN, help=f"HOST:PORT to serve on (default {DEFAULT_LISTEN}).")
    parser.add_argument("--grace-sec", type=float, default=DEFAULT_GRACE_SEC, help="Serve this long after the parent exits.")
    parser.add_argument("--watchdog-interval", type=float, default=WATCHDOG_INTERVAL_SEC, help=argparse.SUPPRESS)
    parser.add_argument("--bind-retry-sec", type=float, default=BIND_RETRY_SEC, help=argparse.SUPPRESS)
    parser.add_argument("--gpu", action="store_true", help="Also run the amd-smi probe (off: use the AMD exporter).")
    parser.add_argument("--server", action="store_true", help="Also scrape the inference server (off: Prometheus does).")
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the exporter until signalled or its watchdog says to exit."""
    args = build_parser().parse_args(argv)
    try:
        host, port = parse_listen(args.listen)
    except ValueError as exc:
        print(f"hyperloom-exporter: {exc}", file=sys.stderr)
        return EXIT_CONFIG_ERROR
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    # An explicit dir is used even before it exists: a fresh run spawns the
    # exporter before state.json is written.
    session_dir = Path(args.session_dir) if args.session_dir else resolve_session_dir(None, model=args.model)
    if session_dir is None:
        print("hyperloom-exporter: no session directory found", file=sys.stderr)
        return EXIT_CONFIG_ERROR

    watchdog = Watchdog(parent_pid=args.parent_pid, session_dir=session_dir, grace_s=args.grace_sec)
    monitor = SessionMonitor(session_dir, gpu=args.gpu, server=args.server)
    register_inference_sd(monitor)
    state = ExporterState(monitor=monitor, version=exporter_version(), parent_alive=watchdog.parent_alive)

    stop = threading.Event()
    for signum in (signal.SIGTERM, signal.SIGINT):
        signal.signal(signum, lambda *_: stop.set())

    with monitor:
        server = bind_with_retry(state, host, port, retry_s=args.bind_retry_sec)
        if server is None:
            return EXIT_OK
        thread = threading.Thread(target=server.serve_forever, name="hyperloom-exporter-http", daemon=True)
        thread.start()
        log.info("exporter: serving http://%s:%s/metrics for %s", host or "0.0.0.0", server.server_address[1], session_dir)
        try:
            while not stop.wait(args.watchdog_interval):
                verdict = watchdog.poll()
                if verdict == EXIT:
                    log.info("exporter: parent gone or session taken over; exiting")
                    break
        finally:
            server.shutdown()
            server.server_close()
    return EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `venv/bin/python -m pytest -q -p no:cacheprovider src/hyperloom/observability/tests/`
Expected: all pass. `test_exporter_exits_after_parent_is_killed` takes about 3–6s.

- [ ] **Step 5: Lint and commit**

```bash
ruff check src/hyperloom/observability/
ruff format src/hyperloom/observability/exporter.py src/hyperloom/observability/tests/test_exporter.py
git add src/hyperloom/observability/exporter.py src/hyperloom/observability/tests/test_exporter.py
git commit -m "feat(observability): exporter lifecycle with parent watchdog and grace"
```

---

### Task 5: Launch the exporter from every optimizer run

**Files:**
- Create: `src/hyperloom/inference_optimizer/cli/metrics_exporter.py`
- Modify: `src/hyperloom/inference_optimizer/cli/parser.py` (insert after the `--closing-grace-sec` argument block, ~line 476–486)
- Modify: `src/hyperloom/inference_optimizer/cli/__init__.py` (~line 2436, immediately before `stop_reason = await coordinator.run(`)
- Test: `src/hyperloom/inference_optimizer/tests/test_metrics_exporter_cli.py`

**Interfaces:**
- Consumes: exporter CLI from Task 4 (`--session-dir`, `--parent-pid`, `--listen`, `--grace-sec`); `hyperloom.common.env.env_bool(name, default, *, env=None)`.
- Produces:
  - `ENV_ENABLE = "HYPERLOOM_METRICS_EXPORTER"`, `ENV_LISTEN = "HYPERLOOM_METRICS_LISTEN"`, `ENV_GRACE = "HYPERLOOM_METRICS_GRACE_SEC"`
  - `@dataclass(frozen=True) class ExporterLaunch(listen: str, grace_sec: float)`
  - `resolve_launch(args: argparse.Namespace, environ: Mapping[str, str]) -> ExporterLaunch | None`
  - `build_command(session_dir: Path, launch: ExporterLaunch, *, parent_pid: int, python: str = sys.executable) -> list[str]`
  - `start_metrics_exporter(session_dir: Path, args: argparse.Namespace, *, environ: Mapping[str, str] | None = None, popen: Callable[..., subprocess.Popen] = subprocess.Popen) -> subprocess.Popen | None`

The defaults are duplicated here as literals instead of imported from `hyperloom.observability.exporter`, so the optimizer's CLI does not import the exporter module at startup. `test_defaults_match_the_exporter` keeps the two copies in sync.

- [ ] **Step 1: Write the failing tests**

Create `src/hyperloom/inference_optimizer/tests/test_metrics_exporter_cli.py`:

```python
# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The optimizer spawns the metrics exporter by default and never depends on it."""

from __future__ import annotations

import inspect
import socket
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

import pytest

from hyperloom.inference_optimizer import cli
from hyperloom.inference_optimizer.cli import metrics_exporter as me


def _parse(argv: list[str]):
    return cli._build_parser().parse_args(["optimize", "--model", "/tmp/m", *argv])


def test_defaults_match_the_exporter() -> None:
    from hyperloom.observability import exporter

    assert me.DEFAULT_LISTEN == exporter.DEFAULT_LISTEN
    assert me.DEFAULT_GRACE_SEC == exporter.DEFAULT_GRACE_SEC


def test_enabled_by_default() -> None:
    assert me.resolve_launch(_parse([]), {}) == me.ExporterLaunch(listen="127.0.0.1:9477", grace_sec=120.0)


@pytest.mark.parametrize(
    ("argv", "env"),
    [(["--no-metrics-exporter"], {}), ([], {"HYPERLOOM_METRICS_EXPORTER": "0"}), ([], {"HYPERLOOM_METRICS_EXPORTER": "false"})],
)
def test_disabled_by_flag_or_env(argv, env) -> None:
    assert me.resolve_launch(_parse(argv), env) is None


def test_env_configures_and_flags_win() -> None:
    env = {"HYPERLOOM_METRICS_LISTEN": "0.0.0.0:9500", "HYPERLOOM_METRICS_GRACE_SEC": "30"}
    assert me.resolve_launch(_parse([]), env) == me.ExporterLaunch(listen="0.0.0.0:9500", grace_sec=30.0)
    flagged = _parse(["--metrics-listen", "127.0.0.1:9600", "--metrics-grace-sec", "5"])
    assert me.resolve_launch(flagged, env) == me.ExporterLaunch(listen="127.0.0.1:9600", grace_sec=5.0)


def test_a_bad_grace_env_falls_back_to_the_default() -> None:
    assert me.resolve_launch(_parse([]), {"HYPERLOOM_METRICS_GRACE_SEC": "soon"}).grace_sec == 120.0


def test_build_command(tmp_path: Path) -> None:
    command = me.build_command(tmp_path, me.ExporterLaunch("127.0.0.1:9477", 120.0), parent_pid=99, python="/py")
    assert command == [
        "/py", "-m", "hyperloom.observability.exporter",
        "--session-dir", str(tmp_path),
        "--parent-pid", "99",
        "--listen", "127.0.0.1:9477",
        "--grace-sec", "120.0",
    ]


def test_spawns_detached_with_its_log_in_runtime(tmp_path: Path) -> None:
    calls: list[tuple[list[str], dict]] = []

    def fake_popen(command, **kwargs):
        calls.append((command, kwargs))
        return "proc"

    assert me.start_metrics_exporter(tmp_path, _parse([]), environ={}, popen=fake_popen) == "proc"
    (command, kwargs), = calls
    assert command[:3] == [sys.executable, "-m", "hyperloom.observability.exporter"]
    assert kwargs["start_new_session"] is True
    assert kwargs["stdin"] is subprocess.DEVNULL
    assert kwargs["stderr"] is subprocess.STDOUT
    assert (tmp_path / "runtime" / "metrics_exporter.log").exists()


def test_disabled_never_spawns(tmp_path: Path) -> None:
    def fail(*_a, **_k):
        raise AssertionError("spawned while disabled")

    assert me.start_metrics_exporter(tmp_path, _parse(["--no-metrics-exporter"]), environ={}, popen=fail) is None


def test_a_spawn_failure_never_fails_the_run(tmp_path: Path, caplog) -> None:
    def broken(*_a, **_k):
        raise OSError("no such interpreter")

    assert me.start_metrics_exporter(tmp_path, _parse([]), environ={}, popen=broken) is None
    assert "metrics exporter" in caplog.text


def test_the_run_starts_the_exporter_before_the_coordinator() -> None:
    source = inspect.getsource(cli._run_optimize)
    spawn = source.index("start_metrics_exporter(session_dir, args)")
    assert spawn < source.index("await coordinator.run(")


def _free_port() -> int:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


def test_spawned_exporter_serves_metrics(tmp_path: Path) -> None:
    session_dir = tmp_path / "model" / "20260930T000000Z"
    session_dir.mkdir(parents=True)
    port = _free_port()
    proc = me.start_metrics_exporter(
        session_dir, _parse(["--metrics-listen", f"127.0.0.1:{port}", "--metrics-grace-sec", "1"]), environ={}
    )
    assert proc is not None
    try:
        deadline = time.monotonic() + 20
        body = ""
        while time.monotonic() < deadline:
            try:
                with urllib.request.urlopen(f"http://127.0.0.1:{port}/metrics", timeout=2) as response:
                    body = response.read().decode()
                break
            except OSError:
                time.sleep(0.2)
        assert "hyperloom_exporter_parent_alive 1" in body
    finally:
        proc.terminate()
        proc.wait(10)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `venv/bin/python -m pytest -q -p no:cacheprovider src/hyperloom/inference_optimizer/tests/test_metrics_exporter_cli.py`
Expected: FAIL with `ImportError: cannot import name 'metrics_exporter'`

- [ ] **Step 3: Write the helper module**

Create `src/hyperloom/inference_optimizer/cli/metrics_exporter.py`:

```python
# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Spawn the Prometheus metrics exporter for one optimizer run.

The exporter is a detached child (``start_new_session=True``) that watches this
process and exits a grace period after it ends, so the run's final state stays
scrapeable. The run never waits for, stops, or depends on it: every failure
here is one warning. See ``docs/superpowers/specs/2026-09-30-prometheus-exporter-design.md``.
"""

from __future__ import annotations

import argparse
import logging
import os
import subprocess
import sys
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path

from hyperloom.common.env import env_bool

log = logging.getLogger(__name__)

ENV_ENABLE = "HYPERLOOM_METRICS_EXPORTER"
ENV_LISTEN = "HYPERLOOM_METRICS_LISTEN"
ENV_GRACE = "HYPERLOOM_METRICS_GRACE_SEC"

# Kept equal to hyperloom.observability.exporter's defaults by a test; not
# imported, so the optimizer CLI does not load the exporter at startup.
DEFAULT_LISTEN = "127.0.0.1:9477"
DEFAULT_GRACE_SEC = 120.0

LOG_RELPATH = ("runtime", "metrics_exporter.log")


@dataclass(frozen=True)
class ExporterLaunch:
    listen: str
    grace_sec: float


def resolve_launch(args: argparse.Namespace, environ: Mapping[str, str]) -> ExporterLaunch | None:
    """Combine flags and env; ``None`` when the exporter is disabled. Flags win."""
    if getattr(args, "no_metrics_exporter", False) or not env_bool(ENV_ENABLE, True, env=environ):
        return None
    listen = getattr(args, "metrics_listen", None) or environ.get(ENV_LISTEN, "").strip() or DEFAULT_LISTEN
    grace = getattr(args, "metrics_grace_sec", None)
    if grace is None:
        raw = environ.get(ENV_GRACE, "").strip()
        try:
            grace = float(raw) if raw else DEFAULT_GRACE_SEC
        except ValueError:
            log.warning("ignoring %s=%r: not a number", ENV_GRACE, raw)
            grace = DEFAULT_GRACE_SEC
    return ExporterLaunch(listen=listen, grace_sec=float(grace))


def build_command(session_dir: Path, launch: ExporterLaunch, *, parent_pid: int, python: str = sys.executable) -> list[str]:
    return [
        python, "-m", "hyperloom.observability.exporter",
        "--session-dir", str(session_dir),
        "--parent-pid", str(parent_pid),
        "--listen", launch.listen,
        "--grace-sec", str(launch.grace_sec),
    ]  # fmt: skip


def start_metrics_exporter(
    session_dir: Path,
    args: argparse.Namespace,
    *,
    environ: Mapping[str, str] | None = None,
    popen: Callable[..., subprocess.Popen] = subprocess.Popen,
) -> subprocess.Popen | None:
    """Spawn the exporter unless disabled; never raises."""
    launch = resolve_launch(args, os.environ if environ is None else environ)
    if launch is None:
        log.info("metrics exporter disabled")
        return None
    command = build_command(Path(session_dir), launch, parent_pid=os.getpid())
    log_path = Path(session_dir).joinpath(*LOG_RELPATH)
    try:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with open(log_path, "ab") as log_file:
            proc = popen(
                command,
                stdin=subprocess.DEVNULL,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                start_new_session=True,
                close_fds=True,
            )
    except OSError:
        log.warning("metrics exporter failed to start; continuing without /metrics", exc_info=True)
        return None
    log.info("metrics exporter pid=%s serving http://%s/metrics (log: %s)", getattr(proc, "pid", "?"), launch.listen, log_path)
    return proc
```

- [ ] **Step 4: Add the flags**

In `src/hyperloom/inference_optimizer/cli/parser.py`, directly after the full `opt.add_argument("--closing-grace-sec", ...)` call, insert:

```python
    opt.add_argument(
        "--no-metrics-exporter",
        action="store_true",
        help="Do not start the Prometheus metrics exporter for this run "
        "(also HYPERLOOM_METRICS_EXPORTER=0).",
    )
    opt.add_argument(
        "--metrics-listen",
        default=None,
        help="HOST:PORT for the metrics exporter (default 127.0.0.1:9477, or "
        "$HYPERLOOM_METRICS_LISTEN). Use 0.0.0.0:9477 for a remote Prometheus.",
    )
    opt.add_argument(
        "--metrics-grace-sec",
        type=float,
        default=None,
        help="Seconds the exporter keeps serving after the run exits (default 120, "
        "or $HYPERLOOM_METRICS_GRACE_SEC).",
    )
```

- [ ] **Step 5: Wire the call**

In `src/hyperloom/inference_optimizer/cli/__init__.py`, find:

```python
    stop_reason: str | None = None
    try:
        stop_reason = await coordinator.run(
```

and change it to:

```python
    # Detached /metrics for this leg; it outlives the run by a grace period and
    # can never fail it.
    from .metrics_exporter import start_metrics_exporter

    start_metrics_exporter(session_dir, args)

    stop_reason: str | None = None
    try:
        stop_reason = await coordinator.run(
```

Before editing, confirm that `session_dir` is the variable name in scope at that point: `grep -n "session_dir" src/hyperloom/inference_optimizer/cli/__init__.py | awk -F: '$1>2380 && $1<2440'` must show `session_dir` used nearby (for example in `_write_cli_terminal_artifacts(session_dir, ...)` just below).

- [ ] **Step 6: Run the tests to verify they pass**

Run: `venv/bin/python -m pytest -q -p no:cacheprovider src/hyperloom/inference_optimizer/tests/test_metrics_exporter_cli.py src/hyperloom/inference_optimizer/tests/test_phase_budget_pct_cli.py src/hyperloom/inference_optimizer/tests/test_status_cli.py`
Expected: all pass

- [ ] **Step 7: Lint and commit**

```bash
ruff check src/hyperloom/inference_optimizer/cli/ src/hyperloom/inference_optimizer/tests/test_metrics_exporter_cli.py
ruff format src/hyperloom/inference_optimizer/cli/metrics_exporter.py src/hyperloom/inference_optimizer/tests/test_metrics_exporter_cli.py
git add src/hyperloom/inference_optimizer/cli/ src/hyperloom/inference_optimizer/tests/test_metrics_exporter_cli.py
git commit -m "feat(cli): start the metrics exporter with every optimizer run"
```

`ruff format` must not reformat `parser.py` or `__init__.py` beyond the inserted lines. Run `git diff --stat` before committing to confirm.

---

### Task 6: Deployment examples, dashboard, docs

**Files:**
- Create: `examples/observability/README.md`
- Create: `examples/observability/dashboards/hyperloom.json`
- Create: `examples/observability/compose/docker-compose.yml`
- Create: `examples/observability/compose/prometheus.yml`
- Create: `examples/observability/compose/grafana/provisioning/datasources/prometheus.yml`
- Create: `examples/observability/compose/grafana/provisioning/dashboards/hyperloom.yml`
- Create: `examples/observability/k8s/scrapeconfig.yaml`
- Modify: `src/hyperloom/skills/hyperloom-setup/SKILL.md`
- Test: `src/hyperloom/observability/tests/test_deploy_examples.py`

**Interfaces:**
- Consumes: `PINNED_METRIC_NAMES` from `hyperloom.observability.tests.test_render_prometheus` (Task 2); endpoint paths and the default port from Tasks 3–4.

- [ ] **Step 1: Write the failing test**

Create `src/hyperloom/observability/tests/test_deploy_examples.py`:

```python
# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The shipped dashboard and scrape configs must track the real metric names."""

from __future__ import annotations

import json
import re
from pathlib import Path

from .test_render_prometheus import PINNED_METRIC_NAMES

EXAMPLES = Path(__file__).resolve().parents[4] / "examples" / "observability"


def _exprs(panel_or_dashboard: dict) -> list[str]:
    found = [t["expr"] for t in panel_or_dashboard.get("targets", []) if "expr" in t]
    for child in panel_or_dashboard.get("panels", []):
        found.extend(_exprs(child))
    return found


def test_dashboard_only_uses_real_hyperloom_metrics() -> None:
    dashboard = json.loads((EXAMPLES / "dashboards" / "hyperloom.json").read_text())
    exprs = _exprs(dashboard)
    assert exprs, "dashboard has no queries"
    used = {name for expr in exprs for name in re.findall(r"\bhyperloom_[a-z_]+\b", expr)}
    assert used, "dashboard queries no hyperloom metrics"
    assert used <= PINNED_METRIC_NAMES, f"dashboard references unknown metrics: {used - PINNED_METRIC_NAMES}"
    variable = json.dumps(dashboard["templating"])
    assert "hyperloom_session_info" in variable


def test_scrape_configs_point_at_the_exporter_endpoints() -> None:
    compose = (EXAMPLES / "compose" / "prometheus.yml").read_text()
    k8s = (EXAMPLES / "k8s" / "scrapeconfig.yaml").read_text()
    for text in (compose, k8s):
        assert ":9477" in text
        assert "/sd/inference" in text
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `venv/bin/python -m pytest -q -p no:cacheprovider src/hyperloom/observability/tests/test_deploy_examples.py`
Expected: FAIL with `FileNotFoundError` for `examples/observability/dashboards/hyperloom.json`

- [ ] **Step 3: Write the dashboard**

Create `examples/observability/dashboards/hyperloom.json`:

```json
{
  "title": "Hyperloom session",
  "uid": "hyperloom-session",
  "schemaVersion": 39,
  "refresh": "15s",
  "time": {"from": "now-6h", "to": "now"},
  "tags": ["hyperloom"],
  "templating": {
    "list": [
      {"name": "datasource", "type": "datasource", "query": "prometheus", "label": "Datasource"},
      {
        "name": "session",
        "label": "Session",
        "type": "query",
        "datasource": {"type": "prometheus", "uid": "${datasource}"},
        "query": {"query": "label_values(hyperloom_session_info, session_id)", "refId": "session"},
        "definition": "label_values(hyperloom_session_info, session_id)",
        "refresh": 2,
        "sort": 2
      }
    ]
  },
  "panels": [
    {
      "id": 1, "type": "stat", "title": "Liveness",
      "gridPos": {"x": 0, "y": 0, "w": 4, "h": 4},
      "datasource": {"type": "prometheus", "uid": "${datasource}"},
      "options": {"textMode": "name", "reduceOptions": {"calcs": ["lastNotNull"]}},
      "targets": [{"refId": "A", "expr": "hyperloom_liveness{session_id=\"$session\"} == 1", "legendFormat": "{{state}}"}]
    },
    {
      "id": 2, "type": "stat", "title": "Current phase",
      "gridPos": {"x": 4, "y": 0, "w": 5, "h": 4},
      "datasource": {"type": "prometheus", "uid": "${datasource}"},
      "options": {"textMode": "name", "reduceOptions": {"calcs": ["lastNotNull"]}},
      "targets": [{"refId": "A", "expr": "hyperloom_phase_current{session_id=\"$session\"} == 1", "legendFormat": "{{phase}}"}]
    },
    {
      "id": 3, "type": "stat", "title": "Gain over baseline",
      "gridPos": {"x": 9, "y": 0, "w": 5, "h": 4},
      "datasource": {"type": "prometheus", "uid": "${datasource}"},
      "fieldConfig": {"defaults": {"unit": "percent", "decimals": 1}},
      "targets": [{"refId": "A", "expr": "hyperloom_gain_percent{session_id=\"$session\"}", "legendFormat": "{{kind}}"}]
    },
    {
      "id": 4, "type": "stat", "title": "Session clock",
      "gridPos": {"x": 14, "y": 0, "w": 6, "h": 4},
      "datasource": {"type": "prometheus", "uid": "${datasource}"},
      "fieldConfig": {"defaults": {"unit": "s"}},
      "targets": [
        {"refId": "A", "expr": "hyperloom_session_elapsed_seconds{session_id=\"$session\"}", "legendFormat": "elapsed"},
        {"refId": "B", "expr": "hyperloom_session_remaining_seconds{session_id=\"$session\"}", "legendFormat": "remaining"}
      ]
    },
    {
      "id": 5, "type": "stat", "title": "Crashes",
      "gridPos": {"x": 20, "y": 0, "w": 4, "h": 4},
      "datasource": {"type": "prometheus", "uid": "${datasource}"},
      "targets": [{"refId": "A", "expr": "hyperloom_crashes{session_id=\"$session\"}"}]
    },
    {
      "id": 6, "type": "state-timeline", "title": "Phase timeline",
      "gridPos": {"x": 0, "y": 4, "w": 24, "h": 5},
      "datasource": {"type": "prometheus", "uid": "${datasource}"},
      "options": {"showValue": "never", "mergeValues": true},
      "targets": [{"refId": "A", "expr": "hyperloom_phase_current{session_id=\"$session\"}", "legendFormat": "{{phase}}"}]
    },
    {
      "id": 7, "type": "timeseries", "title": "Throughput (graded unit)",
      "gridPos": {"x": 0, "y": 9, "w": 12, "h": 8},
      "datasource": {"type": "prometheus", "uid": "${datasource}"},
      "targets": [
        {"refId": "A", "expr": "hyperloom_throughput_baseline{session_id=\"$session\"}", "legendFormat": "baseline"},
        {"refId": "B", "expr": "hyperloom_throughput_best{session_id=\"$session\"}", "legendFormat": "best"}
      ]
    },
    {
      "id": 8, "type": "bargauge", "title": "Phase budget used",
      "gridPos": {"x": 12, "y": 9, "w": 12, "h": 8},
      "datasource": {"type": "prometheus", "uid": "${datasource}"},
      "fieldConfig": {"defaults": {"unit": "percentunit", "min": 0, "max": 1}},
      "options": {"displayMode": "gradient", "orientation": "horizontal", "reduceOptions": {"calcs": ["lastNotNull"]}},
      "targets": [{
        "refId": "A",
        "expr": "hyperloom_phase_elapsed_seconds{session_id=\"$session\"} / hyperloom_phase_budget_seconds{session_id=\"$session\"}",
        "legendFormat": "{{phase}}"
      }]
    },
    {
      "id": 9, "type": "timeseries", "title": "Tasks by state",
      "gridPos": {"x": 0, "y": 17, "w": 12, "h": 8},
      "datasource": {"type": "prometheus", "uid": "${datasource}"},
      "targets": [{"refId": "A", "expr": "hyperloom_tasks{session_id=\"$session\"}", "legendFormat": "{{state}}"}]
    },
    {
      "id": 10, "type": "timeseries", "title": "Lane occupancy",
      "gridPos": {"x": 12, "y": 17, "w": 12, "h": 8},
      "datasource": {"type": "prometheus", "uid": "${datasource}"},
      "targets": [
        {"refId": "A", "expr": "hyperloom_lane_held{session_id=\"$session\"}", "legendFormat": "{{lane}} held"},
        {"refId": "B", "expr": "hyperloom_lane_capacity{session_id=\"$session\"}", "legendFormat": "{{lane}} capacity"}
      ]
    },
    {
      "id": 11, "type": "stat", "title": "Current step: time to deadline",
      "gridPos": {"x": 0, "y": 25, "w": 12, "h": 4},
      "datasource": {"type": "prometheus", "uid": "${datasource}"},
      "fieldConfig": {"defaults": {"unit": "s"}},
      "targets": [{
        "refId": "A",
        "expr": "hyperloom_current_step_deadline_timestamp_seconds{session_id=\"$session\"} - time()",
        "legendFormat": "deadline"
      }]
    },
    {
      "id": 12, "type": "table", "title": "Collection health",
      "gridPos": {"x": 12, "y": 25, "w": 12, "h": 8},
      "datasource": {"type": "prometheus", "uid": "${datasource}"},
      "targets": [
        {"refId": "A", "expr": "hyperloom_source_up{session_id=\"$session\"}", "format": "table", "instant": true},
        {"refId": "B", "expr": "hyperloom_source_age_seconds{session_id=\"$session\"}", "format": "table", "instant": true}
      ],
      "transformations": [{"id": "merge"}]
    },
    {
      "id": 13, "type": "timeseries", "title": "Exporter",
      "gridPos": {"x": 0, "y": 29, "w": 12, "h": 4},
      "datasource": {"type": "prometheus", "uid": "${datasource}"},
      "targets": [
        {"refId": "A", "expr": "hyperloom_exporter_parent_alive{session_id=\"$session\"}", "legendFormat": "optimizer alive"},
        {"refId": "B", "expr": "hyperloom_state_age_seconds{session_id=\"$session\"}", "legendFormat": "state.json age (s)"}
      ]
    }
  ]
}
```

- [ ] **Step 4: Write the Compose stack**

The AMD Device Metrics Exporter tag `v1.5.0` and its default port 5000 come from AMD's Docker installation guide (instinct.docs.amd.com/projects/device-metrics-exporter). Confirm the tag pulls with `docker pull rocm/device-metrics-exporter:v1.5.0` before committing.

Create `examples/observability/compose/docker-compose.yml`:

```yaml
# Local Prometheus + Grafana + AMD GPU metrics for watching a Hyperloom run.
# Host networking so Prometheus reaches the exporter on 127.0.0.1:9477 and any
# loopback-bound inference server it discovers through /sd/inference.
services:
  prometheus:
    image: prom/prometheus:v2.53.0
    network_mode: host
    command:
      - --config.file=/etc/prometheus/prometheus.yml
      - --storage.tsdb.retention.time=30d
      - --web.listen-address=127.0.0.1:9090
    volumes:
      - ./prometheus.yml:/etc/prometheus/prometheus.yml:ro
      - prometheus-data:/prometheus
    restart: unless-stopped

  grafana:
    image: grafana/grafana:11.1.0
    network_mode: host
    environment:
      GF_SERVER_HTTP_PORT: "3000"
      GF_AUTH_ANONYMOUS_ENABLED: "true"
      GF_AUTH_ANONYMOUS_ORG_ROLE: Viewer
    volumes:
      - ./grafana/provisioning:/etc/grafana/provisioning:ro
      - ../dashboards:/var/lib/grafana/dashboards/hyperloom:ro
      - grafana-data:/var/lib/grafana
    restart: unless-stopped

  amd-device-metrics-exporter:
    image: rocm/device-metrics-exporter:v1.5.0
    network_mode: host
    devices:
      - /dev/kfd
      - /dev/dri
    volumes:
      - /sys:/sys:ro  # in-band RAS metrics
    restart: unless-stopped

volumes:
  prometheus-data:
  grafana-data:
```

Create `examples/observability/compose/prometheus.yml`:

```yaml
global:
  scrape_interval: 15s
  evaluation_interval: 15s

scrape_configs:
  # Session state from the exporter every optimizer run starts.
  - job_name: hyperloom
    static_configs:
      - targets: ["127.0.0.1:9477"]

  # The inference server Hyperloom is currently benchmarking; its port changes
  # between launches, so the exporter advertises it.
  - job_name: hyperloom-inference
    http_sd_configs:
      - url: http://127.0.0.1:9477/sd/inference
        refresh_interval: 30s

  - job_name: amd-gpu
    static_configs:
      - targets: ["127.0.0.1:5000"]
```

Create `examples/observability/compose/grafana/provisioning/datasources/prometheus.yml`:

```yaml
apiVersion: 1
datasources:
  - name: Prometheus
    uid: prometheus
    type: prometheus
    access: proxy
    url: http://127.0.0.1:9090
    isDefault: true
```

Create `examples/observability/compose/grafana/provisioning/dashboards/hyperloom.yml`:

```yaml
apiVersion: 1
providers:
  - name: hyperloom
    folder: Hyperloom
    type: file
    options:
      path: /var/lib/grafana/dashboards/hyperloom
```

- [ ] **Step 5: Write the Kubernetes ScrapeConfigs**

Create `examples/observability/k8s/scrapeconfig.yaml`:

```yaml
# prometheus-operator ScrapeConfigs for a Hyperloom host outside the cluster.
# Replace GPU_HOST_IP with the host's address and match the labels to your
# Prometheus's scrapeConfigSelector (kube-prometheus-stack: release: <name>).
# The run must listen beyond loopback: --metrics-listen 0.0.0.0:9477.
apiVersion: monitoring.coreos.com/v1alpha1
kind: ScrapeConfig
metadata:
  name: hyperloom
  labels:
    release: prometheus
spec:
  scrapeInterval: 15s
  staticConfigs:
    - targets: ["GPU_HOST_IP:9477"]
      labels:
        job: hyperloom
---
apiVersion: monitoring.coreos.com/v1alpha1
kind: ScrapeConfig
metadata:
  name: hyperloom-inference
  labels:
    release: prometheus
spec:
  scrapeInterval: 15s
  httpSDConfigs:
    - url: http://GPU_HOST_IP:9477/sd/inference
      refreshInterval: 30s
```

- [ ] **Step 6: Write the README**

Create `examples/observability/README.md`:

````markdown
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

The endpoint has no authentication and reveals the model, framework and
workload shape. Allow port 9477 only from your Prometheus nodes.

## Hyperloom in Docker

Run the ROCm container with `--network=host` (the usual setup), or publish
`-p 9477:9477` and set `HYPERLOOM_METRICS_LISTEN=0.0.0.0:9477` inside it. If
the container exits, the grace period is lost; everything up to that point is
already in Prometheus.

## What is not here

GPU hardware metrics come from the AMD Device Metrics Exporter, and inference
server metrics come from the server's own `/metrics` (found via
`/sd/inference`). The exporter does not duplicate either.
````

- [ ] **Step 7: Document Docker networking in the setup skill**

In `src/hyperloom/skills/hyperloom-setup/SKILL.md`, find the paragraph around line 137 that mentions `` `docker run` / `docker exec` later `` and append this paragraph after it:

```markdown
Metrics exporter: every optimizer run serves Prometheus metrics on
`127.0.0.1:9477` (disable with `--no-metrics-exporter` or
`HYPERLOOM_METRICS_EXPORTER=0`). In Docker mode, reach it from the host by
running the container with `--network=host`, or publish `-p 9477:9477` and set
`HYPERLOOM_METRICS_LISTEN=0.0.0.0:9477`. See `examples/observability/README.md`.
```

- [ ] **Step 8: Run the tests to verify they pass and validate the configs**

Run: `venv/bin/python -m pytest -q -p no:cacheprovider src/hyperloom/observability/tests/test_deploy_examples.py`
Expected: 2 passed

Run: `docker run --rm -v "$PWD/examples/observability/compose/prometheus.yml:/p.yml:ro" --entrypoint promtool prom/prometheus:v2.53.0 check config /p.yml`
Expected: `SUCCESS`

Run: `cd examples/observability/compose && docker compose config -q && cd -`
Expected: no output, exit 0

- [ ] **Step 9: Commit**

```bash
git add examples/observability src/hyperloom/observability/tests/test_deploy_examples.py src/hyperloom/skills/hyperloom-setup/SKILL.md
git commit -m "docs(observability): Compose and Kubernetes examples with a starter dashboard"
```

---

### Task 7: End-to-end verification

**Files:** none are created. This task only verifies.

- [ ] **Step 1: Full observability and CLI test suites**

Run: `venv/bin/python -m pytest -q -p no:cacheprovider src/hyperloom/observability src/hyperloom/inference_optimizer/tests/test_metrics_exporter_cli.py src/hyperloom/inference_optimizer/tests/test_status_cli.py src/hyperloom/inference_optimizer/tests/test_current_step_beacon.py src/hyperloom/inference_optimizer/tests/test_phase_budget_pct_cli.py`
Expected: all pass

- [ ] **Step 2: Lint everything touched**

Run: `ruff check $(git diff --name-only main... -- '*.py') && ruff format --check $(git diff --name-only main... -- '*.py')`
Expected: `All checks passed!` and no files would be reformatted

- [ ] **Step 3: Smoke-test against a real, finished session**

```bash
SD=$(ls -d session/*/2026* | tail -n 1)
venv/bin/python -m hyperloom.observability.exporter --session-dir "$SD" --listen 127.0.0.1:19477 &
sleep 3
curl -s 127.0.0.1:19477/metrics | grep -E '^hyperloom_(phase_current|stop_info|gain_percent|session_observed)'
curl -s 127.0.0.1:19477/sd/inference; echo
kill %1
```

Expected: `hyperloom_session_observed{...} 1`, one `hyperloom_phase_current{...} 1`, a `hyperloom_stop_info{...reason="..."} 1` for a finished session, and `/sd/inference` returning `[]` (no server running). Record the output in the task report.

- [ ] **Step 4: Report**

Summarise the test counts, the lint result, and the smoke output. Do not push; the user decides when.
