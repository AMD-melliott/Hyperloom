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

import logging
import math
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from ..model import Liveness, Snapshot, SourceOutcome


log = logging.getLogger(__name__)

PROMETHEUS_CONTENT_TYPE = "text/plain; version=0.0.4; charset=utf-8"
TASK_STATES = ("queued", "running", "succeeded", "failed", "cancelled")


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
    errors = MetricFamily(
        "hyperloom_exporter_render_errors_total", "counter", "Metric sections that failed to render, across scrapes."
    )
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
            except Exception:  # one broken family must not blank the whole page
                log.warning("prometheus: rendering the %s family failed", name, exc_info=True)
                failed += 1
                if on_family_error is not None:
                    on_family_error(name)
    families.extend(_exporter_families(snapshot, exporter, errors_total=exporter.render_errors_total + failed))
    return format_families(families, const_labels=const)
