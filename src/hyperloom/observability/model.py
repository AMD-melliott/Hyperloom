# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Frozen domain model for the read-only visibility layer.

This module imports nothing from Hyperloom. It is the shared vocabulary every
source produces into and every renderer consumes from, so that adding an output
format never touches data acquisition and vice versa.

Conventions
-----------
* **Units live in field names** (``elapsed_s``, ``tput_tok_s``, ``gain_pct``).
* **``None`` means "not measured" and is never fabricated as ``0``.** Renderers
  show an em dash. A zero that was never observed is a lie a status display
  cannot afford.
* **Warnings travel inside the snapshot.** A degraded source records a warning
  and returns a partial; it never raises. This mirrors the contract the
  ``breakdown`` collectors already follow.

Duck-typing contract with the phase machine
-------------------------------------------
:class:`Snapshot` is deliberately shaped so it can be handed straight to the
budget functions in :mod:`hyperloom.orchestrator.phases.machine_state`, whose
docstrings call their parameter a "Frozen SharedState view". The field names
``phase``, ``phase_started_unix``, ``phase_elapsed_totals``,
``phase_budget_pct``, ``max_minutes``, ``cycle_minutes``, ``start_ts`` and
``explore_elapsed_accum_s`` are therefore **load-bearing** — renaming one
silently changes budget math to a default instead of raising.

Two traps in that contract, both verified against the implementation:

* ``phase_cumulative_seconds`` guards with ``isinstance(totals, dict)``, so
  :attr:`Snapshot.phase_elapsed_totals` must be a real ``dict``. A
  ``MappingProxyType`` would read as "nothing banked" and silently under-report
  every phase's elapsed time.
* ``session_remaining_seconds`` does **not** consult the ``_now_unix`` hook; it
  calls ``datetime.now`` unless ``now_unix`` is passed explicitly. Callers must
  thread :meth:`Snapshot.now_unix` into every call that accepts it, which
  :mod:`hyperloom.observability.progress` does.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable


class Liveness(Enum):
    """Whether the optimizer that owns this session is still running.

    ``UNKNOWN`` is a real answer and must never be rendered as ``LIVE``. The
    session lock file is never unlinked, so its presence proves only that a run
    once started here.
    """

    LIVE = "live"
    STALE = "stale"
    DEAD = "dead"
    UNKNOWN = "unknown"


class Freshness(Enum):
    """How recently ``state.json`` was rewritten, relative to the staleness threshold."""

    FRESH = "fresh"
    STALE = "stale"
    UNKNOWN = "unknown"


class SourceOutcome(Enum):
    """Result of one source read.

    ``ABSENT`` and ``ERROR`` are deliberately distinct: ``coordinator.db`` does
    not exist during early PRELUDE, and reporting that as a failure makes a
    healthy run look broken.
    """

    OK = "ok"
    ABSENT = "absent"
    ERROR = "error"


@dataclass(frozen=True)
class SessionInfo:
    """Identity and workload shape, sourced from ``manifest.json`` with ``state.json`` fallback."""

    session_dir: str
    session_id: str | None = None
    model_name: str | None = None
    framework: str | None = None
    gpu_type: str | None = None
    tp: int | None = None
    ep: int | None = None
    conc: int | None = None
    isl: int | None = None
    osl: int | None = None
    precision: str | None = None
    objective_kind: str | None = None
    objective_value: Any = None
    target_summary: str | None = None
    started_at: str | None = None


@dataclass(frozen=True)
class PhaseProgress:
    """One phase of the pipeline, with cumulative spend against its allotment.

    ``elapsed_s`` sums **every** entry of the phase across macro cycles, not
    just the current one — phases are re-entered on each cycle, and a per-entry
    reading under-reports by a factor of the cycle count.
    """

    name: str
    index: int
    is_current: bool
    has_run: bool
    elapsed_s: float
    budget_total_s: float | None = None
    budget_remaining_s: float | None = None
    cap_s: float | None = None

    @property
    def pct_used(self) -> float | None:
        """Fraction of this phase's budget consumed, or ``None`` when unbudgeted.

        Returns:
            Ratio of elapsed to allotted time. May exceed ``1.0`` — an overrun
            is reported, not clamped.
        """
        if self.budget_total_s is None or self.budget_total_s <= 0:
            return None
        return self.elapsed_s / self.budget_total_s


@dataclass(frozen=True)
class LaneOccupancy:
    """Concurrency-lane occupancy: how many holders against the configured cap."""

    lane: str
    held: int
    capacity: int
    holders: tuple[str, ...] = ()


@dataclass(frozen=True)
class GpuLease:
    """One GPU lease held by a specialist sub-agent."""

    gpu_id: int
    holder_id: str
    task_id: str
    expires_at: str | None = None
    expired: bool = False


@dataclass(frozen=True)
class TaskCounts:
    """Delegated-task tallies by state."""

    queued: int = 0
    running: int = 0
    succeeded: int = 0
    failed: int = 0
    cancelled: int = 0

    @property
    def total(self) -> int:
        """Total tasks recorded across every state."""
        return self.queued + self.running + self.succeeded + self.failed + self.cancelled


@dataclass(frozen=True)
class RunningTask:
    """A task currently in flight."""

    task_id: str
    kind: str
    state: str
    updated_at: str | None = None


@dataclass(frozen=True)
class ResultSummary:
    """Optimization outcome so far.

    ``cumulative_gain_pct`` is the provisional per-round sum; the validated
    figure is re-measured on a fresh server and is the one to trust. Both are
    carried because the gap between them is itself diagnostic.
    """

    baseline_tput: float | None = None
    best_tput: float | None = None
    best_action: str | None = None
    cumulative_gain_pct: float | None = None
    cumulative_gain_validated_pct: float | None = None
    target_gap_pct: float | None = None
    stop_reason: str | None = None
    crash_count: int = 0


@dataclass(frozen=True)
class LifecycleEvent:
    """One operator-facing phase/step boundary."""

    seq: int | None
    ts: str | None
    phase: str | None
    step: str | None
    label: str | None
    status: str | None
    detail: str | None = None
    duration_s: float | None = None


@dataclass(frozen=True)
class CurrentStep:
    """The long-running step a phase is currently blocked on.

    Written by the producer as ``runtime/current_step.json`` immediately before
    a blocking dispatch and removed afterwards. It exists because the phase
    machine stops ticking for the duration of such a dispatch — a GEAK e2e can
    hold the loop for over eight hours — so ``state.json`` and the coordinator
    event log both go silent while the run is perfectly healthy.
    """

    phase: str
    step: str
    detail: str | None = None
    started_unix: float | None = None
    deadline_unix: float | None = None
    artifacts: tuple[tuple[str, str], ...] = ()

    def elapsed_s(self, now_unix: float) -> float | None:
        """Seconds since this step began, or ``None`` when unknown."""
        if not self.started_unix:
            return None
        return max(0.0, float(now_unix) - float(self.started_unix))

    def budget_s(self) -> float | None:
        """Total seconds allowed for this step, or ``None`` when uncapped."""
        if not self.started_unix or not self.deadline_unix:
            return None
        return max(0.0, float(self.deadline_unix) - float(self.started_unix))


@dataclass(frozen=True)
class GpuMetric:
    """One GPU's live utilization.

    Every field is ``None`` when the tool reported it as unsupported. AMD's
    ``amd-smi`` emits the literal string ``"N/A"`` for those, which must not be
    coerced to zero — an idle GPU and an unreportable one look identical once
    that happens.
    """

    index: int
    util_pct: float | None = None
    mem_activity_pct: float | None = None
    mem_used_mb: float | None = None
    mem_total_mb: float | None = None
    power_w: float | None = None

    @property
    def mem_used_fraction(self) -> float | None:
        """VRAM in use as a ``0..1`` ratio, or ``None`` when either side is unknown."""
        if self.mem_used_mb is None or not self.mem_total_mb:
            return None
        return self.mem_used_mb / self.mem_total_mb


@dataclass(frozen=True)
class GpuMetrics:
    """A sample of every visible GPU.

    ``host_global`` is not decoration. These numbers come from the node's
    device driver, so on a shared box they include every tenant's work — they
    answer "is this machine busy", not "is this session busy". Renderers must
    say so rather than implying attribution the data cannot support.
    """

    gpus: tuple[GpuMetric, ...] = ()
    tool: str | None = None
    host_global: bool = True

    @property
    def busy(self) -> tuple[GpuMetric, ...]:
        """GPUs reporting non-trivial utilization or memory residency."""
        return tuple(
            gpu
            for gpu in self.gpus
            if (gpu.util_pct is not None and gpu.util_pct >= 1.0)
            or (gpu.mem_used_fraction is not None and gpu.mem_used_fraction >= 0.05)
        )


@dataclass(frozen=True)
class ServerMetrics:
    """Live counters scraped from an inference server's ``/metrics`` endpoint.

    ``tput_tok_s`` is derived from the delta between two polls, so it is
    ``None`` on the first sample. That is correct rather than inconvenient: a
    rate needs two observations, and reporting ``0.0`` for "not yet known"
    would be indistinguishable from a stalled server.
    """

    url: str
    requests_running: int | None = None
    requests_waiting: int | None = None
    kv_cache_pct: float | None = None
    prompt_tokens_total: float | None = None
    generation_tokens_total: float | None = None
    tput_tok_s: float | None = None


@dataclass(frozen=True)
class RunningWork:
    """A sub-agent run directory that is showing signs of life.

    Keyed on the **run directory**, deliberately not on a coordinator task id.
    A real session was observed refreshing a heartbeat inside a run directory
    whose task had recorded a clean exit seven hours earlier; keying on task id
    would have attributed live work to a finished task. ``task_terminal`` flags
    that mismatch instead of hiding it.
    """

    kind: str
    run_id: str
    status: str | None = None
    note: str | None = None
    turn: int | None = None
    max_turns: int | None = None
    heartbeat_age_s: float | None = None
    log_bytes: int | None = None
    log_growth_bps: float | None = None
    log_age_s: float | None = None
    has_partial_result: bool = False
    task_terminal: bool = False

    @property
    def age_s(self) -> float | None:
        """Age of the freshest liveness signal, mirroring the reaper's rule.

        The reap loop treats either ``heartbeat.json`` or ``process.log`` as
        proof of life; this reports the same signal so the two never disagree.

        Returns:
            Seconds since the most recent write, or ``None`` when neither file
            is readable.
        """
        ages = [age for age in (self.heartbeat_age_s, self.log_age_s) if age is not None]
        return min(ages) if ages else None


@dataclass(frozen=True)
class ActivityEntry:
    """One recently-written file inside the session tree.

    The path is the payload. ``geak/…/round_1/engineer_0/verify/driver.log``
    explains what a run is doing more directly than any status field the
    producer could have thought to emit.
    """

    relpath: str
    age_s: float
    size_bytes: int | None = None


@dataclass(frozen=True)
class GeakProgress:
    """Structured view of a GEAK end-to-end kernel run.

    Backend-specific by construction. It degrades to nothing when the layout
    changes, leaving the generic activity walk as the fallback, so a GEAK
    rename can cost detail but never correctness.
    """

    cycle: int | None = None
    task: str | None = None
    round_no: int | None = None
    engineers: tuple[str, ...] = ()
    active_engineer: str | None = None
    active_stage: str | None = None
    has_result: bool = False


@dataclass(frozen=True)
class SourceHealth:
    """Per-source collection health, carried so the UI can be honest about gaps.

    A status view that silently shows a five-minute-old GPU reading as current
    is the same failure mode as showing a stale ``state.json`` as live. Every
    cached value therefore travels with the provenance needed to caveat it.
    """

    name: str
    outcome: SourceOutcome = SourceOutcome.ABSENT
    age_s: float | None = None
    error: str | None = None
    consecutive_failures: int = 0
    duration_s: float | None = None

    @property
    def degraded(self) -> bool:
        """Whether this source is failing and therefore worth surfacing."""
        return self.outcome is SourceOutcome.ERROR or self.consecutive_failures > 0


@dataclass(frozen=True)
class Snapshot:
    """One read-only observation of a session.

    Field ordering follows the duck-typing contract described in the module
    docstring: the ``machine_state``-facing attributes come first and must keep
    their names.
    """

    # --- structural contract with machine_state (names are load-bearing) ---
    phase: str = ""
    phase_started_unix: float = 0.0
    phase_elapsed_totals: dict[str, float] = field(default_factory=dict)
    phase_budget_pct: dict[str, float] = field(default_factory=dict)
    max_minutes: int = 0
    cycle_minutes: float = 0.0
    start_ts: str = ""
    explore_elapsed_accum_s: float | None = None

    # --- identity ---
    session: SessionInfo = field(default_factory=lambda: SessionInfo(session_dir=""))
    macro_cycle: int = 0
    tick: int = 0

    # --- observation honesty ---
    liveness: Liveness = Liveness.UNKNOWN
    freshness: Freshness = Freshness.UNKNOWN
    observed_at_unix: float = 0.0
    # Paint time, set by ``progress.extrapolate_to``. Zero means the snapshot
    # carries raw collected values and no clock advance has been applied.
    rendered_at_unix: float = 0.0
    state_age_s: float | None = None
    owner_pid: int | None = None

    # --- derived / joined ---
    phases: tuple[PhaseProgress, ...] = ()
    session_remaining_s: float | None = None
    session_elapsed_s: float | None = None
    lanes: tuple[LaneOccupancy, ...] = ()
    gpu_leases: tuple[GpuLease, ...] = ()
    tasks: TaskCounts = field(default_factory=TaskCounts)
    running_tasks: tuple[RunningTask, ...] = ()
    result: ResultSummary = field(default_factory=ResultSummary)
    lifecycle: tuple[LifecycleEvent, ...] = ()
    current_action: str | None = None

    # --- sub-phase visibility: what a blocked phase is actually doing ---
    current_step: CurrentStep | None = None
    running_work: tuple[RunningWork, ...] = ()
    activity: tuple[ActivityEntry, ...] = ()
    geak: GeakProgress | None = None
    last_activity_age_s: float | None = None

    # --- live metrics ---
    gpus: GpuMetrics | None = None
    server: ServerMetrics | None = None

    # --- collection provenance ---
    source_health: tuple[SourceHealth, ...] = ()
    warnings: tuple[str, ...] = ()

    # Injected clock. Read by ``machine_state._now_unix`` via ``hasattr``, which
    # makes every derived value deterministic under test without patching
    # ``time``. Note the caveat in the module docstring: some machine_state
    # helpers ignore this hook, so ``progress`` also threads ``now_unix``
    # explicitly.
    _now_unix: Callable[[], float] = time.time

    def now_unix(self) -> float:
        """Return the snapshot's notion of the current time.

        Returns:
            Seconds since the epoch, from the injected clock.
        """
        return float(self._now_unix())

    @property
    def is_terminal(self) -> bool:
        """Whether the session has recorded a stop reason."""
        return bool(self.result.stop_reason)

    @property
    def current_phase_progress(self) -> PhaseProgress | None:
        """The :class:`PhaseProgress` for the running phase, if any."""
        for row in self.phases:
            if row.is_current:
                return row
        return None

    @property
    def degraded_sources(self) -> tuple[SourceHealth, ...]:
        """Sources currently failing to collect, worth showing to the operator."""
        return tuple(row for row in self.source_health if row.degraded)
