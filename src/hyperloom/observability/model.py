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
