# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Phase-progress derivation — the ONLY module here that imports the orchestrator.

Every other module in :mod:`hyperloom.observability` is orchestrator-free, so
the model and renderers stay cheap to import and easy to test. All coupling to
the phase machine is concentrated in this file, which makes the blast radius of
an upstream change one module wide.

Why delegate instead of reimplement
-----------------------------------
Hyperloom's per-phase budget math is subtle in ways that are invisible from the
outside:

* **Charge-back** — a phase is not given ``max_minutes * pct``. It gets its
  share of the time *still remaining*, renormalized over the current phase and
  the phases yet to come, so an overrunning PRELUDE silently shrinks every
  later phase rather than blowing the session deadline.
* **Long-run cycle windows** — past a 24h threshold the per-cycle window caps
  the charge-back base, so one macro cycle never plans beyond its own window.
* **Cumulative charging** — phases repeat across macro cycles, and each
  re-entry resumes from what the phase has already spent.

A second implementation of that would drift from the real scheduler and display
numbers that disagree with the decisions the optimizer is actually making —
precisely the failure this layer exists to prevent. So we call the production
functions, which are already duck-typed over "a frozen SharedState view".

``now_unix`` is threaded explicitly into every call that accepts it.
``session_remaining_seconds`` ignores the ``_now_unix`` hook and reads the wall
clock otherwise, which would make derived values non-deterministic under test
and inconsistent between fields computed microseconds apart.
"""

from __future__ import annotations

import dataclasses
from typing import Any

from .model import Liveness, PhaseProgress, Snapshot


def phase_names() -> tuple[str, ...]:
    """Return the canonical phase order.

    Returns:
        Phase names in pipeline order, e.g. ``("PRELUDE", ..., "CLOSE")``.
    """
    from hyperloom.orchestrator.phases.machine_state import PHASE_NAMES

    return tuple(PHASE_NAMES)


def stop_reason_vocab() -> frozenset[str]:
    """Return the closed vocabulary of session stop reasons.

    Returns:
        Every valid ``stop_reason`` value.
    """
    from hyperloom.orchestrator.phases.machine_state import STOP_REASON_VOCAB

    return frozenset(STOP_REASON_VOCAB)


def elapsed_totals_from_history(history: Any) -> dict[str, float]:
    """Reconstruct per-phase banked totals from ``phase_history``.

    ``phase_elapsed_totals`` was added in a later state schema, so sessions
    written before it — including any run being inspected after an upgrade —
    have no banked totals and would otherwise report every completed phase as
    zero elapsed.

    ``phase_history`` is capped at 100 rows, so on a very long run this returns
    a **lower bound**. That is the safe direction: it can under-report a phase,
    but it can never invent time the phase did not spend.

    Args:
        history: The raw ``phase_history`` list.

    Returns:
        Phase name to banked seconds; ``{}`` when unreconstructable.
    """
    from hyperloom.orchestrator.phases.machine_state import phase_elapsed_totals_from_history

    return dict(phase_elapsed_totals_from_history(history))


def normalized_budget(snapshot_or_state: Any) -> dict[str, float]:
    """Return the sanitized per-phase budget fractions.

    Always routed through the upstream ``normalize_budget_pct``: reading
    ``phase_budget_pct`` raw would include entries the scheduler itself drops
    (unknown phase names, values outside ``(0.0, 1.0]``), so a display built on
    the raw dict can show a budget the optimizer is not honoring.

    Args:
        snapshot_or_state: Any object exposing ``phase_budget_pct``.

    Returns:
        Phase name to fraction, defaults overlaid with valid overrides.
    """
    from hyperloom.orchestrator.phases.machine_state import normalize_budget_pct

    return normalize_budget_pct(getattr(snapshot_or_state, "phase_budget_pct", None))


def build_phase_progress(snapshot: Snapshot, *, now_unix: float) -> tuple[PhaseProgress, ...]:
    """Derive per-phase progress rows for every phase in the pipeline.

    Args:
        snapshot: A snapshot populated with the ``machine_state`` contract
            fields (phase, ``phase_elapsed_totals``, budget, clock anchors).
        now_unix: Current time, threaded into every upstream call.

    Returns:
        One :class:`~.model.PhaseProgress` per phase, in pipeline order.
    """
    from hyperloom.orchestrator.phases.machine_state import (
        PHASE_NAMES,
        _phase_budget_total_seconds,
        phase_budget_remaining_seconds,
        phase_cap_seconds,
        phase_cumulative_seconds,
    )

    budget = normalized_budget(snapshot)
    current = (snapshot.phase or "").strip().upper()
    totals = snapshot.phase_elapsed_totals if isinstance(snapshot.phase_elapsed_totals, dict) else {}

    rows: list[PhaseProgress] = []
    for index, name in enumerate(PHASE_NAMES):
        is_current = name == current
        elapsed = phase_cumulative_seconds(snapshot, phase=name, now_unix=now_unix)

        # The budget helpers describe *the current phase* only — they read
        # ``state.phase`` internally. For any other phase we can still report
        # the absolute cap, which is phase-addressable and time-independent.
        if is_current:
            total_s = _phase_budget_total_seconds(snapshot, budget_pct=budget, now_unix=now_unix)
            remaining_s = phase_budget_remaining_seconds(snapshot, budget_pct=budget, now_unix=now_unix)
        else:
            total_s = None
            remaining_s = None

        rows.append(
            PhaseProgress(
                name=name,
                index=index,
                is_current=is_current,
                # A phase has run if it banked time, or is running now. Note
                # ``phase_elapsed_totals`` is the durable accumulator, not the
                # capped ``phase_history``, so this stays correct on long runs
                # whose early transitions have aged out of the history window.
                has_run=is_current or float(totals.get(name, 0.0) or 0.0) > 0.0,
                elapsed_s=elapsed,
                budget_total_s=total_s,
                budget_remaining_s=remaining_s,
                cap_s=_safe_cap(phase_cap_seconds, snapshot, name, budget),
            )
        )
    return tuple(rows)


def _safe_cap(cap_fn: Any, snapshot: Snapshot, phase: str, budget: dict[str, float]) -> float | None:
    """Return the absolute cap for ``phase``, or ``None``.

    ``phase_cap_seconds`` reads ``state.phase`` rather than taking a phase
    argument, so a lightweight stand-in carrying the target phase is passed
    instead of mutating the frozen snapshot.

    Args:
        cap_fn: The upstream ``phase_cap_seconds`` callable.
        snapshot: Source of ``max_minutes`` / ``cycle_minutes``.
        phase: Phase to compute the cap for.
        budget: Normalized budget fractions.

    Returns:
        Cap in seconds, or ``None`` when the phase has no finite cap.
    """
    view = _PhaseView(snapshot, phase)
    try:
        return cap_fn(view, budget_pct=budget)
    except (TypeError, ValueError, AttributeError):
        return None


class _PhaseView:
    """Minimal frozen-state stand-in that reports a chosen ``phase``.

    Delegates every other attribute to the snapshot, so upstream helpers see a
    consistent view without the snapshot being copied or mutated.
    """

    __slots__ = ("_snapshot", "phase")

    def __init__(self, snapshot: Snapshot, phase: str) -> None:
        """Bind the snapshot and the phase to report.

        Args:
            snapshot: Backing snapshot.
            phase: Phase name this view reports as current.
        """
        self._snapshot = snapshot
        self.phase = phase

    def __getattr__(self, item: str) -> Any:
        """Delegate unknown attributes to the backing snapshot."""
        return getattr(self._snapshot, item)


def _shift_age(age: float | None, delta: float) -> float | None:
    """Advance an age-since-observation by ``delta``, preserving ``None``."""
    return None if age is None else max(0.0, age + delta)


def extrapolate_to(snapshot: Snapshot, now_unix: float) -> Snapshot:
    """Advance a collected snapshot's clocks to paint time, without any I/O.

    This is what decouples the refresh rate from the collection rate. Data is
    gathered every few seconds, but the operator watches a timer: it must tick
    every frame, or the display reads as frozen even when everything is fine.

    Pure arithmetic — no orchestrator import, no filesystem access — so it is
    cheap enough to call on every repaint.

    Two classes of field move differently:

    * **Ages** (``state_age_s``, heartbeat and activity ages, per-source age)
      always advance. An age is an age; a finished run's "last update 4h ago"
      should keep counting up while you watch it.
    * **Progress** (session and current-phase elapsed, remaining budget) only
      advances while the run can still be making progress — ``LIVE`` or
      ``STALE``, and no recorded stop reason. Extrapolating a dead run's
      elapsed time would reintroduce exactly the runaway-duration bug that
      :func:`~hyperloom.observability.assemble._progress_clock` exists to
      prevent, where a finished 8-hour run inspected a week later reported its
      last phase as lasting a week.

    Args:
        snapshot: A freshly collected snapshot.
        now_unix: Paint-time wall clock.

    Returns:
        A new snapshot with clocks advanced, or ``snapshot`` unchanged when
        there is nothing to advance. Applying this to an already-extrapolated
        snapshot is a no-op rather than a compounding error — callers should
        always extrapolate the pristine collected copy.
    """
    if snapshot.rendered_at_unix:
        return snapshot
    if not snapshot.observed_at_unix:
        return snapshot
    delta = float(now_unix) - float(snapshot.observed_at_unix)
    if delta <= 0:
        return snapshot

    changes: dict[str, Any] = {
        "rendered_at_unix": float(now_unix),
        "state_age_s": _shift_age(snapshot.state_age_s, delta),
        "last_activity_age_s": _shift_age(snapshot.last_activity_age_s, delta),
    }

    if snapshot.running_work:
        changes["running_work"] = tuple(
            dataclasses.replace(
                work,
                heartbeat_age_s=_shift_age(work.heartbeat_age_s, delta),
                log_age_s=_shift_age(work.log_age_s, delta),
            )
            for work in snapshot.running_work
        )
    if snapshot.activity:
        changes["activity"] = tuple(
            dataclasses.replace(entry, age_s=max(0.0, entry.age_s + delta)) for entry in snapshot.activity
        )
    if snapshot.source_health:
        changes["source_health"] = tuple(
            dataclasses.replace(health, age_s=_shift_age(health.age_s, delta)) for health in snapshot.source_health
        )

    progressing = snapshot.liveness in (Liveness.LIVE, Liveness.STALE) and not snapshot.is_terminal
    if progressing:
        cap_s = float(snapshot.max_minutes) * 60.0 if snapshot.max_minutes else None
        if snapshot.session_elapsed_s is not None:
            advanced = snapshot.session_elapsed_s + delta
            # Never extrapolate past the session's own deadline. Past the cap
            # the run is being wound down, not accruing more budgeted time, and
            # a timer reading 25:00/24:00 would look like a display bug.
            changes["session_elapsed_s"] = min(advanced, cap_s) if cap_s else advanced
        if snapshot.session_remaining_s is not None:
            changes["session_remaining_s"] = max(0.0, snapshot.session_remaining_s - delta)
        if snapshot.phases:
            changes["phases"] = tuple(
                dataclasses.replace(
                    row,
                    elapsed_s=row.elapsed_s + delta,
                    budget_remaining_s=(
                        None if row.budget_remaining_s is None else max(0.0, row.budget_remaining_s - delta)
                    ),
                )
                # Only the running phase accrues. A completed phase's banked
                # total is a fact, not a running clock.
                if row.is_current
                else row
                for row in snapshot.phases
            )

    return dataclasses.replace(snapshot, **changes)


def session_timing(snapshot: Snapshot, *, now_unix: float) -> tuple[float | None, float | None]:
    """Return ``(elapsed_s, remaining_s)`` for the whole session.

    Args:
        snapshot: Snapshot exposing ``start_ts`` and ``max_minutes``.
        now_unix: Current time.

    Returns:
        Elapsed seconds since ``start_ts`` (``None`` when unparseable) and
        seconds left against ``max_minutes`` (``None`` for an unbounded run).
    """
    from datetime import datetime, timezone

    from hyperloom.orchestrator.phases.machine_state import session_remaining_seconds

    remaining = session_remaining_seconds(snapshot, now_unix=now_unix)

    elapsed: float | None = None
    start_ts = str(snapshot.start_ts or "").strip()
    if start_ts:
        try:
            start = datetime.fromisoformat(start_ts)
            if start.tzinfo is None:
                start = start.replace(tzinfo=timezone.utc)
            elapsed = max(0.0, now_unix - start.timestamp())
        except (TypeError, ValueError):
            elapsed = None

    return elapsed, remaining
