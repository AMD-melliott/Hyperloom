# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Paint-time clock advance — the decoupling of refresh rate from collection.

The display repaints every second while data is gathered every few; without
advancing the clock arithmetically between collections the timers would visibly
stutter, and a slow probe would make the whole view look frozen.

The rules under test:

============================================= =====================================
Case                                          Expectation
============================================= =====================================
LIVE session, one second later                Session and current-phase elapsed grow
DEAD session                                  Progress pinned; ages still grow
Session with a recorded stop reason           Progress pinned
Completed (non-current) phase                 Banked total untouched
Elapsed near the session cap                  Clamped at the cap
Already-extrapolated snapshot                 No-op, never compounding
============================================= =====================================
"""

from __future__ import annotations

from hyperloom.observability.model import (
    ActivityEntry,
    Liveness,
    PhaseProgress,
    ResultSummary,
    RunningWork,
    Snapshot,
    SourceHealth,
)
from hyperloom.observability.progress import extrapolate_to


def _snapshot(**overrides) -> Snapshot:
    """Build a minimal snapshot positioned mid-run."""
    base = {
        "phase": "KERNEL_AGENT",
        "max_minutes": 60,
        "liveness": Liveness.LIVE,
        "observed_at_unix": 1000.0,
        "state_age_s": 30.0,
        "last_activity_age_s": 4.0,
        "session_elapsed_s": 600.0,
        "session_remaining_s": 3000.0,
        "phases": (
            PhaseProgress(name="PRELUDE", index=0, is_current=False, has_run=True, elapsed_s=300.0),
            PhaseProgress(
                name="KERNEL_AGENT",
                index=1,
                is_current=True,
                has_run=True,
                elapsed_s=300.0,
                budget_total_s=1200.0,
                budget_remaining_s=900.0,
            ),
        ),
    }
    base.update(overrides)
    return Snapshot(**base)


def test_live_session_advances_progress_and_ages() -> None:
    """One paint-second later, everything that should move has moved."""
    advanced = extrapolate_to(_snapshot(), 1010.0)

    assert advanced.session_elapsed_s == 610.0
    assert advanced.session_remaining_s == 2990.0
    assert advanced.state_age_s == 40.0
    assert advanced.last_activity_age_s == 14.0
    assert advanced.rendered_at_unix == 1010.0

    current = advanced.current_phase_progress
    assert current is not None
    assert current.elapsed_s == 310.0
    assert current.budget_remaining_s == 890.0
    # The derived percentage follows for free.
    assert current.pct_used is not None and current.pct_used > 0.25


def test_completed_phases_do_not_accrue() -> None:
    """A finished phase's banked total is a fact, not a running clock."""
    advanced = extrapolate_to(_snapshot(), 1600.0)

    prelude = next(row for row in advanced.phases if row.name == "PRELUDE")
    assert prelude.elapsed_s == 300.0


def test_dead_session_pins_progress_but_ages_keep_growing() -> None:
    """The runaway-duration bug must not come back through the paint path.

    A finished eight-hour run inspected a week later once reported its last
    phase as lasting a week. Progress is therefore frozen for a dead run —
    but "last update N ago" is an age, and an age genuinely does keep growing
    while you watch it.
    """
    advanced = extrapolate_to(_snapshot(liveness=Liveness.DEAD), 1000.0 + 86_400.0)

    assert advanced.session_elapsed_s == 600.0
    assert advanced.current_phase_progress.elapsed_s == 300.0
    assert advanced.state_age_s == 30.0 + 86_400.0


def test_recorded_stop_reason_pins_progress_even_when_liveness_says_live() -> None:
    """The session's own statement that it concluded outranks the classifier."""
    snapshot = _snapshot(liveness=Liveness.LIVE, result=ResultSummary(stop_reason="target_reached"))

    advanced = extrapolate_to(snapshot, 1600.0)
    assert advanced.session_elapsed_s == 600.0


def test_stale_session_still_advances() -> None:
    """A quiet loop is still a working run; its timers must keep moving.

    This is the GEAK case: the phase machine has not ticked for hours because
    it is blocked on a dispatch, and the operator needs to watch that number
    grow.
    """
    advanced = extrapolate_to(_snapshot(liveness=Liveness.STALE), 1060.0)
    assert advanced.session_elapsed_s == 660.0


def test_elapsed_is_clamped_at_the_session_cap() -> None:
    """A timer reading 25:00 / 24:00 looks like a display bug, so it is capped."""
    snapshot = _snapshot(session_elapsed_s=3590.0)  # max_minutes=60 -> 3600 s

    advanced = extrapolate_to(snapshot, 1000.0 + 600.0)
    assert advanced.session_elapsed_s == 3600.0
    assert advanced.session_remaining_s == 2400.0


def test_re_extrapolation_is_a_no_op_not_a_compounding_error() -> None:
    """Callers extrapolate the pristine cached copy; this is the safety net."""
    once = extrapolate_to(_snapshot(), 1010.0)
    twice = extrapolate_to(once, 1020.0)

    assert twice is once
    assert twice.session_elapsed_s == 610.0


def test_no_observation_time_means_nothing_to_advance() -> None:
    """An empty snapshot has no anchor, so it is returned untouched."""
    empty = Snapshot()
    assert extrapolate_to(empty, 5000.0) is empty


def test_clock_going_backwards_is_ignored() -> None:
    """NTP stepping the clock back must not rewind the display."""
    snapshot = _snapshot()
    assert extrapolate_to(snapshot, 990.0) is snapshot


def test_nested_ages_advance() -> None:
    """Work and activity ages are ages too, and must not freeze between polls."""
    snapshot = _snapshot(
        running_work=(RunningWork(kind="specialist", run_id="a1", heartbeat_age_s=10.0, log_age_s=20.0),),
        activity=(ActivityEntry(relpath="geak/x.log", age_s=4.0),),
        source_health=(SourceHealth(name="gpu", age_s=2.0),),
    )

    advanced = extrapolate_to(snapshot, 1005.0)

    assert advanced.running_work[0].heartbeat_age_s == 15.0
    assert advanced.running_work[0].log_age_s == 25.0
    assert advanced.running_work[0].age_s == 15.0
    assert advanced.activity[0].age_s == 9.0
    assert advanced.source_health[0].age_s == 7.0


def test_unmeasured_ages_stay_none() -> None:
    """``None`` means not measured, and adding seconds to it would invent data."""
    snapshot = _snapshot(
        state_age_s=None,
        last_activity_age_s=None,
        running_work=(RunningWork(kind="specialist", run_id="a1", heartbeat_age_s=None, log_age_s=None),),
    )

    advanced = extrapolate_to(snapshot, 1010.0)

    assert advanced.state_age_s is None
    assert advanced.last_activity_age_s is None
    assert advanced.running_work[0].heartbeat_age_s is None
