# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Data-layer contract for :func:`~hyperloom.observability.load_snapshot`.

No rendering: these drive real session directories on disk and assert on the
assembled :class:`~hyperloom.observability.model.Snapshot`.

Coverage matrix
---------------
==========================================  ==================================================
Scenario                                    Contract
==========================================  ==================================================
No ``coordinator.db``                       ABSENT, not a warning; snapshot still returned
Unreadable ``state.json``                   ERROR warning; snapshot still returned
No lock file                                liveness UNKNOWN, never LIVE
Lock pid dead on this host                  liveness DEAD
Lock pid "alive" but session terminal       liveness DEAD (container pid-namespace trap)
Live pid + fresh state                      liveness LIVE
Live pid + stale state                      liveness STALE
Terminal session, wall clock advances       durations pinned to last recorded activity
``state.json`` mtime later than run         durations use in-data timestamps, not mtime
Legacy state, no phase_elapsed_totals       totals reconstructed from phase_history
Ray synthetic GPU slot ids                  filtered out of gpu_leases
Expired lane lease                          not counted as holding the lane
``explore_elapsed_accum_s`` absent          stays None (tri-state), never 0.0
==========================================  ==================================================
"""

from __future__ import annotations

import socket
import subprocess
from pathlib import Path

from hyperloom.observability import Freshness, Liveness, load_snapshot

from .conftest import (
    FROZEN_NOW,
    SESSION_START_UNIX,
    write_coordinator_db,
    write_lock,
    write_manifest,
    write_state,
)


def _dead_pid() -> int:
    """Return a pid that has certainly exited."""
    proc = subprocess.Popen(["true"])  # noqa: S607 — fixed argv, no shell
    proc.wait()
    return proc.pid


# --- sources: absent vs error -------------------------------------------------


def test_missing_coordinator_db_is_absent_not_error(tmp_path: Path, frozen_clock) -> None:
    """An early-run session has no DB yet; that must not read as a failure."""
    sd = tmp_path / "bare"
    write_state(sd)

    snapshot = load_snapshot(sd, now_unix=frozen_clock)

    assert snapshot is not None
    assert snapshot.warnings == ()
    assert snapshot.tasks.total == 0


def test_unreadable_state_reports_warning_and_still_returns(tmp_path: Path, frozen_clock) -> None:
    """A corrupt ``state.json`` degrades to a warning, not an exception."""
    sd = tmp_path / "corrupt"
    sd.mkdir(parents=True)
    (sd / "state.json").write_text("{not json", encoding="utf-8")

    snapshot = load_snapshot(sd, now_unix=frozen_clock)

    assert snapshot is not None
    assert any(w.startswith("state:") for w in snapshot.warnings)


# --- liveness -----------------------------------------------------------------


def test_no_lock_is_unknown_never_live(tmp_path: Path, frozen_clock) -> None:
    """Absent liveness evidence must read UNKNOWN, not LIVE."""
    sd = tmp_path / "nolock"
    write_state(sd)

    snapshot = load_snapshot(sd, now_unix=frozen_clock)

    assert snapshot is not None
    assert snapshot.liveness is Liveness.UNKNOWN


def test_dead_pid_on_this_host_is_dead(tmp_path: Path, frozen_clock) -> None:
    """A lock naming an exited local pid is positive evidence of death."""
    sd = tmp_path / "deadpid"
    write_state(sd)
    write_lock(sd, pid=_dead_pid(), hostname=socket.gethostname())

    snapshot = load_snapshot(sd, now_unix=frozen_clock)

    assert snapshot is not None
    assert snapshot.liveness is Liveness.DEAD


def test_terminal_session_is_dead_even_when_pid_looks_alive(tmp_path: Path, frozen_clock) -> None:
    """A recorded stop reason outranks any pid check.

    Regression guard for a real session: a containerized run recorded
    ``pid: 19`` alongside the *host's* hostname, because the container shares
    it while keeping its own PID namespace. Host pid 19 is an unrelated live
    kernel thread, so the pid probe returned a false positive and a run that
    had finished ten days earlier was reported as still going.
    """
    sd = tmp_path / "terminal"
    write_state(sd, stop_reason="conc_sweep_done", phase="CLOSE")
    # Pid 1 always exists and is never the optimizer.
    write_lock(sd, pid=1, hostname=socket.gethostname())

    snapshot = load_snapshot(sd, now_unix=frozen_clock)

    assert snapshot is not None
    assert snapshot.liveness is Liveness.DEAD
    assert snapshot.is_terminal


def test_live_pid_with_fresh_state_is_live(tmp_path: Path, frozen_clock) -> None:
    """A live local pid corroborated by a fresh state file reads LIVE."""
    sd = tmp_path / "live"
    write_state(sd)
    write_lock(sd, pid=1, hostname=socket.gethostname())

    # Clock at the state file's own mtime so the corroboration is fresh.
    mtime = (sd / "state.json").stat().st_mtime
    snapshot = load_snapshot(sd, now_unix=lambda: mtime)

    assert snapshot is not None
    assert snapshot.liveness is Liveness.LIVE
    assert snapshot.freshness is Freshness.FRESH


def test_live_pid_with_stale_state_is_stale(tmp_path: Path, frozen_clock) -> None:
    """An owner that is up but not publishing progress reads STALE."""
    sd = tmp_path / "wedged"
    write_state(sd)
    write_lock(sd, pid=1, hostname=socket.gethostname())

    mtime = (sd / "state.json").stat().st_mtime
    snapshot = load_snapshot(sd, now_unix=lambda: mtime + 10_000.0)

    assert snapshot is not None
    assert snapshot.liveness is Liveness.STALE
    assert snapshot.freshness is Freshness.STALE


# --- duration math ------------------------------------------------------------


def test_terminal_durations_do_not_grow_with_wall_clock(tmp_path: Path) -> None:
    """A finished run's phase durations must be stable however late we look.

    Without pinning, ``now - phase_started_unix`` keeps accruing forever, so an
    8-hour run inspected a week later reports its last phase as lasting a week
    and every budget percentage becomes meaningless.
    """
    sd = tmp_path / "ended"
    write_state(sd, stop_reason="target_reached", phase="CLOSE", phase_started_unix=SESSION_START_UNIX + 7200.0)

    soon = load_snapshot(sd, now_unix=lambda: FROZEN_NOW)
    much_later = load_snapshot(sd, now_unix=lambda: FROZEN_NOW + 86_400.0 * 7)

    assert soon is not None and much_later is not None
    assert soon.session_elapsed_s == much_later.session_elapsed_s
    assert [p.elapsed_s for p in soon.phases] == [p.elapsed_s for p in much_later.phases]


def test_durations_prefer_in_data_timestamps_over_file_mtime(tmp_path: Path) -> None:
    """Copying a session must not change its reported durations.

    ``mtime`` is a property of the file, not the run: archiving or syncing a
    session directory rewrites it. A real session had an mtime four days past
    its last recorded event for exactly this reason.
    """
    sd = tmp_path / "copied"
    write_state(sd, stop_reason="target_reached", phase="CLOSE")

    before = load_snapshot(sd, now_unix=lambda: FROZEN_NOW + 86_400.0)

    # Simulate an archive/restore touching the file long after the run ended.
    import os

    future = FROZEN_NOW + 86_400.0 * 4
    os.utime(sd / "state.json", (future, future))

    after = load_snapshot(sd, now_unix=lambda: FROZEN_NOW + 86_400.0 * 5)

    assert before is not None and after is not None
    assert before.session_elapsed_s == after.session_elapsed_s


def test_legacy_state_reconstructs_totals_from_history(tmp_path: Path, frozen_clock) -> None:
    """Sessions predating ``phase_elapsed_totals`` still report phase time.

    Reconstruction is a documented lower bound (``phase_history`` is capped at
    100 rows), which is the safe direction: it can under-report a phase but
    never invent time it did not spend.
    """
    sd = tmp_path / "legacy"
    # History says PRELUDE lasted 1800s, deliberately different from the
    # fixture's default banked total of 3600s — so a reconstructed value is
    # distinguishable from the default and the test cannot pass by accident.
    write_state(
        sd,
        phase_elapsed_totals={},  # the pre-v5 shape
        phase_history=[
            {
                "from_phase": "",
                "to_phase": "PRELUDE",
                "reason": "phase_entered",
                "evidence": {},
                "ts_unix": SESSION_START_UNIX,
                "cycle": 0,
            },
            {
                "from_phase": "PRELUDE",
                "to_phase": "EXPLORE",
                "reason": "prelude_done",
                "evidence": {},
                "ts_unix": SESSION_START_UNIX + 1800.0,
                "cycle": 0,
            },
        ],
    )

    snapshot = load_snapshot(sd, now_unix=frozen_clock)

    assert snapshot is not None
    prelude = next(p for p in snapshot.phases if p.name == "PRELUDE")
    assert prelude.elapsed_s == 1800.0
    assert prelude.has_run


def test_banked_totals_win_over_history_when_present(tmp_path: Path, frozen_clock) -> None:
    """The durable accumulator is authoritative; history is only the fallback."""
    sd = tmp_path / "banked"
    write_state(sd, phase_elapsed_totals={"PRELUDE": 9999.0})

    snapshot = load_snapshot(sd, now_unix=frozen_clock)

    assert snapshot is not None
    prelude = next(p for p in snapshot.phases if p.name == "PRELUDE")
    assert prelude.elapsed_s == 9999.0


def test_explore_accum_stays_tri_state(tmp_path: Path, frozen_clock) -> None:
    """``None`` means "legacy resume, unknowable" and must not become 0.0."""
    sd = tmp_path / "tristate"
    write_state(sd)

    snapshot = load_snapshot(sd, now_unix=frozen_clock)

    assert snapshot is not None
    assert snapshot.explore_elapsed_accum_s is None


# --- coordinator.db projections ----------------------------------------------


def test_ray_synthetic_gpu_slots_are_filtered(tmp_path: Path, frozen_clock) -> None:
    """Ray admission-ledger slot ids are accounting rows, not cards."""
    sd = tmp_path / "ray"
    write_state(sd)
    write_coordinator_db(
        sd,
        gpu_leases=[
            (0, "spec-real", "2099-01-01T00:00:00+00:00"),
            (100_001, "spec-ray-slot", "2099-01-01T00:00:00+00:00"),
        ],
    )

    snapshot = load_snapshot(sd, now_unix=frozen_clock)

    assert snapshot is not None
    assert [lease.gpu_id for lease in snapshot.gpu_leases] == [0]


def test_expired_lease_does_not_hold_a_lane(tmp_path: Path, frozen_clock) -> None:
    """An unreaped expired lease must not over-report lane pressure."""
    sd = tmp_path / "expired"
    write_state(sd)
    write_coordinator_db(
        sd,
        leases=[
            ("benchmark_lane", "stale-holder", "2000-01-01T00:00:00+00:00"),
            ("build_lane", "live-holder", "2099-01-01T00:00:00+00:00"),
        ],
        lane_capacity={"benchmark_lane": 1, "build_lane": 1},
    )

    snapshot = load_snapshot(sd, now_unix=frozen_clock)

    assert snapshot is not None
    lanes = {lane.lane: lane for lane in snapshot.lanes}
    assert lanes["benchmark_lane"].held == 0
    assert lanes["build_lane"].held == 1
    assert lanes["build_lane"].holders == ("live-holder",)


def test_task_counts_and_running_tasks(session_dir: Path, frozen_clock) -> None:
    """Task tallies and the in-flight list come straight from ``tasks``."""
    snapshot = load_snapshot(session_dir, now_unix=frozen_clock)

    assert snapshot is not None
    assert snapshot.tasks.succeeded == 1
    assert snapshot.tasks.running == 1
    assert snapshot.tasks.queued == 1
    assert snapshot.tasks.failed == 1
    assert snapshot.tasks.total == 4
    assert [task.kind for task in snapshot.running_tasks] == ["explore"]


# --- identity join ------------------------------------------------------------


def test_manifest_wins_over_state_for_identity(tmp_path: Path, frozen_clock) -> None:
    """The manifest records intent and is never rewritten, so it wins."""
    sd = tmp_path / "identity"
    write_state(sd, model_name="drifted-name", framework="vllm")
    write_manifest(sd, model_name="manifest-name", framework="sglang")

    snapshot = load_snapshot(sd, now_unix=frozen_clock)

    assert snapshot is not None
    assert snapshot.session.model_name == "manifest-name"
    assert snapshot.session.framework == "sglang"
    assert snapshot.session.objective_kind == "gain_pct"
    assert snapshot.session.conc == 64


def test_state_fills_gaps_when_manifest_absent(tmp_path: Path, frozen_clock) -> None:
    """With no manifest, identity falls back to ``state.json``."""
    sd = tmp_path / "nomanifest"
    write_state(sd)

    snapshot = load_snapshot(sd, now_unix=frozen_clock)

    assert snapshot is not None
    assert snapshot.session.model_name == "test-model"
    assert snapshot.session.gpu_type == "mi300x"
    assert snapshot.session.tp == 8


def test_unresolvable_session_returns_none(tmp_path: Path, frozen_clock) -> None:
    """An explicit path that is not a directory resolves to nothing."""
    assert load_snapshot(tmp_path / "does-not-exist", now_unix=frozen_clock) is None
