# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Sub-phase activity: run probes, the bounded walk, and liveness from writes.

Coverage matrix:

======================================================= ===================================
Case                                                    Expectation
======================================================= ===================================
Run with a fresh heartbeat carrying ``note``            Note text is preserved
Run whose only signal is ``process.log``                Still reported, growth derived
Run whose signals are all old                           Filtered out of running work
Heartbeat inside a run with ``*_done.json``             ``task_terminal`` flagged
``__pycache__`` / ``.git`` / ``*.pyc``                  Pruned from the walk
Walk budget exhausted                                   ``truncated`` set, warning raised
No walkable subtrees                                    ``ABSENT``, not ``ERROR``
Container pid absent + fresh activity                   LIVE/STALE, never DEAD
Container pid absent + no activity at all               DEAD stays reachable
======================================================= ===================================
"""

from __future__ import annotations

import socket
from pathlib import Path

from hyperloom.observability.assemble import load_snapshot
from hyperloom.observability.model import Liveness, SourceOutcome
from hyperloom.observability.sources.activity import ActivitySource, probe_runs, walk_activity

from .conftest import FROZEN_NOW, write_activity, write_lock, write_run, write_state


def test_heartbeat_note_is_preserved(tmp_path: Path) -> None:
    """The agent-authored ``note`` is the most useful field on the screen.

    The reap loop already reads this file and keeps only its mtime; the whole
    point of this source is that the text has been on disk unread.
    """
    sd = tmp_path / "s"
    write_run(
        sd,
        run_id="a1e10ec5",
        heartbeat={"ts": "2026-08-06T03:59:50Z", "status": "running", "note": "gdn kernel work"},
        heartbeat_age_s=10.0,
    )

    rows, _ = probe_runs(sd, now_unix=FROZEN_NOW)

    assert len(rows) == 1
    assert rows[0].kind == "specialist"
    assert rows[0].run_id == "a1e10ec5"
    assert rows[0].note == "gdn kernel work"
    assert rows[0].status == "running"
    assert rows[0].heartbeat_age_s == 10.0


def test_dispatcher_heartbeat_schema_also_parses(tmp_path: Path) -> None:
    """Two producers write this file with different shapes; both must read.

    The dispatcher emits turn counters, the agent emits a note. Neither is
    authoritative, so fields are taken opportunistically.
    """
    sd = tmp_path / "s"
    write_run(sd, run_id="b2", heartbeat={"status": "finished", "turn": 3, "max_turns": 120})

    rows, _ = probe_runs(sd, now_unix=FROZEN_NOW)
    assert rows[0].turn == 3
    assert rows[0].max_turns == 120
    assert rows[0].note is None


def test_log_growth_is_derived_across_polls(tmp_path: Path) -> None:
    """Byte growth per second is a progress signal when there is no heartbeat."""
    sd = tmp_path / "s"
    write_run(sd, run_id="c3", heartbeat=None, log_bytes=1000, log_age_s=5.0)

    rows, state = probe_runs(sd, now_unix=FROZEN_NOW)
    assert rows[0].log_bytes == 1000
    assert rows[0].log_growth_bps is None, "a rate needs two samples"

    write_run(sd, run_id="c3", heartbeat=None, log_bytes=3000, log_age_s=1.0)
    rows2, _ = probe_runs(sd, now_unix=FROZEN_NOW + 10.0, previous=state)
    assert rows2[0].log_growth_bps == 200.0  # 2000 bytes over 10 s


def test_finished_runs_are_not_reported_as_work(tmp_path: Path) -> None:
    """Run directories persist forever; only recent ones are current work."""
    sd = tmp_path / "s"
    write_run(sd, run_id="old", heartbeat={"status": "finished"}, heartbeat_age_s=7200.0)
    write_run(sd, run_id="new", heartbeat={"status": "running"}, heartbeat_age_s=30.0)

    rows, _ = probe_runs(sd, now_unix=FROZEN_NOW, active_within_s=900.0)

    assert [row.run_id for row in rows] == ["new"]


def test_heartbeat_in_a_completed_run_is_flagged(tmp_path: Path) -> None:
    """Observed in the wild: a live heartbeat inside a done task's directory.

    Keying on task id would attribute live work to a finished task; this source
    keys on the run directory and reports the mismatch instead.
    """
    sd = tmp_path / "s"
    write_run(sd, run_id="a1e10ec5", heartbeat={"status": "running", "note": "still going"}, done=True)

    rows, _ = probe_runs(sd, now_unix=FROZEN_NOW)
    assert rows[0].task_terminal is True


def test_walk_prunes_build_residue(tmp_path: Path) -> None:
    """``.git`` and ``__pycache__`` mtimes track tooling, not progress."""
    sd = tmp_path / "s"
    write_activity(sd, "geak/e2e_cycle0/verify/driver.log", age_s=4.0)
    write_activity(sd, "geak/e2e_cycle0/__pycache__/x.cpython-312.pyc", age_s=1.0)
    write_activity(sd, "geak/e2e_cycle0/workspace/.git/index", age_s=1.0)
    write_activity(sd, "geak/e2e_cycle0/kernel.py.pyc", age_s=1.0)

    entries, truncated = walk_activity(sd, now_unix=FROZEN_NOW, limit=10)

    paths = [entry.relpath for entry in entries]
    assert paths == ["geak/e2e_cycle0/verify/driver.log"]
    assert truncated is False


def test_walk_is_newest_first(tmp_path: Path) -> None:
    """The freshest write is the best answer to "what is it doing"."""
    sd = tmp_path / "s"
    write_activity(sd, "runs/x/old.log", age_s=600.0)
    write_activity(sd, "geak/new.log", age_s=3.0)
    write_activity(sd, "reports/mid.log", age_s=60.0)

    entries, _ = walk_activity(sd, now_unix=FROZEN_NOW, limit=10)

    assert [entry.relpath for entry in entries] == ["geak/new.log", "reports/mid.log", "runs/x/old.log"]
    assert entries[0].age_s == 3.0


def test_walk_reports_truncation(tmp_path: Path) -> None:
    """A partial view must never be presented as a complete one."""
    sd = tmp_path / "s"
    for index in range(20):
        write_activity(sd, f"geak/f{index}.log", age_s=float(index))

    _, truncated = walk_activity(sd, now_unix=FROZEN_NOW, limit=5, max_files=3)
    assert truncated is True


def test_truncated_walk_raises_a_snapshot_warning(tmp_path: Path, frozen_clock) -> None:
    """The truncation must reach the operator, not just the return value."""
    sd = tmp_path / "s"
    write_state(sd)
    for index in range(50):
        write_activity(sd, f"geak/f{index}.log", age_s=float(index))

    import hyperloom.observability.sources.activity as activity_mod

    original = activity_mod.MAX_FILES
    try:
        activity_mod.MAX_FILES = 3
        snapshot = load_snapshot(sd, now_unix=frozen_clock)
    finally:
        activity_mod.MAX_FILES = original

    assert snapshot is not None
    assert any("partial view" in warning for warning in snapshot.warnings)


def test_no_walkable_subtree_is_absent(tmp_path: Path) -> None:
    """An early-run session with no activity yet is not a broken one."""
    sd = tmp_path / "s"
    sd.mkdir(parents=True)

    assert ActivitySource().read(sd, now_unix=FROZEN_NOW).outcome is SourceOutcome.ABSENT


def test_missing_session_dir_is_absent(tmp_path: Path) -> None:
    """A path that does not exist is absent, not an error."""
    assert ActivitySource().read(tmp_path / "nope", now_unix=FROZEN_NOW).outcome is SourceOutcome.ABSENT


def test_container_pid_with_fresh_activity_is_never_dead(tmp_path: Path, frozen_clock) -> None:
    """Regression: the layer reported ``ended`` for a live 13-hour run.

    ``optimizer.lock`` held a *container* pid (304617) next to the *host's*
    hostname, because the container inherits it. The host had no such process,
    so the pid check concluded DEAD — and the pinned progress clock then
    displayed the running phase as ``0s``. Files were being written four
    seconds earlier the whole time.
    """
    sd = tmp_path / "s"
    write_state(sd, phase="KERNEL_AGENT", stop_reason="")
    # A pid that does not exist on this host, with a matching hostname.
    write_lock(sd, pid=999_999_998, hostname=socket.gethostname())
    write_activity(sd, "geak/e2e_cycle0/verify/driver.log", age_s=4.0)

    snapshot = load_snapshot(sd, now_unix=frozen_clock)

    assert snapshot is not None
    assert snapshot.liveness is not Liveness.DEAD
    assert snapshot.last_activity_age_s == 4.0
    # And the clock is not pinned, so the running phase reports real elapsed.
    current = snapshot.current_phase_progress
    assert current is not None and current.elapsed_s > 0


def test_recorded_pid_namespace_settles_the_question(tmp_path: Path, frozen_clock) -> None:
    """A lock from another PID namespace must not be interpreted locally."""
    from hyperloom.observability.sources.lockfile import LockFileSource

    sd = tmp_path / "s"
    write_state(sd)
    write_lock(sd, pid=1, hostname=socket.gethostname(), pid_ns="pid:[4026543243]")

    data = LockFileSource().read(sd).data
    assert data["same_host"] is True
    assert data["same_pid_ns"] is False
    # pid 1 exists on every host; without the namespace check it would have
    # been read as the optimizer being alive.
    assert data["pid_alive"] is None


def test_dead_is_still_reachable_without_activity(tmp_path: Path, frozen_clock) -> None:
    """The fix must not make DEAD unreachable — only harder to reach wrongly."""
    sd = tmp_path / "s"
    write_state(sd)
    write_lock(sd, pid=999_999_998, hostname=socket.gethostname(), pid_ns=_own_ns())

    snapshot = load_snapshot(sd, now_unix=lambda: FROZEN_NOW + 86_400.0)

    assert snapshot is not None
    assert snapshot.liveness is Liveness.DEAD


def _own_ns() -> str:
    """This process's PID-namespace id, so the fixture lock looks local."""
    import os

    try:
        return os.readlink("/proc/self/ns/pid")
    except OSError:  # pragma: no cover - non-Linux
        return ""
