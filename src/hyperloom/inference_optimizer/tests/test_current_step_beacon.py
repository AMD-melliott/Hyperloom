# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Producer-side tests for the ``runtime/current_step.json`` beacon.

The beacon sits directly in the path of the most expensive work a session does
— the KERNEL_AGENT phase's multi-hour GEAK dispatch — so the properties that
matter are as much about what it must *not* do as what it records.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

from hyperloom.inference_optimizer.session.current_step import (
    beacon_path,
    clear_beacon,
    current_step,
    write_beacon,
)


def test_beacon_is_written_then_removed(tmp_path: Path) -> None:
    """The step is visible for the duration of the block and gone after."""
    path = beacon_path(tmp_path)
    assert not path.exists()

    with current_step(tmp_path, phase="KERNEL_AGENT", step="geak_e2e", deadline_unix=99.0):
        assert path.exists()
        payload = json.loads(path.read_text(encoding="utf-8"))
        assert payload["phase"] == "KERNEL_AGENT"
        assert payload["step"] == "geak_e2e"
        assert payload["deadline_unix"] == 99.0
        assert payload["started_unix"] > 0

    assert not path.exists()


def test_beacon_is_cleared_when_the_block_raises(tmp_path: Path) -> None:
    """A timeout or crash must not strand a phantom step on the display."""
    try:
        with current_step(tmp_path, phase="KERNEL_AGENT", step="geak_e2e"):
            raise TimeoutError("runner exceeded kill_timeout")
    except TimeoutError:
        pass

    assert not beacon_path(tmp_path).exists()


def test_a_write_failure_never_propagates(tmp_path: Path, monkeypatch) -> None:
    """Telemetry must not be able to fail the phase it is describing.

    A full disk or a read-only mount should cost the operator a status line,
    not an eight-hour kernel-optimization run.
    """
    monkeypatch.setattr(
        "hyperloom.inference_optimizer.session.current_step.atomic_write_json",
        lambda *a, **k: (_ for _ in ()).throw(OSError("No space left on device")),
    )

    assert write_beacon(tmp_path, phase="KERNEL_AGENT", step="geak_e2e") is False

    entered = False
    with current_step(tmp_path, phase="KERNEL_AGENT", step="geak_e2e"):
        entered = True
    assert entered, "the guarded block must still run when the beacon cannot be written"


def test_clearing_an_absent_beacon_is_a_no_op(tmp_path: Path) -> None:
    """Removal is attempted on every exit path, including ones that never wrote."""
    clear_beacon(tmp_path)  # must not raise
    assert not beacon_path(tmp_path).exists()


def test_beacon_lands_beside_the_session_lock(tmp_path: Path) -> None:
    """Path derivation must track the lock's, not hard-code ``runtime/``."""
    from hyperloom.inference_optimizer.session import session_paths

    assert beacon_path(tmp_path).parent == session_paths.optimizer_lock_path(tmp_path).parent


def test_artifacts_round_trip_as_strings(tmp_path: Path) -> None:
    """Paths are stringified so a ``Path`` value cannot break serialization."""
    with current_step(
        tmp_path,
        phase="KERNEL_AGENT",
        step="geak_e2e",
        artifacts={"out_dir": tmp_path / "geak", "result": tmp_path / "geak" / "result.json"},
    ):
        payload = json.loads(beacon_path(tmp_path).read_text(encoding="utf-8"))

    assert payload["artifacts"]["out_dir"] == str(tmp_path / "geak")
    assert payload["artifacts"]["result"].endswith("result.json")


def test_observability_reader_sees_what_the_producer_writes(tmp_path: Path) -> None:
    """The two halves of the contract must agree.

    Writer and reader live in different packages, so this is the only place the
    field names are checked end to end.
    """
    from hyperloom.observability.sources.current_step import CurrentStepSource

    deadline_unix = time.time() + 3600.0
    with current_step(
        tmp_path,
        phase="KERNEL_AGENT",
        step="geak_e2e",
        detail="GEAK e2e (from=EXPLORE)",
        deadline_unix=deadline_unix,
        artifacts={"out_dir": "/session/geak"},
    ):
        result = CurrentStepSource().read(tmp_path)

    assert result.ok
    step = result.data
    assert step.phase == "KERNEL_AGENT"
    assert step.step == "geak_e2e"
    assert step.detail == "GEAK e2e (from=EXPLORE)"
    assert step.deadline_unix == deadline_unix
    assert step.artifacts == (("out_dir", "/session/geak"),)
    assert step.budget_s() is not None and step.budget_s() > 0


def test_lock_records_its_pid_namespace(tmp_path: Path) -> None:
    """Without this the reader cannot tell whether the pid means anything.

    A containerized optimizer writes its namespace-local pid next to the host's
    hostname; hostname equality then licenses a pid check that has no standing,
    which produced both a false LIVE and a false DEAD on real sessions.
    """
    import os

    from hyperloom.inference_optimizer.session.lock import SessionLock

    lock = SessionLock(tmp_path)
    try:
        lock.acquire()
        owner = json.loads((tmp_path / "runtime" / "optimizer.lock").read_text(encoding="utf-8"))
    finally:
        lock.release()

    assert owner["pid"] == os.getpid()
    if Path("/proc/self/ns/pid").exists():
        assert owner["pid_ns"] == os.readlink("/proc/self/ns/pid")


def test_lock_heartbeat_advances(tmp_path: Path) -> None:
    """``heartbeat_at`` must actually move; it was written once and never again.

    On a real thirteen-hour run the field equalled ``started_at``, making the
    one signal designed for liveness useless.
    """
    import time

    from hyperloom.inference_optimizer.session.lock import SessionLock

    lock = SessionLock(tmp_path)
    path = tmp_path / "runtime" / "optimizer.lock"
    try:
        lock.acquire()
        first = json.loads(path.read_text(encoding="utf-8"))
        time.sleep(1.1)  # heartbeat_at has second resolution
        lock.heartbeat()
        second = json.loads(path.read_text(encoding="utf-8"))
    finally:
        lock.release()

    assert second["heartbeat_at"] > first["heartbeat_at"]
    assert second["started_at"] == first["started_at"]
