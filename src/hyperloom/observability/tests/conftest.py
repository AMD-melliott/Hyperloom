# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Fixtures for the visibility-layer tests.

Builds synthetic session directories on disk rather than mocking the readers,
so the tests exercise the same parsing, coercion, and SQLite paths a real
session would.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest


# A fixed clock so every derived duration is exact. 2026-08-06T04:00:00Z.
FROZEN_NOW = 1785988800.0

SESSION_START_TS = "2026-08-06T00:00:00+00:00"
SESSION_START_UNIX = 1785974400.0  # FROZEN_NOW - 4h


@pytest.fixture(autouse=True)
def _isolate_session_env(monkeypatch):
    """Keep the ambient session pin out of every test.

    ``INFERENCE_OPTIMIZER_CURRENT_SESSION_DIR`` wins over auto-discovery, so a
    developer with a live run in their shell would otherwise see these tests
    read their real session.
    """
    monkeypatch.delenv("INFERENCE_OPTIMIZER_CURRENT_SESSION_DIR", raising=False)
    monkeypatch.delenv("USER_DATA_PATH", raising=False)


def write_state(session_dir: Path, **overrides) -> Path:
    """Write a ``state.json`` with sensible defaults plus ``overrides``.

    Args:
        session_dir: Session root; created if absent.
        **overrides: Keys merged over the defaults.

    Returns:
        Path to the written file.
    """
    session_dir.mkdir(parents=True, exist_ok=True)
    state = {
        "phase": "EXPLORE",
        "phase_started_unix": FROZEN_NOW - 1800.0,
        "phase_started_ts": "2026-08-06T03:30:00+00:00",
        "phase_elapsed_totals": {"PRELUDE": 3600.0, "EXPLORE": 5400.0},
        "phase_budget_pct": {"PRELUDE": 0.03, "EXPLORE": 0.45, "KERNEL_AGENT": 0.35},
        "max_minutes": 720,
        "cycle_minutes": 0.0,
        "start_ts": SESSION_START_TS,
        "macro_cycle": 0,
        "tick": 7,
        "baseline_tput": 100.0,
        "current_best": {"action": "explore", "tput": 115.0},
        "cumulative_gain": 15.0,
        "cumulative_gain_validated": 12.0,
        "target_gap_pct": 18.0,
        "crash_count": 0,
        "model_name": "test-model",
        "framework": "sglang",
        "gpu_type": "mi300x",
        "tp": 8,
        "conc": 64,
        "isl": 1024,
        "osl": 1024,
        "lifecycle": [
            {
                "seq": 0,
                "ts": "2026-08-06T01:00:00+00:00",
                "phase": "PRELUDE",
                "step": "roofline",
                "label": "TraceLens",
                "status": "END",
                "detail": "hot_kernels=18",
                "duration_s": 3600.0,
            }
        ],
        "phase_history": [
            {
                "from_phase": "",
                "to_phase": "PRELUDE",
                "reason": "phase_entered",
                "evidence": {},
                "ts": SESSION_START_TS,
                "ts_unix": SESSION_START_UNIX,
                "cycle": 0,
            },
            {
                "from_phase": "PRELUDE",
                "to_phase": "EXPLORE",
                "reason": "prelude_done",
                "evidence": {},
                "ts": "2026-08-06T01:00:00+00:00",
                "ts_unix": SESSION_START_UNIX + 3600.0,
                "cycle": 0,
            },
        ],
    }
    state.update(overrides)
    path = session_dir / "state.json"
    path.write_text(json.dumps(state, indent=2, sort_keys=True), encoding="utf-8")
    return path


def write_manifest(session_dir: Path, **overrides) -> Path:
    """Write a ``manifest.json`` with sensible defaults plus ``overrides``."""
    session_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "schema_version": 1,
        "session_id": "test-model_20260806T000000Z_deadbeef",
        "created_at_utc": SESSION_START_TS,
        "model_name": "test-model",
        "framework": "sglang",
        "gpu_type": "mi300x",
        "tp": 8,
        "ep": 1,
        "max_minutes": 720,
        "workload": {"isl": 1024, "osl": 1024, "conc": 64, "precision": "fp8", "max_model_len": 8192},
        "objective": {"kind": "gain_pct", "value": 30.0},
    }
    manifest.update(overrides)
    path = session_dir / "manifest.json"
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return path


def write_lock(session_dir: Path, *, pid: int, hostname: str, heartbeat_at: str = SESSION_START_TS) -> Path:
    """Write a ``runtime/optimizer.lock`` owner document."""
    runtime = session_dir / "runtime"
    runtime.mkdir(parents=True, exist_ok=True)
    path = runtime / "optimizer.lock"
    path.write_text(
        json.dumps(
            {
                "pid": pid,
                "hostname": hostname,
                "started_at": SESSION_START_TS,
                "heartbeat_at": heartbeat_at,
            }
        ),
        encoding="utf-8",
    )
    return path


def write_coordinator_db(
    session_dir: Path,
    *,
    tasks: list[tuple[str, str, str]] | None = None,
    leases: list[tuple[str, str, str]] | None = None,
    gpu_leases: list[tuple[int, str, str]] | None = None,
    lane_capacity: dict[str, int] | None = None,
) -> Path:
    """Build a minimal ``storage/coordinator.db``.

    Only the columns the visibility layer reads are created, which is itself a
    useful property: it proves the reader does not depend on the full schema.

    Args:
        session_dir: Session root.
        tasks: ``(task_id, kind, state)`` rows.
        leases: ``(lane, holder_id, expires_at)`` rows.
        gpu_leases: ``(gpu_id, holder_id, expires_at)`` rows.
        lane_capacity: Lane to capacity.

    Returns:
        Path to the created database.
    """
    storage = session_dir / "storage"
    storage.mkdir(parents=True, exist_ok=True)
    path = storage / "coordinator.db"
    conn = sqlite3.connect(str(path))
    try:
        conn.executescript(
            """
            CREATE TABLE tasks (task_id TEXT PRIMARY KEY, kind TEXT, state TEXT, updated_at TEXT);
            CREATE TABLE leases (lane TEXT, holder_id TEXT, task_id TEXT, action TEXT,
                                 pid INTEGER, acquired_at TEXT, expires_at TEXT, heartbeat_at TEXT);
            CREATE TABLE lane_capacity (lane TEXT PRIMARY KEY, capacity INTEGER);
            CREATE TABLE gpu_leases (gpu_id INTEGER PRIMARY KEY, holder_id TEXT, task_id TEXT,
                                     acquired_at TEXT, expires_at TEXT, heartbeat_at TEXT);
            """
        )
        for task_id, kind, state in tasks or []:
            conn.execute(
                "INSERT INTO tasks(task_id, kind, state, updated_at) VALUES (?, ?, ?, ?)",
                (task_id, kind, state, "2026-08-06T03:00:00+00:00"),
            )
        for lane, holder, expires in leases or []:
            conn.execute(
                "INSERT INTO leases(lane, holder_id, task_id, action, pid, acquired_at, expires_at, heartbeat_at) "
                "VALUES (?, ?, 't1', 'act', 1, ?, ?, ?)",
                (lane, holder, SESSION_START_TS, expires, SESSION_START_TS),
            )
        for lane, capacity in (lane_capacity or {}).items():
            conn.execute("INSERT INTO lane_capacity(lane, capacity) VALUES (?, ?)", (lane, capacity))
        for gpu_id, holder, expires in gpu_leases or []:
            conn.execute(
                "INSERT INTO gpu_leases(gpu_id, holder_id, task_id, acquired_at, expires_at, heartbeat_at) "
                "VALUES (?, ?, 't1', ?, ?, ?)",
                (gpu_id, holder, SESSION_START_TS, expires, SESSION_START_TS),
            )
        conn.commit()
    finally:
        conn.close()
    return path


@pytest.fixture
def session_dir(tmp_path: Path) -> Path:
    """A complete synthetic session: state, manifest, and coordinator DB."""
    sd = tmp_path / "test-model" / "20260806T000000Z"
    write_state(sd)
    write_manifest(sd)
    write_coordinator_db(
        sd,
        tasks=[
            ("t1", "baseline", "succeeded"),
            ("t2", "explore", "running"),
            ("t3", "explore", "queued"),
            ("t4", "kernel_opt", "failed"),
        ],
        leases=[("benchmark_lane", "holder-a", "2099-01-01T00:00:00+00:00")],
        lane_capacity={"benchmark_lane": 1, "research_lane": 4, "build_lane": 1},
        gpu_leases=[(0, "spec-a", "2099-01-01T00:00:00+00:00")],
    )
    return sd


@pytest.fixture
def frozen_clock():
    """Return a callable yielding :data:`FROZEN_NOW`."""
    return lambda: FROZEN_NOW
