# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Structural invariants of the visibility layer.

These are the load-bearing properties. If one of them regresses, the layer is
still "working" in the sense that it produces output — which is exactly why
they need tests rather than review attention.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from hyperloom.observability import load_snapshot
from hyperloom.observability.sources.coordinator_db import RAY_OBS_ID_BASE


def _tree_fingerprint(root: Path) -> dict[str, tuple[float, int]]:
    """Map every file under ``root`` to its ``(mtime, size)``."""
    return {
        str(path.relative_to(root)): (path.stat().st_mtime, path.stat().st_size)
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def test_read_only_invariant(session_dir: Path, frozen_clock) -> None:
    """Reading a session must not create, delete, or touch a single file.

    This is the guarantee that makes the layer safe to point at a live run, and
    it specifically guards the ``bus.storage.connection.open_connection``
    footgun: that helper mkdirs and runs DDL, so any accidental use of it would
    fail here rather than silently mutating an operator's session.
    """
    before = _tree_fingerprint(session_dir)

    snapshot = load_snapshot(session_dir, now_unix=frozen_clock)
    assert snapshot is not None

    after = _tree_fingerprint(session_dir)
    assert after == before, "reading the session mutated it"


def test_read_only_does_not_create_missing_database(tmp_path: Path, frozen_clock) -> None:
    """A session with no ``coordinator.db`` must not gain one by being read."""
    from .conftest import write_state

    sd = tmp_path / "bare"
    write_state(sd)

    snapshot = load_snapshot(sd, now_unix=frozen_clock)

    assert snapshot is not None
    assert not (sd / "storage" / "coordinator.db").exists()
    assert not (sd / "storage").exists()


def test_model_does_not_import_orchestrator() -> None:
    """``model`` must stay orchestrator-free.

    Run in a subprocess because the orchestrator is almost certainly already in
    ``sys.modules`` by the time any other test has run — an in-process
    ``sys.modules`` check would pass for the wrong reason.
    """
    code = (
        "import sys; import hyperloom.observability.model; "
        "leaked = [m for m in sys.modules if m.startswith('hyperloom.orchestrator')]; "
        "print(';'.join(leaked))"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        check=True,
        cwd=str(Path(__file__).resolve().parents[3]),
    )
    assert result.stdout.strip() == "", f"model leaked orchestrator imports: {result.stdout.strip()}"


def test_ray_obs_id_base_matches_orchestrator() -> None:
    """The mirrored Ray sentinel must track its upstream definition.

    ``RAY_OBS_ID_BASE`` is duplicated to keep the sources layer orchestrator-
    free. Pinning it here means an upstream change fails a test instead of
    leaking synthetic slot ids like "GPU 100003" into the display.
    """
    from hyperloom.orchestrator.bus.gpu_pool import _RAY_OBS_ID_BASE

    assert RAY_OBS_ID_BASE == _RAY_OBS_ID_BASE


def test_snapshot_satisfies_machine_state_contract(session_dir: Path, frozen_clock) -> None:
    """The snapshot must be consumable by the real phase-machine helpers.

    The whole design rests on not reimplementing budget math, so this asserts
    the duck-typing contract directly against the production functions rather
    than trusting the field names by eye.
    """
    from hyperloom.orchestrator.phases import machine_state

    snapshot = load_snapshot(session_dir, now_unix=frozen_clock)
    assert snapshot is not None

    # Real dict, not a Mapping proxy: phase_cumulative_seconds guards with
    # isinstance(totals, dict) and would silently bank zero otherwise.
    assert type(snapshot.phase_elapsed_totals) is dict

    assert machine_state.phase_cumulative_seconds(snapshot, phase="PRELUDE", now_unix=frozen_clock()) > 0
    assert machine_state.session_remaining_seconds(snapshot, now_unix=frozen_clock()) is not None
    assert machine_state.normalize_budget_pct(snapshot.phase_budget_pct)["EXPLORE"] == 0.45
