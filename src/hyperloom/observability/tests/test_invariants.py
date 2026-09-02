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


def test_package_import_does_not_pull_the_orchestrator() -> None:
    """Importing the package itself must stay cheap and orchestrator-free.

    ``model`` alone being clean is not enough: the package ``__init__`` re-
    exports from ``assemble`` and ``progress``, and ``progress`` is the one
    module that reaches into ``machine_state``. It does so with function-level
    imports specifically so this stays true, which is easy to undo by moving
    one import to the top of the file.
    """
    code = (
        "import sys; import hyperloom.observability; "
        "print(';'.join(m for m in sys.modules if m.startswith('hyperloom.orchestrator')))"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        check=True,
        cwd=str(Path(__file__).resolve().parents[3]),
    )
    assert result.stdout.strip() == "", f"package leaked orchestrator imports: {result.stdout.strip()}"


def test_ray_obs_id_base_matches_orchestrator() -> None:
    """The mirrored Ray sentinel must track its upstream definition.

    ``RAY_OBS_ID_BASE`` is duplicated to keep the sources layer orchestrator-
    free. Pinning it here means an upstream change fails a test instead of
    leaking synthetic slot ids like "GPU 100003" into the display.
    """
    from hyperloom.orchestrator.bus.gpu_pool import _RAY_OBS_ID_BASE

    assert RAY_OBS_ID_BASE == _RAY_OBS_ID_BASE


def test_exit_limit_sets_match_the_orchestrator() -> None:
    """The mirrored exit-limit sets must track which phases really check them.

    ``PhaseProgress.pct_used`` measures against whichever limit ends the phase,
    which requires knowing that FRAMEWORK_AGENT/KERNEL_AGENT/SWEEP consult both
    ``phase_cap_exceeded`` and ``phase_budget_remaining_seconds`` — and that
    PRELUDE and CLOSE have no exit check at all, so their computable caps are
    never enforced.

    That is policy living in another module, so it is derived from the source
    here rather than trusted: if an ``exit_normal_*`` helper starts or stops
    consulting a limit, this fails instead of the display quietly misreporting
    an overrun (or hiding one).
    """
    import inspect
    import re

    from hyperloom.orchestrator.phases import machine_state

    from ..model import BUDGET_EXIT_PHASES, CAP_EXIT_PHASES

    # The helper names don't all follow the phase name (FRAMEWORK_AGENT's is
    # exit_normal_optimize, having absorbed the retired EXPLORE phase's
    # config-search arm), so the mapping is spelled out rather than derived.
    helpers = {
        "PRELUDE": "exit_normal_prelude",
        "FRAMEWORK_AGENT": "exit_normal_optimize",
        "KERNEL_AGENT": "exit_normal_kernel",
        "SWEEP": "exit_normal_sweep",
        "CLOSE": "exit_normal_close",
    }
    checks_cap: set[str] = set()
    checks_budget: set[str] = set()
    for phase, helper_name in helpers.items():
        helper = getattr(machine_state, helper_name, None)
        if helper is None:
            # No exit helper means no enforced limit, which is the claim being
            # made about PRELUDE and CLOSE.
            assert phase not in CAP_EXIT_PHASES and phase not in BUDGET_EXIT_PHASES
            continue
        # Strip the docstring: it discusses caps and budgets in prose.
        body = re.sub(r'""".*?"""', "", inspect.getsource(helper), count=1, flags=re.DOTALL)
        if "phase_cap_exceeded(" in body:
            checks_cap.add(phase)
        if "phase_budget_remaining_seconds(" in body:
            checks_budget.add(phase)

    assert checks_cap == set(CAP_EXIT_PHASES), (
        f"machine_state enforces a cap for {sorted(checks_cap)}, "
        f"but model.CAP_EXIT_PHASES says {sorted(CAP_EXIT_PHASES)}"
    )
    assert checks_budget == set(BUDGET_EXIT_PHASES), (
        f"machine_state exit checks consult the budget for {sorted(checks_budget)}, "
        f"but model.BUDGET_EXIT_PHASES says {sorted(BUDGET_EXIT_PHASES)}"
    )


def test_session_discovery_matches_what_the_producer_writes(tmp_path: Path, monkeypatch) -> None:
    """Auto-discovery must find a directory the producer actually created.

    ``find_latest_per_session_dir`` recognises per-session directories by the
    shape of their name, so it and ``make_session_dir`` encode one convention in
    two places. When the producer grew a ``-<rand8>`` suffix, a reader still
    requiring the bare 16-character stamp would match nothing — and "no session
    found" is indistinguishable from there being none, which is what this pins.

    Driven through the real producer rather than a fabricated name, so a future
    change to the layout fails here instead of on an operator's terminal.
    """
    from hyperloom.inference_optimizer.session.paths import (
        ENV_CURRENT_SESSION_DIR,
        find_latest_per_session_dir,
        make_session_dir,
    )

    monkeypatch.setenv("USER_DATA_PATH", str(tmp_path))
    monkeypatch.delenv(ENV_CURRENT_SESSION_DIR, raising=False)

    created = make_session_dir("meta-models/Muse-Glimmer-30B")

    assert find_latest_per_session_dir("meta-models/Muse-Glimmer-30B") == created
    assert find_latest_per_session_dir() == created, "unfiltered scan missed the session"


def test_session_discovery_picks_the_newest_by_stamp_not_mtime(tmp_path: Path, monkeypatch) -> None:
    """A resume touching an older session must not make it look newest.

    Selection is a lexical sort on the leading fixed-width timestamp precisely
    so that writing into an older session does not re-order it. Sorting by mtime
    is the tempting "fix" here, and would point the status view at whichever
    session was written to last rather than the one that started last.
    """
    import os

    from hyperloom.inference_optimizer.session.paths import find_latest_per_session_dir

    monkeypatch.setenv("USER_DATA_PATH", str(tmp_path))
    older = tmp_path / "model" / "20260101T000000Z-aaaaaaaa"
    newer = tmp_path / "model" / "20260817T000000Z-bbbbbbbb"
    for path in (older, newer):
        path.mkdir(parents=True)
    # Make the older session the most recently written one.
    os.utime(older, (2_000_000_000, 2_000_000_000))
    os.utime(newer, (1_000_000_000, 1_000_000_000))

    assert find_latest_per_session_dir() == newer


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
    assert machine_state.normalize_budget_pct(snapshot.phase_budget_pct)["FRAMEWORK_AGENT"] == 0.45


def test_read_only_invariant_covers_the_new_sources(tmp_path: Path, frozen_clock) -> None:
    """Activity, GEAK and beacon reads must not touch the session either.

    The original invariant only exercised state/manifest/lock/db. The activity
    walk stats thousands of files and the GEAK source descends an output tree,
    so both get far more opportunity to mutate something by accident.
    """
    from .conftest import write_activity, write_run, write_state

    sd = tmp_path / "busy"
    write_state(sd)
    write_run(sd, run_id="a1", heartbeat={"status": "running", "note": "working"}, log_bytes=64)
    write_activity(sd, "geak/handoff.json", contents="{}")
    write_activity(sd, "geak/e2e_cycle0/kernels/_exp/team_x/task_y/round_1/engineer_0/verify/driver.log")
    write_activity(sd, "reports/optimization_journal.json", contents="{}")

    before = _tree_fingerprint(sd)
    snapshot = load_snapshot(sd, now_unix=frozen_clock)
    assert snapshot is not None
    assert snapshot.running_work, "fixture should have produced running work"
    assert snapshot.geak is not None, "fixture should have produced GEAK progress"

    assert _tree_fingerprint(sd) == before, "reading activity mutated the session"


def test_beacon_source_never_creates_the_beacon(tmp_path: Path, frozen_clock) -> None:
    """The producer removes the beacon on exit; a reader must not resurrect it.

    A recreated beacon would strand a phantom "step in flight" on the display
    for the rest of the session.
    """
    from hyperloom.observability.sources.current_step import CurrentStepSource, beacon_path

    from .conftest import write_state

    sd = tmp_path / "no-beacon"
    write_state(sd)

    assert CurrentStepSource().read(sd).outcome.value == "absent"
    assert not beacon_path(sd).exists()
    assert not (sd / "runtime").exists()


def test_gpu_and_server_sources_ignore_the_session_directory(tmp_path: Path, monkeypatch) -> None:
    """Host-scoped probes must not read or write inside a session.

    They accept ``session_dir`` only for protocol symmetry; taking a dependency
    on it would make a host metric look session-attributable, which the model
    explicitly denies.
    """
    from hyperloom.observability.sources import GpuSource, ServerMetricsSource

    sd = tmp_path / "session"
    sd.mkdir()
    before = _tree_fingerprint(sd)

    monkeypatch.setattr("hyperloom.observability.sources.server.discover_base_url", lambda: None)
    GpuSource().read(sd)
    ServerMetricsSource().read(sd)

    assert _tree_fingerprint(sd) == before
