# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Read-only visibility layer for Hyperloom optimization sessions.

Answers "what step is this run on, and how much budget is left" by reading the
artifacts the optimizer already writes — ``state.json``, ``manifest.json``,
``runtime/optimizer.lock``, and ``storage/coordinator.db``. Nothing here
mutates a session.

The layer is a domain model plus pluggable sources; renderers are consumers of
:class:`~.model.Snapshot` and never touch the filesystem::

    from hyperloom.observability import load_snapshot

    snapshot = load_snapshot()          # auto-resolves the active session
    print(snapshot.phase, snapshot.liveness)

Note that :func:`load_snapshot` is deliberately importable without pulling in
the orchestrator: only :mod:`.progress` reaches into
``hyperloom.orchestrator.phases.machine_state``, and it does so lazily.
"""

from __future__ import annotations

from .assemble import load_snapshot, resolve_session_dir
from .model import (
    Freshness,
    GpuLease,
    LaneOccupancy,
    LifecycleEvent,
    Liveness,
    PhaseProgress,
    ResultSummary,
    RunningTask,
    SessionInfo,
    Snapshot,
    SourceOutcome,
    TaskCounts,
)


__all__ = [
    "Freshness",
    "GpuLease",
    "LaneOccupancy",
    "LifecycleEvent",
    "Liveness",
    "PhaseProgress",
    "ResultSummary",
    "RunningTask",
    "SessionInfo",
    "Snapshot",
    "SourceOutcome",
    "TaskCounts",
    "load_snapshot",
    "resolve_session_dir",
]
