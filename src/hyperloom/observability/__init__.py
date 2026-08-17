# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Read-only visibility layer for Hyperloom optimization sessions.

Answers "what step is this run on, and how much budget is left" by reading the
artifacts the optimizer already writes — ``state.json``, ``manifest.json``,
``runtime/optimizer.lock``, ``runtime/current_step.json``, ``runs/``, and
``storage/coordinator.db`` — plus live GPU and inference-server metrics.
Nothing here mutates a session.

The layer is a domain model plus pluggable sources; renderers are consumers of
:class:`~.model.Snapshot` and never touch the filesystem::

    from hyperloom.observability import load_snapshot

    snapshot = load_snapshot()          # auto-resolves the active session
    print(snapshot.phase, snapshot.liveness)

For a live view, :class:`~.collector.SessionMonitor` polls the sources on
independent cadences into a cache and :func:`~.progress.extrapolate_to`
advances the clocks at paint time, so the display never waits on a probe::

    from hyperloom.observability.collector import SessionMonitor

    with SessionMonitor(session_dir) as monitor:
        snapshot = monitor.current()    # cache read plus arithmetic, no I/O

Note that :func:`load_snapshot` is deliberately importable without pulling in
the orchestrator: only :mod:`.progress` reaches into
``hyperloom.orchestrator.phases.machine_state``, and it does so lazily.
"""

from __future__ import annotations

from .assemble import load_snapshot, resolve_session_dir
from .model import (
    ActivityEntry,
    CurrentStep,
    Freshness,
    GeakProgress,
    GpuLease,
    GpuMetric,
    GpuMetrics,
    LaneOccupancy,
    LifecycleEvent,
    Liveness,
    PhaseProgress,
    ResultSummary,
    RunningTask,
    RunningWork,
    ServerMetrics,
    SessionInfo,
    Snapshot,
    SourceHealth,
    SourceOutcome,
    TaskCounts,
)
from .progress import extrapolate_to


__all__ = [
    "ActivityEntry",
    "CurrentStep",
    "Freshness",
    "GeakProgress",
    "GpuLease",
    "GpuMetric",
    "GpuMetrics",
    "LaneOccupancy",
    "LifecycleEvent",
    "Liveness",
    "PhaseProgress",
    "ResultSummary",
    "RunningTask",
    "RunningWork",
    "ServerMetrics",
    "SessionInfo",
    "Snapshot",
    "SourceHealth",
    "SourceOutcome",
    "TaskCounts",
    "extrapolate_to",
    "load_snapshot",
    "resolve_session_dir",
]
