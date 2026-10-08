# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""``storage/coordinator.db`` source — task queue, lane occupancy, GPU leases.

Opened strictly read-only through :mod:`hyperloom.observability.readonly`.
See that module for why ``bus.storage.connection.open_connection`` must never
be used here.

The database does not exist until the Coordinator boots, so ``ABSENT`` is the
normal answer for the first moments of a run and for a workspace root passed
by mistake.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from hyperloom.common.coerce import to_unix
from hyperloom.inference_optimizer.protocol.action_surfaces import ACTION_CATALOGUE
from hyperloom.inference_optimizer.session.paths import db_path_for

from ..model import RunningTaskSummary
from ..readonly import fetchall, readonly_connection
from .base import SourceResult


# Mirrors ``hyperloom.orchestrator.bus.gpu_pool._RAY_OBS_ID_BASE``. Duplicated
# rather than imported to keep the sources layer free of orchestrator imports;
# ``test_ray_obs_id_base_matches_orchestrator`` pins the two together so a
# change upstream fails a test instead of leaking "GPU 100003" into the display.
#
# Under single-node Ray execution the GPU pool is a count-based admission
# ledger, not a physical-device allocator: pending observation slots take
# synthetic ids at or above this base. They are accounting rows, not cards.
RAY_OBS_ID_BASE = 100000

# Task states, mirroring the CHECK constraint on ``tasks.state``.
TASK_STATES = ("queued", "running", "succeeded", "failed", "cancelled")
RUNNING_TASK_KINDS = tuple(sorted(ACTION_CATALOGUE)) + ("other",)


def _running_task(row: Any) -> dict[str, Any]:
    """Read state transitions and real progress, never lease/DB write times."""
    try:
        history = json.loads(row["history"])
    except (TypeError, json.JSONDecodeError):
        history = []
    if not isinstance(history, list):
        history = []
    started_at = None
    progress_at = None
    started_unix = None
    progress_unix = None
    for entry in history:
        if not isinstance(entry, dict):
            continue
        timestamp = to_unix(entry["ts"]) if isinstance(entry.get("ts"), str) else None
        if timestamp is None:
            continue
        if entry.get("to") == "running":
            started_at, started_unix = entry["ts"], timestamp
            progress_at, progress_unix = None, None
        elif isinstance(entry.get("progress"), dict):
            if progress_unix is None or timestamp > progress_unix:
                progress_at, progress_unix = entry["ts"], timestamp
    return {
        "task_id": str(row["task_id"]),
        "kind": str(row["kind"]),
        "state": str(row["state"]),
        "updated_at": row["updated_at"],
        "started_at": started_at,
        "started_unix": started_unix,
        "progress_at": progress_at,
        "progress_unix": progress_unix,
    }


def _summarize_running(rows: list[dict[str, Any]]) -> tuple[RunningTaskSummary, ...]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        kind = row["kind"] if row["kind"] in ACTION_CATALOGUE else "other"
        grouped.setdefault(kind, []).append(row)
    summaries = []
    for kind in RUNNING_TASK_KINDS:
        tasks = grouped.get(kind, [])
        starts = [row["started_unix"] for row in tasks if row["started_unix"] is not None]
        progress = [row["progress_unix"] for row in tasks if row["progress_unix"] is not None]
        summaries.append(
            RunningTaskSummary(
                kind=kind,
                count=len(tasks),
                oldest_started_unix=min(starts) if starts else None,
                latest_progress_unix=max(progress) if progress else None,
            )
        )
    return tuple(summaries)


def _is_expired(expires_at: Any, *, now_unix: float) -> bool:
    """Return whether an ISO ``expires_at`` is in the past.

    Args:
        expires_at: ISO timestamp string, or anything unparseable.
        now_unix: Current time.

    Returns:
        ``True`` when the deadline has passed. Unparseable values are treated
        as **not** expired: a lease we cannot interpret should not be reported
        as reclaimable.
    """
    if not expires_at:
        return False
    try:
        parsed = datetime.fromisoformat(str(expires_at))
    except (TypeError, ValueError):
        return False
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.timestamp() < now_unix


class CoordinatorDbSource:
    """Reads task, lease, and GPU-lease state from ``coordinator.db``."""

    name = "coordinator_db"

    def __init__(self, *, running_task_limit: int = 12, now_unix: float | None = None) -> None:
        """Configure the read.

        Args:
            running_task_limit: Maximum in-flight tasks to return.
            now_unix: Clock override used for expiry checks; defaults to the
                caller passing it at read time.
        """
        self._running_task_limit = int(running_task_limit)
        self._now_unix = now_unix

    def read(self, session_dir: Path, *, now_unix: float | None = None) -> SourceResult:
        """Read task counts, lane occupancy, and GPU leases.

        Args:
            session_dir: Absolute session root.
            now_unix: Current time, used to classify lease expiry.

        Returns:
            :class:`~.base.SourceResult` carrying task counts, bounded-kind
            running-task summaries, the detailed task list, lanes and GPU leases.
        """
        db_path = db_path_for(Path(session_dir))
        if not db_path.is_file():
            return SourceResult.absent()

        now = float(now_unix if now_unix is not None else (self._now_unix or 0.0))

        with readonly_connection(db_path) as conn:
            if conn is None:
                return SourceResult.error(f"cannot open {db_path.name} read-only")

            counts = {state: 0 for state in TASK_STATES}
            for row in fetchall(conn, "SELECT state, COUNT(*) AS n FROM tasks GROUP BY state"):
                state = str(row["state"])
                if state in counts:
                    counts[state] = int(row["n"])

            running = [
                _running_task(row)
                for row in fetchall(
                    conn,
                    "SELECT task_id, kind, state, updated_at, history FROM tasks "
                    "WHERE state = 'running' ORDER BY updated_at DESC, task_id",
                )
            ]
            summaries = _summarize_running(running)

            capacities = {
                str(row["lane"]): int(row["capacity"])
                for row in fetchall(conn, "SELECT lane, capacity FROM lane_capacity")
            }

            holders: dict[str, list[str]] = {}
            for row in fetchall(conn, "SELECT lane, holder_id, expires_at FROM leases"):
                if _is_expired(row["expires_at"], now_unix=now):
                    # An expired lease has not been reaped yet but is not
                    # holding the lane; counting it would over-report pressure.
                    continue
                holders.setdefault(str(row["lane"]), []).append(str(row["holder_id"]))

            lanes = [
                {
                    "lane": lane,
                    "capacity": capacities.get(lane, 1),
                    "holders": tuple(sorted(holders.get(lane, ()))),
                }
                for lane in sorted(set(capacities) | set(holders))
            ]

            gpu_leases = [
                {
                    "gpu_id": int(row["gpu_id"]),
                    "holder_id": str(row["holder_id"]),
                    "task_id": str(row["task_id"]),
                    "expires_at": row["expires_at"],
                    "expired": _is_expired(row["expires_at"], now_unix=now),
                }
                for row in fetchall(
                    conn,
                    "SELECT gpu_id, holder_id, task_id, expires_at FROM gpu_leases WHERE gpu_id < ? ORDER BY gpu_id",
                    (RAY_OBS_ID_BASE,),
                )
            ]

        return SourceResult.hit(
            {
                "task_counts": counts,
                "running_tasks": running[: max(0, self._running_task_limit)],
                "running_task_summaries": summaries,
                "lanes": lanes,
                "gpu_leases": gpu_leases,
            }
        )
