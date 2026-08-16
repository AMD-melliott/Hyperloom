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

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from hyperloom.inference_optimizer.session.paths import db_path_for

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
            :class:`~.base.SourceResult` carrying ``{"task_counts",
            "running_tasks", "lanes", "gpu_leases"}``.
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
                {
                    "task_id": str(row["task_id"]),
                    "kind": str(row["kind"]),
                    "state": str(row["state"]),
                    "updated_at": row["updated_at"],
                }
                for row in fetchall(
                    conn,
                    "SELECT task_id, kind, state, updated_at FROM tasks "
                    "WHERE state = 'running' ORDER BY updated_at DESC LIMIT ?",
                    (self._running_task_limit,),
                )
            ]

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
                "running_tasks": running,
                "lanes": lanes,
                "gpu_leases": gpu_leases,
            }
        )
