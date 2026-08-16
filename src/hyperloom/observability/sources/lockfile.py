# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""``runtime/optimizer.lock`` source — the primary liveness signal.

The lock file is **never unlinked**, so its presence proves only that a run
once started in this directory. Liveness comes from the recorded ``pid`` plus
``heartbeat_at``, never from the file existing.

One correctness detail the shell watchdog
(``tools/robustness_monitor.sh.example``) does not handle: the lock records the
``hostname`` that wrote it. On a shared filesystem the pid is meaningless on
any other host — the local pid table would be answering a question about a
different machine, and could easily report a live but unrelated process. When
the hostname does not match we return ``UNKNOWN`` and fall back to heartbeat
age, rather than confidently reporting ``LIVE`` or ``DEAD`` from a pid we have
no standing to interpret.
"""

from __future__ import annotations

import os
import socket
from datetime import datetime, timezone
from pathlib import Path

from hyperloom.inference_optimizer.session.lock import read_owner

from .base import SourceResult


def pid_alive(pid: int | None) -> bool:
    """Return whether ``pid`` exists on this host.

    Args:
        pid: Process id to probe; ``None`` and non-positive values are dead.

    Returns:
        ``True`` when the process exists. A ``PermissionError`` counts as alive
        — the process is there, just owned by another user.
    """
    try:
        pid_int = int(pid)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return False
    if pid_int <= 0:
        return False
    try:
        os.kill(pid_int, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return False
    return True


def _age_seconds(ts: str | None, *, now_unix: float) -> float | None:
    """Return the age in seconds of an ISO timestamp, or ``None`` if unparseable."""
    if not ts:
        return None
    try:
        parsed = datetime.fromisoformat(str(ts))
    except (TypeError, ValueError):
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return max(0.0, now_unix - parsed.timestamp())


class LockFileSource:
    """Reads ``<session_dir>/runtime/optimizer.lock``."""

    name = "lock"

    def read(self, session_dir: Path) -> SourceResult:
        """Read the lock's owner document and interpret it against this host.

        Args:
            session_dir: Absolute session root.

        Returns:
            :class:`~.base.SourceResult` carrying ``{"pid", "hostname",
            "heartbeat_at", "started_at", "same_host", "pid_alive"}``, where
            ``pid_alive`` is ``None`` when the lock was written by another host
            and the local pid table cannot answer.
        """
        owner = read_owner(session_dir)
        if owner is None:
            return SourceResult.absent()

        recorded_host = str(owner.get("hostname") or "").strip()
        same_host = bool(recorded_host) and recorded_host == socket.gethostname()
        raw_pid = owner.get("pid")
        try:
            pid: int | None = int(raw_pid) if raw_pid is not None else None
        except (TypeError, ValueError):
            pid = None

        # Only interpret the pid when the lock was written here. A pid from
        # another host would be answered by the wrong process table.
        alive: bool | None = pid_alive(pid) if same_host else None

        return SourceResult.hit(
            {
                "pid": pid,
                "hostname": recorded_host or None,
                "heartbeat_at": owner.get("heartbeat_at"),
                "started_at": owner.get("started_at"),
                "same_host": same_host,
                "pid_alive": alive,
            }
        )


def heartbeat_age_s(lock_data: dict | None, *, now_unix: float) -> float | None:
    """Return the age of the lock's ``heartbeat_at``, or ``None`` when unknown.

    Args:
        lock_data: Payload from :meth:`LockFileSource.read`, or ``None``.
        now_unix: Current time.

    Returns:
        Seconds since the last heartbeat, or ``None``.
    """
    if not lock_data:
        return None
    return _age_seconds(lock_data.get("heartbeat_at"), now_unix=now_unix)
