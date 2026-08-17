# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""A beacon naming the long-running step a phase is blocked on.

The phase machine publishes progress by ticking: each tick rewrites
``state.json`` and appends coordinator events. A phase that makes a blocking
call publishes nothing for the duration, because the tick loop is what would
have done the publishing and it is suspended. The KERNEL_AGENT phase's GEAK
dispatch runs under a timeout of up to twelve hours; on a real session it held
the loop for 4h54m, during which every artifact a reader knows about was
frozen and the status view could only report "no new evidence".

The beacon is written *before* the blocking call, so it survives the silence::

    with current_step(session_dir, phase="KERNEL_AGENT", step="geak_e2e",
                      deadline_unix=now + runner_timeout):
        proc = await asyncio.to_thread(_run)

giving an observer the step name, when it started and when it runs out of time
— none of which is otherwise recoverable while the loop is quiet.

Failures are swallowed by design. This is telemetry sitting directly in the
path of the most expensive work a session does; a full disk or a read-only
mount must degrade the display, never fail the phase.
"""

from __future__ import annotations

import logging
import os
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

from hyperloom.common.io import atomic_write_json

from . import session_paths


log = logging.getLogger(__name__)

BEACON_NAME = "current_step.json"


def beacon_path(session_dir: Path) -> Path:
    """Return the beacon path for a session.

    Mirrors :func:`session_paths.optimizer_lock_path` so the beacon lands
    beside the lock in ``runtime/``.

    Args:
        session_dir: Absolute session root.

    Returns:
        ``<session_dir>/runtime/current_step.json``.
    """
    return session_paths.optimizer_lock_path(Path(session_dir)).parent / BEACON_NAME


def write_beacon(
    session_dir: Path,
    *,
    phase: str,
    step: str,
    detail: str | None = None,
    started_unix: float | None = None,
    deadline_unix: float | None = None,
    artifacts: dict[str, Any] | None = None,
) -> bool:
    """Write the beacon document.

    Args:
        session_dir: Absolute session root.
        phase: Owning phase, e.g. ``"KERNEL_AGENT"``.
        step: Short machine-readable step name, e.g. ``"geak_e2e"``.
        detail: Optional human-readable elaboration.
        started_unix: Start time; defaults to now.
        deadline_unix: When the step will be killed, when it has a cap.
        artifacts: Paths worth surfacing, e.g. the step's output directory.

    Returns:
        ``True`` when the beacon was written.
    """
    path = beacon_path(session_dir)
    payload: dict[str, Any] = {
        "phase": str(phase).strip().upper(),
        "step": str(step).strip(),
        "detail": detail,
        "started_unix": float(started_unix if started_unix is not None else time.time()),
        "deadline_unix": (None if deadline_unix is None else float(deadline_unix)),
        "artifacts": {str(key): str(value) for key, value in (artifacts or {}).items()},
        "pid": os.getpid(),
    }
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        atomic_write_json(path, payload)
        return True
    except OSError as exc:
        log.debug("current_step: could not write beacon at %s: %s", path, exc)
        return False


def clear_beacon(session_dir: Path) -> None:
    """Remove the beacon, if present.

    A stale beacon is worse than none — it would report a step as in flight
    forever — so removal is attempted even on the error paths out of a step.

    Args:
        session_dir: Absolute session root.
    """
    try:
        beacon_path(session_dir).unlink(missing_ok=True)
    except OSError as exc:
        log.debug("current_step: could not clear beacon: %s", exc)


@contextmanager
def current_step(
    session_dir: Path,
    *,
    phase: str,
    step: str,
    detail: str | None = None,
    deadline_unix: float | None = None,
    artifacts: dict[str, Any] | None = None,
) -> Iterator[None]:
    """Publish a step for the duration of the enclosed block.

    The beacon is cleared in a ``finally``, so an exception or a timeout inside
    the block still leaves the session with no phantom step.

    Args:
        session_dir: Absolute session root.
        phase: Owning phase.
        step: Short machine-readable step name.
        detail: Optional human-readable elaboration.
        deadline_unix: When the step will be killed, when it has a cap.
        artifacts: Paths worth surfacing.

    Yields:
        ``None``.
    """
    write_beacon(
        session_dir,
        phase=phase,
        step=step,
        detail=detail,
        deadline_unix=deadline_unix,
        artifacts=artifacts,
    )
    try:
        yield
    finally:
        clear_beacon(session_dir)
