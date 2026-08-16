# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Read-only SQLite access to a session's ``coordinator.db``.

``hyperloom.orchestrator.bus.storage.connection.open_connection`` MUST NOT be
used from a reader: it ``mkdir``s the parent directory and runs the full DDL
under ``BEGIN IMMEDIATE``, so pointing it at a missing or mistyped path
silently *creates* a database. Asking a live session a question must never
change it.

This module is the one sanctioned reader. The ``mode=ro`` URI form it uses was
already duplicated in three call sites
(:mod:`hyperloom.agents.robustness.sources.local_probe` twice and
:mod:`hyperloom.inference_optimizer.breakdown.collectors.telemetry`); this
centralizes it.

Note ``mode=ro`` still requires the sidecar ``-wal`` / ``-shm`` files to be
readable, since the session DB runs in WAL mode by default. That holds for the
operator reading their own session directory, which is the only supported case.
"""

from __future__ import annotations

import logging
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator, Sequence


log = logging.getLogger(__name__)

# Short on purpose: a status reader must never block on a busy writer. The
# Coordinator holds write transactions for meaningful stretches, so a reader
# that waits is a reader that hangs the terminal.
DEFAULT_TIMEOUT_SEC = 2.0


def open_readonly(db_path: Path | None, *, timeout: float = DEFAULT_TIMEOUT_SEC) -> sqlite3.Connection | None:
    """Open ``db_path`` read-only, or return ``None`` when unavailable.

    Never raises and never creates the database: a missing file, an unreadable
    file, and a corrupt file all degrade to ``None`` so callers can treat
    "no database yet" as the ordinary early-run state it is.

    Args:
        db_path: Path to ``coordinator.db``. ``None`` is accepted and yields
            ``None`` so callers need no separate guard.
        timeout: Busy timeout in seconds.

    Returns:
        A connection with ``sqlite3.Row`` as its row factory, or ``None``.
    """
    if db_path is None or not Path(db_path).is_file():
        return None
    try:
        conn = sqlite3.connect(f"file:{Path(db_path)}?mode=ro", uri=True, timeout=float(timeout))
    except sqlite3.Error as exc:
        log.debug("open_readonly: cannot open %s: %s", db_path, exc)
        return None
    conn.row_factory = sqlite3.Row
    return conn


@contextmanager
def readonly_connection(
    db_path: Path | None,
    *,
    timeout: float = DEFAULT_TIMEOUT_SEC,
) -> Iterator[sqlite3.Connection | None]:
    """Context-manager wrapper around :func:`open_readonly` that always closes.

    Args:
        db_path: Path to ``coordinator.db``; ``None`` yields ``None``.
        timeout: Busy timeout in seconds.

    Yields:
        The open connection, or ``None`` when the database is unavailable.
    """
    conn = open_readonly(db_path, timeout=timeout)
    try:
        yield conn
    finally:
        if conn is not None:
            try:
                conn.close()
            except sqlite3.Error:  # noqa: BLE001 — close must never mask the caller's result
                log.debug("readonly_connection: close failed for %s", db_path, exc_info=True)


def fetchall(
    conn: sqlite3.Connection | None,
    sql: str,
    params: Sequence[Any] = (),
) -> list[sqlite3.Row]:
    """Run ``sql`` and return every row, degrading to ``[]`` on any error.

    A schema mismatch (an older or newer ``coordinator.db``) surfaces as
    ``sqlite3.Error`` and is treated as "no rows" rather than a crash: a status
    reader that dies on an unexpected column is worse than one that shows less.

    Args:
        conn: Open read-only connection, or ``None``.
        sql: Query to execute.
        params: Bound parameters.

    Returns:
        The result rows, or ``[]``.
    """
    if conn is None:
        return []
    try:
        return list(conn.execute(sql, tuple(params)).fetchall())
    except sqlite3.Error as exc:
        log.debug("fetchall failed (%s): %s", sql.split()[0:3], exc)
        return []
