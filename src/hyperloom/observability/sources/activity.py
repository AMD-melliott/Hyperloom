# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Sub-phase activity — what a session is doing when the phase machine is quiet.

The phase machine writes ``state.json`` on every tick, but a blocking dispatch
suspends ticking entirely: a GEAK end-to-end run holds the loop for up to
twelve hours. On a real session both ``state.json`` and ``coordinator.db`` sat
untouched for 4h54m while three sub-agents churned through kernel verification
the whole time. The status view correctly reported "no new evidence", which was
true and useless.

This source looks where the work actually is. Two complementary readings:

* **Structured run probes.** Every ``runs/<kind>/<run_id>/`` is inspected for
  ``heartbeat.json`` and ``process.log``. The heartbeat is not a bare
  liveness ping — the specialist prompt (``prompts/specialist_prompt_builder``)
  instructs agents to write ``{"ts", "status", "note"}`` every five minutes,
  where ``note`` is a short description of what they are doing. The reap loop
  at ``orchestrator/loop/conversation.py`` already reads these files and keeps
  only the mtime; the note text has been sitting on disk unread.
* **A bounded activity walk.** The newest-written files under a whitelist of
  subtrees. The path itself is the explanation:
  ``geak/…/round_1/engineer_0/verify/driver.log`` says more about what is
  happening than any status field the producer thought to emit.

Cost was measured before this was built: an 18-hour session tree is ~1800 files
after pruning, walked in 64 ms. The caps below exist for the pathological case,
not the normal one, and a walk that hits them says so rather than presenting a
truncated view as complete.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Iterable

from hyperloom.common.coerce import to_float, to_int
from hyperloom.common.jsonio import read_json

from ..model import ActivityEntry, RunningWork
from .base import SourceResult


# Subtrees worth walking. Everything else in a session directory is either
# static (config, prompts) or enormous and uninformative (model caches).
ACTIVITY_ROOTS: tuple[str, ...] = ("runs", "geak", "kernel-agent", "reports", "critic-workdir", "robustness-workdir")

# Directory names pruned outright. ``.git`` and ``__pycache__`` dominate the
# file count in a GEAK tree (153 of 1808 files on the measured session) and
# their mtimes track tooling, not progress.
PRUNE_DIRS: frozenset[str] = frozenset({".git", "__pycache__", "node_modules", ".mypy_cache", ".pytest_cache"})

# Suffixes that are build residue rather than evidence of work.
PRUNE_SUFFIXES: tuple[str, ...] = (".pyc", ".pyo", ".swp")

# Walk bounds. Generous relative to the measured 1800 files, so hitting one
# means something is genuinely pathological and the operator should be told.
MAX_FILES = 40000
MAX_DIRS = 4000
MAX_WALK_SECONDS = 1.5

DEFAULT_ACTIVITY_LIMIT = 6

# Run kinds under ``runs/``. Directory names are the delegated action kinds.
_HEARTBEAT = "heartbeat.json"
_PROCESS_LOG = "process.log"


def _safe_mtime(path: Path) -> float | None:
    """Return a path's mtime, or ``None`` when it cannot be stated."""
    try:
        return path.stat().st_mtime
    except OSError:
        return None


def _read_heartbeat(path: Path) -> dict[str, Any]:
    """Parse a ``heartbeat.json``, tolerating both writers' schemas.

    Two producers write this file with different shapes: the dispatcher emits
    ``{"turn", "max_turns", "status", "ts", "ts_unix"}``, while the agent
    itself is prompted to write ``{"ts", "status", "note"}``. Neither is
    authoritative and both are useful, so fields are read opportunistically.

    Args:
        path: The heartbeat file.

    Returns:
        The parsed mapping, or ``{}`` when unreadable.
    """
    try:
        payload = read_json(path)
    except Exception:  # noqa: BLE001 - a malformed heartbeat is not an error worth raising
        return {}
    return payload if isinstance(payload, dict) else {}


def probe_runs(
    session_dir: Path,
    *,
    now_unix: float,
    previous: dict[str, tuple[float, int]] | None = None,
    active_within_s: float = 900.0,
) -> tuple[tuple[RunningWork, ...], dict[str, tuple[float, int]]]:
    """Inspect every run directory for signs of life.

    Args:
        session_dir: Absolute session root.
        now_unix: Observation time.
        previous: ``{run_key: (observed_unix, log_bytes)}`` from the prior
            probe, used to derive log growth rate. ``None`` on a cold start.
        active_within_s: Only report runs whose freshest signal is newer than
            this. A finished run's files stay on disk forever; without a
            window every run a session ever launched would be listed as work.

    Returns:
        ``(rows, log_state)`` where ``log_state`` should be handed back as
        ``previous`` on the next call.
    """
    runs_root = session_dir / "runs"
    rows: list[RunningWork] = []
    log_state: dict[str, tuple[float, int]] = {}
    prior = previous or {}

    try:
        kinds = sorted(entry for entry in runs_root.iterdir() if entry.is_dir())
    except OSError:
        return (), log_state

    for kind_dir in kinds:
        try:
            run_dirs = sorted(entry for entry in kind_dir.iterdir() if entry.is_dir())
        except OSError:
            continue
        for run_dir in run_dirs:
            key = f"{kind_dir.name}/{run_dir.name}"
            hb_path = run_dir / _HEARTBEAT
            log_path = run_dir / _PROCESS_LOG

            hb_mtime = _safe_mtime(hb_path)
            log_stat: Any = None
            try:
                log_stat = log_path.stat()
            except OSError:
                log_stat = None

            if hb_mtime is None and log_stat is None:
                continue

            hb_age = None if hb_mtime is None else max(0.0, now_unix - hb_mtime)
            log_age = None if log_stat is None else max(0.0, now_unix - log_stat.st_mtime)
            log_bytes = None if log_stat is None else int(log_stat.st_size)

            if log_bytes is not None:
                log_state[key] = (now_unix, log_bytes)

            freshest = min(age for age in (hb_age, log_age) if age is not None)
            if freshest > active_within_s:
                continue

            growth = None
            if log_bytes is not None and key in prior:
                prev_unix, prev_bytes = prior[key]
                span = now_unix - prev_unix
                if span > 0 and log_bytes >= prev_bytes:
                    growth = (log_bytes - prev_bytes) / span

            heartbeat = _read_heartbeat(hb_path) if hb_mtime is not None else {}

            rows.append(
                RunningWork(
                    kind=kind_dir.name,
                    run_id=run_dir.name,
                    status=(str(heartbeat.get("status")) if heartbeat.get("status") else None),
                    note=(str(heartbeat.get("note")).strip() or None) if heartbeat.get("note") else None,
                    turn=to_int(heartbeat.get("turn")),
                    max_turns=to_int(heartbeat.get("max_turns")),
                    heartbeat_age_s=hb_age,
                    log_bytes=log_bytes,
                    log_growth_bps=growth,
                    log_age_s=log_age,
                    has_partial_result=any(run_dir.glob("*_done.partial.json")),
                    # A live heartbeat inside a run whose done-file is already
                    # written means something is writing into a completed run's
                    # directory. Observed in the wild; flagged rather than
                    # silently attributed to the finished task.
                    task_terminal=any(run_dir.glob("*_done.json")),
                )
            )

    rows.sort(key=lambda row: row.age_s if row.age_s is not None else float("inf"))
    return tuple(rows), log_state


def walk_activity(
    session_dir: Path,
    *,
    now_unix: float,
    limit: int = DEFAULT_ACTIVITY_LIMIT,
    roots: Iterable[str] = ACTIVITY_ROOTS,
    max_files: int | None = None,
    max_dirs: int | None = None,
    deadline_s: float | None = None,
) -> tuple[tuple[ActivityEntry, ...], bool]:
    """Return the most recently written files under the whitelisted subtrees.

    Args:
        session_dir: Absolute session root.
        now_unix: Observation time.
        limit: How many entries to return.
        roots: Subdirectory names to walk.
        max_files: Stat budget; defaults to :data:`MAX_FILES`.
        max_dirs: Directory budget; defaults to :data:`MAX_DIRS`.
        deadline_s: Wall-clock budget; defaults to :data:`MAX_WALK_SECONDS`.

    Returns:
        ``(entries, truncated)`` — newest first. ``truncated`` is ``True`` when
        any budget was exhausted, so the caller can say the view is partial
        instead of implying it is complete.
    """
    # Resolved at call time, not bound as argument defaults, so the module
    # constants remain a single source of truth that tests can vary.
    file_budget = MAX_FILES if max_files is None else max_files
    dir_budget = MAX_DIRS if max_dirs is None else max_dirs
    time_budget = MAX_WALK_SECONDS if deadline_s is None else deadline_s

    started = time.monotonic()
    newest: list[tuple[float, str, int]] = []
    files_seen = 0
    dirs_seen = 0
    truncated = False

    def over_budget() -> bool:
        return dirs_seen >= dir_budget or files_seen >= file_budget or (time.monotonic() - started) > time_budget

    for root_name in roots:
        root = session_dir / root_name
        if not root.is_dir():
            continue
        stack = [root]
        while stack and not truncated:
            if over_budget():
                truncated = True
                break
            current = stack.pop()
            dirs_seen += 1
            try:
                entries = list(current.iterdir())
            except OSError:
                # Permission denied or a directory removed mid-walk. Neither is
                # worth failing the whole source over.
                continue
            for entry in entries:
                # Checked per file, not only per directory: a single directory
                # holding more than the budget would otherwise blow straight
                # through it and report the result as complete.
                if over_budget():
                    truncated = True
                    break
                try:
                    if entry.is_dir():
                        if entry.name not in PRUNE_DIRS:
                            stack.append(entry)
                        continue
                    if entry.name.endswith(PRUNE_SUFFIXES):
                        continue
                    stat = entry.stat()
                except OSError:
                    continue
                files_seen += 1
                try:
                    rel = str(entry.relative_to(session_dir))
                except ValueError:  # pragma: no cover - entry is always under the root
                    rel = str(entry)
                newest.append((stat.st_mtime, rel, int(stat.st_size)))
        if truncated:
            break

    newest.sort(key=lambda row: row[0], reverse=True)
    entries_out = tuple(
        ActivityEntry(relpath=rel, age_s=max(0.0, now_unix - mtime), size_bytes=size)
        for mtime, rel, size in newest[: max(0, limit)]
    )
    return entries_out, truncated


class ActivitySource:
    """Reads sub-phase activity: live run directories plus recent writes."""

    name = "activity"

    def __init__(self, *, limit: int = DEFAULT_ACTIVITY_LIMIT) -> None:
        """Initialise the source.

        Args:
            limit: How many recent-write entries to surface.
        """
        self._limit = limit
        # Retained across polls so log growth rate can be derived. Bounded by
        # the number of run directories, which is small.
        self._log_state: dict[str, tuple[float, int]] = {}

    def read(self, session_dir: Path, *, now_unix: float | None = None) -> SourceResult:
        """Probe run directories and walk for recent writes.

        Args:
            session_dir: Absolute session root.
            now_unix: Observation time; defaults to the wall clock.

        Returns:
            ``OK`` with ``{"running_work", "activity", "truncated",
            "last_activity_age_s"}``, or ``ABSENT`` when the session has no
            walkable subtrees yet.
        """
        now = float(now_unix if now_unix is not None else time.time())
        try:
            if not session_dir.is_dir():
                return SourceResult.absent()
            work, self._log_state = probe_runs(session_dir, now_unix=now, previous=self._log_state)
            activity, truncated = walk_activity(session_dir, now_unix=now, limit=self._limit)
        except OSError as exc:
            return SourceResult.error(str(exc))

        if not work and not activity:
            return SourceResult.absent()

        ages = [entry.age_s for entry in activity]
        ages.extend(row.age_s for row in work if row.age_s is not None)
        return SourceResult.hit(
            {
                "running_work": work,
                "activity": activity,
                "truncated": truncated,
                "last_activity_age_s": min(ages) if ages else None,
            }
        )


def freshest_activity_age_s(data: dict[str, Any] | None) -> float | None:
    """Extract the freshest activity age from an :class:`ActivitySource` payload.

    Used by liveness derivation: any recent write anywhere in the session tree
    is direct evidence the run is alive, and unlike a recorded pid it cannot be
    confounded by PID namespaces or pid reuse.

    Args:
        data: The source payload, or ``None``.

    Returns:
        Seconds since the most recent write, or ``None`` when unknown.
    """
    if not isinstance(data, dict):
        return None
    return to_float(data.get("last_activity_age_s"))
