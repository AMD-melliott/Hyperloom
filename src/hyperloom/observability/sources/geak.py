# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""GEAK end-to-end kernel run progress, read from its output tree.

The KERNEL_AGENT phase shells out to ``geak_runner.py``, which writes
``handoff.json`` on the way in and ``result.json`` on the way out and nothing
whatsoever in between — its child's stdout goes to a pipe the parent does not
drain until exit. For a run capped at eight hours that is a very long silence.

What GEAK *does* leave behind is a directory layout that encodes its own
progress::

    geak/e2e_cycle0/kernels/_exp/team_<task>_<stamp>/<task>/round_1/engineer_0/verify/

so round number, engineer count and current stage are all recoverable by
reading directory names. That makes this source cheap, but also brittle to a
GEAK reorganisation — which is why it is explicitly optional. It returns
``ABSENT`` the moment the layout stops matching, and the generic activity walk
in :mod:`~hyperloom.observability.sources.activity` continues to report the
same work as raw paths. Detail degrades; correctness does not.
"""

from __future__ import annotations

import re
from pathlib import Path

from ..model import GeakProgress
from .base import SourceResult


_CYCLE_DIR = re.compile(r"^e2e_cycle(\d+)$")
_ROUND_DIR = re.compile(r"^round_(\d+)$")
_ENGINEER_DIR = re.compile(r"^engineer_\d+$")

# Stage subdirectories an engineer creates, in the order they are entered.
# Reported as "what this engineer is doing" when one is the freshest thing
# under its directory.
_STAGES: tuple[str, ...] = ("verify", "workspace")


def _newest_cycle(geak_root: Path) -> tuple[int | None, Path | None]:
    """Return the highest-numbered ``e2e_cycleN`` directory."""
    best: tuple[int, Path] | None = None
    try:
        entries = list(geak_root.iterdir())
    except OSError:
        return None, None
    for entry in entries:
        if not entry.is_dir():
            continue
        match = _CYCLE_DIR.match(entry.name)
        if match is None:
            continue
        number = int(match.group(1))
        if best is None or number > best[0]:
            best = (number, entry)
    return (best[0], best[1]) if best else (None, None)


def _newest_dir(parent: Path, pattern: re.Pattern[str] | None = None) -> Path | None:
    """Return the most recently modified child directory, optionally filtered."""
    best: tuple[float, Path] | None = None
    try:
        entries = list(parent.iterdir())
    except OSError:
        return None
    for entry in entries:
        if not entry.is_dir():
            continue
        if pattern is not None and pattern.match(entry.name) is None:
            continue
        try:
            mtime = entry.stat().st_mtime
        except OSError:
            continue
        if best is None or mtime > best[0]:
            best = (mtime, entry)
    return best[1] if best else None


def _subtree_mtime(path: Path, *, budget: int = 200) -> float:
    """Return the newest mtime under ``path``, bounded by a stat budget."""
    newest = 0.0
    stack = [path]
    seen = 0
    while stack and seen < budget:
        current = stack.pop()
        try:
            entries = list(current.iterdir())
        except OSError:
            continue
        for entry in entries:
            seen += 1
            if seen >= budget:
                break
            try:
                stat = entry.stat()
            except OSError:
                continue
            newest = max(newest, stat.st_mtime)
            if entry.is_dir() and entry.name not in {".git", "__pycache__"}:
                stack.append(entry)
    return newest


class GeakSource:
    """Reads GEAK's output tree into a structured progress row."""

    name = "geak"

    def read(self, session_dir: Path, *, now_unix: float | None = None) -> SourceResult:
        """Derive round/engineer progress for the newest GEAK cycle.

        Args:
            session_dir: Absolute session root.
            now_unix: Unused; ages come from the activity source.

        Returns:
            ``OK`` with a :class:`~hyperloom.observability.model.GeakProgress`,
            or ``ABSENT`` when no GEAK tree exists or the layout no longer
            matches.
        """
        del now_unix
        geak_root = session_dir / "geak"
        if not geak_root.is_dir():
            return SourceResult.absent()

        try:
            cycle, cycle_dir = _newest_cycle(geak_root)
            if cycle_dir is None:
                # ``handoff.json`` written but no cycle directory yet: GEAK has
                # been launched and is still setting up.
                if (geak_root / "handoff.json").is_file():
                    return SourceResult.hit(GeakProgress(has_result=(geak_root / "result.json").is_file()))
                return SourceResult.absent()

            has_result = (geak_root / "result.json").is_file()
            exp_root = cycle_dir / "kernels" / "_exp"
            team_dir = _newest_dir(exp_root) if exp_root.is_dir() else None
            if team_dir is None:
                return SourceResult.hit(GeakProgress(cycle=cycle, has_result=has_result))

            task_dir = _newest_dir(team_dir)
            if task_dir is None:
                return SourceResult.hit(GeakProgress(cycle=cycle, has_result=has_result))

            round_dir = _newest_dir(task_dir, _ROUND_DIR)
            if round_dir is None:
                return SourceResult.hit(
                    GeakProgress(cycle=cycle, task=task_dir.name, has_result=has_result),
                )

            round_match = _ROUND_DIR.match(round_dir.name)
            round_no = int(round_match.group(1)) if round_match else None

            engineers: list[str] = []
            try:
                engineers = sorted(
                    entry.name for entry in round_dir.iterdir() if entry.is_dir() and _ENGINEER_DIR.match(entry.name)
                )
            except OSError:
                engineers = []

            active_engineer, active_stage = self._active(round_dir, engineers)

            return SourceResult.hit(
                GeakProgress(
                    cycle=cycle,
                    task=task_dir.name,
                    round_no=round_no,
                    engineers=tuple(engineers),
                    active_engineer=active_engineer,
                    active_stage=active_stage,
                    has_result=has_result,
                )
            )
        except OSError as exc:
            return SourceResult.error(str(exc))

    def _active(self, round_dir: Path, engineers: list[str]) -> tuple[str | None, str | None]:
        """Identify which engineer wrote most recently, and into which stage.

        Args:
            round_dir: The current ``round_N`` directory.
            engineers: Engineer directory names in the round.

        Returns:
            ``(engineer_name, stage_name)``, either of which may be ``None``.
        """
        best: tuple[float, str, str | None] | None = None
        for name in engineers:
            engineer_dir = round_dir / name
            for stage in _STAGES:
                stage_dir = engineer_dir / stage
                if not stage_dir.is_dir():
                    continue
                mtime = _subtree_mtime(stage_dir)
                if mtime and (best is None or mtime > best[0]):
                    best = (mtime, name, stage)
            if best is None or best[1] != name:
                try:
                    mtime = engineer_dir.stat().st_mtime
                except OSError:
                    continue
                if best is None or mtime > best[0]:
                    best = (mtime, name, None)
        if best is None:
            return None, None
        return best[1], best[2]


def describe(progress: GeakProgress) -> str:
    """Render a GEAK progress row as one operator-facing line.

    Args:
        progress: The parsed progress.

    Returns:
        e.g. ``"e2e_cycle0 · round 1 · 3 engineers · engineer_0 verify"``.
    """
    parts: list[str] = []
    if progress.cycle is not None:
        parts.append(f"e2e_cycle{progress.cycle}")
    if progress.task:
        parts.append(progress.task)
    if progress.round_no is not None:
        parts.append(f"round {progress.round_no}")
    if progress.engineers:
        count = len(progress.engineers)
        parts.append(f"{count} engineer{'s' if count != 1 else ''}")
    if progress.active_engineer:
        stage = f" {progress.active_stage}" if progress.active_stage else ""
        parts.append(f"{progress.active_engineer}{stage}")
    if progress.has_result:
        parts.append("result written")
    return " · ".join(parts)
