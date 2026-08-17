# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""``runtime/current_step.json`` — the producer's beacon for a blocked phase.

Written by
:mod:`hyperloom.inference_optimizer.session.current_step` immediately before a
phase makes a long blocking call, and removed afterwards. It is the one signal
that survives the phase machine going quiet, because it is written *before* the
loop stops ticking rather than during.

Read-only here, and strictly so: the beacon is removed by the producer on exit,
so a reader that recreated it would strand a phantom step on the display
forever.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from hyperloom.common.coerce import to_float
from hyperloom.common.jsonio import read_json

from ..model import CurrentStep
from .base import SourceResult


BEACON_RELPATH = ("runtime", "current_step.json")


def beacon_path(session_dir: Path) -> Path:
    """Return the beacon path for a session.

    Args:
        session_dir: Absolute session root.

    Returns:
        ``<session_dir>/runtime/current_step.json``.
    """
    return session_dir.joinpath(*BEACON_RELPATH)


def parse(payload: Any) -> CurrentStep | None:
    """Build a :class:`~hyperloom.observability.model.CurrentStep` from raw JSON.

    Args:
        payload: The decoded beacon document.

    Returns:
        The parsed step, or ``None`` when the document lacks the two fields
        that make it meaningful.
    """
    if not isinstance(payload, dict):
        return None
    phase = str(payload.get("phase") or "").strip()
    step = str(payload.get("step") or "").strip()
    if not phase or not step:
        return None

    raw_artifacts = payload.get("artifacts")
    artifacts: tuple[tuple[str, str], ...] = ()
    if isinstance(raw_artifacts, dict):
        artifacts = tuple((str(key), str(value)) for key, value in sorted(raw_artifacts.items()))

    return CurrentStep(
        phase=phase.upper(),
        step=step,
        detail=(str(payload.get("detail")).strip() or None) if payload.get("detail") else None,
        started_unix=to_float(payload.get("started_unix")),
        deadline_unix=to_float(payload.get("deadline_unix")),
        artifacts=artifacts,
    )


class CurrentStepSource:
    """Reads the producer's current-step beacon."""

    name = "current_step"

    def read(self, session_dir: Path, *, now_unix: float | None = None) -> SourceResult:
        """Read the beacon.

        Args:
            session_dir: Absolute session root.
            now_unix: Unused; the beacon carries its own timestamps.

        Returns:
            ``OK`` with the parsed step, or ``ABSENT`` when no step is in
            flight — which is the normal state for most of a run.
        """
        del now_unix
        path = beacon_path(session_dir)
        if not path.is_file():
            return SourceResult.absent()
        try:
            payload = read_json(path)
        except Exception as exc:  # noqa: BLE001 - a torn read races the producer's removal
            return SourceResult.error(f"unreadable beacon: {exc}")
        step = parse(payload)
        if step is None:
            return SourceResult.error("beacon present but missing phase/step")
        return SourceResult.hit(step)
