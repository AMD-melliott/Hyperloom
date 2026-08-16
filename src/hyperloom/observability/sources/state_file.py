# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""``state.json`` source.

Reads the raw dict rather than round-tripping through
:class:`~hyperloom.orchestrator.state.shared_state.SharedState`. Two reasons:
the typed loader pulls in the whole orchestrator for what is a read-only view,
and it drops keys it does not recognize — so a state written by a newer
Hyperloom would lose fields on the way to the display. The raw dict tolerates
schema drift in both directions, which is what a status tool wants.

The Coordinator rewrites this file atomically (temp file + ``os.replace``) many
times per tick, so a reader either sees the previous complete document or the
next one. Torn reads are not possible; a stale read is, which is what
``state_mtime_unix`` exists to expose.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from hyperloom.common.jsonio import read_json
from hyperloom.inference_optimizer.session.session_paths import state_path

from .base import SourceResult


class StateFileSource:
    """Reads ``<session_dir>/state.json``."""

    name = "state"

    def read(self, session_dir: Path) -> SourceResult:
        """Read and parse ``state.json``.

        Args:
            session_dir: Absolute session root.

        Returns:
            :class:`~.base.SourceResult` carrying ``{"state": dict, "mtime_unix": float}``.
        """
        path = state_path(session_dir)
        if not path.is_file():
            return SourceResult.absent()
        errors: list[BaseException] = []
        data: Any = read_json(path, default=None, require_dict=True, on_error=errors.append)
        if data is None:
            if errors:
                return SourceResult.error(str(errors[0]))
            return SourceResult.error("state.json is not a JSON object")
        try:
            mtime = path.stat().st_mtime
        except OSError:
            mtime = 0.0
        return SourceResult.hit({"state": data, "mtime_unix": float(mtime)})
