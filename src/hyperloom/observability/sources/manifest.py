# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""``manifest.json`` source.

The manifest is written once at session start and never updated, so it is the
authoritative record of what was *requested* — model, framework, topology,
workload shape, objective. ``state.json`` mirrors much of it but can be
mutated mid-run, so where the two disagree the manifest describes intent and
state describes reality.

Its absence is normal for a directory that is being created right now, and for
the workspace root when a caller passes one by mistake.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from hyperloom.common.jsonio import read_json
from hyperloom.inference_optimizer.session.session_paths import manifest_path

from .base import SourceResult


class ManifestSource:
    """Reads ``<session_dir>/manifest.json``."""

    name = "manifest"

    def read(self, session_dir: Path) -> SourceResult:
        """Read and parse ``manifest.json``.

        Args:
            session_dir: Absolute session root.

        Returns:
            :class:`~.base.SourceResult` carrying the manifest dict.
        """
        path = manifest_path(session_dir)
        if not path.is_file():
            return SourceResult.absent()
        errors: list[BaseException] = []
        data: Any = read_json(path, default=None, require_dict=True, on_error=errors.append)
        if data is None:
            if errors:
                return SourceResult.error(str(errors[0]))
            return SourceResult.error("manifest.json is not a JSON object")
        return SourceResult.hit(data)
