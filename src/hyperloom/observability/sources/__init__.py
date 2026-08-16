# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Read-only sources for the visibility layer.

Each source reads one artifact of a session directory, reports its own
degradation, and never raises. See :mod:`.base` for the contract.
"""

from __future__ import annotations

from .base import Source, SourceResult, warning_for
from .coordinator_db import CoordinatorDbSource
from .lockfile import LockFileSource, heartbeat_age_s, pid_alive
from .manifest import ManifestSource
from .state_file import StateFileSource


__all__ = [
    "CoordinatorDbSource",
    "LockFileSource",
    "ManifestSource",
    "Source",
    "SourceResult",
    "StateFileSource",
    "heartbeat_age_s",
    "pid_alive",
    "warning_for",
]
