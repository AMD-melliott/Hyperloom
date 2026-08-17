# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Read-only sources for the visibility layer.

Each source reads one artifact of a session directory, reports its own
degradation, and never raises. See :mod:`.base` for the contract.
"""

from __future__ import annotations

from .activity import ActivitySource, freshest_activity_age_s
from .base import Source, SourceResult, warning_for
from .coordinator_db import CoordinatorDbSource
from .current_step import CurrentStepSource
from .geak import GeakSource, describe as describe_geak
from .gpu import GpuSource, resolve_amd_smi
from .lockfile import LockFileSource, heartbeat_age_s, pid_alive
from .manifest import ManifestSource
from .server import ServerMetricsSource
from .state_file import StateFileSource


__all__ = [
    "ActivitySource",
    "CoordinatorDbSource",
    "CurrentStepSource",
    "GeakSource",
    "GpuSource",
    "LockFileSource",
    "ManifestSource",
    "ServerMetricsSource",
    "Source",
    "SourceResult",
    "StateFileSource",
    "describe_geak",
    "freshest_activity_age_s",
    "heartbeat_age_s",
    "pid_alive",
    "resolve_amd_smi",
    "warning_for",
]
