# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The installed Hyperloom version, read once from package metadata for every layer."""

from __future__ import annotations

from importlib.metadata import packages_distributions, version

UNINSTALLED_VERSION = "0.0.0.dev0"


def hyperloom_version() -> str:
    """Version of the distribution that ships the ``hyperloom`` package; the fallback for a bare source tree."""
    distributions = packages_distributions().get(__name__.partition(".")[0])
    return version(distributions[0]) if distributions else UNINSTALLED_VERSION
