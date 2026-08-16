# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Renderers for :class:`~..model.Snapshot`.

Every renderer returns a string and never prints, so each is unit-testable and
identical across surfaces. Printing is the CLI's job.
"""

from __future__ import annotations

from .format import Style, detect_style
from .json_out import SCHEMA_VERSION, render_json, to_dict
from .text import render_status


__all__ = [
    "SCHEMA_VERSION",
    "Style",
    "detect_style",
    "render_json",
    "render_status",
    "to_dict",
]
