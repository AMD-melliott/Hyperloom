# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Prometheus text exposition of a :class:`~..model.Snapshot`.

Metric names are a stable contract, pinned by
``tests/test_render_prometheus.py``: a dashboard or alert keyed on a name must
not break because a model field was renamed. Like :mod:`.json_out`, ``None`` is
never coerced to ``0``. A sample whose value is unknown is omitted, so a panel
shows "no data" rather than a plausible, wrong zero.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field


PROMETHEUS_CONTENT_TYPE = "text/plain; version=0.0.4; charset=utf-8"


@dataclass
class MetricFamily:
    """One ``# HELP`` / ``# TYPE`` block and its samples."""

    name: str
    kind: str
    help: str
    samples: list[tuple[dict[str, str], float]] = field(default_factory=list)

    def add(self, value: float | int | bool | None, **labels: object) -> None:
        """Append a sample; a ``None`` value is dropped rather than zeroed."""
        if value is None:
            return
        self.samples.append(({key: "" if val is None else str(val) for key, val in labels.items()}, float(value)))


def escape_label_value(value: str) -> str:
    """Escape a label value per the exposition format."""
    return value.replace("\\", "\\\\").replace("\n", "\\n").replace('"', '\\"')


def format_value(value: float) -> str:
    """Render a sample value, including the format's NaN/Inf spellings."""
    if math.isnan(value):
        return "NaN"
    if math.isinf(value):
        return "+Inf" if value > 0 else "-Inf"
    if value.is_integer() and abs(value) < 1e15:
        return str(int(value))
    return repr(value)


def format_families(families: Iterable[MetricFamily], *, const_labels: Mapping[str, str]) -> str:
    """Serialise families, prefixing every sample's labels with ``const_labels``."""
    lines: list[str] = []
    for family in families:
        if not family.samples:
            continue
        lines.append(f"# HELP {family.name} {family.help}")
        lines.append(f"# TYPE {family.name} {family.kind}")
        for labels, value in family.samples:
            merged = {**const_labels, **labels}
            if merged:
                body = ",".join(f'{key}="{escape_label_value(val)}"' for key, val in merged.items())
                lines.append(f"{family.name}{{{body}}} {format_value(value)}")
            else:
                lines.append(f"{family.name} {format_value(value)}")
    return "\n".join(lines) + "\n" if lines else ""
