# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The shipped dashboard and scrape configs must track the real metric names."""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from .test_render_prometheus import PINNED_METRIC_NAMES

EXAMPLES = Path(__file__).resolve().parents[4] / "examples" / "observability"


def _exprs(panel_or_dashboard: dict) -> list[str]:
    found = [t["expr"] for t in panel_or_dashboard.get("targets", []) if "expr" in t]
    for child in panel_or_dashboard.get("panels", []):
        found.extend(_exprs(child))
    return found


@pytest.mark.parametrize("name", ["hyperloom.json", "hyperloom-overview.json"])
def test_dashboard_only_uses_real_hyperloom_metrics(name: str) -> None:
    dashboard = json.loads((EXAMPLES / "dashboards" / name).read_text())
    exprs = _exprs(dashboard)
    assert exprs, "dashboard has no queries"
    used = {name for expr in exprs for name in re.findall(r"\bhyperloom_[a-z_]+\b", expr)}
    assert used, "dashboard queries no hyperloom metrics"
    assert used <= PINNED_METRIC_NAMES, f"dashboard references unknown metrics: {used - PINNED_METRIC_NAMES}"
    variable = json.dumps(dashboard["templating"])
    assert "hyperloom_session_" in variable


def test_scrape_configs_point_at_the_exporter_endpoints() -> None:
    compose = (EXAMPLES / "compose" / "prometheus.yml").read_text()
    k8s = (EXAMPLES / "k8s" / "scrapeconfig.yaml").read_text()
    for text in (compose, k8s):
        assert ":9477" in text
        assert "/sd/inference" in text
