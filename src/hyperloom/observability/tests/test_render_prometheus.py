# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Contract tests for the Prometheus renderer."""

from __future__ import annotations

import dataclasses
import math
import re
from pathlib import Path

import pytest

from hyperloom.observability import load_snapshot
from hyperloom.observability.model import CurrentStep, Liveness, SourceHealth, SourceOutcome
from hyperloom.observability.render import prometheus as prom
from hyperloom.observability.render.prometheus import (
    ExporterInfo,
    MetricFamily,
    escape_label_value,
    format_families,
    format_value,
    render_prometheus,
)


def test_escape_label_value() -> None:
    assert escape_label_value('a"b\\c\nd/e') == 'a\\"b\\\\c\\nd/e'


def test_format_value_specials() -> None:
    assert format_value(1.0) == "1"
    assert format_value(-3600.0) == "-3600"
    assert format_value(0.25) == "0.25"
    assert format_value(math.nan) == "NaN"
    assert format_value(math.inf) == "+Inf"
    assert format_value(-math.inf) == "-Inf"
    assert format_value(1e20) == "1e+20"


def test_none_samples_are_dropped_and_empty_families_skipped() -> None:
    kept = MetricFamily("hyperloom_a", "gauge", "Kept.")
    kept.add(None, phase="X")
    kept.add(2, phase="Y")
    empty = MetricFamily("hyperloom_b", "gauge", "Empty.")
    empty.add(None)

    text = format_families([kept, empty], const_labels={"session_id": "s"})

    assert text == ('# HELP hyperloom_a Kept.\n# TYPE hyperloom_a gauge\nhyperloom_a{session_id="s",phase="Y"} 2\n')


def test_unlabelled_sample_has_no_braces() -> None:
    family = MetricFamily("hyperloom_c", "counter", "Count.")
    family.add(True)
    assert format_families([family], const_labels={}).splitlines()[-1] == "hyperloom_c 1"


PINNED_METRIC_NAMES = frozenset(
    {
        "hyperloom_session_info",
        "hyperloom_session_observed",
        "hyperloom_liveness",
        "hyperloom_state_age_seconds",
        "hyperloom_last_activity_age_seconds",
        "hyperloom_session_start_timestamp_seconds",
        "hyperloom_session_elapsed_seconds",
        "hyperloom_session_remaining_seconds",
        "hyperloom_macro_cycle",
        "hyperloom_tick",
        "hyperloom_phase_current",
        "hyperloom_phase_elapsed_seconds",
        "hyperloom_phase_budget_seconds",
        "hyperloom_phase_budget_remaining_seconds",
        "hyperloom_phase_cap_seconds",
        "hyperloom_tasks",
        "hyperloom_lane_held",
        "hyperloom_lane_capacity",
        "hyperloom_gpu_leased",
        "hyperloom_current_step_info",
        "hyperloom_current_step_start_timestamp_seconds",
        "hyperloom_current_step_deadline_timestamp_seconds",
        "hyperloom_throughput_baseline",
        "hyperloom_throughput_best",
        "hyperloom_gain_percent",
        "hyperloom_target_gap_percent",
        "hyperloom_crashes",
        "hyperloom_stop_info",
        "hyperloom_exporter_build_info",
        "hyperloom_exporter_parent_alive",
        "hyperloom_exporter_render_errors_total",
        "hyperloom_source_up",
        "hyperloom_source_age_seconds",
        "hyperloom_source_collect_duration_seconds",
        "hyperloom_source_consecutive_failures",
    }
)

ALLOWED_LABELS = frozenset(
    {
        "session_id",
        "model",
        "framework",
        "framework_version",
        "gpu_type",
        "precision",
        "tp",
        "ep",
        "conc",
        "isl",
        "osl",
        "objective_kind",
        "state",
        "phase",
        "lane",
        "gpu_id",
        "step",
        "kind",
        "reason",
        "source",
        "version",
    }
)
KNOWN_PHASES = frozenset({"PRELUDE", "ENABLEMENT", "FRAMEWORK_AGENT", "KERNEL_AGENT", "SWEEP", "CLOSE"})

_SAMPLE = re.compile(r"^(?P<name>[a-zA-Z_:][a-zA-Z0-9_:]*)(?:\{(?P<labels>.*)\})? (?P<value>\S+)$")
_LABEL = re.compile(r'(?P<key>[a-zA-Z_][a-zA-Z0-9_]*)="(?P<val>(?:[^"\\]|\\.)*)"')

EXPORTER = ExporterInfo(version="9.9.9", parent_alive=True)


def parse_exposition(text: str) -> tuple[dict[str, str], list[tuple[str, dict[str, str], str]]]:
    """Return ``({family: type}, [(name, labels, raw_value)])`` and assert the format is well-formed."""
    types: dict[str, str] = {}
    samples: list[tuple[str, dict[str, str], str]] = []
    helped: set[str] = set()
    for line in text.splitlines():
        if line.startswith("# HELP "):
            name = line.split()[2]
            assert name not in helped, f"duplicate HELP for {name}"
            helped.add(name)
        elif line.startswith("# TYPE "):
            _, _, name, kind = line.split()
            assert name in helped, f"TYPE before HELP for {name}"
            assert kind in {"gauge", "counter"}
            types[name] = kind
        else:
            match = _SAMPLE.match(line)
            assert match, f"malformed sample line: {line!r}"
            labels = {m["key"]: m["val"] for m in _LABEL.finditer(match["labels"] or "")}
            assert match["name"] in types, f"sample before TYPE: {line!r}"
            samples.append((match["name"], labels, match["value"]))
    return types, samples


def value_of(samples, name: str, **labels: str) -> float:
    hits = [v for n, lab, v in samples if n == name and all(lab.get(k) == w for k, w in labels.items())]
    assert len(hits) == 1, f"{name}{labels}: expected one sample, got {hits}"
    return float(hits[0])


def names_of(samples) -> set[str]:
    return {name for name, _, _ in samples}


@pytest.fixture
def snapshot(session_dir: Path, frozen_clock):
    snap = load_snapshot(session_dir, now_unix=frozen_clock)
    assert snap is not None
    return snap


@pytest.fixture
def full_snapshot(snapshot):
    """The fixture snapshot with every optional field populated."""
    return dataclasses.replace(
        snapshot,
        liveness=Liveness.LIVE,
        last_activity_age_s=5.0,
        current_step=CurrentStep(
            phase="KERNEL_AGENT",
            step="geak_e2e",
            detail="GEAK e2e (from=FRAMEWORK_AGENT)",
            started_unix=1785988000.0,
            deadline_unix=1786019000.0,
        ),
        result=dataclasses.replace(snapshot.result, stop_reason="time_exhausted"),
        source_health=(
            SourceHealth(name="session", outcome=SourceOutcome.OK, age_s=1.5, duration_s=0.02),
            SourceHealth(
                name="inference_sd", outcome=SourceOutcome.ERROR, age_s=30.0, error="x", consecutive_failures=2
            ),
        ),
    )


def test_every_pinned_metric_is_rendered_for_a_full_snapshot(full_snapshot) -> None:
    types, samples = parse_exposition(render_prometheus(full_snapshot, exporter=EXPORTER))
    assert names_of(samples) == PINNED_METRIC_NAMES
    assert types["hyperloom_exporter_render_errors_total"] == "counter"
    assert {kind for name, kind in types.items() if name != "hyperloom_exporter_render_errors_total"} == {"gauge"}


def test_values_follow_the_snapshot(snapshot) -> None:
    _, samples = parse_exposition(render_prometheus(snapshot, exporter=EXPORTER))
    const = {"session_id": "test-model_20260806T000000Z_deadbeef", "model": "test-model", "framework": "sglang"}

    for _, labels, _ in samples:
        assert {k: labels[k] for k in const} == const

    assert value_of(samples, "hyperloom_phase_current", phase="FRAMEWORK_AGENT") == 1
    assert value_of(samples, "hyperloom_phase_current", phase="PRELUDE") == 0
    assert value_of(samples, "hyperloom_phase_elapsed_seconds", phase="FRAMEWORK_AGENT") == 7200
    assert value_of(samples, "hyperloom_phase_cap_seconds", phase="FRAMEWORK_AGENT") == 19440
    assert value_of(samples, "hyperloom_tasks", state="running") == 1
    assert value_of(samples, "hyperloom_tasks", state="cancelled") == 0
    assert value_of(samples, "hyperloom_lane_held", lane="benchmark_lane") == 1
    assert value_of(samples, "hyperloom_lane_capacity", lane="research_lane") == 4
    assert value_of(samples, "hyperloom_gpu_leased", gpu_id="0") == 1
    assert value_of(samples, "hyperloom_throughput_baseline") == 100
    assert value_of(samples, "hyperloom_throughput_best") == 115
    assert value_of(samples, "hyperloom_gain_percent", kind="raw") == 15
    assert value_of(samples, "hyperloom_gain_percent", kind="validated") == 12
    assert value_of(samples, "hyperloom_liveness", state="unknown") == 1
    assert value_of(samples, "hyperloom_liveness", state="live") == 0
    assert value_of(samples, "hyperloom_session_start_timestamp_seconds") == 1785974400
    assert value_of(samples, "hyperloom_session_info", gpu_type="mi300x", tp="8", precision="fp8") == 1
    assert value_of(samples, "hyperloom_session_observed") == 1
    assert value_of(samples, "hyperloom_exporter_parent_alive") == 1
    assert value_of(samples, "hyperloom_exporter_build_info", version="9.9.9") == 1


def test_none_is_omitted_not_zeroed(snapshot) -> None:
    _, samples = parse_exposition(render_prometheus(snapshot, exporter=ExporterInfo(version="1")))
    budget_phases = {lab["phase"] for n, lab, _ in samples if n == "hyperloom_phase_budget_seconds"}
    assert "PRELUDE" not in budget_phases
    assert "FRAMEWORK_AGENT" in budget_phases
    assert "hyperloom_last_activity_age_seconds" not in names_of(samples)
    assert "hyperloom_current_step_info" not in names_of(samples)
    assert "hyperloom_stop_info" not in names_of(samples)
    assert "hyperloom_exporter_parent_alive" not in names_of(samples)


def test_labels_stay_inside_the_allow_list(full_snapshot) -> None:
    _, samples = parse_exposition(render_prometheus(full_snapshot, exporter=EXPORTER))
    for name, labels, _ in samples:
        assert set(labels) <= ALLOWED_LABELS, f"{name} carries unexpected labels {set(labels) - ALLOWED_LABELS}"
        if "phase" in labels:
            assert labels["phase"] in KNOWN_PHASES
        if name == "hyperloom_tasks":
            assert labels["state"] in {"queued", "running", "succeeded", "failed", "cancelled"}
        if name == "hyperloom_liveness":
            assert labels["state"] in {"live", "stale", "dead", "unknown"}
    assert not any("detail" in labels or "holder" in labels for _, labels, _ in samples)


def test_label_values_are_escaped(snapshot) -> None:
    odd = dataclasses.replace(snapshot, session=dataclasses.replace(snapshot.session, model_display='Qwen/"x"\\y\nz'))
    text = render_prometheus(odd, exporter=EXPORTER)
    assert 'model="Qwen/\\"x\\"\\\\y\\nz"' in text
    parse_exposition(text)


def test_negative_and_nan_values_render(snapshot) -> None:
    overrun = dataclasses.replace(
        snapshot,
        session_remaining_s=-120.0,
        result=dataclasses.replace(snapshot.result, best_tput=float("nan")),
    )
    _, samples = parse_exposition(render_prometheus(overrun, exporter=EXPORTER))
    assert value_of(samples, "hyperloom_session_remaining_seconds") == -120
    assert [v for n, _, v in samples if n == "hyperloom_throughput_best"] == ["NaN"]


def test_source_up_is_zero_only_on_error(full_snapshot) -> None:
    _, samples = parse_exposition(render_prometheus(full_snapshot, exporter=EXPORTER))
    assert value_of(samples, "hyperloom_source_up", source="session") == 1
    assert value_of(samples, "hyperloom_source_up", source="inference_sd") == 0
    assert value_of(samples, "hyperloom_source_consecutive_failures", source="inference_sd") == 2


def test_absent_source_counts_as_up(snapshot) -> None:
    _, samples = parse_exposition(render_prometheus(snapshot, exporter=EXPORTER))
    assert value_of(samples, "hyperloom_source_up", source="lock") == 1


def test_minimal_page_without_a_snapshot() -> None:
    text = render_prometheus(None, exporter=ExporterInfo(version="1.2.3", parent_alive=False))
    _, samples = parse_exposition(text)
    assert names_of(samples) == {
        "hyperloom_exporter_build_info",
        "hyperloom_exporter_parent_alive",
        "hyperloom_exporter_render_errors_total",
        "hyperloom_session_observed",
    }
    assert value_of(samples, "hyperloom_session_observed") == 0
    assert value_of(samples, "hyperloom_exporter_parent_alive") == 0
    assert "session_id=" not in text


def test_a_broken_family_does_not_blank_the_page(snapshot, monkeypatch) -> None:
    def boom(_snapshot):
        raise RuntimeError("bug in one family")

    builders = tuple((name, boom if name == "work" else fn) for name, fn in prom._SNAPSHOT_BUILDERS)
    monkeypatch.setattr(prom, "_SNAPSHOT_BUILDERS", builders)
    failed: list[str] = []

    text = render_prometheus(
        snapshot, exporter=ExporterInfo(version="1", render_errors_total=3), on_family_error=failed.append
    )

    _, samples = parse_exposition(text)
    assert failed == ["work"]
    assert "hyperloom_tasks" not in names_of(samples)
    assert "hyperloom_phase_current" in names_of(samples)
    assert value_of(samples, "hyperloom_exporter_render_errors_total") == 4
