# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Renderer tests.

Golden-frame assertions are **substring** checks against a rendered buffer, not
exact-output snapshots: exact matches break on every cosmetic tweak and get
regenerated without being read, which makes them worse than no test at all.
What is asserted here is semantic content and the honesty rules.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from hyperloom.observability import Liveness, load_snapshot
from hyperloom.observability.model import (
    ActivityEntry,
    CurrentStep,
    GeakProgress,
    GpuMetric,
    GpuMetrics,
    PhaseProgress,
    ResultSummary,
    RunningWork,
    ServerMetrics,
    Snapshot,
    SourceHealth,
    SourceOutcome,
)
from hyperloom.observability.render import SCHEMA_VERSION, Style, render_json, render_status, to_dict
from hyperloom.observability.render.format import Style as FormatStyle
from hyperloom.observability.render.format import (
    bar,
    bytes_size,
    clock,
    detect_style,
    duration,
    number,
    percent,
    truncate,
)

from .conftest import write_lock, write_state


PLAIN = Style(color=False, unicode=True, width=100)


# --- formatting primitives ----------------------------------------------------


@pytest.mark.parametrize(
    ("seconds", "expected"),
    [
        (None, "—"),
        (0, "0s"),
        (42, "42s"),
        (59.9, "59s"),
        (60, "1m"),
        (1080, "18m"),
        (3600, "1h00m"),
        (15120, "4h12m"),
        (-5, "0s"),
        ("nonsense", "—"),
    ],
)
def test_duration_rendering(seconds, expected) -> None:
    """Durations render at human scale and never crash on bad input."""
    assert duration(seconds) == expected


@pytest.mark.parametrize(
    ("seconds", "expected"),
    [
        (None, "—"),
        (0, "00:00"),
        (42, "00:00"),
        (60, "00:01"),
        (2520, "00:42"),
        (48_660, "13:31"),
        # Hours are not wrapped at 24: this measures elapsed time against a
        # budget, not a time of day, so a week-long run reads 168:00.
        (604_800, "168:00"),
        (-5, "00:00"),
        ("nonsense", "—"),
    ],
)
def test_clock_rendering(seconds, expected) -> None:
    """``HH:MM`` timers are fixed width and never wrap the hour field."""
    assert clock(seconds) == expected


def test_clock_truncates_rather_than_rounds() -> None:
    """A timer must not display a minute the run has not finished spending."""
    assert clock(119) == "00:01"
    assert clock(3599) == "00:59"


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (None, "—"),
        (0, "0 B"),
        (302, "302 B"),
        (52_428, "51.2 KB"),
        (1_258_291, "1.2 MB"),
    ],
)
def test_bytes_size_rendering(value, expected) -> None:
    """Byte counts render at human scale for the activity block."""
    assert bytes_size(value) == expected


def test_none_never_renders_as_zero() -> None:
    """The core honesty rule: unmeasured is a dash, not a plausible number."""
    assert percent(None) == "—"
    assert number(None) == "—"
    assert duration(None) == "—"
    assert clock(None) == "—"
    assert bytes_size(None) == "—"
    assert percent(0.0) == "0.0%"  # a measured zero still renders as zero
    assert bytes_size(0) == "0 B"


def test_percent_signed() -> None:
    """Signed percentages carry an explicit ``+`` for gains."""
    assert percent(15.0, signed=True) == "+15.0%"
    assert percent(-3.0, signed=True) == "-3.0%"


def test_bar_clamps_overrun_but_caller_shows_true_pct() -> None:
    """A bar cannot exceed its track; the percentage carries the overrun."""
    assert bar(0.0, width=4, style=PLAIN) == "░░░░"
    assert bar(0.5, width=4, style=PLAIN) == "██░░"
    assert bar(1.0, width=4, style=PLAIN) == "████"
    assert bar(2.5, width=4, style=PLAIN) == "████"
    assert bar(None, width=4, style=PLAIN) == "░░░░"


def test_bar_ascii_fallback() -> None:
    """Non-unicode terminals still get a readable bar."""
    ascii_style = Style(color=False, unicode=False, width=80)
    assert bar(0.5, width=4, style=ascii_style) == "##.."
    assert ascii_style.dash == "-"


def test_truncate() -> None:
    """Truncation is hard-limited to the requested width."""
    assert truncate("abcdef", 10) == "abcdef"
    assert truncate("abcdef", 3) == "ab…"
    assert truncate("abcdef", 0) == ""
    assert len(truncate("x" * 200, 40)) == 40


# --- colour gating ------------------------------------------------------------


def test_no_color_env_disables_color(monkeypatch) -> None:
    """``NO_COLOR`` wins over TTY detection."""

    class _Tty:
        encoding = "utf-8"

        def isatty(self) -> bool:
            return True

    monkeypatch.setenv("NO_COLOR", "1")
    monkeypatch.delenv("FORCE_COLOR", raising=False)
    assert detect_style(_Tty(), width=80).color is False


def test_non_tty_disables_color(monkeypatch) -> None:
    """Redirected output stays free of escape codes."""

    class _Pipe:
        encoding = "utf-8"

        def isatty(self) -> bool:
            return False

    monkeypatch.delenv("NO_COLOR", raising=False)
    monkeypatch.delenv("FORCE_COLOR", raising=False)
    assert detect_style(_Pipe(), width=80).color is False


def test_plain_style_emits_no_escape_codes(session_dir: Path, frozen_clock) -> None:
    """A colourless render must contain no ANSI at all."""
    snapshot = load_snapshot(session_dir, now_unix=frozen_clock)
    assert snapshot is not None
    assert "\033[" not in render_status(snapshot, style=PLAIN)


def test_color_style_pads_before_painting(session_dir: Path, frozen_clock) -> None:
    """Colour must not shift column alignment.

    Padding a painted string counts the escape bytes as visible width. Asserting
    the *visible* text is identical with and without colour catches that.
    """
    import re

    snapshot = load_snapshot(session_dir, now_unix=frozen_clock)
    assert snapshot is not None

    plain = render_status(snapshot, style=Style(color=False, unicode=True, width=100))
    colored = render_status(snapshot, style=Style(color=True, unicode=True, width=100))
    stripped = re.sub(r"\033\[[0-9;]*m", "", colored)

    assert stripped == plain


# --- text renderer ------------------------------------------------------------


def test_render_includes_phase_and_workload(session_dir: Path, frozen_clock) -> None:
    """The header answers "what is this and what step is it on"."""
    snapshot = load_snapshot(session_dir, now_unix=frozen_clock)
    assert snapshot is not None
    out = render_status(snapshot, style=PLAIN)

    assert "HYPERLOOM" in out
    assert "test-model" in out
    assert "FRAMEWORK_AGENT" in out
    assert "TP=8" in out
    assert "sglang" in out
    # Labelled, not a bare "1024/1024": an unlabelled pair only reads to
    # someone who already knows which end is which.
    assert "ISL 1024" in out
    assert "OSL 1024" in out
    # ep=1 in the fixture: a dense model's expert parallelism is a constant and
    # would only take up room.
    assert "EP=" not in out


def test_expert_parallelism_appears_only_when_it_is_doing_something(tmp_path: Path, frozen_clock) -> None:
    """EP above 1 is real topology and belongs on the header."""
    from .conftest import write_manifest

    sd = tmp_path / "moe"
    write_state(sd)
    write_manifest(sd, ep=8)

    assert "EP=8" in render_status(load_snapshot(sd, now_unix=frozen_clock), style=PLAIN)


def test_header_shows_the_model_name_not_its_snapshot_sha(tmp_path: Path, frozen_clock) -> None:
    """The reported header regression: a 40-char sha where the model should be.

    Live output was ``HYPERLOOM  a4e59da52a7bc87ae7251dd5545c0dd437c44b68``,
    which identifies nothing. The repo id and the framework version are both
    available in the manifest and belong on that line.
    """
    from .conftest import write_manifest

    sd = tmp_path / "hf"
    write_state(sd)
    write_manifest(
        sd,
        framework="vllm",
        model_name="a4e59da52a7bc87ae7251dd5545c0dd437c44b68",
        model_path=(
            "/data/hf_home/hub/models--meta-models--Muse-Glimmer-30B/snapshots/a4e59da52a7bc87ae7251dd5545c0dd437c44b68"
        ),
        stack_fingerprint={"vllm": "0.27.2rc1.dev150+g311b3513a", "rocm": "7.2.3"},
    )

    out = render_status(load_snapshot(sd, now_unix=frozen_clock), style=PLAIN)
    header = out.splitlines()[0]

    assert "meta-models/Muse-Glimmer-30B" in header
    assert "vllm 0.27.2rc1.dev150+g311b3513a" in header
    assert "a4e59da52a7bc87ae" not in out


def test_a_long_stack_string_wraps_the_workload_instead_of_truncating_it(tmp_path: Path, frozen_clock) -> None:
    """Framework versions are long enough to push the workload off an 80-column line.

    Losing ISL/OSL/conc to a truncation would trade one missing fact for
    several, so the workload moves to its own line rather than being cut.
    """
    from .conftest import write_manifest

    sd = tmp_path / "wrap"
    write_state(sd)
    write_manifest(
        sd,
        framework="vllm",
        model_name="meta-models/Muse-Glimmer-30B",
        stack_fingerprint={"vllm": "0.27.2rc1.dev150+g311b3513a"},
    )

    narrow = render_status(load_snapshot(sd, now_unix=frozen_clock), style=FormatStyle(width=80)).splitlines()
    wide = render_status(load_snapshot(sd, now_unix=frozen_clock), style=FormatStyle(width=200)).splitlines()

    assert "conc=64" in narrow[1] and "conc=64" not in narrow[0]
    assert "conc=64" in wide[0], "a wide terminal should keep the header on one line"
    # Whichever way it lays out, nothing is lost.
    for lines in (narrow, wide):
        assert all(bit in "\n".join(lines[:2]) for bit in ("ISL 1024", "OSL 1024", "TP=8", "fp8"))


def test_header_survives_a_manifest_with_no_identity_at_all(tmp_path: Path, frozen_clock) -> None:
    """An empty session must still render a header rather than raising."""
    from .conftest import write_manifest

    sd = tmp_path / "anonymous"
    write_state(sd, model_name="", framework="", gpu_type="", tp=0, conc=0, isl=0, osl=0)
    write_manifest(sd, model_name="", framework="", gpu_type="", tp=0, ep=0, workload={})

    out = render_status(load_snapshot(sd, now_unix=frozen_clock), style=PLAIN)

    assert out.splitlines()[0].strip() == "HYPERLOOM  (unknown model)"


def test_render_flags_non_live_observation(tmp_path: Path) -> None:
    """A stale or ended session must say so, with the age of the evidence.

    Presenting a week-old snapshot as current is the failure this whole layer
    exists to prevent, so it is asserted rather than left to review.
    """
    sd = tmp_path / "ended"
    write_state(sd, stop_reason="target_reached", phase="CLOSE")

    snapshot = load_snapshot(sd, now_unix=lambda: 1785988800.0 + 86_400.0)
    assert snapshot is not None
    assert snapshot.liveness is Liveness.DEAD

    out = render_status(snapshot, style=PLAIN)
    assert "ended" in out
    assert "last tick" in out
    assert "stop_reason=target_reached" in out


def test_render_shows_warnings(tmp_path: Path, frozen_clock) -> None:
    """Source failures surface in the output rather than being swallowed."""
    sd = tmp_path / "corrupt"
    sd.mkdir(parents=True)
    (sd / "state.json").write_text("{broken", encoding="utf-8")

    snapshot = load_snapshot(sd, now_unix=frozen_clock)
    assert snapshot is not None
    assert "state:" in render_status(snapshot, style=PLAIN)


def test_render_marks_budget_overrun(frozen_clock) -> None:
    """A phase past its budget shows a percentage above 100."""
    snapshot = Snapshot(
        phase="KERNEL_AGENT",
        phases=(
            PhaseProgress(
                name="KERNEL_AGENT",
                index=2,
                is_current=True,
                has_run=True,
                elapsed_s=7200.0,
                budget_total_s=3600.0,
            ),
        ),
        result=ResultSummary(),
        _now_unix=frozen_clock,
    )
    assert "200% of budget" in render_status(snapshot, style=PLAIN)


def test_framework_agent_reports_whichever_limit_actually_binds(frozen_clock) -> None:
    """FRAMEWORK_AGENT's exit helper now checks budget *and* cap.

    ``exit_normal_optimize`` absorbed the retired EXPLORE phase's config-search
    arm, so it now exits on budget exhaustion (``optimize_phase_budget_exhausted``)
    before ever reaching the absolute cap (``optimize_budget_cap``). The display
    must bind on whichever limit is smaller, same as any other budget-exit phase.
    """
    snapshot = Snapshot(
        phase="FRAMEWORK_AGENT",
        phases=(
            PhaseProgress(
                name="FRAMEWORK_AGENT",
                index=1,
                is_current=True,
                has_run=True,
                elapsed_s=4.5 * 3600,
                budget_total_s=3.517 * 3600,
                cap_s=5.067 * 3600,
            ),
        ),
        result=ResultSummary(),
        _now_unix=frozen_clock,
    )
    row = next(line for line in render_status(snapshot, style=PLAIN).splitlines() if "FRAMEWORK_AGENT" in line)

    assert "128% of budget" in row
    assert "89% of cap" not in row
    # Both limits stay visible even though only the smaller one binds.
    assert "03:31" in row
    assert "05:04" in row


def test_a_budget_exit_phase_binds_on_whichever_limit_is_smaller(frozen_clock) -> None:
    """KERNEL_AGENT exits on the budget *or* the cap, so the smaller one governs."""
    from hyperloom.observability.model import BUDGET_EXIT_PHASES

    assert "KERNEL_AGENT" in BUDGET_EXIT_PHASES

    def kernel_agent(budget_s: float, cap_s: float) -> PhaseProgress:
        return PhaseProgress(
            name="KERNEL_AGENT",
            index=2,
            is_current=True,
            has_run=True,
            elapsed_s=3600.0,
            budget_total_s=budget_s,
            cap_s=cap_s,
        )

    budget_binds = kernel_agent(budget_s=2 * 3600, cap_s=10 * 3600)
    cap_binds = kernel_agent(budget_s=10 * 3600, cap_s=2 * 3600)

    assert budget_binds.limit_kind == "budget"
    assert budget_binds.pct_used == pytest.approx(0.5)
    assert cap_binds.limit_kind == "cap"
    assert cap_binds.pct_used == pytest.approx(0.5)
    # The charge-back ratio stays available for both, unchanged.
    assert cap_binds.pct_of_budget == pytest.approx(0.1)


def test_unenforced_caps_are_not_displayed(session_dir: Path, frozen_clock) -> None:
    """PRELUDE and CLOSE have a computable cap that no exit check consults.

    ``phase_cap_seconds`` answers for every phase, but only the four work
    phases have an ``exit_normal_*`` helper that acts on it. Rendering PRELUDE
    at "272% of cap" — as a live run did — points an operator at a limit
    nothing will ever enforce.
    """
    snapshot = load_snapshot(session_dir, now_unix=frozen_clock)
    assert snapshot is not None

    prelude = next(phase for phase in snapshot.phases if phase.name == "PRELUDE")
    # The upstream number is still carried; it is the display that withholds it.
    assert prelude.cap_s is not None
    assert prelude.limit_s is None
    assert prelude.pct_used is None

    rows = {
        line.split()[1]: line
        for line in render_status(snapshot, style=PLAIN).splitlines()
        if len(line.split()) > 1 and line.split()[1] in {"PRELUDE", "CLOSE", "FRAMEWORK_AGENT"}
    }
    assert "%" not in rows["PRELUDE"]
    assert "%" not in rows["CLOSE"]
    assert "%" in rows["FRAMEWORK_AGENT"], "an enforced phase must still report its usage"


def test_a_phase_that_has_not_started_reports_no_percentage(frozen_clock) -> None:
    """ "0% of cap" for an unstarted phase is noise, not information."""
    snapshot = Snapshot(
        phase="KERNEL_AGENT",
        phases=(
            PhaseProgress(
                name="SWEEP",
                index=4,
                is_current=False,
                has_run=False,
                elapsed_s=0.0,
                cap_s=3600.0,
            ),
        ),
        result=ResultSummary(),
        _now_unix=frozen_clock,
    )
    row = next(line for line in render_status(snapshot, style=PLAIN).splitlines() if "SWEEP" in line)

    assert "0%" not in row
    # The allotment ahead is still worth showing.
    assert "01:00" in row


def test_an_unbounded_phase_reports_no_percentage(frozen_clock) -> None:
    """With neither limit set there is nothing to be a percentage of."""
    phase = PhaseProgress(name="KERNEL_AGENT", index=2, is_current=True, has_run=True, elapsed_s=60.0)

    assert phase.limit_s is None
    assert phase.limit_kind is None
    assert phase.pct_used is None


def test_render_unbudgeted_session(frozen_clock) -> None:
    """``max_minutes == 0`` is the unlimited sentinel, not a zero budget."""
    snapshot = Snapshot(phase="KERNEL_AGENT", max_minutes=0, result=ResultSummary(), _now_unix=frozen_clock)
    assert "no time budget" in render_status(snapshot, style=PLAIN)


@pytest.mark.parametrize("width", [20, 40, 80, 100, 200])
def test_render_survives_degenerate_widths(session_dir: Path, frozen_clock, width: int) -> None:
    """Rendering must not raise at any terminal width."""
    snapshot = load_snapshot(session_dir, now_unix=frozen_clock)
    assert snapshot is not None
    out = render_status(snapshot, style=FormatStyle(color=False, unicode=True, width=width))
    assert out


def test_render_empty_snapshot_does_not_raise(frozen_clock) -> None:
    """A default-constructed snapshot renders rather than crashing.

    This is the shape produced by a session directory that exists but holds
    nothing readable yet — the first seconds of a run.
    """
    assert render_status(Snapshot(_now_unix=frozen_clock), style=PLAIN) is not None


# --- JSON renderer ------------------------------------------------------------


def test_json_is_valid_and_versioned(session_dir: Path, frozen_clock) -> None:
    """The machine surface parses and carries its schema version."""
    snapshot = load_snapshot(session_dir, now_unix=frozen_clock)
    assert snapshot is not None
    payload = json.loads(render_json(snapshot))

    assert payload["schema_version"] == SCHEMA_VERSION
    assert payload["status"]["phase"] == "FRAMEWORK_AGENT"
    assert payload["session"]["model_name"] == "test-model"
    assert payload["resources"]["tasks"]["total"] == 4


def test_json_preserves_null_for_unmeasured(frozen_clock) -> None:
    """``None`` must survive as ``null``, never coerced to ``0``.

    A consumer averaging over "not measured" values would otherwise get a
    number that looks plausible and is wrong.
    """
    payload = to_dict(Snapshot(result=ResultSummary(), _now_unix=frozen_clock))

    assert payload["result"]["baseline_tput"] is None
    assert payload["result"]["cumulative_gain_pct"] is None
    assert payload["budget"]["session_elapsed_s"] is None


def test_json_top_level_shape_is_stable(session_dir: Path, frozen_clock) -> None:
    """The top-level keys are a contract; changing them needs a version bump."""
    snapshot = load_snapshot(session_dir, now_unix=frozen_clock)
    assert snapshot is not None

    payload = to_dict(snapshot)
    assert payload["schema_version"] == 2
    assert sorted(payload) == [
        "activity",
        "budget",
        "lifecycle",
        "metrics",
        "observed_at_unix",
        "phases",
        "rendered_at_unix",
        "resources",
        "result",
        "schema_version",
        "session",
        "sources",
        "status",
        "warnings",
    ]


def test_json_v1_keys_survive_the_v2_bump(session_dir: Path, frozen_clock) -> None:
    """Schema 2 is additive: every v1 key must still be present and populated.

    A consumer written against v1 should keep working after the bump, so the
    guarantee is asserted rather than left as an intention in a comment.
    """
    snapshot = load_snapshot(session_dir, now_unix=frozen_clock)
    assert snapshot is not None
    payload = to_dict(snapshot)

    for key in ("budget", "lifecycle", "observed_at_unix", "phases", "resources", "result", "session", "status"):
        assert key in payload, f"v1 key {key} disappeared in schema 2"
    assert "phase" in payload["status"]
    assert "session_elapsed_s" in payload["budget"]
    assert "tasks" in payload["resources"]
    # model_name keeps carrying the manifest's literal value even though it is
    # usually a snapshot sha; the readable name arrived as a new key beside it.
    assert payload["session"]["model_name"] == "test-model"


def test_json_carries_the_derived_identity(tmp_path: Path, frozen_clock) -> None:
    """A machine consumer gets both the raw name and the resolved one."""
    from .conftest import write_manifest

    sd = tmp_path / "hf-json"
    write_state(sd)
    write_manifest(
        sd,
        framework="vllm",
        model_name="a4e59da52a7bc87ae7251dd5545c0dd437c44b68",
        model_path=(
            "/data/hf_home/hub/models--meta-models--Muse-Glimmer-30B/snapshots/a4e59da52a7bc87ae7251dd5545c0dd437c44b68"
        ),
        stack_fingerprint={"vllm": "0.27.2rc1.dev150+g311b3513a"},
    )

    session = json.loads(render_json(load_snapshot(sd, now_unix=frozen_clock)))["session"]

    assert session["model_name"] == "a4e59da52a7bc87ae7251dd5545c0dd437c44b68"
    assert session["model_display"] == "meta-models/Muse-Glimmer-30B"
    assert session["model_revision"] == "a4e59da52a7bc87ae7251dd5545c0dd437c44b68"
    assert session["framework_version"] == "0.27.2rc1.dev150+g311b3513a"
    assert session["model_path"].endswith("/snapshots/a4e59da52a7bc87ae7251dd5545c0dd437c44b68")


def test_json_is_serializable_for_every_liveness(tmp_path: Path) -> None:
    """Enums must serialize as their string values, not as repr."""
    import socket

    sd = tmp_path / "live"
    write_state(sd)
    write_lock(sd, pid=1, hostname=socket.gethostname())

    mtime = (sd / "state.json").stat().st_mtime
    snapshot = load_snapshot(sd, now_unix=lambda: mtime)
    assert snapshot is not None

    payload = json.loads(render_json(snapshot))
    assert payload["status"]["liveness"] == "live"
    assert payload["status"]["freshness"] == "fresh"


# --- sub-phase activity and live metrics --------------------------------------


def _metrics_snapshot(**overrides) -> Snapshot:
    """A snapshot carrying the new activity and metrics blocks."""
    base = {
        "phase": "KERNEL_AGENT",
        "liveness": Liveness.STALE,
        "observed_at_unix": 1000.0,
        "state_age_s": 29_400.0,
        "last_activity_age_s": 4.0,
        "current_step": CurrentStep(
            phase="KERNEL_AGENT",
            step="geak_e2e",
            started_unix=1000.0 - 17_640.0,
            deadline_unix=1000.0 + 13_500.0,
        ),
        "running_work": (
            RunningWork(
                kind="specialist",
                run_id="a1e10ec5999744c785280dc0af20722f",
                status="running",
                note="gdn kernel work",
                heartbeat_age_s=43.0,
            ),
        ),
        "activity": (ActivityEntry(relpath="geak/e2e_cycle0/round_1/engineer_0/verify/driver.log", age_s=4.0),),
        "geak": GeakProgress(cycle=0, round_no=1, engineers=("engineer_0", "engineer_1"), active_engineer="engineer_0"),
        "gpus": GpuMetrics(
            tool="amd-smi",
            gpus=(
                GpuMetric(index=0, util_pct=94.0, mem_used_mb=17153.0, mem_total_mb=196592.0, power_w=612.0),
                GpuMetric(index=1, util_pct=0.0, mem_used_mb=284.0, mem_total_mb=196592.0),
            ),
        ),
    }
    base.update(overrides)
    return Snapshot(**base)


def test_current_step_names_what_a_quiet_loop_is_blocked_on() -> None:
    """The line that makes a five-hour silence legible."""
    out = render_status(_metrics_snapshot(), style=PLAIN)

    assert "KERNEL_AGENT → geak_e2e" in out
    # 17640s elapsed of a 31140s window, both as HH:MM.
    assert "04:54 of 08:39" in out
    assert "deadline" in out


def test_stale_header_distinguishes_a_quiet_loop_from_a_dead_run() -> None:
    """ "Working" plus a fresh activity age, not a bare "STALLED?"."""
    out = render_status(_metrics_snapshot(), style=PLAIN)

    assert "working (loop quiet)" in out
    assert "last tick" in out
    assert "activity 4s ago" in out


def test_heartbeat_note_reaches_the_screen() -> None:
    """The agent's own description of its work is the headline of WORK."""
    out = render_status(_metrics_snapshot(), style=PLAIN)

    assert '"gdn kernel work"' in out
    assert "geak" in out and "round 1" in out and "2 engineers" in out


def test_activity_paths_are_not_duplicated_by_a_listed_run() -> None:
    """A run summarised above must not reappear as a raw path below."""
    snapshot = _metrics_snapshot(
        activity=(
            ActivityEntry(relpath="runs/specialist/a1e10ec5999744c785280dc0af20722f/heartbeat.json", age_s=43.0),
            ActivityEntry(relpath="geak/e2e_cycle0/verify/driver.log", age_s=4.0),
        )
    )

    out = render_status(snapshot, style=PLAIN)
    assert "heartbeat.json" not in out
    assert "driver.log" in out


def test_gpu_block_is_labelled_host_wide() -> None:
    """These readings cover every tenant of the node and must say so."""
    out = render_status(_metrics_snapshot(), style=PLAIN)

    assert "GPU (host-wide)" in out
    assert "0: 94%" in out
    assert "16.8/192 GB" in out
    assert "612W" in out
    # An idle GPU is collapsed rather than given a full cell.
    assert "idle 1" in out


def test_unmeasured_gpu_fields_render_as_a_dash() -> None:
    """``"N/A"`` from amd-smi became ``None`` and must not surface as zero."""
    snapshot = _metrics_snapshot(
        gpus=GpuMetrics(
            tool="amd-smi", gpus=(GpuMetric(index=0, util_pct=None, mem_used_mb=90_000.0, mem_total_mb=196_592.0),)
        )
    )

    out = render_status(snapshot, style=PLAIN)
    assert "0: —" in out
    assert "0%" not in out


def test_server_throughput_is_a_dash_before_two_samples() -> None:
    """A rate needs two readings; ``0 tok/s`` would read as a stalled server."""
    snapshot = _metrics_snapshot(
        server=ServerMetrics(url="http://127.0.0.1:8000", requests_running=12, requests_waiting=3, kv_cache_pct=42.0)
    )

    out = render_status(snapshot, style=PLAIN)
    assert "req 12 running / 3 waiting" in out
    assert "kv 42.0%" in out
    assert "out —" in out


def test_terminal_task_heartbeat_is_flagged() -> None:
    """A heartbeat in a completed run's directory is surfaced, not attributed."""
    snapshot = _metrics_snapshot(
        running_work=(RunningWork(kind="specialist", run_id="a1e10ec5", note="still going", task_terminal=True),)
    )

    out = render_status(snapshot, style=PLAIN)
    assert "already recorded done" in out


def test_sources_footer_is_silent_when_healthy() -> None:
    """A warning row printed every frame stops being read as a warning."""
    healthy = (SourceHealth(name="gpu", outcome=SourceOutcome.OK, age_s=1.0),)

    assert "SOURCES" not in render_status(_metrics_snapshot(source_health=healthy), style=PLAIN)
    # ...but --show-sources renders it without the alarm glyph.
    shown = render_status(_metrics_snapshot(source_health=healthy), style=PLAIN, show_sources=True)
    assert "SOURCES" in shown
    assert "⚠" not in shown


def test_degraded_source_surfaces_with_its_age() -> None:
    """A broken probe must be visible, and say how old the shown value is."""
    degraded = (
        SourceHealth(
            name="gpu",
            outcome=SourceOutcome.ERROR,
            age_s=120.0,
            error="amd-smi timed out after 10s",
            consecutive_failures=3,
        ),
    )

    out = render_status(_metrics_snapshot(source_health=degraded), style=PLAIN)
    assert "SOURCES" in out
    assert "⚠" in out
    assert "amd-smi timed out" in out
    assert "x3" in out
    assert "last ok 2m ago" in out


def test_new_blocks_are_omitted_when_empty(session_dir: Path, frozen_clock) -> None:
    """A session with no GPU, server, or activity renders as it always did."""
    snapshot = load_snapshot(session_dir, now_unix=frozen_clock)
    assert snapshot is not None

    out = render_status(snapshot, style=PLAIN)
    assert "GPU (host-wide)" not in out
    assert "vLLM" not in out
    # Matched as a whole heading line: a bare "WORK" also occurs inside
    # FRAMEWORK_AGENT in the phase table.
    assert "  WORK" not in out.splitlines()
    assert "SOURCES" not in out


@pytest.mark.parametrize("width", [20, 40, 80, 100, 200])
def test_metrics_blocks_survive_degenerate_widths(width: int) -> None:
    """Narrow terminals are a real and historically bug-prone case."""
    style = Style(color=False, unicode=True, width=width)
    out = render_status(_metrics_snapshot(), style=style)
    assert out
