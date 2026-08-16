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
from hyperloom.observability.model import PhaseProgress, ResultSummary, Snapshot
from hyperloom.observability.render import SCHEMA_VERSION, Style, render_json, render_status, to_dict
from hyperloom.observability.render.format import Style as FormatStyle
from hyperloom.observability.render.format import bar, detect_style, duration, number, percent, truncate

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


def test_none_never_renders_as_zero() -> None:
    """The core honesty rule: unmeasured is a dash, not a plausible number."""
    assert percent(None) == "—"
    assert number(None) == "—"
    assert duration(None) == "—"
    assert percent(0.0) == "0.0%"  # a measured zero still renders as zero


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
    assert "EXPLORE" in out
    assert "TP=8" in out
    assert "sglang" in out


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
    assert "last update" in out
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
        phase="EXPLORE",
        phases=(
            PhaseProgress(
                name="EXPLORE",
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
    assert "200%" in render_status(snapshot, style=PLAIN)


def test_render_unbudgeted_session(frozen_clock) -> None:
    """``max_minutes == 0`` is the unlimited sentinel, not a zero budget."""
    snapshot = Snapshot(phase="EXPLORE", max_minutes=0, result=ResultSummary(), _now_unix=frozen_clock)
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
    assert payload["status"]["phase"] == "EXPLORE"
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

    assert sorted(to_dict(snapshot)) == [
        "budget",
        "lifecycle",
        "observed_at_unix",
        "phases",
        "resources",
        "result",
        "schema_version",
        "session",
        "status",
        "warnings",
    ]


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
