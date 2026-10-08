# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""What the journal and ``state.json`` contribute to the optimization table and the accuracy series."""

from __future__ import annotations

from pathlib import Path

from hyperloom.observability import load_snapshot
from hyperloom.observability.assemble import MAX_LEVER_LABEL_CHARS, MAX_OPTIMIZATION_ROWS
from hyperloom.observability.model import SourceOutcome
from hyperloom.observability.sources import JournalSource

from .conftest import write_journal, write_state


def _row(change: str, outcome: str, **extra) -> dict:
    return {"phase": "PRELUDE", "iter": 1, "kind": "other", "change": change, "outcome": outcome, **extra}


JOURNAL = [
    _row("target_analysis", "KEEP", task_id="t0"),
    _row("baseline", "KEEP", kind="baseline", throughput_after=508.3, task_id="t1"),
    _row("specialist", "KEEP", task_id="t2"),
    _row("warm_replay", "KEEP", gain_pct=52.4, throughput_after=774.5, task_id="t3"),
    _row("roofline", "REVERT", error_class="trace_analyze_failed", task_id="t4"),
    _row("noop", "no_promote", task_id="t5"),
]


def _snapshot(session_dir: Path, frozen_clock, **state):
    write_state(session_dir, **state)
    snap = load_snapshot(session_dir, now_unix=frozen_clock)
    assert snap is not None
    return snap


def test_only_measured_keeps_and_every_revert_are_listed(session_dir: Path, frozen_clock) -> None:
    write_journal(session_dir, JOURNAL)
    snap = _snapshot(session_dir, frozen_clock)
    assert [(r.ordinal, r.lever, r.outcome) for r in snap.optimizations] == [
        (3, "warm_replay", "KEEP"),
        (4, "roofline", "REVERT"),
    ]
    assert snap.optimizations[0].gain_pct == 52.4
    assert snap.optimizations[0].tput == 774.5
    assert snap.optimizations[1].tput is None


def test_accuracy_is_joined_from_the_adopted_stack_by_task(session_dir: Path, frozen_clock) -> None:
    write_journal(session_dir, JOURNAL)
    stack = [{"task_id": "t3", "accuracy": 0.9727}]
    snap = _snapshot(session_dir, frozen_clock, optimization_stack=stack, baseline_accuracy=0.975)
    assert snap.optimizations[0].accuracy == 0.9727
    assert snap.optimizations[1].accuracy is None
    assert snap.result.baseline_accuracy == 0.975
    assert snap.result.best_accuracy == 0.9727


def test_accuracy_is_absent_when_nothing_recorded_it(session_dir: Path, frozen_clock) -> None:
    snap = _snapshot(session_dir, frozen_clock)
    assert snap.result.baseline_accuracy is None
    assert snap.result.best_accuracy is None


def test_a_missing_or_corrupt_journal_leaves_the_table_empty(session_dir: Path, frozen_clock) -> None:
    assert _snapshot(session_dir, frozen_clock).optimizations == ()
    write_journal(session_dir, "{not json")
    snap = _snapshot(session_dir, frozen_clock)
    assert snap.optimizations == ()
    assert any(row.name == "journal" and row.outcome is SourceOutcome.ERROR for row in snap.source_health)


def test_the_table_is_bounded_and_keeps_the_newest_rows(session_dir: Path, frozen_clock) -> None:
    write_journal(session_dir, [_row(f"lever-{i}", "REVERT") for i in range(MAX_OPTIMIZATION_ROWS + 25)])
    snap = _snapshot(session_dir, frozen_clock)
    assert len(snap.optimizations) == MAX_OPTIMIZATION_ROWS
    assert snap.optimizations[-1].lever == f"lever-{MAX_OPTIMIZATION_ROWS + 24}"
    assert snap.optimizations[0].ordinal == 25


def test_lever_labels_are_truncated(session_dir: Path, frozen_clock) -> None:
    write_journal(session_dir, [_row("x" * 500, "REVERT")])
    assert len(_snapshot(session_dir, frozen_clock).optimizations[0].lever) == MAX_LEVER_LABEL_CHARS


def test_journal_source_reports_absent_without_a_file(tmp_path: Path) -> None:
    assert JournalSource().read(tmp_path).outcome is SourceOutcome.ABSENT
