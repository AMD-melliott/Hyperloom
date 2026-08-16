# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Human-readable status rendering.

:func:`render_status` returns a string and never prints. Printing happens once,
at the CLI edge, which keeps the renderer unit-testable and identical across
every surface that shows status (one-shot, watch loop, and anything added
later).

Honesty rules this renderer enforces:

* An observation that is not ``LIVE`` says so in the header, with the age of
  the evidence. Silently showing a week-old snapshot as if it were current is
  the failure mode a status tool exists to prevent.
* ``None`` renders as a dash, never as ``0``.
* Colour is decoration only. Every value carries a text label and unit, so the
  output means the same thing piped to a file.
"""

from __future__ import annotations

from ..model import Freshness, Liveness, Snapshot
from .format import ACCENT, BOLD, DIM, ERR, OK, WARN, Style, bar, duration, number, percent, truncate


_LIVENESS_LABEL = {
    Liveness.LIVE: ("running", OK),
    Liveness.STALE: ("STALLED?", WARN),
    Liveness.DEAD: ("ended", DIM),
    Liveness.UNKNOWN: ("liveness unknown", WARN),
}

# Phase glyphs: done / current / not yet reached.
_MARK_UNICODE = {"done": "✓", "current": "▶", "pending": "·"}
_MARK_ASCII = {"done": "+", "current": ">", "pending": "."}


def _phase_mark(style: Style, kind: str) -> str:
    """Return the marker glyph for a phase state."""
    table = _MARK_UNICODE if style.unicode else _MARK_ASCII
    return table[kind]


def _header(snapshot: Snapshot, style: Style) -> list[str]:
    """Render the identity and liveness header."""
    session = snapshot.session
    bits = [b for b in (session.framework, session.gpu_type) if b]
    if session.tp:
        bits.append(f"TP={session.tp}")
    if session.conc:
        bits.append(f"conc={session.conc}")
    if session.isl and session.osl:
        bits.append(f"{session.isl}/{session.osl}")

    model = session.model_name or "(unknown model)"
    title = style.paint("HYPERLOOM", f"{BOLD};{ACCENT}")
    line = f"{title}  {style.paint(truncate(model, 44), BOLD)}"
    if bits:
        line += style.paint("  ·  " + " · ".join(bits), DIM)

    label, color = _LIVENESS_LABEL[snapshot.liveness]
    status_bits = [style.paint(label, color)]
    if snapshot.liveness is not Liveness.LIVE and snapshot.state_age_s is not None:
        status_bits.append(style.paint(f"last update {duration(snapshot.state_age_s, dash=style.dash)} ago", DIM))
    if snapshot.result.stop_reason:
        status_bits.append(style.paint(f"stop_reason={snapshot.result.stop_reason}", DIM))
    if snapshot.freshness is Freshness.UNKNOWN and snapshot.liveness is Liveness.UNKNOWN:
        status_bits.append(style.paint("no session lock found", DIM))

    meta = style.paint(f"tick {snapshot.tick} · cycle {snapshot.macro_cycle}", DIM)
    return [line, f"  {'  '.join(status_bits)}   {meta}"]


def _session_budget(snapshot: Snapshot, style: Style) -> list[str]:
    """Render the whole-session wall-clock budget line."""
    elapsed = snapshot.session_elapsed_s
    if snapshot.max_minutes <= 0:
        return [f"  elapsed {duration(elapsed, dash=style.dash)}  {style.paint('(no time budget)', DIM)}"]

    total_s = snapshot.max_minutes * 60.0
    frac = (elapsed / total_s) if elapsed is not None else None
    track_width = max(10, min(40, style.width - 46))
    over = frac is not None and frac > 1.0
    track = bar(frac, width=track_width, style=style)
    return [
        "  elapsed {elapsed} / {total}  {track} {pct}".format(
            elapsed=duration(elapsed, dash=style.dash),
            total=duration(total_s, dash=style.dash),
            track=style.paint(track, WARN if over else ACCENT),
            pct=style.paint(f"{frac * 100:.0f}%" if frac is not None else style.dash, WARN if over else ""),
        )
    ]


def _phase_chain(snapshot: Snapshot, style: Style) -> list[str]:
    """Render one line per phase with cumulative spend against budget."""
    if not snapshot.phases:
        return []

    # Widest phase name is FRAMEWORK_AGENT (15). Sized off the data rather than
    # a literal so a new phase name cannot silently shift every column.
    name_width = max((len(phase.name) for phase in snapshot.phases), default=0) + 1

    header = "  {mark} {name} {elapsed:>8} {budget:>8} {used:>6}".format(
        mark=" ", name="PHASE".ljust(name_width), elapsed="ELAPSED", budget="BUDGET", used="USED"
    )
    lines = [style.paint(header.rstrip(), DIM)]

    for phase in snapshot.phases:
        if phase.is_current:
            mark, mark_color = _phase_mark(style, "current"), ACCENT
        elif phase.has_run:
            mark, mark_color = _phase_mark(style, "done"), OK
        else:
            mark, mark_color = _phase_mark(style, "pending"), DIM

        used = phase.pct_used
        # Only the current phase has a live charge-back budget; for the others
        # the absolute cap is the honest comparison, so leave BUDGET blank
        # rather than implying a number the scheduler is not using.
        budget_text = duration(phase.budget_total_s, dash="") if phase.budget_total_s else ""
        used_text = f"{used * 100:.0f}%" if used is not None else ""
        used_color = WARN if (used is not None and used > 1.0) else ""

        # Pad on the plain string, then colour. Padding a painted string would
        # count the ANSI escape bytes as visible width and misalign the column.
        cells = "  {mark} {name} {elapsed:>8} {budget:>8} {used:>6}".format(
            mark=style.paint(mark, mark_color),
            name=style.paint(phase.name.ljust(name_width), "" if (phase.has_run or phase.is_current) else DIM),
            elapsed=duration(phase.elapsed_s, dash=style.dash),
            budget=budget_text,
            used=style.paint(used_text, used_color) if used_text else "",
        )
        lines.append(cells.rstrip())
    return lines


def _result(snapshot: Snapshot, style: Style) -> list[str]:
    """Render gain, target, and the current champion."""
    result = snapshot.result
    validated = result.cumulative_gain_validated_pct
    provisional = result.cumulative_gain_pct

    parts = [
        "gain {val} validated".format(
            val=style.paint(
                percent(validated, dash=style.dash, signed=True),
                OK if (validated or 0) > 0 else DIM,
            )
        )
    ]
    # Both figures are shown when they disagree: the gap between provisional
    # and validated gain is itself the diagnostic.
    if provisional is not None and validated is not None and abs(provisional - validated) > 0.05:
        parts.append(style.paint(f"({percent(provisional, signed=True)} provisional)", DIM))
    if result.target_gap_pct:
        parts.append(style.paint(f"gap {percent(result.target_gap_pct)}", DIM))

    lines = ["  " + "   ".join(parts)]

    best_bits = []
    if result.best_tput is not None:
        best_bits.append(number(result.best_tput, unit="tok/s"))
    if result.best_action:
        best_bits.append(f"via {result.best_action}")
    if result.baseline_tput is not None:
        best_bits.append(style.paint(f"baseline {number(result.baseline_tput, unit='tok/s')}", DIM))
    if best_bits:
        lines.append("  best  " + "  ".join(best_bits))
    if result.crash_count:
        lines.append("  " + style.paint(f"crashes {result.crash_count}", ERR))
    return lines


def _resources(snapshot: Snapshot, style: Style) -> list[str]:
    """Render lane occupancy, GPU leases, and task tallies."""
    lines: list[str] = []

    busy = [lane for lane in snapshot.lanes if lane.held]
    if snapshot.lanes:
        shown = busy or []
        if shown:
            cells = [f"{lane.lane.replace('_lane', '')} {lane.held}/{lane.capacity}" for lane in shown]
            lines.append("  lanes " + style.paint("  ".join(cells), ACCENT))
        else:
            lines.append("  lanes " + style.paint("all idle", DIM))

    if snapshot.gpu_leases:
        ids = ",".join(str(lease.gpu_id) for lease in snapshot.gpu_leases)
        holders = sorted({lease.holder_id for lease in snapshot.gpu_leases})
        lines.append(f"  gpu   {ids} held by {truncate(', '.join(holders), 40)}")

    tasks = snapshot.tasks
    if tasks.total:
        cells = [f"running {tasks.running}", f"queued {tasks.queued}", f"done {tasks.succeeded}"]
        if tasks.failed:
            cells.append(style.paint(f"failed {tasks.failed}", WARN))
        if tasks.cancelled:
            cells.append(style.paint(f"cancelled {tasks.cancelled}", DIM))
        lines.append("  tasks " + "  ".join(cells))

    for task in snapshot.running_tasks[:3]:
        lines.append(style.paint(f"        ▸ {task.kind} ({task.task_id[:8]})", DIM))

    return lines


def _lifecycle(snapshot: Snapshot, style: Style, *, limit: int) -> list[str]:
    """Render the tail of the lifecycle log."""
    events = snapshot.lifecycle[-limit:] if limit > 0 else ()
    if not events:
        return []
    lines = [style.paint("  RECENT", DIM)]
    for event in events:
        stamp = (event.ts or "")[11:19]
        detail = f" {event.detail}" if event.detail else ""
        dur = f" {duration(event.duration_s, dash='')}" if event.duration_s else ""
        raw = f"{stamp} {event.phase or ''} {event.label or event.step or ''} {event.status or ''}{dur}{detail}"
        collapsed = " ".join(raw.split())
        lines.append("  " + style.paint(truncate(collapsed, max(1, style.width - 4)), DIM))
    return lines


def render_status(
    snapshot: Snapshot,
    *,
    style: Style | None = None,
    lifecycle_limit: int = 5,
) -> str:
    """Render a snapshot as human-readable status text.

    Args:
        snapshot: The observation to render.
        style: Rendering capabilities; defaults to plain 100-column ASCII-safe
            output so the function is deterministic when called without a
            terminal.
        lifecycle_limit: Number of trailing lifecycle events to show.

    Returns:
        The rendered block, without a trailing newline.
    """
    style = style or Style()

    blocks: list[list[str]] = [
        _header(snapshot, style),
        _session_budget(snapshot, style),
        _phase_chain(snapshot, style),
        _result(snapshot, style),
        _resources(snapshot, style),
        _lifecycle(snapshot, style, limit=lifecycle_limit),
    ]

    if snapshot.warnings:
        blocks.append([style.paint(f"  ! {warning}", WARN) for warning in snapshot.warnings])

    lines: list[str] = []
    for block in blocks:
        if not block:
            continue
        if lines:
            lines.append("")
        lines.extend(block)
    return "\n".join(lines)
