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

from ..model import Freshness, Liveness, Snapshot, SourceOutcome
from .format import (
    ACCENT,
    BOLD,
    DIM,
    ERR,
    OK,
    WARN,
    Style,
    bar,
    bytes_size,
    clock,
    duration,
    number,
    percent,
    truncate,
)


_LIVENESS_LABEL = {
    Liveness.LIVE: ("running", OK),
    # Not "STALLED?" any more. A blocking dispatch legitimately suspends the
    # phase machine for hours, and the sub-phase activity block below shows
    # what is happening during that window, so the header should describe the
    # loop's state rather than guess at the run's health.
    Liveness.STALE: ("working (loop quiet)", WARN),
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
        status_bits.append(style.paint(f"last tick {duration(snapshot.state_age_s, dash=style.dash)} ago", DIM))
    if snapshot.liveness is Liveness.STALE and snapshot.last_activity_age_s is not None:
        # The distinction that makes a quiet loop legible: the phase machine
        # has not ticked, but the session tree is still being written to.
        status_bits.append(style.paint(f"activity {duration(snapshot.last_activity_age_s, dash=style.dash)} ago", OK))
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
        return [f"  elapsed {clock(elapsed, dash=style.dash)}  {style.paint('(no time budget)', DIM)}"]

    total_s = snapshot.max_minutes * 60.0
    frac = (elapsed / total_s) if elapsed is not None else None
    track_width = max(10, min(40, style.width - 46))
    over = frac is not None and frac > 1.0
    track = bar(frac, width=track_width, style=style)
    return [
        "  elapsed {elapsed} / {total}  {track} {pct}".format(
            elapsed=clock(elapsed, dash=style.dash),
            total=clock(total_s, dash=style.dash),
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
    # Columns are HH:MM rather than human-scaled: a fixed-width duration keeps
    # the numbers aligned down the column, which is what makes an overrun
    # visible at a glance.
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
        budget_text = clock(phase.budget_total_s, dash="") if phase.budget_total_s else ""
        used_text = f"{used * 100:.0f}%" if used is not None else ""
        used_color = WARN if (used is not None and used > 1.0) else ""

        # Pad on the plain string, then colour. Padding a painted string would
        # count the ANSI escape bytes as visible width and misalign the column.
        cells = "  {mark} {name} {elapsed:>8} {budget:>8} {used:>6}".format(
            mark=style.paint(mark, mark_color),
            name=style.paint(phase.name.ljust(name_width), "" if (phase.has_run or phase.is_current) else DIM),
            elapsed=clock(phase.elapsed_s, dash=style.dash),
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


def _current_step(snapshot: Snapshot, style: Style) -> list[str]:
    """Render the long-running step a blocked phase is waiting on.

    This is the line that makes a five-hour silence legible: the phase machine
    has stopped ticking, but the beacon says what it stopped to do and when
    that work runs out of time.
    """
    step = snapshot.current_step
    if step is None:
        return []

    now = snapshot.rendered_at_unix or snapshot.observed_at_unix or snapshot.now_unix()
    elapsed = step.elapsed_s(now)
    budget = step.budget_s()

    arrow = "→" if style.unicode else "->"
    parts = [style.paint(f"{step.phase} {arrow} {step.step}", f"{BOLD};{ACCENT}")]
    if elapsed is not None:
        timing = clock(elapsed, dash=style.dash)
        if budget:
            over = elapsed > budget
            timing = style.paint(f"{timing} of {clock(budget)}", WARN if over else "")
        parts.append(timing)
    if step.deadline_unix:
        from datetime import datetime, timezone

        stamp = datetime.fromtimestamp(step.deadline_unix, tz=timezone.utc).strftime("%H:%MZ")
        parts.append(style.paint(f"deadline {stamp}", DIM))
    if step.detail:
        parts.append(style.paint(truncate(step.detail, 40), DIM))
    return ["  " + "   ".join(parts)]


def _gpu(snapshot: Snapshot, style: Style) -> list[str]:
    """Render per-GPU utilization.

    Labelled host-wide because that is what the driver reports: on a shared
    node these numbers include every tenant, and implying otherwise would
    invite the operator to read another job's load as their own.
    """
    metrics = snapshot.gpus
    if metrics is None or not metrics.gpus:
        return []

    scope = style.paint("(host-wide)" if metrics.host_global else "", DIM)
    busy = metrics.busy
    idle = [gpu.index for gpu in metrics.gpus if gpu not in busy]

    cells: list[str] = []
    for gpu in busy:
        bits = [f"{gpu.index}:"]
        # Whole percent: GPU utilization is a sampled instant, and a tenth of a
        # percent implies a precision the driver is not offering.
        bits.append(f"{gpu.util_pct:.0f}%" if gpu.util_pct is not None else style.dash)
        if gpu.mem_used_mb is not None and gpu.mem_total_mb:
            bits.append(f"{gpu.mem_used_mb / 1024:.1f}/{gpu.mem_total_mb / 1024:.0f} GB")
        if gpu.power_w is not None:
            bits.append(f"{gpu.power_w:.0f}W")
        cells.append(" ".join(bits))

    if idle:
        cells.append(style.paint(f"idle {_compact_ids(idle)}", DIM))

    return [f"  GPU {scope}  " + "   ".join(cells)]


def _compact_ids(ids: list[int]) -> str:
    """Collapse a sorted id list into ranges: ``[1,2,3,5]`` becomes ``1-3,5``."""
    if not ids:
        return ""
    ordered = sorted(ids)
    spans: list[str] = []
    start = previous = ordered[0]
    for value in ordered[1:]:
        if value == previous + 1:
            previous = value
            continue
        spans.append(str(start) if start == previous else f"{start}-{previous}")
        start = previous = value
    spans.append(str(start) if start == previous else f"{start}-{previous}")
    return ",".join(spans)


def _server(snapshot: Snapshot, style: Style) -> list[str]:
    """Render live inference-server counters."""
    server = snapshot.server
    if server is None:
        return []
    bits = []
    if server.requests_running is not None or server.requests_waiting is not None:
        bits.append(f"req {server.requests_running or 0} running / {server.requests_waiting or 0} waiting")
    if server.kv_cache_pct is not None:
        bits.append(f"kv {percent(server.kv_cache_pct)}")
    # None rather than 0 until two samples exist — a rate needs two readings,
    # and "0 tok/s" would read as a stalled server.
    bits.append(f"out {number(server.tput_tok_s, unit='tok/s', dash=style.dash)}")
    return ["  vLLM  " + "   ".join(bits) + style.paint(f"   {server.url}", DIM)]


def _work(snapshot: Snapshot, style: Style, *, limit: int = 4) -> list[str]:
    """Render what long-running sub-agents are doing right now.

    Ordered by how recently each showed a sign of life, so the top line is the
    best available answer to "what is it doing".
    """
    if not (snapshot.running_work or snapshot.activity or snapshot.geak):
        return []

    lines = [style.paint("  WORK", DIM)]
    width = max(20, style.width - 4)

    if snapshot.geak is not None:
        from ..sources.geak import describe

        text = describe(snapshot.geak)
        if text:
            lines.append("    " + style.paint(truncate(f"geak  {text}", width), ACCENT))

    shown_work = snapshot.running_work[:limit]
    # Any run already summarised above is reported far better by its heartbeat
    # note than by the raw path of the file that heartbeat lives in, so its
    # subtree is suppressed from the recent-writes list below.
    covered = tuple(f"runs/{work.kind}/{work.run_id}/" for work in shown_work)

    for work in shown_work:
        bits = [f"{work.kind} {work.run_id[:8]}"]
        if work.note:
            # Agent-authored free text, written every five minutes per the
            # specialist prompt contract. Usually the most informative thing
            # on the whole screen.
            bits.append(f'"{work.note}"')
        elif work.status:
            bits.append(work.status)
        if work.turn is not None:
            bits.append(f"turn {work.turn}" + (f"/{work.max_turns}" if work.max_turns else ""))
        if work.log_growth_bps:
            bits.append(f"+{bytes_size(work.log_growth_bps)}/s")
        age = f"{duration(work.age_s, dash=style.dash)} ago" if work.age_s is not None else ""
        line = truncate("  ".join(bits), max(10, width - len(age) - 6))
        lines.append(f"    {line}  {style.paint(age, DIM)}")
        if work.task_terminal:
            lines.append(
                style.paint(
                    f"      ! heartbeat is refreshing in {work.run_id[:8]}, whose task already recorded done",
                    WARN,
                )
            )

    remaining = [entry for entry in snapshot.activity if not entry.relpath.startswith(covered)]
    for entry in remaining[:limit]:
        age = f"{duration(entry.age_s, dash=style.dash)} ago"
        size = bytes_size(entry.size_bytes, dash="")
        # The path is the payload: a truncated head loses the run identity, so
        # keep the tail where the round/engineer/stage segments live.
        path = entry.relpath
        budget = max(10, width - len(age) - len(size) - 10)
        if len(path) > budget:
            path = "…" + path[-(budget - 1) :]
        lines.append(f"    {style.paint(path, DIM)}  {style.paint(age, DIM)}  {style.paint(size, DIM)}")

    return lines


def _sources(snapshot: Snapshot, style: Style, *, always: bool = False) -> list[str]:
    """Render per-source collection health.

    Shown only when something is degraded unless ``always``. A healthy footer
    every frame is noise the operator learns to skip, which is exactly what
    must not happen to a warning line.
    """
    rows = snapshot.source_health if always else snapshot.degraded_sources
    if not rows:
        return []
    cells: list[str] = []
    for row in rows:
        detail = row.error or row.outcome.value
        age = f" (last ok {duration(row.age_s)} ago)" if row.age_s else ""
        repeats = f" x{row.consecutive_failures}" if row.consecutive_failures > 1 else ""
        color = ERR if row.outcome is SourceOutcome.ERROR else DIM
        cells.append(style.paint(f"{row.name}: {truncate(detail, 48)}{repeats}{age}", color))
    # The warning glyph is reserved for an actual problem. Printing it beside a
    # healthy footer every frame is how a warning marker stops being read.
    marker = ("⚠ " if style.unicode else "! ") if snapshot.degraded_sources else ""
    return [f"  SOURCES  {marker}" + "   ".join(cells)]


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
    show_sources: bool = False,
) -> str:
    """Render a snapshot as human-readable status text.

    Args:
        snapshot: The observation to render.
        style: Rendering capabilities; defaults to plain 100-column ASCII-safe
            output so the function is deterministic when called without a
            terminal.
        lifecycle_limit: Number of trailing lifecycle events to show.
        show_sources: Always render the per-source health footer, rather than
            only when a source is degraded.

    Returns:
        The rendered block, without a trailing newline.
    """
    style = style or Style()

    blocks: list[list[str]] = [
        _header(snapshot, style),
        _session_budget(snapshot, style),
        _current_step(snapshot, style),
        _phase_chain(snapshot, style),
        _result(snapshot, style),
        _gpu(snapshot, style) + _server(snapshot, style),
        _resources(snapshot, style),
        _work(snapshot, style),
        _lifecycle(snapshot, style, limit=lifecycle_limit),
        _sources(snapshot, style, always=show_sources),
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
