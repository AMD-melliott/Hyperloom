# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Pure formatting helpers for the visibility renderers.

Every function here is total: any input produces a string, and ``None`` always
produces the "not measured" placeholder rather than a fabricated zero. That
distinction is the whole point — a status display that prints ``0.0%`` for a
metric it never observed is actively misleading.

Conventions:

* Durations are human-scaled (``4h12m``, ``18m``, ``42s``) — a status view is
  read at a glance, not parsed.
* Every value carries its unit in the rendered text, so meaning never depends
  on color alone.
* Colour is opt-in and centrally gated; see :class:`Style`.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from typing import IO


# "Not measured". ASCII fallback keeps output legible on terminals that mangle
# multi-byte characters, which still turn up over serial consoles and in CI logs.
DASH_UNICODE = "—"
DASH_ASCII = "-"


@dataclass(frozen=True)
class Style:
    """Rendering capabilities of the target output stream.

    Attributes:
        color: Emit ANSI colour codes.
        unicode: Use box-drawing and multi-byte glyphs.
        width: Column budget for wrapping and truncation.
    """

    color: bool = False
    unicode: bool = True
    width: int = 100

    @property
    def dash(self) -> str:
        """Placeholder for an unmeasured value."""
        return DASH_UNICODE if self.unicode else DASH_ASCII

    def paint(self, text: str, code: str) -> str:
        """Wrap ``text`` in an ANSI colour when colour is enabled.

        Args:
            text: Text to colour.
            code: SGR parameter, e.g. ``"1;32"``.

        Returns:
            The text, coloured or untouched.
        """
        if not self.color or not code:
            return text
        return f"\033[{code}m{text}\033[0m"


# Semantic colour slots. Names describe meaning, not hue, so a theme change
# never requires touching call sites.
DIM = "2"
BOLD = "1"
OK = "32"
WARN = "33"
ERR = "31"
ACCENT = "36"


def detect_style(stream: IO[str] | None = None, *, width: int | None = None) -> Style:
    """Infer rendering capabilities from the environment and stream.

    Honors ``NO_COLOR`` (any value disables colour, per the informal standard)
    and ``FORCE_COLOR``, and disables colour for non-TTY streams so redirected
    output stays clean for ``grep`` and file capture.

    Args:
        stream: Output stream; defaults to ``sys.stdout``.
        width: Explicit column budget; detected from the terminal when ``None``.

    Returns:
        The inferred :class:`Style`.
    """
    stream = stream if stream is not None else sys.stdout

    try:
        is_tty = bool(stream.isatty())
    except (AttributeError, ValueError):
        is_tty = False

    color = is_tty
    if os.environ.get("NO_COLOR") is not None:
        color = False
    elif os.environ.get("FORCE_COLOR"):
        color = True

    if width is None:
        # Recomputed on every render rather than cached, which makes terminal
        # resize handling implicit.
        import shutil

        width = shutil.get_terminal_size(fallback=(100, 24)).columns if is_tty else 100

    encoding = (getattr(stream, "encoding", "") or "").lower()
    unicode_ok = "utf" in encoding or not is_tty

    return Style(color=color, unicode=unicode_ok, width=max(20, int(width)))


def duration(seconds: float | None, *, dash: str = DASH_UNICODE) -> str:
    """Render a duration at human scale.

    Args:
        seconds: Duration; ``None`` yields the placeholder.
        dash: Placeholder for ``None``.

    Returns:
        e.g. ``"4h12m"``, ``"18m"``, ``"42s"``, ``"0s"``.
    """
    if seconds is None:
        return dash
    try:
        total = max(0, int(float(seconds)))
    except (TypeError, ValueError):
        return dash
    if total < 60:
        return f"{total}s"
    minutes, _ = divmod(total, 60)
    if minutes < 60:
        return f"{minutes}m"
    hours, rem_min = divmod(minutes, 60)
    return f"{hours}h{rem_min:02d}m"


def percent(value: float | None, *, dash: str = DASH_UNICODE, signed: bool = False) -> str:
    """Render a percentage already expressed in percent units.

    Args:
        value: Percentage, e.g. ``15.0`` for 15%.
        dash: Placeholder for ``None``.
        signed: Force a leading ``+`` for non-negative values.

    Returns:
        e.g. ``"15.0%"``, ``"+15.0%"``, or the placeholder.
    """
    if value is None:
        return dash
    try:
        number = float(value)
    except (TypeError, ValueError):
        return dash
    sign = "+" if signed and number >= 0 else ""
    return f"{sign}{number:.1f}%"


def ratio_percent(value: float | None, *, dash: str = DASH_UNICODE) -> str:
    """Render a 0..1 ratio as a percentage.

    Args:
        value: Ratio, e.g. ``0.76``.
        dash: Placeholder for ``None``.

    Returns:
        e.g. ``"76%"``, or the placeholder.
    """
    if value is None:
        return dash
    try:
        return f"{float(value) * 100:.0f}%"
    except (TypeError, ValueError):
        return dash


def number(value: float | None, *, unit: str = "", dash: str = DASH_UNICODE) -> str:
    """Render a float with one decimal and an optional unit.

    Args:
        value: Value to render.
        unit: Unit suffix, appended after a space when non-empty.
        dash: Placeholder for ``None``.

    Returns:
        e.g. ``"281.5 tok/s"``, or the placeholder.
    """
    if value is None:
        return dash
    try:
        rendered = f"{float(value):.1f}"
    except (TypeError, ValueError):
        return dash
    return f"{rendered} {unit}".rstrip() if unit else rendered


def bar(fraction: float | None, *, width: int, style: Style) -> str:
    """Render a proportional progress bar.

    Overrun is shown rather than clamped away: a phase at 140% of its budget is
    a thing the operator needs to see, so the bar fills completely and the
    caller renders the true percentage alongside.

    Args:
        fraction: Completion in ``0..1``; ``None`` yields an empty track.
        width: Bar width in columns.
        style: Rendering capabilities.

    Returns:
        The bar string, without surrounding brackets.
    """
    width = max(0, int(width))
    if width == 0:
        return ""
    filled_char, empty_char = ("█", "░") if style.unicode else ("#", ".")
    if fraction is None:
        return empty_char * width
    try:
        value = max(0.0, min(1.0, float(fraction)))
    except (TypeError, ValueError):
        return empty_char * width
    filled = int(round(value * width))
    return filled_char * filled + empty_char * (width - filled)


def truncate(text: str, width: int, *, dash: str = DASH_UNICODE) -> str:
    """Hard-truncate ``text`` to ``width`` columns with an ellipsis.

    Args:
        text: Text to shorten.
        width: Maximum columns.
        dash: Unused; accepted for signature symmetry with the other helpers.

    Returns:
        Text no longer than ``width``.
    """
    del dash
    width = max(0, int(width))
    if width == 0:
        return ""
    if len(text) <= width:
        return text
    if width == 1:
        return "…"
    return text[: width - 1] + "…"
