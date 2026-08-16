# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Source protocol for the visibility layer.

A source reads exactly one artifact of a session directory and reports its own
degradation. Sources never raise, never mutate, and never fabricate: a missing
artifact is :attr:`~hyperloom.observability.model.SourceOutcome.ABSENT`, an
unreadable one is ``ERROR`` with a message, and both leave the rest of the
snapshot intact.

Keeping ``ABSENT`` and ``ERROR`` distinct matters more than it looks:
``coordinator.db`` does not exist until the Coordinator boots, and
``manifest.json`` is written a moment after the session directory appears.
Reporting either as an error would make a perfectly healthy early-run session
look broken, and operators would learn to ignore the warning line.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from ..model import SourceOutcome


@dataclass(frozen=True)
class SourceResult:
    """Outcome of one source read.

    Attributes:
        outcome: Whether the read succeeded, found nothing, or failed.
        data: Payload on success; ``None`` otherwise.
        message: Human-readable detail, set when ``outcome`` is ``ERROR``.
    """

    outcome: SourceOutcome
    data: Any = None
    message: str | None = None

    @property
    def ok(self) -> bool:
        """Whether the read produced usable data."""
        return self.outcome is SourceOutcome.OK

    @classmethod
    def hit(cls, data: Any) -> SourceResult:
        """Build a successful result carrying ``data``."""
        return cls(SourceOutcome.OK, data, None)

    @classmethod
    def absent(cls) -> SourceResult:
        """Build a result for an artifact that does not exist yet."""
        return cls(SourceOutcome.ABSENT, None, None)

    @classmethod
    def error(cls, message: str) -> SourceResult:
        """Build a failed result carrying ``message``."""
        return cls(SourceOutcome.ERROR, None, message)


@runtime_checkable
class Source(Protocol):
    """One readable artifact of a session directory."""

    @property
    def name(self) -> str:
        """Short identifier used to attribute warnings (e.g. ``coordinator_db``)."""
        ...

    def read(self, session_dir: Path) -> SourceResult:
        """Read this source's artifact.

        Args:
            session_dir: Absolute session root.

        Returns:
            The read outcome. Implementations must not raise.
        """
        ...


def warning_for(source_name: str, result: SourceResult) -> str | None:
    """Render a source failure as an attributed warning line.

    Args:
        source_name: The reporting source's :attr:`Source.name`.
        result: The result to describe.

    Returns:
        A ``"<source>: <message>"`` line, or ``None`` when there is nothing to
        report (``OK`` and ``ABSENT`` are both silent).
    """
    if result.outcome is not SourceOutcome.ERROR:
        return None
    return f"{source_name}: {result.message or 'unreadable'}"
