# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""``reports/optimization_journal.json`` source.

The journal is the only artifact that records REVERTed attempts: ``state.json``
keeps the adopted stack (KEEPs) but forgets what was tried and rolled back. The
optimizer flushes it after every decision, so it is readable mid-run.

Rows are returned as the raw ``entries`` list; deciding which are worth
exporting is the assembler's job, not the reader's.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from hyperloom.common.jsonio import read_json
from hyperloom.inference_optimizer.session.optimization_journal import JOURNAL_FILENAME
from hyperloom.inference_optimizer.session.session_paths import reports_dir

from .base import SourceResult


class JournalSource:
    """Reads ``<session_dir>/reports/optimization_journal.json``."""

    name = "journal"

    def read(self, session_dir: Path) -> SourceResult:
        """Read the journal's decision rows.

        Args:
            session_dir: Absolute session root.

        Returns:
            :class:`~.base.SourceResult` carrying the ``entries`` list of dicts.
        """
        path = reports_dir(session_dir) / JOURNAL_FILENAME
        if not path.is_file():
            return SourceResult.absent()
        errors: list[BaseException] = []
        data: Any = read_json(path, default=None, require_dict=True, on_error=errors.append)
        if data is None:
            if errors:
                return SourceResult.error(str(errors[0]))
            return SourceResult.error("optimization_journal.json is not a JSON object")
        entries = data.get("entries")
        return SourceResult.hit([row for row in entries if isinstance(row, dict)] if isinstance(entries, list) else [])
