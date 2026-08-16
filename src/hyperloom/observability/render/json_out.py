# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Machine-readable rendering of a :class:`~..model.Snapshot`.

This is a stable contract, so it is built by hand rather than by reflecting
over the dataclasses: an incidental field rename in the model must not silently
break a downstream consumer's parser. ``SCHEMA_VERSION`` is bumped when the
shape changes incompatibly.

``None`` is preserved as JSON ``null`` and never coerced to ``0``. A consumer
computing an average over "not measured" values would otherwise get a number
that looks plausible and is wrong.
"""

from __future__ import annotations

import json
from typing import Any

from ..model import Snapshot


SCHEMA_VERSION = 1


def to_dict(snapshot: Snapshot) -> dict[str, Any]:
    """Project a snapshot into the stable JSON shape.

    Args:
        snapshot: The observation to project.

    Returns:
        A JSON-serializable dict.
    """
    session = snapshot.session
    result = snapshot.result

    return {
        "schema_version": SCHEMA_VERSION,
        "observed_at_unix": snapshot.observed_at_unix,
        "session": {
            "session_dir": session.session_dir,
            "session_id": session.session_id,
            "model_name": session.model_name,
            "framework": session.framework,
            "gpu_type": session.gpu_type,
            "tp": session.tp,
            "ep": session.ep,
            "conc": session.conc,
            "isl": session.isl,
            "osl": session.osl,
            "precision": session.precision,
            "objective_kind": session.objective_kind,
            "objective_value": session.objective_value,
            "started_at": session.started_at,
        },
        "status": {
            "phase": snapshot.phase or None,
            "macro_cycle": snapshot.macro_cycle,
            "tick": snapshot.tick,
            "liveness": snapshot.liveness.value,
            "freshness": snapshot.freshness.value,
            "state_age_s": snapshot.state_age_s,
            "owner_pid": snapshot.owner_pid,
            "current_action": snapshot.current_action,
            "is_terminal": snapshot.is_terminal,
            "stop_reason": result.stop_reason,
        },
        "budget": {
            "max_minutes": snapshot.max_minutes,
            "session_elapsed_s": snapshot.session_elapsed_s,
            "session_remaining_s": snapshot.session_remaining_s,
        },
        "phases": [
            {
                "name": phase.name,
                "index": phase.index,
                "is_current": phase.is_current,
                "has_run": phase.has_run,
                "elapsed_s": phase.elapsed_s,
                "budget_total_s": phase.budget_total_s,
                "budget_remaining_s": phase.budget_remaining_s,
                "cap_s": phase.cap_s,
                "pct_used": phase.pct_used,
            }
            for phase in snapshot.phases
        ],
        "result": {
            "baseline_tput": result.baseline_tput,
            "best_tput": result.best_tput,
            "best_action": result.best_action,
            "cumulative_gain_pct": result.cumulative_gain_pct,
            "cumulative_gain_validated_pct": result.cumulative_gain_validated_pct,
            "target_gap_pct": result.target_gap_pct,
            "crash_count": result.crash_count,
        },
        "resources": {
            "lanes": [
                {
                    "lane": lane.lane,
                    "held": lane.held,
                    "capacity": lane.capacity,
                    "holders": list(lane.holders),
                }
                for lane in snapshot.lanes
            ],
            "gpu_leases": [
                {
                    "gpu_id": lease.gpu_id,
                    "holder_id": lease.holder_id,
                    "task_id": lease.task_id,
                    "expires_at": lease.expires_at,
                    "expired": lease.expired,
                }
                for lease in snapshot.gpu_leases
            ],
            "tasks": {
                "queued": snapshot.tasks.queued,
                "running": snapshot.tasks.running,
                "succeeded": snapshot.tasks.succeeded,
                "failed": snapshot.tasks.failed,
                "cancelled": snapshot.tasks.cancelled,
                "total": snapshot.tasks.total,
            },
            "running_tasks": [
                {
                    "task_id": task.task_id,
                    "kind": task.kind,
                    "state": task.state,
                    "updated_at": task.updated_at,
                }
                for task in snapshot.running_tasks
            ],
        },
        "lifecycle": [
            {
                "seq": event.seq,
                "ts": event.ts,
                "phase": event.phase,
                "step": event.step,
                "label": event.label,
                "status": event.status,
                "detail": event.detail,
                "duration_s": event.duration_s,
            }
            for event in snapshot.lifecycle
        ],
        "warnings": list(snapshot.warnings),
    }


def render_json(snapshot: Snapshot, *, indent: int | None = 2) -> str:
    """Render a snapshot as a JSON document.

    Args:
        snapshot: The observation to render.
        indent: Indentation; pass ``None`` for a single compact line.

    Returns:
        The JSON text, without a trailing newline.
    """
    return json.dumps(to_dict(snapshot), indent=indent, sort_keys=True)
