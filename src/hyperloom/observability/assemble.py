# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Session resolution and the source-to-:class:`~.model.Snapshot` join.

This module is the composition root of the read layer: it resolves which
session to look at, runs every source, derives liveness and phase progress, and
returns one immutable :class:`~.model.Snapshot`. Renderers consume that object
and nothing else.

Nothing here writes. That is asserted, not assumed — see
``test_read_only_invariant``.
"""

from __future__ import annotations

import os
from pathlib import Path, PurePosixPath
from typing import Any, Callable

from hyperloom.common.coerce import to_float, to_int
from hyperloom.inference_optimizer.session.paths import (
    ENV_CURRENT_SESSION_DIR,
    find_latest_per_session_dir,
    workspace_root,
)

from .model import (
    ActivityEntry,
    Freshness,
    GpuLease,
    LaneOccupancy,
    LifecycleEvent,
    Liveness,
    ResultSummary,
    RunningTask,
    RunningWork,
    SessionInfo,
    Snapshot,
    SourceHealth,
    SourceOutcome,
    TaskCounts,
)
from .progress import build_phase_progress, elapsed_totals_from_history, session_timing
from .sources import (
    ActivitySource,
    CoordinatorDbSource,
    CurrentStepSource,
    GeakSource,
    LockFileSource,
    ManifestSource,
    StateFileSource,
    heartbeat_age_s,
    warning_for,
)


# ``state.json`` is rewritten many times per tick, so a long gap suggests a
# wedged Coordinator. It is advisory, not proof: a single long-running action
# (a TraceLens pass can run for over an hour) can legitimately exceed it. STALE
# means "worth a look", never "broken".
DEFAULT_STALE_AFTER_S = 900.0

# A heartbeat this recent is treated as positive evidence of life even when the
# pid cannot be interrogated, e.g. a lock written on another host.
LIVE_HEARTBEAT_S = 120.0

DEFAULT_LIFECYCLE_LIMIT = 12


def _last_activity_unix(state: dict[str, Any], *, state_mtime_unix: float) -> float:
    """Return the most recent timestamp the session itself recorded.

    Preferred over the ``state.json`` mtime, which is filesystem metadata and
    therefore a property of the *file* rather than the run: copying, syncing,
    or archiving a session directory rewrites it. A real session observed here
    had an mtime four days after its last recorded event purely because the
    directory had been copied — enough to make every duration wrong.

    The timestamps inside the document move only when the optimizer moves.

    Args:
        state: Parsed ``state.json``.
        state_mtime_unix: Fallback when the document carries no timestamps.

    Returns:
        Best available "last activity" timestamp, or ``0.0`` when unknown.
    """
    from datetime import datetime, timezone

    best = 0.0

    history = state.get("phase_history")
    if isinstance(history, list):
        for row in history:
            if isinstance(row, dict):
                value = to_float(row.get("ts_unix"))
                if value is not None:
                    best = max(best, value)

    lifecycle = state.get("lifecycle")
    if isinstance(lifecycle, list):
        for row in lifecycle:
            if not isinstance(row, dict):
                continue
            raw = row.get("ts")
            if not raw:
                continue
            try:
                parsed = datetime.fromisoformat(str(raw))
            except (TypeError, ValueError):
                continue
            if parsed.tzinfo is None:
                parsed = parsed.replace(tzinfo=timezone.utc)
            best = max(best, parsed.timestamp())

    return best if best > 0 else float(state_mtime_unix)


def _progress_clock(
    *,
    now_unix: float,
    state_mtime_unix: float,
    liveness: Liveness,
    stop_reason: str | None,
) -> float:
    """Return the time that duration math should be measured against.

    Durations are ``now - phase_started_unix``, which for a session that has
    already ended keeps growing with wall-clock time — an 8-hour run inspected
    a week later would report its final phase as lasting a week. That is not a
    cosmetic problem: it makes budget percentages meaningless on exactly the
    artifacts operators inspect most, namely finished runs.

    So when the run can no longer be making progress, the clock is pinned to
    the last evidence we actually have — the final ``state.json`` write. The
    reported duration then converges on the phase's true final length.

    A ``STALE`` session is deliberately *not* pinned: the process is still up
    and a growing "this phase has run for 3h" is the signal the operator wants.

    Args:
        now_unix: Wall-clock observation time.
        state_mtime_unix: Last ``state.json`` write, or ``0.0`` when unknown.
        liveness: Derived liveness.
        stop_reason: Session stop reason, when recorded.

    Returns:
        The timestamp to treat as "now" for duration math.
    """
    ended = bool(stop_reason) or liveness is Liveness.DEAD
    if ended and state_mtime_unix > 0:
        return min(now_unix, state_mtime_unix)
    return now_unix


def resolve_session_dir(explicit: Path | str | None = None, *, model: str | None = None) -> Path | None:
    """Resolve which session directory to read.

    Precedence: explicit argument, then ``$INFERENCE_OPTIMIZER_CURRENT_SESSION_DIR``
    (set by the CLI during a run), then the newest per-model timestamped
    session under the workspace root, then the workspace root itself.

    The auto-discovery step is what the existing operator scripts lack — their
    ``--help`` text warns that falling back to the workspace root usually finds
    no ``coordinator.db``, because sessions actually live one or two levels
    down under ``<model>/<timestamp>/``.

    Args:
        explicit: Caller-supplied directory; wins outright when set.
        model: Optional model basename to narrow auto-discovery.

    Returns:
        The resolved directory, or ``None`` when nothing plausible exists.
    """
    if explicit:
        path = Path(explicit).expanduser()
        return path if path.is_dir() else None

    pinned = os.environ.get(ENV_CURRENT_SESSION_DIR)
    if pinned:
        path = Path(pinned).expanduser()
        if path.is_dir():
            return path

    # NOTE: selection inside is a lexical sort on the ``%Y%m%dT%H%M%SZ``
    # directory name, not mtime. That is correct for this layout and
    # deliberately not "fixed" to mtime, which a resume would perturb.
    latest = find_latest_per_session_dir(model)
    if latest is not None and Path(latest).is_dir():
        return Path(latest)

    root = workspace_root()
    return root if root.is_dir() else None


def _derive_liveness(
    lock: dict[str, Any] | None,
    *,
    state_age_s: float | None,
    activity_age_s: float | None,
    now_unix: float,
    stale_after_s: float,
    stop_reason: str | None,
) -> Liveness:
    """Classify whether the owning optimizer is still running.

    ``UNKNOWN`` is a real answer. The lock file is never unlinked, so presence
    proves only that a run once started here.

    **Filesystem activity outranks the pid**, and that ordering is the result
    of a bug this function shipped. Two hazards make a recorded pid weak
    evidence in both directions:

    * **PID namespaces.** A containerized run writes the *container's* pid
      alongside the *host's* hostname, because the container inherits it. A
      hostname match therefore does not license interpreting the pid locally.
      Both failure directions were observed on real sessions: a container pid
      of ``19`` matching an unrelated live kernel thread on the host (false
      LIVE), and a container pid of ``304617`` with no host counterpart, which
      this function reported as ``DEAD`` for a run that was thirteen hours in
      and actively working (false DEAD). The second is the worse error — it
      also pinned the duration clock, so the running phase displayed ``0s``.
    * **PID reuse.** Even same-namespace, a long-dead optimizer's pid may have
      been recycled.

    So: a pid that is *absent* is treated as uninterpretable rather than as
    proof of death, and is only allowed to conclude ``DEAD`` when the session
    tree corroborates it by showing no recent writes either. Conversely, recent
    writes anywhere in the session tree are positive evidence of life that no
    namespace can confound.

    ``heartbeat_at`` from the lock is deliberately weak input: on a real
    thirteen-hour run it was written once at startup and never refreshed.

    Args:
        lock: Payload from :class:`~.sources.LockFileSource`, or ``None``.
        state_age_s: Age of ``state.json``, or ``None``.
        activity_age_s: Age of the freshest write anywhere in the session tree,
            or ``None`` when the activity source could not read one.
        now_unix: Current time.
        stale_after_s: Age past which corroborating evidence is not fresh.
        stop_reason: Recorded session stop reason, when any.

    Returns:
        The liveness classification.
    """
    # A recorded stop reason is the session's own statement that it concluded.
    # It outranks every other signal, which can only produce false positives.
    if stop_reason:
        return Liveness.DEAD

    activity_fresh = activity_age_s is not None and activity_age_s <= stale_after_s
    state_fresh = state_age_s is not None and state_age_s <= stale_after_s

    if not lock:
        # No lock to interrogate, but something is writing. Files do not write
        # themselves; that is enough to say the run is alive.
        return Liveness.STALE if activity_fresh else Liveness.UNKNOWN

    pid_alive = lock.get("pid_alive")
    hb_age = heartbeat_age_s(lock, now_unix=now_unix)
    claims_alive = bool(pid_alive) or (hb_age is not None and hb_age <= LIVE_HEARTBEAT_S)

    if claims_alive or activity_fresh:
        # Fresh state.json means the loop is publishing. Fresh activity without
        # it means a long blocking dispatch is in flight — the run is working,
        # just not ticking, which is STALE rather than LIVE.
        return Liveness.LIVE if state_fresh else Liveness.STALE

    if lock.get("same_host") and pid_alive is False:
        # The pid is gone AND nothing in the tree has been written recently.
        # Only with both is death the honest conclusion; the pid alone is not
        # enough, because it may belong to another namespace entirely.
        if activity_age_s is not None or state_age_s is not None:
            return Liveness.DEAD
        return Liveness.UNKNOWN

    return Liveness.UNKNOWN


# Hugging Face lays its cache out as
# ``<root>/hub/models--<org>--<repo>/snapshots/<revision>/``, so the directory a
# server is actually pointed at is named after a 40-char commit sha. The
# manifest's ``model_name`` is ``Path(model).name`` of exactly that directory
# (``cli.bootstrap.resolve_model_display_name``), which is why an operator sees
# ``a4e59da52a7bc87ae...`` where the model should be.
_HF_REPO_PREFIX = "models--"


def parse_hf_cache_path(model_path: str | None) -> tuple[str | None, str | None]:
    """Recover the repo id and revision from a Hugging Face cache path.

    ``huggingface_hub`` encodes a repo id by replacing ``/`` with ``--``, so the
    inverse is a plain substitution. Anything that is not that layout — a plain
    directory of weights, a quantization export — yields ``(None, None)`` and
    the caller falls back to the declared name.

    Args:
        model_path: Filesystem path the run was pointed at.

    Returns:
        ``(repo_id, revision)``, either element ``None`` when not recoverable.
    """
    if not model_path:
        return None, None
    parts = PurePosixPath(str(model_path)).parts
    for position, part in enumerate(parts):
        if part != "snapshots" or position == 0:
            continue
        folder = parts[position - 1]
        if not folder.startswith(_HF_REPO_PREFIX):
            continue
        repo = folder[len(_HF_REPO_PREFIX) :].replace("--", "/")
        revision = parts[position + 1] if position + 1 < len(parts) else None
        return repo or None, revision or None
    return None, None


def framework_version(manifest: dict[str, Any], framework: str | None) -> str | None:
    """Pull the running framework's version out of the stack fingerprint.

    ``common.provenance.detect_stack_fingerprint`` records one entry per stack
    component, writing the literal string ``"unknown"`` for anything it could
    not probe. That is an absence, not a version, so it maps to ``None`` rather
    than being displayed.

    Args:
        manifest: Parsed ``manifest.json``.
        framework: The framework in use, e.g. ``"vllm"``.

    Returns:
        The version string, or ``None``.
    """
    fingerprint = manifest.get("stack_fingerprint")
    if not isinstance(fingerprint, dict) or not framework:
        return None
    value = str(fingerprint.get(str(framework).strip().lower()) or "").strip()
    if not value or value.lower() == "unknown":
        return None
    return value


def _build_session_info(session_dir: Path, manifest: dict[str, Any], state: dict[str, Any]) -> SessionInfo:
    """Join manifest and state into workload identity.

    The manifest records what was requested and is never rewritten; state
    mirrors much of it but can drift mid-run. Manifest wins, state fills gaps.

    Args:
        session_dir: Absolute session root.
        manifest: Parsed ``manifest.json``, possibly empty.
        state: Parsed ``state.json``, possibly empty.

    Returns:
        The joined :class:`~.model.SessionInfo`.
    """
    workload = manifest.get("workload") if isinstance(manifest.get("workload"), dict) else {}
    objective = manifest.get("objective") if isinstance(manifest.get("objective"), dict) else {}

    def pick(key: str, *, state_key: str | None = None) -> Any:
        value = manifest.get(key)
        if value in (None, ""):
            value = state.get(state_key or key)
        return value if value not in (None, "") else None

    declared_model = pick("model_name")
    model_path = pick("model_path")
    repo, revision = parse_hf_cache_path(model_path)
    # The repo id wins when the path yields one: it is the only place the real
    # model name survives, because the declared name is the snapshot directory.
    display = repo or declared_model or (PurePosixPath(str(model_path)).name if model_path else None)
    framework = pick("framework")

    return SessionInfo(
        session_dir=str(session_dir),
        session_id=pick("session_id"),
        model_name=declared_model,
        model_display=display or None,
        model_path=model_path,
        model_revision=revision,
        framework=framework,
        framework_version=framework_version(manifest, framework),
        gpu_type=pick("gpu_type"),
        tp=to_int(pick("tp")),
        ep=to_int(pick("ep")),
        conc=to_int(workload.get("conc") if workload else state.get("conc")),
        isl=to_int(workload.get("isl") if workload else state.get("isl")),
        osl=to_int(workload.get("osl") if workload else state.get("osl")),
        precision=(workload.get("precision") if workload else state.get("precision")) or None,
        objective_kind=(objective.get("kind") if objective else None) or None,
        objective_value=objective.get("value") if objective else None,
        target_summary=state.get("target_summary") or None,
        started_at=manifest.get("created_at_utc") or state.get("start_ts") or None,
    )


def _build_result(state: dict[str, Any]) -> ResultSummary:
    """Extract the optimization outcome so far from ``state.json``."""
    best = state.get("current_best") if isinstance(state.get("current_best"), dict) else {}
    return ResultSummary(
        baseline_tput=to_float(state.get("baseline_tput")),
        best_tput=to_float(best.get("tput")) if best else None,
        best_action=(best.get("action") or None) if best else None,
        cumulative_gain_pct=to_float(state.get("cumulative_gain")),
        cumulative_gain_validated_pct=to_float(state.get("cumulative_gain_validated")),
        target_gap_pct=to_float(state.get("target_gap_pct")),
        stop_reason=(state.get("stop_reason") or None),
        crash_count=to_int(state.get("crash_count"), 0) or 0,
    )


def _build_source_health(reads: list[tuple[str, Any]], *, now_unix: float) -> tuple[SourceHealth, ...]:
    """Summarise the outcome of each inline source read.

    Inline reads are either fresh or they failed just now, so ``age_s`` is
    zero for successes. The background collector overwrites these rows with
    its own, which carry real ages because its values can be minutes old.

    Args:
        reads: ``(source_name, SourceResult)`` pairs.
        now_unix: Unused; accepted so the signature matches the collector's.

    Returns:
        One :class:`~.model.SourceHealth` per source, in read order.
    """
    del now_unix
    return tuple(
        SourceHealth(
            name=name,
            outcome=result.outcome,
            age_s=0.0 if result.outcome is SourceOutcome.OK else None,
            error=result.message,
            consecutive_failures=1 if result.outcome is SourceOutcome.ERROR else 0,
        )
        for name, result in reads
    )


def _build_lifecycle(state: dict[str, Any], *, limit: int) -> tuple[LifecycleEvent, ...]:
    """Project the tail of the lifecycle log into model rows."""
    raw = state.get("lifecycle")
    if not isinstance(raw, list):
        return ()
    rows = [row for row in raw if isinstance(row, dict)]
    tail = rows[-limit:] if limit > 0 else []
    return tuple(
        LifecycleEvent(
            seq=to_int(row.get("seq")),
            ts=row.get("ts") or None,
            phase=row.get("phase") or None,
            step=row.get("step") or None,
            label=row.get("label") or None,
            status=row.get("status") or None,
            detail=row.get("detail") or None,
            duration_s=to_float(row.get("duration_s")),
        )
        for row in tail
    )


def load_snapshot(
    session_dir: Path | str | None = None,
    *,
    model: str | None = None,
    now_unix: Callable[[], float] | None = None,
    stale_after_s: float = DEFAULT_STALE_AFTER_S,
    lifecycle_limit: int = DEFAULT_LIFECYCLE_LIMIT,
    activity: bool = True,
    extras: dict[str, Any] | None = None,
    extra_reads: list[tuple[str, Any]] | None = None,
) -> Snapshot | None:
    """Read one session directory and return an immutable snapshot.

    Args:
        session_dir: Session to read; auto-resolved when ``None``.
        model: Optional model basename to narrow auto-resolution.
        now_unix: Clock override. Injecting one makes every derived value
            deterministic, which is how the tests avoid patching ``time``.
        stale_after_s: ``state.json`` age past which a live owner reads STALE.
        lifecycle_limit: Number of trailing lifecycle events to carry.
        activity: Read sub-phase activity (run heartbeats, recent writes, GEAK
            progress). Disable for the cheapest possible read.
        extras: Out-of-band values folded into the snapshot, keyed by field
            name — used by the background collector to attach ``gpus``,
            ``server`` and ``source_health``, which are gathered on their own
            cadences rather than inline here.
        extra_reads: ``(source_name, SourceResult)`` pairs from sources the
            caller ran itself. They join the inline reads for warning and
            health purposes, so a caller-run probe that fails is reported the
            same way an inline one would be rather than vanishing.

    Returns:
        The snapshot, or ``None`` when no session directory could be resolved.
        A resolved directory with no readable artifacts still yields a snapshot
        — carrying warnings, which is more useful than nothing.
    """
    resolved = resolve_session_dir(session_dir, model=model)
    if resolved is None:
        return None

    clock: Callable[[], float] = now_unix if now_unix is not None else __import__("time").time
    now = float(clock())

    warnings: list[str] = []

    state_src, manifest_src, lock_src = StateFileSource(), ManifestSource(), LockFileSource()
    db_src = CoordinatorDbSource()
    step_src = CurrentStepSource()

    state_res = state_src.read(resolved)
    manifest_res = manifest_src.read(resolved)
    lock_res = lock_src.read(resolved)
    db_res = db_src.read(resolved, now_unix=now)
    step_res = step_src.read(resolved)

    reads = [
        (state_src.name, state_res),
        (manifest_src.name, manifest_res),
        (lock_src.name, lock_res),
        (db_src.name, db_res),
        (step_src.name, step_res),
    ]

    activity_res = None
    geak_res = None
    if activity:
        activity_src, geak_src = ActivitySource(), GeakSource()
        activity_res = activity_src.read(resolved, now_unix=now)
        geak_res = geak_src.read(resolved)
        reads.append((activity_src.name, activity_res))
        reads.append((geak_src.name, geak_res))

    reads.extend(extra_reads or [])

    for src_name, res in reads:
        warning = warning_for(src_name, res)
        if warning:
            warnings.append(warning)

    activity_data: dict[str, Any] = activity_res.data if (activity_res and activity_res.ok) else {}
    running_work: tuple[RunningWork, ...] = tuple(activity_data.get("running_work", ()) or ())
    activity_rows: tuple[ActivityEntry, ...] = tuple(activity_data.get("activity", ()) or ())
    last_activity_age = to_float(activity_data.get("last_activity_age_s"))
    if activity_data.get("truncated"):
        warnings.append("activity: walk hit its budget; showing a partial view of recent writes")

    state: dict[str, Any] = {}
    state_mtime = 0.0
    if state_res.ok:
        state = state_res.data["state"]
        state_mtime = float(state_res.data["mtime_unix"])

    manifest: dict[str, Any] = manifest_res.data if manifest_res.ok else {}
    lock: dict[str, Any] | None = lock_res.data if lock_res.ok else None

    state_age = max(0.0, now - state_mtime) if state_mtime > 0 else None
    freshness = Freshness.UNKNOWN
    if state_age is not None:
        freshness = Freshness.FRESH if state_age <= stale_after_s else Freshness.STALE

    totals = state.get("phase_elapsed_totals")
    # Must be a real dict: ``phase_cumulative_seconds`` guards with
    # ``isinstance(totals, dict)`` and would silently bank zero otherwise.
    totals_dict: dict[str, float] = {}
    if isinstance(totals, dict):
        for key, value in totals.items():
            coerced = to_float(value)
            if coerced is not None:
                totals_dict[str(key).strip().upper()] = coerced
    if not totals_dict:
        # Pre-``phase_elapsed_totals`` state schema: rebuild what we can from
        # the (capped) transition log rather than reporting every completed
        # phase as zero. Lower bound by construction — see the helper.
        totals_dict = elapsed_totals_from_history(state.get("phase_history"))

    budget = state.get("phase_budget_pct")
    budget_dict: dict[str, float] = {}
    if isinstance(budget, dict):
        for key, value in budget.items():
            coerced = to_float(value)
            if coerced is not None:
                budget_dict[str(key)] = coerced

    db_data: dict[str, Any] = db_res.data if db_res.ok else {}
    counts_raw: dict[str, int] = db_data.get("task_counts", {}) or {}

    result = _build_result(state)
    liveness = _derive_liveness(
        lock,
        state_age_s=state_age,
        activity_age_s=last_activity_age,
        now_unix=now,
        stale_after_s=stale_after_s,
        stop_reason=result.stop_reason,
    )
    progress_now = _progress_clock(
        now_unix=now,
        state_mtime_unix=_last_activity_unix(state, state_mtime_unix=state_mtime),
        liveness=liveness,
        stop_reason=result.stop_reason,
    )

    base = Snapshot(
        phase=str(state.get("phase") or "").strip().upper(),
        phase_started_unix=to_float(state.get("phase_started_unix"), 0.0) or 0.0,
        phase_elapsed_totals=totals_dict,
        phase_budget_pct=budget_dict,
        max_minutes=to_int(state.get("max_minutes"), 0) or 0,
        cycle_minutes=to_float(state.get("cycle_minutes"), 0.0) or 0.0,
        start_ts=str(state.get("start_ts") or ""),
        explore_elapsed_accum_s=to_float(state.get("explore_elapsed_accum_s")),
        session=_build_session_info(resolved, manifest, state),
        macro_cycle=to_int(state.get("macro_cycle"), 0) or 0,
        tick=to_int(state.get("tick"), 0) or 0,
        liveness=liveness,
        freshness=freshness,
        observed_at_unix=now,
        state_age_s=state_age,
        owner_pid=to_int((lock or {}).get("pid")),
        lanes=tuple(
            LaneOccupancy(
                lane=str(row["lane"]),
                held=len(row["holders"]),
                capacity=int(row["capacity"]),
                holders=tuple(row["holders"]),
            )
            for row in db_data.get("lanes", []) or []
        ),
        gpu_leases=tuple(
            GpuLease(
                gpu_id=int(row["gpu_id"]),
                holder_id=str(row["holder_id"]),
                task_id=str(row["task_id"]),
                expires_at=row.get("expires_at"),
                expired=bool(row.get("expired")),
            )
            for row in db_data.get("gpu_leases", []) or []
        ),
        tasks=TaskCounts(
            queued=int(counts_raw.get("queued", 0)),
            running=int(counts_raw.get("running", 0)),
            succeeded=int(counts_raw.get("succeeded", 0)),
            failed=int(counts_raw.get("failed", 0)),
            cancelled=int(counts_raw.get("cancelled", 0)),
        ),
        running_tasks=tuple(
            RunningTask(
                task_id=str(row["task_id"]),
                kind=str(row["kind"]),
                state=str(row["state"]),
                updated_at=row.get("updated_at"),
            )
            for row in db_data.get("running_tasks", []) or []
        ),
        result=result,
        lifecycle=_build_lifecycle(state, limit=lifecycle_limit),
        current_action=(state.get("current_action") or None),
        current_step=(step_res.data if step_res.ok else None),
        running_work=running_work,
        activity=activity_rows,
        geak=(geak_res.data if (geak_res and geak_res.ok) else None),
        last_activity_age_s=last_activity_age,
        source_health=_build_source_health(reads, now_unix=now),
        warnings=tuple(warnings),
        _now_unix=clock,
    )
    if extras:
        base = replace_derived(base, **extras)

    # Phase progress needs the assembled snapshot as its "frozen state view",
    # so it is derived in a second pass and folded in. Duration math uses
    # ``progress_now`` (pinned to the last write for an ended run), not the
    # observation time — see :func:`_progress_clock`.
    elapsed_s, remaining_s = session_timing(base, now_unix=progress_now)
    return replace_derived(
        base,
        phases=build_phase_progress(base, now_unix=progress_now),
        session_elapsed_s=elapsed_s,
        session_remaining_s=remaining_s,
    )


def replace_derived(snapshot: Snapshot, **changes: Any) -> Snapshot:
    """Return a copy of ``snapshot`` with derived fields filled in.

    Args:
        snapshot: The partially-built snapshot.
        **changes: Derived field values.

    Returns:
        A new frozen snapshot.
    """
    import dataclasses

    return dataclasses.replace(snapshot, **changes)
