# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Background collection, so the display never waits on a probe.

The one-shot path gathers inline and that is correct for it — a single render
should be deterministic and finish. A live view is different: it repaints on a
timer, and every collector it calls inline becomes a chance to freeze the
frame. ``amd-smi`` takes 0.3 s normally but can hang indefinitely against a
wedged driver; an HTTP scrape waits out its timeout; a session directory on a
network filesystem can stall on ``stat``. Any one of those, called from the
paint loop, stops the clock the operator is watching.

So collection runs on its own threads and the paint loop only ever reads a
cache. The two consequences that shape everything here:

* **A slow source degrades only itself.** Sources are scheduled independently,
  each with its own interval and timeout, and a source still in flight is
  skipped rather than queued again. One wedged probe cannot delay another or
  pile up work.
* **Failure shows the last good value, not a gap.** A source that breaks keeps
  its previous reading, stamped with the age it actually has and flagged in
  :class:`~hyperloom.observability.model.SourceHealth`. Blanking the field
  would throw away true information; silently showing it as current would be a
  lie. Showing it with its age is neither.

What is never done is fabricate. A value that has never been read stays
``None`` and renders as an em dash. ``0`` is reserved for measurements that
were actually zero.

Every thread is a daemon and every probe is individually bounded, so a hung
collector can delay a clean shutdown by at most one timeout — and cannot
prevent it.
"""

from __future__ import annotations

import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from .model import SourceHealth, SourceOutcome
from .sources.base import SourceResult


# Imported lazily inside methods: ``assemble`` reaches the orchestrator through
# ``progress``, and keeping that off this module's import path means the cheap
# parts stay cheap to import.
DEFAULT_LIFECYCLE_LIMIT = 12


log = logging.getLogger(__name__)


# Per-source refresh intervals. The session artifacts are cheap local reads and
# can be frequent; the probes that leave the process are slower and their data
# changes on a human timescale anyway.
DEFAULT_INTERVALS: dict[str, float] = {
    "session": 2.0,
    "gpu": 5.0,
    "server": 5.0,
}

# Per-source wall-clock ceilings. Deliberately generous relative to measured
# cost (session read ~70 ms, amd-smi ~320 ms, HTTP ~50 ms on localhost) so a
# merely slow poll is not mistaken for a hung one.
DEFAULT_TIMEOUTS: dict[str, float] = {
    "session": 20.0,
    "gpu": 10.0,
    "server": 5.0,
}


@dataclass(frozen=True)
class CachedValue:
    """The last reading of one source, with everything needed to caveat it.

    Attributes:
        value: Last **successful** payload, retained across later failures.
            ``None`` means never successfully read — not zero, not empty.
        observed_at_unix: When ``value`` was read; ``0.0`` when never.
        outcome: Outcome of the most recent *attempt*, which may differ from
            the state of ``value``: a stale-but-present value alongside
            ``ERROR`` is the case this whole module exists to represent.
        error: Message from the most recent failed attempt.
        duration_s: How long the most recent attempt took.
        consecutive_failures: Failures since the last success.
        attempts: Total attempts, successful or not.
    """

    value: Any = None
    observed_at_unix: float = 0.0
    outcome: SourceOutcome = SourceOutcome.ABSENT
    error: str | None = None
    duration_s: float | None = None
    consecutive_failures: int = 0
    attempts: int = 0

    def age_s(self, now_unix: float) -> float | None:
        """Age of the cached value, or ``None`` when never read."""
        if not self.observed_at_unix:
            return None
        return max(0.0, float(now_unix) - self.observed_at_unix)

    def health(self, name: str, *, now_unix: float) -> SourceHealth:
        """Project this entry into the model's health row."""
        return SourceHealth(
            name=name,
            outcome=self.outcome,
            age_s=self.age_s(now_unix),
            error=self.error,
            consecutive_failures=self.consecutive_failures,
            duration_s=self.duration_s,
        )


class MetricCache:
    """Thread-safe store of the latest reading per source.

    Reads never block on collection — they take a short lock held only for a
    dict lookup — which is what lets the paint loop run at a fixed rate
    regardless of what the collectors are doing.
    """

    def __init__(self) -> None:
        """Create an empty cache."""
        self._lock = threading.Lock()
        self._entries: dict[str, CachedValue] = {}

    def get(self, name: str) -> CachedValue:
        """Return the entry for ``name``, or an empty one when unseen."""
        with self._lock:
            return self._entries.get(name, CachedValue())

    def snapshot(self) -> dict[str, CachedValue]:
        """Return a shallow copy of every entry."""
        with self._lock:
            return dict(self._entries)

    def record(self, name: str, result: SourceResult, *, now_unix: float, duration_s: float) -> None:
        """Fold one attempt into the cache.

        A successful read replaces the value. ``ABSENT`` and ``ERROR`` update
        the outcome and counters but **keep the previous value**, so a probe
        that breaks mid-run leaves the last good reading on screen with an
        honest age rather than a hole.

        Args:
            name: Source name.
            result: Outcome of the attempt.
            now_unix: When the attempt completed.
            duration_s: How long it took.
        """
        with self._lock:
            previous = self._entries.get(name, CachedValue())
            if result.outcome is SourceOutcome.OK:
                self._entries[name] = CachedValue(
                    value=result.data,
                    observed_at_unix=now_unix,
                    outcome=SourceOutcome.OK,
                    error=None,
                    duration_s=duration_s,
                    consecutive_failures=0,
                    attempts=previous.attempts + 1,
                )
                return
            failures = previous.consecutive_failures + (1 if result.outcome is SourceOutcome.ERROR else 0)
            self._entries[name] = CachedValue(
                value=previous.value,
                observed_at_unix=previous.observed_at_unix,
                outcome=result.outcome,
                error=result.message,
                duration_s=duration_s,
                consecutive_failures=failures,
                attempts=previous.attempts + 1,
            )

    def health(self, *, now_unix: float) -> tuple[SourceHealth, ...]:
        """Return a health row per source, ordered by name."""
        return tuple(entry.health(name, now_unix=now_unix) for name, entry in sorted(self.snapshot().items()))


@dataclass
class _Registration:
    """One scheduled source."""

    name: str
    fn: Callable[[], SourceResult]
    interval_s: float
    timeout_s: float
    next_due: float = 0.0
    in_flight: bool = False
    started_at: float = field(default=0.0)


class Collector:
    """Polls registered sources on independent cadences into a cache.

    Usage is a context manager; ``__exit__`` stops the scheduler and does not
    wait for in-flight probes, which is what keeps Ctrl-C responsive when a
    source is hung.
    """

    def __init__(self, *, cache: MetricCache | None = None, clock: Callable[[], float] = time.time) -> None:
        """Initialise the collector.

        Args:
            cache: Cache to populate; a fresh one is created when omitted.
            clock: Wall-clock source, injectable for tests.
        """
        self.cache = cache if cache is not None else MetricCache()
        self._clock = clock
        self._sources: list[_Registration] = []
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._pool: ThreadPoolExecutor | None = None
        self._lock = threading.Lock()

    def register(
        self,
        name: str,
        fn: Callable[[], SourceResult],
        *,
        interval_s: float | None = None,
        timeout_s: float | None = None,
    ) -> None:
        """Add a source to the schedule.

        Args:
            name: Cache key and health-row label.
            fn: Zero-argument probe returning a :class:`SourceResult`. Must not
                raise; exceptions are caught and recorded as ``ERROR`` anyway.
            interval_s: Seconds between polls; falls back to
                :data:`DEFAULT_INTERVALS` then 5 s.
            timeout_s: Soft ceiling used to flag a probe as hung. The probe
                itself must enforce its own hard timeout — a thread cannot be
                interrupted from outside, which is why every source in this
                package bounds its own subprocess and socket calls.
        """
        self._sources.append(
            _Registration(
                name=name,
                fn=fn,
                interval_s=float(interval_s if interval_s is not None else DEFAULT_INTERVALS.get(name, 5.0)),
                timeout_s=float(timeout_s if timeout_s is not None else DEFAULT_TIMEOUTS.get(name, 15.0)),
            )
        )

    def collect_once(self, name: str | None = None) -> None:
        """Run every registered source synchronously.

        Used for the initial fill, so the first frame is not empty, and by
        tests that want deterministic ordering.

        Args:
            name: Run only this source when given.
        """
        for registration in self._sources:
            if name is not None and registration.name != name:
                continue
            self._run(registration)

    def start(self) -> Collector:
        """Start the scheduler thread.

        Returns:
            ``self``, for use as a context manager.
        """
        if self._thread is not None:
            return self
        # One worker per source: a source is never queued behind another, so a
        # hung probe cannot starve the rest. The count is small and bounded by
        # registration, not by workload.
        self._pool = ThreadPoolExecutor(
            max_workers=max(1, len(self._sources)),
            thread_name_prefix="hl-observe",
        )
        self._thread = threading.Thread(target=self._loop, name="hl-observe-sched", daemon=True)
        self._thread.start()
        return self

    def stop(self) -> None:
        """Signal the scheduler to stop, without waiting for in-flight probes."""
        self._stop.set()
        thread, self._thread = self._thread, None
        if thread is not None:
            # Bounded join: the scheduler only ever sleeps in small slices, so
            # this returns promptly even mid-cycle.
            thread.join(timeout=2.0)
        pool, self._pool = self._pool, None
        if pool is not None:
            # Explicitly do not wait. An in-flight ``amd-smi`` against a wedged
            # driver would otherwise hold up the operator's Ctrl-C; the threads
            # are daemons and will not outlive the process.
            pool.shutdown(wait=False, cancel_futures=True)

    def __enter__(self) -> Collector:
        """Start collecting on context entry."""
        return self.start()

    def __exit__(self, *_exc: object) -> None:
        """Stop collecting on context exit."""
        self.stop()

    def hung_sources(self, *, now_unix: float | None = None) -> tuple[str, ...]:
        """Names of sources that have been in flight past their timeout."""
        now = float(now_unix if now_unix is not None else self._clock())
        with self._lock:
            return tuple(reg.name for reg in self._sources if reg.in_flight and (now - reg.started_at) > reg.timeout_s)

    def _loop(self) -> None:
        """Scheduler: submit due sources, never wait on them."""
        while not self._stop.is_set():
            now = float(self._clock())
            for registration in self._sources:
                with self._lock:
                    if registration.in_flight or now < registration.next_due:
                        continue
                    registration.in_flight = True
                    registration.started_at = now
                pool = self._pool
                if pool is None:  # pragma: no cover - stop() raced us
                    with self._lock:
                        registration.in_flight = False
                    continue
                try:
                    pool.submit(self._run, registration)
                except RuntimeError:  # pragma: no cover - pool shut down mid-submit
                    with self._lock:
                        registration.in_flight = False
            # Short slices keep stop() responsive without busy-waiting.
            self._stop.wait(0.25)

    def _run(self, registration: _Registration) -> None:
        """Execute one probe and record its outcome.

        Args:
            registration: The source to run.
        """
        started = time.monotonic()
        try:
            result = registration.fn()
        except Exception as exc:  # noqa: BLE001 - a source must never take the collector down
            log.debug("collector: %s raised", registration.name, exc_info=True)
            result = SourceResult.error(f"{type(exc).__name__}: {exc}")
        duration = time.monotonic() - started
        try:
            self.cache.record(
                registration.name,
                result,
                now_unix=float(self._clock()),
                duration_s=duration,
            )
        finally:
            with self._lock:
                registration.in_flight = False
                # Schedule from completion, not from the due time, so a probe
                # that takes longer than its interval does not immediately
                # become due again and monopolise its worker.
                registration.next_due = float(self._clock()) + registration.interval_s


class SessionMonitor:
    """A :class:`Collector` wired to one session, producing render-ready snapshots.

    Composition root for live mode. The session artifacts, the GPU probe and
    the server scrape are three independent cadences feeding one cache;
    :meth:`current` joins them and advances the clock to paint time, doing no
    I/O at all.
    """

    SESSION = "session"
    GPU = "gpu"
    SERVER = "server"

    def __init__(
        self,
        session_dir: Path,
        *,
        gpu: bool = True,
        server: bool = True,
        server_url: str | None = None,
        lifecycle_limit: int = DEFAULT_LIFECYCLE_LIMIT,
        session_interval_s: float | None = None,
        clock: Callable[[], float] = time.time,
    ) -> None:
        """Bind collectors to a session.

        Args:
            session_dir: Absolute session root.
            gpu: Register the GPU probe.
            server: Register the inference-server scrape.
            server_url: Explicit server base URL; discovered per poll when
                ``None``.
            lifecycle_limit: Trailing lifecycle events to carry.
            session_interval_s: Seconds between session-artifact reads;
                defaults to :data:`DEFAULT_INTERVALS`.
            clock: Wall-clock source, injectable for tests.
        """
        from .sources import GpuSource, ServerMetricsSource

        self.session_dir = Path(session_dir)
        self._clock = clock
        self._lifecycle_limit = lifecycle_limit
        self.collector = Collector(clock=clock)

        self.collector.register(self.SESSION, self._read_session, interval_s=session_interval_s)
        if gpu:
            gpu_source = GpuSource()
            self.collector.register(self.GPU, lambda: gpu_source.read())
        if server:
            server_source = ServerMetricsSource(base_url=server_url)
            self.collector.register(self.SERVER, lambda: server_source.read(now_unix=self._clock()))

    def _read_session(self) -> SourceResult:
        """Read the session artifacts into a snapshot."""
        from .assemble import load_snapshot

        snapshot = load_snapshot(
            self.session_dir,
            now_unix=self._clock,
            lifecycle_limit=self._lifecycle_limit,
        )
        if snapshot is None:
            return SourceResult.error(f"session directory disappeared: {self.session_dir}")
        return SourceResult.hit(snapshot)

    def start(self) -> SessionMonitor:
        """Fill the cache once, then start background collection.

        The synchronous first pass means the opening frame is populated rather
        than a screen of em dashes that fills in a moment later.

        Returns:
            ``self``.
        """
        self.collector.collect_once()
        self.collector.start()
        return self

    def stop(self) -> None:
        """Stop background collection."""
        self.collector.stop()

    def __enter__(self) -> SessionMonitor:
        """Start on context entry."""
        return self.start()

    def __exit__(self, *_exc: object) -> None:
        """Stop on context exit."""
        self.stop()

    def current(self, *, now_unix: float | None = None) -> Any | None:
        """Join the cached readings into one snapshot advanced to paint time.

        Pure cache reads plus arithmetic — no filesystem, no subprocess, no
        sockets — so this is safe to call on every frame.

        Args:
            now_unix: Paint time; defaults to the injected clock.

        Returns:
            A :class:`~hyperloom.observability.model.Snapshot`, or ``None``
            when the session has never been read successfully.
        """
        from .assemble import replace_derived
        from .progress import extrapolate_to

        now = float(now_unix if now_unix is not None else self._clock())
        entries = self.collector.cache.snapshot()

        session_entry = entries.get(self.SESSION, CachedValue())
        snapshot = session_entry.value
        if snapshot is None:
            return None

        gpu_entry = entries.get(self.GPU)
        server_entry = entries.get(self.SERVER)

        health = list(snapshot.source_health)
        # Collector-owned rows replace the inline ones: their ages are real,
        # where an inline read is by definition zero seconds old.
        health = [row for row in health if row.name not in entries]
        for name, entry in sorted(entries.items()):
            health.append(entry.health(name, now_unix=now))

        warnings = list(snapshot.warnings)
        for name in self.collector.hung_sources(now_unix=now):
            warnings.append(f"{name}: collection has been running longer than its timeout")

        joined = replace_derived(
            snapshot,
            gpus=(gpu_entry.value if gpu_entry else None),
            server=(server_entry.value if server_entry else None),
            source_health=tuple(health),
            warnings=tuple(dict.fromkeys(warnings)),
        )
        return extrapolate_to(joined, now)
