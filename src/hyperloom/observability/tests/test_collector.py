# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Contract tests for background collection.

The properties asserted here are the reasons the collector exists, so each one
is pinned rather than left to review:

============================================ ==========================================
Case                                         Expectation
============================================ ==========================================
A source hangs past its timeout              Other sources keep collecting
A source hangs past its timeout              It is reported by ``hung_sources``
A source starts failing                      Its last good value survives, with its age
A source has never succeeded                 Value stays ``None`` — never ``0``
A source raises                              Recorded as ``ERROR``, collector survives
``stop()`` with a probe in flight            Returns without waiting for it
Failure then recovery                        ``consecutive_failures`` resets
============================================ ==========================================
"""

from __future__ import annotations

import threading
import time

from hyperloom.observability.collector import CachedValue, Collector, MetricCache
from hyperloom.observability.model import SourceOutcome
from hyperloom.observability.sources.base import SourceResult


def test_failure_keeps_the_last_good_value_with_its_real_age() -> None:
    """A broken probe must degrade to a stale reading, not to a hole.

    Blanking the field throws away information that is still true; showing it
    as current is a lie. Keeping it with an honest age is neither.
    """
    cache = MetricCache()
    cache.record("gpu", SourceResult.hit({"util": 94}), now_unix=100.0, duration_s=0.3)
    cache.record("gpu", SourceResult.error("amd-smi timed out"), now_unix=160.0, duration_s=8.0)

    entry = cache.get("gpu")
    assert entry.value == {"util": 94}
    assert entry.outcome is SourceOutcome.ERROR
    assert entry.error == "amd-smi timed out"
    assert entry.consecutive_failures == 1
    # Age is measured from the last SUCCESS, not the last attempt.
    assert entry.age_s(160.0) == 60.0


def test_never_read_stays_none_rather_than_zero() -> None:
    """A value never observed is ``None``; ``0`` is reserved for a real zero."""
    cache = MetricCache()
    cache.record("server", SourceResult.absent(), now_unix=100.0, duration_s=0.01)

    entry = cache.get("server")
    assert entry.value is None
    assert entry.age_s(200.0) is None
    assert entry.outcome is SourceOutcome.ABSENT
    # ABSENT is not a failure: nothing listening is the normal state between
    # phases, and counting it would train operators to ignore the warning.
    assert entry.consecutive_failures == 0


def test_recovery_resets_the_failure_counter() -> None:
    """A source that comes back must stop being reported as degraded."""
    cache = MetricCache()
    cache.record("gpu", SourceResult.error("boom"), now_unix=10.0, duration_s=0.1)
    cache.record("gpu", SourceResult.error("boom"), now_unix=20.0, duration_s=0.1)
    assert cache.get("gpu").consecutive_failures == 2

    cache.record("gpu", SourceResult.hit(1), now_unix=30.0, duration_s=0.1)
    entry = cache.get("gpu")
    assert entry.consecutive_failures == 0
    assert entry.outcome is SourceOutcome.OK


def test_a_hung_source_does_not_delay_the_others() -> None:
    """The whole point: one wedged probe must not stop the rest collecting.

    ``amd-smi`` against a wedged driver is the real-world case. Inline, it
    would freeze the frame; here it must cost only its own field.
    """
    release = threading.Event()
    collector = Collector()
    collector.register("hung", lambda: (release.wait(30), SourceResult.hit("late"))[1], interval_s=0.05, timeout_s=0.2)
    collector.register("quick", lambda: SourceResult.hit("value"), interval_s=0.05, timeout_s=1.0)

    try:
        collector.start()
        deadline = time.time() + 5.0
        while time.time() < deadline and collector.cache.get("quick").value is None:
            time.sleep(0.02)

        assert collector.cache.get("quick").value == "value"
        assert collector.cache.get("hung").value is None

        # Wait for the hung source to exceed its timeout AND for the fast one
        # to have polled again, which together are the property under test:
        # collection continues while one probe is stuck.
        deadline = time.time() + 10.0
        while time.time() < deadline:
            if "hung" in collector.hung_sources() and collector.cache.get("quick").attempts > 1:
                break
            time.sleep(0.05)

        assert "hung" in collector.hung_sources()
        assert collector.cache.get("quick").attempts > 1
        assert collector.cache.get("hung").attempts == 0
    finally:
        release.set()
        collector.stop()


def test_stop_returns_without_waiting_for_an_in_flight_probe() -> None:
    """Ctrl-C must be responsive even while a probe is stuck.

    Waiting on the pool here would make a wedged GPU driver hold the operator's
    terminal hostage for the length of its timeout.
    """
    release = threading.Event()
    collector = Collector()
    collector.register("hung", lambda: (release.wait(30), SourceResult.hit("late"))[1], interval_s=0.05)

    try:
        collector.start()
        time.sleep(0.4)
        started = time.monotonic()
        collector.stop()
        assert time.monotonic() - started < 3.0
    finally:
        release.set()


def test_a_raising_source_is_recorded_not_propagated() -> None:
    """A source that raises must be an ``ERROR`` row, not a dead collector."""
    collector = Collector()

    def explode() -> SourceResult:
        raise RuntimeError("driver gone")

    collector.register("bad", explode)
    collector.collect_once()

    entry = collector.cache.get("bad")
    assert entry.outcome is SourceOutcome.ERROR
    assert "RuntimeError" in (entry.error or "")
    assert "driver gone" in (entry.error or "")


def test_health_rows_carry_provenance() -> None:
    """Health projection must expose enough to caveat a stale value."""
    cache = MetricCache()
    cache.record("gpu", SourceResult.hit(1), now_unix=100.0, duration_s=0.3)
    cache.record("gpu", SourceResult.error("timed out"), now_unix=130.0, duration_s=8.0)

    (row,) = cache.health(now_unix=160.0)
    assert row.name == "gpu"
    assert row.outcome is SourceOutcome.ERROR
    assert row.age_s == 60.0
    assert row.error == "timed out"
    assert row.degraded is True


def test_empty_cached_value_is_inert() -> None:
    """The default entry must not pretend to be a reading."""
    empty = CachedValue()
    assert empty.value is None
    assert empty.age_s(1000.0) is None
    assert empty.attempts == 0

    health = empty.health("unseen", now_unix=1000.0)
    assert health.outcome is SourceOutcome.ABSENT
    assert health.age_s is None
    # An unseen source is not a broken one, so it stays out of the warning row.
    assert health.degraded is False
