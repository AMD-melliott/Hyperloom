# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Live counters from the inference server's Prometheus ``/metrics`` endpoint.

Finding the endpoint is most of the work. The persistent benchmark server
takes a fresh port on every launch, and the only record of it is the
lifecycle pid file the optimizer keeps under ``runs/`` while the server runs.
So discovery walks a ladder from explicit configuration, through this
session's live lifecycle server, down to the host's framework-default ports,
and gives up quietly rather than guessing loudly.

An absent server is ``ABSENT``, never ``ERROR``. There is no server at all
during a KERNEL_AGENT phase — the framework is torn down while kernels are
compiled and benchmarked — and reporting that as a failure would train
operators to ignore the warning line.

Throughput is derived from the delta between two polls of the monotonic token
counters, so the first sample yields ``None``. A rate genuinely needs two
observations; reporting ``0.0`` for "not known yet" is indistinguishable from a
wedged server.

Stdlib only — ``urllib.request`` rather than ``httpx`` — because this package
declares no third-party dependencies.
"""

from __future__ import annotations

import json
import os
import re
import time
import urllib.error
import urllib.request
from pathlib import Path

from ..model import ServerMetrics
from .base import SourceResult
from .lockfile import pid_alive


ENV_SERVER_URL = "HYPERLOOM_VLLM_URL"

DEFAULT_BASE_URL = "http://127.0.0.1:8000"
DEFAULT_TIMEOUT_S = 3.0

# Ports worth probing when socket discovery finds nothing. vLLM defaults to
# 8000; 8080 is the common override in this repo's launch scripts.
FALLBACK_PORTS: tuple[int, ...] = (8000, 8080)

# ``runs/**/{framework}_{port}.pid`` holding ``PID [PGID]``, beside a ``.json``
# meta; written by the orchestrator's server-lifecycle protocol and removed on
# teardown.
_LIFECYCLE_PID_FILE = re.compile(r"^.+_(?P<port>\d+)\.pid$")

# ``metric_name{labels} value`` — labels are ignored because a single-model
# server emits one series per metric and summing across labels would be wrong
# for gauges anyway.
_SAMPLE = re.compile(r"^(?P<name>[a-zA-Z_:][a-zA-Z0-9_:]*)(?:\{[^}]*\})?\s+(?P<value>[-+0-9.eENaN]+)\s*$")

# Metric name to model field. Both the modern ``vllm:`` prefixed names and the
# unprefixed forms older builds emit are accepted.
_WANTED: dict[str, str] = {
    "vllm:num_requests_running": "requests_running",
    "vllm:num_requests_waiting": "requests_waiting",
    "vllm:gpu_cache_usage_perc": "kv_cache_pct",
    "vllm:prompt_tokens_total": "prompt_tokens_total",
    "vllm:generation_tokens_total": "generation_tokens_total",
    "num_requests_running": "requests_running",
    "num_requests_waiting": "requests_waiting",
    "gpu_cache_usage_perc": "kv_cache_pct",
    "prompt_tokens_total": "prompt_tokens_total",
    "generation_tokens_total": "generation_tokens_total",
}


def parse_prometheus(text: str) -> dict[str, float]:
    """Extract the metrics of interest from a Prometheus text exposition.

    Args:
        text: Raw ``/metrics`` body.

    Returns:
        Model field name to value, omitting anything absent or unparseable.
    """
    found: dict[str, float] = {}
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        match = _SAMPLE.match(line)
        if match is None:
            continue
        field = _WANTED.get(match.group("name"))
        if field is None or field in found:
            continue
        try:
            value = float(match.group("value"))
        except ValueError:
            continue
        if value != value:  # NaN — the server has the metric but no reading
            continue
        found[field] = value
    return found


def _listening_ports() -> tuple[int, ...]:
    """Return locally-listening TCP ports, newest kernel table first.

    Reads ``/proc/net/tcp`` directly rather than shelling out to ``ss``, which
    keeps the probe dependency-free and fast. Returns an empty tuple on any
    platform that lacks the file.

    Returns:
        Ports in listen state.
    """
    ports: list[int] = []
    for proc_file in ("/proc/net/tcp", "/proc/net/tcp6"):
        try:
            with open(proc_file, "r", encoding="utf-8", errors="replace") as handle:
                lines = handle.readlines()[1:]
        except OSError:
            continue
        for line in lines:
            parts = line.split()
            if len(parts) < 4:
                continue
            # State 0A is TCP_LISTEN.
            if parts[3] != "0A":
                continue
            local = parts[1]
            _, _, port_hex = local.partition(":")
            try:
                ports.append(int(port_hex, 16))
            except ValueError:
                continue
    return tuple(sorted(set(ports)))


def _configured_url() -> str | None:
    configured = os.environ.get(ENV_SERVER_URL, "").strip()
    return configured.rstrip("/") or None


def _meta_base_url(meta_file: Path, port: str) -> str | None:
    try:
        meta = json.loads(meta_file.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(meta, dict):
        return None
    if isinstance(meta.get("base_url"), str) and meta["base_url"]:
        return meta["base_url"].rstrip("/")
    if isinstance(meta.get("host"), str) and meta["host"]:
        return f"http://{meta['host']}:{port}"
    return None


def find_session_server(session_dir: Path) -> str | None:
    """Resolve the inference server this session is benchmarking right now.

    ``$HYPERLOOM_VLLM_URL`` wins; otherwise the most recently started lifecycle
    server under ``runs/`` whose pid is alive, at the host its meta records or
    ``127.0.0.1``. Never probes host-wide ports, so another session's or an
    unrelated service is not mistaken for this one.

    Returns:
        A base URL, or ``None`` when this session has no live server.
    """
    configured = _configured_url()
    if configured:
        return configured
    newest: tuple[float, str] | None = None
    for pid_file in (Path(session_dir) / "runs").rglob("*.pid"):
        match = _LIFECYCLE_PID_FILE.match(pid_file.name)
        if match is None:
            continue
        try:
            fields = pid_file.read_text(encoding="utf-8").split()
            started = pid_file.stat().st_mtime
        except OSError:
            continue
        if not fields or not pid_alive(fields[0]):
            continue
        port = match.group("port")
        url = _meta_base_url(pid_file.with_suffix(".json"), port) or f"http://127.0.0.1:{port}"
        if newest is None or started > newest[0]:
            newest = (started, url)
    return newest[1] if newest else None


def discover_base_url(session_dir: Path | None = None) -> str | None:
    """Resolve the inference server's base URL.

    Order: :func:`find_session_server` when a session is given (else just
    ``$HYPERLOOM_VLLM_URL``), then a listening port that matches one of the
    known framework defaults, then nothing.

    Returns:
        A base URL, or ``None`` when no plausible server is listening.
    """
    found = find_session_server(session_dir) if session_dir is not None else _configured_url()
    if found:
        return found
    listening = set(_listening_ports())
    for port in FALLBACK_PORTS:
        if port in listening:
            return f"http://127.0.0.1:{port}"
    return None


def _fetch(url: str, *, timeout_s: float) -> str | None:
    """GET ``url`` and return its body, or ``None`` when unreachable."""
    request = urllib.request.Request(url, headers={"Accept": "text/plain"})  # http(s) only, caller-built
    try:
        with urllib.request.urlopen(request, timeout=timeout_s) as response:
            return response.read().decode("utf-8", errors="replace")
    except (urllib.error.URLError, OSError, ValueError):
        return None


class ServerMetricsSource:
    """Scrapes ``/metrics`` from a running inference server."""

    name = "server"

    def __init__(self, *, base_url: str | None = None, timeout_s: float = DEFAULT_TIMEOUT_S) -> None:
        """Initialise the source.

        Args:
            base_url: Explicit server base URL; discovered per-poll when
                ``None``, since a server comes and goes across phases.
            timeout_s: HTTP timeout. Short by design — a hung server must not
                hold up the collector.
        """
        self._base_url = base_url.rstrip("/") if base_url else None
        self._timeout_s = timeout_s
        # Previous ``(observed_unix, total_tokens)`` for rate derivation.
        self._previous: tuple[float, float] | None = None

    def read(self, session_dir: Path | None = None, *, now_unix: float | None = None) -> SourceResult:
        """Scrape the server's metrics endpoint.

        Args:
            session_dir: Session whose lifecycle server to prefer when no
                base URL was given.
            now_unix: Observation time, used to derive the token rate.

        Returns:
            ``OK`` with a :class:`~hyperloom.observability.model.ServerMetrics`,
            ``ABSENT`` when nothing is listening, or ``ERROR`` when a server
            answered but its response was unusable.
        """
        now = float(now_unix if now_unix is not None else time.time())

        base = self._base_url or discover_base_url(session_dir)
        if not base:
            self._previous = None
            return SourceResult.absent()

        body = _fetch(f"{base}/metrics", timeout_s=self._timeout_s)
        if body is None:
            # Discovery said something was listening, but it did not answer.
            # Between phases that is a server mid-teardown, not a fault.
            self._previous = None
            return SourceResult.absent()

        values = parse_prometheus(body)
        if not values:
            return SourceResult.error(f"{base}/metrics returned no recognised vllm metrics")

        tput = self._rate(values, now_unix=now)

        return SourceResult.hit(
            ServerMetrics(
                url=base,
                requests_running=_as_int(values.get("requests_running")),
                requests_waiting=_as_int(values.get("requests_waiting")),
                kv_cache_pct=_as_pct(values.get("kv_cache_pct")),
                prompt_tokens_total=values.get("prompt_tokens_total"),
                generation_tokens_total=values.get("generation_tokens_total"),
                tput_tok_s=tput,
            )
        )

    def _rate(self, values: dict[str, float], *, now_unix: float) -> float | None:
        """Derive output tokens per second from consecutive counter readings.

        Args:
            values: Parsed metric values.
            now_unix: Observation time.

        Returns:
            Tokens per second, or ``None`` on the first sample or after a
            counter reset.
        """
        total = values.get("generation_tokens_total")
        if total is None:
            self._previous = None
            return None

        previous = self._previous
        self._previous = (now_unix, total)
        if previous is None:
            return None

        prev_unix, prev_total = previous
        span = now_unix - prev_unix
        if span <= 0 or total < prev_total:
            # A counter that went backwards means the server restarted; the
            # delta across that boundary is meaningless.
            return None
        return (total - prev_total) / span


def _as_int(value: float | None) -> int | None:
    """Round a gauge to an integer, preserving ``None``."""
    return None if value is None else int(round(value))


def _as_pct(value: float | None) -> float | None:
    """Normalise a cache-usage gauge to percent.

    vLLM reports ``gpu_cache_usage_perc`` as a ``0..1`` ratio despite the name,
    but some builds emit ``0..100``. Values at or below 1 are treated as a
    ratio, which is correct except for the degenerate case of a server at
    exactly 1% usage reporting in the percent convention.

    Args:
        value: Raw gauge reading.

    Returns:
        Percentage in ``0..100``, or ``None``.
    """
    if value is None:
        return None
    return value * 100.0 if value <= 1.0 else value
