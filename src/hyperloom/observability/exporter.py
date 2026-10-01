# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Serve a session's snapshot as Prometheus metrics.

Spawned by the optimizer for the life of one run (see
``inference_optimizer/cli/metrics_exporter.py``), or run by hand::

    python -m hyperloom.observability.exporter --session-dir SD --listen 127.0.0.1:9477

Endpoints: ``/metrics`` (exposition format), ``/healthz``, and
``/sd/inference`` (Prometheus HTTP-SD JSON naming the inference server, whose
port changes between launches). Every request renders from the monitor's cache,
so a scrape never waits on session I/O. Like the rest of the package, this
never writes the session directory.
"""

from __future__ import annotations

import json
import logging
import threading
from collections.abc import Callable
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlsplit

from .collector import SessionMonitor
from .render.prometheus import PROMETHEUS_CONTENT_TYPE, ExporterInfo, render_prometheus, session_labels
from .sources.base import SourceResult
from .sources.server import discover_base_url

log = logging.getLogger(__name__)

SD_SOURCE = "inference_sd"
SD_INTERVAL_SEC = 15.0


def exporter_version() -> str:
    """Installed Hyperloom version, or ``"unknown"`` for an uninstalled tree."""
    from importlib.metadata import PackageNotFoundError, version

    try:
        return version("hyperloom")
    except PackageNotFoundError:
        return "unknown"


def _read_inference_target() -> SourceResult:
    # Always a hit, even with no server: an ABSENT result would keep the
    # previous URL cached and keep advertising a server that has gone away.
    return SourceResult.hit({"url": discover_base_url()})


def register_inference_sd(monitor: SessionMonitor, *, interval_s: float = SD_INTERVAL_SEC) -> None:
    """Poll for the inference server on the monitor's collector."""
    monitor.collector.register(SD_SOURCE, _read_inference_target, interval_s=interval_s)


class ExporterState:
    """What the HTTP handler renders from."""

    def __init__(
        self,
        *,
        monitor: SessionMonitor,
        version: str,
        parent_alive: Callable[[], bool | None] = lambda: None,
    ) -> None:
        self.monitor = monitor
        self.version = version
        self._parent_alive = parent_alive
        self._render_errors = 0
        self._lock = threading.Lock()

    def _count_render_error(self, _family: str) -> None:
        with self._lock:
            self._render_errors += 1

    def metrics_text(self) -> str:
        """Render the current snapshot."""
        with self._lock:
            errors = self._render_errors
        info = ExporterInfo(version=self.version, parent_alive=self._parent_alive(), render_errors_total=errors)
        return render_prometheus(self.monitor.current(), exporter=info, on_family_error=self._count_render_error)

    def sd_json(self) -> str:
        """HTTP-SD target list for the inference server; ``[]`` when none is listening."""
        cached = self.monitor.collector.cache.get(SD_SOURCE).value or {}
        url = cached.get("url")
        if not url:
            return "[]"
        netloc = urlsplit(url).netloc
        if not netloc:
            return "[]"
        snapshot = self.monitor.current()
        labels = session_labels(snapshot) if snapshot is not None else {}
        return json.dumps([{"targets": [netloc], "labels": labels}])


class ExporterHTTPServer(ThreadingHTTPServer):
    """Threaded server that can rebind a port left in TIME_WAIT."""

    allow_reuse_address = True
    daemon_threads = True


def make_server(state: ExporterState, host: str, port: int) -> ExporterHTTPServer:
    """Bind the exporter's HTTP server; raises ``OSError`` when the bind fails."""
    routes: dict[str, tuple[Callable[[], str], str]] = {
        # Resolved per request, not bound at build time.
        "/metrics": (lambda: state.metrics_text(), PROMETHEUS_CONTENT_TYPE),
        "/healthz": (lambda: "ok\n", "text/plain; charset=utf-8"),
        "/sd/inference": (lambda: state.sd_json(), "application/json"),
    }

    class Handler(BaseHTTPRequestHandler):
        server_version = "hyperloom-exporter"

        def do_GET(self) -> None:
            path = self.path.split("?", 1)[0]
            route = routes.get(path)
            if route is None:
                self.send_error(404)
                return
            render, content_type = route
            try:
                body = render().encode("utf-8")
            except Exception:  # noqa: BLE001 - a failed request must not take the server down
                log.warning("exporter: rendering %s failed", path, exc_info=True)
                self.send_error(500)
                return
            self.send_response(200)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, format: str, *args: object) -> None:
            log.debug("exporter: " + format, *args)

    return ExporterHTTPServer((host, port), Handler)
