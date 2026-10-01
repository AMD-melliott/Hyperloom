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

import argparse
import errno
import json
import logging
import os
import signal
import sys
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlsplit

from .assemble import resolve_session_dir
from .collector import SessionMonitor
from .render.prometheus import PROMETHEUS_CONTENT_TYPE, ExporterInfo, render_prometheus, session_labels
from .sources.base import SourceResult
from .sources.lockfile import LockFileSource
from .sources.server import discover_base_url

log = logging.getLogger(__name__)

SD_SOURCE = "inference_sd"
SD_INTERVAL_SEC = 15.0
DEFAULT_LISTEN = "127.0.0.1:9477"
DEFAULT_GRACE_SEC = 120.0
WATCHDOG_INTERVAL_SEC = 2.0
BIND_RETRY_SEC = 15.0

# Mirrors ``inference_optimizer/tools/status.py``: the exit code reports whether
# the command ran, so an exporter that could not bind still exits 0.
EXIT_OK = 0
EXIT_CONFIG_ERROR = 3

RUN = "run"
GRACE = "grace"
EXIT = "exit"


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
        # A stalled client must not hold a handler thread forever.
        timeout = 10

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


def parse_listen(value: str) -> tuple[str, int]:
    """Split ``HOST:PORT``; an empty host binds every IPv4 interface."""
    host, sep, port_text = value.rpartition(":")
    if not sep:
        raise ValueError(f"listen address {value!r} is not HOST:PORT")
    if host.startswith("["):
        raise ValueError(f"listen address {value!r}: IPv6 is not supported")
    try:
        port = int(port_text)
    except ValueError:
        raise ValueError(f"listen address {value!r} has a non-numeric port") from None
    if not 0 <= port <= 65535:
        raise ValueError(f"listen address {value!r} has an out-of-range port")
    return host, port


@dataclass
class Watchdog:
    """Decide, once per interval, whether the exporter should keep serving.

    The optimizer spawns the exporter directly, so ``os.getppid()`` stops
    matching the moment the optimizer exits (however it exits) and the child is
    reparented. That avoids the PID-reuse race a bare ``kill(pid, 0)`` has.
    """

    parent_pid: int | None
    session_dir: Path
    grace_s: float
    getppid: Callable[[], int] = os.getppid
    read_lock: Callable[[Path], SourceResult] = field(default_factory=lambda: LockFileSource().read)
    clock: Callable[[], float] = time.monotonic
    _parent_gone_at: float | None = None

    def parent_alive(self) -> bool | None:
        """``None`` when running without a parent."""
        if self.parent_pid is None:
            return None
        return self.getppid() == self.parent_pid

    def poll(self) -> str:
        """Return :data:`RUN`, :data:`GRACE` or :data:`EXIT`."""
        if self.parent_pid is None:
            return RUN
        if self._taken_over():
            return EXIT
        if self.parent_alive():
            return RUN
        now = self.clock()
        if self._parent_gone_at is None:
            self._parent_gone_at = now
        return EXIT if now - self._parent_gone_at >= self.grace_s else GRACE

    def _taken_over(self) -> bool:
        # A resumed run took the session lock: free the port for its exporter.
        result = self.read_lock(self.session_dir)
        if not result.ok or not isinstance(result.data, dict):
            return False
        pid = result.data.get("pid")
        return pid is not None and pid != self.parent_pid and result.data.get("pid_alive") is True


def bind_with_retry(
    state: ExporterState,
    host: str,
    port: int,
    *,
    retry_s: float = BIND_RETRY_SEC,
    sleep: Callable[[float], object] = time.sleep,
    clock: Callable[[], float] = time.monotonic,
) -> ExporterHTTPServer | None:
    """Bind, retrying a busy port for ``retry_s``; ``None`` means give up quietly.

    A truthy return from ``sleep`` (``Event.wait`` when stopped) abandons the retry.
    """
    deadline = clock() + retry_s
    delay = 0.25
    while True:
        try:
            return make_server(state, host, port)
        except OSError as exc:
            if exc.errno != errno.EADDRINUSE or clock() >= deadline:
                log.warning("exporter: cannot bind %s:%s (%s); not serving metrics", host, port, exc)
                return None
        if sleep(delay):
            return None
        delay = min(delay * 2, 2.0)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="hyperloom-exporter", description=__doc__.split("\n\n")[0])
    parser.add_argument("--session-dir", default=None, help="Session directory; auto-discovered when omitted.")
    parser.add_argument("--model", default=None, help="Narrow session auto-discovery to one model basename.")
    parser.add_argument("--parent-pid", type=int, default=None, help="Exit a grace period after this process ends.")
    parser.add_argument("--listen", default=DEFAULT_LISTEN, help=f"HOST:PORT to serve on (default {DEFAULT_LISTEN}).")
    parser.add_argument(
        "--grace-sec", type=float, default=DEFAULT_GRACE_SEC, help="Serve this long after the parent exits."
    )
    parser.add_argument("--watchdog-interval", type=float, default=WATCHDOG_INTERVAL_SEC, help=argparse.SUPPRESS)
    parser.add_argument("--bind-retry-sec", type=float, default=BIND_RETRY_SEC, help=argparse.SUPPRESS)
    parser.add_argument("--gpu", action="store_true", help="Also run the amd-smi probe (off: use the AMD exporter).")
    parser.add_argument(
        "--server", action="store_true", help="Also scrape the inference server (off: Prometheus does)."
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the exporter until signalled or its watchdog says to exit."""
    args = build_parser().parse_args(argv)
    try:
        host, port = parse_listen(args.listen)
    except ValueError as exc:
        print(f"hyperloom-exporter: {exc}", file=sys.stderr)
        return EXIT_CONFIG_ERROR
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    # An explicit dir is used even before it exists: a fresh run spawns the
    # exporter before state.json is written.
    session_dir = Path(args.session_dir) if args.session_dir else resolve_session_dir(None, model=args.model)
    if session_dir is None:
        print("hyperloom-exporter: no session directory found", file=sys.stderr)
        return EXIT_CONFIG_ERROR

    watchdog = Watchdog(parent_pid=args.parent_pid, session_dir=session_dir, grace_s=args.grace_sec)
    monitor = SessionMonitor(session_dir, gpu=args.gpu, server=args.server)
    register_inference_sd(monitor)
    state = ExporterState(monitor=monitor, version=exporter_version(), parent_alive=watchdog.parent_alive)

    stop = threading.Event()
    for signum in (signal.SIGTERM, signal.SIGINT):
        signal.signal(signum, lambda *_: stop.set())

    with monitor:
        server = bind_with_retry(state, host, port, retry_s=args.bind_retry_sec, sleep=stop.wait)
        if server is None:
            return EXIT_OK
        thread = threading.Thread(target=server.serve_forever, name="hyperloom-exporter-http", daemon=True)
        thread.start()
        log.info(
            "exporter: serving http://%s:%s/metrics for %s", host or "0.0.0.0", server.server_address[1], session_dir
        )
        try:
            while not stop.wait(args.watchdog_interval):
                verdict = watchdog.poll()
                if verdict == EXIT:
                    log.info("exporter: parent gone or session taken over; exiting")
                    break
        finally:
            server.shutdown()
            server.server_close()
    return EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
