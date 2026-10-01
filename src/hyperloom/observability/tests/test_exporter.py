# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Exporter process tests: HTTP surface and lifecycle."""

from __future__ import annotations

import argparse
import errno
import json
import os
import signal
import socket
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path

import pytest

from hyperloom.observability import exporter as exp, exporter_config
from hyperloom.observability.collector import SessionMonitor
from hyperloom.observability.sources.base import SourceResult

from .test_invariants import _tree_fingerprint


@contextmanager
def serving(state: exp.ExporterState) -> Iterator[str]:
    server = exp.make_server(state, "127.0.0.1", 0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_address[1]}"
    finally:
        server.shutdown()
        server.server_close()


@contextmanager
def monitored(session_dir: Path) -> Iterator[SessionMonitor]:
    monitor = SessionMonitor(session_dir, gpu=False, server=False)
    exp.register_inference_sd(monitor)
    monitor.start()
    try:
        yield monitor
    finally:
        monitor.stop()


def get(url: str) -> tuple[int, str, str]:
    try:
        with urllib.request.urlopen(url, timeout=5) as response:
            return response.status, response.headers["Content-Type"], response.read().decode()
    except urllib.error.HTTPError as err:
        return err.code, err.headers.get("Content-Type", ""), ""


def test_metrics_endpoint_serves_the_session(session_dir: Path, monkeypatch) -> None:
    monkeypatch.setattr(exp, "find_session_server", lambda _sd: None)
    with monitored(session_dir) as monitor, serving(exp.ExporterState(monitor=monitor, version="t")) as base:
        status, ctype, body = get(f"{base}/metrics")
    assert status == 200
    assert ctype == "text/plain; version=0.0.4; charset=utf-8"
    assert 'hyperloom_phase_current{session_id="test-model_20260806T000000Z_deadbeef"' in body
    assert "hyperloom_session_observed{" in body
    assert (
        'hyperloom_source_up{session_id="test-model_20260806T000000Z_deadbeef",model="test-model",framework="sglang",source="inference_sd"} 1'
        in body
    )


def test_healthz_and_unknown_path(session_dir: Path, monkeypatch) -> None:
    monkeypatch.setattr(exp, "find_session_server", lambda _sd: None)
    with monitored(session_dir) as monitor, serving(exp.ExporterState(monitor=monitor, version="t")) as base:
        assert get(f"{base}/healthz")[0] == 200
        assert get(f"{base}/nope")[0] == 404


def test_inference_sd_advertises_the_discovered_server(session_dir: Path, monkeypatch) -> None:
    monkeypatch.setattr(exp, "find_session_server", lambda _sd: "http://127.0.0.1:8000")
    with monitored(session_dir) as monitor, serving(exp.ExporterState(monitor=monitor, version="t")) as base:
        status, ctype, body = get(f"{base}/sd/inference")
    assert status == 200
    assert ctype == "application/json"
    assert json.loads(body) == [
        {
            "targets": ["127.0.0.1:8000"],
            "labels": {
                "session_id": "test-model_20260806T000000Z_deadbeef",
                "model": "test-model",
                "framework": "sglang",
            },
        }
    ]


def test_inference_sd_is_empty_without_a_server(session_dir: Path, monkeypatch) -> None:
    monkeypatch.setattr(exp, "find_session_server", lambda _sd: None)
    with monitored(session_dir) as monitor, serving(exp.ExporterState(monitor=monitor, version="t")) as base:
        assert json.loads(get(f"{base}/sd/inference")[2]) == []


def _write_lifecycle_pid(session_dir: Path, pid: int, port: int) -> None:
    pid_dir = session_dir / "runs" / "baseline" / "abc123"
    pid_dir.mkdir(parents=True)
    (pid_dir / f"sglang_{port}.pid").write_text(f"{pid} {pid}\n", encoding="utf-8")


def test_inference_sd_advertises_the_sessions_live_server(session_dir: Path, monkeypatch) -> None:
    monkeypatch.setattr("hyperloom.observability.sources.server._listening_ports", lambda: (8000, 8080))
    _write_lifecycle_pid(session_dir, os.getpid(), 43210)
    with monitored(session_dir) as monitor, serving(exp.ExporterState(monitor=monitor, version="t")) as base:
        assert json.loads(get(f"{base}/sd/inference")[2])[0]["targets"] == ["127.0.0.1:43210"]


def test_inference_sd_is_empty_once_the_sessions_server_is_gone(session_dir: Path, monkeypatch) -> None:
    monkeypatch.setattr("hyperloom.observability.sources.server._listening_ports", lambda: (8000, 8080))
    dead = subprocess.Popen([sys.executable, "-c", "pass"])
    dead.wait()
    _write_lifecycle_pid(session_dir, dead.pid, 43210)
    with monitored(session_dir) as monitor, serving(exp.ExporterState(monitor=monitor, version="t")) as base:
        assert json.loads(get(f"{base}/sd/inference")[2]) == []


def test_metrics_before_session_exists(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(exp, "find_session_server", lambda _sd: None)
    missing = tmp_path / "model" / "20260930T000000Z"
    with monitored(missing) as monitor, serving(exp.ExporterState(monitor=monitor, version="t")) as base:
        status, _, body = get(f"{base}/metrics")
    assert status == 200
    assert "hyperloom_session_observed 0" in body
    assert not missing.exists(), "the exporter must not create the session dir"


def test_metrics_answer_while_a_source_hangs(session_dir: Path, monkeypatch) -> None:
    monkeypatch.setattr(exp, "find_session_server", lambda _sd: None)
    release = threading.Event()
    calls = {"n": 0}

    def stuck() -> SourceResult:
        calls["n"] += 1
        if calls["n"] > 1:
            release.wait(30)
        return SourceResult.hit(None)

    monitor = SessionMonitor(session_dir, gpu=False, server=False)
    monitor.collector.register("stuck", stuck, interval_s=0.05)
    monitor.start()
    try:
        time.sleep(0.3)
        with serving(exp.ExporterState(monitor=monitor, version="t")) as base:
            started = time.monotonic()
            status, _, _ = get(f"{base}/metrics")
            assert status == 200
            assert time.monotonic() - started < 1.0
    finally:
        release.set()
        monitor.stop()


def test_a_failing_render_returns_500_and_the_server_survives(session_dir: Path, monkeypatch) -> None:
    monkeypatch.setattr(exp, "find_session_server", lambda _sd: None)
    with monitored(session_dir) as monitor:
        state = exp.ExporterState(monitor=monitor, version="t")
        original = state.metrics_text
        monkeypatch.setattr(state, "metrics_text", lambda: (_ for _ in ()).throw(RuntimeError("x")))
        with serving(state) as base:
            assert get(f"{base}/metrics")[0] == 500
            monkeypatch.setattr(state, "metrics_text", original)
            assert get(f"{base}/metrics")[0] == 200


def test_serving_never_writes_the_session(session_dir: Path, monkeypatch) -> None:
    monkeypatch.setattr(exp, "find_session_server", lambda _sd: "http://127.0.0.1:8000")
    before = _tree_fingerprint(session_dir)
    with monitored(session_dir) as monitor, serving(exp.ExporterState(monitor=monitor, version="t")) as base:
        for path in ("/metrics", "/sd/inference", "/healthz"):
            get(f"{base}{path}")
    assert _tree_fingerprint(session_dir) == before


def test_render_errors_accumulate_across_scrapes(session_dir: Path, monkeypatch) -> None:
    from hyperloom.observability.render import prometheus as prom

    def boom(_snapshot):
        raise RuntimeError("x")

    monkeypatch.setattr(prom, "_SNAPSHOT_BUILDERS", (("session", boom),))
    monkeypatch.setattr(exp, "find_session_server", lambda _sd: None)
    with monitored(session_dir) as monitor:
        state = exp.ExporterState(monitor=monitor, version="t")
        state.metrics_text()
        page = state.metrics_text()
    assert "hyperloom_exporter_render_errors_total 2" in page.replace(
        '{session_id="test-model_20260806T000000Z_deadbeef",model="test-model",framework="sglang"}', ""
    )


def test_exporter_reports_the_optimizer_version() -> None:
    import hyperloom.inference_optimizer
    from hyperloom.common.version import UNINSTALLED_VERSION

    assert exp.hyperloom_version() == hyperloom.inference_optimizer.__version__
    assert exp.hyperloom_version() != UNINSTALLED_VERSION, "the test environment installs the package"


class FakeClock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


def lock_result(pid: int | None, alive: bool | None) -> Callable[[Path], SourceResult]:
    return lambda _sd: SourceResult.hit({"pid": pid, "pid_alive": alive})


@pytest.mark.parametrize(
    ("value", "expected"),
    [("127.0.0.1:9477", ("127.0.0.1", 9477)), ("0.0.0.0:0", ("0.0.0.0", 0)), (":9477", ("", 9477))],
)
def test_parse_listen_accepts(value: str, expected: tuple[str, int]) -> None:
    assert exporter_config.parse_listen(value) == expected


@pytest.mark.parametrize("value", ["9477", "host:abc", "host:70000", "[::1]:9477", ""])
def test_parse_listen_rejects(value: str) -> None:
    with pytest.raises(argparse.ArgumentTypeError):
        exporter_config.parse_listen(value)


@pytest.mark.parametrize(("value", "expected"), [("0", 0.0), ("1.5", 1.5), ("120", 120.0)])
def test_parse_grace_sec_accepts(value: str, expected: float) -> None:
    assert exporter_config.parse_grace_sec(value) == expected


@pytest.mark.parametrize("value", ["soon", "nan", "inf", "-inf", "-1", ""])
def test_parse_grace_sec_rejects(value: str) -> None:
    with pytest.raises(argparse.ArgumentTypeError):
        exporter_config.parse_grace_sec(value)


def test_watchdog_without_a_parent_runs_forever(tmp_path: Path) -> None:
    dog = exp.Watchdog(parent_pid=None, session_dir=tmp_path, grace_s=1.0, read_lock=lock_result(None, None))
    assert dog.parent_alive() is None
    assert dog.poll() == exp.RUN


def test_watchdog_grace_then_exit(tmp_path: Path) -> None:
    clock = FakeClock()
    ppid = {"value": 4242}
    dog = exp.Watchdog(
        parent_pid=4242,
        session_dir=tmp_path,
        grace_s=120.0,
        getppid=lambda: ppid["value"],
        read_lock=lock_result(4242, True),
        clock=clock,
    )
    assert dog.poll() == exp.RUN
    assert dog.parent_alive() is True

    ppid["value"] = 1
    clock.now = 10.0
    assert dog.poll() == exp.GRACE
    assert dog.parent_alive() is False
    clock.now = 129.0
    assert dog.poll() == exp.GRACE
    clock.now = 130.0
    assert dog.poll() == exp.EXIT


def test_watchdog_exits_on_lock_takeover(tmp_path: Path) -> None:
    dog = exp.Watchdog(
        parent_pid=4242,
        session_dir=tmp_path,
        grace_s=120.0,
        getppid=lambda: 1,
        read_lock=lock_result(5555, True),
        clock=FakeClock(),
    )
    assert dog.poll() == exp.EXIT


@pytest.mark.parametrize(("pid", "alive"), [(4242, True), (5555, False), (5555, None), (None, None)])
def test_watchdog_ignores_a_lock_that_is_not_a_live_successor(tmp_path: Path, pid, alive) -> None:
    dog = exp.Watchdog(
        parent_pid=4242,
        session_dir=tmp_path,
        grace_s=120.0,
        getppid=lambda: 4242,
        read_lock=lock_result(pid, alive),
        clock=FakeClock(),
    )
    assert dog.poll() == exp.RUN


def _occupy_port() -> tuple[socket.socket, int]:
    blocker = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    blocker.bind(("127.0.0.1", 0))
    blocker.listen(1)
    return blocker, blocker.getsockname()[1]


def test_bind_without_a_parent_tries_once(session_dir: Path, caplog) -> None:
    blocker, port = _occupy_port()
    slept: list[float] = []
    try:
        with monitored(session_dir) as monitor:
            server = exp.bind_with_retry(
                exp.ExporterState(monitor=monitor, version="t"),
                "127.0.0.1",
                port,
                keep_waiting=lambda: False,
                sleep=slept.append,
            )
    finally:
        blocker.close()
    assert server is None
    assert slept == []
    assert len([r for r in caplog.records if r.name == exp.log.name]) == 1


def test_bind_waits_while_the_parent_runs_and_stops_when_it_says_exit(session_dir: Path, caplog) -> None:
    blocker, port = _occupy_port()
    verdicts = iter([True, True, True, False])
    slept: list[float] = []
    try:
        with monitored(session_dir) as monitor:
            server = exp.bind_with_retry(
                exp.ExporterState(monitor=monitor, version="t"),
                "127.0.0.1",
                port,
                keep_waiting=lambda: next(verdicts),
                sleep=slept.append,
            )
    finally:
        blocker.close()
    assert server is None
    assert slept == [0.25, 0.5, 1.0]
    assert caplog.text.count("is busy; waiting") == 1


def test_bind_stops_retrying_when_sleep_reports_a_stop(session_dir: Path) -> None:
    blocker, port = _occupy_port()
    slept: list[float] = []

    def stopped(seconds: float) -> bool:
        slept.append(seconds)
        return True

    try:
        with monitored(session_dir) as monitor:
            server = exp.bind_with_retry(
                exp.ExporterState(monitor=monitor, version="t"),
                "127.0.0.1",
                port,
                keep_waiting=lambda: True,
                sleep=stopped,
            )
    finally:
        blocker.close()
    assert server is None
    assert len(slept) == 1


def test_bind_retries_until_port_frees(session_dir: Path) -> None:
    blocker, port = _occupy_port()

    def sleep(_seconds: float) -> None:
        blocker.close()

    with monitored(session_dir) as monitor:
        server = exp.bind_with_retry(
            exp.ExporterState(monitor=monitor, version="t"), "127.0.0.1", port, keep_waiting=lambda: True, sleep=sleep
        )
    assert server is not None
    assert server.server_address[1] == port
    server.server_close()


def test_bind_does_not_retry_other_errors(session_dir: Path, monkeypatch) -> None:
    def refuse(*_args, **_kwargs):
        raise OSError(errno.EADDRNOTAVAIL, "not here")

    monkeypatch.setattr(exp, "make_server", refuse)
    slept: list[float] = []
    with monitored(session_dir) as monitor:
        server = exp.bind_with_retry(
            exp.ExporterState(monitor=monitor, version="t"),
            "10.255.255.1",
            9477,
            keep_waiting=lambda: True,
            sleep=slept.append,
        )
    assert server is None
    assert slept == []


def test_a_busy_port_without_a_parent_exits_zero(session_dir: Path) -> None:
    blocker, port = _occupy_port()
    try:
        exporter = subprocess.run(
            [
                sys.executable,
                "-m",
                "hyperloom.observability.exporter",
                "--session-dir",
                str(session_dir),
                "--listen",
                f"127.0.0.1:{port}",
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )
    finally:
        blocker.close()
    assert exporter.returncode == 0
    assert "not serving metrics" in exporter.stderr


@pytest.mark.parametrize(("flag", "value"), [("--listen", "nonsense"), ("--grace-sec", "nan")])
def test_main_rejects_a_bad_flag(session_dir: Path, flag: str, value: str) -> None:
    with pytest.raises(SystemExit) as exit_info:
        exp.main(["--session-dir", str(session_dir), flag, value])
    assert exit_info.value.code == 2


def _free_port() -> int:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


PARENT_SCRIPT = """
import subprocess, sys, time
child = subprocess.Popen(
    [sys.executable, "-m", "hyperloom.observability.exporter", "--session-dir", sys.argv[1],
     "--parent-pid", str(__import__("os").getpid()), "--listen", sys.argv[2],
     "--grace-sec", "1", "--watchdog-interval", "0.2"],
    start_new_session=True,
)
print(child.pid, flush=True)
time.sleep(600)
"""


def _pid_running(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    # A reaped-by-init zombie still answers kill(0); check its state.
    try:
        return Path(f"/proc/{pid}/stat").read_text().split()[2] != "Z"
    except FileNotFoundError:
        return False


def _wait_for_healthz(port: int) -> None:
    deadline = time.monotonic() + 20
    while time.monotonic() < deadline:
        try:
            if get(f"http://127.0.0.1:{port}/healthz")[0] == 200:
                return
        except OSError:
            pass
        time.sleep(0.1)
    pytest.fail("exporter never started serving")


def test_exporter_exits_after_parent_is_killed(session_dir: Path) -> None:
    port = _free_port()
    parent = subprocess.Popen(
        [sys.executable, "-c", PARENT_SCRIPT, str(session_dir), f"127.0.0.1:{port}"],
        stdout=subprocess.PIPE,
        text=True,
    )
    child_pid = int(parent.stdout.readline())
    try:
        _wait_for_healthz(port)

        status, _, body = get(f"http://127.0.0.1:{port}/metrics")
        assert status == 200
        assert "hyperloom_exporter_parent_alive" in body

        parent.send_signal(signal.SIGKILL)
        parent.wait(10)

        deadline = time.monotonic() + 15
        while time.monotonic() < deadline and _pid_running(child_pid):
            time.sleep(0.2)
        assert not _pid_running(child_pid), "exporter outlived its grace period"
    finally:
        if _pid_running(child_pid):
            os.kill(child_pid, signal.SIGKILL)
        if parent.poll() is None:
            parent.kill()


def test_exporter_exits_cleanly_on_sigterm(session_dir: Path) -> None:
    port = _free_port()
    exporter = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "hyperloom.observability.exporter",
            "--session-dir",
            str(session_dir),
            "--listen",
            f"127.0.0.1:{port}",
        ]
    )
    try:
        _wait_for_healthz(port)
        exporter.send_signal(signal.SIGTERM)
        assert exporter.wait(5) == 0
    finally:
        if exporter.poll() is None:
            exporter.kill()


@pytest.fixture(autouse=True)
def _no_real_discovery_env(monkeypatch):
    monkeypatch.delenv("HYPERLOOM_VLLM_URL", raising=False)
