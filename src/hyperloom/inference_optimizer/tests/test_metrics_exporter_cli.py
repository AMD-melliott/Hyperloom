# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The optimizer spawns the metrics exporter by default and never depends on it."""

from __future__ import annotations

import inspect
import socket
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

import pytest

from hyperloom.inference_optimizer import cli
from hyperloom.inference_optimizer.cli import metrics_exporter as me


def _parse(argv: list[str]):
    return cli._build_parser().parse_args(["optimize", "--model", "/tmp/m", *argv])


def test_enabled_by_default() -> None:
    assert me.resolve_launch(_parse([]), {}) == me.ExporterLaunch(listen=("127.0.0.1", 9477), grace_sec=120.0)


@pytest.mark.parametrize(
    ("argv", "env"),
    [
        (["--no-metrics-exporter"], {}),
        ([], {"HYPERLOOM_METRICS_EXPORTER": "0"}),
        ([], {"HYPERLOOM_METRICS_EXPORTER": "false"}),
    ],
)
def test_disabled_by_flag_or_env(argv, env) -> None:
    assert me.resolve_launch(_parse(argv), env) is None


def test_env_configures_and_flags_win() -> None:
    env = {"HYPERLOOM_METRICS_LISTEN": "0.0.0.0:9500", "HYPERLOOM_METRICS_GRACE_SEC": "30"}
    assert me.resolve_launch(_parse([]), env) == me.ExporterLaunch(listen=("0.0.0.0", 9500), grace_sec=30.0)
    flagged = _parse(["--metrics-listen", "127.0.0.1:9600", "--metrics-grace-sec", "5"])
    assert me.resolve_launch(flagged, env) == me.ExporterLaunch(listen=("127.0.0.1", 9600), grace_sec=5.0)


@pytest.mark.parametrize(
    "argv",
    [
        ["--metrics-listen", "nonsense"],
        ["--metrics-listen", "127.0.0.1:70000"],
        ["--metrics-grace-sec", "soon"],
        ["--metrics-grace-sec", "nan"],
        ["--metrics-grace-sec", "inf"],
        ["--metrics-grace-sec", "-1"],
    ],
)
def test_a_bad_flag_is_rejected_by_the_parser(argv, capsys) -> None:
    with pytest.raises(SystemExit) as exit_info:
        _parse(argv)
    assert exit_info.value.code == 2
    assert argv[0] in capsys.readouterr().err


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("HYPERLOOM_METRICS_LISTEN", "nonsense"),
        ("HYPERLOOM_METRICS_GRACE_SEC", "soon"),
        ("HYPERLOOM_METRICS_GRACE_SEC", "nan"),
        ("HYPERLOOM_METRICS_GRACE_SEC", "inf"),
        ("HYPERLOOM_METRICS_GRACE_SEC", "-5"),
    ],
)
def test_a_bad_env_value_warns_and_keeps_the_default(name, value, caplog) -> None:
    launch = me.resolve_launch(_parse([]), {name: value})
    assert launch == me.ExporterLaunch(listen=("127.0.0.1", 9477), grace_sec=120.0)
    assert [record.levelname for record in caplog.records] == ["WARNING"]
    assert name in caplog.text


def test_a_malformed_enable_env_keeps_the_exporter_on(caplog) -> None:
    launch = me.resolve_launch(_parse([]), {"HYPERLOOM_METRICS_EXPORTER": "maybe"})
    assert launch == me.ExporterLaunch(listen=("127.0.0.1", 9477), grace_sec=120.0)
    assert "HYPERLOOM_METRICS_EXPORTER" in caplog.text


def test_build_command(tmp_path: Path) -> None:
    command = me.build_command(tmp_path, me.ExporterLaunch(("127.0.0.1", 9477), 120.0), parent_pid=99, python="/py")
    assert command == [
        "/py",
        "-m",
        "hyperloom.observability.exporter",
        "--session-dir",
        str(tmp_path),
        "--parent-pid",
        "99",
        "--listen",
        "127.0.0.1:9477",
        "--grace-sec",
        "120.0",
    ]


def test_spawns_detached_with_its_log_in_runtime(tmp_path: Path, capsys) -> None:
    calls: list[tuple[list[str], dict]] = []

    def fake_popen(command, **kwargs):
        calls.append((command, kwargs))
        return "proc"

    assert me.start_metrics_exporter(tmp_path, _parse([]), environ={}, popen=fake_popen) == "proc"
    ((command, kwargs),) = calls
    assert command[:3] == [sys.executable, "-m", "hyperloom.observability.exporter"]
    assert kwargs["start_new_session"] is True
    assert kwargs["stdin"] is subprocess.DEVNULL
    assert kwargs["stderr"] is subprocess.STDOUT
    assert (tmp_path / "runtime" / "metrics_exporter.log").exists()
    assert "[metrics] exporter pid=? starting on http://127.0.0.1:9477/metrics" in capsys.readouterr().err


def test_disabled_never_spawns(tmp_path: Path) -> None:
    def fail(*_a, **_k):
        raise AssertionError("spawned while disabled")

    assert me.start_metrics_exporter(tmp_path, _parse(["--no-metrics-exporter"]), environ={}, popen=fail) is None


def test_a_spawn_failure_never_fails_the_run(tmp_path: Path, caplog) -> None:
    def broken(*_a, **_k):
        raise OSError("no such interpreter")

    assert me.start_metrics_exporter(tmp_path, _parse([]), environ={}, popen=broken) is None
    assert "metrics exporter" in caplog.text


def test_the_run_starts_the_exporter_before_the_coordinator() -> None:
    source = inspect.getsource(cli._run_optimize)
    spawn = source.index("start_metrics_exporter(session_dir, args)")
    assert source.rindex("_acquire_session_lock_or_exit(session_dir)") < spawn
    assert spawn < source.index("await coordinator.run(")


def _free_port() -> int:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


def test_spawned_exporter_serves_metrics(tmp_path: Path) -> None:
    session_dir = tmp_path / "model" / "20260930T000000Z"
    session_dir.mkdir(parents=True)
    port = _free_port()
    proc = me.start_metrics_exporter(
        session_dir, _parse(["--metrics-listen", f"127.0.0.1:{port}", "--metrics-grace-sec", "1"]), environ={}
    )
    assert proc is not None
    try:
        deadline = time.monotonic() + 20
        body = ""
        while time.monotonic() < deadline:
            try:
                with urllib.request.urlopen(f"http://127.0.0.1:{port}/metrics", timeout=2) as response:
                    body = response.read().decode()
                break
            except OSError:
                time.sleep(0.2)
        assert "hyperloom_exporter_parent_alive 1" in body
    finally:
        proc.terminate()
        proc.wait(10)
