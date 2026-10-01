# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Exporter process tests: HTTP surface and lifecycle."""

from __future__ import annotations

import json
import threading
import time
import urllib.error
import urllib.request
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import pytest

from hyperloom.observability import exporter as exp
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
    monkeypatch.setattr(exp, "discover_base_url", lambda: None)
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
    monkeypatch.setattr(exp, "discover_base_url", lambda: None)
    with monitored(session_dir) as monitor, serving(exp.ExporterState(monitor=monitor, version="t")) as base:
        assert get(f"{base}/healthz")[0] == 200
        assert get(f"{base}/nope")[0] == 404


def test_inference_sd_advertises_the_discovered_server(session_dir: Path, monkeypatch) -> None:
    monkeypatch.setattr(exp, "discover_base_url", lambda: "http://127.0.0.1:8000")
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
    monkeypatch.setattr(exp, "discover_base_url", lambda: None)
    with monitored(session_dir) as monitor, serving(exp.ExporterState(monitor=monitor, version="t")) as base:
        assert json.loads(get(f"{base}/sd/inference")[2]) == []


def test_metrics_before_session_exists(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(exp, "discover_base_url", lambda: None)
    missing = tmp_path / "model" / "20260930T000000Z"
    with monitored(missing) as monitor, serving(exp.ExporterState(monitor=monitor, version="t")) as base:
        status, _, body = get(f"{base}/metrics")
    assert status == 200
    assert "hyperloom_session_observed 0" in body
    assert not missing.exists(), "the exporter must not create the session dir"


def test_metrics_answer_while_a_source_hangs(session_dir: Path, monkeypatch) -> None:
    monkeypatch.setattr(exp, "discover_base_url", lambda: None)
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
    monkeypatch.setattr(exp, "discover_base_url", lambda: None)
    with monitored(session_dir) as monitor:
        state = exp.ExporterState(monitor=monitor, version="t")
        original = state.metrics_text
        monkeypatch.setattr(state, "metrics_text", lambda: (_ for _ in ()).throw(RuntimeError("x")))
        with serving(state) as base:
            assert get(f"{base}/metrics")[0] == 500
            monkeypatch.setattr(state, "metrics_text", original)
            assert get(f"{base}/metrics")[0] == 200


def test_serving_never_writes_the_session(session_dir: Path, monkeypatch) -> None:
    monkeypatch.setattr(exp, "discover_base_url", lambda: "http://127.0.0.1:8000")
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
    monkeypatch.setattr(exp, "discover_base_url", lambda: None)
    with monitored(session_dir) as monitor:
        state = exp.ExporterState(monitor=monitor, version="t")
        state.metrics_text()
        page = state.metrics_text()
    assert "hyperloom_exporter_render_errors_total 2" in page.replace(
        '{session_id="test-model_20260806T000000Z_deadbeef",model="test-model",framework="sglang"}', ""
    )


def test_exporter_version_is_a_string() -> None:
    assert isinstance(exp.exporter_version(), str) and exp.exporter_version()


@pytest.fixture(autouse=True)
def _no_real_discovery_env(monkeypatch):
    monkeypatch.delenv("HYPERLOOM_VLLM_URL", raising=False)
