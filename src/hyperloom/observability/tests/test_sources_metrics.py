# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Parsing and discovery tests for the GPU and inference-server sources.

The GPU fixture below is real ``amd-smi metric --json`` output captured from an
MI300X node running ROCm 7.2.1 (AMDSMI 26.2.2), trimmed to the fields the layer
reads. Keeping it verbatim — including the ``"N/A"`` strings — is the point: a
hand-written fixture would omit exactly the shapes that break parsers.
"""

from __future__ import annotations

import json
import subprocess

import pytest

from hyperloom.observability.model import SourceOutcome
from hyperloom.observability.sources.gpu import (
    GpuSource,
    parse_amd_smi,
    parse_rocm_smi_csv,
    resolve_amd_smi,
)
from hyperloom.observability.sources.server import (
    ServerMetricsSource,
    parse_prometheus,
)


AMD_SMI_FIXTURE = {
    "gpu_data": [
        {
            "gpu": 0,
            "usage": {
                "gfx_activity": {"value": 94, "unit": "%"},
                "umc_activity": {"value": 61, "unit": "%"},
                "mm_activity": "N/A",
            },
            "power": {"socket_power": {"value": 612, "unit": "W"}, "gfx_voltage": "N/A"},
            "mem_usage": {
                "total_vram": {"value": 196592, "unit": "MB"},
                "used_vram": {"value": 17153, "unit": "MB"},
            },
        },
        {
            "gpu": 1,
            "usage": {"gfx_activity": {"value": 0, "unit": "%"}, "umc_activity": "N/A"},
            "power": {"socket_power": "N/A"},
            "mem_usage": {
                "total_vram": {"value": 196592, "unit": "MB"},
                "used_vram": {"value": 284, "unit": "MB"},
            },
        },
    ]
}


def test_amd_smi_parses_real_output() -> None:
    """The captured MI300X document maps onto the model."""
    gpus = parse_amd_smi(AMD_SMI_FIXTURE)

    assert len(gpus) == 2
    assert gpus[0].index == 0
    assert gpus[0].util_pct == 94.0
    assert gpus[0].mem_activity_pct == 61.0
    assert gpus[0].mem_used_mb == 17153.0
    assert gpus[0].power_w == 612.0
    assert gpus[0].mem_used_fraction == pytest.approx(17153 / 196592)


def test_amd_smi_maps_not_available_to_none_not_zero() -> None:
    """``"N/A"`` means unsupported, which is not the same as measuring zero.

    Coerced to ``0.0`` an unreportable metric becomes indistinguishable from an
    idle one — the exact confusion this layer exists to prevent.
    """
    gpus = parse_amd_smi(AMD_SMI_FIXTURE)

    assert gpus[1].power_w is None
    assert gpus[1].mem_activity_pct is None
    # A genuine zero survives as zero, so the distinction is real and not just
    # blanket nulling.
    assert gpus[1].util_pct == 0.0


def test_amd_smi_tolerates_junk() -> None:
    """A shape change degrades to no rows rather than an exception."""
    assert parse_amd_smi({"unexpected": 1}) == ()
    assert parse_amd_smi(None) == ()
    assert parse_amd_smi({"gpu_data": ["not-a-dict"]}) == ()


def test_gpu_source_reports_a_broken_binary_as_error(monkeypatch) -> None:
    """``Invalid platform`` must surface, not be silently swallowed.

    A stale ``amd-smi`` earlier on PATH than the working ROCm one is the
    observed real configuration on a multi-ROCm host, and it exits non-zero
    with that message.
    """
    source = GpuSource()
    monkeypatch.setattr(source, "_amd_smi_command", lambda: "/usr/local/bin/amd-smi")
    monkeypatch.setattr(
        "hyperloom.observability.sources.gpu._run",
        lambda cmd, *, timeout_s: subprocess.CompletedProcess(cmd, 1, "", "Invalid platform\n"),
    )

    result = source.read()
    assert result.outcome is SourceOutcome.ERROR
    assert "Invalid platform" in (result.message or "")


def test_gpu_source_reports_a_timeout_rather_than_hanging(monkeypatch) -> None:
    """A wedged driver becomes an ``ERROR`` row, never a blocked caller."""
    source = GpuSource(timeout_s=2.0)
    monkeypatch.setattr(source, "_amd_smi_command", lambda: "amd-smi")

    def _timeout(cmd, *, timeout_s):
        raise subprocess.TimeoutExpired(cmd, timeout_s)

    monkeypatch.setattr("hyperloom.observability.sources.gpu._run", _timeout)

    result = source.read()
    assert result.outcome is SourceOutcome.ERROR
    assert "timed out" in (result.message or "")


def test_validation_rejects_a_zero_exit_invalid_platform(monkeypatch) -> None:
    """Exit status alone is not enough to trust a binary.

    The stale ``/usr/local/bin/amd-smi`` observed on a multi-ROCm host prints
    ``Invalid platform`` and exits **zero**. An ``rc == 0`` check accepts it,
    and the failure then surfaces as unparseable JSON several layers away.
    """
    from hyperloom.observability.sources.gpu import _validates

    monkeypatch.setattr(
        "hyperloom.observability.sources.gpu._run",
        lambda cmd, *, timeout_s: subprocess.CompletedProcess(cmd, 0, "Invalid platform\n", ""),
    )
    assert _validates("/usr/local/bin/amd-smi") is False

    monkeypatch.setattr(
        "hyperloom.observability.sources.gpu._run",
        lambda cmd, *, timeout_s: subprocess.CompletedProcess(
            cmd,
            0,
            "AMDSMI Tool: 26.2.2 | AMDSMI Library version: 26.2.2 | ROCm version: 7.2.1\n",
            "",
        ),
    )
    assert _validates("/opt/rocm/bin/amd-smi") is True


def test_resolver_skips_a_binary_that_does_not_validate(monkeypatch) -> None:
    """Existence on PATH is not proof of function; the resolver must move on."""
    seen: list[str] = []

    def fake_validates(binary: str) -> bool:
        seen.append(binary)
        return binary == "amd-smi"  # only the PATH one works in this scenario

    monkeypatch.setattr("hyperloom.observability.sources.gpu._validates", fake_validates)
    monkeypatch.setattr("hyperloom.observability.sources.gpu.Path.exists", lambda self: True)
    monkeypatch.setattr("hyperloom.observability.sources.gpu.shutil.which", lambda name: f"/usr/bin/{name}")
    monkeypatch.delenv("HYPERLOOM_AMD_SMI", raising=False)

    assert resolve_amd_smi() == "amd-smi"
    assert seen[0] == "/opt/rocm/bin/amd-smi", "the explicit ROCm path must be tried before PATH"


def test_gpu_source_absent_when_no_tool_exists(monkeypatch) -> None:
    """No GPU tooling is ``ABSENT`` — a CPU box is not a broken box."""
    source = GpuSource()
    monkeypatch.setattr(source, "_amd_smi_command", lambda: None)
    monkeypatch.setattr("hyperloom.observability.sources.gpu.shutil.which", lambda name: None)

    assert source.read().outcome is SourceOutcome.ABSENT


def test_rocm_smi_fallback_converts_vram_bytes_to_mb() -> None:
    """The deprecated tool reports VRAM in bytes where amd-smi uses MB."""
    csv = (
        "device,GPU use (%),Power (W),VRAM Total Memory (B),VRAM Total Used Memory (B)\n"
        "card0,73,415.0,206158430208,18253611008\n"
        "card1,0,140.0,206158430208,297795584\n"
    )
    gpus = parse_rocm_smi_csv(csv)

    assert [g.index for g in gpus] == [0, 1]
    assert gpus[0].util_pct == 73.0
    assert gpus[0].mem_total_mb == pytest.approx(196608.0, rel=1e-3)
    assert gpus[0].mem_used_mb == pytest.approx(17408.0, rel=1e-3)


PROMETHEUS_FIXTURE = """
# HELP vllm:num_requests_running Number of requests currently running.
# TYPE vllm:num_requests_running gauge
vllm:num_requests_running{model_name="qwen"} 12.0
vllm:num_requests_waiting{model_name="qwen"} 3.0
vllm:gpu_cache_usage_perc{model_name="qwen"} 0.42
vllm:prompt_tokens_total{model_name="qwen"} 1000.0
vllm:generation_tokens_total{model_name="qwen"} 5000.0
vllm:time_to_first_token_seconds_bucket{le="0.001"} 0.0
"""


def test_prometheus_parse_extracts_only_what_is_modelled() -> None:
    """Label sets are ignored and unrelated series are skipped."""
    values = parse_prometheus(PROMETHEUS_FIXTURE)

    assert values["requests_running"] == 12.0
    assert values["requests_waiting"] == 3.0
    assert values["kv_cache_pct"] == 0.42
    assert values["generation_tokens_total"] == 5000.0
    assert "time_to_first_token_seconds_bucket" not in values


def test_throughput_needs_two_samples(monkeypatch) -> None:
    """A rate is undefined on the first reading, and ``None`` says so.

    Reporting ``0.0`` there would be indistinguishable from a stalled server —
    the one thing an operator is watching this number to detect.
    """
    source = ServerMetricsSource(base_url="http://127.0.0.1:8000")
    bodies = iter([PROMETHEUS_FIXTURE, PROMETHEUS_FIXTURE.replace("5000.0", "5600.0")])
    monkeypatch.setattr(
        "hyperloom.observability.sources.server._fetch",
        lambda url, *, timeout_s: next(bodies),
    )

    first = source.read(now_unix=1000.0)
    assert first.ok
    assert first.data.tput_tok_s is None

    second = source.read(now_unix=1010.0)
    assert second.data.tput_tok_s == pytest.approx(60.0)  # 600 tokens / 10 s


def test_counter_reset_yields_none_not_a_negative_rate(monkeypatch) -> None:
    """A restarted server's counters go backwards; that delta is meaningless."""
    source = ServerMetricsSource(base_url="http://127.0.0.1:8000")
    bodies = iter([PROMETHEUS_FIXTURE, PROMETHEUS_FIXTURE.replace("5000.0", "12.0")])
    monkeypatch.setattr(
        "hyperloom.observability.sources.server._fetch",
        lambda url, *, timeout_s: next(bodies),
    )

    source.read(now_unix=1000.0)
    after_restart = source.read(now_unix=1010.0)
    assert after_restart.data.tput_tok_s is None


def test_no_server_listening_is_absent_not_error(monkeypatch) -> None:
    """No server during a KERNEL_AGENT phase is normal, not a fault."""
    monkeypatch.setattr("hyperloom.observability.sources.server.discover_base_url", lambda: None)

    assert ServerMetricsSource().read().outcome is SourceOutcome.ABSENT


def test_unreachable_discovered_server_is_absent(monkeypatch) -> None:
    """A port that stops answering mid-teardown is not an error either."""
    monkeypatch.setattr("hyperloom.observability.sources.server._fetch", lambda url, *, timeout_s: None)

    result = ServerMetricsSource(base_url="http://127.0.0.1:8000").read()
    assert result.outcome is SourceOutcome.ABSENT


def test_wrong_service_on_the_port_is_an_error(monkeypatch) -> None:
    """Something answered but is not vLLM — worth telling the operator."""
    monkeypatch.setattr(
        "hyperloom.observability.sources.server._fetch",
        lambda url, *, timeout_s: "# nothing we recognise\nfoo_bar 1.0\n",
    )

    result = ServerMetricsSource(base_url="http://127.0.0.1:9999").read()
    assert result.outcome is SourceOutcome.ERROR
    assert "no recognised vllm metrics" in (result.message or "")


def test_kv_cache_ratio_is_normalised_to_percent(monkeypatch) -> None:
    """vLLM reports a 0..1 ratio despite the ``_perc`` suffix."""
    monkeypatch.setattr(
        "hyperloom.observability.sources.server._fetch",
        lambda url, *, timeout_s: PROMETHEUS_FIXTURE,
    )

    result = ServerMetricsSource(base_url="http://127.0.0.1:8000").read()
    assert result.data.kv_cache_pct == pytest.approx(42.0)


def test_env_override_wins_over_discovery(monkeypatch) -> None:
    """``$HYPERLOOM_VLLM_URL`` short-circuits socket scanning."""
    from hyperloom.observability.sources.server import discover_base_url

    monkeypatch.setenv("HYPERLOOM_VLLM_URL", "http://gpu-node:8123/")
    assert discover_base_url() == "http://gpu-node:8123"


def test_json_roundtrip_of_the_gpu_fixture() -> None:
    """The fixture is valid JSON, so the source's decode path is exercised."""
    gpus = parse_amd_smi(json.loads(json.dumps(AMD_SMI_FIXTURE)))
    assert len(gpus) == 2
