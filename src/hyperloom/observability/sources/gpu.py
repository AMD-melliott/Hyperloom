# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""GPU utilization via ``amd-smi``, with ``rocm-smi`` as a fallback.

``rocm-smi`` is deprecated in favour of ``amd-smi``, so ``amd-smi`` is tried
first and the older tool is kept only as a fallback for hosts that predate it.

Finding a *working* ``amd-smi`` is the interesting part. On a development host
with five ROCm trees under ``/opt``, ``shutil.which("amd-smi")`` resolved to a
stale standalone binary at ``/usr/local/bin/amd-smi`` that exits non-zero with
``Invalid platform`` — while ``/opt/rocm/bin/amd-smi`` (the symlink into the
current ROCm's ``amdsmi_cli``) worked perfectly. PATH order is not a proxy for
"functional", so each candidate is *validated* with a cheap ``version`` call
before being trusted, and the winner is cached for the process lifetime.

Everything here is stdlib. This package declares no third-party dependencies,
which is why the well-tested ``rocm-smi`` CSV parser in
``agents/robustness/sources/local_probe.py`` is prior art rather than an
import — that module pulls in ``httpx``.

Scope caveat, deliberately carried into the model: these readings come from the
node's device driver and cover every tenant of the machine. On a shared box
they answer "is this host busy", not "is this session busy".
:attr:`~hyperloom.observability.model.GpuMetrics.host_global` records that so
renderers can label it rather than implying attribution.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any, Sequence

from ..model import GpuMetric, GpuMetrics
from .base import SourceResult


log = logging.getLogger(__name__)

# Override for hosts where neither the default location nor PATH is right.
ENV_AMD_SMI = "HYPERLOOM_AMD_SMI"

# Probed in order. The explicit ROCm path precedes PATH lookup precisely
# because PATH is where the broken copy tends to live.
AMD_SMI_CANDIDATES: tuple[str, ...] = (
    "/opt/rocm/bin/amd-smi",
    "amd-smi",
)

DEFAULT_TIMEOUT_S = 8.0
_VALIDATE_TIMEOUT_S = 5.0

# amd-smi reports unsupported fields as this literal. It must map to ``None``:
# coerced to 0.0 an unreportable metric becomes indistinguishable from an idle
# one, which is the exact class of lie this layer exists to avoid.
_NOT_AVAILABLE = "N/A"


def _run(cmd: Sequence[str], *, timeout_s: float) -> subprocess.CompletedProcess[str]:
    """Run a command capturing text output, never raising on non-zero exit."""
    return subprocess.run(  # noqa: S603 - fixed argv, no shell
        list(cmd),
        capture_output=True,
        text=True,
        timeout=timeout_s,
        check=False,
    )


def _validates(binary: str) -> bool:
    """Return whether ``binary`` is a functional amd-smi on this host.

    Exit status alone is **not** enough. The stale standalone build observed at
    ``/usr/local/bin/amd-smi`` prints ``Invalid platform`` and then exits
    **zero**, so an ``rc == 0`` check accepts the one binary this resolver
    exists to reject — and the failure only surfaces later as unparseable JSON.
    A working tool identifies itself (``AMDSMI Tool: 26.2.2 | ... | ROCm
    version: 7.2.1``), so the output is required to say so.

    Args:
        binary: Path or PATH-resolvable name.

    Returns:
        ``True`` when the binary answers ``version`` as a real amd-smi.
    """
    try:
        proc = _run([binary, "version"], timeout_s=_VALIDATE_TIMEOUT_S)
    except (OSError, subprocess.SubprocessError):
        return False
    if proc.returncode != 0:
        return False
    return "amdsmi" in f"{proc.stdout}{proc.stderr}".lower()


def resolve_amd_smi() -> str | None:
    """Locate a working ``amd-smi``.

    Order: ``$HYPERLOOM_AMD_SMI``, then ``/opt/rocm/bin/amd-smi``, then PATH.
    Each is validated before being accepted.

    Returns:
        The resolved command, or ``None`` when no candidate validates.
    """
    override = os.environ.get(ENV_AMD_SMI, "").strip()
    candidates = (override,) + AMD_SMI_CANDIDATES if override else AMD_SMI_CANDIDATES
    for candidate in candidates:
        if not candidate:
            continue
        if os.path.sep in candidate:
            if not Path(candidate).exists():
                continue
        elif shutil.which(candidate) is None:
            continue
        if _validates(candidate):
            return candidate
        log.debug("gpu: %s exists but did not validate; trying the next candidate", candidate)
    return None


def _value(node: Any) -> float | None:
    """Extract a numeric reading from an amd-smi ``{"value", "unit"}`` node.

    Args:
        node: The raw JSON node, which may be a mapping, a bare number, or the
            string ``"N/A"``.

    Returns:
        The reading, or ``None`` when unsupported or unparseable.
    """
    if isinstance(node, dict):
        node = node.get("value")
    if node is None or node == _NOT_AVAILABLE:
        return None
    try:
        return float(node)
    except (TypeError, ValueError):
        return None


def parse_amd_smi(payload: Any) -> tuple[GpuMetric, ...]:
    """Parse ``amd-smi metric --json`` output into model rows.

    Args:
        payload: The decoded JSON document.

    Returns:
        One :class:`~hyperloom.observability.model.GpuMetric` per GPU, in the
        order reported.
    """
    if isinstance(payload, dict):
        entries = payload.get("gpu_data")
    else:
        entries = payload
    if not isinstance(entries, list):
        return ()

    rows: list[GpuMetric] = []
    for position, entry in enumerate(entries):
        if not isinstance(entry, dict):
            continue
        usage = entry.get("usage") if isinstance(entry.get("usage"), dict) else {}
        mem = entry.get("mem_usage") if isinstance(entry.get("mem_usage"), dict) else {}
        power = entry.get("power") if isinstance(entry.get("power"), dict) else {}
        index = entry.get("gpu")
        try:
            index_int = int(index)
        except (TypeError, ValueError):
            index_int = position
        rows.append(
            GpuMetric(
                index=index_int,
                util_pct=_value(usage.get("gfx_activity")),
                mem_activity_pct=_value(usage.get("umc_activity")),
                mem_used_mb=_value(mem.get("used_vram")),
                mem_total_mb=_value(mem.get("total_vram")),
                power_w=_value(power.get("socket_power")),
            )
        )
    return tuple(rows)


def parse_rocm_smi_csv(text: str) -> tuple[GpuMetric, ...]:
    """Parse ``rocm-smi --csv`` output into model rows.

    Fallback path only. ``rocm-smi`` emits a header row followed by one row per
    ``card<N>``; column names vary across ROCm releases, so lookup is by
    substring rather than exact match.

    Args:
        text: Raw CSV output.

    Returns:
        One row per parsed card.
    """
    lines = [line for line in (raw.strip() for raw in text.splitlines()) if line]
    if len(lines) < 2:
        return ()
    header = [col.strip().lower() for col in lines[0].split(",")]

    def column(*needles: str) -> int | None:
        for position, name in enumerate(header):
            if all(needle in name for needle in needles):
                return position
        return None

    idx_use = column("gpu use")
    idx_power = column("power")
    idx_vram_used = column("vram", "used")
    idx_vram_total = column("vram", "total")

    def cell(parts: list[str], index: int | None) -> float | None:
        if index is None or index >= len(parts):
            return None
        raw = parts[index].strip()
        if not raw or raw.upper() == _NOT_AVAILABLE:
            return None
        try:
            return float(raw)
        except ValueError:
            return None

    rows: list[GpuMetric] = []
    for position, line in enumerate(lines[1:]):
        parts = line.split(",")
        name = parts[0].strip().lower() if parts else ""
        try:
            index_int = int(name.removeprefix("card"))
        except ValueError:
            index_int = position
        used = cell(parts, idx_vram_used)
        total = cell(parts, idx_vram_total)
        rows.append(
            GpuMetric(
                index=index_int,
                util_pct=cell(parts, idx_use),
                # rocm-smi reports VRAM in bytes where amd-smi uses MB.
                mem_used_mb=None if used is None else used / (1024.0 * 1024.0),
                mem_total_mb=None if total is None else total / (1024.0 * 1024.0),
                power_w=cell(parts, idx_power),
            )
        )
    return tuple(rows)


class GpuSource:
    """Samples GPU utilization from the node's device driver.

    The resolved tool is cached after the first successful probe so the
    validation cost is paid once, not on every poll.
    """

    name = "gpu"

    def __init__(self, *, timeout_s: float = DEFAULT_TIMEOUT_S) -> None:
        """Initialise the source.

        Args:
            timeout_s: Per-invocation subprocess timeout. A wedged GPU driver
                can hang ``amd-smi`` indefinitely, and the collector must not
                inherit that.
        """
        self._timeout_s = timeout_s
        self._amd_smi: str | None = None
        self._resolved = False

    def _amd_smi_command(self) -> str | None:
        """Return the cached amd-smi command, resolving it on first use."""
        if not self._resolved:
            self._amd_smi = resolve_amd_smi()
            self._resolved = True
        return self._amd_smi

    def read(self, session_dir: Path | None = None, *, now_unix: float | None = None) -> SourceResult:
        """Sample every visible GPU.

        Args:
            session_dir: Unused; accepted for protocol symmetry. GPU state is a
                property of the host, not of a session directory.
            now_unix: Unused; the reading is instantaneous.

        Returns:
            ``OK`` with a :class:`~hyperloom.observability.model.GpuMetrics`,
            ``ABSENT`` when no GPU tool is installed, or ``ERROR`` when a tool
            is present but failed.
        """
        del session_dir, now_unix

        command = self._amd_smi_command()
        if command:
            try:
                proc = _run(
                    [command, "metric", "--usage", "--power", "--mem-usage", "--json"],
                    timeout_s=self._timeout_s,
                )
            except subprocess.TimeoutExpired:
                return SourceResult.error(f"amd-smi timed out after {self._timeout_s:g}s")
            except (OSError, subprocess.SubprocessError) as exc:
                return SourceResult.error(f"amd-smi failed: {exc}")
            if proc.returncode == 0:
                try:
                    gpus = parse_amd_smi(json.loads(proc.stdout))
                except (ValueError, TypeError) as exc:
                    return SourceResult.error(f"amd-smi emitted unparseable JSON: {exc}")
                if gpus:
                    return SourceResult.hit(GpuMetrics(gpus=gpus, tool="amd-smi", host_global=True))
                return SourceResult.error("amd-smi reported no GPUs")
            detail = (proc.stderr or proc.stdout or "").strip().splitlines()
            return SourceResult.error(f"amd-smi rc={proc.returncode}: {detail[0] if detail else 'no output'}")

        return self._read_rocm_smi()

    def _read_rocm_smi(self) -> SourceResult:
        """Fall back to the deprecated ``rocm-smi``."""
        if shutil.which("rocm-smi") is None:
            return SourceResult.absent()
        try:
            proc = _run(
                ["rocm-smi", "--showuse", "--showmemuse", "--showpower", "--csv"],
                timeout_s=self._timeout_s,
            )
        except subprocess.TimeoutExpired:
            return SourceResult.error(f"rocm-smi timed out after {self._timeout_s:g}s")
        except (OSError, subprocess.SubprocessError) as exc:
            return SourceResult.error(f"rocm-smi failed: {exc}")
        if proc.returncode != 0:
            return SourceResult.error(f"rocm-smi rc={proc.returncode}")
        gpus = parse_rocm_smi_csv(proc.stdout)
        if not gpus:
            return SourceResult.error("rocm-smi output had no parseable rows")
        return SourceResult.hit(GpuMetrics(gpus=gpus, tool="rocm-smi", host_global=True))
