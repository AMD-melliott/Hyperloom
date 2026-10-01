# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Spawn the Prometheus metrics exporter for one optimizer run.

The exporter is a detached child (``start_new_session=True``) that watches this
process and exits a grace period after it ends, so the run's final state stays
scrapeable. The run never waits for, stops, or depends on it: every failure
here is one warning. See ``docs/superpowers/specs/2026-09-30-prometheus-exporter-design.md``.
"""

from __future__ import annotations

import argparse
import logging
import os
import subprocess
import sys
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import TypeVar

from hyperloom.common.env import EnvValueError, env_bool
from hyperloom.observability.exporter_config import DEFAULT_GRACE_SEC, DEFAULT_LISTEN, parse_grace_sec, parse_listen

log = logging.getLogger(__name__)

T = TypeVar("T")

ENV_ENABLE = "HYPERLOOM_METRICS_EXPORTER"
ENV_LISTEN = "HYPERLOOM_METRICS_LISTEN"
ENV_GRACE = "HYPERLOOM_METRICS_GRACE_SEC"

LOG_RELPATH = ("runtime", "metrics_exporter.log")


@dataclass(frozen=True)
class ExporterLaunch:
    listen: tuple[str, int]
    grace_sec: float


def _from_env(environ: Mapping[str, str], name: str, parse: Callable[[str], T], default: T) -> T:
    raw = environ.get(name, "").strip()
    if raw:
        try:
            return parse(raw)
        except argparse.ArgumentTypeError as exc:
            log.warning("ignoring %s=%r: %s; using the default", name, raw, exc)
    return default


def resolve_launch(args: argparse.Namespace, environ: Mapping[str, str]) -> ExporterLaunch | None:
    """Combine flags and env; ``None`` when the exporter is disabled. Flags win."""
    if getattr(args, "no_metrics_exporter", False):
        return None
    try:
        enabled = env_bool(ENV_ENABLE, True, env=environ)
    except EnvValueError:
        log.warning("ignoring %s=%r: not a boolean; exporter stays enabled", ENV_ENABLE, environ.get(ENV_ENABLE))
        enabled = True
    if not enabled:
        return None
    listen = getattr(args, "metrics_listen", None)
    if listen is None:
        listen = _from_env(environ, ENV_LISTEN, parse_listen, parse_listen(DEFAULT_LISTEN))
    grace = getattr(args, "metrics_grace_sec", None)
    if grace is None:
        grace = _from_env(environ, ENV_GRACE, parse_grace_sec, DEFAULT_GRACE_SEC)
    return ExporterLaunch(listen=listen, grace_sec=grace)


def build_command(
    session_dir: Path, launch: ExporterLaunch, *, parent_pid: int, python: str = sys.executable
) -> list[str]:
    host, port = launch.listen
    return [
        python, "-m", "hyperloom.observability.exporter",
        "--session-dir", str(session_dir),
        "--parent-pid", str(parent_pid),
        "--listen", f"{host}:{port}",
        "--grace-sec", str(launch.grace_sec),
    ]  # fmt: skip


def start_metrics_exporter(
    session_dir: Path,
    args: argparse.Namespace,
    *,
    environ: Mapping[str, str] | None = None,
    popen: Callable[..., subprocess.Popen] = subprocess.Popen,
) -> subprocess.Popen | None:
    """Spawn the exporter unless disabled; never raises."""
    launch = resolve_launch(args, os.environ if environ is None else environ)
    if launch is None:
        log.info("metrics exporter disabled")
        return None
    command = build_command(Path(session_dir), launch, parent_pid=os.getpid())
    log_path = Path(session_dir).joinpath(*LOG_RELPATH)
    try:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with open(log_path, "ab") as log_file:
            proc = popen(
                command,
                stdin=subprocess.DEVNULL,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                start_new_session=True,
                close_fds=True,
            )
    except OSError:
        log.warning("metrics exporter failed to start; continuing without /metrics", exc_info=True)
        return None
    host, port = launch.listen
    print(
        f"[metrics] exporter pid={getattr(proc, 'pid', '?')} starting on http://{host or '0.0.0.0'}:{port}/metrics"
        f" (log: {log_path})",
        file=sys.stderr,
    )
    return proc
