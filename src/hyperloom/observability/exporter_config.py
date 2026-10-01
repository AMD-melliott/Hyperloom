# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Exporter launch settings shared by the exporter and the optimizer CLI.

The parsers double as argparse ``type=`` callables, so a bad flag fails at the
command line rather than inside the detached exporter. Kept free of the
exporter's imports so the optimizer can validate its flags without loading it.
"""

from __future__ import annotations

import argparse
import math

DEFAULT_LISTEN = "127.0.0.1:9477"
DEFAULT_GRACE_SEC = 120.0


def parse_listen(value: str) -> tuple[str, int]:
    """Split ``HOST:PORT``; an empty host binds every IPv4 interface."""
    host, sep, port_text = value.rpartition(":")
    if not sep:
        raise argparse.ArgumentTypeError(f"listen address {value!r} is not HOST:PORT")
    if host.startswith("["):
        raise argparse.ArgumentTypeError(f"listen address {value!r}: IPv6 is not supported")
    try:
        port = int(port_text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"listen address {value!r} has a non-numeric port") from None
    if not 0 <= port <= 65535:
        raise argparse.ArgumentTypeError(f"listen address {value!r} has an out-of-range port")
    return host, port


def parse_grace_sec(value: str) -> float:
    """A finite, non-negative number of seconds."""
    try:
        seconds = float(value)
    except ValueError:
        raise argparse.ArgumentTypeError(f"grace period {value!r} is not a number") from None
    if not math.isfinite(seconds) or seconds < 0:
        raise argparse.ArgumentTypeError(f"grace period {value!r} must be a finite number >= 0")
    return seconds
