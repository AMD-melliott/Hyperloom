#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Print the current phase, budget, and resource state of a Hyperloom session.

Read-only: it never writes to the session directory, so it is safe against a
live run.

Usage::

    python -m hyperloom.inference_optimizer.tools.status
    python -m hyperloom.inference_optimizer.tools.status --session-dir SD
    python -m hyperloom.inference_optimizer.tools.status --json

Unlike the other operator scripts, this auto-discovers the newest
``$USER_DATA_PATH/<model>/<timestamp>/`` session when no directory is given,
rather than stopping at the workspace root (which normally holds no session
artifacts at all).

This module is argv handling and printing only. All logic lives in
:mod:`hyperloom.observability`, which is library code under test — the ``tools``
tree is excluded from coverage as operator CLIs.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from hyperloom.observability import load_snapshot
from hyperloom.observability.render import detect_style, render_json, render_status


# Mirrors the exit-code contract documented in
# ``hyperloom.inference_optimizer.multi_node.cli``. Following ``rocm examine``,
# the exit code reports whether the command RAN, not what it found: a dead or
# stalled session is a successful observation and exits 0.
EXIT_OK = 0
EXIT_CONFIG_ERROR = 3
EXIT_INTERRUPT = 130

# Fast enough to feel live, slow enough that a networked session directory is
# not hammered. state.json is rewritten many times per tick, so there is no
# value in polling faster than the operator can read.
DEFAULT_INTERVAL_SEC = 2.0


def add_status_arguments(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Attach the status flags to ``parser``.

    Shared by the standalone tool and the ``inference_optimizer status``
    subcommand so the two surfaces cannot drift apart.

    Args:
        parser: Parser or subparser to populate.

    Returns:
        The same parser, for chaining.
    """
    parser.add_argument(
        "--session-dir",
        default=None,
        help=(
            "Session directory. Defaults to $INFERENCE_OPTIMIZER_CURRENT_SESSION_DIR, "
            "then the newest per-model timestamped session under the workspace root."
        ),
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Narrow session auto-discovery to one model basename.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit the machine-readable snapshot instead of the text view.",
    )
    parser.add_argument(
        "--watch",
        action="store_true",
        help="Repaint continuously until interrupted.",
    )
    parser.add_argument(
        "--interval",
        type=float,
        default=DEFAULT_INTERVAL_SEC,
        help=f"Seconds between refreshes in --watch mode (default: {DEFAULT_INTERVAL_SEC}).",
    )
    parser.add_argument(
        "--lifecycle-limit",
        type=int,
        default=5,
        help="Number of recent lifecycle events to show (default: 5).",
    )
    parser.add_argument(
        "--no-color",
        action="store_true",
        help="Disable ANSI colour even on a TTY.",
    )
    return parser


def build_parser() -> argparse.ArgumentParser:
    """Build the standalone argument parser.

    Returns:
        The configured parser.
    """
    return add_status_arguments(
        argparse.ArgumentParser(
            prog="hyperloom-status",
            description=__doc__,
            formatter_class=argparse.RawDescriptionHelpFormatter,
        )
    )


def run(args: argparse.Namespace) -> int:
    """Execute the status command from already-parsed arguments.

    Shared entry point for the standalone tool and the
    ``inference_optimizer status`` subcommand.

    Args:
        args: Parsed arguments carrying the flags from
            :func:`add_status_arguments`.

    Returns:
        Process exit code.
    """
    if args.watch:
        return _watch(args)

    code, text = _render_once(args)
    print(text, file=sys.stderr if code != EXIT_OK else sys.stdout)
    return code


def _render_once(args: argparse.Namespace) -> tuple[int, str]:
    """Load and render one snapshot.

    Args:
        args: Parsed arguments.

    Returns:
        ``(exit_code, text)``. A missing session is a configuration error; any
        session that resolves renders successfully whatever state it is in.
    """
    session_dir = Path(args.session_dir) if args.session_dir else None
    snapshot = load_snapshot(session_dir, model=args.model, lifecycle_limit=max(args.lifecycle_limit, 12))
    if snapshot is None:
        target = args.session_dir or "$INFERENCE_OPTIMIZER_CURRENT_SESSION_DIR / workspace root"
        return EXIT_CONFIG_ERROR, f"no Hyperloom session found at {target}"

    if args.json:
        return EXIT_OK, render_json(snapshot)

    style = detect_style()
    if args.no_color:
        style = type(style)(color=False, unicode=style.unicode, width=style.width)
    return EXIT_OK, render_status(snapshot, style=style, lifecycle_limit=args.lifecycle_limit)


class _Terminated(BaseException):
    """Raised from a signal handler to unwind the watch loop through ``finally``."""

    def __init__(self, signum: int) -> None:
        """Record which signal fired.

        Args:
            signum: The delivered signal number.
        """
        super().__init__(signum)
        self.signum = signum


def _watch(args: argparse.Namespace) -> int:
    """Repaint the status view until interrupted.

    Uses the alternate screen buffer so the operator's scrollback survives.
    Non-TTY output degrades to appended one-shot renders, which keeps
    ``--watch | tee`` useful instead of filling a file with escape codes.

    SIGTERM and SIGHUP are converted into an exception so the loop unwinds
    through the teardown below. Their default disposition kills the process
    outright, which would skip ``finally`` entirely and leave the operator's
    terminal in the alternate screen with the cursor still hidden — a wrecked
    terminal after any ``kill``, closed pane, or supervisor stop. Ctrl-C is
    already safe, because SIGINT raises.

    Args:
        args: Parsed arguments.

    Returns:
        The process exit code.
    """
    import signal

    interactive = sys.stdout.isatty() and not args.json

    def _raise_terminated(signum: int, _frame: object) -> None:
        raise _Terminated(signum)

    previous: list[tuple[int, object]] = []
    for name in ("SIGTERM", "SIGHUP"):
        signum = getattr(signal, name, None)
        if signum is None:
            continue  # not present on this platform
        try:
            previous.append((signum, signal.signal(signum, _raise_terminated)))
        except (OSError, ValueError):
            # Not the main thread, or the signal cannot be handled here.
            pass

    if interactive:
        sys.stdout.write("\033[?1049h\033[?25l")  # alternate screen, hide cursor
        sys.stdout.flush()
    try:
        while True:
            code, text = _render_once(args)
            if code != EXIT_OK:
                print(text, file=sys.stderr)
                return code
            if interactive:
                # Home the cursor and clear forward rather than clearing first,
                # which avoids the flash a full erase produces on each frame.
                sys.stdout.write("\033[H\033[J" + text + "\n")
                sys.stdout.flush()
            else:
                print(text, flush=True)
            time.sleep(max(0.2, float(args.interval)))
    except KeyboardInterrupt:
        return EXIT_INTERRUPT
    except _Terminated as terminated:
        return 128 + terminated.signum
    finally:
        if interactive:
            # Best-effort teardown. If the controlling terminal has already
            # gone away these writes fail, and that must not turn a clean exit
            # into a non-zero one.
            try:
                sys.stdout.write("\033[?25h\033[?1049l")
                sys.stdout.flush()
            except (OSError, ValueError):
                pass
        for signum, handler in previous:
            try:
                signal.signal(signum, handler)  # type: ignore[arg-type]
            except (OSError, ValueError):
                pass


def main(argv: list[str] | None = None) -> int:
    """Entry point.

    Args:
        argv: Argument vector; defaults to ``sys.argv[1:]``.

    Returns:
        Process exit code.
    """
    return run(build_parser().parse_args(argv))


if __name__ == "__main__":
    sys.exit(main())
