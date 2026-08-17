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

# Paint rate. Distinct from the collection rate below: the display repaints
# every second so the elapsed timers tick visibly, while the underlying data is
# gathered on slower, per-source cadences. Decoupling the two is what keeps the
# clock moving when a probe is slow or hung.
DEFAULT_INTERVAL_SEC = 1.0

# Session-artifact collection rate. state.json is rewritten many times per
# tick, so there is no value in re-reading it faster than this.
DEFAULT_COLLECT_INTERVAL_SEC = 2.0


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
        help=f"Seconds between repaints in --watch mode (default: {DEFAULT_INTERVAL_SEC}).",
    )
    parser.add_argument(
        "--collect-interval",
        type=float,
        default=DEFAULT_COLLECT_INTERVAL_SEC,
        help=(
            "Seconds between session-artifact reads in --watch mode "
            f"(default: {DEFAULT_COLLECT_INTERVAL_SEC}). Independent of --interval: "
            "the timers keep ticking between collections."
        ),
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
    parser.add_argument(
        "--no-gpu",
        action="store_true",
        help="Skip the amd-smi GPU probe.",
    )
    parser.add_argument(
        "--vllm-url",
        default=None,
        help=(
            "Inference server base URL for /metrics. Auto-discovered from listening "
            "ports when omitted; no server is normal during a KERNEL_AGENT phase."
        ),
    )
    parser.add_argument(
        "--no-server",
        action="store_true",
        help="Skip the inference-server /metrics scrape.",
    )
    parser.add_argument(
        "--show-sources",
        action="store_true",
        help="Always show the per-source collection health footer, not only when degraded.",
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


def _style_for(args: argparse.Namespace):
    """Build the render style, honouring ``--no-color``."""
    style = detect_style()
    if args.no_color:
        style = type(style)(color=False, unicode=style.unicode, width=style.width)
    return style


def _render(args: argparse.Namespace, snapshot) -> str:
    """Render one snapshot in the requested format."""
    if args.json:
        return render_json(snapshot)
    return render_status(
        snapshot,
        style=_style_for(args),
        lifecycle_limit=args.lifecycle_limit,
        show_sources=args.show_sources,
    )


def _render_once(args: argparse.Namespace) -> tuple[int, str]:
    """Load and render one snapshot synchronously.

    The one-shot path deliberately stays inline rather than going through the
    background collector: a single invocation should be deterministic, finish,
    and — for ``--json`` — produce exactly one document from exactly one read.

    Args:
        args: Parsed arguments.

    Returns:
        ``(exit_code, text)``. A missing session is a configuration error; any
        session that resolves renders successfully whatever state it is in.
    """
    session_dir = Path(args.session_dir) if args.session_dir else None
    extras: dict[str, object] = {}
    # Carried into the snapshot alongside the inline reads so a probe that
    # fails here is warned about and appears in the SOURCES footer, rather than
    # leaving a blank where a metric should be with nothing to explain it.
    extra_reads: list[tuple[str, object]] = []

    if not args.no_gpu:
        from hyperloom.observability.sources import GpuSource

        source = GpuSource()
        result = source.read()
        extra_reads.append((source.name, result))
        if result.ok:
            extras["gpus"] = result.data

    if not args.no_server:
        from hyperloom.observability.sources import ServerMetricsSource

        source = ServerMetricsSource(base_url=args.vllm_url)
        result = source.read()
        extra_reads.append((source.name, result))
        if result.ok:
            extras["server"] = result.data

    snapshot = load_snapshot(
        session_dir,
        model=args.model,
        lifecycle_limit=max(args.lifecycle_limit, 12),
        extras=extras or None,
        extra_reads=extra_reads or None,
    )
    if snapshot is None:
        target = args.session_dir or "$INFERENCE_OPTIMIZER_CURRENT_SESSION_DIR / workspace root"
        return EXIT_CONFIG_ERROR, f"no Hyperloom session found at {target}"

    return EXIT_OK, _render(args, snapshot)


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

    Data gathering runs on background threads and the paint loop only reads a
    cache, so a slow ``amd-smi``, an unresponsive server, or a stalled network
    filesystem degrades one field rather than freezing the frame. The elapsed
    timers are advanced arithmetically on every repaint, which is why they tick
    once a second off a two-second collection cadence.

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

    from hyperloom.observability.assemble import resolve_session_dir
    from hyperloom.observability.collector import SessionMonitor

    session_dir = resolve_session_dir(Path(args.session_dir) if args.session_dir else None, model=args.model)
    if session_dir is None:
        target = args.session_dir or "$INFERENCE_OPTIMIZER_CURRENT_SESSION_DIR / workspace root"
        print(f"no Hyperloom session found at {target}", file=sys.stderr)
        return EXIT_CONFIG_ERROR

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

    monitor = SessionMonitor(
        session_dir,
        gpu=not args.no_gpu,
        server=not args.no_server,
        server_url=args.vllm_url,
        lifecycle_limit=max(args.lifecycle_limit, 12),
        session_interval_s=max(0.5, float(args.collect_interval)),
    )

    if interactive:
        sys.stdout.write("\033[?1049h\033[?25l")  # alternate screen, hide cursor
        sys.stdout.flush()
    try:
        monitor.start()
        while True:
            snapshot = monitor.current()
            if snapshot is None:
                # Nothing has been read successfully yet. Keep waiting rather
                # than exiting: a session directory that is mid-creation
                # resolves a moment later.
                time.sleep(max(0.2, float(args.interval)))
                continue
            text = _render(args, snapshot)
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
        monitor.stop()
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
