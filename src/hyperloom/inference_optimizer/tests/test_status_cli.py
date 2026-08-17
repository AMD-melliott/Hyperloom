# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""CLI-surface tests for ``tools/status.py``.

The rendering and data logic are tested in ``hyperloom/observability/tests``;
what is covered here is argv handling, exit codes, and terminal hygiene.
"""

from __future__ import annotations

import json
import os
import pty
import signal
import socket
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import pytest

from hyperloom.inference_optimizer.tools import status as status_tool


REPO_SRC = Path(__file__).resolve().parents[3]


def _write_session(tmp_path: Path) -> Path:
    """Create a minimal readable session directory."""
    sd = tmp_path / "model" / "20260806T000000Z"
    sd.mkdir(parents=True)
    (sd / "state.json").write_text(
        json.dumps(
            {
                "phase": "EXPLORE",
                "start_ts": "2026-08-06T00:00:00+00:00",
                "max_minutes": 720,
                "phase_started_unix": 1785974400.0,
                "model_name": "model",
            }
        ),
        encoding="utf-8",
    )
    return sd


def test_exit_ok_for_readable_session(tmp_path: Path, capsys) -> None:
    """A readable session exits 0 and prints to stdout."""
    sd = _write_session(tmp_path)

    code = status_tool.main(["--session-dir", str(sd)])

    assert code == status_tool.EXIT_OK
    assert "HYPERLOOM" in capsys.readouterr().out


def test_exit_config_error_for_missing_session(tmp_path: Path, capsys) -> None:
    """An unresolvable session is a config error on stderr, not a crash."""
    code = status_tool.main(["--session-dir", str(tmp_path / "nope")])

    assert code == status_tool.EXIT_CONFIG_ERROR
    assert "no Hyperloom session found" in capsys.readouterr().err


def test_exit_ok_for_terminal_session(tmp_path: Path, capsys) -> None:
    """Exit code reports whether the command RAN, not what it found.

    A dead session is a successful observation, so scripts can distinguish
    "could not look" from "looked, and the run is over".
    """
    sd = _write_session(tmp_path)
    state = json.loads((sd / "state.json").read_text(encoding="utf-8"))
    state["stop_reason"] = "target_reached"
    (sd / "state.json").write_text(json.dumps(state), encoding="utf-8")

    code = status_tool.main(["--session-dir", str(sd)])

    assert code == status_tool.EXIT_OK
    assert "ended" in capsys.readouterr().out


def test_json_flag_emits_parseable_document(tmp_path: Path, capsys) -> None:
    """``--json`` emits only JSON on stdout."""
    sd = _write_session(tmp_path)

    code = status_tool.main(["--session-dir", str(sd), "--json"])

    assert code == status_tool.EXIT_OK
    assert json.loads(capsys.readouterr().out)["status"]["phase"] == "EXPLORE"


def test_no_color_flag_suppresses_escapes(tmp_path: Path, capsys, monkeypatch) -> None:
    """``--no-color`` wins even when colour would otherwise be forced on."""
    sd = _write_session(tmp_path)
    monkeypatch.setenv("FORCE_COLOR", "1")

    status_tool.main(["--session-dir", str(sd), "--no-color"])

    assert "\033[" not in capsys.readouterr().out


@pytest.mark.skipif(sys.platform == "win32", reason="pty is POSIX-only")
def test_watch_restores_terminal_on_sigterm(tmp_path: Path) -> None:
    """SIGTERM must not leave the terminal in the alternate screen.

    ``SIGTERM``'s default disposition kills the process without running
    ``finally``, so without an explicit handler a ``kill``, a closed pane, or a
    supervisor stop would leave the operator with a hidden cursor and their
    scrollback replaced. Regression guard for exactly that.
    """
    sd = _write_session(tmp_path)

    controller, worker = pty.openpty()
    env = {**os.environ, "PYTHONPATH": str(REPO_SRC), "TERM": "xterm"}
    proc = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "hyperloom.inference_optimizer.tools.status",
            "--session-dir",
            str(sd),
            "--watch",
            "--interval",
            "1",
        ],
        stdout=worker,
        stderr=subprocess.DEVNULL,
        stdin=worker,
        env=env,
        # Neutral cwd: the repo root holds a `pip install --target .` copy of
        # `hyperloom/` that shadows `src/` when cwd is on sys.path.
        cwd="/",
        close_fds=True,
    )
    os.close(worker)

    try:
        deadline = time.time() + 15
        seen = b""
        while time.time() < deadline and b"\033[?1049h" not in seen:
            seen += os.read(controller, 65536)

        assert b"\033[?1049h" in seen, "watch never entered the alternate screen"

        proc.send_signal(signal.SIGTERM)
        proc.wait(timeout=15)

        tail = b""
        try:
            while True:
                chunk = os.read(controller, 65536)
                if not chunk:
                    break
                tail += chunk
        except OSError:
            pass

        combined = seen + tail
        assert b"\033[?1049l" in combined, "alternate screen was never exited"
        assert b"\033[?25h" in combined, "cursor was never restored"
        assert proc.returncode == 128 + signal.SIGTERM
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait(timeout=5)
        os.close(controller)


def test_new_flags_are_accepted(tmp_path: Path, capsys) -> None:
    """The metric flags parse and do not change the exit contract."""
    sd = _write_session(tmp_path)

    code = status_tool.main(["--session-dir", str(sd), "--no-gpu", "--no-server", "--show-sources", "--no-color"])

    assert code == status_tool.EXIT_OK
    out = capsys.readouterr().out
    assert "SOURCES" in out
    # --no-gpu / --no-server must actually skip the probes, not render empties.
    assert "GPU (host-wide)" not in out
    assert "vLLM" not in out


def test_paint_and_collection_intervals_are_independent(tmp_path: Path) -> None:
    """The elapsed timer must advance between collections, not only after one.

    This is the decoupling the whole collector exists for: with a 60s
    collection cadence and a sub-second paint rate, ``observed_at_unix`` should
    stay fixed while ``session_elapsed_s`` keeps climbing.
    """
    sd = _write_session(tmp_path)
    # Start the session ten minutes ago. The fixture's default start is a fixed
    # date in the past, which is comfortably beyond its own 720-minute budget,
    # and extrapolation deliberately clamps at the cap — so the timer would sit
    # pinned at 43200s and the test would prove nothing.
    now = time.time()
    state = json.loads((sd / "state.json").read_text(encoding="utf-8"))
    state["start_ts"] = datetime.fromtimestamp(now - 600.0, tz=timezone.utc).isoformat()
    state["phase_started_unix"] = now - 300.0
    (sd / "state.json").write_text(json.dumps(state), encoding="utf-8")

    # A lock owned by this very process, so liveness reads LIVE. Progress is
    # deliberately NOT extrapolated for an UNKNOWN or DEAD session, so without
    # this the timer would correctly stay put and the test would prove nothing.
    runtime = sd / "runtime"
    runtime.mkdir(parents=True, exist_ok=True)
    owner = {
        "pid": os.getpid(),
        "hostname": socket.gethostname(),
        "started_at": "2026-08-06T00:00:00+00:00",
        "heartbeat_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    try:
        owner["pid_ns"] = os.readlink("/proc/self/ns/pid")
    except OSError:  # pragma: no cover - non-Linux
        pass
    (runtime / "optimizer.lock").write_text(json.dumps(owner), encoding="utf-8")

    with subprocess.Popen(
        [
            sys.executable,
            "-m",
            "hyperloom.inference_optimizer.tools.status",
            "--session-dir",
            str(sd),
            "--json",
            "--watch",
            "--interval",
            "0.3",
            "--collect-interval",
            "60",
            "--no-gpu",
            "--no-server",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
        env={**os.environ, "PYTHONPATH": str(REPO_SRC)},
        # Neutral cwd: the repo root holds a `pip install --target .` copy of
        # `hyperloom/` that shadows `src/` when cwd is on sys.path.
        cwd="/",
        close_fds=True,
    ) as proc:
        time.sleep(3.0)
        proc.terminate()
        stdout, _ = proc.communicate(timeout=10)

    docs = _split_json_objects(stdout)
    assert len(docs) >= 3, f"expected several frames, got {len(docs)}"

    observed = {doc["observed_at_unix"] for doc in docs}
    assert len(observed) == 1, "collection should not have re-run at a 60s cadence"

    elapsed = [doc["budget"]["session_elapsed_s"] for doc in docs]
    assert elapsed == sorted(elapsed)
    assert elapsed[-1] > elapsed[0], "the timer must advance between collections"


def _split_json_objects(text: str) -> list[dict]:
    """Split concatenated pretty-printed JSON documents by brace depth."""
    docs: list[dict] = []
    buf = ""
    depth = 0
    in_string = False
    escaped = False
    for ch in text:
        buf += ch
        if in_string:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                docs.append(json.loads(buf))
                buf = ""
    return docs
