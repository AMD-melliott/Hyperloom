# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Regression: KERNEL phase-budget-pct CLI override must reach KERNEL_AGENT.

Both ``--*-kernel-pct`` flag spellings must parse and survive
``normalize_budget_pct`` under the canonical ``KERNEL_AGENT`` key.
"""

from __future__ import annotations

import inspect

import pytest

from hyperloom.inference_optimizer import cli
from hyperloom.orchestrator.phases.machine_state import (
    DEFAULT_PHASE_BUDGET_PCT,
    PHASE_FRAMEWORK_AGENT,
    PHASE_KERNEL_AGENT,
    normalize_budget_pct,
    redistribute_budget_pct,
)


def _parse_optimize(argv: list[str]) -> object:
    parser = cli._build_parser()
    return parser.parse_args(["optimize", "--model", "/tmp/m", *argv])


@pytest.mark.parametrize(
    "flag",
    ["--max-minutes-kernel-pct", "--phase-budget-kernel-pct"],
)
def test_kernel_pct_override_reaches_kernel_agent(flag: str) -> None:
    """Both flag spellings must survive normalize_budget_pct as KERNEL_AGENT."""
    args = _parse_optimize([flag, "0.78"])
    raw = cli._build_phase_budget_pct(args)
    assert raw.get(PHASE_KERNEL_AGENT) == pytest.approx(0.78)

    normalized = normalize_budget_pct(raw)
    assert normalized[PHASE_KERNEL_AGENT] == pytest.approx(0.78)
    # Must not silently fall back to the default.
    assert normalized[PHASE_KERNEL_AGENT] != pytest.approx(DEFAULT_PHASE_BUDGET_PCT[PHASE_KERNEL_AGENT])


def test_kernel_pct_key_is_canonical_phase_name() -> None:
    """The override key must be the canonical phase name, not the legacy alias."""
    args = _parse_optimize(["--phase-budget-kernel-pct", "0.5"])
    raw = cli._build_phase_budget_pct(args)
    assert "KERNEL" not in raw
    assert PHASE_KERNEL_AGENT in raw


@pytest.mark.parametrize(
    "flag",
    ["--max-minutes-framework-pct", "--phase-budget-framework-pct"],
)
def test_framework_pct_override_reaches_framework_agent(flag: str) -> None:
    """FRAMEWORK_AGENT is a budgeted phase, so both flag spellings must parse
    and survive normalize_budget_pct as FRAMEWORK_AGENT."""
    args = _parse_optimize([flag, "0.42"])
    raw = cli._build_phase_budget_pct(args)
    assert raw.get(PHASE_FRAMEWORK_AGENT) == pytest.approx(0.42)

    normalized = normalize_budget_pct(raw)
    assert normalized[PHASE_FRAMEWORK_AGENT] == pytest.approx(0.42)
    # Must not silently fall back to the default.
    assert normalized[PHASE_FRAMEWORK_AGENT] != pytest.approx(DEFAULT_PHASE_BUDGET_PCT[PHASE_FRAMEWORK_AGENT])


def test_framework_pct_key_is_canonical_phase_name() -> None:
    """The override key must be the canonical phase name, not the legacy alias."""
    args = _parse_optimize(["--phase-budget-framework-pct", "0.2"])
    raw = cli._build_phase_budget_pct(args)
    assert "FRAMEWORK" not in raw
    assert PHASE_FRAMEWORK_AGENT in raw


def test_all_phase_budget_pct_spellings_parse() -> None:
    """Every phase accepts both the legacy and the phase-budget spelling."""
    argv = [
        "--phase-budget-prelude-pct",
        "0.05",
        "--phase-budget-framework-pct",
        "0.20",
        "--phase-budget-kernel-pct",
        "0.40",
        "--phase-budget-sweep-pct",
        "0.15",
        "--phase-budget-close-pct",
        "0.03",
    ]
    args = _parse_optimize(argv)
    normalized = normalize_budget_pct(cli._build_phase_budget_pct(args))
    assert normalized["PRELUDE"] == pytest.approx(0.05)
    assert normalized[PHASE_FRAMEWORK_AGENT] == pytest.approx(0.20)
    assert normalized[PHASE_KERNEL_AGENT] == pytest.approx(0.40)
    assert normalized["SWEEP"] == pytest.approx(0.15)
    assert normalized["CLOSE"] == pytest.approx(0.03)


def test_qwen3_8b_3h_no_kernel_budget_shape() -> None:
    """The 3h demo budget stays normalized after disabling framework/kernel.

    The demo passes explicit optimisation/SWEEP caps because disabled phase
    shares are redistributed onto the remaining work phases; a lone 0.95
    override would combine with defaults to over-budget after redistribution.

    The two literals below MUST stay in lockstep with the flags documented in
    ``examples/hyperloom-qwen3-8b-3h/SKILL.md``: they are chosen so the demo's
    overrides plus the *defaults* for the phases it does not override still sum
    to exactly 1.0, so they move whenever a default they lean on moves.
    """
    args = _parse_optimize(
        [
            "--max-hours",
            "3",
            "--max-minutes-framework-pct",
            "0.44",
            "--max-minutes-sweep-pct",
            "0.01",
            "--no-framework-agent",
            "--no-kernel",
            "--no-enable-conc-sweep",
        ]
    )
    normalized = normalize_budget_pct(cli._build_phase_budget_pct(args))
    assert sum(normalized.values()) == pytest.approx(1.0)

    out = redistribute_budget_pct(
        normalized,
        kernel_enabled=not args.no_kernel,
        optimize_enabled=not args.no_framework_agent,
    )
    assert out[PHASE_FRAMEWORK_AGENT] == 0.0
    assert out[PHASE_KERNEL_AGENT] == 0.0
    assert out["PRELUDE"] == pytest.approx(0.03)
    assert out["CLOSE"] == pytest.approx(0.02)
    # SWEEP is the only work phase left, so it absorbs both freed shares.
    assert out["SWEEP"] == pytest.approx(0.95)
    assert sum(out.values()) == pytest.approx(1.0)


def test_optimize_path_is_wired_to_helper() -> None:
    """Guard: the live optimize path must build the budget via the helper.

    Asserts the helper is actually called and the buggy inline literal is gone
    from the module.
    """
    src = inspect.getsource(cli)
    assert "_build_phase_budget_pct(args)" in src
    assert '("phase_budget_kernel_pct", "KERNEL")' not in src


def test_redistribute_disabled_phase_share_goes_to_work_phases() -> None:
    """A disabled phase's pct is zeroed and spread across enabled work phases.

    PRELUDE/CLOSE are fixed overhead and must not absorb; the total is preserved.
    """
    base = dict(DEFAULT_PHASE_BUDGET_PCT)
    out = redistribute_budget_pct(base, kernel_enabled=False, optimize_enabled=True)
    assert out[PHASE_KERNEL_AGENT] == 0.0
    assert out["PRELUDE"] == base["PRELUDE"]
    assert out["CLOSE"] == base["CLOSE"]
    # Freed KERNEL share landed on the enabled work phases.
    assert out[PHASE_FRAMEWORK_AGENT] > base[PHASE_FRAMEWORK_AGENT]
    assert out["SWEEP"] > base["SWEEP"]
    assert sum(out.values()) == pytest.approx(sum(base.values()))


def test_redistribute_all_enabled_is_noop_and_idempotent() -> None:
    """No disabled phase → unchanged; re-running never drifts."""
    base = dict(DEFAULT_PHASE_BUDGET_PCT)
    once = redistribute_budget_pct(base, kernel_enabled=True, optimize_enabled=True)
    assert once == base
    twice = redistribute_budget_pct(once, kernel_enabled=True, optimize_enabled=True)
    assert twice == once


def test_a_disabled_phase_survives_normalization() -> None:
    """Zero means disabled, and must not be read back as a missing override.

    ``normalize_budget_pct`` treated ``0.0`` as out of range and restored the
    phase's library default, which silently undid ``redistribute_budget_pct``.
    On a ``--no-kernel`` run the persisted map came back with KERNEL_AGENT at
    0.35 and summed to 1.35; the phantom share then inflated the charge-back
    denominator, shrinking every remaining phase's allotment by about a quarter.
    """
    disabled = redistribute_budget_pct(
        dict(DEFAULT_PHASE_BUDGET_PCT),
        kernel_enabled=False,
        optimize_enabled=True,
    )
    assert disabled[PHASE_KERNEL_AGENT] == 0.0

    round_tripped = normalize_budget_pct(disabled)

    assert round_tripped[PHASE_KERNEL_AGENT] == 0.0
    assert round_tripped == disabled, "normalization must be a fixed point on a redistributed map"
    assert sum(round_tripped.values()) == pytest.approx(1.0)


def test_redistribute_is_idempotent_across_a_normalize_round_trip() -> None:
    """State is persisted and reloaded every tick, so the cycle must not drift.

    ``redistribute_budget_pct`` documents itself as idempotent, but the live
    path runs it on a map that has been through ``normalize_budget_pct`` on the
    way back out of ``state.json`` — which is where the drift actually happened.
    """
    base = dict(DEFAULT_PHASE_BUDGET_PCT)
    settled = redistribute_budget_pct(base, kernel_enabled=False, optimize_enabled=True)

    for _ in range(3):
        settled = redistribute_budget_pct(
            normalize_budget_pct(settled),
            kernel_enabled=False,
            optimize_enabled=True,
        )

    assert settled == redistribute_budget_pct(base, kernel_enabled=False, optimize_enabled=True)
    assert sum(settled.values()) == pytest.approx(1.0)


def test_an_out_of_range_override_is_still_rejected() -> None:
    """Widening the floor to include 0.0 must not let nonsense through."""
    for bad in (-0.1, 1.5, float("nan"), "abc", None):
        assert normalize_budget_pct({PHASE_FRAMEWORK_AGENT: bad})[PHASE_FRAMEWORK_AGENT] == pytest.approx(
            DEFAULT_PHASE_BUDGET_PCT[PHASE_FRAMEWORK_AGENT]
        )
