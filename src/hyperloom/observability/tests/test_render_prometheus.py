# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Contract tests for the Prometheus renderer."""

from __future__ import annotations

import math

from hyperloom.observability.render.prometheus import (
    MetricFamily,
    escape_label_value,
    format_families,
    format_value,
)


def test_escape_label_value() -> None:
    assert escape_label_value('a"b\\c\nd/e') == 'a\\"b\\\\c\\nd/e'


def test_format_value_specials() -> None:
    assert format_value(1.0) == "1"
    assert format_value(-3600.0) == "-3600"
    assert format_value(0.25) == "0.25"
    assert format_value(math.nan) == "NaN"
    assert format_value(math.inf) == "+Inf"
    assert format_value(-math.inf) == "-Inf"
    assert format_value(1e20) == "1e+20"


def test_none_samples_are_dropped_and_empty_families_skipped() -> None:
    kept = MetricFamily("hyperloom_a", "gauge", "Kept.")
    kept.add(None, phase="X")
    kept.add(2, phase="Y")
    empty = MetricFamily("hyperloom_b", "gauge", "Empty.")
    empty.add(None)

    text = format_families([kept, empty], const_labels={"session_id": "s"})

    assert text == ('# HELP hyperloom_a Kept.\n# TYPE hyperloom_a gauge\nhyperloom_a{session_id="s",phase="Y"} 2\n')


def test_unlabelled_sample_has_no_braces() -> None:
    family = MetricFamily("hyperloom_c", "counter", "Count.")
    family.add(True)
    assert format_families([family], const_labels={}).splitlines()[-1] == "hyperloom_c 1"
