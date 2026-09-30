"""Ranks, shifts, differences, changes and fills across the rows of a frame.

pandas ranks, fills and interpolates along `axis=1` by working down the frame
turned on its side. It shifts along `axis=1` by moving whole columns, each
keeping its type, and takes a difference or a change as the frame against
those shifted columns. Each test here runs the same code on both libraries
and compares what they print.
"""

from __future__ import annotations

from typing import Any

import pandas as pd
import pytest


def whole(lib: Any) -> Any:
    return lib.DataFrame({"a": [1, 2], "b": [3, 1], "c": [5, 6]}, index=["x", "y"])


def holed(lib: Any) -> Any:
    return lib.DataFrame({"a": [1.0, None], "b": [3.0, 4.0], "c": [None, 6.0]})


BUILDS = {
    "rank": lambda lib: whole(lib).rank(axis=1),
    "rank pct": lambda lib: whole(lib).rank(axis=1, pct=True),
    "rank descending": lambda lib: whole(lib).rank(axis=1, ascending=False, method="min"),
    "shift": lambda lib: whole(lib).shift(1, axis=1),
    "shift back": lambda lib: whole(lib).shift(-1, axis=1),
    "shift two": lambda lib: whole(lib).shift(2, axis=1),
    "shift filled": lambda lib: whole(lib).shift(1, axis=1, fill_value=0),
    "diff": lambda lib: whole(lib).diff(axis=1),
    "diff back": lambda lib: whole(lib).diff(-1, axis=1),
    "diff gaps": lambda lib: holed(lib).diff(axis=1),
    "pct_change": lambda lib: whole(lib).pct_change(axis=1),
    "pct_change down": lambda lib: whole(lib).pct_change(),
    "interpolate": lambda lib: holed(lib).interpolate(axis=1),
    "ffill": lambda lib: holed(lib).ffill(axis=1),
    "bfill": lambda lib: holed(lib).bfill(axis=1),
    "ffill limit": lambda lib: lib.DataFrame(
        {"a": [1.0, 2.0], "b": [None, 3.0], "c": [None, None]}, dtype="float64"
    ).ffill(axis=1, limit=1),
}


@pytest.mark.parametrize("make", BUILDS.values(), ids=BUILDS.keys())
def test_transforms_across_rows_are_pandas(firepanda: Any, make: Any) -> None:
    assert repr(make(firepanda)) == repr(make(pd))
