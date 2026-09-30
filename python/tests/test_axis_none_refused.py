"""`axis=None` on the methods pandas reads it as no axis at all rather than the default.

pandas takes `axis=None` as the default on the reductions and on the methods
that fill, mask or align, and refuses it everywhere else with `No axis named
None`. Each test runs the same call on both libraries and compares the error,
or the answer where pandas takes None.
"""

from __future__ import annotations

from typing import Any

import pandas as pd
import pytest


def frame(lib: Any) -> Any:
    return lib.DataFrame({"a": [1.0, 2.0, 7.0], "b": [4.0, 5.0, 3.0]})


def column(lib: Any) -> Any:
    return lib.Series([1.0, 2.0, 7.0])


REFUSED = {
    "frame rank": lambda lib: frame(lib).rank(axis=None),
    "frame diff": lambda lib: frame(lib).diff(axis=None),
    "frame shift": lambda lib: frame(lib).shift(axis=None),
    "frame pct_change": lambda lib: frame(lib).pct_change(axis=None),
    "frame sort_values": lambda lib: frame(lib).sort_values("a", axis=None),
    "frame sort_index": lambda lib: frame(lib).sort_index(axis=None),
    "frame dropna": lambda lib: frame(lib).dropna(axis=None),
    "frame mode": lambda lib: frame(lib).mode(axis=None),
    "frame apply": lambda lib: frame(lib).apply(sum, axis=None),
    "frame drop": lambda lib: frame(lib).drop("a", axis=None),
    "frame take": lambda lib: frame(lib).take([0], axis=None),
    "frame corrwith": lambda lib: frame(lib).corrwith(frame(lib), axis=None),
    "frame set_axis": lambda lib: frame(lib).set_axis(["x", "y", "z"], axis=None),
    "frame interpolate": lambda lib: frame(lib).interpolate(axis=None),
    "frame idxmin": lambda lib: frame(lib).idxmin(axis=None),
    "frame idxmax": lambda lib: frame(lib).idxmax(axis=None),
    "series rank": lambda lib: column(lib).rank(axis=None),
    "series shift": lambda lib: column(lib).shift(axis=None),
    "series sort_values": lambda lib: column(lib).sort_values(axis=None),
    "series sort_index": lambda lib: column(lib).sort_index(axis=None),
    "series drop": lambda lib: column(lib).drop(0, axis=None),
    "series take": lambda lib: column(lib).take([0], axis=None),
    "series take by position": lambda lib: column(lib).take([0], None),
    "series set_axis": lambda lib: column(lib).set_axis(["x", "y", "z"], axis=None),
    "series interpolate": lambda lib: column(lib).interpolate(axis=None),
    "series idxmin": lambda lib: column(lib).idxmin(axis=None),
    "series idxmax": lambda lib: column(lib).idxmax(axis=None),
}

TAKEN = {
    "frame sum": lambda lib: frame(lib).sum(axis=None),
    "frame cumsum": lambda lib: frame(lib).cumsum(axis=None),
    "frame fillna": lambda lib: frame(lib).fillna(0, axis=None),
    "frame clip": lambda lib: frame(lib).clip(1, 3, axis=None),
    "frame where": lambda lib: frame(lib).where(frame(lib) > 2, axis=None),
    "frame ffill": lambda lib: frame(lib).ffill(axis=None),
    "series dropna": lambda lib: column(lib).dropna(axis=None),
    "frame rank default": lambda lib: frame(lib).rank(),
    "series take by position": lambda lib: column(lib).take([0], 0),
}


@pytest.mark.parametrize("make", REFUSED.values(), ids=REFUSED.keys())
def test_a_none_axis_is_refused_as_pandas_refuses_it(firepanda: Any, make: Any) -> None:
    with pytest.raises(ValueError) as theirs:
        make(pd)
    with pytest.raises(ValueError) as mine:
        make(firepanda)
    assert str(mine.value) == str(theirs.value)


@pytest.mark.parametrize("make", TAKEN.values(), ids=TAKEN.keys())
def test_a_none_axis_is_taken_where_pandas_takes_it(firepanda: Any, make: Any) -> None:
    assert str(make(firepanda)) == str(make(pd))
