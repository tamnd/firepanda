"""`unstack` of several levels at once.

pandas spreads the pairs of the levels handed in across the columns, in the
order the pairs first appear, under columns of several levels named after
them, and keeps the other levels as the row labels. A frame puts its own
column names on top. Each test here runs the same code on both libraries and
compares what they print.
"""

from __future__ import annotations

from typing import Any

import pandas as pd
import pytest


def series(lib: Any) -> Any:
    labels = lib.MultiIndex.from_tuples(
        [("r1", "b", "x"), ("r1", "a", "y"), ("r2", "b", "y"), ("r2", "a", "x"), ("r1", "b", "y")],
        names=["r", "p", "q"],
    )
    return lib.Series([1, 2, 3, 4, 5], index=labels, name="v")


BUILDS = {
    "two": lambda lib: series(lib).unstack(["p", "q"]),
    "two reversed": lambda lib: series(lib).unstack(["q", "p"]),
    "numbers": lambda lib: series(lib).unstack([1, 2]),
    "fill": lambda lib: series(lib).unstack(["p", "q"], fill_value=0),
    "names": lambda lib: list(series(lib).unstack(["p", "q"]).columns.names),
    "frame": lambda lib: series(lib).to_frame().unstack(["p", "q"]),
    "frame names": lambda lib: list(series(lib).to_frame().unstack(["p", "q"]).columns.names),
    "one in a list": lambda lib: series(lib).unstack(["q"]),
}


@pytest.mark.parametrize("make", BUILDS.values(), ids=BUILDS.keys())
def test_several_levels_are_pandas(firepanda: Any, make: Any) -> None:
    assert repr(make(firepanda)) == repr(make(pd))
