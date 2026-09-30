"""`skipna=False` on a group by's `any`, `all`, `idxmax` and `idxmin`, the way pandas reads it.

pandas counts a gap that is not skipped as true in `any` and `all`, the way
numpy reads NaN, and refuses to name the place of the largest or smallest
value of a group that holds a gap. Each test runs the same code on both
libraries and compares what they print.
"""

from __future__ import annotations

from typing import Any

import pandas as pd
import pytest


def frame(lib: Any) -> Any:
    return lib.DataFrame(
        {
            "k": ["a", "a", "b", "b", "c"],
            "v": [0.0, None, 0.0, 0.0, None],
            "w": [1.0, None, 2.0, 0.0, 3.0],
            "s": ["", None, "x", "", "y"],
        }
    )


BUILDS = {
    "any": lambda lib: frame(lib).groupby("k").any(skipna=False),
    "all": lambda lib: frame(lib).groupby("k").all(skipna=False),
    "column any": lambda lib: frame(lib).groupby("k")["v"].any(skipna=False),
    "column all": lambda lib: frame(lib).groupby("k")["w"].all(skipna=False),
    "as_index": lambda lib: frame(lib).groupby("k", as_index=False).any(skipna=False),
    "skipped": lambda lib: frame(lib).groupby("k").any(skipna=True),
    "idxmax whole": lambda lib: frame(lib).dropna().groupby("k")["w"].idxmax(),
}


@pytest.mark.parametrize("make", BUILDS.values(), ids=BUILDS.keys())
def test_group_skipna_is_pandas(firepanda: Any, make: Any) -> None:
    assert repr(make(firepanda)) == repr(make(pd))


@pytest.mark.parametrize(
    "build",
    [
        lambda lib: frame(lib).groupby("k")["v"].idxmax(skipna=False),
        lambda lib: frame(lib).groupby("k")["w"].idxmin(skipna=False),
        lambda lib: frame(lib).groupby("k")[["v", "w"]].idxmax(skipna=False),
    ],
)
def test_a_gap_that_is_not_skipped_has_no_place(firepanda: Any, build: Any) -> None:
    with pytest.raises(ValueError) as theirs:
        build(pd)
    with pytest.raises(ValueError) as mine:
        build(firepanda)
    assert str(mine.value) == str(theirs.value)
