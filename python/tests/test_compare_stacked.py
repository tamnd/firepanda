"""`Series.compare` with the two sides taking turns down the rows.

With `align_axis=0` pandas answers a series rather than a frame, each label
there twice, once for each side, under labels of two levels: the label and the
side. The other arguments of `compare` work the same. Each test here runs the
same code on both libraries and compares what they print.
"""

from __future__ import annotations

from typing import Any

import pandas as pd
import pytest


def mine(lib: Any) -> Any:
    return lib.Series([1, 2, 3, 4], index=["w", "x", "y", "z"], name="n")


def theirs(lib: Any) -> Any:
    return lib.Series([1, 5, 3, 7], index=["w", "x", "y", "z"], name="n")


BUILDS = {
    "rows": lambda lib: mine(lib).compare(theirs(lib), align_axis=0),
    "index word": lambda lib: mine(lib).compare(theirs(lib), align_axis="index"),
    "shape": lambda lib: mine(lib).compare(theirs(lib), align_axis=0, keep_shape=True),
    "equal": lambda lib: mine(lib).compare(
        theirs(lib), align_axis=0, keep_shape=True, keep_equal=True
    ),
    "names": lambda lib: mine(lib).compare(theirs(lib), align_axis=0, result_names=("l", "r")),
    "text": lambda lib: lib.Series(["p", "q"]).compare(lib.Series(["p", "r"]), align_axis=0),
    "same": lambda lib: mine(lib).compare(mine(lib), align_axis=0),
    "same labels": lambda lib: list(mine(lib).compare(mine(lib), align_axis=0).index.names),
}


@pytest.mark.parametrize("make", BUILDS.values(), ids=BUILDS.keys())
def test_stacked_sides_are_pandas(firepanda: Any, make: Any) -> None:
    assert repr(make(firepanda)) == repr(make(pd))
