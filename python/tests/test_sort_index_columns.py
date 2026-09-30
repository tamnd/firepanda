"""`DataFrame.sort_index(axis=1)` with every argument pandas takes there.

The column labels sort a level at a time as the row labels do: through a
`key`, on the levels asked for with or without the rest, a direction for each
level, and renumbered under `ignore_index`. Each test runs the same code on
both libraries and compares what they print.
"""

from __future__ import annotations

from typing import Any

import pandas as pd
import pytest


def flat(lib: Any) -> Any:
    return lib.DataFrame({"b": [1, 2], "A": [3.0, 4.0], "c": ["x", "y"]})


def deep(lib: Any) -> Any:
    pairs = [("b", 2), ("a", 2), ("b", 1), ("a", 1)]
    return lib.DataFrame(
        [[1, 2, 3, 4]], columns=lib.MultiIndex.from_tuples(pairs, names=["x", "y"])
    )


def lowered(labels: Any) -> Any:
    return labels.str.lower()


def negated(labels: Any) -> Any:
    return -labels if labels.name == "y" else labels


BUILDS = {
    "key": lambda lib: flat(lib).sort_index(axis=1, key=lowered),
    "key falling": lambda lib: flat(lib).sort_index(axis=1, ascending=False, key=lowered),
    "ignore_index": lambda lib: flat(lib).sort_index(axis=1, ignore_index=True),
    "na_position": lambda lib: flat(lib).sort_index(axis=1, na_position="first"),
    "levels": lambda lib: deep(lib).sort_index(axis=1),
    "level": lambda lib: deep(lib).sort_index(axis=1, level="y"),
    "level alone": lambda lib: deep(lib).sort_index(axis=1, level="y", sort_remaining=False),
    "directions": lambda lib: deep(lib).sort_index(axis=1, ascending=[True, False]),
    "level key": lambda lib: deep(lib).sort_index(axis=1, level=1, key=negated),
}


@pytest.mark.parametrize("make", BUILDS.values(), ids=BUILDS.keys())
def test_column_sort_is_pandas(firepanda: Any, make: Any) -> None:
    assert repr(make(firepanda)) == repr(make(pd))


def test_in_place_sorts_the_frame_itself(firepanda: Any) -> None:
    mine, theirs = flat(firepanda), flat(pd)
    assert mine.sort_index(axis=1, key=lowered, inplace=True) is None
    theirs.sort_index(axis=1, key=lowered, inplace=True)
    assert repr(mine) == repr(theirs)
