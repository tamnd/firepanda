"""`limit_area` on `ffill` and `bfill`, which keeps only some of the fills.

pandas fills the gaps between two present values under "inside" and only the
gaps at either end under "outside", and it reads any other word as "inside".
Each case runs the same code on both libraries and compares what they print.
"""

from __future__ import annotations

from typing import Any

import pandas as pd
import pytest


def holed(lib: Any) -> Any:
    return lib.Series([None, 1.0, None, None, 3.0, None, None])


def frame(lib: Any) -> Any:
    return lib.DataFrame(
        {
            "a": [None, 1.0, None, 2.0, None],
            "b": [1, 2, 3, 4, 5],
            "c": [None, None, "x", None, None],
        }
    )


def filled_in_place(lib: Any) -> Any:
    column = holed(lib)
    column.ffill(limit_area="inside", inplace=True)
    return column


BUILDS = {
    "ffill inside": lambda lib: holed(lib).ffill(limit_area="inside"),
    "bfill inside": lambda lib: holed(lib).bfill(limit_area="inside"),
    "ffill outside": lambda lib: holed(lib).ffill(limit_area="outside"),
    "bfill outside": lambda lib: holed(lib).bfill(limit_area="outside"),
    "ffill inside limit": lambda lib: holed(lib).ffill(limit_area="inside", limit=1),
    "bfill inside limit": lambda lib: holed(lib).bfill(limit_area="inside", limit=1),
    "ffill outside limit": lambda lib: holed(lib).ffill(limit_area="outside", limit=1),
    "another word": lambda lib: holed(lib).bfill(limit_area="Inside"),
    "in place": filled_in_place,
    "empty": lambda lib: lib.Series([], dtype="float64").ffill(limit_area="inside"),
    "no gaps": lambda lib: lib.Series([1, 2]).ffill(limit_area="outside"),
    "repeated labels": lambda lib: lib.Series(["a", None, "b", None], index=[3, 3, 1, 0]).ffill(
        limit_area="inside"
    ),
    "frame inside": lambda lib: frame(lib).ffill(limit_area="inside"),
    "frame outside": lambda lib: frame(lib).bfill(limit_area="outside"),
    "frame across": lambda lib: frame(lib)[["a", "c"]].ffill(axis=1, limit_area="inside"),
    "frame dtypes": lambda lib: [str(t) for t in frame(lib).ffill(limit_area="inside").dtypes],
}


@pytest.mark.parametrize("make", BUILDS.values(), ids=BUILDS.keys())
def test_fill_area_is_pandas(firepanda: Any, make: Any) -> None:
    assert repr(make(firepanda)) == repr(make(pd))
