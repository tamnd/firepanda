"""Arithmetic on one level of a MultiIndex, pandas' `level=`.

pandas reads a Series on flat labels once for every row of a MultiIndex, by
the labels on the level named, and works out each row against the value it
finds there. A row whose label the other side lacks is NaN, or takes
`fill_value`. Each test here runs the same code on both libraries and compares
what they print.
"""

from __future__ import annotations

from typing import Any

import pandas as pd
import pytest


def rows(lib: Any) -> Any:
    return lib.MultiIndex.from_arrays(
        [["p", "p", "q", "r"], ["x", "y", "z", "x"]], names=["o", "i"]
    )


def series(lib: Any) -> Any:
    return lib.Series([1.0, 2.0, 4.0, 8.0], index=rows(lib), name="v")


def outer(lib: Any) -> Any:
    return lib.Series([10.0, 20.0], index=["p", "q"])


def inner(lib: Any) -> Any:
    return lib.Series([100, 200, 300], index=["x", "y", "z"], name="u")


def frame(lib: Any) -> Any:
    return lib.DataFrame({"a": [1.0, 2.0, 4.0, 8.0], "b": [1, 2, 3, 4]}, index=rows(lib))


def wide(lib: Any) -> Any:
    columns = lib.MultiIndex.from_tuples([("A", "x"), ("A", "y"), ("B", "x")])
    return lib.DataFrame([[1.0, 2.0, 3.0]], columns=columns)


BUILDS = {
    "add by name": lambda lib: series(lib).add(outer(lib), level="o"),
    "sub by number": lambda lib: series(lib).sub(inner(lib), level=1),
    "radd": lambda lib: series(lib).radd(outer(lib), level=0),
    "mul fill": lambda lib: series(lib).mul(outer(lib), level="o", fill_value=1),
    "flat first": lambda lib: outer(lib).add(series(lib), level="o"),
    "scalar": lambda lib: series(lib).add(1, level=0),
    "truediv": lambda lib: series(lib).truediv(outer(lib), level="o"),
    "eq": lambda lib: series(lib).eq(outer(lib), level="o"),
    "frame rows": lambda lib: frame(lib).mul(outer(lib), axis=0, level=0),
    "frame rows inner": lambda lib: frame(lib).add(inner(lib), axis=0, level="i"),
    "frame columns": lambda lib: wide(lib).mul(
        lib.Series([10.0, 100.0], index=["A", "B"]), axis=1, level=0
    ),
}


@pytest.mark.parametrize("make", BUILDS.values(), ids=BUILDS.keys())
def test_arithmetic_on_one_level_is_pandas(firepanda: Any, make: Any) -> None:
    assert repr(make(firepanda)) == repr(make(pd))


def test_two_multiindexes_on_one_level_are_refused(firepanda: Any) -> None:
    with pytest.raises(NotImplementedError, match="level="):
        series(firepanda).add(series(firepanda), level=0)
    with pytest.raises(NotImplementedError, match="level="):
        frame(firepanda).add(frame(firepanda), level=0)
