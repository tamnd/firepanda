"""`apply` and `agg` of a function over resample bins, in pandas' order.

pandas first takes the function as an aggregation, one value a bin, reading
a series of one as the value in it, and over a frame it tries each column.
An answer that is not one value makes it an `apply` over the whole bin, whose
pieces are put end to end, under the bin labels when `group_keys` is set.
Each test runs the same code on both libraries and compares what they print.
"""

from __future__ import annotations

from typing import Any

import pandas as pd
import pytest


def series(lib: Any) -> Any:
    index = lib.date_range("2024-01-01", periods=9, freq="25min")
    return lib.Series(range(9), index=index)


def frame(lib: Any) -> Any:
    index = lib.date_range("2024-01-01", periods=6, freq="25min")
    return lib.DataFrame({"v": range(6), "w": [1.0, 2, 3, 4, 5, 6]}, index=index)


def one(x: Any) -> Any:
    return x.head(1)


def two(x: Any) -> Any:
    return x.head(2)


def double(x: Any) -> Any:
    return x * 2


BUILDS = {
    "head1": lambda lib: series(lib).resample("h").apply(one),
    "agg head1": lambda lib: series(lib).resample("h").agg(one),
    "list answer": lambda lib: series(lib).resample("h").apply(lambda x: [x.sum()]),
    "labelled one": lambda lib: (
        series(lib).resample("h").apply(lambda x: lib.Series([x.sum()], index=["t"]))
    ),
    "head2": lambda lib: series(lib).resample("h").apply(two),
    "same": lambda lib: series(lib).resample("h").apply(double),
    "head2 keyed": lambda lib: series(lib).resample("h", group_keys=True).apply(two),
    "same keyed": lambda lib: series(lib).resample("h", group_keys=True).apply(double),
    "reset keyed": lambda lib: (
        series(lib).resample("h", group_keys=True).apply(lambda x: x.reset_index(drop=True))
    ),
    "frame sum": lambda lib: frame(lib).resample("h").apply(lambda x: x.sum()),
    "frame head1": lambda lib: frame(lib).resample("h").apply(one),
    "frame head2": lambda lib: frame(lib).resample("h").apply(two),
    "frame head2 keyed": lambda lib: frame(lib).resample("h", group_keys=True).apply(two),
    "frame same keyed": lambda lib: frame(lib).resample("h", group_keys=True).apply(double),
    "frame col": lambda lib: frame(lib).resample("h").apply(lambda x: x["v"].sum()),
    "frame col series": lambda lib: frame(lib).resample("h").apply(lambda x: x["v"].head(2)),
}


@pytest.mark.parametrize("make", BUILDS.values(), ids=BUILDS.keys())
def test_resample_apply_order_is_pandas(firepanda: Any, make: Any) -> None:
    assert repr(make(firepanda)) == repr(make(pd))
