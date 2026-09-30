"""`SeriesGroupBy.value_counts` with `bins`, and row labels that are intervals, as in pandas.

A number of bins cuts each group's values on their own and a list of edges cuts
every group alike, each group listing every bin, the empty ones too. The bins
label the rows beside the keys, so a `MultiIndex` holds intervals and a list of
intervals is an `IntervalIndex`. Each test runs the same code on both libraries
and compares what they print.
"""

from __future__ import annotations

from typing import Any

import pandas as pd
import pytest


def grouped(lib: Any) -> Any:
    values = [1.0, 2.0, 3.0, 10.0, 9.0, 1.5]
    frame = lib.DataFrame({"k": ["a", "b", "a", "b", "a", "a"], "v": values})
    return frame.groupby("k")["v"]


EDGES = [0, 2, 5, 10]

BUILDS = {
    "bins": lambda lib: grouped(lib).value_counts(bins=3),
    "edges": lambda lib: grouped(lib).value_counts(bins=EDGES),
    "unsorted": lambda lib: grouped(lib).value_counts(bins=3, sort=False),
    "normalize": lambda lib: grouped(lib).value_counts(bins=EDGES, normalize=True),
    "ascending": lambda lib: grouped(lib).value_counts(bins=EDGES, ascending=True),
    "whole numbers": lambda lib: (
        lib.DataFrame({"k": [1, 1, 2], "v": [1, 2, 3]}).groupby("k")["v"].value_counts(bins=2)
    ),
    "gap": lambda lib: (
        lib.DataFrame({"k": [1, 1, 2], "v": [1.0, None, 3.0]})
        .groupby("k")["v"]
        .value_counts(bins=[0, 2, 4])
    ),
    "outside": lambda lib: (
        lib.DataFrame({"k": [1, 1, 2], "v": [1.0, 7.0, 3.0]})
        .groupby("k")["v"]
        .value_counts(bins=[0, 2, 4])
    ),
    "swaplevel": lambda lib: grouped(lib).value_counts(bins=EDGES).swaplevel(),
    "reset_index": lambda lib: grouped(lib).value_counts(bins=EDGES).reset_index(),
    "index of intervals": lambda lib: lib.Index([lib.Interval(0, 1), lib.Interval(1, 2)]),
}


@pytest.mark.parametrize("make", BUILDS.values(), ids=BUILDS.keys())
def test_group_value_counts_bins_is_pandas(firepanda: Any, make: Any) -> None:
    assert repr(make(firepanda)) == repr(make(pd))


def test_an_interval_label_reads_back_as_it_was(firepanda: Any) -> None:
    labels = [("a", firepanda.Interval(0.5, 2.0)), ("b", firepanda.Interval(1, 3, closed="left"))]
    index = firepanda.MultiIndex.from_tuples(labels)
    assert list(index) == labels
