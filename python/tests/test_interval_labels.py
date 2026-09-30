"""Intervals as row labels, which `IntervalIndex` is an `Index` of.

pandas labels a series by intervals the way it labels one by anything else,
and `value_counts(bins=)` answers its counts under an `IntervalIndex`. `loc`
by an interval finds that interval, and by a point finds the intervals that
hold it. Each test runs the same code on both libraries and compares what
they print.
"""

from __future__ import annotations

from typing import Any

import pandas as pd
import pytest

NAN = float("nan")


def labelled(lib: Any) -> Any:
    return lib.Series(range(4), index=lib.interval_range(0, 4))


BUILDS = {
    "value_counts bins": lambda lib: lib.Series([1.0, 2.0, 2.5, 7.0, 9.0]).value_counts(bins=3),
    "bins unsorted": lambda lib: lib.Series([1.0, 2.0, 7.0]).value_counts(bins=2, sort=False),
    "bins index type": lambda lib: type(lib.Series([1.0, 5.0]).value_counts(bins=2).index).__name__,
    "labels": lambda lib: labelled(lib),
    "label type": lambda lib: type(labelled(lib).index).__name__,
    "loc point": lambda lib: int(labelled(lib).loc[2.5]),
    "loc interval": lambda lib: int(labelled(lib).loc[lib.Interval(1, 2)]),
    "frame loc point": lambda lib: lib.DataFrame(
        {"a": range(3)}, index=lib.interval_range(0, 3)
    ).loc[0.5],
    "sort_index": lambda lib: labelled(lib).sort_index(ascending=False),
    "slice": lambda lib: lib.interval_range(0, 4)[1:3],
    "mask": lambda lib: lib.interval_range(0, 4)[[True, False, True, False]],
    "item": lambda lib: lib.interval_range(0, 4)[2],
    "sort_values": lambda lib: lib.interval_range(0, 4)[::-1].sort_values(),
    "unique": lambda lib: lib.IntervalIndex.from_tuples([(0, 1), (0, 1), (1, 2)]).unique(),
    "append": lambda lib: lib.interval_range(0, 2).append(lib.interval_range(2, 4)),
    "empty": lambda lib: lib.IntervalIndex([]),
    "gap": lambda lib: lib.IntervalIndex.from_tuples([(0, 1), NAN]),
    "contains": lambda lib: (
        lib.Interval(1, 2) in lib.interval_range(0, 4),
        1.5 in lib.interval_range(0, 4),
    ),
    "get_loc": lambda lib: int(lib.interval_range(0, 4).get_loc(lib.Interval(2, 3))),
    "is index": lambda lib: isinstance(lib.interval_range(0, 2), lib.Index),
}


@pytest.mark.parametrize("make", BUILDS.values(), ids=BUILDS.keys())
def test_interval_labels_are_pandas(firepanda: Any, make: Any) -> None:
    assert repr(make(firepanda)) == repr(make(pd))


def test_a_point_no_interval_holds_is_missing(firepanda: Any) -> None:
    with pytest.raises(KeyError):
        labelled(firepanda).loc[9]
    with pytest.raises(KeyError):
        labelled(firepanda).loc[firepanda.Interval(0, 2)]
