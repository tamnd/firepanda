"""Interval columns and the labels `cut` and `qcut` write, against pandas, spec 102.

Each case runs in both libraries and is compared by repr, and the Arrow export is
compared by type, since pandas writes intervals as its own `pandas.interval` type.
"""

from __future__ import annotations

from collections.abc import Callable
from types import ModuleType
from typing import Any

import pytest

import firepanda as fp

pd = pytest.importorskip("pandas")
pa = pytest.importorskip("pyarrow")


def ten(lib: ModuleType) -> Any:
    return lib.Series([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])


CASES: dict[str, Callable[[ModuleType], Any]] = {
    "float-column": lambda lib: lib.Series(
        [lib.Interval(0, 1), lib.Interval(1, 2.5), None], name="x"
    ),
    "int-column": lambda lib: lib.Series([lib.Interval(1, 2), lib.Interval(0, 1)]),
    "left-closed": lambda lib: lib.Series([lib.Interval(0, 1, "left")]),
    "dtype": lambda lib: lib.Series([lib.Interval(0, 1)]).dtype,
    "tolist": lambda lib: lib.Series([lib.Interval(0, 1), None]).tolist(),
    "isna": lambda lib: lib.Series([lib.Interval(0, 1), None]).isna(),
    "take": lambda lib: lib.Series([lib.Interval(0, 1), None]).take([1, 0]),
    "category": lambda lib: lib.Series(
        [lib.Interval(1, 2), lib.Interval(0, 1), lib.Interval(0, 1)]
    ).astype("category"),
    "categories": lambda lib: (
        lib.Series([lib.Interval(1, 2), lib.Interval(0, 1)]).astype("category").cat.categories
    ),
    "from-breaks": lambda lib: lib.IntervalIndex.from_breaks([0, 1, 2]),
    "from-tuples": lambda lib: lib.IntervalIndex.from_tuples([(0, 1.5), (2, 3)], closed="left"),
    "left-mid": lambda lib: (
        lib.IntervalIndex.from_breaks([0, 1, 3]).left,
        lib.IntervalIndex.from_breaks([0, 1, 3]).mid,
    ),
    "frame": lambda lib: lib.DataFrame({"a": [lib.Interval(0, 1)], "b": [1]}),
    "cut-count": lambda lib: lib.cut(ten(lib), 3),
    "cut-edges": lambda lib: lib.cut(lib.Series([1, 5, 7, 12]), [0, 5, 10, 15]),
    "cut-left": lambda lib: lib.cut(lib.Series([1, 5, 7, 12]), [0, 5, 10, 15], right=False),
    "cut-lowest": lambda lib: lib.cut(
        lib.Series([0, 5, 7, 12]), [0, 5, 10, 15], include_lowest=True
    ),
    "cut-small": lambda lib: lib.cut(lib.Series([0.001, 0.002, 0.0035, 0.01]), 4),
    "cut-gap": lambda lib: lib.cut(lib.Series([1.0, None, 3.0, 20.0]), [0, 2, 4]),
    "cut-counts": lambda lib: lib.cut(ten(lib), 3).value_counts(),
    "cut-categories": lambda lib: lib.cut(lib.Series([1, 2, 3, 10]), 2).cat.categories,
    "qcut": lambda lib: lib.qcut(lib.Series(range(10)), 4),
    "qcut-precision": lambda lib: lib.qcut(
        lib.Series([0.11, 0.2, 0.33, 0.5, 0.91]), 2, precision=1
    ),
    "qcut-drop": lambda lib: lib.qcut(lib.Series([1, 1, 1, 2, 3]), 3, duplicates="drop"),
    "group-by-bin": lambda lib: (
        lib.DataFrame({"v": [1, 2, 3]})
        .groupby(lib.cut(lib.Series([1, 5, 9]), 2), observed=True)["v"]
        .sum()
    ),
    "equals-interval": lambda lib: (
        lib.cut(lib.Series([1, 5, 9]), [0, 5, 10]) == lib.Interval(0, 5)
    ).tolist(),
    "as-text": lambda lib: lib.cut(lib.Series([1, 5, 9]), [0, 5, 10]).astype(str).tolist(),
}


def outcome(build: Callable[[], Any]) -> str:
    try:
        return repr(build())
    except Exception as error:
        return f"{type(error).__name__}: {error}"


@pytest.mark.parametrize("case", CASES.values(), ids=CASES.keys())
def test_intervals_answer_as_pandas(case: Callable[[ModuleType], Any]) -> None:
    assert outcome(lambda: case(fp)) == outcome(lambda: case(pd))


EXPORTS: dict[str, Callable[[ModuleType], Any]] = {
    "column": lambda lib: pa.array(lib.Series([lib.Interval(0, 1), None])),
    "category": lambda lib: pa.array(lib.cut(ten(lib), 3)),
    "counts-index": lambda lib: pa.array(lib.cut(ten(lib), 3).value_counts().index),
    "frame": lambda lib: pa.table(lib.DataFrame({"a": lib.cut(ten(lib), 3)})).column(0),
    "number-category": lambda lib: pa.array(lib.Series([2, 1]).astype("category")),
}


@pytest.mark.parametrize("case", EXPORTS.values(), ids=EXPORTS.keys())
def test_intervals_export_as_pandas(case: Callable[[ModuleType], Any]) -> None:
    mine, theirs = case(fp), case(pd)
    kind = mine.type
    wanted = theirs.type
    if pa.types.is_dictionary(kind):
        assert pa.types.is_dictionary(wanted)
        assert kind.ordered == wanted.ordered
        kind, wanted = kind.value_type, wanted.value_type
    assert kind == wanted
    assert mine.to_pylist() == theirs.to_pylist()


RANGES: dict[str, Callable[[ModuleType], Any]] = {
    "start-end": lambda lib: lib.interval_range(0, 5),
    "start-periods": lambda lib: lib.interval_range(start=0, periods=4),
    "end-periods": lambda lib: lib.interval_range(end=5, periods=4),
    "float-step": lambda lib: lib.interval_range(0, 10, freq=2.5),
    "split": lambda lib: lib.interval_range(0, 1, periods=4),
    "whole-split": lambda lib: lib.interval_range(0, 8, periods=4),
    "float-ends": lambda lib: lib.interval_range(0.0, 3.0),
    "closed-named": lambda lib: lib.interval_range(0, 3, closed="both", name="x"),
    "step-past-end": lambda lib: lib.interval_range(0, 7, freq=2),
    "all-four": lambda lib: lib.interval_range(0, 5, periods=2, freq=1),
    "text-start": lambda lib: lib.interval_range("a", 5),
    "part-periods": lambda lib: lib.interval_range(0, periods=1.5),
    "cut-by-range": lambda lib: lib.cut(lib.Series([1, 3, 5]), lib.interval_range(0, 6, freq=2)),
    "cut-by-left": lambda lib: lib.cut(
        lib.Series([0, 2, 3.5, 6, None]), lib.interval_range(0, 6, freq=2, closed="left")
    ),
    "cut-outside": lambda lib: lib.cut(lib.Series([0, 2, 6, 7]), lib.interval_range(0, 6, freq=2)),
    "cut-between": lambda lib: lib.cut(
        lib.Series([1, 5]), lib.IntervalIndex.from_tuples([(0, 1), (4, 6)])
    ),
    "cut-open": lambda lib: lib.cut(
        lib.Series([1, 2, 3]), lib.IntervalIndex.from_tuples([(0, 2), (2, 4)], closed="neither")
    ),
    "cut-overlap": lambda lib: lib.cut(
        lib.Series([1]), lib.IntervalIndex.from_tuples([(0, 2), (1, 4)])
    ),
    "days": lambda lib: lib.interval_range(lib.Timestamp("2024-01-01"), periods=3),
    "days-to-end": lambda lib: lib.interval_range(
        lib.Timestamp("2024-01-01"), lib.Timestamp("2024-01-05")
    ),
    "days-by-two": lambda lib: lib.interval_range(
        lib.Timestamp("2024-01-01"), lib.Timestamp("2024-01-05"), freq="2D"
    ),
    "hours": lambda lib: lib.interval_range(lib.Timestamp("2024-01-01"), periods=2, freq="12h"),
    "days-back": lambda lib: lib.interval_range(end=lib.Timestamp("2024-01-05"), periods=2),
    "days-split": lambda lib: lib.interval_range(
        lib.Timestamp("2024-01-01"), lib.Timestamp("2024-01-05"), periods=3
    ),
    "months": lambda lib: lib.interval_range(lib.Timestamp("2024-01-01"), periods=2, freq="MS"),
    "zoned": lambda lib: lib.interval_range(lib.Timestamp("2024-01-01", tz="UTC"), periods=2),
    "spans": lambda lib: lib.interval_range(lib.Timedelta("1h"), periods=2, freq="30min"),
    "spans-short": lambda lib: lib.interval_range(lib.Timedelta("0h"), lib.Timedelta("3h")),
    "spans-split": lambda lib: lib.interval_range(
        lib.Timedelta("0h"), lib.Timedelta("3h"), periods=4
    ),
    "instant-and-number": lambda lib: lib.interval_range(lib.Timestamp("2024-01-01"), 5),
    "days-left": lambda lib: lib.interval_range(
        lib.Timestamp("2024-01-01"), periods=2, closed="left", name="k"
    ),
}


def plain_outcome(build: Callable[[], Any]) -> str:
    """The repr, or the mistake as its builtin class and message."""
    try:
        return repr(build())
    except Exception as error:
        kind = next(k.__name__ for k in type(error).__mro__ if k.__module__ == "builtins")
        return f"{kind}: {error}"


@pytest.mark.parametrize("case", RANGES.values(), ids=RANGES.keys())
def test_ranges_answer_as_pandas(case: Callable[[ModuleType], Any]) -> None:
    assert plain_outcome(lambda: case(fp)) == plain_outcome(lambda: case(pd))
