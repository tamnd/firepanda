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
