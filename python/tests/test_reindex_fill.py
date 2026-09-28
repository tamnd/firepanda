"""`reindex` filling from a neighbour, and reindexing on instants, compared with pandas.

Each case builds the same column or frame in both libraries and reindexes it the
same way. The answer is compared as values, dtype, labels and label name, and a
mistake is compared by its message, with firepanda's error allowed to be a
subclass of pandas' own.
"""

from __future__ import annotations

import datetime as dt
from collections.abc import Callable
from types import ModuleType
from typing import Any

import pytest

import firepanda as fp

pd = pytest.importorskip("pandas")


def numbers(lib: ModuleType) -> Any:
    return lib.Series([1, 2, 3], index=[10, 20, 30])


def moments(lib: ModuleType) -> Any:
    index = lib.DatetimeIndex(["2024-01-01", "2024-01-03"], name="t")
    return lib.Series([1.0, 2.0], index=index)


def days(lib: ModuleType, count: int = 4) -> Any:
    return lib.date_range("2024-01-01", periods=count, freq="D")


def outcome(call: Callable[[ModuleType], Any], lib: ModuleType) -> Any:
    """The answer as plain values, or the mistake as its class and message."""
    try:
        got = call(lib)
    except Exception as error:
        return type(error), str(error)
    labels = [str(one) for one in got.index]
    if hasattr(got, "columns"):
        values = {name: got[name].tolist() for name in got.columns}
        kinds = [str(got[name].dtype) for name in got.columns]
    else:
        values, kinds = got.tolist(), [str(got.dtype)]
    return repr(values), kinds, labels, got.index.name


CASES: list[Callable[[ModuleType], Any]] = [
    lambda lib: numbers(lib).reindex([5, 15, 30], method="ffill"),
    lambda lib: numbers(lib).reindex([5, 15, 30], method="pad"),
    lambda lib: numbers(lib).reindex([5, 15, 35], method="bfill"),
    lambda lib: numbers(lib).reindex([5, 15, 35], method="backfill"),
    lambda lib: numbers(lib).reindex([5, 15, 26, 35], method="nearest"),
    lambda lib: numbers(lib).reindex([15, 16, 17, 25], method="ffill", limit=2),
    lambda lib: numbers(lib).reindex([11, 12, 13, 25], method="bfill", limit=1),
    lambda lib: numbers(lib).reindex([15, 26, 40], method="nearest", tolerance=4),
    lambda lib: numbers(lib).reindex([15, 26], method="nearest", tolerance=[1, 5]),
    lambda lib: numbers(lib).reindex([15, 26], method="ffill", fill_value=0),
    lambda lib: numbers(lib).reindex([25, 5], method="ffill"),
    lambda lib: lib.Series([1, 2], index=[3, 1]).reindex([2], method="ffill"),
    lambda lib: lib.Series([1, 2], index=[3, 1]).reindex([2, 4, 0], method="bfill"),
    lambda lib: lib.Series([1, 2], index=["a", "c"]).reindex(["b", "d"], method="bfill"),
    lambda lib: moments(lib).reindex(days(lib), method="ffill"),
    lambda lib: moments(lib).reindex(days(lib), method="nearest", tolerance="12h"),
    lambda lib: moments(lib).reindex(days(lib)),
    lambda lib: moments(lib).reindex([dt.datetime(2024, 1, 2)], fill_value=0),
    lambda lib: moments(lib).reindex(days(lib, 3).rename("z")),
    lambda lib: moments(lib).to_frame("v").reindex(days(lib), method="bfill"),
    lambda lib: moments(lib).to_frame("v").reindex(index=days(lib), columns=["v", "w"]),
    lambda lib: lib.Series([1, 2], index=lib.Index([1, 2], name="k")).reindex(
        lib.Index([2, 3], name="z")
    ),
    lambda lib: lib.DataFrame({"a": [1]}, index=lib.Index([3], name="k")).reindex(
        lib.Index([3, 4], name="z")
    ),
]


@pytest.mark.parametrize("call", CASES)
def test_the_rows_are_pandas_rows(call: Callable[[ModuleType], Any]) -> None:
    assert outcome(call, fp) == outcome(call, pd)


MISTAKES: list[Callable[[ModuleType], Any]] = [
    lambda lib: numbers(lib).reindex([5], method="zz"),
    lambda lib: numbers(lib).reindex([5], method="ffill", limit=0),
    lambda lib: numbers(lib).reindex([5], method="ffill", limit=1.5),
    lambda lib: numbers(lib).reindex([25, 15], method="ffill", limit=1),
    lambda lib: lib.Series([1, 2, 3], index=[3, 1, 2]).reindex([2], method="ffill"),
    lambda lib: lib.Series([1, 2], index=[1, 1]).reindex([2], method="ffill"),
    lambda lib: numbers(lib).reindex([15], method="nearest", tolerance="x"),
    lambda lib: numbers(lib).reindex([15, 26, 27], method="nearest", tolerance=[1, 2]),
    lambda lib: lib.Series([1, 2], index=["a", "c"]).reindex(["b"], method="nearest"),
    lambda lib: numbers(lib).reindex([15], tolerance=1),
]


@pytest.mark.parametrize("call", MISTAKES)
def test_the_mistakes_are_pandas_mistakes(call: Callable[[ModuleType], Any]) -> None:
    ours, theirs = outcome(call, fp), outcome(call, pd)
    assert isinstance(ours[0], type) and issubclass(ours[0], theirs[0])
    assert ours[1] == theirs[1]


def test_reindex_like_fills_as_pandas_does() -> None:
    def call(lib: ModuleType) -> Any:
        return moments(lib).reindex_like(lib.Series(0.0, index=days(lib)), method="ffill")

    assert outcome(call, fp) == outcome(call, pd)


def test_an_index_of_instants_is_an_index_of_instants() -> None:
    values = [dt.datetime(2024, 1, 1), dt.datetime(2024, 1, 2, 3)]
    ours, theirs = fp.Index(values, name="t"), pd.Index(values, name="t")
    assert type(ours).__name__ == type(theirs).__name__ == "DatetimeIndex"
    assert [str(one) for one in ours] == [str(one) for one in theirs]
    assert ours.name == "t"
    spans = [dt.timedelta(hours=1), dt.timedelta(days=2)]
    assert type(fp.Index(spans)).__name__ == type(pd.Index(spans)).__name__
