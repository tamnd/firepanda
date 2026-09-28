"""The grouped `kurt` and `SeriesGroupBy.ohlc`, compared with pandas.

The core has no grouped kurtosis, so it is built from grouped sums, and these
cases check that the answer, the row labels and the rules for small and flat
groups come out as pandas' own kernel gives them.
"""

from __future__ import annotations

import math
from typing import Any

import pytest

import firepanda as fp

pd = pytest.importorskip("pandas")

DATA: dict[str, Any] = {
    "k": ["a", "a", "a", "a", "b", "b", "b", "b", "b", "c", None, None, None, None],
    "v": [1.0, 2.0, 4.0, 9.0, 3.0, 3.5, None, 8.0, 1.0, 2.0, 1.0, 1.0, 1.0, 1.0],
    "w": [1, 5, 2, 7, 3, 3, 4, 1, 0, 2, 5, 5, 5, 6],
    "b": [True, False] * 7,
}

GROUPINGS: list[dict[str, Any]] = [
    {},
    {"as_index": False},
    {"dropna": False},
    {"sort": False},
    {"dropna": False, "sort": False},
    {"dropna": False, "as_index": False},
]


def plain(value: Any) -> Any:
    """A value as Python has it, NaN and None as None and floats rounded past noise."""
    if hasattr(value, "item"):
        value = value.item()
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    if isinstance(value, float):
        return round(value, 9)
    return value


def shown(answer: Any) -> Any:
    """The labels, the names, the types and the values of a frame or a series."""
    if not hasattr(answer, "columns"):
        return (
            [plain(label) for label in answer.index],
            answer.name,
            str(answer.dtype),
            [plain(value) for value in answer.tolist()],
        )
    return (
        [plain(label) for label in answer.index],
        answer.index.name,
        [str(name) for name in answer.columns],
        [str(kind).replace("string", "str") for kind in answer.dtypes],
        [[plain(value) for value in answer[name].tolist()] for name in answer.columns],
    )


def outcome(make: Any) -> Any:
    """What a call gives, or the mistake it makes."""
    try:
        return shown(make())
    except Exception as error:
        return f"{type(error).__name__}: {error}"


@pytest.mark.parametrize("grouping", GROUPINGS, ids=repr)
def test_kurt_over_a_frame(grouping: dict[str, Any]) -> None:
    def make(lib: Any) -> Any:
        return lib.DataFrame(DATA).groupby("k", **grouping).kurt()

    assert outcome(lambda: make(fp)) == outcome(lambda: make(pd))


@pytest.mark.parametrize("grouping", GROUPINGS, ids=repr)
@pytest.mark.parametrize("column", ["v", "w", "b"])
def test_kurt_over_a_column(grouping: dict[str, Any], column: str) -> None:
    def make(lib: Any) -> Any:
        return lib.DataFrame(DATA).groupby("k", **grouping)[column].kurt()

    assert outcome(lambda: make(fp)) == outcome(lambda: make(pd))


@pytest.mark.parametrize("grouping", GROUPINGS, ids=repr)
@pytest.mark.parametrize("column", ["v", "w", "b"])
def test_ohlc_over_a_column(grouping: dict[str, Any], column: str) -> None:
    def make(lib: Any) -> Any:
        return lib.DataFrame(DATA).groupby("k", **grouping)[column].ohlc()

    assert outcome(lambda: make(fp)) == outcome(lambda: make(pd))


@pytest.mark.parametrize(
    "call",
    [
        lambda lib: lib.DataFrame(DATA).groupby("w").kurt(),
        lambda lib: lib.DataFrame(DATA).groupby("w")["k"].kurt(),
        lambda lib: lib.DataFrame(DATA).groupby("w")["k"].ohlc(),
        lambda lib: lib.DataFrame(DATA).groupby("k").agg("kurt"),
        lambda lib: lib.DataFrame(DATA).groupby("k")["v"].agg("kurt"),
        lambda lib: lib.DataFrame(DATA).groupby("k")[["v", "w"]].kurt(),
        lambda lib: (
            lib.DataFrame({"k": ["a"] * 6, "t": lib.to_datetime(["2024-01-01"] * 6)})
            .groupby("k")
            .kurt()
        ),
        lambda lib: (
            lib.DataFrame({"k": ["a"] * 5, "v": [1e8, 1e8, 1e8, 1e8, 1e8 + 1]}).groupby("k").kurt()
        ),
    ],
    ids=["text", "text-column", "text-ohlc", "agg", "column-agg", "selection", "dates", "flat"],
)
def test_other_calls_and_mistakes(call: Any) -> None:
    assert outcome(lambda: call(fp)) == outcome(lambda: call(pd))


def test_the_signatures_are_pandas_signatures() -> None:
    import inspect

    for name, klass in (("DataFrameGroupBy", "kurt"), ("SeriesGroupBy", "kurt")):
        ours = inspect.signature(getattr(fp._frame.__dict__[name], klass))
        theirs = inspect.signature(getattr(getattr(pd.core.groupby, name), klass))
        assert list(ours.parameters) == list(theirs.parameters)
    ours = inspect.signature(fp._frame.SeriesGroupBy.ohlc)
    assert list(ours.parameters) == list(
        inspect.signature(pd.core.groupby.SeriesGroupBy.ohlc).parameters
    )
