"""Group by answers with two levels of column labels, compared with pandas.

`agg` with a list, `agg` with a list for a column, `describe` and `ohlc` over
a frame name each column by the pair of the column and the reduction. Each case
runs in both libraries and the answers are compared by their repr, or for a
mistake by its class name.
"""

from __future__ import annotations

from collections.abc import Callable
from types import ModuleType
from typing import Any

import pytest

import firepanda as fp

pd = pytest.importorskip("pandas")


def frame(lib: ModuleType) -> Any:
    return lib.DataFrame(
        {"k": ["a", "b", "a"], "j": ["u", "u", "w"], "x": [1, 2, 3], "y": [1.5, 2.5, 3.5]}
    )


def texts(lib: ModuleType) -> Any:
    return lib.DataFrame({"k": ["a", "b", "a"], "x": [1, 2, 3], "s": ["p", "q", "r"]})


CASES: dict[str, Callable[[ModuleType], Any]] = {
    "agg-list": lambda lib: frame(lib).groupby("k")[["x", "y"]].agg(["sum", "mean"]),
    "agg-list-columns": lambda lib: frame(lib).groupby("k")[["x", "y"]].agg(["min"]).columns,
    "agg-dict-of-lists": lambda lib: frame(lib).groupby("k").agg({"x": ["sum", "max"], "y": "min"}),
    "agg-lambda": lambda lib: frame(lib).groupby("k")[["x"]].agg(["sum", lambda v: v.max()]),
    "agg-two-lambdas": lambda lib: (
        frame(lib).groupby("k")[["x"]].agg([lambda v: v.min(), lambda v: v.max()])
    ),
    "agg-list-as-index-false": lambda lib: (
        frame(lib).groupby("k", as_index=False)[["x", "y"]].agg(["sum", "mean"])
    ),
    "agg-list-two-keys": lambda lib: frame(lib).groupby(["k", "j"]).agg(["min", "max"]),
    "describe": lambda lib: frame(lib).groupby("k")[["x", "y"]].describe(),
    "describe-percentiles": lambda lib: frame(lib).groupby("k")[["x"]].describe(percentiles=[0.1]),
    "describe-as-index-false": lambda lib: (
        frame(lib).groupby("k", as_index=False)[["x", "y"]].describe()
    ),
    "describe-numbers-only": lambda lib: texts(lib).groupby("k").describe(),
    "ohlc": lambda lib: frame(lib).groupby("k")[["x", "y"]].ohlc(),
    "ohlc-as-index-false": lambda lib: frame(lib).groupby("k", as_index=False)[["x", "y"]].ohlc(),
    "ohlc-two-keys": lambda lib: frame(lib).groupby(["k", "j"]).ohlc(),
    "ohlc-text": lambda lib: texts(lib).groupby("k").ohlc(),
    "agg-repeated": lambda lib: frame(lib).groupby("k").agg(["sum", "sum"]),
}


def outcome(build: Callable[[], Any]) -> str:
    try:
        return repr(build())
    except Exception as error:
        return type(error).__name__


@pytest.mark.parametrize("case", CASES.values(), ids=CASES.keys())
def test_group_levels_answer_as_pandas(case: Callable[[ModuleType], Any]) -> None:
    assert outcome(lambda: case(fp)) == outcome(lambda: case(pd))


def test_a_truncated_header_prints_the_label_past_the_dots() -> None:
    def build(lib: ModuleType) -> str:
        return repr(frame(lib).groupby("k")[["x", "y"]].describe())

    assert build(fp) == build(pd)
