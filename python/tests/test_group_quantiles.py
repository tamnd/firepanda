"""A group by's `quantile` at a list of quantiles, or under a rule that picks a value.

pandas answers a list with one more index level after the keys, the quantiles
in the order they were asked for, and answers the four rules that pick one of
the two values either side of a quantile with each group's own quantile, so
whole numbers stay whole. Each test here runs the same code on both libraries
and compares what they print.
"""

from __future__ import annotations

from typing import Any

import pandas as pd
import pytest


def frame(lib: Any) -> Any:
    return lib.DataFrame(
        {
            "k": ["b", "a", "b", "a", "b"],
            "j": [1, 1, 2, 1, 2],
            "v": [1.0, 2.0, 3.0, 4.0, 5.0],
            "w": [5, 4, 3, 2, 1],
        }
    )


def gapped(lib: Any) -> Any:
    return lib.DataFrame({"k": ["b", None, "b", "a"], "v": [1.0, 2.0, None, 4.0]})


BUILDS = {
    "frame list": lambda lib: frame(lib).groupby("k").quantile([0.25, 0.75]),
    "series list": lambda lib: frame(lib).groupby("k")["v"].quantile([0.5, 0.1]),
    "two keys": lambda lib: frame(lib).groupby(["k", "j"]).quantile([0.5, 1.0]),
    "unsorted": lambda lib: frame(lib).groupby("k", sort=False)["w"].quantile([0.5]),
    "tuple": lambda lib: frame(lib).groupby("k")["v"].quantile((0.2,)),
    "lower": lambda lib: frame(lib).groupby("k").quantile(0.3, interpolation="lower"),
    "nearest list": lambda lib: (
        frame(lib).groupby("k")["w"].quantile([0.3, 0.6], interpolation="nearest")
    ),
    "midpoint": lambda lib: frame(lib).groupby("k")["v"].quantile(0.3, interpolation="midpoint"),
    "higher two keys": lambda lib: (
        frame(lib).groupby(["k", "j"])["w"].quantile(0.5, interpolation="higher")
    ),
    "keys as columns": lambda lib: frame(lib).groupby("k", as_index=False).quantile([0.5]),
    "missing key kept": lambda lib: (
        gapped(lib).groupby("k", dropna=False)["v"].quantile([0.5, 0.9])
    ),
    "level": lambda lib: gapped(lib).set_index("k").groupby(level=0)["v"].quantile([0.5]),
    "no rows": lambda lib: gapped(lib).iloc[:0].groupby("k")["v"].quantile([0.5]),
}


@pytest.mark.parametrize("make", BUILDS.values(), ids=BUILDS.keys())
def test_group_quantiles_are_pandas(firepanda: Any, make: Any) -> None:
    assert repr(make(firepanda)) == repr(make(pd))


def test_a_quantile_outside_zero_and_one_is_refused_in_pandas_words(firepanda: Any) -> None:
    with pytest.raises(ValueError, match="Each 'q' must be between 0 and 1"):
        frame(firepanda).groupby("k")["v"].quantile([1.5])


def test_a_text_column_is_refused(firepanda: Any) -> None:
    text = firepanda.DataFrame({"k": ["a", "b"], "t": ["x", "y"]})
    with pytest.raises(TypeError):
        text.groupby("k").quantile([0.5])
