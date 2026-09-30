"""`quantile` over a column of categories, which pandas reads through the codes.

pandas takes the quantile of an ordered or unordered categorical by its codes:
a picked interpolation answers a category, a blended one answers a float, and
a frame's list of quantiles keeps the column a categorical. A group by has no
kernel for categories and refuses. Each test runs the same code on both
libraries and compares what they print.
"""

from __future__ import annotations

from typing import Any

import pandas as pd
import pytest


def ordered(lib: Any) -> Any:
    return lib.Series(
        lib.Categorical(["b", "a", "c", "b", "a"], categories=["c", "b", "a"], ordered=True)
    )


def frame(lib: Any) -> Any:
    return lib.DataFrame({"c": ordered(lib), "v": [1, 2, 3, 4, 5]})


BUILDS = {
    "nearest": lambda lib: ordered(lib).quantile(0.5, interpolation="nearest"),
    "lower": lambda lib: ordered(lib).quantile(0.3, interpolation="lower"),
    "higher": lambda lib: ordered(lib).quantile(0.3, interpolation="higher"),
    "linear": lambda lib: float(ordered(lib).quantile(0.3)),
    "midpoint": lambda lib: float(ordered(lib).quantile(0.3, interpolation="midpoint")),
    "list nearest": lambda lib: ordered(lib).quantile([0.1, 0.9], interpolation="nearest"),
    "list linear": lambda lib: ordered(lib).quantile([0.1, 0.9]),
    "frame list": lambda lib: frame(lib).quantile([0.25, 0.5], numeric_only=False),
    "frame list dtypes": lambda lib: frame(lib).quantile([0.5], numeric_only=False).dtypes,
    "frame list picked": lambda lib: frame(lib).quantile(
        [0.5], numeric_only=False, interpolation="lower"
    ),
    "frame picked dtypes": lambda lib: (
        frame(lib).quantile([0.5], numeric_only=False, interpolation="lower").dtypes
    ),
}


@pytest.mark.parametrize("make", BUILDS.values(), ids=BUILDS.keys())
def test_category_quantile_is_pandas(firepanda: Any, make: Any) -> None:
    assert repr(make(firepanda)) == repr(make(pd))


def test_a_group_by_refuses_categories(firepanda: Any) -> None:
    for lib in (firepanda, pd):
        grouped = lib.DataFrame({"k": [1, 1, 2, 2, 2], "c": ordered(lib)}).groupby("k")["c"]
        with pytest.raises(TypeError, match="No matching signature found"):
            grouped.quantile(0.5)
