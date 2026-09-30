"""A list of reductions over a window of a frame, answered with two levels of labels.

pandas reduces each column of the frame as a column of its own and puts the
answers under the column and then the reduction. A dict with a list for a
column does the same for that column, and a window read along a column named
by `on` is refused, since the column alone has no such column. Each test here
runs the same code on both libraries and compares what they print.
"""

from __future__ import annotations

from typing import Any

import pandas as pd
import pytest


def frame(lib: Any) -> Any:
    return lib.DataFrame({"a": [1.0, 2.0, 3.0, 4.0], "b": [4, 3, 2, 1]})


BUILDS = {
    "rolling": lambda lib: frame(lib).rolling(2).agg(["sum", "mean"]),
    "expanding": lambda lib: frame(lib).expanding().agg(["sum", "max"]),
    "ewm": lambda lib: frame(lib).ewm(com=1).agg(["mean", "std"]),
    "one name": lambda lib: frame(lib).rolling(2).aggregate(["min"]),
    "tuple": lambda lib: frame(lib).rolling(3, min_periods=1).agg(("count", "median")),
    "dict of lists": lambda lib: frame(lib).rolling(2).agg({"a": ["sum", "min"], "b": "max"}),
    "columns": lambda lib: frame(lib).rolling(2).agg(["sum", "mean"]).columns,
}


@pytest.mark.parametrize("make", BUILDS.values(), ids=BUILDS.keys())
def test_reductions_under_two_levels_are_pandas(firepanda: Any, make: Any) -> None:
    assert repr(make(firepanda)) == repr(make(pd))


def test_a_window_along_a_column_is_refused_in_pandas_words(firepanda: Any) -> None:
    timed = firepanda.DataFrame(
        {"t": firepanda.date_range("2024", periods=3), "v": [1.0, 2.0, 3.0]}
    )
    with pytest.raises(ValueError, match="invalid on specified as t"):
        timed.rolling("2D", on="t").agg(["sum", "mean"])


def test_a_mapping_inside_a_dict_is_refused(firepanda: Any) -> None:
    from firepanda.errors import SpecificationError

    with pytest.raises(SpecificationError, match="nested renamer"):
        frame(firepanda).rolling(2).agg({"a": {"s": "sum"}})
