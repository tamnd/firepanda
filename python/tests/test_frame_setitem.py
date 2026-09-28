"""`df[key] = value` and `del df[key]`, against pandas.

A write rebinds the one slot a frame holds to the frame `assign` or `mask`
answers, so a copy or a column taken before the write keeps what it had, which
is pandas' copy on write.
"""

from __future__ import annotations

import importlib.util
import math
from collections.abc import Callable
from types import ModuleType
from typing import Any

import pytest

needs_pandas = pytest.mark.skipif(
    importlib.util.find_spec("pandas") is None, reason="pandas is not installed"
)


def base(m: ModuleType) -> Any:
    return m.DataFrame({"a": [1, 2, 3], "b": [1.5, None, 3.5]}, index=[10, 20, 30])


def spelled(value: Any) -> Any:
    return "nan" if isinstance(value, float) and math.isnan(value) else value


def shown(frame: Any) -> Any:
    """The names, the kinds, the labels and the values of a frame."""
    kinds = ["str" if str(t) in ("str", "string") else str(t) for t in frame.dtypes]
    values = {c: [spelled(v) for v in frame[c].tolist()] for c in frame.columns}
    return list(frame.columns), kinds, frame.index.tolist(), values


WRITES: dict[str, Callable[[ModuleType, Any], None]] = {
    "new scalar": lambda m, d: d.__setitem__("c", 5),
    "new text": lambda m, d: d.__setitem__("c", "x"),
    "new flag": lambda m, d: d.__setitem__("c", True),
    "new instant": lambda m, d: d.__setitem__("c", m.Timestamp("2020-01-01")),
    "nan": lambda m, d: d.__setitem__("c", float("nan")),
    "replace keeps place": lambda m, d: d.__setitem__("a", [7, 8, 9]),
    "replace with a float": lambda m, d: d.__setitem__("a", 0.5),
    "series lined up": lambda m, d: d.__setitem__("c", m.Series([1, 2], index=[30, 10])),
    "series from itself": lambda m, d: d.__setitem__("c", d["a"] * 2),
    "range": lambda m, d: d.__setitem__("c", range(3)),
    "tuple": lambda m, d: d.__setitem__("c", (4, 5, 6)),
    "mapping": lambda m, d: d.__setitem__("c", {10: 1, 30: 3}),
    "frame of one": lambda m, d: d.__setitem__("c", m.DataFrame({"x": [1, 2, 3]}, index=d.index)),
    "names from rows": lambda m, d: d.__setitem__(["c", "d"], [[1, 2]] * 3),
    "names from a frame": lambda m, d: d.__setitem__(
        ["c", "a"], m.DataFrame({"x": [1, 2, 3], "y": [4, 5, 6]}, index=d.index)
    ),
    "names from one value": lambda m, d: d.__setitem__(["c", "d"], 0),
    "names from one each": lambda m, d: d.__setitem__(["c", "d"], [1, 2.5]),
    "marked rows": lambda m, d: d.__setitem__(d["a"] > 1, 0),
    "marked rows by list": lambda m, d: d.__setitem__([True, False, True], 9),
    "marked cells": lambda m, d: d.__setitem__(d > 2, 0),
    "del": lambda m, d: d.__delitem__("a"),
    "twice": lambda m, d: (d.__setitem__("c", d["a"] + 1), d.__setitem__("c", d["c"] * 10)),
}


@needs_pandas
@pytest.mark.parametrize("name", list(WRITES))
def test_a_write_lands_as_in_pandas(firepanda: ModuleType, name: str) -> None:
    """The frame after the write matches pandas' frame after the same write."""
    import pandas as pd

    mine, theirs = base(firepanda), base(pd)
    WRITES[name](firepanda, mine)
    WRITES[name](pd, theirs)
    assert shown(mine) == shown(theirs)


def test_an_empty_frame_takes_its_first_column(firepanda: ModuleType) -> None:
    """The rows come from the values, or there are none for one value."""
    frame = firepanda.DataFrame()
    frame["c"] = [1, 2]
    assert frame["c"].tolist() == [1, 2] and frame.index.tolist() == [0, 1]
    frame = firepanda.DataFrame()
    frame["c"] = 1
    assert len(frame) == 0 and list(frame.columns) == ["c"]


def test_a_copy_and_a_column_taken_before_keep_what_they_had(firepanda: ModuleType) -> None:
    """Nothing but the frame written to sees the write."""
    frame = base(firepanda)
    copy, column, same = frame.copy(), frame["a"], frame
    frame["a"] = 0
    assert copy["a"].tolist() == [1, 2, 3] and column.tolist() == [1, 2, 3]
    assert same["a"].tolist() == [0, 0, 0]


@pytest.mark.parametrize(
    ("key", "value", "error"),
    [
        ("c", [1, 2], ValueError),
        (["c", "d"], [[1, 2, 3]] * 3, ValueError),
        (["c", "d"], [1, 2, 3], ValueError),
        ([True, False], 0, ValueError),
    ],
    ids=["length", "row width", "one each", "marks"],
)
def test_a_wrong_shape_raises_as_in_pandas(
    firepanda: ModuleType, key: Any, value: Any, error: type[Exception]
) -> None:
    """The same kind of error pandas raises."""
    frame = base(firepanda)
    with pytest.raises(error):
        frame[key] = value


def test_what_needs_an_object_column_is_refused(firepanda: ModuleType) -> None:
    """None and a function are objects in pandas."""
    frame = base(firepanda)
    for value in (None, len):
        with pytest.raises(firepanda.errors.UnsupportedError):
            frame["c"] = value
    assert list(frame.columns) == ["a", "b"]


def test_del_of_a_missing_column_raises_key_error(firepanda: ModuleType) -> None:
    """As pandas does."""
    frame = base(firepanda)
    with pytest.raises(KeyError):
        del frame["z"]
