"""`merge` and `DataFrame.merge`, checked against pandas.

The core pairs the rows and the Python layer does what pandas does around the
pairing: a suffix on both sides, one key type for a pair, a missing key that
matches a missing key, the sort an outer join always has, and an integer column
with a gap in it widened to float64. Every test builds the same merge in both
libraries and compares the columns, the types and every row, in order, because
pandas 3 promises the order.
"""

from __future__ import annotations

import importlib.util
import inspect
from collections.abc import Callable
from types import ModuleType
from typing import Any

import pytest

pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("pandas") is None, reason="pandas is not installed"
)

LEFT = {"k": ["b", "a", "b", "c", None], "x": [1, 2, 3, 4, 5]}
RIGHT = {"k": ["a", "b", "d", "b", None], "y": [10, 20, 30, 40, 50]}
NUMBERS = {"k": [1, 2, 2], "x": [1, 2, 3]}
OTHERS = {"k": [2, 3], "y": [1, 2], "x": [5, 6]}
PAIRS = {"a": ["x", "x", "y", "y"], "b": [1, 2, 1, 3], "v": [0, 1, 2, 3]}


def same(got: list[Any], want: list[Any]) -> bool:
    """Equal row by row, with NaN equal to NaN and None equal to NaN."""

    def missing(value: Any) -> bool:
        return value is None or value != value

    return len(got) == len(want) and all(
        (missing(a) and missing(b)) or a == b for a, b in zip(got, want, strict=True)
    )


def agrees(got: Any, want: Any) -> None:
    """The same columns, types, rows and row labels as pandas."""
    assert list(got.columns) == list(want.columns)
    assert list(got.index) == list(want.index)
    for name in want.columns:
        printed = str(want[name].dtype)
        assert got[name].dtype == ("string" if printed == "str" else printed), name
        assert same(got[name].tolist(), want[name].tolist()), name


@pytest.mark.parametrize("how", ["inner", "left", "right", "outer"])
def test_each_join_matches_pandas_row_for_row(firepanda: ModuleType, how: str) -> None:
    """A missing key on both sides pairs with itself, as pandas pairs it."""
    import pandas as pd

    got = firepanda.merge(firepanda.DataFrame(LEFT), firepanda.DataFrame(RIGHT), on="k", how=how)
    want = pd.merge(pd.DataFrame(LEFT), pd.DataFrame(RIGHT), on="k", how=how)
    agrees(got, want)


BUILDS: list[Callable[[Any], Any]] = [
    lambda m: m.merge(m.DataFrame(NUMBERS), m.DataFrame(OTHERS), on="k"),
    lambda m: m.merge(m.DataFrame(NUMBERS), m.DataFrame(OTHERS)),
    lambda m: m.merge(m.DataFrame(NUMBERS), m.DataFrame(OTHERS), on="k", suffixes=("_l", None)),
    lambda m: m.merge(m.DataFrame(NUMBERS), m.DataFrame(OTHERS), on="k", suffixes=["_a", "_b"]),
    lambda m: m.merge(m.DataFrame(NUMBERS), m.DataFrame(OTHERS), left_on="k", right_on="y"),
    lambda m: m.merge(m.DataFrame(NUMBERS), m.DataFrame(NUMBERS), left_on="k", right_on="k"),
    lambda m: m.merge(
        m.DataFrame(NUMBERS).iloc[::-1], m.DataFrame(OTHERS), on="k", how="left", sort=True
    ),
    lambda m: m.merge(
        m.DataFrame(NUMBERS), m.DataFrame(OTHERS).astype({"k": "float64"}), on="k", how="outer"
    ),
    lambda m: m.merge(m.DataFrame(PAIRS), m.DataFrame(PAIRS), on=["a", "b"]),
    lambda m: m.merge(m.DataFrame(PAIRS), m.DataFrame(PAIRS), on="a", how="right"),
    lambda m: m.merge(
        m.DataFrame(NUMBERS), m.DataFrame(OTHERS), left_on="x", right_on="y", how="outer"
    ),
    lambda m: m.DataFrame(NUMBERS).merge(m.DataFrame(OTHERS), on="k", how="left"),
    lambda m: m.merge(m.DataFrame(NUMBERS), m.DataFrame(NUMBERS)["x"], on="x"),
    lambda m: m.merge(m.DataFrame(NUMBERS), m.DataFrame(OTHERS), on="k", validate="many_to_one"),
]


@pytest.mark.parametrize("build", BUILDS)
def test_a_merge_is_pandas_merge(firepanda: ModuleType, build: Callable[[Any], Any]) -> None:
    """Keys, suffixes, types, the sort and the method spelling."""
    import pandas as pd

    agrees(build(firepanda), build(pd))


MISTAKES: list[Callable[[Any], Any]] = [
    lambda m: m.merge(
        m.DataFrame(NUMBERS), m.DataFrame(OTHERS).rename(columns={"k": "a", "x": "b"})
    ),
    lambda m: m.merge(m.DataFrame(NUMBERS), m.DataFrame(OTHERS), on="k", left_on="k"),
    lambda m: m.merge(m.DataFrame(NUMBERS), m.DataFrame(OTHERS), left_on="k"),
    lambda m: m.merge(m.DataFrame(NUMBERS), m.DataFrame(OTHERS), right_on="k"),
    lambda m: m.merge(m.DataFrame(NUMBERS), m.DataFrame(OTHERS), left_on=["k", "x"], right_on="k"),
    lambda m: m.merge(m.DataFrame(NUMBERS), m.DataFrame(OTHERS), on="k", how="bad"),
    lambda m: m.merge(m.DataFrame(NUMBERS), m.DataFrame(OTHERS), on="k", suffixes=(None, None)),
    lambda m: m.merge(m.DataFrame(NUMBERS), m.DataFrame(OTHERS), on="k", validate="one_to_one"),
    lambda m: m.merge(m.DataFrame(OTHERS), m.DataFrame(NUMBERS), on="k", validate="m:1"),
    lambda m: m.merge(m.DataFrame(NUMBERS), m.DataFrame(OTHERS), on="k", validate="bad"),
    lambda m: m.merge(m.DataFrame(NUMBERS), m.Series([1])),
    lambda m: m.merge(m.DataFrame(NUMBERS), m.DataFrame({"k": ["a", "b"], "y": [1, 2]}), on="k"),
]


@pytest.mark.parametrize("build", MISTAKES)
def test_a_mistake_is_pandas_mistake(firepanda: ModuleType, build: Callable[[Any], Any]) -> None:
    """The same class, or one of the same name, and the same first line."""
    import pandas as pd

    with pytest.raises(Exception) as theirs:
        build(pd)
    with pytest.raises(Exception) as mine:
        build(firepanda)
    kind = type(theirs.value)
    if kind.__module__ == "builtins":
        assert isinstance(mine.value, kind)
    else:
        assert type(mine.value).__name__ == kind.__name__
        assert isinstance(mine.value, kind.__mro__[1])
    assert str(mine.value).split("\n")[0] == str(theirs.value).split("\n")[0]


def test_a_missing_key_is_a_key_error(firepanda: ModuleType) -> None:
    """Named for the column, as pandas names it."""
    with pytest.raises(KeyError, match="zz"):
        firepanda.merge(firepanda.DataFrame(NUMBERS), firepanda.DataFrame(OTHERS), on="zz")


def test_the_unwritten_options_are_refused(firepanda: ModuleType) -> None:
    """Refused by name rather than ignored."""
    left, right = firepanda.DataFrame(NUMBERS), firepanda.DataFrame(OTHERS)
    for options in ({"how": "cross"}, {"left_index": True}, {"indicator": True}):
        with pytest.raises(NotImplementedError):
            firepanda.merge(left, right, **options)


def test_both_spellings_have_the_pandas_signature(firepanda: ModuleType) -> None:
    """The module function and the method, parameter for parameter."""
    import pandas as pd

    for ours, theirs in (
        (firepanda.merge, pd.merge),
        (firepanda.DataFrame.merge, pd.DataFrame.merge),
    ):
        mine = inspect.signature(ours).parameters
        yours = inspect.signature(theirs).parameters
        assert list(mine) == list(yours)
        for name in mine:
            if name != "copy":
                assert mine[name].default == yours[name].default, name
