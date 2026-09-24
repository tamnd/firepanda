"""`concat`, checked against pandas.

The core stacks the parts and the Python layer does what pandas does around the
stacking: the union of the columns in the order they first appear, one type per
column by numpy's rule, float64 for an integer column with a gap, the row labels
carried along or numbered again, and on the other axis the parts lined up on
their row labels. Every test builds the same concat in both libraries and
compares the columns, the types, the row labels and every row.
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

FIRST = {"a": [1, 2], "b": ["x", "y"]}
SECOND = {"b": ["z"], "a": [3]}
OTHER = {"c": [1.5, 2.5, 3.5]}


def same(got: list[Any], want: list[Any]) -> bool:
    """Equal row by row, with NaN equal to NaN and None equal to NaN."""

    def missing(value: Any) -> bool:
        return value is None or value != value

    return len(got) == len(want) and all(
        (missing(a) and missing(b)) or a == b for a, b in zip(got, want, strict=True)
    )


def printed(dtype: Any) -> str:
    """A pandas type in the words firepanda uses for it."""
    return "string" if str(dtype) == "str" else str(dtype)


def agrees(got: Any, want: Any) -> None:
    """The same columns, types, rows and row labels as pandas, frame or series."""
    assert list(got.index) == list(want.index)
    assert got.index.name == want.index.name
    if hasattr(want, "columns"):
        assert list(got.columns) == list(want.columns)
        for name in want.columns:
            assert got[name].dtype == printed(want[name].dtype), name
            assert same(got[name].tolist(), want[name].tolist()), name
    else:
        assert got.name == want.name
        assert got.dtype == printed(want.dtype)
        assert same(got.tolist(), want.tolist())


BUILDS: list[Callable[[Any], Any]] = [
    lambda m: m.concat([m.DataFrame(FIRST), m.DataFrame(SECOND)]),
    lambda m: m.concat([m.DataFrame(FIRST), m.DataFrame(SECOND)], ignore_index=True),
    lambda m: m.concat([m.DataFrame(FIRST), m.DataFrame(OTHER)]),
    lambda m: m.concat([m.DataFrame(FIRST), m.DataFrame(OTHER)], sort=True),
    lambda m: m.concat([m.DataFrame(FIRST), m.DataFrame({"a": [7]})], join="inner"),
    lambda m: m.concat([m.DataFrame(FIRST), None, m.DataFrame(SECOND)]),
    lambda m: m.concat((m.DataFrame(FIRST), m.DataFrame(FIRST).iloc[:0])),
    lambda m: m.concat([m.DataFrame({"a": [1]}), m.DataFrame({"a": [0.5]})]),
    lambda m: m.concat(
        [m.DataFrame({"a": [1]}).astype("int8"), m.DataFrame({"a": [2]}).astype("uint8")]
    ),
    lambda m: m.concat(
        [m.DataFrame({"a": [1]}).astype("int16"), m.DataFrame({"a": [2]}).astype("float32")]
    ),
    lambda m: m.concat(
        [m.DataFrame({"a": [1]}).astype("int32"), m.DataFrame({"a": [2]}).astype("float32")]
    ),
    lambda m: m.concat(
        [m.DataFrame({"a": [1]}).astype("uint64"), m.DataFrame({"a": [2]}).astype("int64")]
    ),
    lambda m: m.concat([m.DataFrame({"a": [True]}), m.DataFrame({"a": [2]})]),
    lambda m: m.concat([m.DataFrame({"a": [1.5]}).astype("float32"), m.DataFrame({"b": [1]})]),
    lambda m: m.concat([m.DataFrame(FIRST).set_index("b"), m.DataFrame(SECOND).set_index("b")]),
    lambda m: m.concat([m.Series([1, 2], name="s"), m.Series([3], name="s")]),
    lambda m: m.concat([m.Series([1, 2], name="s"), m.Series([3.5], name="t")]),
    lambda m: m.concat([m.Series(["p"]), m.Series(["q"])], ignore_index=True),
    lambda m: m.concat([m.DataFrame(FIRST), m.Series([9], name="a")]),
    lambda m: m.concat([m.Series([1, 2], name="a"), m.Series([3.5, 4.5], name="b")], axis=1),
    lambda m: m.concat([m.DataFrame(FIRST), m.DataFrame(OTHER)], axis=1),
    lambda m: m.concat([m.DataFrame(FIRST), m.DataFrame(OTHER)], axis="columns", join="inner"),
    lambda m: m.concat([m.DataFrame(FIRST).iloc[::-1], m.DataFrame(OTHER)], axis=1, sort=True),
    lambda m: m.concat([m.DataFrame(FIRST), m.DataFrame(FIRST)], verify_integrity=False),
]


@pytest.mark.parametrize("build", BUILDS)
def test_a_concat_is_pandas_concat(firepanda: ModuleType, build: Callable[[Any], Any]) -> None:
    """Columns, types, labels and rows, down the rows and across."""
    import pandas as pd

    agrees(build(firepanda), build(pd))


MISTAKES: list[Callable[[Any], Any]] = [
    lambda m: m.concat([]),
    lambda m: m.concat([None, None]),
    lambda m: m.concat([m.DataFrame(FIRST), 1]),
    lambda m: m.concat([m.DataFrame(FIRST)], join="left"),
    lambda m: m.concat([m.DataFrame(FIRST)], axis=2),
    lambda m: m.concat([m.DataFrame(FIRST), m.DataFrame(FIRST)], verify_integrity=True),
    lambda m: m.concat([m.DataFrame(FIRST), m.DataFrame(FIRST)], axis=1, verify_integrity=True),
]


@pytest.mark.parametrize("build", MISTAKES)
def test_a_mistake_is_pandas_mistake(firepanda: ModuleType, build: Callable[[Any], Any]) -> None:
    """The same class, or a subclass of it, and the same first line."""
    import pandas as pd

    with pytest.raises(Exception) as theirs:
        build(pd)
    with pytest.raises(Exception) as mine:
        build(firepanda)
    assert isinstance(mine.value, type(theirs.value))
    assert str(mine.value).split("\n")[0] == str(theirs.value).split("\n")[0]


REFUSED: list[Callable[[Any], Any]] = [
    lambda m: m.concat([m.DataFrame(FIRST), m.DataFrame(SECOND)], keys=["p", "q"]),
    lambda m: m.concat({"p": m.DataFrame(FIRST)}),
    lambda m: m.concat([m.DataFrame(FIRST)], names=["n"]),
    lambda m: m.concat([m.DataFrame({"a": [1]}), m.DataFrame({"a": ["x"]})]),
    lambda m: m.concat([m.DataFrame({"a": [True]}), m.DataFrame({"b": [1]})]),
    lambda m: m.concat([m.Series([True]), m.Series([1])]),
    lambda m: m.concat([m.DataFrame(FIRST), m.Series([1])]),
    lambda m: m.concat([m.DataFrame(FIRST), m.DataFrame(OTHER)], axis=1, ignore_index=True),
    lambda m: m.concat([m.DataFrame(FIRST), m.DataFrame(FIRST).set_index("b")]),
]


@pytest.mark.parametrize("build", REFUSED)
def test_what_pandas_answers_with_object_or_a_multiindex_is_refused(
    firepanda: ModuleType, build: Callable[[Any], Any]
) -> None:
    """Refused by name rather than answered with something pandas would not give."""
    with pytest.raises(NotImplementedError):
        build(firepanda)


def test_the_signature_is_pandas_signature(firepanda: ModuleType) -> None:
    """Parameter for parameter, with the same defaults but for the sentinels."""
    import pandas as pd

    mine = inspect.signature(firepanda.concat).parameters
    yours = inspect.signature(pd.concat).parameters
    assert list(mine) == list(yours)
    for name in mine:
        if name not in ("sort", "copy"):
            assert mine[name].default == yours[name].default, name
