"""The other names pandas has for a method, and the small helpers, checked against pandas.

`div`, `multiply`, `isnull` and the rest are a second name for a method this
library already has, and `pipe`, `pop`, `insert`, `equals`, `to_dict` and the
rest are a few lines over methods it already has. Every test runs the same call
in both libraries and compares what comes back.
"""

from __future__ import annotations

import collections
import importlib.util
import inspect
from collections.abc import Callable
from types import ModuleType
from typing import Any

import pytest

pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("pandas") is None, reason="pandas is not installed"
)

NAN = float("nan")
FRAME = {"a": [1, 2, 3], "b": [NAN, 4.0, 5.5], "s": ["x", None, "z"]}
LABELS = [5, 6, 7]


def frame(m: ModuleType) -> Any:
    """The frame, with row labels that are not the positions."""
    return m.DataFrame(FRAME, index=LABELS)


def column(m: ModuleType) -> Any:
    """A float column with a gap, labelled with text."""
    return m.Series([NAN, 2.0, 3.5, NAN], index=["p", "q", "r", "s"], name="v")


def plain(value: Any) -> Any:
    """A value with a missing value made comparable and numpy's scalars made Python's.

    pandas spells a missing text as NaN and this library as None, and both are missing.
    """
    if hasattr(value, "item") and not hasattr(value, "index"):
        value = value.item()
    if value is None or (isinstance(value, float) and value != value):
        return "missing"
    if isinstance(value, dict):
        return {plain(key): plain(each) for key, each in value.items()}
    if isinstance(value, list):
        return [plain(each) for each in value]
    return value


def agrees(got: Any, want: Any) -> None:
    """The same answer, whether a frame, a column or a plain value."""
    if hasattr(want, "columns"):
        assert list(got.columns) == list(want.columns)
        assert list(got.index) == list(want.index)
        for name in want.columns:
            assert plain(got[name].tolist()) == plain(want[name].tolist()), name
    elif hasattr(want, "tolist") and hasattr(want, "name"):
        assert list(got.index) == list(want.index)
        assert got.name == want.name
        assert plain(got.tolist()) == plain(want.tolist())
    else:
        assert plain(got) == plain(want)


BUILDS: list[Callable[[Any], Any]] = [
    lambda m: frame(m)[["a", "b"]].div(2),
    lambda m: frame(m)[["a", "b"]].divide(2),
    lambda m: frame(m)[["a", "b"]].rdiv(2),
    lambda m: frame(m)[["a", "b"]].multiply(3),
    lambda m: frame(m)[["a", "b"]].subtract(1),
    lambda m: frame(m).isnull(),
    lambda m: frame(m).notnull(),
    lambda m: column(m).div(2),
    lambda m: column(m).divide(column(m)),
    lambda m: column(m).rdiv(1),
    lambda m: column(m).multiply(2),
    lambda m: column(m).subtract(1),
    lambda m: column(m).isnull(),
    lambda m: column(m).notnull(),
    lambda m: frame(m).add_prefix("col_"),
    lambda m: frame(m).add_suffix("_col"),
    lambda m: frame(m).add_prefix("row_", axis=0),
    lambda m: frame(m).add_suffix("_row", axis="index"),
    lambda m: column(m).add_prefix("x"),
    lambda m: column(m).add_suffix("x"),
    lambda m: frame(m).pipe(lambda d, n: d.head(n), 2),
    lambda m: frame(m).pipe((lambda n, data: data.head(n), "data"), 1),
    lambda m: column(m).pipe(lambda s, k=1: s * k, k=3),
    lambda m: frame(m).first_valid_index(),
    lambda m: frame(m).last_valid_index(),
    lambda m: frame(m)[["b"]].first_valid_index(),
    lambda m: column(m).first_valid_index(),
    lambda m: column(m).last_valid_index(),
    lambda m: m.Series([NAN, NAN]).first_valid_index(),
    lambda m: m.Series([], dtype="float64").last_valid_index(),
    lambda m: frame(m).equals(frame(m)),
    lambda m: frame(m).equals(frame(m).head(2)),
    lambda m: frame(m).equals(frame(m).rename(columns={"a": "z"})),
    lambda m: frame(m).equals(frame(m).astype({"a": "float64"})),
    lambda m: frame(m).equals(m.DataFrame(FRAME)),
    lambda m: frame(m).equals(1),
    lambda m: column(m).equals(column(m).rename("other")),
    lambda m: column(m).equals(column(m).fillna(0)),
    lambda m: column(m).equals(frame(m)),
    lambda m: m.Series([1, 2]).equals(m.Series([1, 2], index=[1, 2])),
    lambda m: frame(m)[["a", "b"]].to_dict(),
    lambda m: frame(m)[["a", "b"]].to_dict("list"),
    lambda m: frame(m)[["a", "b"]].to_dict("split"),
    lambda m: frame(m)[["a", "b"]].to_dict("split", index=False),
    lambda m: frame(m)[["a", "b"]].to_dict("tight"),
    lambda m: frame(m)[["a", "b"]].to_dict("records"),
    lambda m: frame(m)[["a", "b"]].to_dict("index"),
    lambda m: frame(m).to_dict("list"),
    lambda m: frame(m)[["a"]].to_dict(into=collections.OrderedDict),
    lambda m: column(m).to_dict(),
    lambda m: m.Series(["x", None]).to_dict(),
    lambda m: column(m).to_list(),
    lambda m: column(m).is_unique,
    lambda m: m.Series([1, 2, 3]).is_unique,
    lambda m: m.Series(["a", None, None]).is_unique,
    lambda m: m.Series([7.5]).item(),
    lambda m: m.Series(["only"]).item(),
    lambda m: m.isna(None),
    lambda m: m.isna(NAN),
    lambda m: m.isna("x"),
    lambda m: m.isna(3),
    lambda m: m.notna(NAN),
    lambda m: m.isnull(None),
    lambda m: m.notnull(1.5),
    lambda m: m.isna(column(m)),
    lambda m: m.notna(frame(m)),
]


@pytest.mark.parametrize("build", BUILDS)
def test_the_answer_is_pandas_answer(firepanda: ModuleType, build: Callable[[Any], Any]) -> None:
    """Each alias and helper, on a frame, a column or a scalar."""
    import pandas as pd

    agrees(build(firepanda), build(pd))


def test_pop_takes_the_column_out(firepanda: ModuleType) -> None:
    """The column comes back and is gone from the frame."""
    import pandas as pd

    mine, theirs = frame(firepanda), frame(pd)
    agrees(mine.pop("b"), theirs.pop("b"))
    agrees(mine, theirs)
    mine, theirs = column(firepanda), column(pd)
    agrees(mine.pop("q"), theirs.pop("q"))
    agrees(mine, theirs)


@pytest.mark.parametrize(
    ("loc", "value"), [(0, 9), (1, [7, 8, 9]), (3, "t"), (2, "series"), (-0, 1.5)]
)
def test_insert_puts_the_column_in_place(firepanda: ModuleType, loc: int, value: Any) -> None:
    """A scalar, a list or a column, at the front, the middle or the end."""
    import pandas as pd

    mine, theirs = frame(firepanda), frame(pd)
    if value == "series":
        assert mine.insert(loc, "new", mine["a"] * 2) is None
        theirs.insert(loc, "new", theirs["a"] * 2)
    else:
        assert mine.insert(loc, "new", value) is None
        theirs.insert(loc, "new", value)
    agrees(mine, theirs)


MISTAKES: list[Callable[[Any], Any]] = [
    lambda m: frame(m).insert(5, "c", 1),
    lambda m: frame(m).insert(1.0, "c", 1),
    lambda m: frame(m).insert(-1, "c", 1),
    lambda m: frame(m).insert(0, "a", 1),
    lambda m: frame(m).insert(0, "c", [1, 2]),
    lambda m: frame(m).pop("nope"),
    lambda m: m.Series([1, 2]).item(),
    lambda m: m.Series([], dtype="float64").item(),
    lambda m: frame(m).pipe((lambda data: data, "data"), data=1),
    lambda m: frame(m).add_prefix("p", axis=2),
    lambda m: column(m).add_prefix("p", axis=1),
    lambda m: frame(m).to_dict("bogus"),
    lambda m: frame(m).to_dict("dict", index=False),
    lambda m: frame(m).to_dict(into=list),
    lambda m: frame(m).to_dict(into=collections.defaultdict),
]


@pytest.mark.parametrize("build", MISTAKES)
def test_a_mistake_is_pandas_mistake(firepanda: ModuleType, build: Callable[[Any], Any]) -> None:
    """The same class, or a subclass of it, and the same first line."""
    import pandas as pd

    with pytest.raises(Exception) as theirs:
        build(pd)
    with pytest.raises(Exception) as mine:
        build(firepanda)
    assert isinstance(mine.value, type(theirs.value)), (mine.value, theirs.value)
    assert str(mine.value).split("\n")[0] == str(theirs.value).split("\n")[0]


def test_what_is_not_written_is_refused(firepanda: ModuleType) -> None:
    """A list for isna, which pandas answers with numpy, and a duplicate column."""
    with pytest.raises(NotImplementedError):
        firepanda.isna([1, None])
    with pytest.raises(NotImplementedError):
        frame(firepanda).insert(0, "a", 1, allow_duplicates=True)


FRAME_METHODS = [
    "div",
    "divide",
    "rdiv",
    "multiply",
    "subtract",
    "isnull",
    "notnull",
    "add_prefix",
    "add_suffix",
    "pipe",
    "pop",
    "insert",
    "equals",
    "first_valid_index",
    "last_valid_index",
    "to_dict",
]
SERIES_METHODS = [
    *[name for name in FRAME_METHODS if name != "insert"],
    "to_list",
    "item",
]
METHODS = [("DataFrame", name) for name in FRAME_METHODS] + [
    ("Series", name) for name in SERIES_METHODS
]


@pytest.mark.parametrize(("owner", "name"), METHODS)
def test_the_signature_is_pandas_signature(firepanda: ModuleType, owner: str, name: str) -> None:
    """Parameter for parameter, with the same defaults."""
    import pandas as pd

    ours = inspect.signature(getattr(getattr(firepanda, owner), name)).parameters
    yours = inspect.signature(getattr(getattr(pd, owner), name)).parameters
    assert [(p.name, p.kind) for p in ours.values()] == [(p.name, p.kind) for p in yours.values()]
    for each in ours:
        mine, theirs = ours[each].default, yours[each].default
        if repr(theirs) == "<no_default>":
            assert repr(mine) == "<no_default>", each
        else:
            assert mine == theirs, each


@pytest.mark.parametrize("name", ["isna", "isnull", "notna", "notnull"])
def test_the_module_function_is_there(firepanda: ModuleType, name: str) -> None:
    """The four names, each taking one object as pandas' do."""
    import pandas as pd

    ours = inspect.signature(getattr(firepanda, name)).parameters
    yours = inspect.signature(getattr(pd, name)).parameters
    assert list(ours) == list(yours)
