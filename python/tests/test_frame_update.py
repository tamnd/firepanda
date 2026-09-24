"""`update`, `isetitem`, `from_dict` and `from_records` on a frame, `filter` on a
column and `map` on an index, checked against pandas.

`update` changes the frame it is called on and answers None, so a build here
answers the frame after the call.
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


def plain(values: list[Any]) -> list[Any]:
    """The values with every spelling of missing as one word."""
    return ["missing" if value is None or value != value else value for value in values]


def agrees(got: Any, want: Any) -> None:
    """The same labels, values, types and names."""
    assert list(got.index) == list(want.index)
    assert got.index.name == want.index.name
    if hasattr(want, "columns"):
        assert list(got.columns) == list(want.columns)
        for name in want.columns:
            agrees(got[name], want[name])
        return
    assert plain(got.tolist()) == plain(want.tolist())
    assert str(got.dtype).replace("string", "str") == str(want.dtype)
    assert got.name == want.name


def updated(m: ModuleType, call: Callable[[Any], Any]) -> Any:
    """A frame after an in-place call, which answers None."""
    target = m.DataFrame({"a": [1.0, None, 3.0], "b": [4, 5, 6]}, index=["x", "y", "z"])
    assert call(target) is None
    return target


def gaps(m: ModuleType) -> Any:
    """The other side of an update, with a gap and a column the frame lacks."""
    return m.DataFrame({"a": [9.0, 8.0, None], "c": [1, 1, 1]}, index=["x", "y", "z"])


BUILDS: list[Callable[[Any], Any]] = [
    lambda m: updated(m, lambda f: f.update(gaps(m))),
    lambda m: updated(m, lambda f: f.update(gaps(m), overwrite=False)),
    lambda m: updated(m, lambda f: f.update(gaps(m), filter_func=lambda v: v > 2)),
    lambda m: updated(m, lambda f: f.update(m.DataFrame({"b": [7]}, index=["y"]))),
    lambda m: updated(m, lambda f: f.update(m.Series([7, 8, 9], index=["x", "y", "z"], name="b"))),
    lambda m: updated(m, lambda f: f.isetitem(1, [7, 8, 9])),
    lambda m: updated(m, lambda f: f.isetitem(-1, [7.5, 8.5, 9.5])),
    lambda m: m.DataFrame.from_dict({"a": [1, 2], "b": [3, 4]}),
    lambda m: m.DataFrame.from_dict({"p": [1, 2], "q": [3, 4]}, orient="index", columns=["x", "y"]),
    lambda m: m.DataFrame.from_dict({"p": {"x": 1}, "q": {"y": 2}}, orient="index"),
    lambda m: m.DataFrame.from_dict(
        {
            "index": ["p", "q"],
            "columns": ["x"],
            "data": [[1], [2]],
            "index_names": [None],
            "column_names": [None],
        },
        orient="tight",
    ),
    lambda m: m.DataFrame.from_records([(1, "a"), (2, "b")], columns=["n", "s"]),
    lambda m: m.DataFrame.from_records([{"n": 1, "s": "a"}, {"n": 2}]),
    lambda m: m.DataFrame.from_records([(1, "a"), (2, "b")], columns=["n", "s"], index="s"),
    lambda m: m.DataFrame.from_records(
        [(1, "a", 3), (2, "b", 4)], columns=["n", "s", "t"], exclude=["t"]
    ),
    lambda m: m.DataFrame.from_records([(1, "a")], columns=["n", "s"], index=["r"]),
    lambda m: m.DataFrame({"a": [1, 2]}).infer_objects(),
    lambda m: m.Series([1, 2]).infer_objects(),
    lambda m: m.Series([1, 2, 3], index=["ab", "b", "ca"]).filter(like="a"),
    lambda m: m.Series([1, 2, 3], index=["ab", "b", "ca"]).filter(regex="^b"),
    lambda m: m.Series([1, 2, 3], index=["ab", "b", "ca"]).filter(items=["ca", "zz", "b"]),
    lambda m: m.Series([1, 2], index=[10, 20]).filter(items=[20]),
]


@pytest.mark.parametrize("build", BUILDS)
def test_the_answer_is_pandas_answer(firepanda: ModuleType, build: Callable[[Any], Any]) -> None:
    """Each rule of update, positions, the three orients, records and filters."""
    import pandas as pd

    agrees(build(firepanda), build(pd))


@pytest.mark.parametrize(
    "build",
    [
        lambda m: m.Index([1, 2], name="k").map(lambda v: v * 2),
        lambda m: m.Index(["a", "b"]).map({"a": "z"}),
        lambda m: m.Index([1, 2]).ravel(),
    ],
)
def test_an_index_answer_is_pandas_answer(
    firepanda: ModuleType, build: Callable[[Any], Any]
) -> None:
    """The labels mapped, and the index itself for ravel."""
    import pandas as pd

    got, want = build(firepanda), build(pd)
    assert plain(got.tolist()) == plain(want.tolist())
    assert str(got.dtype).replace("string", "str") == str(want.dtype)
    assert got.name == want.name


MISTAKES: list[Callable[[Any], Any]] = [
    lambda m: updated(m, lambda f: f.update(gaps(m), errors="raise")),
    lambda m: updated(m, lambda f: f.update(gaps(m), join="outer")),
    lambda m: m.DataFrame.from_dict({"a": [1]}, columns=["x"]),
    lambda m: m.DataFrame.from_dict({"a": [1]}, orient="bad"),
    lambda m: m.Series([1]).filter(items=["a"], like="a"),
    lambda m: m.Series([1]).filter(),
    lambda m: updated(m, lambda f: f.isetitem(5, [1, 2, 3])),
]


@pytest.mark.parametrize("build", MISTAKES)
def test_a_mistake_is_pandas_mistake(firepanda: ModuleType, build: Callable[[Any], Any]) -> None:
    """The same class or a subclass of it, and the same message."""
    import pandas as pd

    with pytest.raises(Exception) as theirs:
        build(pd)
    with pytest.raises(Exception) as mine:
        build(firepanda)
    assert isinstance(mine.value, type(theirs.value))
    assert str(mine.value) == str(theirs.value)


def test_a_value_a_column_cannot_hold_is_refused(firepanda: ModuleType) -> None:
    """A fraction into whole numbers is a TypeError, as in pandas."""
    target = firepanda.DataFrame({"a": [1, 2]})
    with pytest.raises(TypeError):
        target.update(firepanda.DataFrame({"a": [1.5, None]}))


@pytest.mark.parametrize(
    ("owner", "name"),
    [
        ("DataFrame", "update"),
        ("DataFrame", "isetitem"),
        ("DataFrame", "from_dict"),
        ("DataFrame", "from_records"),
        ("DataFrame", "infer_objects"),
        ("Series", "infer_objects"),
        ("Series", "filter"),
        ("Index", "map"),
        ("Index", "ravel"),
    ],
)
def test_the_signature_is_pandas_signature(firepanda: ModuleType, owner: str, name: str) -> None:
    """Parameter for parameter, with the same defaults."""
    import pandas as pd

    ours = inspect.signature(getattr(getattr(firepanda, owner), name)).parameters
    yours = inspect.signature(getattr(getattr(pd, owner), name)).parameters
    assert [(p.name, p.kind, repr(p.default)) for p in ours.values()] == [
        (p.name, p.kind, repr(p.default)) for p in yours.values()
    ]
