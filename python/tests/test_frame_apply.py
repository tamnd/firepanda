"""`apply`, `agg`, `transform`, `mode` and `transpose` on a frame, and `agg` and
`transform` on a column, checked against pandas.

pandas calls a function on each column, or on each row with `axis=1`, and puts
the answers together: values become a column and columns become a frame. A name
is the method of that name, a list answers a row per function and a dict picks
the functions for each column.
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
    """The same labels, values, types and names, or the same single value."""
    if not hasattr(want, "index"):
        assert got == want
        return
    assert list(got.index) == list(want.index)
    if hasattr(want, "columns"):
        assert list(got.columns) == list(want.columns)
        for name in want.columns:
            agrees(got[name], want[name])
        return
    assert plain(got.tolist()) == plain(want.tolist())
    assert str(got.dtype).replace("string", "str") == str(want.dtype)
    assert got.name == want.name


def numbers(m: ModuleType) -> Any:
    """Whole numbers and floats, with text row labels."""
    return m.DataFrame({"a": [1, 2, 3], "b": [4.0, 5.0, 6.0]}, index=["x", "y", "z"])


def whole(m: ModuleType) -> Any:
    """Whole numbers only, with a repeat for the mode."""
    return m.DataFrame({"a": [1, 1, 2], "b": [3, 4, 5]}, index=["x", "y", "z"])


BUILDS: list[Callable[[Any], Any]] = [
    lambda m: numbers(m).apply(sum),
    lambda m: numbers(m).apply(lambda c: c * 2),
    lambda m: numbers(m).apply(lambda c: c.max() - c.min()),
    lambda m: numbers(m).apply(lambda r: r["a"] + r["b"], axis=1),
    lambda m: numbers(m).apply(lambda r: r * 2, axis=1),
    lambda m: numbers(m).apply(lambda c, k: c.sum() + k, args=(1,)),
    lambda m: numbers(m).apply("sum"),
    lambda m: numbers(m).apply(["sum", "max"]),
    lambda m: numbers(m).agg("sum"),
    lambda m: numbers(m).agg(["sum", "min"]),
    lambda m: numbers(m).agg({"a": "sum", "b": "max"}),
    lambda m: numbers(m).agg({"a": ["sum", "min"], "b": "max"}),
    lambda m: numbers(m).aggregate(lambda c: c.sum()),
    lambda m: numbers(m).transform(lambda c: c * 2),
    lambda m: numbers(m).transform("abs"),
    lambda m: numbers(m)["a"].agg("sum"),
    lambda m: numbers(m)["a"].agg(["sum", "max"]),
    lambda m: numbers(m)["a"].agg({"total": "sum", "top": "max"}),
    lambda m: numbers(m)["a"].agg(lambda c: c.sum()),
    lambda m: numbers(m)["a"].transform(lambda v: v + 1),
    lambda m: numbers(m)["a"].transform("abs"),
    lambda m: whole(m).mode(),
    lambda m: m.DataFrame({"s": ["p", "q", "q"], "n": [1, 2, 3]}).mode(),
    lambda m: numbers(m).T,
    lambda m: whole(m).transpose(),
    lambda m: m.DataFrame({"f": [True, False]}, index=["p", "q"]).T,
]


@pytest.mark.parametrize("build", BUILDS)
def test_the_answer_is_pandas_answer(firepanda: ModuleType, build: Callable[[Any], Any]) -> None:
    """Functions on columns and rows, names, lists, dicts, modes and the transpose."""
    import pandas as pd

    agrees(build(firepanda), build(pd))


MISTAKES: list[Callable[[Any], Any]] = [
    lambda m: numbers(m).transform(lambda c: c.head(1)),
    lambda m: numbers(m)["a"].transform(lambda c: c.sum()),
    lambda m: numbers(m).agg("nope"),
    lambda m: numbers(m)["a"].agg("nope"),
    lambda m: numbers(m).apply(sum, axis=2),
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


def test_a_transpose_needs_text_row_labels(firepanda: ModuleType) -> None:
    """The row labels become column names, and a column here is named by text."""
    with pytest.raises(NotImplementedError, match="text labels"):
        firepanda.DataFrame({"a": [1, 2]}).transpose()


@pytest.mark.parametrize(
    ("owner", "name"),
    [
        ("DataFrame", "apply"),
        ("DataFrame", "agg"),
        ("DataFrame", "aggregate"),
        ("DataFrame", "transform"),
        ("DataFrame", "mode"),
        ("DataFrame", "transpose"),
        ("Series", "agg"),
        ("Series", "aggregate"),
        ("Series", "transform"),
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
