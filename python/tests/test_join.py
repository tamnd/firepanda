"""`DataFrame.join` and `merge` on the row labels, checked against pandas.

A side keyed on its row labels has them put in a column, the merge is the one
on columns, and what comes back is relabelled the way pandas labels it: by the
key when both sides are on their labels, and by the other side's labels when
only one is, where a row that side does not have loses its label and an integer
label becomes float64. Every test builds the same join in both libraries and
compares the columns, the types, the row labels and their name, and every row.
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

LEFT = {"k": ["b", "a", "b", "c"], "x": [1, 2, 3, 4]}
RIGHT = {"k": ["a", "b", "d"], "y": [10, 20, 30]}
NUMBERS = {"n": [3, 1, 2], "x": [1, 2, 3]}
OTHERS = {"n": [1, 0, 3], "y": [5, 6, 7]}


def same(got: list[Any], want: list[Any]) -> bool:
    """Equal row by row, with NaN equal to NaN and None equal to NaN."""

    def missing(value: Any) -> bool:
        return value is None or value != value

    return len(got) == len(want) and all(
        (missing(a) and missing(b)) or a == b for a, b in zip(got, want, strict=True)
    )


def agrees(got: Any, want: Any) -> None:
    """The same columns, types, rows, row labels and label name as pandas."""
    assert list(got.columns) == list(want.columns)
    assert same(list(got.index), list(want.index))
    assert got.index.name == want.index.name
    assert got.index.dtype == ("string" if str(want.index.dtype) == "str" else want.index.dtype)
    for name in want.columns:
        printed = str(want[name].dtype)
        assert got[name].dtype == ("string" if printed == "str" else printed), name
        assert same(got[name].tolist(), want[name].tolist()), name


def keyed(m: Any, data: dict[str, list[Any]], name: str = "k") -> Any:
    """A frame labelled by one of its columns."""
    return m.DataFrame(data).set_index(name)


HOWS = ["inner", "left", "right", "outer"]

BUILDS: list[Callable[[Any, str], Any]] = [
    lambda m, how: keyed(m, LEFT).join(keyed(m, RIGHT), how=how),
    lambda m, how: m.merge(
        keyed(m, LEFT), keyed(m, RIGHT), left_index=True, right_index=True, how=how
    ),
    lambda m, how: keyed(m, LEFT).join(keyed(m, RIGHT).rename_axis("j"), how=how),
    lambda m, how: keyed(m, LEFT).rename_axis(None).join(keyed(m, RIGHT), how=how),
    lambda m, how: m.DataFrame(LEFT).join(keyed(m, RIGHT), on="k", how=how),
    lambda m, how: m.merge(
        m.DataFrame(LEFT).rename_axis("n"), keyed(m, RIGHT), left_on="k", right_index=True, how=how
    ),
    lambda m, how: m.merge(
        keyed(m, LEFT), m.DataFrame(RIGHT), left_index=True, right_on="k", how=how
    ),
    lambda m, how: keyed(m, NUMBERS, "n").join(keyed(m, OTHERS, "n"), how=how),
    lambda m, how: m.merge(
        m.DataFrame(NUMBERS), keyed(m, OTHERS, "n"), left_on="n", right_index=True, how=how
    ),
    lambda m, how: keyed(m, LEFT).join(keyed(m, LEFT), how=how, lsuffix="_l", rsuffix="_r"),
    lambda m, how: m.merge(
        keyed(m, LEFT), keyed(m, LEFT), left_index=True, right_index=True, how=how
    ),
    lambda m, how: keyed(m, LEFT).join(keyed(m, RIGHT)["y"], how=how),
    lambda m, how: keyed(m, LEFT).join([keyed(m, RIGHT)], how=how, sort=True),
]


@pytest.mark.parametrize("how", HOWS)
@pytest.mark.parametrize("build", BUILDS)
def test_a_join_is_pandas_join(
    firepanda: ModuleType, build: Callable[[Any, str], Any], how: str
) -> None:
    """Every join on the labels, one side or both, in each of the four hows."""
    import pandas as pd

    agrees(build(firepanda, how), build(pd, how))


MISTAKES: list[Callable[[Any], Any]] = [
    lambda m: keyed(m, LEFT).join(keyed(m, LEFT)),
    lambda m: keyed(m, LEFT).join(m.Series([1, 2])),
    lambda m: keyed(m, LEFT).join(keyed(m, RIGHT), how="bad"),
    lambda m: m.DataFrame(LEFT).join(keyed(m, RIGHT), on="zz"),
    lambda m: keyed(m, LEFT).join(keyed(m, RIGHT), validate="1:1"),
    lambda m: m.merge(keyed(m, LEFT), keyed(m, RIGHT), left_index=True, right_index=True, on="k"),
    lambda m: m.merge(keyed(m, LEFT), keyed(m, RIGHT), left_index=True),
    lambda m: m.merge(keyed(m, LEFT), keyed(m, RIGHT), right_index=True),
    lambda m: m.merge(
        m.DataFrame(LEFT), keyed(m, RIGHT), left_index=True, left_on="k", right_index=True
    ),
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


def test_what_is_not_written_is_refused(firepanda: ModuleType) -> None:
    """A cross join, a list of frames, and text labels that go missing."""
    left, right = keyed(firepanda, LEFT), keyed(firepanda, RIGHT)
    for build in (
        lambda: left.join(right, how="cross"),
        lambda: left.join([right, right]),
        lambda: keyed(firepanda, {"t": ["p", "q", "r", "s"], **LEFT}, "t").join(
            right, on="k", how="outer"
        ),
    ):
        with pytest.raises(NotImplementedError):
            build()


def test_the_method_has_the_pandas_signature(firepanda: ModuleType) -> None:
    """Parameter for parameter, with the same defaults."""
    import pandas as pd

    mine = inspect.signature(firepanda.DataFrame.join).parameters
    yours = inspect.signature(pd.DataFrame.join).parameters
    assert list(mine) == list(yours)
    for name in mine:
        assert mine[name].default == yours[name].default, name
