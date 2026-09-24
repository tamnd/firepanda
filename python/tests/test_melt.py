"""`DataFrame.melt` and `pandas.melt`, checked against pandas.

A melt repeats the id columns once for every value column, names the column
each row came from and stacks the values, so the tests cover which columns are
picked by default, the names of the two new columns, the types the stacked
values widen to, the row labels kept or dropped, and the mistakes pandas names.
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

NAN = float("nan")
FRAME = {
    "a": [1, 2, 3],
    "b": [1.5, NAN, 2.5],
    "c": ["x", "y", None],
    "d": [3, 4, 5],
    "e": [True, False, True],
}


def labelled(m: Any) -> Any:
    """The frame on labels that repeat, under a named axis."""
    return m.DataFrame({"i": ["p", "q", "p"], **FRAME}).set_index("i")


def same(got: list[Any], want: list[Any]) -> bool:
    """Equal row by row, with NaN and None equal to each other."""

    def missing(value: Any) -> bool:
        return value is None or value != value

    return len(got) == len(want) and all(
        (missing(a) and missing(b)) or a == b for a, b in zip(got, want, strict=True)
    )


def agrees(got: Any, want: Any) -> None:
    """The same columns, types, rows and row labels as pandas."""
    assert list(got.columns) == list(want.columns)
    assert list(got.index) == list(want.index)
    assert got.index.name == want.index.name
    for name in want.columns:
        printed = str(want[name].dtype)
        assert got[name].dtype == ("string" if printed == "str" else printed), name
        assert same(got[name].tolist(), want[name].tolist()), name


BUILDS: list[Callable[[Any], Any]] = [
    lambda m: m.DataFrame(FRAME).melt(id_vars="a", value_vars=["b", "d"]),
    lambda m: m.DataFrame(FRAME).melt(id_vars=["a", "c"], value_vars="b"),
    lambda m: m.DataFrame(FRAME).melt(id_vars=("a",), value_vars=("d", "a")),
    lambda m: m.DataFrame(FRAME)[["a", "d"]].melt(),
    lambda m: m.DataFrame(FRAME)[["b", "c"]].melt(id_vars="c"),
    lambda m: m.DataFrame(FRAME)[["c", "e"]].melt(id_vars=["e"], var_name="k", value_name="v"),
    lambda m: m.DataFrame(FRAME).melt(id_vars=["c"], value_vars=["c"]),
    lambda m: m.DataFrame(FRAME).melt(id_vars=["a", "c"], value_vars=[]),
    lambda m: m.DataFrame(FRAME)[["a", "d"]].melt(value_vars=[]),
    lambda m: m.DataFrame(FRAME).melt(id_vars="a", value_vars=["b"], col_level=0),
    lambda m: m.DataFrame(FRAME).head(0).melt(id_vars="a", value_vars=["d"]),
    lambda m: labelled(m).melt(id_vars="c", value_vars=["a", "d"]),
    lambda m: labelled(m).melt(id_vars="c", value_vars=["a", "d"], ignore_index=False),
    lambda m: labelled(m)[["b"]].melt(ignore_index=False),
    lambda m: m.melt(m.DataFrame(FRAME), id_vars=["a"], value_vars=["b", "d"]),
    lambda m: m.melt(m.DataFrame(FRAME)[["d", "b"]], value_name="amount"),
]


@pytest.mark.parametrize("build", BUILDS)
def test_a_melt_is_pandas_melt(firepanda: ModuleType, build: Callable[[Any], Any]) -> None:
    """Names, defaults, widened values, empties and row labels kept or dropped."""
    import pandas as pd

    agrees(build(firepanda), build(pd))


MISTAKES: list[Callable[[Any], Any]] = [
    lambda m: m.DataFrame(FRAME).melt(id_vars=["z"]),
    lambda m: m.DataFrame(FRAME).melt(id_vars="a", value_vars=["z", "y"]),
    lambda m: m.DataFrame(FRAME).melt(id_vars="a", value_name="a"),
    lambda m: m.DataFrame(FRAME).melt(value_name="d"),
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
    lambda m: m.DataFrame(FRAME).melt(id_vars="a", value_vars=["b", "c"]),
    lambda m: m.DataFrame(FRAME).melt(id_vars="a", value_vars=["d", "e"]),
    lambda m: m.DataFrame(FRAME).melt(id_vars="a", value_vars=["b"], var_name="a"),
    lambda m: m.DataFrame(FRAME).melt(id_vars="a", value_vars=["b"], col_level=1),
]


@pytest.mark.parametrize("build", REFUSED)
def test_what_is_not_written_is_refused(firepanda: ModuleType, build: Callable[[Any], Any]) -> None:
    """Values pandas holds in an object column, a repeated name and a second level."""
    with pytest.raises(NotImplementedError):
        build(firepanda)


def test_the_signatures_are_pandas_signatures(firepanda: ModuleType) -> None:
    """Parameter for parameter, with the same defaults, for the method and the function."""
    import pandas as pd

    for ours, yours in [(firepanda.DataFrame.melt, pd.DataFrame.melt), (firepanda.melt, pd.melt)]:
        mine = inspect.signature(ours).parameters
        theirs = inspect.signature(yours).parameters
        assert list(mine) == list(theirs)
        for each in mine:
            assert mine[each].default == theirs[each].default, each
