"""The index members a column already has, checked against pandas.

Each goes through the index's labels as a column and back, so the rules are the
column's rules, and the answer keeps the index's class and name. The ones about
levels answer for the one level a flat index has.
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
    """The same answer, an index or a column by labels, values, type and name."""
    if getattr(want, "ndim", 0) == 0:
        assert got == want
        assert type(got) is type(want.item() if hasattr(want, "item") else want)
        return
    assert plain(got.tolist()) == plain(want.tolist())
    assert str(got.dtype).replace("string", "str") == str(want.dtype)
    assert got.name == want.name
    if hasattr(want, "index"):
        assert list(got.index) == list(want.index)
        assert got.index.name == want.index.name


def gaps(m: ModuleType) -> Any:
    """Named float labels out of order with a gap."""
    return m.Index([3.0, 1.0, None, 2.0], name="k")


BUILDS: list[Callable[[Any], Any]] = [
    lambda m: gaps(m).all(),
    lambda m: gaps(m).any(),
    lambda m: m.Index([0, 1]).all(),
    lambda m: m.Index([0, 0]).any(),
    lambda m: m.Index(["a", ""]).all(),
    lambda m: m.Index([1]).delete(0).all(),
    lambda m: m.Index([1]).delete(0).any(),
    lambda m: gaps(m).argmax(),
    lambda m: gaps(m).argmin(),
    lambda m: m.Index(["b", "c", "a"]).argmax(),
    lambda m: m.Index([5]).item(),
    lambda m: m.Index(["x"]).item(),
    lambda m: gaps(m).fillna(0.0),
    lambda m: gaps(m).where([True, False, True, True]),
    lambda m: gaps(m).where([True, False, True, False], 9.0),
    lambda m: gaps(m).diff(),
    lambda m: m.Index([1, 4, 9]).diff(2),
    lambda m: m.Index([1.26, 2.5], name="r").round(1),
    lambda m: m.Index(["a", "b", "a"], name="k").value_counts(),
    lambda m: m.Index([1, 2, 2]).value_counts(normalize=True),
    lambda m: gaps(m).T,
    lambda m: gaps(m).transpose(),
    lambda m: m.Index([1, 2], name="k").astype("float64"),
    lambda m: m.Index([1, 2], name="k").astype("str"),
    lambda m: gaps(m).droplevel([]),
    lambda m: gaps(m).get_level_values(0),
    lambda m: gaps(m).get_level_values("k"),
    lambda m: gaps(m).to_flat_index(),
    lambda m: gaps(m).infer_objects(),
]


@pytest.mark.parametrize("build", BUILDS)
def test_the_answer_is_pandas_answer(firepanda: ModuleType, build: Callable[[Any], Any]) -> None:
    """Truth, extremes, one label, fills, differences, counts, types and levels."""
    import pandas as pd

    agrees(build(firepanda), build(pd))


def test_a_datetime_index_keeps_its_class(firepanda: ModuleType) -> None:
    """Filling and choosing labels answer an index of instants."""
    stamps = firepanda.to_datetime(firepanda.Series(["2024-01-01", None, "2024-01-03"]))
    index = firepanda.DatetimeIndex(stamps, name="t")
    filled = index.fillna(firepanda.Timestamp("2024-01-02"))
    assert isinstance(filled, firepanda.DatetimeIndex)
    assert filled.name == "t"
    assert [str(value) for value in filled.tolist()] == [
        "2024-01-01 00:00:00",
        "2024-01-02 00:00:00",
        "2024-01-03 00:00:00",
    ]


def test_memory_usage_is_the_column_of_labels(firepanda: ModuleType) -> None:
    """The number the labels weigh as a column, which counts the missing value bits."""
    index = firepanda.Index([1, 2, 3])
    assert index.memory_usage() == index.to_series().memory_usage(index=False)


@pytest.mark.parametrize("value", ["2024-01-02", "Timestamp"])
def test_a_column_of_instants_fills_with_an_instant(firepanda: ModuleType, value: str) -> None:
    """An instant with no zone, or text naming one, keeps the column's type, as in pandas."""
    import pandas as pd

    def build(m: ModuleType) -> Any:
        fill = m.Timestamp("2024-01-02") if value == "Timestamp" else value
        return m.to_datetime(m.Series(["2024-01-01", None])).fillna(fill)

    agrees(build(firepanda), build(pd))


MISTAKES: list[Callable[[Any], Any]] = [
    lambda m: gaps(m).argmin(skipna=False),
    lambda m: gaps(m).argmax(axis=1),
    lambda m: gaps(m).item(),
    lambda m: gaps(m).droplevel(),
    lambda m: gaps(m).droplevel([0, 0]),
    lambda m: gaps(m).get_level_values(1),
    lambda m: gaps(m).get_level_values("z"),
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


NAMES = [
    "all",
    "any",
    "argmax",
    "argmin",
    "item",
    "fillna",
    "where",
    "diff",
    "round",
    "value_counts",
    "transpose",
    "memory_usage",
    "astype",
    "droplevel",
    "get_level_values",
    "to_flat_index",
    "infer_objects",
]


@pytest.mark.parametrize("name", NAMES)
def test_the_signature_is_pandas_signature(firepanda: ModuleType, name: str) -> None:
    """Parameter for parameter, with the same defaults."""
    import pandas as pd

    ours = inspect.signature(getattr(firepanda.Index, name)).parameters
    yours = inspect.signature(getattr(pd.Index, name)).parameters
    assert [(p.name, p.kind, p.default) for p in ours.values()] == [
        (p.name, p.kind, p.default) for p in yours.values()
    ]


def test_t_is_a_property(firepanda: ModuleType) -> None:
    """As in pandas, so `index.T` is the index and `index.T()` is a mistake."""
    assert isinstance(inspect.getattr_static(firepanda.Index, "T"), property)
