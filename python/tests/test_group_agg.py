"""`groupby(...).agg` and `aggregate` by name, checked against pandas.

Every function is a name, and each output column is the method of that name run
on its own, so what these tests watch is the shape: which columns come back,
in what order, under what names, with the keys as labels or as columns. Every
test builds the same aggregation in both libraries and compares the columns,
the types, the labels and every row.
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
DATA = {
    "k": ["a", "b", "a", "c", "b", "a", None],
    "j": [1, 1, 2, 2, 1, 1, 2],
    "v": [1, 2, 3, 4, 5, 6, 7],
    "w": [1.5, NAN, 3.5, 4.0, 2.5, NAN, 9.0],
    "s": ["p", "q", "r", "s", "t", "u", "v"],
}


def frame(m: Any) -> Any:
    """Keys with a gap, whole numbers, floats with gaps and text."""
    return m.DataFrame(DATA)


def numbers(m: Any) -> Any:
    """The same frame without the text, for the reductions text has no answer to."""
    return m.DataFrame({name: DATA[name] for name in ("k", "j", "v", "w")})


def same(got: list[Any], want: list[Any]) -> bool:
    """Equal row by row, with NaN equal to NaN."""

    def missing(value: Any) -> bool:
        return value is None or value != value

    return len(got) == len(want) and all(
        (missing(a) and missing(b)) or a == b for a, b in zip(got, want, strict=True)
    )


def printed(dtype: Any) -> str:
    """A pandas type in the words firepanda uses for it."""
    return "string" if str(dtype) == "str" else str(dtype)


def agrees(got: Any, want: Any) -> None:
    """The same types, rows and row labels as pandas, frame or series."""
    assert same(list(got.index), list(want.index))
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
    lambda m: numbers(m).groupby("k").agg("sum"),
    lambda m: frame(m).groupby("k").aggregate("max"),
    lambda m: frame(m).groupby("k").agg("count"),
    lambda m: frame(m).groupby("k").agg("size"),
    lambda m: frame(m).groupby("k").agg("first"),
    lambda m: frame(m).groupby("k", dropna=False).agg("min"),
    lambda m: numbers(m).groupby("k", sort=False).agg("sum"),
    lambda m: numbers(m).groupby("k", as_index=False).agg("sum"),
    lambda m: numbers(m).groupby(["k", "j"], as_index=False).agg("sum"),
    lambda m: frame(m).groupby("k")["v"].agg("sum"),
    lambda m: frame(m).groupby("k")["w"].agg("mean"),
    lambda m: frame(m).groupby("k")["v"].agg(["sum", "mean", "count"]),
    lambda m: frame(m).groupby("k")["w"].agg(["min", "max", "std", "nunique"]),
    lambda m: frame(m).groupby("k")["s"].agg(("first", "last")),
    lambda m: frame(m).groupby("k", sort=False)["v"].agg(["sum", "max"]),
    lambda m: frame(m).groupby("k", as_index=False)["v"].agg(["sum", "mean"]),
    lambda m: frame(m).groupby("k")["v"].agg(total="sum", most="max"),
    lambda m: frame(m).groupby("k").agg({"w": "max", "v": "sum"}),
    lambda m: frame(m).groupby("k").agg({"s": "last"}),
    lambda m: frame(m).groupby("k", as_index=False).agg({"v": "sum", "w": "mean"}),
    lambda m: frame(m).groupby("k").agg(total=("v", "sum"), rows=("w", "count")),
    lambda m: frame(m).groupby("k").agg(low=("v", "min"), high=("v", "max")),
    lambda m: frame(m).groupby("k").agg(total=m.NamedAgg("v", "sum")),
    lambda m: frame(m).groupby("k", as_index=False).agg(t=("v", "sum"), u=("w", "mean")),
    lambda m: frame(m).groupby(["k", "j"], as_index=False).agg(t=("v", "sum")),
    lambda m: frame(m).groupby("k", dropna=False, sort=False).agg(t=("w", "sum")),
    lambda m: frame(m).groupby("j")[["v", "w"]].agg("mean"),
    lambda m: frame(m).groupby("k")["v"].agg([]),
]


@pytest.mark.parametrize("build", BUILDS)
def test_an_aggregation_is_pandas_aggregation(
    firepanda: ModuleType, build: Callable[[Any], Any]
) -> None:
    """A name, a list, a mapping and the named form, with the keys either way."""
    import pandas as pd

    agrees(build(firepanda), build(pd))


MISTAKES: list[Callable[[Any], Any]] = [
    lambda m: frame(m).groupby("k").agg(),
    lambda m: frame(m).groupby("k").agg(t="sum"),
    lambda m: frame(m).groupby("k").agg("nope"),
    lambda m: frame(m).groupby("k")["v"].agg("nope"),
    lambda m: frame(m).groupby("k").agg({"nope": "sum"}),
    lambda m: frame(m).groupby("k").agg(t=("nope", "sum")),
    lambda m: frame(m).groupby("k").agg(t=("v", "nope")),
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
    lambda m: frame(m).groupby("k")["v"].agg(lambda group: group.max()),
    lambda m: frame(m).groupby("k")["v"].agg(["sum", len]),
    lambda m: frame(m).groupby("k").agg(["min", "max"]),
    lambda m: frame(m).groupby("k").agg({"v": ["min", "max"]}),
    lambda m: frame(m).groupby("k")["v"].agg(["sum", "sum"]),
    lambda m: frame(m).groupby("k").agg("sum", engine="numba"),
    lambda m: frame(m).groupby("k")["v"].agg("sum", 1),
]


@pytest.mark.parametrize("build", REFUSED)
def test_what_is_not_written_is_refused(firepanda: ModuleType, build: Callable[[Any], Any]) -> None:
    """A function, two levels of column labels, a repeated name and the engine."""
    with pytest.raises(NotImplementedError):
        build(firepanda)


def test_agg_is_aggregate(firepanda: ModuleType) -> None:
    """One method under two names, as in pandas."""
    grouped = type(frame(firepanda).groupby("k"))
    assert grouped.agg is grouped.aggregate
    column = type(frame(firepanda).groupby("k")["v"])
    assert column.agg is column.aggregate


def test_a_mapping_for_one_column_is_a_specification_error(firepanda: ModuleType) -> None:
    """The class pandas names, a plain `Exception`, with pandas' words."""
    import pandas as pd

    with pytest.raises(pd.errors.SpecificationError, match="nested renamer"):
        frame(pd).groupby("k")["v"].agg({"x": "sum"})
    with pytest.raises(firepanda.errors.SpecificationError, match=r"^nested renamer is not"):
        frame(firepanda).groupby("k")["v"].agg({"x": "sum"})
    assert issubclass(firepanda.errors.SpecificationError, Exception)
    assert not issubclass(firepanda.errors.SpecificationError, ValueError)


def test_a_named_agg_is_pandas_named_agg(firepanda: ModuleType) -> None:
    """The same fields and the same printed form, and pandas' own one is read too."""
    import pandas as pd

    assert repr(firepanda.NamedAgg("v", "sum")) == repr(pd.NamedAgg("v", "sum"))
    agrees(
        frame(firepanda).groupby("k").agg(total=pd.NamedAgg("v", "sum")),
        frame(pd).groupby("k").agg(total=pd.NamedAgg("v", "sum")),
    )
    with pytest.raises(NotImplementedError):
        frame(firepanda).groupby("k").agg(total=firepanda.NamedAgg("v", "quantile", 0.5))


@pytest.mark.parametrize("owner", ["DataFrameGroupBy", "SeriesGroupBy"])
def test_the_signature_is_pandas_signature(firepanda: ModuleType, owner: str) -> None:
    """Parameter for parameter, with the same kinds and defaults."""
    import pandas as pd

    build = frame(firepanda).groupby("k"), frame(pd).groupby("k")
    if owner == "SeriesGroupBy":
        build = build[0]["v"], build[1]["v"]
    mine = inspect.signature(type(build[0]).aggregate).parameters
    yours = inspect.signature(type(build[1]).aggregate).parameters
    assert [(p.name, p.kind, p.default) for p in mine.values()] == [
        (p.name, p.kind, p.default) for p in yours.values()
    ]
