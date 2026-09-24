"""`DataFrame.resample`, `Series.resample` and the resampler, checked against pandas.

A resample is a group by on the bin every timestamp falls in, with every bin
from the first to the last in the answer, empty or not. Every test builds the
same answer in both libraries and compares the labels, the types and the rows,
since what an empty bin holds, and whether an integer column widens to float
because of it, is where a resample goes wrong without raising.
"""

from __future__ import annotations

import datetime
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
TIMES = [
    "2026-01-01 00:10:00",
    "2026-01-01 01:20:00",
    "2026-01-01 04:00:00",
    "2026-01-01 04:30:00",
    "2026-01-01 06:00:00",
    "2026-01-02 09:59:59",
]
EXTREMES = [
    datetime.datetime(1677, 9, 22),
    datetime.datetime(1900, 1, 1),
    datetime.datetime(1969, 12, 31, 23, 59, 59),
    datetime.datetime(1970, 1, 1),
    datetime.datetime(2262, 4, 11),
    datetime.datetime(2300, 1, 1),
]


def frame(m: Any) -> Any:
    """Six rows on a datetime index, with gaps between them and a missing float."""
    return m.DataFrame(
        {
            "t": m.to_datetime(m.Series(TIMES)),
            "a": [1, 2, 3, 4, 5, 6],
            "x": [1.5, NAN, 2.0, 3.0, -1.0, 0.25],
            "s": ["p", "q", "r", "s", "t", "u"],
        }
    ).set_index("t")


def numbers(m: Any) -> Any:
    """The frame without its text column."""
    return frame(m)[["a", "x"]]


def extremes(m: Any) -> Any:
    """Six rows in seconds from 1677 to 2300, the compat suite's temporal_range frame."""
    return m.DataFrame(
        {"row": [0, 1, 2, 3, 4, 5], "second": m.Series(EXTREMES).dt.as_unit("s")}
    ).set_index("second")["row"]


def same(got: list[Any], want: list[Any]) -> bool:
    """Equal row by row to a relative 1e-9, with NaN equal to NaN."""

    def missing(value: Any) -> bool:
        return value is None or value != value

    def equal(a: Any, b: Any) -> bool:
        if isinstance(a, float) or isinstance(b, float):
            return a == b or abs(a - b) <= 1e-9 * max(abs(a), abs(b))
        return a == b

    return len(got) == len(want) and all(
        (missing(a) and missing(b)) or equal(a, b) for a, b in zip(got, want, strict=True)
    )


def printed(dtype: Any) -> str:
    """A type the way firepanda prints it."""
    text = str(dtype)
    return "string" if text == "str" else text


def agrees(got: Any, want: Any) -> None:
    """The same labels, index name, types and rows as pandas, frame or series."""
    assert [str(x) for x in got.index.tolist()] == [str(x) for x in want.index.tolist()]
    assert got.index.name == want.index.name
    assert got.index.dtype == str(want.index.dtype)
    if hasattr(want, "columns"):
        assert list(got.columns) == list(want.columns)
        for name in want.columns:
            assert got[name].dtype == printed(want[name].dtype), name
            assert same(got[name].tolist(), want[name].tolist()), name
    else:
        assert got.name == want.name
        assert got.dtype == printed(want.dtype)
        assert same(got.tolist(), want.tolist())


REDUCTIONS = ["sum", "prod", "mean", "median", "min", "max", "first", "last", "std", "var"]
REDUCTIONS += ["sem", "count", "nunique", "quantile"]


@pytest.mark.parametrize("how", REDUCTIONS)
@pytest.mark.parametrize("rule", ["h", "90min", "1.5h", "D", "2D", "30s"])
def test_a_reduction_is_pandas_reduction(firepanda: ModuleType, rule: str, how: str) -> None:
    """Every reduction over steps from seconds to days, with empty bins between."""
    import pandas as pd

    agrees(
        getattr(numbers(firepanda).resample(rule), how)(),
        getattr(numbers(pd).resample(rule), how)(),
    )


BUILDS: list[Callable[[Any], Any]] = [
    lambda m: numbers(m).resample("h", closed="right").sum(),
    lambda m: numbers(m).resample("h", label="right").mean(),
    lambda m: numbers(m).resample("2h", closed="right", label="right").max(),
    lambda m: numbers(m).resample("h", closed="left", label="left").count(),
    lambda m: numbers(m).resample("h", origin="start").sum(),
    lambda m: numbers(m).resample("7min", origin="epoch").mean(),
    lambda m: numbers(m).resample("7min", origin="start_day").mean(),
    lambda m: numbers(m).resample("7min", origin="start").mean(),
    lambda m: numbers(m).resample("D", origin="start").sum(),
    lambda m: numbers(m).resample("h").sum(min_count=1),
    lambda m: numbers(m).resample("h").prod(min_count=1),
    lambda m: numbers(m).resample("h").std(ddof=0),
    lambda m: numbers(m).resample("h").quantile(0.25),
    lambda m: numbers(m).resample("h").size(),
    lambda m: frame(m).resample("h").count(),
    lambda m: frame(m).resample("h").first(),
    lambda m: frame(m).resample("h").nunique(),
    lambda m: frame(m).resample("h").sum(numeric_only=True),
    lambda m: frame(m).resample("h").mean(numeric_only=True),
    lambda m: frame(m)["a"].resample("h").sum(),
    lambda m: frame(m)["x"].resample("h").mean(),
    lambda m: frame(m)["a"].resample("h").ohlc(),
    lambda m: frame(m)["a"].resample("2h", closed="right", label="right").ohlc(),
    lambda m: frame(m)["x"].resample("30min").ohlc(),
    lambda m: frame(m)["a"].resample("h").size(),
    lambda m: frame(m).resample("h")["a"].sum(),
    lambda m: frame(m).resample("h")[["a", "x"]].max(),
    lambda m: frame(m).resample("h").x.mean(),
    lambda m: numbers(m).reset_index().resample("h", on="t").sum(),
    lambda m: frame(m).reset_index().resample("h", on="t")["x"].min(),
    lambda m: numbers(m).resample("h").agg("sum"),
    lambda m: numbers(m).resample("h").aggregate("mean"),
    lambda m: numbers(m).resample("h").agg({"a": "sum", "x": "max"}),
    lambda m: numbers(m).resample("h").agg({"x": "count"}),
    lambda m: numbers(m).resample("h").apply("sum"),
    lambda m: numbers(m).resample("h").transform("sum"),
    lambda m: frame(m)["x"].resample("2h").transform("mean"),
    lambda m: numbers(m).resample("h").pipe(lambda r: r.sum()),
    lambda m: numbers(m).resample("h").asfreq(),
    lambda m: numbers(m).resample("h").asfreq(fill_value=0),
    lambda m: numbers(m).resample("h").get_group("2026-01-01 04:00"),
    lambda m: frame(m)["a"].resample("h").get_group(m.Timestamp("2026-01-01 00:00")),
    lambda m: numbers(m).astype({"a": "int32"}).resample("h").max(),
    lambda m: numbers(m).astype({"a": "float32"}).resample("h").mean(),
    lambda m: numbers(m).astype({"a": "uint8"}).resample("D").max(),
    lambda m: numbers(m).head(0).resample("h").sum(),
    lambda m: numbers(m).head(1).resample("h").mean(),
    lambda m: (
        frame(m)
        .reset_index()
        .assign(t=lambda f: f.t.dt.as_unit("s"))
        .set_index("t")
        .resample("h")
        .sum(numeric_only=True)
    ),
    lambda m: (
        frame(m)
        .head(2)
        .reset_index()
        .assign(t=lambda f: f.t.dt.as_unit("ns"))
        .set_index("t")
        .resample("250ms")
        .count()
    ),
    lambda m: extremes(m).resample("D").sum(),
    lambda m: extremes(m).resample("D").count(),
]


@pytest.mark.parametrize("build", BUILDS)
def test_a_resample_is_pandas_resample(firepanda: ModuleType, build: Callable[[Any], Any]) -> None:
    """Edges, labels, origins, columns, times in a column, widths and units."""
    import pandas as pd

    agrees(build(firepanda), build(pd))


def test_the_bins_are_pandas_bins(firepanda: ModuleType) -> None:
    """How many bins, the rows in each, the edges and the timestamps binned."""
    import pandas as pd

    mine = numbers(firepanda).resample("h")
    theirs = numbers(pd).resample("h")
    assert mine.ngroups == theirs.ngroups
    assert mine.ndim == theirs.ndim == 2
    assert frame(firepanda)["a"].resample("h").ndim == 1
    assert {str(k): int(v) for k, v in mine.groups.items()} == {
        str(k): int(v) for k, v in theirs.groups.items()
    }
    assert {str(k): v for k, v in mine.indices.items()} == {
        str(k): [int(i) for i in v] for k, v in theirs.indices.items()
    }
    assert [str(x) for x in mine.binner.tolist()] == [str(x) for x in theirs.binner.tolist()]
    assert [str(x) for x in mine.ax.tolist()] == [str(x) for x in theirs.ax.tolist()]
    assert list(mine.obj.columns) == ["a", "x"]


def test_a_missing_timestamp_is_in_no_bin(firepanda: ModuleType) -> None:
    """The row is left out, as pandas leaves it out."""
    import pandas as pd

    def build(m: Any) -> Any:
        times = m.to_datetime(m.Series(["2026-01-01 00:10:00", None, "2026-01-01 02:00:00"]))
        return m.DataFrame({"t": times, "a": [1, 2, 3]}).set_index("t").resample("h").sum()

    agrees(build(firepanda), build(pd))


MISTAKES: list[Callable[[Any], Any]] = [
    lambda m: numbers(m).resample("h", closed="middle"),
    lambda m: numbers(m).resample("h", label="x"),
    lambda m: numbers(m).resample("h", origin="nope"),
    lambda m: numbers(m).resample("0h"),
    lambda m: numbers(m).resample("H"),
    lambda m: numbers(m).resample("nope"),
    lambda m: numbers(m).reset_index().resample("h"),
    lambda m: numbers(m).reset_index(drop=True).set_index("x").resample("h"),
    lambda m: numbers(m).resample("h").get_group("2026-01-01 02:00"),
    lambda m: numbers(m).resample("h").nope,
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
    assert (
        str(mine.value).split("\n")[0].split(".")[0]
        == str(theirs.value).split("\n")[0].split(".")[0]
    )


REFUSED: list[Callable[[Any], Any]] = [
    lambda m: numbers(m).resample("W").sum(),
    lambda m: numbers(m).resample("ME").sum(),
    lambda m: numbers(m).resample("h", offset="5min").sum(),
    lambda m: numbers(m).resample("h", convention="end").sum(),
    lambda m: numbers(m).resample("h", group_keys=True).sum(),
    lambda m: numbers(m).resample("h", origin="end").sum(),
    lambda m: numbers(m).resample("ns").sum(),
    lambda m: numbers(m).resample("h").ohlc(),
    lambda m: numbers(m).resample("h").agg(["sum", "max"]),
    lambda m: numbers(m).resample("h").agg(lambda x: x.sum()),
    lambda m: numbers(m).resample("h").apply(lambda x: x.sum()),
    lambda m: numbers(m).resample("h").transform(lambda x: x),
    lambda m: numbers(m).resample("h").ffill(),
    lambda m: numbers(m).resample("h").bfill(),
    lambda m: numbers(m).resample("h").nearest(),
    lambda m: numbers(m).resample("h").interpolate(),
    lambda m: numbers(m).resample("h").first(skipna=False),
]


@pytest.mark.parametrize("build", REFUSED)
def test_what_is_not_written_is_refused(firepanda: ModuleType, build: Callable[[Any], Any]) -> None:
    """Calendar rules, shifted bins, finer units, two levels of labels and upsampling."""
    with pytest.raises(NotImplementedError):
        build(firepanda)


@pytest.mark.parametrize("owner", ["Series", "DataFrame"])
def test_the_resample_signature_is_pandas_signature(firepanda: ModuleType, owner: str) -> None:
    """Parameter for parameter, with the same defaults."""
    import pandas as pd

    ours = inspect.signature(getattr(firepanda, owner).resample).parameters
    yours = inspect.signature(getattr(pd, owner).resample).parameters
    assert [(p.name, p.kind) for p in ours.values()] == [(p.name, p.kind) for p in yours.values()]
    for each in ours:
        assert ours[each].default == yours[each].default, each


METHODS = [*REDUCTIONS, "agg", "aggregate", "apply", "transform", "pipe", "asfreq", "ohlc"]
METHODS += ["size", "get_group", "ffill", "bfill", "nearest", "interpolate"]


@pytest.mark.parametrize("name", METHODS)
def test_a_method_signature_is_pandas_signature(firepanda: ModuleType, name: str) -> None:
    """Parameter for parameter, with the same defaults."""
    import pandas as pd

    mine = type(numbers(firepanda).resample("h"))
    theirs = type(numbers(pd).resample("h"))
    ours = inspect.signature(getattr(mine, name)).parameters
    yours = inspect.signature(getattr(theirs, name)).parameters
    assert [(p.name, p.kind) for p in ours.values()] == [(p.name, p.kind) for p in yours.values()]
    for each in ours:
        assert ours[each].default == yours[each].default, each
