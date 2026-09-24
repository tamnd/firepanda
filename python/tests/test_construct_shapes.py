"""`DataFrame(...)` and `Series(...)` on every shape of data pandas takes, checked against pandas.

A frame is built from a mapping, a list of records, a list of rows, a two
dimensional numpy array or another frame, with `index=` and `columns=` beside
any of them, and a series from a list, a numpy array, one value or another
series. Every test builds the same thing in both libraries and compares the
columns, the types, the labels and the rows, because a constructor that guesses
a type or a label wrong is wrong everywhere after it.
"""

from __future__ import annotations

import importlib.util
from collections.abc import Callable
from types import ModuleType
from typing import Any

import pytest

pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("pandas") is None or importlib.util.find_spec("numpy") is None,
    reason="pandas and numpy are not installed",
)


def keyed(m: Any, labels: list[Any], values: list[Any]) -> Any:
    """A series on chosen labels, built the one way both libraries share."""
    return m.DataFrame({"i": labels, "v": values}).set_index("i")["v"].rename(None)


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
    """The same columns, types, rows, labels and label name as pandas."""
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


def np() -> Any:
    """numpy, imported late so the file loads without it."""
    import numpy

    return numpy


FRAMES: list[Callable[[Any], Any]] = [
    lambda m: m.DataFrame([{"a": 1, "b": "x"}, {"b": "y", "c": 2.5}]),
    lambda m: m.DataFrame([{"a": 1}, {"a": 2}], index=["p", "q"]),
    lambda m: m.DataFrame([[1, "x"], [2, "y"]], columns=["a", "b"]),
    lambda m: m.DataFrame([(1, "x"), (2, "y")], columns=["a", "b"], index=[5, 6]),
    lambda m: m.DataFrame(np().arange(6).reshape(3, 2), columns=["a", "b"], index=["x", "y", "z"]),
    lambda m: m.DataFrame(np().arange(4, dtype="float32").reshape(2, 2), columns=["a", "b"]),
    lambda m: m.DataFrame({"a": [1, 2]}, index=["x", "y"]),
    lambda m: m.DataFrame({"a": [1, 2], "b": [3, 4]}, columns=["b", "a"]),
    lambda m: m.DataFrame({"a": 1, "b": [3, 4]}),
    lambda m: m.DataFrame({"a": 1, "b": "s"}, index=[5, 6]),
    lambda m: m.DataFrame({"a": keyed(m, ["x", "y"], [1, 2]), "b": keyed(m, ["y", "z"], [3, 4])}),
    lambda m: m.DataFrame({"a": keyed(m, ["y", "x"], [1, 2]), "b": keyed(m, ["y", "x"], [3, 4])}),
    lambda m: m.DataFrame({"a": keyed(m, ["y", "x"], [1, 2]), "b": [3, 4]}),
    lambda m: m.DataFrame({"a": keyed(m, ["y", "x"], [1, 2])}, index=["x", "q"]),
    lambda m: m.DataFrame(
        {"a": np().array([1, 2], dtype="int16"), "b": np().array([1.5, np().nan])}
    ),
    lambda m: m.DataFrame({"k": ["x", "y"], "c": m.Series(["p", "q"]).astype("category")}),
    lambda m: m.DataFrame({"c": keyed(m, ["y", "x"], ["p", "q"]).astype("category"), "n": [1, 2]}),
    lambda m: m.DataFrame(m.DataFrame({"a": [1], "b": [2]}), columns=["b"]),
    lambda m: m.DataFrame(m.DataFrame({"a": [1, 2]}), index=[1, 5]),
    lambda m: m.DataFrame({"v": [1, 2]}, index=m.Index(["a", "b"], name="k")),
    lambda m: m.DataFrame({"a": [np().int64(1), np().int32(2)]}),
    lambda m: m.DataFrame(),
    lambda m: m.DataFrame({}),
]


@pytest.mark.parametrize("build", FRAMES)
def test_a_frame_is_pandas_frame(firepanda: ModuleType, build: Callable[[Any], Any]) -> None:
    """Records, rows, arrays, labels, picked columns, broadcasts and aligned series."""
    import pandas as pd

    agrees(build(firepanda), build(pd))


SERIES: list[Callable[[Any], Any]] = [
    lambda m: m.Series([1, 2], index=["a", "b"], name="n"),
    lambda m: m.Series(5),
    lambda m: m.Series(5.5, index=["a", "b"]),
    lambda m: m.Series("s", index=[1, 2, 3], name="t"),
    lambda m: m.Series(keyed(m, ["a", "b"], [1, 2]), index=["b", "c"]),
    lambda m: m.Series(keyed(m, ["a", "b"], [1, 2]), index=["b"], name="z"),
    lambda m: m.Series(np().array([1, 2], dtype="int32")),
    lambda m: m.Series(np().array([1, 2], dtype="uint8")),
    lambda m: m.Series(np().array([True, False])),
    lambda m: m.Series(np().array(["a", "b"])),
    lambda m: m.Series(np().array([1.0, None], dtype="float64")),
    lambda m: m.Series([np().int64(1), np().int32(2)]),
    lambda m: m.Series([np().float32(1.5), 2.5]),
    lambda m: m.Series([1, 2], index=m.Index(["a", "b"], name="k")),
    lambda m: m.Series([1, 2], index=["a", "b"], dtype="float32"),
    lambda m: m.Series(np().array([1, 2]), dtype="float64"),
    lambda m: m.Series(range(3), index=[3, 2, 1]),
]


@pytest.mark.parametrize("build", SERIES)
def test_a_series_is_pandas_series(firepanda: ModuleType, build: Callable[[Any], Any]) -> None:
    """Labels, one value repeated, a reindex and numpy's own types."""
    import pandas as pd

    agrees(build(firepanda), build(pd))


@pytest.mark.parametrize("unit", ["s", "ms", "us", "ns", "D"])
def test_a_numpy_instant_keeps_its_unit(firepanda: ModuleType, unit: str) -> None:
    """The type pandas keeps, a day read as seconds, and the same moments."""
    import pandas as pd

    rows = np().array(["2024-01-01", "2024-03-02"], dtype=f"datetime64[{unit}]")
    got, want = firepanda.Series(rows), pd.Series(rows)
    assert got.dtype == str(want.dtype)
    assert got.dt.year.tolist() == want.dt.year.tolist()
    assert got.dt.day.tolist() == want.dt.day.tolist()
    framed = firepanda.DataFrame({"t": rows})
    assert framed["t"].dtype == str(pd.DataFrame({"t": rows})["t"].dtype)
    assert framed["t"].dt.month.tolist() == [1, 3]


MISTAKES: list[Callable[[Any], Any]] = [
    lambda m: m.DataFrame([[1, "x"], [2, "y"]], columns=["a"]),
    lambda m: m.DataFrame({"a": [1, 2]}, index=["x"]),
    lambda m: m.DataFrame({"a": 1}),
    lambda m: m.DataFrame(np().arange(6).reshape(3, 2), columns=["a", "b", "c"]),
    lambda m: m.Series([1, 2], index=["a"]),
    lambda m: m.Series(np().zeros((2, 2))),
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
    lambda m: m.DataFrame([[1, 2]]),
    lambda m: m.DataFrame(np().zeros((2, 2))),
    lambda m: m.DataFrame({"a": [1]}, columns=["a", "b"]),
    lambda m: m.Series(np().array([1], dtype="timedelta64[s]")),
    lambda m: m.DataFrame({"a": [1]}, copy=True),
]


@pytest.mark.parametrize("build", REFUSED)
def test_a_shape_firepanda_cannot_hold_is_refused(
    firepanda: ModuleType, build: Callable[[Any], Any]
) -> None:
    """Integer column names, object columns and spans are refused by name."""
    with pytest.raises(NotImplementedError):
        build(firepanda)
