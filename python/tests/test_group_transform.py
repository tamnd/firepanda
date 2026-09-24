"""The grouped transforms, checked against pandas.

`cumsum`, `cumprod`, `cummax`, `cummin`, `shift`, `cumcount` and `ngroup` on a
group by keep the row count. Every test builds the same transform in both
libraries and compares the columns, the types, the row labels and every row,
including the rows whose key is missing, which belong to no group and answer
missing.
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

DATA = {
    "k": ["b", "a", "b", None, "a", "c", "b"],
    "j": [1, 1, 2, 2, 1, 1, 2],
    "x": [1, 2, 3, 4, 5, 6, 7],
    "y": [1.5, None, 2.5, 3.5, -1.0, 0.5, 2.0],
}


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
    if hasattr(want, "columns"):
        assert list(got.columns) == list(want.columns)
        for name in want.columns:
            assert got[name].dtype == printed(want[name].dtype), name
            assert same(got[name].tolist(), want[name].tolist()), name
    else:
        assert got.name == want.name
        assert got.dtype == printed(want.dtype)
        assert same(got.tolist(), want.tolist())


def frame(m: Any) -> Any:
    """The frame every test groups, with its labels not the default ones."""
    return m.DataFrame(DATA).set_index("j", drop=False).rename_axis(None)


SCANS = ["cumsum", "cumprod", "cummax", "cummin", "shift", "cumcount", "ngroup"]

BUILDS: list[Callable[[Any, str], Any]] = [
    lambda m, kind: getattr(frame(m).groupby("k")["x"], kind)(),
    lambda m, kind: getattr(frame(m).groupby("k")["y"], kind)(),
    lambda m, kind: getattr(frame(m).groupby("k")[["x", "y"]], kind)(),
    lambda m, kind: getattr(frame(m).groupby("k", dropna=False)["x"], kind)(),
    lambda m, kind: getattr(frame(m).groupby("k", sort=False)["y"], kind)(),
    lambda m, kind: getattr(frame(m).groupby(["j", "k"])["x"], kind)(),
    lambda m, kind: getattr(frame(m).groupby(["k", "j"], dropna=False)["y"], kind)(),
    lambda m, kind: getattr(frame(m).drop(columns="k").groupby("j"), kind)(),
    lambda m, kind: getattr(frame(m).astype({"x": "int8"}).groupby("j")["x"], kind)(),
    lambda m, kind: getattr(frame(m).astype({"y": "float32"}).groupby("k")["y"], kind)(),
]


@pytest.mark.parametrize("kind", SCANS)
@pytest.mark.parametrize("build", BUILDS)
def test_a_transform_is_pandas_transform(
    firepanda: ModuleType, build: Callable[[Any, str], Any], kind: str
) -> None:
    """Every transform, on a column and on a frame, with and without missing keys."""
    import pandas as pd

    agrees(build(firepanda, kind), build(pd, kind))


SHIFTS: list[Callable[[Any], Any]] = [
    lambda m: frame(m).groupby("k")["x"].shift(2),
    lambda m: frame(m).groupby("k")["y"].shift(-1),
    lambda m: frame(m).groupby("k")["x"].shift(0),
    lambda m: frame(m).groupby("j")[["x", "y"]].shift(-2),
    lambda m: frame(m).groupby("k", dropna=False)["y"].shift(periods=1),
]


@pytest.mark.parametrize("build", SHIFTS)
def test_a_shift_is_pandas_shift(firepanda: ModuleType, build: Callable[[Any], Any]) -> None:
    """Forward, back, by nothing and past the end of a group."""
    import pandas as pd

    agrees(build(firepanda), build(pd))


def test_a_bool_column_folds_the_way_pandas_folds_it(firepanda: ModuleType) -> None:
    """A sum or a product of truths counts, and a running extreme stays a truth."""
    import pandas as pd

    data = {"k": [1, 2, 1, 1, 2], "t": [True, False, True, False, True]}
    for kind in ("cumsum", "cumprod", "cummax", "cummin"):
        got = getattr(firepanda.DataFrame(data).groupby("k")["t"], kind)()
        want = getattr(pd.DataFrame(data).groupby("k")["t"], kind)()
        agrees(got, want)


def test_what_is_not_written_is_refused(firepanda: ModuleType) -> None:
    """Counting backwards, a fill, a frequency, a suffix and a list of periods."""
    grouped = frame(firepanda).groupby("k")["x"]
    for build in (
        lambda: grouped.cumcount(ascending=False),
        lambda: grouped.ngroup(ascending=False),
        lambda: grouped.shift(fill_value=0),
        lambda: grouped.shift(freq="D"),
        lambda: grouped.shift([1, 2]),
        lambda: grouped.shift(1, suffix="_s"),
        lambda: grouped.cumsum(numeric_only=True),
    ):
        with pytest.raises(NotImplementedError):
            build()


@pytest.mark.parametrize("name", SCANS)
@pytest.mark.parametrize("cls", ["DataFrameGroupBy", "SeriesGroupBy"])
def test_the_signatures_are_pandas_signatures(firepanda: ModuleType, cls: str, name: str) -> None:
    """Parameter for parameter, with the same defaults but for the sentinel."""
    import pandas as pd

    mine = inspect.signature(getattr(getattr(firepanda._frame, cls), name)).parameters
    theirs = inspect.signature(getattr(getattr(pd.api.typing, cls), name)).parameters
    assert list(mine) == list(theirs)
    for key in mine:
        if key != "fill_value":
            assert mine[key].default == theirs[key].default, key
