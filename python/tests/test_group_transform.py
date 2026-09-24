"""The grouped transforms, checked against pandas.

`cumsum`, `cumprod`, `cummax`, `cummin`, `shift`, `diff`, `cumcount`, `ngroup`
and `transform` on a group by keep the row count. Every test builds the same transform in both
libraries and compares the columns, the types, the row labels and every row,
including the rows whose key is missing, which belong to no group and answer
missing.
"""

from __future__ import annotations

import importlib.util
import inspect
import math
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
    """Equal row by row, with NaN equal to NaN and None equal to NaN.

    A float may be one rounding away, since a deviation sums its squares in a
    different order from pandas.
    """

    def missing(value: Any) -> bool:
        return value is None or value != value

    def close(a: Any, b: Any) -> bool:
        if isinstance(a, float) and isinstance(b, float):
            return math.isclose(a, b, rel_tol=1e-12)
        return bool(a == b)

    return len(got) == len(want) and all(
        (missing(a) and missing(b)) or close(a, b) for a, b in zip(got, want, strict=True)
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


SCANS = ["cumsum", "cumprod", "cummax", "cummin", "shift", "diff", "cumcount", "ngroup"]

GROUPS: list[Callable[[Any], Any]] = [
    lambda m: frame(m).groupby("k")["x"],
    lambda m: frame(m).groupby("k")["y"],
    lambda m: frame(m).groupby("k")[["x", "y"]],
    lambda m: frame(m).groupby("k", dropna=False)["x"],
    lambda m: frame(m).groupby("k", sort=False)["y"],
    lambda m: frame(m).groupby(["j", "k"])["x"],
    lambda m: frame(m).groupby(["k", "j"], dropna=False)["y"],
    lambda m: frame(m).drop(columns="k").groupby("j"),
    lambda m: frame(m).astype({"x": "int8"}).groupby("j")["x"],
    lambda m: frame(m).astype({"y": "float32"}).groupby("k")["y"],
]


@pytest.mark.parametrize("kind", SCANS)
@pytest.mark.parametrize("group", GROUPS)
def test_a_transform_is_pandas_transform(
    firepanda: ModuleType, group: Callable[[Any], Any], kind: str
) -> None:
    """Every transform, on a column and on a frame, with and without missing keys."""
    import pandas as pd

    agrees(getattr(group(firepanda), kind)(), getattr(group(pd), kind)())


NAMED = ["sum", "mean", "min", "max", "count", "first", "last", "median", "std", "prod"]


@pytest.mark.parametrize("func", [*NAMED, "cumsum", "shift", "diff"])
@pytest.mark.parametrize("group", GROUPS)
def test_a_named_transform_is_pandas_transform(
    firepanda: ModuleType, group: Callable[[Any], Any], func: str
) -> None:
    """A reduction put on every row of its group, and a scan named as a string."""
    import pandas as pd

    agrees(group(firepanda).transform(func), group(pd).transform(func))


SHIFTS: list[Callable[[Any], Any]] = [
    lambda m: frame(m).groupby("k")["x"].shift(2),
    lambda m: frame(m).groupby("k")["y"].shift(-1),
    lambda m: frame(m).groupby("k")["x"].shift(0),
    lambda m: frame(m).groupby("j")[["x", "y"]].shift(-2),
    lambda m: frame(m).groupby("k", dropna=False)["y"].shift(periods=1),
    lambda m: frame(m).groupby("k")["x"].diff(2),
    lambda m: frame(m).groupby("k")["y"].diff(-1),
    lambda m: frame(m).groupby("k")["x"].diff(0),
    lambda m: frame(m).astype({"x": "int16"}).groupby("j")["x"].diff(),
    lambda m: frame(m).astype({"x": "uint8"}).groupby("j")["x"].diff(),
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
        lambda: grouped.transform(lambda v: v),
        lambda: grouped.transform("sum", engine="numba"),
        lambda: frame(firepanda).astype({"x": "bool"}).groupby("k")["x"].diff(),
    ):
        with pytest.raises(NotImplementedError):
            build()


def test_a_bad_name_is_pandas_mistake(firepanda: ModuleType) -> None:
    """The same class and sentence for a name transform does not know."""
    import pandas as pd

    with pytest.raises(ValueError) as theirs:
        frame(pd).groupby("k")["x"].transform("nope")
    with pytest.raises(ValueError) as mine:
        frame(firepanda).groupby("k")["x"].transform("nope")
    assert str(mine.value) == str(theirs.value)


@pytest.mark.parametrize("name", [*SCANS, "transform"])
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
