"""`Series.rank`, `DataFrame.rank` and `groupby(...).rank()`, checked against pandas.

All three are one walk in the core, a sort and a pass over the ties, with a
grouping handed in or not. Every test builds the same rank in both libraries
and compares the types, the labels and every row, because the ties, the missing
values and the fractions are where a rank goes wrong, and each of them moves a
number rather than raising.
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
    "k": ["a", "b", "a", "b", "a", None, "b", "a"],
    "x": [3.0, 1.0, 3.0, NAN, 2.0, 5.0, 1.0, NAN],
    "n": [4, 4, 1, 2, 2, 7, 9, 4],
    "s": ["q", "p", "q", "r", "p", "s", "p", "t"],
    "b": [True, False, True, True, False, True, False, False],
}
METHODS = ["average", "min", "max", "first", "dense"]
PLACES = ["keep", "top", "bottom"]


def labelled(m: Any, labels: list[str], values: list[Any], name: str = "v") -> Any:
    """A series on text labels, built the one way both libraries share."""
    return m.DataFrame({"i": labels, name: values}).set_index("i")[name]


def same(got: list[Any], want: list[Any]) -> bool:
    """Equal row by row, with NaN equal to NaN."""

    def missing(value: Any) -> bool:
        return value is None or value != value

    return len(got) == len(want) and all(
        (missing(a) and missing(b)) or abs(a - b) < 1e-12 for a, b in zip(got, want, strict=True)
    )


def agrees(got: Any, want: Any) -> None:
    """The same types, rows and row labels as pandas, frame or series."""
    assert list(got.index) == list(want.index)
    if hasattr(want, "columns"):
        assert list(got.columns) == list(want.columns)
        for name in want.columns:
            assert got[name].dtype == str(want[name].dtype), name
            assert same(got[name].tolist(), want[name].tolist()), name
    else:
        assert got.name == want.name
        assert got.dtype == str(want.dtype)
        assert same(got.tolist(), want.tolist())


@pytest.mark.parametrize("pct", [False, True])
@pytest.mark.parametrize("ascending", [True, False])
@pytest.mark.parametrize("na_option", PLACES)
@pytest.mark.parametrize("method", METHODS)
def test_a_series_rank_is_pandas_rank(
    firepanda: ModuleType, method: str, na_option: str, ascending: bool, pct: bool
) -> None:
    """Every method, placement, direction and fraction over floats with gaps."""
    import pandas as pd

    options = {"method": method, "na_option": na_option, "ascending": ascending, "pct": pct}
    agrees(
        firepanda.Series(FRAME["x"], name="x").rank(**options),
        pd.Series(FRAME["x"], name="x").rank(**options),
    )


@pytest.mark.parametrize("pct", [False, True])
@pytest.mark.parametrize("ascending", [True, False])
@pytest.mark.parametrize("na_option", PLACES)
@pytest.mark.parametrize("method", METHODS)
def test_a_group_rank_is_pandas_rank(
    firepanda: ModuleType, method: str, na_option: str, ascending: bool, pct: bool
) -> None:
    """Every column that is not a key, the missing key answering NaN."""
    import pandas as pd

    options = {"method": method, "na_option": na_option, "ascending": ascending, "pct": pct}
    agrees(
        firepanda.DataFrame(FRAME).groupby("k").rank(**options),
        pd.DataFrame(FRAME).groupby("k").rank(**options),
    )


BUILDS: list[Callable[[Any], Any]] = [
    lambda m: m.DataFrame(FRAME).rank(),
    lambda m: m.DataFrame(FRAME).rank(method="dense", pct=True),
    lambda m: m.DataFrame(FRAME).rank(numeric_only=True),
    lambda m: m.DataFrame(FRAME).rank(ascending=False, na_option="top"),
    lambda m: m.DataFrame(FRAME).set_index("s").rank(),
    lambda m: m.DataFrame(FRAME).rank(axis="index"),
    lambda m: m.Series(FRAME["s"], name="s").rank(),
    lambda m: m.Series(FRAME["b"]).rank(method="min"),
    lambda m: m.Series(FRAME["n"]).astype("int32").rank(pct=True),
    lambda m: m.Series(FRAME["n"]).astype("uint8").rank(method="first", ascending=False),
    lambda m: m.Series(FRAME["n"]).rank(numeric_only=True),
    lambda m: m.Series(FRAME["k"]).rank(na_option="bottom"),
    lambda m: m.Series(FRAME["s"]).astype("category").rank(),
    lambda m: (
        m.Series(["b", "a", "c"]).astype("category").cat.set_categories(["c", "b", "a"]).rank()
    ),
    lambda m: (
        m.Series(["b", "a", "c", None])
        .astype("category")
        .cat.set_categories(["c", "b", "a"], ordered=True)
        .rank()
    ),
    lambda m: m.Series([], dtype="float64").rank(),
    lambda m: labelled(m, ["x", "y", "z"], [2.0, 1.0, 2.0]).rank(),
    lambda m: m.DataFrame(FRAME).groupby("k")["x"].rank(),
    lambda m: m.DataFrame(FRAME).groupby("k")["s"].rank(method="dense", ascending=False),
    lambda m: m.DataFrame(FRAME).groupby("k", dropna=False).rank(),
    lambda m: m.DataFrame(FRAME).groupby("k", dropna=False)["n"].rank(pct=True),
    lambda m: m.DataFrame(FRAME).groupby("k", sort=False).rank(method="max"),
    lambda m: m.DataFrame(FRAME).groupby("k", as_index=False).rank(),
    lambda m: m.DataFrame(FRAME).groupby(["k", "b"]).rank(),
    lambda m: m.DataFrame(FRAME).groupby("b")["x"].rank(na_option="top", pct=True),
    lambda m: m.DataFrame(FRAME).set_index("s").groupby("k")["n"].rank(),
    lambda m: m.DataFrame(FRAME).groupby("k")["x"].transform("rank"),
]


@pytest.mark.parametrize("build", BUILDS)
def test_a_rank_is_pandas_rank(firepanda: ModuleType, build: Callable[[Any], Any]) -> None:
    """Types, labels, keys, text, flags and categories in their own order."""
    import pandas as pd

    agrees(build(firepanda), build(pd))


MISTAKES: list[Callable[[Any], Any]] = [
    lambda m: m.Series([1.0]).rank(method="nope"),
    lambda m: m.Series([1.0]).rank(na_option="nope"),
    lambda m: m.Series([1.0]).rank(axis=1),
    lambda m: m.Series(["a"]).rank(numeric_only=True),
    lambda m: m.DataFrame(FRAME).rank(method="nope"),
    lambda m: m.DataFrame(FRAME).rank(na_option="nope"),
    lambda m: m.DataFrame(FRAME).groupby("k").rank(method="nope"),
    lambda m: m.DataFrame(FRAME).groupby("k")["x"].rank(na_option="nope"),
    lambda m: m.DataFrame(FRAME).astype({"s": "category"}).groupby("k").rank(),
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


def test_a_rank_across_a_row_is_refused(firepanda: ModuleType) -> None:
    """Refused by name rather than answered down the columns."""
    with pytest.raises(NotImplementedError):
        firepanda.DataFrame(FRAME).rank(axis=1)


OWNERS: list[Callable[[Any], Any]] = [
    lambda m: m.Series,
    lambda m: m.DataFrame,
    lambda m: type(m.DataFrame(FRAME).groupby("k")),
    lambda m: type(m.DataFrame(FRAME).groupby("k")["x"]),
]


@pytest.mark.parametrize("owner", OWNERS)
def test_the_signature_is_pandas_signature(
    firepanda: ModuleType, owner: Callable[[Any], Any]
) -> None:
    """Parameter for parameter, with the same defaults."""
    import pandas as pd

    ours = inspect.signature(owner(firepanda).rank).parameters
    yours = inspect.signature(owner(pd).rank).parameters
    assert list(ours) == list(yours)
    for each in ours:
        assert ours[each].default == yours[each].default, each
