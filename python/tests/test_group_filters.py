"""`head`, `tail` and `nth` on a group by, `idxmax` and `idxmin`, and masks.

The three filters answer rows of the frame, in the frame's order and with its
labels, which is what pandas has done since 2.0, and a row whose key is not a
group is left out. `idxmax` and `idxmin` answer one label a group. A boolean
mask picks rows by label, so a mask that was reordered on the way still picks
the rows it was computed for, and `df[mask]` is `df.loc[mask]`. Every test
builds the same thing in both libraries and compares it.
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
    "i": [10, 20, 30, 40, 50, 60, 70, 80],
    "k": ["a", "b", "a", None, "b", "a", "c", "a"],
    "v": [3.0, 1.0, 5.0, 2.0, 1.0, NAN, 8.0, 5.0],
    "w": [1, 2, 3, 4, 5, 6, 7, 0],
}


def frame(m: Any) -> Any:
    """Keys with a gap, floats with a gap and ties, on labels that are not places."""
    return m.DataFrame(DATA).set_index("i")


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
    lambda m: frame(m).groupby("k").head(1),
    lambda m: frame(m).groupby("k").head(2),
    lambda m: frame(m).groupby("k").head(),
    lambda m: frame(m).groupby("k").head(0),
    lambda m: frame(m).groupby("k").head(-1),
    lambda m: frame(m).groupby("k").tail(1),
    lambda m: frame(m).groupby("k").tail(-2),
    lambda m: frame(m).groupby("k", dropna=False).head(1),
    lambda m: frame(m).groupby("k", sort=False).tail(2),
    lambda m: frame(m).groupby("k", as_index=False).head(1),
    lambda m: frame(m).groupby(["k", "w"]).head(1),
    lambda m: frame(m).groupby("k")["v"].head(1),
    lambda m: frame(m).groupby("k")["w"].tail(2),
    lambda m: frame(m).groupby("k").nth(0),
    lambda m: frame(m).groupby("k").nth(1),
    lambda m: frame(m).groupby("k").nth(-1),
    lambda m: frame(m).groupby("k").nth(5),
    lambda m: frame(m).groupby("k").nth([0, -1]),
    lambda m: frame(m).groupby("k").nth[1],
    lambda m: frame(m).groupby("k").nth([]),
    lambda m: frame(m).groupby("k", dropna=False).nth(-1),
    lambda m: frame(m).groupby("k")["v"].nth(0),
    lambda m: frame(m).drop(index=60).groupby("k")["v"].idxmax(),
    lambda m: frame(m).drop(index=60).groupby("k")["v"].idxmin(),
    lambda m: frame(m).drop(index=60).groupby("k")["w"].idxmax(),
    lambda m: frame(m).drop(index=60).groupby("k").idxmin(),
    lambda m: frame(m).drop(index=60).groupby("k").idxmax(),
    lambda m: frame(m).drop(index=60).groupby("k", sort=False)["w"].idxmin(),
    lambda m: frame(m).drop(index=60).groupby("k", dropna=False)["v"].idxmax(),
]


@pytest.mark.parametrize("build", BUILDS)
def test_a_group_filter_is_pandas_filter(
    firepanda: ModuleType, build: Callable[[Any], Any]
) -> None:
    """Front, back, negative, several places, the missing key and one column."""
    import pandas as pd

    agrees(build(firepanda), build(pd))


MASKS: list[Callable[[Any], Any]] = [
    lambda m: frame(m)[frame(m)["v"] > 2],
    lambda m: frame(m)[(frame(m)["v"] > 2).iloc[::-1]],
    lambda m: frame(m)[(frame(m)["v"] > 2).sort_values()],
    lambda m: frame(m)[[True, False] * 4],
    lambda m: frame(m)[lambda f: f["w"] % 2 == 0],
    lambda m: frame(m).loc[(frame(m)["w"] > 2).iloc[::-1]],
    lambda m: frame(m).loc[(frame(m)["w"] > 2).iloc[::-1], "v"],
    lambda m: frame(m)["v"][(frame(m)["w"] > 2).iloc[::-1]],
    lambda m: frame(m)["v"].loc[(frame(m)["w"] > 2).iloc[::-1]],
    lambda m: frame(m)[frame(m)["w"] > 100],
]


@pytest.mark.parametrize("build", MASKS)
def test_a_mask_is_pandas_mask(firepanda: ModuleType, build: Callable[[Any], Any]) -> None:
    """A mask lined up on labels, a list by place, a callable and an empty answer."""
    import pandas as pd

    agrees(build(firepanda), build(pd))


def test_a_numpy_mask_is_read_by_place(firepanda: ModuleType) -> None:
    """An array of booleans, in `[]` and in `loc`."""
    np = pytest.importorskip("numpy")
    import pandas as pd

    keep = np.array([True, False, True, True, False, False, True, False])
    agrees(frame(firepanda)[keep], frame(pd)[keep])
    agrees(frame(firepanda).loc[keep], frame(pd).loc[keep])


MISTAKES: list[Callable[[Any], Any]] = [
    lambda m: frame(m)[(frame(m)["v"] > 2).iloc[:3]],
    lambda m: frame(m).loc[(frame(m)["v"] > 2).iloc[:3]],
    lambda m: frame(m)[[True, False]],
    lambda m: m.DataFrame({"k": ["a", "b"], "v": [1.0, NAN]}).groupby("k")["v"].idxmax(),
    lambda m: m.DataFrame({"k": ["a", "b"], "v": [1.0, NAN]}).groupby("k").idxmin(),
]


@pytest.mark.parametrize("build", MISTAKES)
def test_a_mistake_is_pandas_mistake(firepanda: ModuleType, build: Callable[[Any], Any]) -> None:
    """The same first line, and the same class where pandas' is not its own."""
    import pandas as pd

    with pytest.raises(Exception) as theirs:
        build(pd)
    with pytest.raises(Exception) as mine:
        build(firepanda)
    assert type(mine.value).__name__ == type(theirs.value).__name__ or isinstance(
        mine.value, type(theirs.value)
    )
    assert str(mine.value).split("\n")[0] == str(theirs.value).split("\n")[0]


REFUSED: list[Callable[[Any], Any]] = [
    lambda m: frame(m).groupby("k").nth(slice(0, 2)),
    lambda m: frame(m).groupby("k").nth(0, dropna="any"),
    lambda m: frame(m).groupby("k")["v"].idxmax(skipna=False),
    lambda m: frame(m).groupby("k", as_index=False).idxmax(),
]


@pytest.mark.parametrize("build", REFUSED)
def test_what_is_not_written_is_refused(firepanda: ModuleType, build: Callable[[Any], Any]) -> None:
    """A slice of places, the deprecated `dropna`, `skipna` and keys as columns."""
    with pytest.raises(NotImplementedError):
        build(firepanda)


@pytest.mark.parametrize("name", ["head", "tail", "idxmax", "idxmin"])
@pytest.mark.parametrize("owner", ["DataFrameGroupBy", "SeriesGroupBy"])
def test_the_signature_is_pandas_signature(firepanda: ModuleType, owner: str, name: str) -> None:
    """Parameter for parameter, with the same defaults."""
    import pandas as pd

    build = frame(firepanda).groupby("k"), frame(pd).groupby("k")
    if owner == "SeriesGroupBy":
        build = build[0]["v"], build[1]["v"]
    mine = inspect.signature(getattr(type(build[0]), name)).parameters
    yours = inspect.signature(getattr(type(build[1]), name)).parameters
    assert [(p.name, p.default) for p in mine.values()] == [
        (p.name, p.default) for p in yours.values()
    ]
