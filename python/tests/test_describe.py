"""`describe` and `kurt` on a series and a frame, checked against pandas.

`describe` is a count, a mean, a spread, the two ends and the percentiles
between them, each one a reduction firepanda already has, and `kurt` is pandas'
own sum of fourth powers written with the column's arithmetic. Every test builds
the same answer in both libraries and compares the labels, the types and the
numbers, the last few bits apart, since two orders of adding are not the same
number to the last bit.
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
    "x": [3.0, 1.0, 3.0, NAN, 2.0, 5.0, 1.0, 11.5],
    "n": [4, 4, 1, 2, 2, 7, 9, 4],
    "u": [1, 2, 3, 4, 5, 6, 7, 8],
    "s": ["q", "p", "q", "r", "p", "s", "p", "t"],
    "b": [True, False, True, True, False, True, False, False],
}


def close(got: list[Any], want: list[Any]) -> bool:
    """Equal row by row to a relative 1e-9, with NaN equal to NaN."""

    def missing(value: Any) -> bool:
        return value is None or value != value

    return len(got) == len(want) and all(
        (missing(a) and missing(b)) or (a == b or abs(a - b) <= 1e-9 * max(abs(a), abs(b)))
        for a, b in zip(got, want, strict=True)
    )


def agrees(got: Any, want: Any) -> None:
    """The same labels, types and numbers as pandas, frame, series or one number."""
    if not hasattr(want, "index"):
        assert close([got], [want])
        return
    assert list(got.index) == list(want.index)
    if hasattr(want, "columns"):
        assert list(got.columns) == list(want.columns)
        for name in want.columns:
            assert got[name].dtype == str(want[name].dtype), name
            assert close(got[name].tolist(), want[name].tolist()), name
    else:
        assert got.name == want.name
        assert got.dtype == str(want.dtype)
        assert close(got.tolist(), want.tolist())


BUILDS: list[Callable[[Any], Any]] = [
    lambda m: m.Series(FRAME["x"], name="x").kurt(),
    lambda m: m.Series(FRAME["x"]).kurtosis(),
    lambda m: m.Series(FRAME["x"]).kurt(skipna=False),
    lambda m: m.Series(FRAME["n"]).kurt(),
    lambda m: m.Series(FRAME["n"]).astype("uint8").kurt(),
    lambda m: m.Series(FRAME["b"]).kurt(),
    lambda m: m.Series([1.0, 2.0, 3.0]).kurt(),
    lambda m: m.Series([2.5, 2.5, 2.5, 2.5, 2.5]).kurt(),
    lambda m: m.Series([1e8 + 0.1, 1e8 + 0.1, 1e8 + 0.1, 1e8 + 0.1]).kurt(),
    lambda m: m.Series([], dtype="float64").kurt(),
    lambda m: m.Series([1.0, float("inf"), 2.0, float("-inf"), 3.0]).kurt(),
    lambda m: m.Series([1.0, float("inf"), 2.0, 5.0, 3.0]).kurt(),
    lambda m: m.Series([NAN, NAN], dtype="float64").kurt(),
    lambda m: m.DataFrame(FRAME).kurt(numeric_only=True),
    lambda m: m.DataFrame(FRAME).drop(columns="s").kurt(),
    lambda m: m.DataFrame(FRAME)[["x", "n"]].kurtosis(skipna=False),
    lambda m: m.DataFrame(FRAME)[["x"]].kurt(axis="index"),
    lambda m: m.Series(FRAME["x"], name="x").describe(),
    lambda m: m.Series(FRAME["n"], name="n").describe(),
    lambda m: m.Series(FRAME["n"]).astype("int32").describe(percentiles=[0.1, 0.9]),
    lambda m: m.Series(FRAME["x"]).describe(percentiles=[0.333333, 0.25]),
    lambda m: m.Series(FRAME["x"]).describe(percentiles=[0.01999, 0.02001, 0.5, 0.9999]),
    lambda m: m.Series(FRAME["x"]).describe(percentiles=[0, 1]),
    lambda m: m.Series(FRAME["x"]).describe(percentiles=[]),
    lambda m: m.Series([], dtype="float64").describe(),
    lambda m: m.Series([NAN, NAN], name="gone").describe(),
    lambda m: m.Series([7.0]).describe(),
    lambda m: m.DataFrame(FRAME).describe(),
    lambda m: m.DataFrame(FRAME).describe(percentiles=[0.05, 0.5, 0.95]),
    lambda m: m.DataFrame(FRAME)[["x", "u"]].describe(include="all"),
    lambda m: m.DataFrame(FRAME)[["x", "s", "u"]].describe(),
    lambda m: m.DataFrame(FRAME).set_index("s")[["x"]].describe(),
]


@pytest.mark.parametrize("build", BUILDS)
def test_a_description_is_pandas_description(
    firepanda: ModuleType, build: Callable[[Any], Any]
) -> None:
    """Numbers of every width, flags, gaps, constants, empties and percentiles."""
    import pandas as pd

    agrees(build(firepanda), build(pd))


MISTAKES: list[Callable[[Any], Any]] = [
    lambda m: m.Series(FRAME["s"]).kurt(),
    lambda m: m.Series(FRAME["s"]).astype("category").kurt(),
    lambda m: m.to_datetime(m.Series(["2024-01-01", "2024-01-03"])).kurt(),
    lambda m: m.DataFrame(FRAME).kurt(),
    lambda m: m.Series([1.0]).kurt(axis=1),
    lambda m: m.Series([1.0]).describe(percentiles=[1.5]),
    lambda m: m.Series([1.0]).describe(percentiles=[0.5, 0.5]),
    lambda m: m.Series([1.0]).describe(percentiles=0.3),
    lambda m: m.DataFrame().describe(),
    lambda m: m.DataFrame(FRAME).describe(include="all", exclude=["object"]),
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


def test_a_percentile_that_is_not_a_number_is_a_type_error(firepanda: ModuleType) -> None:
    """pandas' message names numpy's string type, so only the class is compared."""
    with pytest.raises(TypeError):
        firepanda.Series([1.0]).describe(percentiles=["a"])


REFUSED: list[Callable[[Any], Any]] = [
    lambda m: m.Series(FRAME["s"]).describe(),
    lambda m: m.Series(FRAME["b"]).describe(),
    lambda m: m.DataFrame(FRAME)[["s", "b"]].describe(),
    lambda m: m.DataFrame(FRAME).describe(include="all"),
    lambda m: m.DataFrame(FRAME).describe(include=["number"]),
    lambda m: m.DataFrame(FRAME)[["x"]].kurt(axis=1),
]


@pytest.mark.parametrize("build", REFUSED)
def test_what_is_not_written_is_refused(firepanda: ModuleType, build: Callable[[Any], Any]) -> None:
    """A mixed answer pandas holds in an object column, a list of types and a row."""
    with pytest.raises(NotImplementedError):
        build(firepanda)


@pytest.mark.parametrize("owner", ["Series", "DataFrame"])
@pytest.mark.parametrize("name", ["kurt", "kurtosis", "describe"])
def test_the_signature_is_pandas_signature(firepanda: ModuleType, owner: str, name: str) -> None:
    """Parameter for parameter, with the same defaults."""
    import pandas as pd

    ours = inspect.signature(getattr(getattr(firepanda, owner), name)).parameters
    yours = inspect.signature(getattr(getattr(pd, owner), name)).parameters
    assert [(p.name, p.kind) for p in ours.values()] == [(p.name, p.kind) for p in yours.values()]
    for each in ours:
        assert ours[each].default == yours[each].default, each
