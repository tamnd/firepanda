"""`Series.round` and `DataFrame.round`, checked against pandas.

The core scales, rounds half to even and scales back, all in the column's own
floating point type, which is what numpy does and so what pandas does. Every
test builds the same round in both libraries and compares the types and every
row exactly, because a round that is one place off numpy's is the divergence
this file exists to catch.
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

FLOATS = [0.5, 1.5, 2.5, -0.5, -2.5, 1.005, 2.675, 0.125, 1234.5, 1250.0, None]
WHOLE = [15, 25, -15, -25, 14, 250, 0]


def labelled(m: Any, labels: list[str], values: list[Any], name: str = "n") -> Any:
    """A series on text labels, built the one way both libraries share."""
    return m.DataFrame({"i": labels, name: values}).set_index("i")[name]


FRAME = {"a": [1.26, 2.35, None], "b": ["x", "y", "z"], "c": [15, 25, 35], "d": [True, False, True]}


def same(got: list[Any], want: list[Any]) -> bool:
    """Equal row by row, NaN equal to NaN, and the sign of a zero kept."""

    def missing(value: Any) -> bool:
        return value is None or value != value

    def equal(a: Any, b: Any) -> bool:
        if isinstance(a, float) and isinstance(b, float) and a == 0 and b == 0:
            return math.copysign(1, a) == math.copysign(1, b)
        return a == b

    return len(got) == len(want) and all(
        (missing(a) and missing(b)) or equal(a, b) for a, b in zip(got, want, strict=True)
    )


def printed(dtype: Any) -> str:
    """A pandas type in the words firepanda uses for it."""
    return "string" if str(dtype) == "str" else str(dtype)


def agrees(got: Any, want: Any) -> None:
    """The same types, rows and row labels as pandas, frame or series."""
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


@pytest.mark.parametrize("decimals", [-3, -2, -1, 0, 1, 2, 3, 10, 400, -400])
@pytest.mark.parametrize("dtype", ["float64", "float32"])
def test_a_float_column_rounds_as_numpy_rounds(
    firepanda: ModuleType, dtype: str, decimals: int
) -> None:
    """Half to even after scaling, in the column's own width, overflow and all."""
    import pandas as pd

    build: Callable[[Any], Any] = lambda m: m.Series(FLOATS, name="v").astype(dtype)  # noqa: E731
    agrees(build(firepanda).round(decimals), build(pd).round(decimals))


@pytest.mark.parametrize("decimals", [-2, -1, 0, 2])
@pytest.mark.parametrize(
    "dtype", ["int64", "int32", "int16", "uint8", "uint16", "uint32", "uint64"]
)
def test_an_integer_column_rounds_only_to_tens(
    firepanda: ModuleType, dtype: str, decimals: int
) -> None:
    """Unchanged for places of zero or more, and half to even among the tens."""
    import pandas as pd

    rows = [abs(value) for value in WHOLE] if dtype.startswith("u") else WHOLE
    if dtype == "int16":
        rows = [value for value in rows if abs(value) < 200]
    agrees(
        firepanda.Series(rows).astype(dtype).round(decimals),
        pd.Series(rows).astype(dtype).round(decimals),
    )


BUILDS: list[Callable[[Any], Any]] = [
    lambda m: m.Series(["p", "q"]).round(2),
    lambda m: m.Series([True, False]).round(),
    lambda m: labelled(m, ["x", "y"], [1.25, 2.5]).round(1),
    lambda m: m.Series([1.25, 2.5]).round(True),
    lambda m: m.Series([1.26]).round(1, None),
    lambda m: m.Series([1.26]).round(1, out=None),
    lambda m: round(m.Series([1.26, 2.5])),
    lambda m: round(m.Series([1.26, 2.5]), 1),
    lambda m: m.DataFrame(FRAME).round(),
    lambda m: m.DataFrame(FRAME).round(1),
    lambda m: m.DataFrame(FRAME).round(-1),
    lambda m: m.DataFrame(FRAME).round({"a": 0, "c": -1, "zz": 3}),
    lambda m: m.DataFrame(FRAME).round(labelled(m, ["a"], [1])),
    lambda m: m.DataFrame(FRAME).round({}),
    lambda m: m.DataFrame(FRAME).set_index("b").round(1),
    lambda m: round(m.DataFrame(FRAME), 1),
    lambda m: m.DataFrame({"a": [1.26], "b": [2.35]}).round({"b": 1}),
]


@pytest.mark.parametrize("build", BUILDS)
def test_a_round_is_pandas_round(firepanda: ModuleType, build: Callable[[Any], Any]) -> None:
    """Labels, names, the columns left alone and the spellings numpy sends."""
    import pandas as pd

    agrees(build(firepanda), build(pd))


MISTAKES: list[Callable[[Any], Any]] = [
    lambda m: m.Series([1.26]).round(1.5),
    lambda m: m.Series([1.26]).round("x"),
    lambda m: m.Series([1.26]).round(1, 5),
    lambda m: m.Series([1.26]).round(1, None, None),
    lambda m: m.Series([1.26]).round(1, out=1),
    lambda m: m.Series([1.26]).round(1, foo=1),
    lambda m: m.DataFrame(FRAME).round(1.5),
    lambda m: m.DataFrame(FRAME).round([1]),
    lambda m: m.DataFrame(FRAME).round(True),
    lambda m: m.DataFrame(FRAME).round({"a": 1.5}),
    lambda m: m.DataFrame(FRAME).round({"a": True}),
    lambda m: m.DataFrame(FRAME).round(labelled(m, ["a", "a"], [1, 2])),
    lambda m: m.DataFrame(FRAME).round(1, 5),
    lambda m: m.DataFrame(FRAME).round(1, foo=1),
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


@pytest.mark.parametrize("name", ["round", "__round__"])
@pytest.mark.parametrize("kind", ["Series", "DataFrame"])
def test_the_signature_is_pandas_signature(firepanda: ModuleType, kind: str, name: str) -> None:
    """Parameter for parameter, with the same defaults."""
    import pandas as pd

    mine = inspect.signature(getattr(getattr(firepanda, kind), name)).parameters
    yours = inspect.signature(getattr(getattr(pd, kind), name)).parameters
    assert list(mine) == list(yours)
    for each in mine:
        assert mine[each].default == yours[each].default, each
