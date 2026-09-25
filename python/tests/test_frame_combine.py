"""`DataFrame.combine`, `DataFrame.corrwith` and `convert_dtypes`, checked against pandas.

A frame is compared by its columns, labels, column types and values, with
every missing value as None, because a float column here holds a gap where
pandas holds NaN. pandas' nullable types are read as the type here that holds
the same values: `Int64` is `int64`, `Float64` is `float64`, `string` is
`str` and `boolean` is `bool`, and the same for the Arrow backed ones. A
correlation is compared to twelve places, and the gaps are counted apart from
NaN, which `convert_dtypes` turns into gaps.
"""

from __future__ import annotations

import importlib.util
import math
import re
from collections.abc import Callable
from types import ModuleType
from typing import Any

import pytest

pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("pandas") is None, reason="pandas is not installed"
)

NULLABLE = {
    "Int64": "int64",
    "Float64": "float64",
    "string": "str",
    "boolean": "bool",
    "int64[pyarrow]": "int64",
    "double[pyarrow]": "float64",
    "string[pyarrow]": "str",
    "bool[pyarrow]": "bool",
}


def plain(value: Any) -> Any:
    """A value with every missing one as None and floats rounded."""
    if value is None or type(value).__name__ == "NAType" or value != value:
        return None
    if isinstance(value, float):
        return round(value, 12)
    return value


def kind(dtype: Any) -> str:
    """A column type with pandas' nullable names read as the ones here."""
    text = str(dtype)
    return NULLABLE.get(text, "str" if text == "string" else text)


def converted(answer: Any) -> Any:
    """What is compared for `convert_dtypes`, with the gaps counted apart from NaN."""
    if hasattr(answer, "columns"):
        return facts(answer), [gaps(answer[c]) for c in answer.columns]
    return facts(answer), gaps(answer), answer.name


def gaps(column: Any) -> int:
    """How many values are missing rather than NaN."""
    return sum(v is None or type(v).__name__ == "NAType" for v in column.tolist())


def facts(answer: Any) -> Any:
    """What is compared: labels, types and values."""
    if hasattr(answer, "columns"):
        return (
            list(answer.columns),
            answer.index.tolist(),
            [kind(answer[c].dtype) for c in answer.columns],
            [[plain(v) for v in answer[c].tolist()] for c in answer.columns],
        )
    return answer.index.tolist(), kind(answer.dtype), [plain(v) for v in answer.tolist()]


def pair(m: ModuleType) -> tuple[Any, Any]:
    """Two frames that share some labels and one column."""
    left = m.DataFrame({"x": [1.0, 2.0, None], "y": [4.0, 5.0, 6.0]}, index=[0, 1, 2])
    right = m.DataFrame({"x": [10, 20, 30], "z": [7.0, 8.0, 9.0]}, index=[1, 2, 3])
    return left, right


def bigger(p: Any, q: Any) -> Any:
    """The larger of each pair, the other side's where this one is not larger."""
    return p.where(p > q, q)


def plus(p: Any, q: Any) -> Any:
    """Each pair added."""
    return p + q


def whole(m: ModuleType, **columns: Any) -> Any:
    """A frame from its columns."""
    return m.DataFrame(columns)


def numbers(m: ModuleType) -> tuple[Any, Any]:
    """Frames to correlate, with a gap and a text column, on labels that partly agree."""
    left = m.DataFrame(
        {
            "p": [1.0, 2.0, None, 4.0, 5.0],
            "q": [4.0, 1.0, 2.0, 0.5, 3.0],
            "r": ["a", "b", "c", "d", "e"],
        }
    )
    right = m.DataFrame(
        {"p": [2.0, 4.0, 5.0, None, 9.0], "q": [1.0, 3.0, 2.0, 2.0, 0.0], "w": [1.0] * 5},
        index=[0, 1, 2, 3, 7],
    )
    return left, right


def against(m: ModuleType) -> Any:
    """A column to correlate with."""
    return m.Series([1.0, 2.0, 3.0, 5.0, 4.0])


BUILDS: list[Callable[[Any], Any]] = [
    lambda m: pair(m)[0].combine(pair(m)[1], bigger),
    lambda m: pair(m)[0].combine(pair(m)[1], bigger, fill_value=0),
    lambda m: pair(m)[0].combine(pair(m)[1], bigger, overwrite=False),
    lambda m: pair(m)[0].combine(pair(m)[0] * 2, plus),
    lambda m: whole(m, x=[1, 2]).combine(whole(m, x=[3, 4]), plus),
    lambda m: whole(m, x=[True, False]).combine(whole(m, x=[True, True]), lambda p, q: p & q),
    lambda m: whole(m, b=[1, 2], a=[3, 4]).combine(whole(m, c=[1, 2], a=[5, 6]), plus),
    lambda m: m.DataFrame({"b": [1, 2], "a": [3, 4]}, index=[5, 1]).combine(
        m.DataFrame({"a": [5, 6]}, index=[1, 0]), plus
    ),
    lambda m: m.DataFrame({"b": [1, 2], "a": [3, 4]}, index=[5, 1]).combine(
        m.DataFrame({"a": [5, 6]}, index=[5, 1]), plus
    ),
    lambda m: whole(m, a=[1, 2]).combine(whole(m, a=[1.5, 2.0]), plus),
    lambda m: whole(m, a=[1, 2]).combine(whole(m, a=[1.0, 2.0]), plus),
    lambda m: whole(m, a=[1, 2]).combine(whole(m, b=[1, 2]), plus),
    lambda m: whole(m, a=[1, 2]).combine(whole(m, a=[3]), plus),
    lambda m: whole(m, a=[1, 2]).combine(whole(m, a=[3]), plus, fill_value=0),
    lambda m: whole(m, a=[1, 2]).combine(m.DataFrame(), plus),
    lambda m: m.DataFrame().combine(whole(m, a=[1, 2]), plus),
    lambda m: whole(m, a=[1, 2]).combine(whole(m, a=[3, 4]), lambda p, q: p / q),
    lambda m: whole(m, a=[1, 2]).combine(whole(m, a=[4, 4]), lambda p, q: q / 2),
    lambda m: numbers(m)[0][["p", "q"]].corrwith(against(m)),
    lambda m: numbers(m)[0].corrwith(against(m), numeric_only=True),
    lambda m: numbers(m)[0][["p", "q"]].corrwith(against(m), min_periods=4),
    lambda m: numbers(m)[0][["p", "q"]].corrwith(against(m), min_periods=5),
    lambda m: numbers(m)[0][["p", "q"]].corrwith(against(m), method=lambda a, b: 0.5),
    lambda m: numbers(m)[0][["p", "q"]].corrwith(against(m).iloc[::-1]),
    lambda m: numbers(m)[0][["p", "q"]].corrwith(m.Series([1.0] * 5)),
    lambda m: numbers(m)[0][["p", "q"]].iloc[:0].corrwith(against(m)),
    lambda m: numbers(m)[0][["p", "q"]].corrwith(m.Series([1.0, 2.0], index=["p", "q"]), axis=1),
    lambda m: numbers(m)[0][["p", "q"]].corrwith(numbers(m)[1]),
    lambda m: numbers(m)[0][["q", "p"]].corrwith(numbers(m)[1][["p", "q"]]),
    lambda m: numbers(m)[0][["p", "q"]].corrwith(numbers(m)[1], drop=True),
    lambda m: numbers(m)[0].corrwith(numbers(m)[1]),
    lambda m: numbers(m)[0].corrwith(numbers(m)[1], numeric_only=True),
    lambda m: numbers(m)[0][["p", "q"]].corrwith(numbers(m)[1], method=lambda a, b: 0.5),
    lambda m: numbers(m)[0][["p", "q"]].corrwith(numbers(m)[1], axis=1),
    lambda m: numbers(m)[0][["p", "q"]].corrwith(numbers(m)[1], axis="columns", drop=True),
    lambda m: whole(m, p=[1, 2, 3]).corrwith(m.Series([1, 2, 4])),
    lambda m: whole(m, p=[True, False, True]).corrwith(m.Series([1, 2, 4])),
]


@pytest.mark.parametrize("build", BUILDS)
def test_the_answer_is_pandas_answer(firepanda: ModuleType, build: Callable[[Any], Any]) -> None:
    """Labels and columns on either side, casting, gaps and every way to correlate."""
    import pandas as pd

    assert facts(build(firepanda)) == facts(build(pd))


CONVERTS: list[Callable[[Any], Any]] = [
    lambda m: whole(m, i=[1, 2, 3], g=[1.5, 2.0, None], s=["a", None, "c"]).convert_dtypes(),
    lambda m: whole(m, f=[1.0, 2.0, 3.0], b=[True, False, True]).convert_dtypes(),
    lambda m: whole(m, f=[1.0, 2.0, 3.0]).convert_dtypes(convert_integer=False),
    lambda m: whole(m, f=[1.0, math.inf]).convert_dtypes(),
    lambda m: whole(m, f=[1.5, math.nan]).convert_dtypes(),
    lambda m: whole(m, f=[1.5, math.nan]).convert_dtypes(convert_integer=False),
    lambda m: m.Series([1.5, math.nan], index=[4, 2], name="v").convert_dtypes(),
    lambda m: whole(m, f=[1.0, 2.0]).convert_dtypes(dtype_backend="pyarrow"),
    lambda m: m.Series([1.0, 2.0], name="v").convert_dtypes(),
    lambda m: m.Series(["a", "b"]).convert_dtypes(),
]


@pytest.mark.parametrize("build", CONVERTS)
def test_the_conversion_is_pandas_conversion(
    firepanda: ModuleType, build: Callable[[Any], Any]
) -> None:
    """Types, values, and a NaN in a float column turned into a gap."""
    import pandas as pd

    assert converted(build(firepanda)) == converted(build(pd))


MISTAKES: list[Callable[[Any], Any]] = [
    lambda m: pair(m)[0].combine(5, bigger),
    lambda m: pair(m)[0].combine(m.Series([1, 2]), bigger),
    lambda m: pair(m)[0].combine(pair(m)[1], 5),
    lambda m: whole(m, a=[1, 2]).combine(whole(m, a=[3, 4]), lambda p, q: (p + q).tolist()),
    lambda m: whole(m, a=[1, 2]).combine(whole(m, a=[3, 4]), lambda p, q: 7),
    lambda m: numbers(m)[0].corrwith(against(m)),
    lambda m: numbers(m)[0][["p", "q"]].corrwith([1, 2, 3, 4]),
    lambda m: numbers(m)[0][["p", "q"]].corrwith(numbers(m)[1], method="x"),
    lambda m: numbers(m)[0][["p", "q"]].corrwith(against(m), method="x"),
    lambda m: numbers(m)[0][["p", "q"]].corrwith(numbers(m)[1], axis=2),
    lambda m: whole(m, a=[1.0]).convert_dtypes(dtype_backend="x"),
    lambda m: m.Series([1.0]).convert_dtypes(dtype_backend="x"),
]


@pytest.mark.parametrize("build", MISTAKES)
def test_mistakes_raise_as_pandas_raises(
    firepanda: ModuleType, build: Callable[[Any], Any]
) -> None:
    """The same error type and words as pandas."""
    import pandas as pd

    with pytest.raises(Exception) as expected:
        build(pd)
    with pytest.raises(type(expected.value), match="^" + re.escape(str(expected.value)) + "$"):
        build(firepanda)


def test_spearman_against_a_frame_is_spearman_of_each_pair(firepanda: ModuleType) -> None:
    """pandas asks scipy for this, so it is checked against the column's own answer."""
    left, right = numbers(firepanda)
    answer = left[["p", "q"]].corrwith(right, method="spearman", drop=True)
    for name in ["p", "q"]:
        rows = [0, 1, 2, 3]
        expected = left[name].loc[rows].corr(right[name].loc[rows], method="spearman")
        assert answer[name] == pytest.approx(expected)


def test_whole_floats_with_a_gap_stay_floats(firepanda: ModuleType) -> None:
    """pandas makes this its nullable `Int64`; a whole number column here takes no gap."""
    column = firepanda.Series([1.0, None]).convert_dtypes()
    assert str(column.dtype) == "float64"
