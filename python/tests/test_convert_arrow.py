"""`convert_dtypes(dtype_backend="pyarrow")`, compared with pandas type for type.

pandas takes the masked type it would pick and moves it to Arrow when that
type's switch is on, keeping each width. Moments and durations always move,
and categories and columns of objects stay. The values are compared as text,
so a moment or a duration has to come back as pandas' own scalar to print
the same. firepanda spells pandas' `str` as `string`, which is read as `str`.
"""

from __future__ import annotations

import importlib.util
from collections.abc import Callable
from types import ModuleType
from typing import Any

import pytest

import firepanda as fp

pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("pandas") is None or importlib.util.find_spec("pyarrow") is None,
    reason="pandas or pyarrow is not installed",
)


def frame(m: ModuleType) -> Any:
    moments = m.to_datetime(["2020-01-01", None, "2021-01-01"])
    return m.DataFrame(
        {
            "i": [1, 2, 3],
            "f": [1.5, None, 2.0],
            "w": [1.0, None, 3.0],
            "b": [True, False, True],
            "s": ["x", None, "z"],
            "t": moments,
            "d": m.to_timedelta([1, None, 3], unit="s"),
            "c": m.Series(["a", "b", "a"], dtype="category"),
            "z": m.Series(moments).dt.tz_localize("UTC"),
            "o": [1, "a", 2.5],
            "m": m.Series([1, None, 3], dtype="Int64"),
        }
    )


def shown(answer: Any) -> Any:
    def typed(dtype: Any) -> str:
        text = str(dtype)
        return "str" if text == "string" else text

    if hasattr(answer, "columns"):
        return (
            [typed(t) for t in answer.dtypes],
            [[str(v) for v in answer[c].tolist()] for c in answer.columns],
            answer.index.tolist(),
        )
    return typed(answer.dtype), [str(v) for v in answer.tolist()], answer.index.tolist()


SWITCHES: list[dict[str, Any]] = [
    {},
    {"convert_integer": False},
    {"convert_floating": False},
    {"convert_boolean": False},
    {"convert_string": False},
    {"infer_objects": False},
]


@pytest.mark.parametrize("switches", SWITCHES)
def test_a_frame_moves_to_arrow_as_pandas_moves_it(switches: dict[str, Any]) -> None:
    import pandas

    def made(m: ModuleType) -> Any:
        return shown(frame(m).convert_dtypes(dtype_backend="pyarrow", **switches))

    assert made(fp) == made(pandas)


SERIES: list[Callable[[ModuleType], Any]] = [
    lambda m: m.Series([1, 2], dtype="int32"),
    lambda m: m.Series([1, 2], dtype="uint8"),
    lambda m: m.Series([1.5, None], dtype="float32"),
    lambda m: m.Series(["a", "b"], name="x", index=[5, 6]),
    lambda m: m.Series(m.to_datetime(["2020-01-01 00:00:01", None])),
    lambda m: m.Series([True, None]),
    lambda m: m.Series([1, None], dtype="Int8"),
    lambda m: m.Series([1.0, 2.5], dtype="Float32"),
    lambda m: m.Series([True, False], dtype="boolean"),
    lambda m: m.Series([1, 2]).convert_dtypes(dtype_backend="pyarrow"),
    lambda m: m.Series(m.period_range("2020", periods=2, freq="Y")),
    lambda m: m.Series([1.0, 2.0], name="a"),
]


@pytest.mark.parametrize("build", SERIES)
def test_a_series_moves_to_arrow_as_pandas_moves_it(build: Callable[[ModuleType], Any]) -> None:
    import pandas

    def made(m: ModuleType) -> Any:
        return shown(build(m).convert_dtypes(dtype_backend="pyarrow"))

    assert made(fp) == made(pandas)


@pytest.mark.parametrize("backend", ["numpy_nullable", "pyarrow"])
def test_flags_are_whole_numbers_when_flags_are_not_converted(backend: str) -> None:
    import pandas

    def made(m: ModuleType) -> Any:
        column = m.Series([True, False])
        return shown(column.convert_dtypes(convert_boolean=False, dtype_backend=backend))

    assert made(fp) == made(pandas)


def test_moments_sit_next_to_a_column_of_objects() -> None:
    import pandas

    def made(m: ModuleType) -> Any:
        return shown(m.DataFrame({"t": m.to_datetime(["2020-01-01", None]), "o": [1, "a"]}))

    assert made(fp) == made(pandas)
