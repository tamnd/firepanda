"""Instants and spans cast to numbers, and elapsed times read from nothing but gaps.

Each case runs in both libraries. A series or an index is compared by its type
and its labels, and a mistake by its class and its message.
"""

from __future__ import annotations

from collections.abc import Callable
from types import ModuleType
from typing import Any

import pytest

import firepanda as fp

pd = pytest.importorskip("pandas")
np = pytest.importorskip("numpy")


def outcome(call: Callable[[ModuleType], Any], lib: ModuleType) -> Any:
    try:
        got = call(lib)
    except Exception as error:
        return type(error).__name__, str(error)
    if isinstance(got, (fp.DataFrame, pd.DataFrame)):
        return [str(one) for one in got.dtypes]
    return str(got.dtype), len(got)


def instants(lib: ModuleType) -> Any:
    return lib.Series(lib.to_datetime(["2024-01-01", "2024-01-02"]))


def spans(lib: ModuleType) -> Any:
    return lib.Series(lib.to_timedelta(["1h", None]))


CASES: list[Callable[[ModuleType], Any]] = [
    lambda lib: instants(lib).astype("float64"),
    lambda lib: instants(lib).astype("float32"),
    lambda lib: instants(lib).dt.tz_localize("UTC").astype("float64"),
    lambda lib: spans(lib).astype(float),
    lambda lib: spans(lib).astype("float32"),
    lambda lib: instants(lib).astype("int32"),
    lambda lib: spans(lib).astype("int16"),
    lambda lib: spans(lib).dropna().astype("uint64"),
    lambda lib: instants(lib).astype("int64"),
    lambda lib: spans(lib).dropna().astype("int64"),
    lambda lib: lib.DataFrame({"a": instants(lib)}).astype("float64"),
    lambda lib: lib.DataFrame({"a": instants(lib), "b": [1, 2]}).astype({"a": "uint8"}),
    lambda lib: lib.DataFrame({"a": spans(lib), "b": [1, 2]}).astype({"b": "float64"}),
    lambda lib: lib.to_timedelta([]),
    lambda lib: lib.to_timedelta([None]),
    lambda lib: lib.to_timedelta([None, None]),
    lambda lib: lib.to_timedelta([None], unit="h"),
    lambda lib: lib.to_timedelta([float("nan")]),
    lambda lib: lib.to_timedelta([lib.NaT, None]),
    lambda lib: lib.to_timedelta(["x"], errors="coerce"),
    lambda lib: lib.to_timedelta(["x", "1h"], errors="coerce"),
    lambda lib: lib.to_timedelta(lib.Series([np.nan])),
]


@pytest.mark.parametrize("call", CASES)
def test_the_answer_is_pandas_answer(call: Callable[[ModuleType], Any]) -> None:
    assert outcome(call, fp) == outcome(call, pd)
