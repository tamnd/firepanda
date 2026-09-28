"""A column of spans scaled by a number or divided by a span, compared with pandas.

Each case runs in both libraries. A series is compared by its type, its name,
its labels and its index, an index by its class, type, labels and frequency,
and a mistake by its class name.
"""

from __future__ import annotations

import datetime as dt
from collections.abc import Callable
from types import ModuleType
from typing import Any

import pytest

import firepanda as fp

pd = pytest.importorskip("pandas")
np = pytest.importorskip("numpy")


def spans(lib: ModuleType) -> Any:
    return lib.Series(lib.to_timedelta(["1h", "90min", None, "-3h"]), name="s")


def counts(lib: ModuleType) -> Any:
    return lib.Series([2, 0, 3, 4], name="n")


def zeros(lib: ModuleType) -> Any:
    return lib.Series(lib.to_timedelta(["0h", "0h", "1h", "0h"]))


def hours(lib: ModuleType) -> Any:
    return lib.timedelta_range("1h", periods=3, freq="h")


def seconds(lib: ModuleType, *spelled: str) -> Any:
    return lib.Series(lib.to_timedelta(list(spelled)).as_unit("s"))


def outcome(call: Callable[[ModuleType], Any], lib: ModuleType) -> Any:
    try:
        got = call(lib)
    except Exception as error:
        return type(error).__name__
    if isinstance(got, (fp.Index, pd.Index)):
        return (
            type(got).__name__,
            str(got.dtype),
            str(got.tolist()),
            str(getattr(got, "freq", None)),
        )
    return str(got.dtype), got.name, str(got.tolist()), got.index.tolist()


CASES: list[Callable[[ModuleType], Any]] = [
    lambda lib: spans(lib) * 3,
    lambda lib: 3 * spans(lib),
    lambda lib: spans(lib) * 1.5,
    lambda lib: spans(lib) / 2,
    lambda lib: spans(lib) / 7,
    lambda lib: spans(lib) // 7,
    lambda lib: spans(lib) / 0,
    lambda lib: spans(lib) // 0,
    lambda lib: spans(lib).mul(2),
    lambda lib: seconds(lib, "1s", "3s") / 2,
    lambda lib: spans(lib) * counts(lib),
    lambda lib: counts(lib) * spans(lib),
    lambda lib: spans(lib) / counts(lib),
    lambda lib: spans(lib) // counts(lib),
    lambda lib: spans(lib) * lib.Series([1.0, float("nan"), 2.0, 2.0]),
    lambda lib: spans(lib) * lib.Series([1, 2], index=[1, 7]),
    lambda lib: spans(lib) / lib.Timedelta("1h"),
    lambda lib: spans(lib) // lib.Timedelta("1h"),
    lambda lib: spans(lib).dropna() // lib.Timedelta("1h"),
    lambda lib: spans(lib) % lib.Timedelta("1h"),
    lambda lib: spans(lib) % lib.Timedelta("-1h"),
    lambda lib: lib.Timedelta("1h") / spans(lib),
    lambda lib: lib.Timedelta("1h") // spans(lib),
    lambda lib: spans(lib).truediv(lib.Timedelta("30min")),
    lambda lib: spans(lib) // dt.timedelta(minutes=45),
    lambda lib: spans(lib) / spans(lib),
    lambda lib: spans(lib) // spans(lib),
    lambda lib: spans(lib) % spans(lib),
    lambda lib: spans(lib) / zeros(lib),
    lambda lib: spans(lib) // zeros(lib),
    lambda lib: spans(lib) % zeros(lib),
    lambda lib: spans(lib) // lib.Timedelta(1, unit="ns"),
    lambda lib: spans(lib) % lib.Timedelta(7, unit="ns"),
    lambda lib: seconds(lib, "1s") / seconds(lib, "1ms").dt.as_unit("ms"),
    lambda lib: seconds(lib, "1s") % seconds(lib, "3ms").dt.as_unit("ms"),
    lambda lib: seconds(lib, "1s") % lib.Timedelta("3ms"),
    lambda lib: spans(lib) + np.timedelta64(1, "h"),
    lambda lib: spans(lib) / np.timedelta64(30, "m"),
    lambda lib: spans(lib) + lib.offsets.Day(1),
    lambda lib: spans(lib) - lib.offsets.Hour(2),
    lambda lib: lib.Series(lib.to_datetime(["2024-01-01"])) + np.timedelta64(1, "D"),
    lambda lib: lib.Series(lib.to_datetime(["2024-01-01"])) - np.datetime64("2023-12-31"),
    lambda lib: spans(lib) * True,
    lambda lib: 2 / spans(lib),
    lambda lib: hours(lib) * 2,
    lambda lib: hours(lib) * 1.5,
    lambda lib: hours(lib) / 2,
    lambda lib: hours(lib) / lib.Timedelta("1h"),
    lambda lib: hours(lib) // lib.Timedelta("1h"),
    lambda lib: hours(lib) % lib.Timedelta("1h"),
]


@pytest.mark.parametrize("call", CASES)
def test_span_arithmetic_is_pandas_arithmetic(call: Callable[[ModuleType], Any]) -> None:
    assert outcome(call, fp) == outcome(call, pd)


NUMPY: list[Callable[[ModuleType], Any]] = [
    lambda lib: lib.Timedelta(np.timedelta64(30, "m")),
    lambda lib: lib.Timedelta(np.timedelta64(5, "ns")),
    lambda lib: lib.Timedelta(np.timedelta64("NaT")),
    lambda lib: lib.Timestamp(np.datetime64("2024-01-01T01")),
    lambda lib: lib.Timestamp(np.datetime64("2024-01-01T01:02:03.004")),
]


@pytest.mark.parametrize("call", NUMPY)
def test_a_numpy_scalar_reads_as_pandas_reads_it(call: Callable[[ModuleType], Any]) -> None:
    ours, theirs = call(fp), call(pd)
    assert str(ours) == str(theirs)
    assert getattr(ours, "unit", None) == getattr(theirs, "unit", None)
