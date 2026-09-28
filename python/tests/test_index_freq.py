"""The frequency an index of instants or spans holds, compared with pandas.

Each case builds the same index in both libraries and compares `freq`, `freqstr`,
the class and the repr. A mistake is compared by its class and the start of its
message.
"""

from __future__ import annotations

import datetime as dt
from collections.abc import Callable
from types import ModuleType
from typing import Any

import pytest

import firepanda as fp

pd = pytest.importorskip("pandas")


def days(lib: ModuleType) -> Any:
    return lib.date_range("2024-01-01", periods=5, freq="D", name="t")


def hours(lib: ModuleType) -> Any:
    return lib.timedelta_range("1h", periods=4, freq="h")


def outcome(call: Callable[[ModuleType], Any], lib: ModuleType) -> Any:
    got = call(lib)
    return repr(got.freq), got.freqstr, type(got).__name__, repr(got)


CASES: list[Callable[[ModuleType], Any]] = [
    days,
    lambda lib: days(lib)[1:3],
    lambda lib: days(lib)[::2],
    lambda lib: days(lib)[::-1],
    lambda lib: days(lib)[[0, 1]],
    lambda lib: days(lib)[:0],
    lambda lib: days(lib).copy(),
    lambda lib: days(lib).rename("z"),
    lambda lib: days(lib).as_unit("s"),
    lambda lib: days(lib).shift(1),
    lambda lib: days(lib).shift(2, freq="h"),
    lambda lib: days(lib).shift(1, freq="ME"),
    lambda lib: lib.Index(days(lib)),
    lambda lib: lib.DatetimeIndex(days(lib)),
    lambda lib: lib.DatetimeIndex(days(lib), freq=None),
    lambda lib: lib.date_range("2024-01-01", "2024-01-03", periods=3),
    lambda lib: lib.date_range("2024-01-01", periods=3, freq=dt.timedelta(hours=1)),
    lambda lib: lib.date_range("2024-01-01", periods=3, freq=dt.timedelta(days=2)),
    lambda lib: lib.date_range("2024-01-01", periods=3, freq="2ME"),
    lambda lib: lib.date_range("2024-01-01", periods=3, freq="W"),
    lambda lib: lib.date_range("2024-01-01", periods=3, freq=lib.offsets.MonthBegin(2)),
    lambda lib: lib.date_range("2024-01-01", periods=3, freq="h", tz="UTC"),
    lambda lib: lib.date_range("2024-01-01", "2024-01-05", inclusive="neither"),
    lambda lib: lib.date_range("2024-01-31", periods=3, freq="ME")[::-1],
    lambda lib: lib.bdate_range("2024-01-01", periods=3),
    lambda lib: lib.DatetimeIndex(["2024-01-01", "2024-01-02", "2024-01-03"], freq="infer"),
    lambda lib: lib.DatetimeIndex(["2024-01-01", "2024-01-02"], freq="infer"),
    lambda lib: lib.DatetimeIndex(["2024-01-01", "2024-01-02"], freq="D"),
    hours,
    lambda lib: hours(lib)[::2],
    lambda lib: hours(lib).as_unit("s"),
    lambda lib: lib.timedelta_range("1D", periods=3),
    lambda lib: lib.timedelta_range("1D", "3D", periods=3),
    lambda lib: lib.TimedeltaIndex(["1h", "2h", "3h"], freq="infer"),
]


@pytest.mark.parametrize("call", CASES)
def test_the_frequency_is_pandas_frequency(call: Callable[[ModuleType], Any]) -> None:
    assert outcome(call, fp) == outcome(call, pd)


def test_setting_the_frequency_is_checked_as_in_pandas() -> None:
    for lib in (fp, pd):
        index = lib.DatetimeIndex(["2024-01-01", "2024-01-02"])
        index.freq = "D"
        assert index.freqstr == "D"
        index.freq = None
        assert index.freq is None


MISTAKES: list[Callable[[ModuleType], Any]] = [
    lambda lib: lib.DatetimeIndex(["2024-01-01", "2024-01-03"], freq="D"),
    lambda lib: setattr(lib.date_range("2024-01-01", periods=3), "freq", "h"),
    lambda lib: lib.TimedeltaIndex(["1h", "3h"], freq="h"),
    lambda lib: lib.DatetimeIndex(["2024-01-01"], freq="zz"),
    lambda lib: lib.DatetimeIndex(["2024-01-01"]).shift(1),
]


@pytest.mark.parametrize("call", MISTAKES)
def test_the_mistakes_are_pandas_mistakes(call: Callable[[ModuleType], Any]) -> None:
    with pytest.raises(Exception) as theirs:
        call(pd)
    with pytest.raises(Exception) as ours:
        call(fp)
    # pandas' errors that firepanda defines again match by name rather than by class.
    assert isinstance(ours.value, type(theirs.value)) or (
        type(ours.value).__name__ == type(theirs.value).__name__
    )
    assert str(theirs.value).startswith(str(ours.value))
