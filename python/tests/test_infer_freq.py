"""`infer_freq`, `inferred_freq` and `dt.freq`, compared with pandas.

Each case builds the same labels in both libraries, as text read back into a
`DatetimeIndex` so that no frequency is carried along, and asks both for the
frequency the labels keep. A mistake is compared by its message, with
firepanda's error allowed to be a subclass of pandas' own.
"""

from __future__ import annotations

from collections.abc import Callable
from types import ModuleType
from typing import Any

import pytest

import firepanda as fp

pd = pytest.importorskip("pandas")


def labels(lib: ModuleType, start: str, count: int, freq: str, tz: str | None = None) -> Any:
    """Labels at a fixed frequency, with the frequency forgotten."""
    moments = [str(one) for one in pd.date_range(start, periods=count, freq=freq, tz=tz)]
    if tz is None:
        return lib.DatetimeIndex(moments)
    return lib.to_datetime(moments, utc=True).tz_convert(tz)


RULES = [
    ("2024-01-01", 5, "D", None),
    ("2024-01-01", 5, "3D", None),
    ("2024-01-01", 5, "h", None),
    ("2024-01-01", 5, "2h", None),
    ("2024-01-01", 5, "15min", None),
    ("2024-01-01", 5, "s", None),
    ("2024-01-01", 5, "10ms", None),
    ("2024-01-03", 5, "W-WED", None),
    ("2024-01-01", 5, "2W", None),
    ("2024-01-01", 5, "ME", None),
    ("2024-01-01", 5, "MS", None),
    ("2024-01-01", 6, "BME", None),
    ("2024-01-01", 6, "BMS", None),
    ("2024-01-01", 5, "QE", None),
    ("2024-02-01", 5, "QS-FEB", None),
    ("2024-01-01", 4, "YE", None),
    ("2024-01-01", 4, "YS-MAR", None),
    ("2024-01-01", 4, "2YS", None),
    ("2024-01-01", 10, "B", None),
    ("2024-01-01", 5, "WOM-2TUE", None),
    ("2024-03-01", 40, "D", "America/New_York"),
    ("2024-03-09", 40, "h", "America/New_York"),
]


@pytest.mark.parametrize(("start", "count", "freq", "tz"), RULES)
def test_the_frequency_is_pandas_frequency(
    start: str, count: int, freq: str, tz: str | None
) -> None:
    ours, theirs = labels(fp, start, count, freq, tz), labels(pd, start, count, freq, tz)
    assert fp.infer_freq(ours) == pd.infer_freq(theirs)
    assert ours.inferred_freq == theirs.inferred_freq
    assert fp.Series(ours).dt.freq == pd.Series(theirs).dt.freq


def outcome(call: Callable[[ModuleType], Any], lib: ModuleType) -> Any:
    try:
        return call(lib)
    except Exception as error:
        return type(error), str(error)


CASES: list[Callable[[ModuleType], Any]] = [
    lambda lib: lib.infer_freq(lib.DatetimeIndex(["2024-01-01", "2024-01-02", "2024-01-05"])),
    lambda lib: lib.infer_freq(lib.DatetimeIndex(["2024-01-03", "2024-01-02", "2024-01-01"])),
    lambda lib: lib.infer_freq(lib.DatetimeIndex(["2024-01-01", "2024-01-01", "2024-01-02"])),
    lambda lib: lib.infer_freq(lib.DatetimeIndex(["2024-01-01", None, "2024-01-03"])),
    lambda lib: lib.infer_freq(["2024-01-01", "2024-01-02", "2024-01-03"]),
    lambda lib: lib.infer_freq(lib.timedelta_range("1D", periods=4, freq="D")),
    lambda lib: lib.infer_freq(lib.timedelta_range("1h", periods=4, freq="90min")),
    lambda lib: lib.infer_freq(lib.timedelta_range("0D", periods=4, freq="7D")),
    lambda lib: lib.timedelta_range("1h", periods=4, freq="h").inferred_freq,
    lambda lib: lib.infer_freq(
        lib.DatetimeIndex(
            ["2024-01-05 15:00", "2024-01-05 16:00", "2024-01-08 09:00", "2024-01-08 10:00"]
        )
    ),
    lambda lib: lib.infer_freq(
        lib.Series(lib.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03"])).dt.as_unit("s")
    ),
    lambda lib: lib.DatetimeIndex(["2024-01-01", "2024-01-02"]).inferred_freq,
    lambda lib: lib.Series(lib.to_datetime(["2024-01-01", "2024-01-02"])).dt.freq,
]


@pytest.mark.parametrize("call", CASES)
def test_the_answer_is_pandas_answer(call: Callable[[ModuleType], Any]) -> None:
    assert outcome(call, fp) == outcome(call, pd)


MISTAKES: list[Callable[[ModuleType], Any]] = [
    lambda lib: lib.infer_freq(lib.DatetimeIndex(["2024-01-01", "2024-01-02"])),
    lambda lib: lib.infer_freq(lib.Index([1, 2, 3])),
    lambda lib: lib.infer_freq(lib.Series([1, 2, 3])),
    lambda lib: lib.infer_freq(lib.Series(["a", "b", "c"])),
]


@pytest.mark.parametrize("call", MISTAKES)
def test_the_mistakes_are_pandas_mistakes(call: Callable[[ModuleType], Any]) -> None:
    ours, theirs = outcome(call, fp), outcome(call, pd)
    assert isinstance(ours[0], type) and issubclass(ours[0], theirs[0])
    assert ours[1] == theirs[1]
