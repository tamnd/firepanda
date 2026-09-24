"""`date_range`, checked against pandas.

pandas counts from one end in steps of `freq`, or spaces the points evenly when
both ends and a count are given, in the finest unit the ends carry. With a zone
a step of days keeps to the wall clock and every other step to absolute time.
"""

from __future__ import annotations

import datetime
import importlib.util
import inspect
from collections.abc import Callable
from types import ModuleType
from typing import Any

import pytest

pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("pandas") is None, reason="pandas is not installed"
)

ZONE = "America/New_York"


def agrees(got: Any, want: Any) -> None:
    """The same instants, type and name."""
    assert [str(value) for value in got.tolist()] == [str(value) for value in want.tolist()]
    assert str(got.dtype) == str(want.dtype)
    assert got.name == want.name


BUILDS: list[Callable[[Any], Any]] = [
    lambda m: m.date_range("2024-01-01", periods=4),
    lambda m: m.date_range("2024-01-01", "2024-01-02", periods=5),
    lambda m: m.date_range("2024-01-01", "2024-01-01 00:00:10", periods=4),
    lambda m: m.date_range("2024-01-01", "2024-01-02", periods=3, tz=ZONE),
    lambda m: m.date_range("2024-01-01", "2024-01-03", inclusive="neither"),
    lambda m: m.date_range("2024-01-01", periods=3, inclusive="neither"),
    lambda m: m.date_range("2024-01-01", "2024-01-05", freq="2D", inclusive="right"),
    lambda m: m.date_range(end="2024-01-01", periods=3, inclusive="right"),
    lambda m: m.date_range(None, "2024-01-01", 3, "D"),
    lambda m: m.date_range("2024-01-01", periods=0),
    lambda m: m.date_range("2024-01-03", "2024-01-01"),
    lambda m: m.date_range("2024-01-01", periods=3, freq="-1D"),
    lambda m: m.date_range("2024-01-01 10:30", periods=2, normalize=True, name="n"),
    lambda m: m.date_range("2024-01-01 10:30", "2024-01-03 09:00", normalize=True),
    lambda m: m.date_range("2024-01-01", "2024-01-01 00:00:00.000000001", freq="ns"),
    lambda m: m.date_range("2024-01-01", periods=3, unit="s"),
    lambda m: m.date_range("2024-01-01", periods=3, freq="1.5h"),
    lambda m: m.date_range("2024-01-01", periods=3, freq=datetime.timedelta(hours=1)),
    lambda m: m.date_range(datetime.date(2024, 1, 1), periods=2, freq="h"),
    lambda m: m.date_range(datetime.date(2024, 1, 1), periods=2, freq="ms"),
    lambda m: m.date_range(datetime.date(2024, 1, 1), datetime.datetime(2024, 1, 3)),
    lambda m: m.date_range("2024-03-09", periods=6, freq="12h", tz=ZONE),
    lambda m: m.date_range("2024-11-02", periods=3, freq="D", tz=ZONE),
    lambda m: m.date_range("2024-11-02", periods=3, freq="24h", tz=ZONE),
    lambda m: m.date_range("2024-03-09 12:00", "2024-03-11", freq="6h", tz=ZONE, inclusive="left"),
    lambda m: m.date_range("2024-01-01", "2024-01-02 05:00", freq="7h", tz="UTC"),
    lambda m: m.date_range("2024-01-01 00:00-05:00", periods=2, freq="h"),
    lambda m: m.date_range(m.Timestamp("2024-01-01", tz="UTC"), periods=2),
]


@pytest.mark.parametrize("build", BUILDS)
def test_the_answer_is_pandas_answer(firepanda: ModuleType, build: Callable[[Any], Any]) -> None:
    """Each three of four, the inclusive ends, normalising, units and zones."""
    import pandas as pd

    agrees(build(firepanda), build(pd))


def test_a_start_in_seconds_keeps_the_answer_in_seconds(firepanda: ModuleType) -> None:
    """A timestamp read out of a column in seconds carries its unit into the range."""
    start = firepanda.to_datetime(firepanda.Series([86_400]), unit="s").iloc[0]
    answer = firepanda.date_range(start=start, periods=3, freq="D")
    assert str(answer.dtype) == "datetime64[s]"
    assert [str(value) for value in answer.tolist()] == [
        "1970-01-02 00:00:00",
        "1970-01-03 00:00:00",
        "1970-01-04 00:00:00",
    ]


MISTAKES: list[Callable[[Any], Any]] = [
    lambda m: m.date_range("2024-01-01", periods=10.0),
    lambda m: m.date_range("2024-01-01", periods=True),
    lambda m: m.date_range("2024-01-01", periods=-1),
    lambda m: m.date_range(periods=3),
    lambda m: m.date_range("2024-01-01", "2024-01-02", periods=3, freq="h"),
    lambda m: m.date_range("2024-01-01", periods=3, inclusive="x"),
    lambda m: m.date_range("2024-01-01", periods=3, freq="bogus"),
    lambda m: m.date_range("2024-01-01", periods=3, freq="0h"),
    lambda m: m.date_range("2024-01-01", periods=3, unit="D"),
    lambda m: m.date_range("2024-01-01", periods=3, unit="s", freq="ms"),
    lambda m: m.date_range("2024-01-01", periods=3, foo=1),
    lambda m: m.date_range("2024-01-01 00:00-05:00", periods=2, tz="UTC"),
    lambda m: m.date_range("2024-03-10 02:30", periods=3, freq="h", tz=ZONE),
]


@pytest.mark.parametrize("build", MISTAKES)
def test_a_mistake_is_pandas_mistake(firepanda: ModuleType, build: Callable[[Any], Any]) -> None:
    """The same class or a subclass of it, and the same message."""
    import pandas as pd

    with pytest.raises(Exception) as theirs:
        build(pd)
    with pytest.raises(Exception) as mine:
        build(firepanda)
    assert isinstance(mine.value, type(theirs.value))
    assert str(mine.value) == str(theirs.value)


@pytest.mark.parametrize("freq", ["ME", "W", "2QS", "B"])
def test_a_calendar_offset_is_refused(firepanda: ModuleType, freq: str) -> None:
    """Months, weeks, quarters and business days are not all one length."""
    with pytest.raises(NotImplementedError, match="calendar offset"):
        firepanda.date_range("2024-01-01", periods=3, freq=freq)


def test_the_answer_is_a_datetime_index(firepanda: ModuleType) -> None:
    """With the calendar on it, like pandas' answer."""
    answer = firepanda.date_range("2024-01-30", periods=3)
    assert isinstance(answer, firepanda.DatetimeIndex)
    assert answer.day.tolist() == [30, 31, 1]


def test_the_signature_is_pandas_signature(firepanda: ModuleType) -> None:
    """Parameter for parameter, with the same defaults."""
    import pandas as pd

    ours = inspect.signature(firepanda.date_range).parameters
    yours = inspect.signature(pd.date_range).parameters
    assert [(p.name, p.kind, p.default) for p in ours.values()] == [
        (p.name, p.kind, p.default) for p in yours.values()
    ]
