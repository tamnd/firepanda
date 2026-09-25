"""Calendar steps in `date_range` and `bdate_range`, checked against pandas.

Business days, weeks on a weekday, and the first or last day, plain or business,
of every month, quarter and year, with multiples, backwards steps, zones,
`inclusive`, `normalize`, and the week mask and holidays of the custom business
day. The labels, their type and the name are compared.
"""

from __future__ import annotations

import datetime
import importlib.util
import re
import warnings
from collections.abc import Callable
from types import ModuleType
from typing import Any

import pytest

pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("pandas") is None, reason="pandas is not installed"
)


def facts(found: Any) -> tuple[Any, ...]:
    """What is compared: the type, the name and the labels as text."""
    return str(found.dtype), found.name, [str(value) for value in found.tolist()]


BUILDS: list[Callable[[Any], Any]] = [
    lambda m: m.date_range(end="2020-03-15", periods=3, freq="ME"),
    lambda m: m.date_range(end="2020-03-15 06:00", periods=3, freq="B"),
    lambda m: m.date_range(start="2020-01-15 12:00", end="2020-03-31", freq="ME"),
    lambda m: m.date_range(start="2020-01-15 12:00", end="2020-04-30", freq="2ME"),
    lambda m: m.date_range(start="2020-01-04 12:00", periods=3, freq="B"),
    lambda m: m.date_range(start="2020-01-04", periods=3, freq="W"),
    lambda m: m.date_range(start="2020-01-04", periods=3, freq="W-WED"),
    lambda m: m.date_range(start="2020-01-08", periods=3, freq="W-WED"),
    lambda m: m.date_range(start="2020-02-04", periods=3, freq="QE"),
    lambda m: m.date_range(start="2020-02-04", periods=3, freq="QE-JAN"),
    lambda m: m.date_range(start="2020-02-04", periods=3, freq="QS-FEB"),
    lambda m: m.date_range(start="2020-02-04", periods=3, freq="QS"),
    lambda m: m.date_range(start="2020-02-04", periods=3, freq="YS"),
    lambda m: m.date_range(start="2020-02-04", periods=3, freq="YE"),
    lambda m: m.date_range(start="2020-02-04", periods=3, freq="YE-JUN"),
    lambda m: m.date_range(start="2020-02-04", periods=3, freq="BYE"),
    lambda m: m.date_range(start="2020-02-04", periods=3, freq="BYS"),
    lambda m: m.date_range(start="2020-02-04", periods=3, freq="BQE"),
    lambda m: m.date_range(start="2020-02-04", periods=3, freq="BQS"),
    lambda m: m.date_range(start="2020-02-04", periods=4, freq="BME"),
    lambda m: m.date_range(start="2020-02-04", periods=4, freq="BMS"),
    lambda m: m.date_range(start="2020-02-04", periods=4, freq="MS"),
    lambda m: m.date_range(start="2020-03-04", end="2020-01-01", freq="-1ME"),
    lambda m: m.date_range(end="2020-03-15", periods=2, freq="-1ME"),
    lambda m: m.date_range(start="2020-03-15", periods=2, freq="-1ME"),
    lambda m: m.date_range(start="2020-03-15", periods=3, freq="-2B"),
    lambda m: m.date_range(start="2020-03-15", periods=3, freq="C"),
    lambda m: m.date_range(start="2020-02-04", periods=3, freq="ME", unit="s"),
    lambda m: m.date_range(start="2020-03-07", periods=3, freq="B", tz="America/New_York"),
    lambda m: m.date_range(start="2020-03-01", periods=3, freq="W", tz="America/New_York"),
    lambda m: m.date_range(start="2020-01-31", end="2020-04-30", freq="ME", inclusive="neither"),
    lambda m: m.date_range(start="2020-01-31", end="2020-04-30", freq="ME", inclusive="left"),
    lambda m: m.date_range(start="2020-01-15", end="2020-04-30", freq="ME", inclusive="right"),
    lambda m: m.date_range(
        start="2020-01-31 10:00", periods=3, freq="ME", normalize=True, name="x"
    ),
    lambda m: m.date_range(start="2020-03-31", end="2020-01-01", freq="ME"),
    lambda m: m.date_range(start="2020-02-04", periods=0, freq="ME"),
    lambda m: m.date_range(start=datetime.date(2020, 2, 4), periods=2, freq="ME"),
    lambda m: m.bdate_range(start="2020-03-13 10:00", periods=3),
    lambda m: m.bdate_range(start="2020-03-13", end="2020-03-24"),
    lambda m: m.bdate_range(start="2020-03-13", periods=3, freq="C", holidays=["2020-03-16"]),
    lambda m: m.bdate_range(start="2020-03-13", periods=3, freq="C", weekmask="Mon Wed"),
    lambda m: m.bdate_range(start="2020-03-13", periods=3, freq="C", weekmask="1010100"),
    lambda m: m.bdate_range(
        start="2020-03-13",
        periods=3,
        freq="C",
        weekmask="Mon Wed",
        holidays=[datetime.date(2020, 3, 16)],
    ),
    lambda m: m.bdate_range(start="2020-03-13", periods=3, freq="h"),
    lambda m: m.bdate_range(start="2020-03-13 10:00", periods=3, freq="h", normalize=False),
    lambda m: m.bdate_range(end="2020-03-15", periods=3, name="b"),
]


@pytest.mark.parametrize("build", BUILDS)
def test_the_answer_is_pandas_answer(firepanda: ModuleType, build: Callable[[Any], Any]) -> None:
    """Every calendar step, and every way of giving the ends."""
    import pandas as pd

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        expected = facts(build(pd))
    assert facts(build(firepanda)) == expected


MISTAKES: list[Callable[[Any], Any]] = [
    lambda m: m.date_range(start="2020-02-04", periods=3, freq="M"),
    lambda m: m.date_range(start="2020-02-04", periods=3, freq="Q"),
    lambda m: m.date_range(start="2020-02-04", periods=3, freq="W-foo"),
    lambda m: m.date_range(start="2020-02-04", periods=3, freq="QE-FOO"),
    lambda m: m.date_range(start="2020-02-04", periods=3, freq="ME-JAN"),
    lambda m: m.date_range(start="2020-02-04", periods=3, freq="1.5ME"),
    lambda m: m.bdate_range(start="2020-03-13", periods=3, freq=None),
    lambda m: m.bdate_range(start="2020-03-13", periods=3, holidays=["2020-03-16"]),
    lambda m: m.bdate_range(start="2020-03-13", periods=3, freq="2C", holidays=["2020-03-17"]),
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


def test_steps_not_counted_yet_are_refused(firepanda: ModuleType) -> None:
    """Semi month, business hour and week of month steps are refused by name."""
    for freq in ("SME", "BH", "WOM-1MON"):
        with pytest.raises(NotImplementedError, match="calendar offset"):
            firepanda.date_range("2020-01-01", periods=3, freq=freq)
