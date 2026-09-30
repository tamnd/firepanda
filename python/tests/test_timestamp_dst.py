"""A moment localized onto an hour that repeats or does not exist, as pandas settles it.

On the night the clocks go back an hour happens twice, and on the night they
go forward an hour is skipped. pandas refuses both unless told what to do:
`ambiguous` picks the first or second of a repeated hour or gives NaT, and
`nonexistent` gives NaT or moves a skipped moment to either edge of the gap.
A `Timestamp` answers these as an index of it does, and the refusals read as
pandas' own sentences. Each test here runs the same code on both libraries and
compares what they print.
"""

from __future__ import annotations

import datetime
from typing import Any

import pandas as pd
import pytest

REPEATED = "2024-11-03 01:30"
SKIPPED = "2024-03-10 02:30"
EAST = "US/Eastern"

BUILDS = {
    "first of two": lambda lib: lib.Timestamp(REPEATED).tz_localize(EAST, ambiguous=True),
    "second of two": lambda lib: lib.Timestamp(REPEATED).tz_localize(EAST, ambiguous=False),
    "repeated gap": lambda lib: lib.Timestamp(REPEATED).tz_localize(EAST, ambiguous="NaT"),
    "skipped gap": lambda lib: lib.Timestamp(SKIPPED).tz_localize(EAST, nonexistent="NaT"),
    "forward": lambda lib: lib.Timestamp(SKIPPED).tz_localize(EAST, nonexistent="shift_forward"),
    "backward": lambda lib: lib.Timestamp(SKIPPED).tz_localize(EAST, nonexistent="shift_backward"),
    "by an hour": lambda lib: lib.Timestamp(SKIPPED).tz_localize(
        EAST, nonexistent=datetime.timedelta(hours=1)
    ),
    "back an hour": lambda lib: lib.Timestamp(SKIPPED).tz_localize(
        EAST, nonexistent=datetime.timedelta(hours=-1)
    ),
    "southern": lambda lib: lib.Timestamp("2024-04-07 02:30").tz_localize(
        "Australia/Sydney", ambiguous=True
    ),
    "plain hour": lambda lib: lib.Timestamp("2024-07-01 12:00:00.000000001").tz_localize(EAST),
}


@pytest.mark.parametrize("make", BUILDS.values(), ids=BUILDS.keys())
def test_a_moment_on_a_changing_hour_is_pandas(firepanda: Any, make: Any) -> None:
    assert repr(make(firepanda)) == repr(make(pd))


REFUSED = {
    "repeated": lambda lib: lib.Timestamp(REPEATED).tz_localize(EAST),
    "skipped": lambda lib: lib.Timestamp(SKIPPED).tz_localize(EAST),
    "index repeated": lambda lib: lib.DatetimeIndex([REPEATED]).tz_localize(EAST),
    "column skipped": lambda lib: lib.Series(lib.to_datetime([SKIPPED])).dt.tz_localize(EAST),
}


@pytest.mark.parametrize("make", REFUSED.values(), ids=REFUSED.keys())
def test_an_unsettled_hour_is_refused_in_pandas_words(firepanda: Any, make: Any) -> None:
    with pytest.raises(ValueError) as mine:
        make(firepanda)
    with pytest.raises(ValueError) as theirs:
        make(pd)
    assert str(mine.value) == str(theirs.value)
