"""`DatetimeIndex` handed a clock or a type, and `TimedeltaIndex` handed a type.

pandas puts freshly read instants on the clock `tz=` names, as `tz_localize`
would, keeps them when they already carry that clock, and refuses another one.
`dtype=` names the unit, and for instants the clock too. Spans read out of
nothing are counted in seconds. Each test here runs the same code on both
libraries and compares what they print.
"""

from __future__ import annotations

from typing import Any

import pandas as pd
import pytest


def aware(lib: Any) -> Any:
    return lib.date_range("2024-03-10", periods=3, freq="h", tz="UTC")


BUILDS = {
    "tz": lambda lib: lib.DatetimeIndex(["2024-01-01", "2024-01-02"], tz="UTC"),
    "tz named": lambda lib: lib.DatetimeIndex(
        ["2024-01-01 10:00"], tz="America/New_York", name="x"
    ),
    "same clock": lambda lib: lib.DatetimeIndex(aware(lib), tz="UTC"),
    "step kept": lambda lib: lib.DatetimeIndex(lib.date_range("2024", periods=2), tz="UTC"),
    "ambiguous": lambda lib: lib.DatetimeIndex(
        ["2024-11-03 01:30"], tz="America/New_York", ambiguous=True
    ),
    "unit": lambda lib: lib.DatetimeIndex(["2024-01-01"], dtype="datetime64[ns]"),
    "unit and clock": lambda lib: lib.DatetimeIndex(["2024-01-01"], dtype="datetime64[ns, UTC]"),
    "copy": lambda lib: lib.DatetimeIndex(["2024-01-01"], copy=True),
    "spans empty": lambda lib: lib.TimedeltaIndex([]),
    "spans unit": lambda lib: lib.TimedeltaIndex(["1D"], dtype="timedelta64[s]"),
    "spans index unit": lambda lib: lib.TimedeltaIndex(
        lib.TimedeltaIndex(["1D"]), dtype="timedelta64[ms]"
    ),
    "spans copy": lambda lib: lib.TimedeltaIndex(["1D"], copy=False),
}


@pytest.mark.parametrize("make", BUILDS.values(), ids=BUILDS.keys())
def test_the_index_is_pandas(firepanda: Any, make: Any) -> None:
    assert repr(make(firepanda)) == repr(make(pd))


def test_another_clock_is_refused(firepanda: Any) -> None:
    with pytest.raises(TypeError, match="data is already tz-aware UTC, unable to set"):
        firepanda.DatetimeIndex(aware(firepanda), tz="Asia/Tokyo")


def test_no_clock_for_labels_on_one_is_refused(firepanda: Any) -> None:
    with pytest.raises(ValueError, match="incompatible with 'tz=None'"):
        firepanda.DatetimeIndex(aware(firepanda), tz=None)


def test_two_clocks_are_refused(firepanda: Any) -> None:
    with pytest.raises(ValueError, match="cannot supply both a tz and a dtype with a tz"):
        firepanda.DatetimeIndex(["2024-01-01"], tz="UTC", dtype="datetime64[ns, Asia/Tokyo]")


def test_a_type_that_is_not_instants_is_refused(firepanda: Any) -> None:
    with pytest.raises(ValueError, match="Unexpected value for 'dtype'"):
        firepanda.DatetimeIndex(["2024-01-01"], dtype="int64")


def test_a_type_that_is_not_spans_is_refused(firepanda: Any) -> None:
    with pytest.raises(ValueError, match="dtype 'int64' is invalid"):
        firepanda.TimedeltaIndex(["1D"], dtype="int64")
