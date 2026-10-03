"""The fields and methods pandas gives its instant, span and period arrays.

Each array asks the index of its own kind and hands the answer back as pandas'
array does: numbers and flags in numpy, with NaN for a gap, other labels as an
array of their own, and a table counted from 0.
"""

from __future__ import annotations

from types import ModuleType

import numpy as np
import pytest


def _instants(firepanda: ModuleType) -> object:
    return firepanda.array(
        firepanda.to_datetime(["2024-01-31 10:15:30", None, "2024-03-01 00:00:00"])
    )


def test_instant_fields(firepanda: ModuleType) -> None:
    """Numbers are floats around a gap, and a flag is False there."""
    values = _instants(firepanda)
    years = values.year
    assert isinstance(years, np.ndarray) and str(years.dtype) == "float64"
    assert years[0] == 2024 and np.isnan(years[1])
    assert values.dayofweek.tolist()[::2] == [2, 4]
    assert values.weekday.tolist()[::2] == [2, 4]
    assert values.days_in_month.tolist()[::2] == [31, 31]
    assert values.is_month_start.tolist() == [False, False, True]
    assert values.is_leap_year.tolist() == [True, False, True]
    assert str(firepanda.array(firepanda.to_datetime(["2024-01-31"])).year.dtype) == "int32"
    assert values.unit == "us"
    assert values.resolution == "second"
    assert values.tz is None and values.freq is None
    assert values.asi8[1] == np.iinfo(np.int64).min
    assert not values.is_normalized


def test_instant_methods_keep_the_kind(firepanda: ModuleType) -> None:
    """Rounding gives instants back, names give text, the calendar gives a table."""
    values = _instants(firepanda)
    floored = values.floor("h")
    assert type(floored).__name__ == "DatetimeArray"
    assert str(floored[0]) == "2024-01-31 10:00:00"
    assert str(values.ceil("h")[0]) == "2024-01-31 11:00:00"
    assert str(values.normalize()[0]) == "2024-01-31 00:00:00"
    assert str(values.as_unit("s").dtype) == "datetime64[s]"
    assert values.day_name().tolist()[0] == "Wednesday"
    assert values.month_name().tolist()[2] == "March"
    assert type(values.to_period("M")).__name__ == "PeriodArray"
    table = values.isocalendar()
    assert table.index.tolist() == [0, 1, 2]
    assert table["week"].tolist()[0] == 5
    assert str(values.tz_localize("UTC").dtype) == "datetime64[us, UTC]"
    assert values.to_pydatetime().dtype == object


def test_span_fields(firepanda: ModuleType) -> None:
    """Days and seconds as pandas splits a span, a negative one counting down from a day."""
    spans = firepanda.array(firepanda.to_timedelta(["1D 2h", None, "-3s"]))
    days = spans.days
    assert days[0] == 1 and np.isnan(days[1]) and days[2] == -1
    assert spans.seconds[2] == 86397
    assert spans.total_seconds()[0] == 93600.0
    assert spans.components.index.tolist() == [0, 1, 2]
    assert spans.to_pytimedelta().dtype == object
    assert str(spans.floor("h")[2]) == "-1 days +23:00:00"
    assert str(spans.as_unit("s").dtype) == "timedelta64[s]"
    assert spans.unit == "us" and spans.inferred_freq is None


def test_period_fields(firepanda: ModuleType) -> None:
    """Calendar numbers in int64, the ends as instants and other frequencies as periods."""
    months = firepanda.array(firepanda.period_range("2024-01", periods=2, freq="M"))
    assert months.year.tolist() == [2024, 2024]
    assert months.day.tolist() == [31, 29]
    assert months.week.tolist() == [5, 9]
    assert months.weekofyear.tolist() == [5, 9]
    assert months.freqstr == "M"
    assert type(months.start_time).__name__ == "DatetimeArray"
    assert str(months.end_time[1]) == "2024-02-29 23:59:59.999999"
    assert str(months.asfreq("D")[0]) == "2024-01-31"
    assert str(months.to_timestamp()[1]) == "2024-02-01 00:00:00"
    assert months.strftime("%Y").tolist() == ["2024", "2024"]


@pytest.mark.parametrize("unit", ["q", "D"])
def test_as_unit_names_the_units_held(firepanda: ModuleType, unit: str) -> None:
    """Only the four units a column is held in, worded as pandas words it."""
    with pytest.raises(ValueError, match=r"^Supported units are 's', 'ms', 'us', 'ns'$"):
        _instants(firepanda).as_unit(unit)
    with pytest.raises(ValueError, match=r"^Supported units are"):
        firepanda.Series(firepanda.to_timedelta(["1D"])).dt.as_unit(unit)
    with pytest.raises(ValueError, match=r"^Supported units are"):
        firepanda.DatetimeIndex(["2024-01-01"]).as_unit(unit)
