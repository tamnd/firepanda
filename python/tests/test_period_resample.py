"""`resample` over rows labelled by periods, pandas' `PeriodIndexResampler`.

Going to a coarser frequency each row joins the bin of the period it falls in,
and the bins run over every period between the first row's and the last's.
Going to a finer one each row lands at the start or the end of its period, as
`convention` says, and the periods between are gaps to fill. A reduction on a
finer or equal frequency is that upsample, and one between frequencies neither
of which divides the other is refused. Each test runs the same code on both
libraries and compares what they print.
"""

from __future__ import annotations

from typing import Any

import pandas as pd
import pytest


def months(lib: Any) -> Any:
    return lib.Series([1, 2, 4], index=lib.PeriodIndex(["2024-01", "2024-02", "2024-04"], freq="M"))


BUILDS = {
    "sum": lambda lib: months(lib).resample("Q").sum(),
    "mean": lambda lib: months(lib).resample("Q").mean(),
    "count": lambda lib: months(lib).resample("Q").count(),
    "empty bins": lambda lib: (
        lib.Series([1, 2], index=lib.PeriodIndex(["2024-01", "2024-12"], freq="M"))
        .resample("Q")
        .sum()
    ),
    "ohlc": lambda lib: months(lib).resample("Q").ohlc(),
    "agg": lambda lib: months(lib).resample("Q").agg(["sum", "max"]),
    "frame": lambda lib: (
        lib.DataFrame({"a": [1, 2, 4], "b": [1.0, 2, 3]}, index=months(lib).index)
        .resample("Q")
        .sum()
    ),
    "days to months": lambda lib: (
        lib.Series(range(40), index=lib.period_range("2024-01-01", periods=40, freq="D"))
        .resample("M")
        .mean()
    ),
    "anchored year": lambda lib: months(lib).resample("Y-JUN").sum(),
    "counted bins": lambda lib: (
        lib.Series(range(6), index=lib.period_range("2024-01", periods=6, freq="M"))
        .resample("2Q")
        .sum()
    ),
    "closed ignored": lambda lib: months(lib).resample("Q", closed="left", label="right").sum(),
    "missing period": lambda lib: (
        lib.Series([1, 2, 3], index=lib.PeriodIndex(["2024-01", "NaT", "2024-03"], freq="M"))
        .resample("Q")
        .sum()
    ),
    "same frequency": lambda lib: months(lib).resample("M").sum(),
    "asfreq": lambda lib: months(lib).resample("W").asfreq(),
    "asfreq end": lambda lib: months(lib).resample("W", convention="end").asfreq(),
    "ffill": lambda lib: months(lib).resample("W").ffill(limit=2),
    "bfill": lambda lib: months(lib).resample("W", convention="e").bfill(),
    "interpolate": lambda lib: months(lib).resample("W").interpolate(),
    "to days": lambda lib: months(lib).resample("D").ffill().head(3),
    "to hours": lambda lib: (
        lib.Series([1, 2], index=lib.period_range("2024-01-01", periods=2, freq="D"))
        .resample("12h")
        .asfreq()
    ),
    "groups": lambda lib: {str(k): int(v) for k, v in months(lib).resample("Q").groups.items()},
}


@pytest.mark.parametrize("make", BUILDS.values(), ids=BUILDS.keys())
def test_period_resample_is_pandas(firepanda: Any, make: Any) -> None:
    assert repr(make(firepanda)) == repr(make(pd))


def test_frequencies_that_do_not_divide_are_refused(firepanda: Any) -> None:
    with pytest.raises(firepanda.errors.IncompatibleFrequency, match="not sub or super periods"):
        months(firepanda).resample("W").sum()


PERIODS = {
    "same base": lambda lib: lib.Period(lib.Period("2024-01-01 00:00", "12h"), freq="12h"),
    "same base count": lambda lib: lib.Period(lib.Period("2024-01-01 05:00", "h"), freq="3h"),
    "range from start": lambda lib: lib.period_range(
        lib.Period("2024-01-01 00:00", "12h"), periods=3, freq="12h"
    ),
    "range to end": lambda lib: lib.period_range(
        end=lib.Period("2024-01-01 00:00", "12h"), periods=3, freq="12h"
    ),
}


@pytest.mark.parametrize("make", PERIODS.values(), ids=PERIODS.keys())
def test_period_of_a_period_is_pandas(firepanda: Any, make: Any) -> None:
    assert repr(make(firepanda)) == repr(make(pd))


def test_period_labels_cross_to_arrow_as_pandas_does(firepanda: Any) -> None:
    import pyarrow as pa

    index = firepanda.PeriodIndex(["2024-01", "NaT", "2024-04"], freq="M")
    ours = pa.array(index)
    theirs = pa.array(pd.PeriodIndex(["2024-01", "NaT", "2024-04"], freq="M").array)
    assert ours.type == theirs.type
    assert ours.storage.to_pylist() == theirs.storage.to_pylist()
