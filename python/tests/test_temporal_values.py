"""Moments, spans and days going into a column and coming back out, checked against pandas.

A temporal column stores whole numbers, a count of units since the epoch for a
moment, a count of units for a span and a count of days for a date. What pandas
hands out for one of those numbers is a `Timestamp`, a `Timedelta` or a
`datetime.date`, from `tolist`, from iterating, from reading one cell and from
the reductions that answer a value of the column's own kind. And a list of
`datetime` or `timedelta` values handed to a constructor is a temporal column
in pandas, at the finest unit any value is quoted at. Every test builds the
same thing in both libraries and compares the values, which compare equal
across the two because both scalars are the standard library's classes
underneath. A missing value is None here and NaT in pandas.
"""

from __future__ import annotations

import datetime
import importlib.util
from collections.abc import Callable
from types import ModuleType
from typing import Any

import pytest

pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("pandas") is None, reason="pandas is not installed"
)

TEXT = ["2024-01-01 10:00:00", None, "2023-06-15 23:59:59", "2024-02-29 00:00:00"]
UTC = datetime.UTC


def moments(m: Any) -> Any:
    """A column of moments with a gap, built the one way both libraries share."""
    return m.to_datetime(m.Series(TEXT))


def same(got: Any, want: Any) -> bool:
    """Equal, with None standing for NaT and NaN, and the zone offset compared too."""

    def missing(value: Any) -> bool:
        return value is None or value != value or str(value) == "NaT"

    if isinstance(want, (list, tuple)):
        return len(got) == len(want) and all(same(a, b) for a, b in zip(got, want, strict=True))
    if missing(want):
        return missing(got)
    zoned = isinstance(want, datetime.datetime) and want.tzinfo is not None
    if zoned and got.utcoffset() != want.utcoffset():
        return False
    temporal = isinstance(want, (datetime.date, datetime.timedelta))
    return got == want and (not temporal or type(got).__name__ == type(want).__name__)


READS: list[Callable[[Any], Any]] = [
    lambda m: moments(m).tolist(),
    lambda m: list(moments(m)),
    lambda m: [value for _, value in moments(m).items()],
    lambda m: moments(m).iloc[0],
    lambda m: moments(m).iloc[-1],
    lambda m: moments(m).loc[2],
    lambda m: moments(m).iat[0],
    lambda m: moments(m).max(),
    lambda m: moments(m).min(),
    lambda m: moments(m).median(),
    lambda m: moments(m).dt.as_unit("s").tolist(),
    lambda m: moments(m).dt.as_unit("ms").tolist(),
    lambda m: moments(m).dt.as_unit("ns").tolist(),
    lambda m: moments(m).dt.tz_localize("UTC").tolist(),
    lambda m: moments(m).dt.tz_localize("UTC").dt.tz_convert("Asia/Tokyo").tolist(),
    lambda m: moments(m).dt.tz_localize("UTC").dt.tz_convert("+05:30").tolist(),
    lambda m: moments(m).dt.tz_localize("UTC").max(),
    lambda m: moments(m).dt.date.tolist(),
    lambda m: (moments(m) - moments(m).min()).tolist(),
    lambda m: (moments(m) - moments(m).min()).max(),
    lambda m: (moments(m) - moments(m).min()).sum(),
    lambda m: (moments(m) - moments(m).min()).iloc[2],
    lambda m: m.DataFrame({"t": moments(m), "v": [1, 2, 3, 4]}).set_index("t").index.tolist(),
    lambda m: list(m.DataFrame({"t": moments(m), "v": [1, 2, 3, 4]}).set_index("t").index),
    lambda m: m.DataFrame({"t": moments(m), "v": [1, 2, 3, 4]}).iloc[0, 0],
    lambda m: m.DataFrame({"t": moments(m), "v": [1, 2, 3, 4]}).at[2, "t"],
    lambda m: [tuple(row) for row in m.DataFrame({"t": moments(m)}).itertuples()],
    lambda m: moments(m).count(),
    lambda m: (moments(m) > datetime.datetime(2024, 1, 1)).tolist(),
    lambda m: (moments(m) == m.Timestamp("2024-02-29")).tolist(),
    lambda m: (moments(m) + datetime.timedelta(days=1, hours=2)).tolist(),
    lambda m: (datetime.datetime(2025, 1, 1) - moments(m)).tolist(),
    lambda m: moments(m).dt.as_unit("ns").sub(moments(m).min()).tolist(),
    lambda m: ((moments(m) - moments(m).min()) > datetime.timedelta(days=200)).tolist(),
]


@pytest.mark.parametrize("read", READS)
def test_a_value_read_out_is_pandas_value(
    firepanda: ModuleType, read: Callable[[Any], Any]
) -> None:
    """Lists, iteration, cells, labels, reductions, zones, units, spans and days."""
    import pandas as pd

    assert same(read(firepanda), read(pd))


def D(*parts: int, **zone: Any) -> datetime.datetime:
    """A plain `datetime`, short to write."""
    return datetime.datetime(*parts, **zone)  # type: ignore[arg-type]


BUILDS: list[Callable[[Any], Any]] = [
    lambda m: m.Series([D(2024, 1, 1), None, D(2023, 5, 6, 7, 8, 9, 10)]),
    lambda m: m.Series([m.Timestamp("2024-01-01"), m.Timestamp("2024-01-01 00:00:00.000000001")]),
    lambda m: m.Series([m.Timestamp("2024-01-01").as_unit("s")], name="t"),
    lambda m: m.Series([D(2024, 1, 1, tzinfo=UTC), D(2024, 6, 1, tzinfo=UTC)]),
    lambda m: m.Series([datetime.timedelta(days=1, seconds=5), None]),
    lambda m: m.Series([m.Timedelta(1, "s"), m.Timedelta(3, "s")]),
    lambda m: m.Series([m.Timedelta(1, "s"), m.Timedelta(1, "ns")]),
    lambda m: m.Series([D(2024, 1, 1), D(2024, 1, 2)], index=["a", "b"]),
    lambda m: m.DataFrame({"a": [1, 2], "t": [D(2024, 1, 1), D(2024, 2, 1)], "s": ["x", "y"]}),
    lambda m: m.DataFrame({"t": [D(2024, 1, 1), None], "d": [datetime.timedelta(1), None]}),
    lambda m: m.DataFrame([{"t": D(2024, 1, 1)}, {"t": D(2024, 3, 1)}]),
    lambda m: m.DataFrame({"t": moments(m), "v": 1}),
]


@pytest.mark.parametrize("build", BUILDS)
def test_a_list_of_moments_is_pandas_column(
    firepanda: ModuleType, build: Callable[[Any], Any]
) -> None:
    """The type pandas infers, the unit and zone in it, and the values back out."""
    import pandas as pd

    mine, theirs = build(firepanda), build(pd)
    if hasattr(theirs, "columns"):
        assert list(mine.columns) == list(theirs.columns)
        for name in theirs.columns:
            assert mine[name].dtype == str(theirs[name].dtype).replace("str", "string"), name
            assert same(mine[name].tolist(), theirs[name].tolist()), name
    else:
        assert mine.dtype == str(theirs.dtype)
        assert mine.name == theirs.name
        assert list(mine.index) == list(theirs.index)
        assert same(mine.tolist(), theirs.tolist())


def test_moments_with_and_without_a_zone_are_refused(firepanda: ModuleType) -> None:
    """pandas refuses the mix, and so does this, with a ValueError."""
    with pytest.raises(ValueError, match="tz-aware"):
        firepanda.Series([D(2024, 1, 1), D(2024, 1, 1, tzinfo=UTC)])


def test_a_count_is_still_a_number(firepanda: ModuleType) -> None:
    """Only a value of the column's own kind is made a moment."""
    column = moments(firepanda)
    assert column.count() == 3
    assert column.dt.year.tolist() == [2024, None, 2023, 2024]
