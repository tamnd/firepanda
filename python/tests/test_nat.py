"""`NaT`, the missing moment and span, against pandas.

pandas hands a gap in a column of instants or spans out as `NaT`, one value of
its own class that is a `datetime` which never equals anything, sorts against
nothing, and turns any arithmetic it meets into itself. firepanda keeps the
gap in a validity bitmap and hands it out the same way, and takes `NaT` back in
wherever pandas does.
"""

from __future__ import annotations

import copy
import datetime
import importlib.util
import math
import pickle
from types import ModuleType
from typing import Any

import pytest

needs_pandas = pytest.mark.skipif(
    importlib.util.find_spec("pandas") is None, reason="pandas is not installed"
)


def spelled(value: Any) -> Any:
    """A value named so two libraries' answers compare, with `NaT` and NaN as words."""
    if isinstance(value, list):
        return [spelled(item) for item in value]
    if type(value).__name__ == "NaTType":
        return "NaT"
    if isinstance(value, float) and math.isnan(value):
        return "nan"
    return repr(value)


def outcome(call: Any) -> Any:
    """What a call answers, or the kind of error it raises."""
    try:
        return spelled(call())
    except Exception as error:
        return type(error).__name__


SCALAR: dict[str, Any] = {
    "repr": lambda m: [repr(m.NaT), str(m.NaT), bool(m.NaT)],
    "hash": lambda m: hash(m.NaT),
    "is a datetime": lambda m: isinstance(m.NaT, datetime.datetime),
    "equals": lambda m: [m.NaT == m.NaT, m.NaT != m.NaT, m.NaT == 1, m.NaT == "NaT"],
    "orders": lambda m: [m.NaT < m.NaT, m.NaT >= datetime.datetime(2020, 1, 1)],
    "orders a number": lambda m: m.NaT < 1,
    "orders a date": lambda m: m.NaT < datetime.date(2020, 1, 1),
    "adds": lambda m: [m.NaT + datetime.timedelta(1), m.NaT - m.NaT, m.NaT + 3, 3 - m.NaT],
    "adds a float": lambda m: m.NaT + 1.5,
    "multiplies": lambda m: [m.NaT * 2, m.NaT / 2, m.NaT / m.NaT, m.NaT // 2],
    "divides a span": lambda m: [datetime.timedelta(1) / m.NaT, m.NaT % datetime.timedelta(1)],
    "signs": lambda m: [-m.NaT, +m.NaT],
    "abs": lambda m: abs(m.NaT),
    "fields": lambda m: [m.NaT.year, m.NaT.day, m.NaT.nanosecond, m.NaT.is_leap_year],
    "zone": lambda m: [m.NaT.tz, m.NaT.tzinfo, m.NaT.value],
    "gives itself": lambda m: [m.NaT.floor("D"), m.NaT.date(), m.NaT.replace(year=1)],
    "gives nan": lambda m: [m.NaT.day_name(), m.NaT.weekday(), m.NaT.total_seconds()],
    "strftime": lambda m: m.NaT.strftime("%Y"),
    "timestamp": lambda m: m.NaT.timestamp(),
    "isoformat": lambda m: [m.NaT.isoformat(), f"{m.NaT}"],
    "Timestamp": lambda m: [m.Timestamp(None), m.Timestamp("NaT"), m.Timestamp(math.nan)],
    "Timedelta": lambda m: [m.Timedelta(None), m.Timedelta(""), m.Timedelta("nat")],
    "to_datetime": lambda m: [m.to_datetime(None), m.to_timedelta(None)],
    "isna": lambda m: [m.isna(m.NaT), m.notna(m.NaT)],
}


@needs_pandas
@pytest.mark.parametrize("name", list(SCALAR))
def test_nat_behaves_as_pandas_nat(firepanda: ModuleType, name: str) -> None:
    """The scalar, its operators, its fields and its methods."""
    import pandas as pd

    call = SCALAR[name]
    assert outcome(lambda: call(firepanda)) == outcome(lambda: call(pd))


def test_nat_is_one_value(firepanda: ModuleType) -> None:
    """Every road to `NaT` answers the same object, pickled and copied too."""
    nat = firepanda.NaT
    assert type(nat)() is nat
    assert pickle.loads(pickle.dumps(nat)) is nat
    assert copy.copy(nat) is nat and copy.deepcopy(nat) is nat
    assert firepanda.Timestamp(None) is nat and firepanda.Timedelta(None) is nat


def instants(m: ModuleType) -> Any:
    return m.Series([m.Timestamp("2020-01-02"), None, m.Timestamp("2021-03-04 05:06")], name="w")


def spans(m: ModuleType) -> Any:
    return m.Series([m.Timedelta("1 day"), None, m.Timedelta("2h")])


COLUMN: dict[str, Any] = {
    "tolist": lambda m: instants(m).tolist(),
    "iat": lambda m: [instants(m).iat[1], spans(m).iat[1]],
    "iter": lambda m: list(instants(m)),
    "spans": lambda m: spans(m).tolist(),
    "frame cell": lambda m: m.DataFrame({"a": instants(m)}).iat[1, 0],
    "index": lambda m: m.DatetimeIndex([m.Timestamp("2020-01-02"), None]).tolist(),
    "built with NaT": lambda m: m.Series([m.Timestamp("2020-01-02"), m.NaT]).tolist(),
    "only NaT": lambda m: str(m.Series([m.NaT, m.NaT]).dtype),
    "isna": lambda m: m.Series([m.Timestamp("2020-01-02"), m.NaT]).isna().tolist(),
    "fillna": lambda m: instants(m).fillna(m.NaT).tolist(),
    "where": lambda m: instants(m).where([True, True, False], m.NaT).tolist(),
    "replace": lambda m: instants(m).replace(m.Timestamp("2020-01-02"), m.NaT).tolist(),
    "min of gaps": lambda m: [m.Series([m.NaT, m.NaT]).min(), spans(m).iloc[1:2].max()],
    "index min": lambda m: m.DatetimeIndex([m.NaT, m.NaT]).min(),
    "map": lambda m: instants(m).map(lambda v: v is m.NaT).tolist(),
    "compare": lambda m: [(instants(m) == m.NaT).tolist(), (instants(m) != m.NaT).tolist()],
    "less than": lambda m: (instants(m) < m.NaT).tolist(),
    "minus NaT": lambda m: [str((instants(m) - m.NaT).dtype), (instants(m) - m.NaT).tolist()],
    "span plus NaT": lambda m: [str((spans(m) + m.NaT).dtype), (spans(m) + m.NaT).tolist()],
    "isin": lambda m: instants(m).isin([m.NaT]).tolist(),
    "get_loc": lambda m: m.DatetimeIndex([m.Timestamp("2020-01-02"), m.NaT]).get_loc(m.NaT),
    "contains": lambda m: m.NaT in m.DatetimeIndex([m.Timestamp("2020-01-02"), m.NaT]),
}


@needs_pandas
@pytest.mark.parametrize("name", list(COLUMN))
def test_a_column_gap_is_nat_as_in_pandas(firepanda: ModuleType, name: str) -> None:
    """A gap read out of a column of instants or spans, and `NaT` given back in."""
    import pandas as pd

    call = COLUMN[name]
    assert outcome(lambda: call(firepanda)) == outcome(lambda: call(pd))
