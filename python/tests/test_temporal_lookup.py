"""Looking rows up by instants and spans in the labels, checked against pandas.

An index of instants answers a Timestamp, a `datetime`, text for one moment,
and text that names a whole year, quarter, month, day or hour, which pandas
reads as every row inside that period. Slices take the same keys, zoned labels
take naive text on their own clock, and an index of spans answers a Timedelta
or its text. The answers are compared as labels and values.
"""

from __future__ import annotations

import datetime
import importlib.util
import re
from collections.abc import Callable
from types import ModuleType
from typing import Any

import pytest

pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("pandas") is None, reason="pandas is not installed"
)

MOMENTS = ("2020-01-01 00:00", "2020-01-01 12:00", "2020-01-02 06:00", "2020-02-01 00:00")
DAYS = ("2020-01-01", "2020-01-02", "2020-01-02", "2020-01-05")
MIXED = ("2020-01-03 00:00", "2020-01-01 00:00", "2020-01-02 00:00", "2020-01-01 12:00")


def timed(m: ModuleType, texts: tuple[str, ...] = MOMENTS, zone: str | None = None) -> Any:
    """A frame with instants for labels and whole numbers counting up."""
    instants = m.to_datetime(m.Series(list(texts)))
    if zone:
        instants = instants.dt.tz_localize(zone)
    return m.DataFrame({"d": instants, "v": list(range(1, len(texts) + 1))}).set_index("d")


def spans(m: ModuleType) -> Any:
    """A frame with spans for labels."""
    return m.DataFrame(
        {"d": m.to_timedelta(m.Series(["1D", "2D", "3D"])), "v": [1, 2, 3]}
    ).set_index("d")


def answer(value: Any) -> Any:
    """What is compared: labels and values of a frame or column, or the value itself."""
    if hasattr(value, "columns"):
        return [str(v) for v in value.index.tolist()], value["v"].tolist()
    if hasattr(value, "index") and hasattr(value, "tolist"):
        return [str(v) for v in value.index.tolist()], value.tolist()
    if isinstance(value, slice):
        return tuple(None if v is None else int(v) for v in (value.start, value.stop, value.step))
    if hasattr(value, "item") and not hasattr(value, "value"):
        return value.item()
    return str(value) if hasattr(value, "value") else value


BUILDS: list[Callable[[Any], Any]] = [
    lambda m: timed(m).loc[m.Timestamp("2020-01-02 06:00")],
    lambda m: timed(m).loc[datetime.datetime(2020, 1, 1, 12)],
    lambda m: timed(m).loc["2020-01-02 06:00:00"],
    lambda m: timed(m).loc["2020-01-01"],
    lambda m: timed(m).loc["2020-01"],
    lambda m: timed(m).loc["2020"],
    lambda m: timed(m).loc["2020Q1"],
    lambda m: timed(m).loc["2020-01-01 12"],
    lambda m: timed(m).loc["2020-01-01 12:00:00.000"],
    lambda m: timed(m).loc["2020-01-01":"2020-01-02"],
    lambda m: timed(m).loc["2020-01-01":"2020-01-01"],
    lambda m: timed(m).loc["2020-01-02":"2020-01-01"],
    lambda m: timed(m).loc[m.Timestamp("2020-01-01 06:00") : m.Timestamp("2020-01-02 06:00")],
    lambda m: timed(m).loc[m.Timestamp("2020-01-01 12:00"), "v"],
    lambda m: timed(m).at[m.Timestamp("2020-01-01 12:00"), "v"],
    lambda m: timed(m)["v"][m.Timestamp("2020-01-01 12:00")],
    lambda m: timed(m)["v"]["2020-01-01"],
    lambda m: timed(m)["v"][m.Timestamp("2020-01-02 06:00") :],
    lambda m: timed(m)["v"].loc["2020-01"],
    lambda m: timed(m).index[1],
    lambda m: timed(m).index[-1],
    lambda m: timed(m).index.get_loc(m.Timestamp("2020-01-01 12:00")),
    lambda m: timed(m).index.get_loc("2020-01-01"),
    lambda m: timed(m).index.slice_indexer("2020-01-01 06:00", "2020-01-02"),
    lambda m: m.Timestamp("2020-01-01 12:00") in timed(m).index,
    lambda m: m.Timestamp("2020-01-01 13:00") in timed(m).index,
    lambda m: "2020-01" in timed(m).index,
    lambda m: timed(m, zone="Asia/Tokyo").loc["2020-01-01"],
    lambda m: timed(m, zone="Asia/Tokyo").loc[m.Timestamp("2020-01-01 12:00", tz="Asia/Tokyo")],
    lambda m: timed(m, zone="Asia/Tokyo").loc[m.Timestamp("2020-01-01 03:00", tz="UTC")],
    lambda m: timed(m, DAYS).loc["2020-01-02"],
    lambda m: timed(m, DAYS).loc["2020-01-01"],
    lambda m: timed(m, DAYS).loc["2020-01"],
    lambda m: timed(m, DAYS).loc["2020-01-02":],
    lambda m: timed(m, DAYS).loc[:"2020-01-02"],
    lambda m: timed(m, DAYS).loc["2020-01-03":"2020-01-04"],
    lambda m: timed(m, MIXED).loc["2020-01-01"],
    lambda m: timed(m, MIXED).loc[m.Timestamp("2020-01-02")],
    lambda m: spans(m).loc[m.Timedelta("2D")],
    lambda m: spans(m).loc["2 days"],
    lambda m: spans(m).index[0],
    lambda m: spans(m).loc[m.Timedelta("1D") : m.Timedelta("2D")],
    lambda m: m.Timestamp(2020, 1, 1, 6, 30, 15, 7),
]


@pytest.mark.parametrize("build", BUILDS)
def test_the_answer_is_pandas_answer(firepanda: ModuleType, build: Callable[[Any], Any]) -> None:
    """Every kind of key, period, slice, zone and span."""
    import pandas as pd

    assert answer(build(firepanda)) == answer(build(pd))


MISTAKES: list[Callable[[Any], Any]] = [
    lambda m: timed(m).loc[m.Timestamp("2021-01-01")],
    lambda m: timed(m).loc["2019"],
    lambda m: timed(m).loc[5],
    lambda m: timed(m).index[7],
    lambda m: timed(m, DAYS).loc["2020-01-03"],
    lambda m: timed(m, zone="Asia/Tokyo").loc[m.Timestamp("2020-01-01 12:00")],
    lambda m: spans(m).loc[m.Timedelta("5D")],
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
