"""The members only an index of instants has, checked against pandas.

`shift` and `snap` along fixed and calendar frequencies, the two time of day
indexers, `isocalendar`, `mean`, `std`, `to_julian_date`, `to_pydatetime`,
`time`, `timetz`, `tzinfo` and `resolution`, on a naive index with a missing
label and on one with a clock that crosses a change of clocks.
"""

from __future__ import annotations

import datetime
import importlib.util
import re
import zoneinfo
from collections.abc import Callable
from types import ModuleType
from typing import Any

import pytest

pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("pandas") is None, reason="pandas is not installed"
)

STAMPS = ["2024-01-01 09:30:00", "2024-01-02 12:00:00", "2024-01-06 18:45:10", None]
HOUR = datetime.timedelta(hours=1)
CLOCKED = ["2024-03-10 01:30:00", "2024-07-01 12:00:00"]


def facts(answer: Any) -> Any:
    """What is compared: an index as its type, labels, name and dtype, a frame by column."""
    if hasattr(answer, "columns"):
        return (
            list(answer.columns),
            [facts(answer[column]) for column in answer.columns],
            facts(answer.index),
        )
    if hasattr(answer, "tolist") and hasattr(answer, "__len__"):
        kind = type(answer).__name__
        shown = (kind, str(answer.dtype)) if kind.endswith("Index") else ()
        return (*shown, [plain(value) for value in answer.tolist()], getattr(answer, "name", None))
    if isinstance(answer, list):
        return ([plain(value) for value in answer], None)
    return plain(answer)


def plain(value: Any) -> Any:
    """A value as compared: a gap as None, an instant or a span as text."""
    if value is None or (isinstance(value, float) and value != value):
        return None
    if str(value) in ("NaT", "<NA>"):
        return None
    if type(value).__name__ in ("Timestamp", "Timedelta"):
        return str(value)
    return value.item() if hasattr(value, "item") and not hasattr(value, "__len__") else value


def plain_index(m: ModuleType) -> Any:
    """The naive index with a missing label."""
    return m.DatetimeIndex(STAMPS, name="w")


def clocked(m: ModuleType) -> Any:
    """Two labels on New York's clock, one either side of the spring change."""
    return m.DatetimeIndex(CLOCKED).tz_localize("US/Eastern")


BUILDS: list[Callable[[Any], Any]] = [
    lambda m: plain_index(m).shift(1, freq="h"),
    lambda m: plain_index(m).shift(2, freq="D"),
    lambda m: plain_index(m).shift(-1, freq="2h"),
    lambda m: plain_index(m).shift(1, freq="90s"),
    lambda m: plain_index(m).shift(1, freq="MS"),
    lambda m: plain_index(m).shift(-2, freq="MS"),
    lambda m: plain_index(m).shift(1, freq="ME"),
    lambda m: plain_index(m).shift(1, freq="W-MON"),
    lambda m: plain_index(m).shift(3, freq="B"),
    lambda m: plain_index(m).shift(0, freq="MS"),
    lambda m: clocked(m).shift(2, freq="h"),
    lambda m: plain_index(m).snap("D"),
    lambda m: plain_index(m)[:3].snap("MS"),
    lambda m: plain_index(m)[:3].snap("W-MON"),
    lambda m: plain_index(m)[:3].snap("B"),
    lambda m: plain_index(m).indexer_at_time("12:00"),
    lambda m: plain_index(m).indexer_at_time(datetime.time(9, 30)),
    lambda m: plain_index(m).indexer_at_time("12:00PM"),
    lambda m: plain_index(m).indexer_at_time("0930"),
    lambda m: clocked(m).indexer_at_time("01:30"),
    lambda m: clocked(m).indexer_at_time(datetime.time(6, 30, tzinfo=datetime.UTC)),
    lambda m: clocked(m).indexer_at_time(datetime.time(9, tzinfo=zoneinfo.ZoneInfo("Asia/Tokyo"))),
    lambda m: clocked(m).indexer_at_time(datetime.time(17, 30, tzinfo=datetime.timezone(HOUR))),
    lambda m: plain_index(m).indexer_between_time("09:00", "13:00"),
    lambda m: plain_index(m).indexer_between_time("09:30", "12:00", include_start=False),
    lambda m: plain_index(m).indexer_between_time("09:30", "12:00", include_end=False),
    lambda m: plain_index(m).indexer_between_time("12:00", "10:00"),
    lambda m: plain_index(m).indexer_between_time(datetime.time(12), datetime.time(19)),
    lambda m: plain_index(m).indexer_between_time("0930", "1200"),
    lambda m: plain_index(m).mean(),
    lambda m: plain_index(m).mean(skipna=False),
    lambda m: plain_index(m)[:0].mean(),
    lambda m: clocked(m).mean(),
    lambda m: plain_index(m).std(),
    lambda m: plain_index(m).std(ddof=0),
    lambda m: plain_index(m).std(skipna=False),
    lambda m: plain_index(m)[:1].std(),
    lambda m: plain_index(m).to_julian_date(),
    lambda m: clocked(m).to_julian_date(),
    lambda m: list(plain_index(m).to_pydatetime()),
    lambda m: list(clocked(m).to_pydatetime()),
    lambda m: list(plain_index(m).time),
    lambda m: list(clocked(m).time),
    lambda m: list(clocked(m).timetz),
    lambda m: str(clocked(m).tzinfo),
    lambda m: str(clocked(m).tz_convert("UTC").tzinfo),
    lambda m: plain_index(m).tzinfo,
    lambda m: plain_index(m).resolution,
    lambda m: clocked(m).resolution,
    lambda m: m.DatetimeIndex(["2024-01-01", "2024-01-02"]).resolution,
    lambda m: plain_index(m)[:0].resolution,
    lambda m: plain_index(m)[3:].resolution,
    lambda m: type(plain_index(m)[1:]).__name__,
    lambda m: type(plain_index(m)[[0, 2]]).__name__,
    lambda m: type(plain_index(m)[[True, False, True, False]]).__name__,
]


@pytest.mark.parametrize("build", BUILDS)
def test_the_answer_is_pandas_answer(firepanda: ModuleType, build: Callable[[Any], Any]) -> None:
    """Every member against pandas, positions and arrays compared as lists."""
    import pandas as pd

    assert facts(build(firepanda)) == facts(build(pd))


def test_isocalendar_answers_the_three_columns_on_the_index(firepanda: ModuleType) -> None:
    """The same year, week and day, in unsigned columns where pandas has nullable ones.

    The labels are compared and not the index type, since a frame's index
    answers a plain `Index` in firepanda whatever its labels are.
    """
    import pandas as pd

    mine, theirs = plain_index(firepanda).isocalendar(), plain_index(pd).isocalendar()
    assert list(mine.columns) == list(theirs.columns)
    for column in mine.columns:
        assert [plain(v) for v in mine[column].tolist()] == [
            plain(v) for v in theirs[column].tolist()
        ]
    assert facts(mine.index)[2:] == facts(theirs.index)[2:]


MISTAKES: list[Callable[[Any], Any]] = [
    lambda m: plain_index(m).shift(1),
    lambda m: plain_index(m).snap("S"),
    lambda m: plain_index(m).snap("W-MON"),
    lambda m: plain_index(m).indexer_at_time("12:00", asof=True),
    lambda m: plain_index(m).indexer_at_time(5),
    lambda m: plain_index(m).indexer_at_time(datetime.time(9, tzinfo=datetime.UTC)),
    lambda m: plain_index(m).indexer_between_time("noon", "13:00"),
    lambda m: plain_index(m).indexer_between_time(5, "13:00"),
    lambda m: plain_index(m).indexer_between_time("25:00", "13:00"),
    lambda m: plain_index(m).mean(axis=1),
]


@pytest.mark.parametrize("build", MISTAKES)
def test_mistakes_raise_as_pandas_raises(
    firepanda: ModuleType, build: Callable[[Any], Any]
) -> None:
    """The same error type as pandas, compared by name, and the same words.

    An invalid frequency keeps its first sentence only, since pandas quotes
    its own parser's error after it.
    """
    import pandas as pd

    with pytest.raises(Exception) as expected:
        build(pd)
    words, ending = str(expected.value), "$"
    if words.startswith("Invalid frequency"):
        words, ending = words.split(" Failed")[0], ""
    with pytest.raises(Exception, match="^" + re.escape(words) + ending) as found:
        build(firepanda)
    named = type(found.value).__name__.replace("InvalidArgumentError", "ValueError")
    assert named == type(expected.value).__name__


def test_a_bad_time_of_day_to_look_for_is_a_value_error(firepanda: ModuleType) -> None:
    """pandas raises dateutil's parser error, which is a ValueError."""
    with pytest.raises(ValueError, match=r"^Unknown string format: noon$"):
        plain_index(firepanda).indexer_at_time("noon")
