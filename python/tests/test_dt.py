"""The `dt` accessor, checked against a running pandas.

Same argument as `test_reductions.py` and `test_transformations.py`, and one
thing that is different. Those two build their columns out of Python floats,
because a float column is what `firepanda.Series([...])` makes. There is no
constructor spelling for a timestamp column yet, so every column here comes in
through Arrow, which is the import path document 15 argues is the real one
anyway.

The answers go back out through Arrow as well. They were written that way when
`tolist` on a temporal column handed back the stored integer, which was #348.
`tolist` hands out a `Timestamp` now, and `test_temporal_values.py` checks it,
but reading through `pyarrow.array` still gives the `datetime` a pandas program
would compare against, so these tests were left as they are.

Two differences from pandas are asserted rather than worked around, because both
are decisions.

The first is a missing row on a boolean part. `is_month_end` on a NaT is False in
pandas and missing here, and pandas is that way because its answer is a numpy
bool array, which has nowhere to put a missing value. Arrow has somewhere, so
this says missing.

The second is the five names that are not here at all: `time`, `timetz`,
`to_period`, `to_pydatetime` and `freq`. Every one of them needs a type firepanda
does not have, and none of them is declared and refused, which is the opposite of
what the arguments do. An absent name is honest about being unimplemented and a
declared one that always raises is not, and the line between the two is that a
caller can see a name before they call it and cannot see an argument's fate until
they pass it.
"""

from __future__ import annotations

import datetime as dt
import importlib.util
import inspect
import re
from types import ModuleType
from typing import Any

import pytest

needs = {
    name: pytest.mark.skipif(
        importlib.util.find_spec(name) is None, reason=f"{name} is not installed"
    )
    for name in ("pyarrow", "pandas")
}

both = pytest.mark.skipif(
    importlib.util.find_spec("pyarrow") is None or importlib.util.find_spec("pandas") is None,
    reason="pyarrow and pandas are not both installed",
)

STAMPS = [
    dt.datetime(2024, 1, 1, 0, 0, 0),
    dt.datetime(2024, 2, 29, 13, 45, 30, 123456),
    dt.datetime(2023, 12, 31, 23, 59, 59),
    None,
]
"""Four rows chosen so that no part of the accessor answers the same thing twice.

The first is the start of a month, a quarter and a year at once. The second is a
leap day, which is the only way `is_leap_year` and `days_in_month` disagree with
a fixed table. The third is the end of a month, a quarter and a year, and it is
in a different ISO year from its calendar year, which is the whole reason
`isocalendar` exists. The fourth is missing, which is where every part has to
decide what to do with a row that is not there.
"""

SPANS = [
    dt.timedelta(days=1),
    dt.timedelta(days=2, hours=3, minutes=4, seconds=5),
    dt.timedelta(seconds=-90),
    None,
]
"""Four durations, one of them negative, for the two parts a duration column has."""

PROPERTIES = [
    "year",
    "month",
    "day",
    "hour",
    "minute",
    "second",
    "microsecond",
    "nanosecond",
    "dayofweek",
    "day_of_week",
    "weekday",
    "dayofyear",
    "day_of_year",
    "quarter",
    "days_in_month",
    "daysinmonth",
    "date",
]
"""The parts that answer a number or a date, and take nothing.

The seven boolean ones are not in here, because a missing row makes them
disagree with pandas on purpose and they are checked separately.
"""

FLAGS = [
    "is_leap_year",
    "is_month_start",
    "is_month_end",
    "is_quarter_start",
    "is_quarter_end",
    "is_year_start",
    "is_year_end",
]
"""The parts that answer a yes or a no."""

ABSENT = ["time", "timetz", "to_period", "to_pydatetime", "freq"]
"""The five pandas has and this does not, for the reason the module docstring gives."""


def stamps(firepanda: ModuleType, values: list[Any] = STAMPS, unit: str = "us") -> Any:
    """Builds a timestamp column, which is only spellable through Arrow for now."""
    import pyarrow as pa

    table = pa.table({"t": pa.array(values, type=pa.timestamp(unit))})
    return firepanda.from_arrow(table)["t"]


def zoned(series: Any) -> Any:
    """Puts a zone on a timestamp column, for the tests that need one.

    Both libraries take this spelling, so it is written once here rather than
    twice in the parametrized rows.
    """
    return series.dt.tz_localize("UTC")


def spans(firepanda: ModuleType) -> Any:
    """Builds a duration column, the same way."""
    import pyarrow as pa

    table = pa.table({"d": pa.array(SPANS, type=pa.duration("us"))})
    return firepanda.from_arrow(table)["d"]


def read(series: Any) -> list[Any]:
    """Reads a firepanda column out as Python values, through Arrow.

    The module docstring says why this is not `tolist`.
    """
    import pyarrow as pa

    return list(pa.array(series).to_pylist())


def theirs(values: list[Any] = STAMPS, unit: str = "us") -> Any:
    """The same column in pandas, at the same resolution.

    The resolution has to be said out loud. pandas picks nanoseconds for a list
    of `datetime` objects and firepanda keeps whatever Arrow handed it, and a
    part that reads the calendar gives the same answer either way while `unit`
    does not.
    """
    import pandas as pd

    return pd.Series(pd.array(values, dtype=f"datetime64[{unit}]"))


def absent(one: Any) -> bool:
    """Whether a value is missing, counting every spelling of it as one.

    pandas has four: `NaT` for a timestamp, `nan` for a float, `pd.NA` for the
    nullable integer `isocalendar` answers with, and `None`. firepanda has one.
    """
    import pandas as pd

    return one is None or one is pd.NaT or one is pd.NA or (isinstance(one, float) and one != one)


def like(mine: list[Any], them: list[Any]) -> bool:
    """Whether two lists agree, once every spelling of missing is the same thing."""
    if len(mine) != len(them):
        return False
    return all(
        absent(a) == absent(b) and (absent(a) or a == b) for a, b in zip(mine, them, strict=True)
    )


@both
@pytest.mark.parametrize("name", PROPERTIES)
def test_a_part_gives_the_pandas_answer(firepanda: ModuleType, name: str) -> None:
    """The whole point, once per part that takes nothing and answers a column."""
    mine = read(getattr(stamps(firepanda).dt, name))
    them = list(getattr(theirs().dt, name))
    assert like(mine, them), f"{name}: {mine} against {them}"


@both
@pytest.mark.parametrize("name", FLAGS)
def test_a_flag_gives_the_pandas_answer_where_the_row_is_there(
    firepanda: ModuleType, name: str
) -> None:
    """The three present rows agree, and the missing one is the documented difference.

    pandas answers False for a NaT because its answer is a numpy bool array and
    there is no third value to put in one. This answers missing, and the reason
    is that Arrow has a validity bit and a row that is not there has no month to
    be the end of.
    """
    mine = read(getattr(stamps(firepanda).dt, name))
    them = list(getattr(theirs().dt, name))
    assert mine[:3] == them[:3], f"{name}: {mine} against {them}"
    assert mine[3] is None
    assert them[3] is False


@both
def test_the_aliases_are_the_same_part(firepanda: ModuleType) -> None:
    """pandas spells three of these twice and one of them three times.

    `dayofweek`, `day_of_week` and `weekday` are one field, and `days_in_month`
    and `daysinmonth` are another. They are separate rows in the generator's
    table and one word at the boundary, and this is what says the table did not
    point two of them at different fields.
    """
    column = stamps(firepanda)
    week = read(column.dt.dayofweek)
    assert read(column.dt.day_of_week) == week
    assert read(column.dt.weekday) == week
    assert read(column.dt.days_in_month) == read(column.dt.daysinmonth)
    assert read(column.dt.dayofyear) == read(column.dt.day_of_year)


@both
def test_normalize_moves_every_clock_to_midnight(firepanda: ModuleType) -> None:
    """A method rather than a property, which is why it is not in the table."""
    assert like(read(stamps(firepanda).dt.normalize()), list(theirs().dt.normalize()))


@both
@pytest.mark.parametrize("freq", ["s", "min", "h", "D"])
@pytest.mark.parametrize("name", ["floor", "ceil", "round"])
def test_the_three_roundings_reach_the_kernel(firepanda: ModuleType, name: str, freq: str) -> None:
    """Three directions and four frequencies, because a dropped freq passes at one.

    `round` is the one worth having all four for. Rounding to the second and
    rounding to the day are the same code with a different divisor and the
    second row is far enough past the half way point of a day to tell them
    apart.
    """
    mine = read(getattr(stamps(firepanda).dt, name)(freq))
    them = list(getattr(theirs().dt, name)(freq))
    assert like(mine, them), f"{name}({freq}): {mine} against {them}"


@both
@pytest.mark.parametrize("unit", ["s", "ms", "us"])
def test_as_unit_restates_the_column(firepanda: ModuleType, unit: str) -> None:
    """Casting down loses the sub unit part, and pandas loses the same part.

    Nanoseconds are not in the list because the source is microseconds and
    casting up cannot lose anything, so it would pass whatever the cast did.
    """
    moved = stamps(firepanda).dt.as_unit(unit)
    assert moved.dt.unit == unit
    assert like(read(moved), list(theirs().dt.as_unit(unit)))


@both
@pytest.mark.parametrize("name", ["day_name", "month_name"])
def test_the_names_are_the_english_ones(firepanda: ModuleType, name: str) -> None:
    """No locale, which is the only locale there is."""
    mine = read(getattr(stamps(firepanda).dt, name)())
    them = list(getattr(theirs().dt, name)())
    assert like(mine, them), f"{name}: {mine} against {them}"


@both
@pytest.mark.parametrize("fmt", ["%Y-%m-%d", "%H:%M:%S", "%Y-%m-%dT%H:%M:%S", "%j of %Y", "%%Y"])
def test_strftime_writes_the_rows_out(firepanda: ModuleType, fmt: str) -> None:
    """Including an escaped percent, which is the one directive that is not one."""
    mine = read(stamps(firepanda).dt.strftime(fmt))
    them = list(theirs().dt.strftime(fmt))
    assert like(mine, them), f"{fmt}: {mine} against {them}"


@both
def test_isocalendar_answers_a_frame_of_three_columns(firepanda: ModuleType) -> None:
    """The one part whose answer is a frame, which is why it has its own door.

    The third row is the reason this is not the same as `year` and `dayofweek`.
    The thirty first of December 2023 is in ISO week 52 of ISO year 2023 and its
    calendar year is also 2023, but the first of January 2024 falls in ISO week
    1 of ISO year 2024 while the thirty first of December 2024 falls in ISO week
    1 of ISO year 2025. Two of the three field names are spelled the same as
    ordinary fields and mean something else, which is why the core refuses to
    reach them by name.
    """
    mine = stamps(firepanda).dt.isocalendar()
    them = theirs().dt.isocalendar()
    assert mine.columns == ["year", "week", "day"]
    assert list(them.columns) == ["year", "week", "day"]
    for column in ("year", "week", "day"):
        assert like(read(mine[column]), list(them[column])), column


@both
def test_isocalendar_keeps_the_labels_of_the_column_it_read(firepanda: ModuleType) -> None:
    """A frame built out of three new columns still belongs to the rows it came from."""
    column = stamps(firepanda)
    assert column.dt.isocalendar().index.tolist() == column.index.tolist()


@both
def test_a_naive_column_has_no_zone(firepanda: ModuleType) -> None:
    """`tz` is None rather than the empty string the boundary hands over.

    The two are the same thing and only one of them fits in a Mojo `String`, so
    the Python layer is where the empty one turns back into the absent one.
    """
    assert stamps(firepanda).dt.tz is None
    assert theirs().dt.tz is None


@both
@pytest.mark.parametrize("unit", ["s", "ms", "us", "ns"])
def test_the_unit_is_the_stored_resolution(firepanda: ModuleType, unit: str) -> None:
    """Four resolutions, and the column keeps whichever one Arrow handed it."""
    assert stamps(firepanda, unit=unit).dt.unit == unit
    assert theirs(unit=unit).dt.unit == unit


@both
def test_a_zone_goes_on_and_comes_off(firepanda: ModuleType) -> None:
    """UTC is the one zone that needs no transition table, so it is the one tested.

    Naming a real zone is refused, and that refusal is tested below. Putting UTC
    on keeps every reading and changes what it means, converting moves the
    reading, and taking the clock off keeps the instants. The round trip is the
    check that all three agree with each other as well as with pandas.
    """
    column = stamps(firepanda)
    put = column.dt.tz_localize("UTC")
    assert put.dt.tz == "UTC"
    assert like(read(put), list(theirs().dt.tz_localize("UTC")))
    back = put.dt.tz_localize(None)
    assert back.dt.tz is None
    assert read(back) == read(column)


@both
def test_converting_to_the_same_zone_changes_nothing(firepanda: ModuleType) -> None:
    """A convert is a relabelling, so UTC to UTC has to be the identity."""
    put = stamps(firepanda).dt.tz_localize("UTC")
    assert read(put.dt.tz_convert("UTC")) == read(put)
    assert put.dt.tz_convert("UTC").dt.tz == "UTC"


@both
def test_a_duration_column_answers_its_own_two_parts(firepanda: ModuleType) -> None:
    """`days` and `total_seconds` are the two names that need a duration column.

    pandas puts them on `TimedeltaProperties` and this puts them on the one
    accessor, which is the difference the class docstring writes down. The
    negative row is the one worth having: pandas floors `days` towards minus
    infinity rather than towards zero, so a span of minus ninety seconds is
    minus one day and not zero.
    """
    import pandas as pd

    column = spans(firepanda)
    them = pd.Series(pd.array(SPANS, dtype="timedelta64[us]"))
    assert like(read(column.dt.days), list(them.dt.days))
    assert like(read(column.dt.total_seconds()), list(them.dt.total_seconds()))


@both
@pytest.mark.parametrize("name", ABSENT)
def test_the_five_unwritten_names_are_absent_rather_than_refusing(
    firepanda: ModuleType, name: str
) -> None:
    """pandas has them and this does not, and the difference is deliberate.

    Every one needs a type that does not exist yet: a time of day column for
    `time` and `timetz`, a period for `to_period`, a column of Python objects
    for `to_pydatetime`, and frequency inference over an index for `freq`. A
    name that is declared and always raises reads as a failure on a conformance
    board and a name that is not there reads as unimplemented, and unimplemented
    is what these are.
    """
    assert hasattr(theirs().dt, name)
    assert not hasattr(stamps(firepanda).dt, name)


@both
@pytest.mark.parametrize(
    ("call", "expected"),
    [
        (lambda s: zoned(s).dt.round("h", ambiguous="infer"), "ambiguous="),
        (lambda s: s.dt.as_unit("s", round_ok=False), "round_ok="),
        (lambda s: s.dt.tz_convert(None), r"tz_convert\(None\)"),
        (lambda s: s.dt.floor(1), "freq has to be a string"),
        (lambda s: s.dt.tz_localize(3), "tz has to be a zone name"),
    ],
)
def test_a_declared_argument_that_is_not_written_says_so(
    firepanda: ModuleType, call: Any, expected: str
) -> None:
    """Five refusals, one test each, so none of them can be dropped by a tidy up.

    The argument document 26 makes for the reductions and document 27 repeats
    for the transformations. A parameter that is accepted and ignored is right at
    its default and wrong everywhere else, and where it is wrong is where a real
    program uses it.

    The `infer` row puts a zone on the column before it calls, because pandas
    reads the two daylight saving policies there only when there is one, and a
    naive column comes back rounded with them unread. That is what the test
    below this one is about.
    """
    with pytest.raises(NotImplementedError, match=expected):
        call(stamps(firepanda))


@both
@pytest.mark.parametrize("policy", ["nonexistent", "ambiguous"])
def test_a_naive_column_reads_neither_daylight_saving_policy(
    firepanda: ModuleType, policy: str
) -> None:
    """A column with no zone rounds with whatever was passed for the two policies.

    Measured against pandas rather than reasoned about. There is no daylight
    saving without a zone, so there is nothing for a policy to decide, and pandas
    hands the column straight back with a misspelling in its arguments unread.
    Refusing here would be firepanda turning away input pandas takes, which is
    the one direction of difference this library does not get to have, and it is
    worse than a wrong message because the caller's program stops.
    """
    mine = stamps(firepanda).dt.floor("h", **{policy: "not a policy"})
    assert like(read(mine), list(theirs().dt.floor("h", **{policy: "not a policy"})))


@both
@pytest.mark.parametrize(
    "call",
    [
        lambda s: s.dt.tz_localize("UTC", nonexistent="not a policy"),
        lambda s: zoned(s).dt.floor("h", nonexistent="not a policy"),
        lambda s: zoned(s).dt.ceil("h", nonexistent="not a policy"),
        lambda s: zoned(s).dt.round("h", nonexistent="not a policy"),
    ],
)
def test_a_misspelled_policy_is_a_typo_and_not_a_gap(firepanda: ModuleType, call: Any) -> None:
    """The class says which of the two mistakes this is, and they are different.

    A word pandas takes and firepanda has not written is a schedule, and it comes
    back `NotImplementedError` naming the issue. A word pandas does not take
    either is a typo in the caller's own line, and pandas answers that with a
    `ValueError` listing the words that would have worked, so this does too.
    Handing somebody who misspelled `shift_forward` a `NotImplementedError` sends
    them to read a changelog about a feature they never wanted.

    `ambiguous` has no row here on purpose. pandas does not check that word on a
    column at all, and the reason is written where the check is not.
    """
    with pytest.raises(ValueError, match="nonexistent argument must be one of"):
        call(stamps(firepanda))
    with pytest.raises(ValueError, match="nonexistent argument must be one of"):
        call(theirs())


@both
def test_a_locale_is_refused_rather_than_answered_in_english(firepanda: ModuleType) -> None:
    """The one refusal the core owns rather than the Python layer.

    pandas hands the locale to the C library, so what `day_name(locale="fr_FR")`
    answers depends on which locales the machine has installed. Refusing is the
    only answer that is the same everywhere, and it is written in the kernel
    rather than here because the kernel is what holds the English names.
    """
    with pytest.raises(Exception, match="locale"):
        stamps(firepanda).dt.day_name(locale="fr_FR")
    with pytest.raises(TypeError, match="locale has to be a string"):
        stamps(firepanda).dt.month_name(locale=3)


@both
def test_the_temporal_refusals_share_the_words_pandas_uses(firepanda: ModuleType) -> None:
    """Three messages, asserted against both libraries with one pattern each.

    A user who converts a naive column, localizes a zoned one, or misspells a
    frequency does the same thing next in every case, which is to paste the
    message into a search box and land on the pandas documentation page that
    explains what to do. A message that describes our internals accurately and
    shares no words with that page sends them nowhere, so pandas' sentence goes
    first here and ours follows it.

    Running the same pattern against pandas is the point of `@both`. A test that
    only asserted our string would keep passing on the day pandas reworded
    theirs, which is the day the phrase stops being the one anybody searches for.
    """
    naive = stamps(firepanda)
    with pytest.raises(TypeError, match="tz-naive"):
        naive.dt.tz_convert("UTC")
    with pytest.raises(TypeError, match="Already tz-aware"):
        naive.dt.tz_localize("UTC").dt.tz_localize("UTC")
    with pytest.raises(ValueError, match="Invalid frequency"):
        naive.dt.floor("not a frequency")


@both
def test_naming_a_zone_with_a_transition_table_matches(firepanda: ModuleType) -> None:
    """A rule zone reads its offset out of the zone database, row by row.

    January and July are both in the rows, so a column on one offset for the
    whole year would get one of them wrong.
    """
    import pandas as pd

    values = [dt.datetime(2024, 1, 1, 12), dt.datetime(2024, 7, 1, 12), None]
    mine = stamps(firepanda, values).dt.tz_localize("America/New_York")
    theirs = pd.Series(values, dtype="datetime64[us]").dt.tz_localize("America/New_York")
    assert str(mine.dtype) == str(theirs.dtype)
    assert mine.dt.tz_convert("UTC").dt.hour.tolist()[:2] == [17, 16]
    assert mine.dt.tz_localize(None).dt.hour.tolist()[:2] == [12, 12]


@both
@pytest.mark.parametrize(
    ("reading", "words"),
    [
        (dt.datetime(2024, 3, 10, 2, 30), "is a nonexistent time due to daylight savings time"),
        (dt.datetime(2024, 11, 3, 1, 30), "Cannot infer dst time from 2024-11-03 01:30:00"),
    ],
)
def test_a_reading_the_clock_skipped_or_repeated_is_a_value_error(
    firepanda: ModuleType, reading: dt.datetime, words: str
) -> None:
    """Both are what pandas raises when it is not told what to do with them."""
    with pytest.raises(ValueError, match=words):
        stamps(firepanda, [reading]).dt.tz_localize("America/New_York")


GAP = [
    dt.datetime(2024, 3, 10, 2, 30),
    dt.datetime(2024, 3, 10, 2, 0),
    dt.datetime(2024, 3, 10, 2, 59, 59, 999999),
    dt.datetime(2024, 3, 10, 1, 30),
    None,
]
"""Readings in the hour New York skipped in 2024, one either side of it, and a null."""

FOLD = [dt.datetime(2024, 11, 3, 1, 30), dt.datetime(2024, 11, 3, 0, 30), None]
"""A reading in the hour New York repeated in 2024, one before it, and a null."""


def instants(series: Any) -> list[Any]:
    """The UTC instants a zoned column holds, with the zone taken off."""
    return list(series.dt.tz_convert("UTC").dt.tz_localize(None))


@both
@pytest.mark.parametrize(
    ("zone", "values", "policies"),
    [
        ("America/New_York", GAP, {"nonexistent": "NaT"}),
        ("America/New_York", GAP, {"nonexistent": "shift_forward"}),
        ("America/New_York", GAP, {"nonexistent": "shift_backward"}),
        ("America/New_York", GAP, {"nonexistent": dt.timedelta(hours=1)}),
        ("America/New_York", GAP, {"nonexistent": dt.timedelta(hours=-1)}),
        ("America/New_York", GAP, {"nonexistent": dt.timedelta(hours=2)}),
        (
            "Australia/Lord_Howe",
            [dt.datetime(2024, 10, 6, 2, 15)],
            {"nonexistent": "shift_forward"},
        ),
        (
            "Australia/Lord_Howe",
            [dt.datetime(2024, 10, 6, 2, 15)],
            {"nonexistent": "shift_backward"},
        ),
        ("America/New_York", FOLD, {"ambiguous": "NaT"}),
        ("America/New_York", FOLD, {"ambiguous": True}),
        ("America/New_York", FOLD, {"ambiguous": False}),
        ("America/New_York", FOLD, {"ambiguous": [True, False, True]}),
        ("America/New_York", FOLD, {"ambiguous": [False, True, False]}),
        (
            "America/New_York",
            [*FOLD, dt.datetime(2024, 11, 3, 1, 45)],
            {"ambiguous": [True, True, False, False]},
        ),
        ("Australia/Sydney", [dt.datetime(2024, 4, 7, 2, 30)], {"ambiguous": True}),
        ("Australia/Sydney", [dt.datetime(2024, 4, 7, 2, 30)], {"ambiguous": False}),
        ("Europe/London", [dt.datetime(2024, 10, 27, 1, 30)], {"ambiguous": True}),
    ],
)
def test_a_policy_for_a_skipped_or_repeated_reading_matches(
    firepanda: ModuleType, zone: str, values: list[Any], policies: dict[str, Any]
) -> None:
    """Every word pandas takes for the two policies, against pandas.

    The shifts out of a gap go by the whole hour of the clock rather than by
    where the gap ends, which Lord Howe is here to show, since its clock goes
    forward half an hour and pandas still shifts forward to the hour after it.
    """
    mine = stamps(firepanda, values).dt.tz_localize(zone, **policies)
    them = theirs(values).dt.tz_localize(zone, **policies)
    assert str(mine.dtype) == str(them.dtype)
    assert like(read(mine.dt.tz_convert("UTC").dt.tz_localize(None)), instants(them))


@both
def test_a_policy_holds_when_a_zoned_column_is_rounded(firepanda: ModuleType) -> None:
    """Rounding reads the local clock and puts the answer back on it, where a fold can be."""
    values = [dt.datetime(2024, 11, 3, 5, 40), dt.datetime(2024, 11, 3, 6, 40)]
    for ambiguous in ("NaT", True, False, [True, False]):
        mine = stamps(firepanda, values).dt.tz_localize("UTC").dt.tz_convert("America/New_York")
        them = theirs(values).dt.tz_localize("UTC").dt.tz_convert("America/New_York")
        got = mine.dt.floor("h", ambiguous=ambiguous)
        want = them.dt.floor("h", ambiguous=ambiguous)
        assert like(read(got.dt.tz_convert("UTC").dt.tz_localize(None)), instants(want))


@both
@pytest.mark.parametrize("shift", [dt.timedelta(minutes=20), dt.timedelta(minutes=-20)])
def test_a_shift_that_stays_in_the_skipped_hour_is_refused(
    firepanda: ModuleType, shift: dt.timedelta
) -> None:
    """pandas checks that a shift leaves the hour, and names the shift when it does not."""
    words = f"The provided timedelta will relocalize on a nonexistent time: {shift}"
    with pytest.raises(ValueError, match=re.escape(words)):
        stamps(firepanda, GAP).dt.tz_localize("America/New_York", nonexistent=shift)
    with pytest.raises(ValueError, match=re.escape(words)):
        theirs(GAP).dt.tz_localize("America/New_York", nonexistent=shift)


@both
def test_a_list_of_flags_of_the_wrong_length_is_refused(firepanda: ModuleType) -> None:
    words = "Length of ambiguous bool-array must be the same size as vals"
    with pytest.raises(ValueError, match=words):
        stamps(firepanda, FOLD).dt.tz_localize("America/New_York", ambiguous=[True])
    with pytest.raises(ValueError, match=words):
        theirs(FOLD).dt.tz_localize("America/New_York", ambiguous=[True])


@both
def test_a_zone_the_database_does_not_hold_is_refused(firepanda: ModuleType) -> None:
    with pytest.raises(ValueError, match="No time zone found with key Nowhere/Land"):
        stamps(firepanda).dt.tz_localize("Nowhere/Land")


@both
def test_a_part_a_column_does_not_have_is_a_dtype_error(firepanda: ModuleType) -> None:
    """A calendar field on a duration, and a duration field on a timestamp.

    This is the visible half of the decision to carry both pandas accessors on
    one class. pandas would raise AttributeError for either, because the name is
    not on the accessor it built. Here the name is on the accessor and the
    column's type is what refuses it, which is a different error with the same
    cause in it.
    """
    with pytest.raises(firepanda.errors.DTypeError, match="timedelta"):
        _ = spans(firepanda).dt.year
    with pytest.raises(firepanda.errors.DTypeError):
        _ = stamps(firepanda).dt.days


@needs["pyarrow"]
@pytest.mark.parametrize(
    ("kind", "expected"),
    [
        ("cumquat", "unknown datetime part"),
        ("tz", "does not answer a column"),
        ("unit", "does not answer a column"),
        ("isocalendar", "does not answer a column"),
    ],
)
def test_the_boundary_refuses_a_word_before_it_looks_at_the_column(
    firepanda: ModuleType, kind: str, expected: str
) -> None:
    """Reaching past the Python layer, which is the only way to send a bad word.

    `_inner` is the extension object, so the message arrives with its tag on the
    front rather than as the typed error a caller would see. That is the point:
    a word nobody implements has to come back tagged `value`, because it is a
    value the caller chose, and it would come back tagged `dtype` if the check
    happened inside the handler that turns a kernel complaint into one.
    """
    column = stamps(firepanda)
    with pytest.raises(Exception, match=f"firepanda:value: .*{expected}"):
        column._inner.temporal_part(kind, "")


@needs["pyarrow"]
def test_the_word_door_refuses_a_word_that_answers_a_column(firepanda: ModuleType) -> None:
    """The same check from the other side, which is what makes it a rule.

    Three doors, and each one refuses the words that belong to the other two
    rather than trusting the layer above to keep them apart. Two things
    separated by a convention that nothing checks are separated by a comment.
    """
    column = stamps(firepanda)
    with pytest.raises(Exception, match=r"firepanda:value: .*does not answer a word"):
        column._inner.temporal_word("year")
    with pytest.raises(Exception, match=r"firepanda:value: .*unknown datetime part"):
        column._inner.temporal_word("cumquat")


@needs["pyarrow"]
def test_the_accessor_carries_no_dictionary(firepanda: ModuleType) -> None:
    """Slotted like everything else, and built fresh on every lookup.

    pandas builds one per `.dt` as well. There is nothing to cache, since the
    object is one reference and finding a cached one would cost the lookup that
    building it saves.
    """
    column = stamps(firepanda)
    assert not hasattr(column.dt, "__dict__")
    assert column.dt is not column.dt
    with pytest.raises(AttributeError):
        column.dt.spare = 1


def test_the_accessor_read_off_the_class_is_the_class(firepanda: ModuleType) -> None:
    """`Series.dt` is the accessor class, which is what pandas answers there.

    A property would answer itself here, and a program that reads the accessor's
    members off the class rather than off a column would find a property object
    with none of them on it. The conformance board is one such program, so this
    is the shape that has to hold. The parameter name is part of it, because the
    board compares the accessor's signature with the pandas one and pandas calls
    it `data`.

    The class is not reached by name from the top of the package, because pandas
    does not put its accessor classes there either. Both libraries hand it out
    through this attribute and nowhere else, so this is the only door to check.
    """
    accessor = firepanda.Series.dt
    assert isinstance(accessor, type)
    assert accessor.__name__ == "DatetimeProperties"
    assert callable(accessor)
    assert list(inspect.signature(accessor).parameters) == ["data"]
