"""Tests for `DatetimeIndex`, which is an index with a calendar on it.

Every assertion is against a running pandas, for the reason `test_ewm.py` gives
at more length: what is being measured here is a surface rather than a kernel,
and a surface is signatures, defaults, the class of an exception and what type
comes back, none of which can be written down and trusted.

The calendar arithmetic itself is already measured in `test_dt.py` against the
same kernels through the accessor door, so nothing here re-measures what a
February in a leap year is. What these hold is that the index door and the
accessor door agree, that the answer is an `Index` where pandas gives an `Index`
and a `DatetimeIndex` where pandas gives a `DatetimeIndex`, and that the
arguments pandas declares are the arguments this declares.
"""

from __future__ import annotations

import importlib.util
from types import ModuleType
from typing import Any

import pytest

needs_pandas = pytest.mark.skipif(
    importlib.util.find_spec("pandas") is None, reason="pandas is not installed"
)

STAMPS = [
    "2020-01-01 00:00:00",
    "2020-02-29 13:45:30",
    "2021-06-15 23:59:59",
    "2023-12-31 06:30:00",
]
"""Four instants chosen so that every calendar field has more than one answer.

The second is a leap day, the third is late enough in the day that a floor to a
day moves it, and the fourth is the last day of a year, of a quarter and of a
month at once.
"""

HOLED = ["2020-01-01 00:00:00", None, "2021-06-15 23:59:59"]
"""Three labels with a gap in the middle, since a missing instant has no year."""

NUMBERED = (
    "year",
    "month",
    "day",
    "hour",
    "minute",
    "second",
    "microsecond",
    "nanosecond",
    "quarter",
    "dayofweek",
    "day_of_week",
    "weekday",
    "dayofyear",
    "day_of_year",
    "days_in_month",
    "daysinmonth",
)
"""The sixteen calendar fields that answer a number per label."""

ASKED = (
    "is_leap_year",
    "is_month_start",
    "is_month_end",
    "is_quarter_start",
    "is_quarter_end",
    "is_year_start",
    "is_year_end",
)
"""The seven that answer a yes or a no per label.

Apart for one reason, which is what a missing label answers. pandas hands these
back as a numpy boolean array, which has no missing value, so a label that is
not an instant at all reads as `False` and cannot be told apart from a label
that is an instant and is not the first of its month. firepanda answers a
missing value, as the `dt` accessor already does, and `test_dt.py` owns that
decision. So these share the value test and not the gap test.
"""

FIELDS = NUMBERED + ASKED
"""The twenty three calendar fields that answer one value per label."""


def made(firepanda: ModuleType, values: list[Any] = STAMPS, **kwargs: Any) -> Any:
    """Builds the firepanda index under test."""
    return firepanda.DatetimeIndex(values, **kwargs)


def theirs(values: list[Any] = STAMPS, **kwargs: Any) -> Any:
    """Builds the pandas index the answer is compared against."""
    import pandas as pd

    return pd.DatetimeIndex(values, **kwargs)


def same(mine: Any, them: Any) -> None:
    """Holds that two indexes carry the same labels and the same level name.

    The comparison is through lists rather than through `equals`, because the
    two objects come from different libraries and neither one's `equals` will
    look at the other. A missing label reads as equal to a missing label, which
    `None == None` already does and `nan == nan` would not, and pandas answers
    `NaT` rather than `None` for a missing calendar field, so the check asks
    whether both sides are missing before it asks whether they are equal.

    The level name is only compared when the pandas side carries one. Seven of
    the calendar fields come back from pandas as a numpy array rather than as an
    index, and an array has no name to compare against. This answers an index
    for all twenty three, which is the shape `ASKED` explains.
    """
    import pandas as pd

    got = list(mine.tolist())
    want = list(them.tolist())
    assert len(got) == len(want)
    for a, b in zip(got, want, strict=True):
        if b is None or b is pd.NaT or (isinstance(b, float) and b != b):
            assert a is None
        else:
            assert a == b
    if hasattr(them, "name"):
        assert mine.name == them.name


def instants(mine: Any, them: Any) -> None:
    """Holds that two indexes of instants name the same instants.

    Written out through a format rather than compared as labels, because an
    index of instants reads back as the whole numbers it stores, which is issue
    #348 and is the same on a column as it is here. Going through the format
    also makes the check independent of the resolution each side happens to
    hold, which is what `as_unit` is being asked about two tests below.
    """
    shape = "%Y-%m-%d %H:%M:%S"
    assert list(mine.strftime(shape).tolist()) == list(them.strftime(shape))


@needs_pandas
@pytest.mark.parametrize("field", FIELDS)
def test_every_calendar_field_matches(firepanda: ModuleType, field: str) -> None:
    """The twenty three fields read the same values off the same labels."""
    same(getattr(made(firepanda), field), getattr(theirs(), field))


@needs_pandas
@pytest.mark.parametrize("field", NUMBERED)
def test_a_missing_label_has_no_calendar_field(firepanda: ModuleType, field: str) -> None:
    """A gap in the labels is a gap in every number read off them."""
    same(getattr(made(firepanda, HOLED), field), getattr(theirs(HOLED), field))


@needs_pandas
@pytest.mark.parametrize("field", ASKED)
def test_a_missing_label_answers_neither_yes_nor_no(firepanda: ModuleType, field: str) -> None:
    """The gap test the seven yes or no fields have instead, and why they have it.

    pandas answers `False` for a label that is not an instant, because the array
    it hands back has no missing value to answer with. That reads as a label
    which is an instant and is not the first of its month, which is not what was
    asked. This answers a missing value, and the two rows either side of the gap
    still agree.
    """
    mine = getattr(made(firepanda, HOLED), field).tolist()
    theirs_ = list(getattr(theirs(HOLED), field))
    assert mine[1] is None
    assert bool(theirs_[1]) is False
    assert mine[0] == bool(theirs_[0])
    assert mine[2] == bool(theirs_[2])


@needs_pandas
def test_a_calendar_field_is_an_index_and_not_an_index_of_instants(
    firepanda: ModuleType,
) -> None:
    """The year of an instant is a number, so it comes back as a plain index."""
    got = made(firepanda).year
    assert isinstance(got, firepanda.Index)
    assert not isinstance(got, firepanda.DatetimeIndex)


@needs_pandas
@pytest.mark.parametrize("freq", ["D", "h", "min", "s"])
@pytest.mark.parametrize("kind", ["floor", "ceil", "round"])
def test_the_three_ways_of_moving_to_a_frequency_match(
    firepanda: ModuleType, kind: str, freq: str
) -> None:
    """Rounding the labels answers the same instants pandas answers."""
    instants(getattr(made(firepanda), kind)(freq), getattr(theirs(), kind)(freq))


@needs_pandas
@pytest.mark.parametrize("kind", ["floor", "ceil", "round", "normalize"])
def test_moving_the_labels_gives_back_an_index_of_instants(
    firepanda: ModuleType, kind: str
) -> None:
    """Anything that answers instants answers a `DatetimeIndex`, as pandas does."""
    made_it = made(firepanda)
    got = getattr(made_it, kind)("D") if kind != "normalize" else made_it.normalize()
    assert isinstance(got, firepanda.DatetimeIndex)


@needs_pandas
def test_the_midnights_match(firepanda: ModuleType) -> None:
    """`normalize` takes the time of day off every label."""
    instants(made(firepanda).normalize(), theirs().normalize())


@needs_pandas
def test_whether_the_labels_are_already_midnight_matches(firepanda: ModuleType) -> None:
    """`is_normalized` is False for a column with a time of day in it and True after."""
    assert made(firepanda).is_normalized == theirs().is_normalized
    assert made(firepanda).normalize().is_normalized == theirs().normalize().is_normalized


@needs_pandas
@pytest.mark.parametrize("kind", ["day_name", "month_name"])
def test_the_written_out_names_match(firepanda: ModuleType, kind: str) -> None:
    """The day and the month write out the same words."""
    same(getattr(made(firepanda), kind)(), getattr(theirs(), kind)())


@needs_pandas
def test_a_locale_that_is_not_a_word_is_refused(firepanda: ModuleType) -> None:
    """A locale has to be a string, since it is looked up rather than called."""
    with pytest.raises(TypeError):
        made(firepanda).day_name(7)


@needs_pandas
@pytest.mark.parametrize("shape", ["%Y-%m-%d", "%H:%M:%S", "%Y", "%d/%m/%Y %H:%M"])
def test_the_format_string_matches(firepanda: ModuleType, shape: str) -> None:
    """Writing the labels out through a format gives the same text."""
    same(made(firepanda).strftime(shape), theirs().strftime(shape))


@needs_pandas
def test_a_format_that_is_not_a_word_is_refused(firepanda: ModuleType) -> None:
    """A format is a string and nothing else is read as one."""
    with pytest.raises(TypeError):
        made(firepanda).strftime(7)


@needs_pandas
def test_the_dates_match(firepanda: ModuleType) -> None:
    """`date` drops the time of day and leaves a calendar date.

    Compared as days since the epoch rather than as written out dates, because
    a date column reads back as the whole number it is stored as, which is
    issue #348 and is the same on a column as it is here.
    """
    import datetime

    epoch = datetime.date(1970, 1, 1)
    got = made(firepanda).date.tolist()
    want = [(x - epoch).days for x in theirs().date.tolist()]
    assert got == want


@needs_pandas
@pytest.mark.parametrize("unit", ["s", "ms", "us"])
def test_restating_the_resolution_matches(firepanda: ModuleType, unit: str) -> None:
    """`as_unit` counts the same instants in another resolution."""
    mine = made(firepanda).as_unit(unit)
    assert mine.unit == theirs().as_unit(unit).unit
    instants(mine, theirs().as_unit(unit))


@needs_pandas
def test_the_resolution_arrives_the_way_pandas_reports_it(firepanda: ModuleType) -> None:
    """The unit of a parsed column is the one pandas reads out of the same text."""
    assert made(firepanda).unit == theirs().unit


@needs_pandas
def test_a_cast_that_would_round_is_refused_by_name(firepanda: ModuleType) -> None:
    """`round_ok=False` is declared and is not implemented, so it says so."""
    with pytest.raises(NotImplementedError):
        made(firepanda).as_unit("s", round_ok=False)


@needs_pandas
def test_labels_with_no_clock_report_no_clock(firepanda: ModuleType) -> None:
    """A naive index answers None for its zone, which is what pandas answers."""
    assert made(firepanda).tz is None
    assert theirs().tz is None


@needs_pandas
def test_putting_the_labels_on_a_clock_matches(firepanda: ModuleType) -> None:
    """`tz_localize` keeps the readings and says what they were read against."""
    mine = made(firepanda).tz_localize("UTC")
    assert str(mine.tz) == str(theirs().tz_localize("UTC").tz)
    assert isinstance(mine, firepanda.DatetimeIndex)


@needs_pandas
def test_reading_the_labels_against_another_clock_matches(firepanda: ModuleType) -> None:
    """`tz_convert` keeps the instants and changes what they read as.

    Against a fixed offset rather than against a named zone, because a named
    zone is ahead of UTC by an amount that changes twice a year and firepanda
    has no time zone database to look that up in yet, which is issue #349. The
    two spell the offset differently once it is on, pandas as `UTC+09:00` and
    this as `+09:00`, so what is held here is that the hours agree.
    """
    mine = made(firepanda).tz_localize("UTC").tz_convert("+09:00")
    them = theirs().tz_localize("UTC").tz_convert("+09:00")
    assert mine.tz == "+09:00"
    assert str(them.tz) == "UTC+09:00"
    assert mine.hour.tolist() == list(them.hour)


@needs_pandas
def test_taking_the_clock_off_matches(firepanda: ModuleType) -> None:
    """`tz_localize(None)` keeps the instants and drops what they were read against."""
    mine = made(firepanda).tz_localize("UTC").tz_localize(None)
    assert mine.tz is None
    instants(mine, theirs().tz_localize("UTC").tz_localize(None))


@needs_pandas
@pytest.mark.parametrize("kind", ["tz_localize", "tz_convert"])
def test_a_zone_that_is_not_a_name_is_refused_by_name(firepanda: ModuleType, kind: str) -> None:
    """A zone is a string here, because the kernel reads the zone out of one."""
    import datetime

    with pytest.raises(NotImplementedError):
        getattr(made(firepanda), kind)(datetime.UTC)


@needs_pandas
def test_converting_a_naive_index_is_refused_the_way_pandas_refuses_it(
    firepanda: ModuleType,
) -> None:
    """Reading labels against another clock needs them to be on one first."""
    with pytest.raises(TypeError):
        made(firepanda).tz_convert("UTC")


@needs_pandas
def test_the_class_is_named_what_pandas_names_it(firepanda: ModuleType) -> None:
    """A program that checks the type sees the same word."""
    assert type(made(firepanda)).__name__ == "DatetimeIndex"


@needs_pandas
def test_it_is_an_index_as_well(firepanda: ModuleType) -> None:
    """pandas makes this a kind of index and so does this, which `isinstance` reads."""
    import pandas as pd

    assert isinstance(made(firepanda), firepanda.Index)
    assert isinstance(theirs(), pd.Index)


@needs_pandas
def test_the_index_surface_underneath_still_answers(firepanda: ModuleType) -> None:
    """Everything an index does, an index of instants does, since it is one."""
    mine = made(firepanda)
    them = theirs()
    assert mine.is_unique == them.is_unique
    assert mine.is_monotonic_increasing == them.is_monotonic_increasing
    assert len(mine) == len(them)
    assert mine.shape == them.shape
    assert mine.ndim == them.ndim
    assert mine.nlevels == them.nlevels
    assert mine.empty == them.empty
    assert mine.has_duplicates == them.has_duplicates


@needs_pandas
def test_the_level_name_is_carried_and_reported(firepanda: ModuleType) -> None:
    """A name given to the constructor comes back off the index and off its fields."""
    mine = made(firepanda, name="when")
    assert mine.name == theirs(name="when").name
    assert mine.year.name == theirs(name="when").year.name


@needs_pandas
def test_nothing_at_all_is_an_empty_index(firepanda: ModuleType) -> None:
    """An empty list of labels is an empty index, which is a thing pandas builds.

    Worth its own test because an empty list is not obviously a list of instants.
    There is nothing in it to look at, so it infers as a column of floats, and
    the parser has to decide to read a column with no rows as instants rather
    than complain about the type of the rows it does not have.
    """
    mine = firepanda.DatetimeIndex([])
    assert len(mine) == len(theirs([]))
    assert mine.dtype.startswith("datetime64")
    assert mine.year.tolist() == []


@needs_pandas
def test_labels_that_are_already_instants_are_not_read_again(firepanda: ModuleType) -> None:
    """Building one out of another keeps the resolution rather than reparsing it."""
    first = made(firepanda).as_unit("s")
    again = firepanda.DatetimeIndex(first, name="again")
    assert again.unit == "s"
    assert again.name == "again"
    instants(again, theirs().as_unit("s"))


@needs_pandas
def test_a_column_of_instants_becomes_an_index_of_them(firepanda: ModuleType) -> None:
    """A series that already holds instants is taken as it is."""
    column = firepanda.to_datetime(firepanda.Series(STAMPS))
    instants(firepanda.DatetimeIndex(column), theirs())


@needs_pandas
def test_whole_numbers_are_read_the_way_the_parser_reads_them(firepanda: ModuleType) -> None:
    """The one parser in the library is the one that reads these, as it reads a column."""
    mine = firepanda.DatetimeIndex([0, 1_000_000_000])
    them = firepanda.to_datetime(firepanda.Series([0, 1_000_000_000]))
    assert mine.tolist() == them.tolist()


@needs_pandas
def test_a_column_of_text_is_refused_as_labels(firepanda: ModuleType) -> None:
    """Text that is not an instant is refused by the parser and not by this."""
    with pytest.raises(Exception, match="temporal"):
        firepanda.DatetimeIndex(["not an instant"])


@needs_pandas
@pytest.mark.parametrize(
    "kwargs",
    [
        {"freq": "D"},
        {"tz": "UTC"},
        {"ambiguous": "NaT"},
        {"dayfirst": True},
        {"yearfirst": True},
        {"dtype": "datetime64[ns]"},
        {"copy": True},
    ],
)
def test_the_arguments_with_no_implementation_behind_them_are_refused_by_name(
    firepanda: ModuleType, kwargs: dict[str, Any]
) -> None:
    """A declared argument that is not honoured says so rather than being ignored."""
    with pytest.raises(NotImplementedError):
        firepanda.DatetimeIndex(STAMPS, **kwargs)


@needs_pandas
def test_the_stored_whole_numbers_are_reachable(firepanda: ModuleType) -> None:
    """`asi8` is the labels as the integers they are, in the index's own unit."""
    got = made(firepanda).as_unit("s").asi8
    want = list(theirs().as_unit("s").asi8)
    assert got == want
