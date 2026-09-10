"""`Timestamp` and `Timedelta`, checked against a running pandas.

Every answer here is compared against the pandas answer rather than against a
number somebody typed out, for the reason the other differential tests give: a
number typed out records what the author believed pandas does, and the whole
point of these two classes is what pandas actually does. Several of the
behaviours below were written the other way round first and were wrong, and the
comparison is what said so.

Two of the comparisons are worth naming here because they are not obvious. The
first is `test_the_scalar_and_the_column_round_the_same_way`, which the module
docstring of `firepanda/_scalars.py` promises by name. There are two frequency
parsers in firepanda, one in Mojo for columns and one in Python for scalars, and
nothing about the arrangement makes them agree. That test is what makes them
agree, so if it is ever deleted the drift starts that day.

The second is the exception comparison. It asks whether the class firepanda
raises is the class pandas raises, and it asks it through the standard library
ancestor rather than by name, because firepanda's errors are named for what went
wrong and pandas' are named for pandas. `InvalidArgumentError` and `ValueError`
are the same answer to a caller writing `except ValueError`, and that is the
question the tests ask. The message is deliberately not compared; message
bucketing is spec 13's subject and it is not settled.

Three differences from pandas are asserted rather than worked around. `NaT` does
not exist in firepanda, so the two constructors refuse the input pandas turns
into it. A moment beyond the year 9999 is out of range here and is not in pandas,
because pandas' `Timestamp` is not really a `datetime` and this one is. And
`to_julian_date` hands back a plain float where pandas hands back a
`numpy.float64`, which is the same decision as `Series.dtype` being a string.
"""

from __future__ import annotations

import datetime as dt
import importlib.util
import warnings
from types import ModuleType
from typing import Any

import pytest

needs_pandas = pytest.mark.skipif(
    importlib.util.find_spec("pandas") is None, reason="pandas is not installed"
)

both = pytest.mark.skipif(
    importlib.util.find_spec("pyarrow") is None or importlib.util.find_spec("pandas") is None,
    reason="pyarrow and pandas are not both installed",
)

MOMENTS = [
    "2026-09-05 13:45:06.123456789",
    "2026-09-05",
    "1970-01-01",
    "2020-02-29 23:59:59.999999999",
    "1677-09-22",
    "2262-04-10",
    "2024-01-01 00:00:00",
    "1000-01-01 12:00:00",
]
"""Eight moments chosen so that no two of them exercise the same thing.

The first has nanoseconds, which is the digit `datetime` cannot hold and the
reason the class exists. The second and the seventh are midnight, where the
printed form drops its fractional part entirely. The third is the epoch, which
is the only value whose count is zero. The fourth is a leap day one nanosecond
before midnight, so every carry in the calendar happens at once. The fifth and
the sixth are within days of the range a nanosecond count reaches, which is
where the bounds checks live. The eighth is before the year 1000, which is
outside the nanosecond range entirely and inside the microsecond one, and it is
also the only one whose year prints in three digits.
"""

SPANS = [
    "1 days 02:03:04.000005006",
    "-1 days +02:03:04.000005006",
    "0 days 00:00:00",
    "3 days",
    "-02:03:04",
    "00:00:00.000000001",
    "-1 days +23:59:59.999999999",
]
"""Seven elapsed times, four of them negative, which is where the split is hard.

A negative span floors to a negative day and a positive remainder, so the last
one is minus a single nanosecond written the way pandas writes it, and the
second is minus a day plus two hours rather than minus the whole reading. Both
spellings round trip through the printed form and both are here because getting
one right does not get the other right.
"""

TEXT_SPANS = [
    "1D",
    "1 D",
    "1 day",
    "1 days",
    "2W",
    "1h",
    "1 hr",
    "30min",
    "30m",
    "1 sec",
    "5ms",
    "7us",
    "3ns",
    "1 nanos",
    "1h30min",
    "1 h 30 min",
    "1D2h",
    "2W3D4h5min6s7ms8us9ns",
    "-1D",
    "+1D",
    "- 1D",
    " 1 D ",
    "1 day, 2:03:04",
    "1 day, 2:03:04.5",
    "2:03:04",
    "0D",
    "0ns",
    "0NS",
    "1000ns",
    "1000NS",
    "1.5D",
    "1.5s",
    "1.5us",
    "1.25us",
    "1.0us",
    "1.0009us",
    "1.5ns",
    "2.5ns",
    "-2.5ns",
    "0.0000001s",
    "0.000001s",
    "1.000000000s",
    "0.5ms",
    "1D 500ns",
    "1 day 2 day",
]
"""The count and unit form, one entry per thing that decides an answer.

The first group is one spelling of each unit. The second is what happens when
several of them are written together, with and without spaces. The third is the
signs, which pandas only allows in the leading position. The fourth is the forms
that overlap with the printed shape, since a parser that tries the printed shape
first has to hand the rest over rather than refuse them.

The rest are fractions, which is where every surprise in this form lives.
`1.0009us` rounds its fraction up to a whole nanosecond and `1.5ns` truncates
its own, `1.000000000s` is quoted in nanoseconds because of the digits it wrote
rather than the value they came to, and `0NS` is quoted in microseconds where
`0ns` is quoted in nanoseconds, which is a wart in the deprecated spelling.
"""

FREQUENCIES = ["D", "h", "min", "s", "ms", "us", "ns", "2h", "15min", "-2h", "3D"]
"""The fixed frequencies with and without a count, and one with a sign."""

UNITS = ["s", "ms", "us", "ns"]
"""The four resolutions an Arrow temporal column can be quoted at."""


def standard(cls: type) -> str:
    """Names the standard library exception a firepanda or pandas error is one of.

    Returns:
        The name of the first class in the method resolution order that came out
        of `builtins`, which is the class a caller writes in an `except`.
    """
    for base in cls.__mro__:
        if base.__module__ == "builtins":
            return base.__name__
    return cls.__name__


def answer(call: Any) -> Any:
    """Runs something and reports either its value or the kind of its failure.

    Returns:
        The repr of the value, or the name of the standard library exception
        class it raised. The message is not in it, on purpose.
    """
    try:
        return repr(call())
    except Exception as raised:
        return f"raised {standard(type(raised))}"


def agree(mine: Any, theirs: Any, note: str = "") -> None:
    """Asserts that firepanda and pandas gave the same answer or failed the same way."""
    left, right = answer(mine), answer(theirs)
    # The class names differ between the two libraries and the repr carries the
    # module for firepanda's, so a firepanda Timestamp reads `Timestamp(...)`
    # once this is off it. What is left is the value.
    left = left.replace("firepanda.", "")
    assert left == right, f"{note}: firepanda said {left}, pandas said {right}"


@needs_pandas
@pytest.mark.parametrize("text", MOMENTS)
@pytest.mark.parametrize(
    "name",
    [
        "value",
        "unit",
        "nanosecond",
        "tz",
        "tzinfo",
        "dayofweek",
        "day_of_week",
        "dayofyear",
        "day_of_year",
        "days_in_month",
        "daysinmonth",
        "quarter",
        "week",
        "weekofyear",
        "is_leap_year",
        "is_month_start",
        "is_month_end",
        "is_quarter_start",
        "is_quarter_end",
        "is_year_start",
        "is_year_end",
        "year",
        "month",
        "day",
        "hour",
        "minute",
        "second",
        "microsecond",
        "resolution",
    ],
)
def test_every_timestamp_property_agrees_with_pandas(
    firepanda: ModuleType, text: str, name: str
) -> None:
    """The parts a moment breaks into, one row per part per moment.

    `resolution` is in here rather than with the class attributes because it is
    not one. pandas answers it two ways, one nanosecond off the class and one
    unit off a value, and a moment read out of text is quoted in microseconds,
    so the answer for most of these is a microsecond.
    """
    import pandas as pd

    agree(
        lambda: getattr(firepanda.Timestamp(text), name),
        lambda: getattr(pd.Timestamp(text), name),
        f"Timestamp({text!r}).{name}",
    )


@needs_pandas
@pytest.mark.parametrize("text", MOMENTS)
@pytest.mark.parametrize(
    "name",
    [
        "day_name",
        "month_name",
        "normalize",
        "to_pydatetime",
        "isoformat",
        "timestamp",
        "__str__",
        "__repr__",
        "__hash__",
        "date",
        "time",
        "timetz",
        "toordinal",
        "weekday",
        "isoweekday",
        "ctime",
        "timetuple",
        "utctimetuple",
        "isocalendar",
        "utcoffset",
        "dst",
        "tzname",
    ],
)
@pytest.mark.filterwarnings("ignore:Discarding nonzero nanoseconds")
def test_every_timestamp_method_that_takes_nothing_agrees_with_pandas(
    firepanda: ModuleType, text: str, name: str
) -> None:
    """The methods with no arguments, including the ones inherited from `datetime`.

    The inherited ones are here because inheriting is not the same as agreeing.
    `timestamp` is the clearest case: the one `datetime` provides reads a naive
    value as local time and pandas reads it as UTC, so the inherited answer is
    wrong by however many hours the machine happens to be from Greenwich.

    The warning that is filtered out is one both libraries raise, in the same
    words, when `to_pydatetime` throws the nanoseconds away. It is filtered here
    rather than fixed because raising it is the correct behaviour and there is a
    test below that checks it is raised.
    """
    import pandas as pd

    agree(
        lambda: getattr(firepanda.Timestamp(text), name)(),
        lambda: getattr(pd.Timestamp(text), name)(),
        f"Timestamp({text!r}).{name}()",
    )


@needs_pandas
@pytest.mark.parametrize("text", MOMENTS)
def test_the_hash_of_a_moment_matches_the_datetime_it_equals(
    firepanda: ModuleType, text: str
) -> None:
    """Equal things hash the same, which is a rule and not a preference.

    A moment on a whole microsecond is equal to the plain `datetime` holding it,
    so it has to hash like one or a dictionary keyed on datetimes quietly grows
    two entries for one key. A moment with nanoseconds in it is equal to no
    `datetime` at all, so it is free to hash on its own count, and it does.
    """
    import pandas as pd

    mine, theirs = firepanda.Timestamp(text), pd.Timestamp(text)
    assert hash(mine) == hash(theirs)
    if mine.nanosecond == 0:
        plain = dt.datetime(
            mine.year, mine.month, mine.day, mine.hour, mine.minute, mine.second, mine.microsecond
        )
        assert mine == plain
        assert hash(mine) == hash(plain)


@needs_pandas
@pytest.mark.parametrize("freq", FREQUENCIES)
@pytest.mark.parametrize("how", ["round", "floor", "ceil"])
@pytest.mark.parametrize("text", MOMENTS)
def test_rounding_a_moment_agrees_with_pandas(
    firepanda: ModuleType, text: str, how: str, freq: str
) -> None:
    """Three directions over eleven frequencies over eight moments.

    The half to even rule is the part worth the size of this table. Rounding to
    the day puts a great many of these exactly on a boundary or exactly half way
    across one, and the tie has to break towards the even multiple rather than
    away from zero or upwards, which are the two rules somebody would write by
    hand.
    """
    import pandas as pd

    agree(
        lambda: getattr(firepanda.Timestamp(text), how)(freq),
        lambda: getattr(pd.Timestamp(text), how)(freq),
        f"Timestamp({text!r}).{how}({freq!r})",
    )


@needs_pandas
@pytest.mark.parametrize("freq", ["W", "ME", "QE", "YE", "B", "nonsense", 5, 1.5])
@pytest.mark.parametrize("how", ["round", "floor", "ceil"])
def test_rounding_to_something_that_is_not_a_fixed_frequency_fails_the_way_pandas_fails(
    firepanda: ModuleType, how: str, freq: Any
) -> None:
    """A week is not a fixed length and neither is a month, so neither can be rounded to.

    They are not refused for being unimplemented. A month is between twenty
    eight and thirty one days and there is no answer to what rounding to one
    means, which is why pandas refuses them too, and refusing with the same
    class is what lets a caller's `except ValueError` work either way round.

    A number is refused as the wrong type rather than the wrong value, which is
    also what pandas does and is worth a row here because the two are easy to
    swap. `None` is not in the list and has its own test below.
    """
    import pandas as pd

    agree(
        lambda: getattr(firepanda.Timestamp("2024-05-06 07:08:09"), how)(freq),
        lambda: getattr(pd.Timestamp("2024-05-06 07:08:09"), how)(freq),
        f"round to {freq!r}",
    )


@needs_pandas
@pytest.mark.parametrize("how", ["round", "floor", "ceil"])
def test_rounding_to_none_is_refused_as_a_type_where_pandas_leaks_an_attribute_error(
    firepanda: ModuleType, how: str
) -> None:
    """The one place refusing differently from pandas is the better answer.

    pandas hands `None` to its own machinery and it comes back out as
    `AttributeError: 'NoneType' object has no attribute 'nanos'`, which is an
    implementation detail escaping rather than an answer. Every other wrong type
    gets a `TypeError` there naming the argument, and this gives that one for
    `None` as well. Nobody writes `except AttributeError` around a rounding call,
    so nothing that works against pandas stops working against this.
    """
    with pytest.raises(TypeError):
        getattr(firepanda.Timestamp("2024-05-06 07:08:09"), how)(None)


@needs_pandas
@pytest.mark.parametrize("how", ["round", "floor", "ceil"])
def test_an_offset_object_rounds_by_spelling_itself(firepanda: ModuleType, how: str) -> None:
    """A pandas offset is accepted, and it is accepted without knowing what one is.

    Every offset can spell itself through `freqstr`, so the parser reads that and
    goes on as if a string had been passed. It costs nothing, it does not make
    firepanda depend on pandas, and it means code holding an offset does not have
    to translate it by hand.
    """
    import pandas as pd

    offset = pd.tseries.offsets.Hour(2)
    agree(
        lambda: getattr(firepanda.Timestamp("2024-05-06 07:08:09"), how)(offset),
        lambda: getattr(pd.Timestamp("2024-05-06 07:08:09"), how)(offset),
        f"{how} to an offset",
    )


@needs_pandas
@pytest.mark.parametrize("unit", [*UNITS, "D", "nonsense"])
@pytest.mark.parametrize("text", MOMENTS)
def test_as_unit_on_a_moment_agrees_with_pandas(
    firepanda: ModuleType, text: str, unit: str
) -> None:
    """Restating a moment at another resolution, including the two it refuses."""
    import pandas as pd

    agree(
        lambda: firepanda.Timestamp(text).as_unit(unit),
        lambda: pd.Timestamp(text).as_unit(unit),
        f"as_unit({unit!r})",
    )


@needs_pandas
@pytest.mark.parametrize("fmt", ["%Y-%m-%d", "%Y-%m-%d %H:%M:%S", "%j %U %A %B", "%f", "%z"])
@pytest.mark.parametrize("text", MOMENTS)
def test_strftime_agrees_with_pandas(firepanda: ModuleType, text: str, fmt: str) -> None:
    """The inherited formatter, which is inherited and therefore has to be checked."""
    import pandas as pd

    agree(
        lambda: firepanda.Timestamp(text).strftime(fmt),
        lambda: pd.Timestamp(text).strftime(fmt),
        f"strftime({fmt!r})",
    )


@needs_pandas
@pytest.mark.parametrize(
    "tz", ["UTC", "Europe/Paris", "America/New_York", "Asia/Tokyo", "+01:00", None, "Nowhere/Here"]
)
@pytest.mark.parametrize(
    "text", ["2026-09-05 13:45:06.123456789", "2026-01-05 13:45:06", "1677-09-22"]
)
def test_tz_localize_agrees_with_pandas(firepanda: ModuleType, text: str, tz: Any) -> None:
    """Attaching a zone keeps the clock and moves the instant.

    The 1677 row is the one that catches a printer. `Europe/Paris` was nine
    minutes and twenty one seconds off Greenwich until 1911, so the offset there
    has seconds in it, and a printer that stops at minutes prints a moment that
    is not the moment.
    """
    import pandas as pd

    agree(
        lambda: firepanda.Timestamp(text).tz_localize(tz),
        lambda: pd.Timestamp(text).tz_localize(tz),
        f"tz_localize({tz!r})",
    )


@needs_pandas
@pytest.mark.parametrize(
    ("policy", "given"),
    [
        ("ambiguous", "not a policy"),
        ("ambiguous", "infer"),
        ("ambiguous", 3),
        ("nonexistent", "not a policy"),
        ("nonexistent", 3),
    ],
)
def test_a_misspelled_zone_policy_fails_the_way_pandas_fails(
    firepanda: ModuleType, policy: str, given: Any
) -> None:
    """Neither policy is written yet, and the class still has to be the right one.

    A word pandas takes and firepanda has not written is a schedule, and it comes
    back `NotImplementedError` naming the issue. A word pandas does not take
    either is a typo in the caller's own line, and pandas answers that with a
    `ValueError` listing the words that would have worked. `agree` asks only
    which of those two happened, which is the whole question.

    `infer` is on the list because it is the surprise. It is a real word on a
    column and is not one on a scalar, since a single moment has no neighbours to
    infer a direction from, and pandas refuses it here while taking it there.
    """
    import pandas as pd

    agree(
        lambda: firepanda.Timestamp("2026-09-05 13:45:06").tz_localize("UTC", **{policy: given}),
        lambda: pd.Timestamp("2026-09-05 13:45:06").tz_localize("UTC", **{policy: given}),
        f"tz_localize({policy}={given!r})",
    )


@needs_pandas
@pytest.mark.parametrize("policy", ["ambiguous", "nonexistent"])
@pytest.mark.parametrize("tz", ["UTC", None])
def test_rounding_reads_the_zone_policies_only_when_there_is_a_zone(
    firepanda: ModuleType, policy: str, tz: Any
) -> None:
    """A naive moment rounds with a misspelled policy in its arguments unread.

    Measured against pandas rather than reasoned about. There is no daylight
    saving without a zone, so there is nothing for either policy to decide, and
    pandas hands the moment straight back. Refusing it would be firepanda turning
    away input pandas takes, which is the one direction of difference this
    library does not get to have. Put a zone on the same moment and both
    libraries refuse the same misspelling.
    """
    import pandas as pd

    agree(
        lambda: firepanda.Timestamp("2026-09-05 13:45:06", tz=tz).floor(
            "h", **{policy: "not a policy"}
        ),
        lambda: pd.Timestamp("2026-09-05 13:45:06", tz=tz).floor("h", **{policy: "not a policy"}),
        f"floor({policy}=) at tz={tz!r}",
    )


@needs_pandas
@pytest.mark.parametrize("tz", ["UTC", "Europe/Paris", "America/New_York", "+01:00", None])
@pytest.mark.parametrize("text", ["2026-09-05 13:45:06.123456789", "2026-01-05 13:45:06"])
def test_tz_convert_agrees_with_pandas(firepanda: ModuleType, text: str, tz: Any) -> None:
    """Converting keeps the instant and moves the clock, and refuses a naive moment."""
    import pandas as pd

    agree(
        lambda: firepanda.Timestamp(text).tz_convert(tz),
        lambda: pd.Timestamp(text).tz_convert(tz),
        f"naive tz_convert({tz!r})",
    )
    agree(
        lambda: firepanda.Timestamp(text, tz="Europe/Berlin").tz_convert(tz),
        lambda: pd.Timestamp(text, tz="Europe/Berlin").tz_convert(tz),
        f"zoned tz_convert({tz!r})",
    )


@needs_pandas
@pytest.mark.parametrize("tz", ["UTC", "Europe/Paris", "America/New_York", "Australia/Sydney"])
@pytest.mark.parametrize("text", ["2026-09-05 13:45:06", "2026-01-05 13:45:06"])
@pytest.mark.parametrize(
    "name",
    ["value", "hour", "isoformat", "__str__", "__repr__", "timestamp", "utcoffset", "tzname"],
)
def test_a_zoned_moment_reads_the_same_as_pandas(
    firepanda: ModuleType, text: str, tz: str, name: str
) -> None:
    """A zone named in the constructor localizes, and both halves of the year are here.

    Two dates six months apart in four zones covers both sides of a daylight
    saving change in each hemisphere, and it is the difference between attaching
    a zone and converting into one: the clock says the same thing either way in
    January and does not in September.
    """
    import pandas as pd

    mine, theirs = firepanda.Timestamp(text, tz=tz), pd.Timestamp(text, tz=tz)
    found = getattr(mine, name)
    agree(
        lambda: found() if callable(found) else found,
        lambda: (
            getattr(theirs, name)() if callable(getattr(theirs, name)) else getattr(theirs, name)
        ),
        f"{tz} {name}",
    )


@needs_pandas
@pytest.mark.parametrize(
    "fields",
    [
        {"year": 2020, "month": 1, "day": 2},
        {"year": 2020, "month": 1, "day": 2, "hour": 3, "nanosecond": 7},
        {"year": 2020, "month": 1, "day": 2, "tz": "UTC"},
        {"year": 2020, "month": 13, "day": 2},
        {"year": 2020},
    ],
)
def test_building_a_moment_from_fields_agrees_with_pandas(
    firepanda: ModuleType, fields: dict[str, Any]
) -> None:
    """The second constructor, which shares a door with the first one."""
    import pandas as pd

    agree(
        lambda: firepanda.Timestamp(**fields),
        lambda: pd.Timestamp(**fields),
        f"Timestamp(**{fields})",
    )


@needs_pandas
@pytest.mark.parametrize(
    "value",
    [
        "2026-09-05T13:45:06+02:00",
        1600000000,
        1600000000.5,
        dt.datetime(2020, 1, 1, 2, 3, 4),
        dt.date(2020, 1, 1),
        "not a date",
        [],
        True,
    ],
)
def test_building_a_moment_from_a_value_agrees_with_pandas(
    firepanda: ModuleType, value: Any
) -> None:
    """The first constructor, over the kinds of thing that can go in it.

    The unit that comes back is the interesting half and it is measured rather
    than chosen. A whole number is nanoseconds, a float is nanoseconds, a `date`
    is seconds because a date has no time in it, and a `datetime` is microseconds
    because that is as fine as one goes.
    """
    import pandas as pd

    agree(lambda: firepanda.Timestamp(value), lambda: pd.Timestamp(value), f"Timestamp({value!r})")
    agree(
        lambda: firepanda.Timestamp(value).unit,
        lambda: pd.Timestamp(value).unit,
        f"Timestamp({value!r}).unit",
    )


@needs_pandas
@pytest.mark.parametrize(
    ("unit", "value"),
    [
        ("s", 1600000000),
        ("ms", 1600000000),
        ("us", 1600000000),
        ("ns", 1600000000),
        ("h", 500000),
        ("m", 30000000),
        ("D", 20000),
        ("nonsense", 1),
    ],
)
def test_a_whole_number_with_a_unit_agrees_with_pandas(
    firepanda: ModuleType, unit: str, value: int
) -> None:
    """`unit=` scales the number and also decides what the answer is quoted at.

    The count differs per unit so that every row lands in a year both libraries
    can hold. The rows that do not are their own test further down, because a
    year past 9999 is a real difference and not a thing to hide by choosing
    smaller numbers.
    """
    import pandas as pd

    agree(
        lambda: firepanda.Timestamp(value, unit=unit),
        lambda: pd.Timestamp(value, unit=unit),
        f"Timestamp({value}, unit={unit!r})",
    )


@needs_pandas
@pytest.mark.parametrize("text", MOMENTS)
@pytest.mark.parametrize("spelled", SPANS)
@pytest.mark.parametrize("op", ["__add__", "__sub__", "__radd__"])
def test_moving_a_moment_by_a_span_agrees_with_pandas(
    firepanda: ModuleType, text: str, spelled: str, op: str
) -> None:
    """A moment plus a span, over every combination of the two tables."""
    import pandas as pd

    agree(
        lambda: getattr(firepanda.Timestamp(text), op)(firepanda.Timedelta(spelled)),
        lambda: getattr(pd.Timestamp(text), op)(pd.Timedelta(spelled)),
        f"{text} {op} {spelled}",
    )


@needs_pandas
@pytest.mark.parametrize("later", MOMENTS)
@pytest.mark.parametrize("earlier", MOMENTS)
def test_subtracting_two_moments_gives_the_span_pandas_gives(
    firepanda: ModuleType, earlier: str, later: str
) -> None:
    """Every pair of the eight, which is where the unit of the answer is decided.

    The answer is quoted at the finer of the two units, so subtracting a date
    from a nanosecond reading keeps the nanoseconds. Two of these pairs are far
    enough apart that the answer does not fit in a nanosecond count at all, and
    the span comes back quoted in microseconds rather than failing.
    """
    import pandas as pd

    agree(
        lambda: firepanda.Timestamp(later) - firepanda.Timestamp(earlier),
        lambda: pd.Timestamp(later) - pd.Timestamp(earlier),
        f"{later} - {earlier}",
    )


@needs_pandas
@pytest.mark.parametrize(
    "fields",
    [
        {"year": 1999},
        {"nanosecond": 5},
        {"tzinfo": dt.UTC},
        {"month": 13},
        {"hour": 1, "minute": 2},
    ],
)
@pytest.mark.parametrize("text", [one for one in MOMENTS if not one.startswith("1000")])
def test_replace_agrees_with_pandas(
    firepanda: ModuleType, text: str, fields: dict[str, Any]
) -> None:
    """The override with the wrong signature, which is the right signature here.

    pandas reads None as leave the field alone where the standard library reads
    a missing argument that way, and pandas puts `nanosecond` in the middle of
    the list. Matching pandas is what the class is for.

    The year 1000 is left out and has its own test below, because that is the
    one moment out of the eight where matching pandas would mean copying an
    inconsistency in it.
    """
    import pandas as pd

    agree(
        lambda: firepanda.Timestamp(text).replace(**fields),
        lambda: pd.Timestamp(text).replace(**fields),
        f"replace(**{fields})",
    )


@needs_pandas
def test_the_class_level_attributes_agree_with_pandas(firepanda: ModuleType) -> None:
    """`min`, `max` and `resolution`, which are values and not methods.

    `min` is the one that was wrong first. The obvious spelling rounds the
    nanosecond count off at the thousand, and pandas does not: its lowest moment
    is one nanosecond above the lowest signed 64 bit number, and the last three
    digits of it are 193.
    """
    import pandas as pd

    assert repr(firepanda.Timestamp.min) == repr(pd.Timestamp.min)
    assert repr(firepanda.Timestamp.max) == repr(pd.Timestamp.max)
    assert repr(firepanda.Timestamp.resolution) == repr(pd.Timestamp.resolution)
    assert repr(firepanda.Timedelta.min) == repr(pd.Timedelta.min)
    assert repr(firepanda.Timedelta.max) == repr(pd.Timedelta.max)
    assert repr(firepanda.Timedelta.resolution) == repr(pd.Timedelta.resolution)


@needs_pandas
def test_the_class_methods_agree_with_pandas(firepanda: ModuleType) -> None:
    """The four ways of building a moment that are not the constructor."""
    import pandas as pd

    agree(
        lambda: firepanda.Timestamp.fromordinal(737000),
        lambda: pd.Timestamp.fromordinal(737000),
        "fromordinal",
    )
    agree(
        lambda: firepanda.Timestamp.fromisoformat("2020-01-01T02:03:04"),
        lambda: pd.Timestamp.fromisoformat("2020-01-01T02:03:04"),
        "fromisoformat",
    )
    agree(
        lambda: firepanda.Timestamp.combine(dt.date(2020, 1, 1), dt.time(2, 3)),
        lambda: pd.Timestamp.combine(dt.date(2020, 1, 1), dt.time(2, 3)),
        "combine",
    )
    agree(
        lambda: firepanda.Timestamp.fromtimestamp(1600000000),
        lambda: pd.Timestamp.fromtimestamp(1600000000),
        "fromtimestamp",
    )


@needs_pandas
def test_every_public_name_pandas_puts_on_the_two_classes_is_here(firepanda: ModuleType) -> None:
    """The surface, which is the question the conformance board asks by reflection.

    This is the same measurement the board takes and it is here as well because
    the board runs somewhere else and this runs on every commit. A name that
    disappears from one of these classes should fail a test in the repository
    that owns the class, not only a run in the repository that scores it.
    """
    import pandas as pd

    for mine, theirs in ((firepanda.Timestamp, pd.Timestamp), (firepanda.Timedelta, pd.Timedelta)):
        missing = {n for n in dir(theirs) if not n.startswith("_")} - set(dir(mine))
        assert missing == set(), f"{theirs.__name__} is missing {sorted(missing)}"


@needs_pandas
def test_every_callable_takes_the_parameters_pandas_takes(firepanda: ModuleType) -> None:
    """The second half of the surface, which is the second question the board asks.

    Names and kinds and not defaults, because the repr of a pandas sentinel is not
    portable and comparing defaults would produce a wall of differences that are
    all about sentinel identity. A wrong default shows up where a user would
    notice it, which is in an answer.

    A callable pandas itself cannot introspect is skipped, since both sides
    failing to report a signature is not evidence of anything. That is why the
    two constructors are compared separately below: pandas can read those.

    The members are read off a built moment and a built span rather than off the
    two classes, which is what the board does and is not a detail. A method read
    off a class carries its `self` and a method read off an instance does not,
    and the inherited C level ones disagree about that parameter's kind between
    the two libraries in a way no caller can see.
    """
    import inspect

    import pandas as pd

    def parameters(obj: Any) -> list[str] | None:
        try:
            return [f"{p.name}:{p.kind.name}" for p in inspect.signature(obj).parameters.values()]
        except (TypeError, ValueError):
            return None

    differences = []
    pairs: list[tuple[str, Any, Any]] = [
        ("Timestamp", firepanda.Timestamp("2026-01-01"), pd.Timestamp("2026-01-01")),
        ("Timedelta", firepanda.Timedelta("1D"), pd.Timedelta("1D")),
    ]
    for label, mine, theirs in pairs:
        for name in sorted(n for n in dir(theirs) if not n.startswith("_")):
            want = parameters(getattr(theirs, name))
            if want is None:
                continue
            got = parameters(getattr(mine, name))
            if got != want:
                differences.append(f"{label}.{name}: {got}, pandas takes {want}")
        want = parameters(type(theirs))
        got = parameters(type(mine))
        if got != want:
            differences.append(f"{label}(): {got}, pandas takes {want}")
    assert differences == [], "\n".join(differences)


@needs_pandas
@pytest.mark.parametrize("spelled", SPANS)
@pytest.mark.parametrize(
    "name",
    [
        "value",
        "unit",
        "days",
        "seconds",
        "microseconds",
        "nanoseconds",
        "components",
        "resolution_string",
        "resolution",
    ],
)
def test_every_timedelta_property_agrees_with_pandas(
    firepanda: ModuleType, spelled: str, name: str
) -> None:
    """The parts a span breaks into, which floor rather than take an absolute value.

    Only the days can be negative. Minus one nanosecond is minus one day and
    twenty three hours and the rest, not a day of zero and a nanosecond of minus
    one, and the second spelling is the one somebody writes by hand.
    """
    import pandas as pd

    agree(
        lambda: getattr(firepanda.Timedelta(spelled), name),
        lambda: getattr(pd.Timedelta(spelled), name),
        f"Timedelta({spelled!r}).{name}",
    )


@needs_pandas
@pytest.mark.parametrize("spelled", SPANS)
@pytest.mark.parametrize(
    "name",
    [
        "total_seconds",
        "isoformat",
        "to_pytimedelta",
        "__str__",
        "__repr__",
        "__neg__",
        "__abs__",
        "__hash__",
        "__bool__",
    ],
)
def test_every_timedelta_method_that_takes_nothing_agrees_with_pandas(
    firepanda: ModuleType, spelled: str, name: str
) -> None:
    """The methods with no arguments, two of which the base class answers wrongly.

    `__bool__` is the clear one. A span of a single nanosecond rounds to nothing
    in the microseconds the base class keeps, so the inherited answer is False
    and the span is real. `total_seconds` is the same shape of problem: pandas
    adds up the days, the seconds and the microseconds and never reaches the
    nanoseconds, so one nanosecond is zero seconds in both.
    """
    import pandas as pd

    agree(
        lambda: getattr(firepanda.Timedelta(spelled), name)(),
        lambda: getattr(pd.Timedelta(spelled), name)(),
        f"Timedelta({spelled!r}).{name}()",
    )


@needs_pandas
@pytest.mark.parametrize("spelled", SPANS)
def test_a_span_prints_in_a_form_that_reads_back_as_itself(
    firepanda: ModuleType, spelled: str
) -> None:
    """The printed form is a parseable form, and for a negative span that is subtle.

    pandas prints minus one day plus two hours as `-1 days +02:03:04`, and the
    minus covers the days alone. A parser that reads it as covering the whole
    reading gets a different number back than the one that was printed, which
    would make the round trip silently lossy rather than loudly broken.
    """
    import pandas as pd

    mine = firepanda.Timedelta(spelled)
    assert firepanda.Timedelta(str(mine)).value == mine.value
    assert pd.Timedelta(str(mine)).value == mine.value


@needs_pandas
@pytest.mark.parametrize("freq", FREQUENCIES)
@pytest.mark.parametrize("how", ["round", "floor", "ceil"])
@pytest.mark.parametrize("spelled", SPANS)
def test_rounding_a_span_agrees_with_pandas(
    firepanda: ModuleType, spelled: str, how: str, freq: str
) -> None:
    """The same three directions over the same frequencies, on the other class."""
    import pandas as pd

    agree(
        lambda: getattr(firepanda.Timedelta(spelled), how)(freq),
        lambda: getattr(pd.Timedelta(spelled), how)(freq),
        f"Timedelta({spelled!r}).{how}({freq!r})",
    )


@needs_pandas
@pytest.mark.parametrize("unit", [*UNITS, "D", "nonsense"])
@pytest.mark.parametrize("spelled", SPANS)
def test_as_unit_on_a_span_agrees_with_pandas(
    firepanda: ModuleType, spelled: str, unit: str
) -> None:
    """Restating a span at another resolution, including the two it refuses."""
    import pandas as pd

    agree(
        lambda: firepanda.Timedelta(spelled).as_unit(unit),
        lambda: pd.Timedelta(spelled).as_unit(unit),
        f"Timedelta({spelled!r}).as_unit({unit!r})",
    )


@needs_pandas
@pytest.mark.parametrize("other", [2, 2.5, 0, -1, "1 days"])
@pytest.mark.parametrize(
    "op", ["__add__", "__sub__", "__mul__", "__truediv__", "__floordiv__", "__eq__", "__lt__"]
)
@pytest.mark.parametrize("spelled", SPANS)
def test_span_arithmetic_agrees_with_pandas(
    firepanda: ModuleType, spelled: str, op: str, other: Any
) -> None:
    """Spans against numbers and against spans, which give different kinds of answer.

    Divided by a number a span is a span, divided by a span it is a number, and
    the truncation on the first of those is towards zero rather than the round
    somebody would write. The two disagree on every value that lands on a
    fraction.
    """
    import pandas as pd

    mine = firepanda.Timedelta(other) if isinstance(other, str) else other
    theirs = pd.Timedelta(other) if isinstance(other, str) else other
    agree(
        lambda: getattr(firepanda.Timedelta(spelled), op)(mine),
        lambda: getattr(pd.Timedelta(spelled), op)(theirs),
        f"Timedelta({spelled!r}).{op}({other!r})",
    )


@needs_pandas
@pytest.mark.parametrize(
    "fields",
    [
        {"days": 1},
        {"hours": 2, "minutes": 3},
        {"weeks": 1},
        {"nanoseconds": 7},
        {"milliseconds": 1.5},
        {"years": 1},
        {"seconds": -1},
    ],
)
def test_building_a_span_from_fields_agrees_with_pandas(
    firepanda: ModuleType, fields: dict[str, Any]
) -> None:
    """The keyword constructor, whose unit depends on whether nanoseconds were named."""
    import pandas as pd

    agree(
        lambda: firepanda.Timedelta(**fields),
        lambda: pd.Timedelta(**fields),
        f"Timedelta({fields})",
    )
    agree(
        lambda: firepanda.Timedelta(**fields).unit,
        lambda: pd.Timedelta(**fields).unit,
        f"Timedelta({fields}).unit",
    )


@needs_pandas
@pytest.mark.parametrize("unit", [None, "W", "D", "h", "min", "s", "ms", "us", "ns", "Y"])
@pytest.mark.parametrize("value", [1, 1.5, -3, 0])
def test_building_a_span_from_a_number_agrees_with_pandas(
    firepanda: ModuleType, value: Any, unit: Any
) -> None:
    """A number and a unit, where the unit of the answer is not the unit passed in.

    A whole number of days is a whole number of seconds, so it is quoted in
    seconds. The same count with a fraction on it is quoted in nanoseconds,
    whatever the unit said. Neither is obvious and both are measured.
    """
    import pandas as pd

    agree(
        lambda: firepanda.Timedelta(value, unit),
        lambda: pd.Timedelta(value, unit),
        f"Timedelta({value!r}, {unit!r})",
    )
    agree(
        lambda: firepanda.Timedelta(value, unit).unit,
        lambda: pd.Timedelta(value, unit).unit,
        f"Timedelta({value!r}, {unit!r}).unit",
    )


@needs_pandas
@pytest.mark.parametrize(
    "value",
    [
        "1 days 02:03:04",
        "-1 days 02:03:04",
        "-1 days +02:03:04",
        "nope",
        dt.timedelta(days=2),
        True,
    ],
)
def test_building_a_span_from_a_value_agrees_with_pandas(firepanda: ModuleType, value: Any) -> None:
    """Text and timedeltas, including the two sign spellings that mean the same thing."""
    import pandas as pd

    agree(lambda: firepanda.Timedelta(value), lambda: pd.Timedelta(value), f"Timedelta({value!r})")


@needs_pandas
@pytest.mark.parametrize("text", TEXT_SPANS)
def test_a_span_written_as_counts_and_units_agrees_with_pandas(
    firepanda: ModuleType, text: str
) -> None:
    """The shape people type, as against the shape a span prints as.

    Both the nanosecond count and the unit are compared, because the unit is
    where this form is least obvious: three separate rules decide it and all
    three were measured rather than reasoned about. A whole sweep of eight
    hundred and eighty five spellings ran against pandas while this was written
    and the list below is the part of it worth keeping, one entry per thing that
    can go wrong rather than one per spelling.
    """
    import pandas as pd

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        agree(
            lambda: (firepanda.Timedelta(text).value, firepanda.Timedelta(text).unit),
            lambda: (pd.Timedelta(text).value, pd.Timedelta(text).unit),
            f"Timedelta({text!r})",
        )


@needs_pandas
@pytest.mark.parametrize("spelled", ["w", "d", "H", "S", "MIN", "MS", "US", "NS"])
def test_the_deprecated_unit_spellings_warn_the_way_pandas_warns(
    firepanda: ModuleType, spelled: str
) -> None:
    """A program under `-W error::DeprecationWarning` has to break in both libraries.

    The class is `DeprecationWarning` here and `Pandas4Warning` there, which is
    a subclass of it, so a filter written against the base catches both and a
    filter written against the pandas name catches only pandas. That is the
    smaller of the two possible mistakes and document 31 made it first.
    """
    import pandas as pd

    with pytest.warns(DeprecationWarning, match=f"'{spelled}' is deprecated"):
        firepanda.Timedelta("1" + spelled)
    with pytest.warns(DeprecationWarning, match=f"'{spelled}' is deprecated"):
        pd.Timedelta("1" + spelled)


@needs_pandas
@pytest.mark.parametrize(
    "text", ["1 M", "1 Y", "1 y", "1 zz", "hello", "--1D", "1D-2h", "1e3s", "1_000s", "1 T", "1 L"]
)
def test_a_span_that_names_no_length_is_refused_the_way_pandas_refuses_it(
    firepanda: ModuleType, text: str
) -> None:
    """A month has no fixed length and `1D-2h` has two readings, so neither is guessed.

    `T` and `L` are in here because they used to be minutes and milliseconds and
    are not any more, so a program carrying them has to find out rather than get
    a number.
    """
    import pandas as pd

    with pytest.raises(ValueError):
        firepanda.Timedelta(text)
    with pytest.raises(ValueError):
        pd.Timedelta(text)


@needs_pandas
def test_throwing_the_nanoseconds_away_says_so_the_way_pandas_says_so(
    firepanda: ModuleType,
) -> None:
    """`to_pydatetime` warns when it loses something, and the words are pandas' words.

    A silent loss of precision is the kind of thing somebody finds a year later
    in a number that is slightly wrong, and the whole reason this class exists
    rather than a plain `datetime` is that nanoseconds are worth keeping. So the
    conversion that cannot keep them is loud about it, and it is loud in the same
    words as pandas so that a project filtering the warning keeps filtering it.
    """
    import pandas as pd

    with pytest.warns(UserWarning, match="Discarding nonzero nanoseconds"):
        firepanda.Timestamp("2026-09-05 13:45:06.123456789").to_pydatetime()
    with pytest.warns(UserWarning, match="Discarding nonzero nanoseconds"):
        pd.Timestamp("2026-09-05 13:45:06.123456789").to_pydatetime()

    # No warning when there is nothing to lose, which is the other half of it.
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        firepanda.Timestamp("2026-09-05 13:45:06").to_pydatetime()


@needs_pandas
def test_the_two_things_pandas_turns_into_nat_are_refused_here(firepanda: ModuleType) -> None:
    """firepanda has no NaT, so the input that makes one has nowhere to go.

    This is asserted rather than worked around because it is a gap and not a
    decision. A missing temporal value inside a column is an Arrow null and that
    works; a missing temporal value as a scalar needs a singleton that is equal
    to nothing including itself, and there is not one. Until there is, refusing
    with a clear class beats returning None and letting it travel.
    """
    with pytest.raises(TypeError):
        firepanda.Timestamp(None)
    with pytest.raises(ValueError):
        firepanda.Timedelta(None)


@needs_pandas
@pytest.mark.parametrize("unit", ["D", "h", "m"])
def test_a_moment_past_the_year_nine_thousand_is_out_of_range_here(
    firepanda: ModuleType, unit: str
) -> None:
    """The one place the range is narrower than pandas', and it is narrower on purpose.

    pandas' `Timestamp` claims to be a `datetime` and is not really one, so it
    can hold the year 4382621 and print it. This one is a `datetime`, which is
    the property that makes every library holding a `datetime` keep working, and
    a `datetime` stops at 9999. The trade is worth making in that direction.
    """
    with pytest.raises(ValueError):
        firepanda.Timestamp(1600000000, unit=unit)


@needs_pandas
def test_replacing_a_field_keeps_a_moment_pandas_gives_up_on(firepanda: ModuleType) -> None:
    """A moment before 1678 survives `replace` here, and in pandas it does not.

    pandas holds that moment quite happily, prints it and does arithmetic on it,
    and then refuses to replace a field on it because `replace` goes back through
    a constructor that checks the nanosecond range rather than the range of the
    unit the value is quoted at. It is an inconsistency inside pandas rather than
    a rule, the rest of this file already checks the range against the unit, and
    a moment that exists should stay usable, so this one is left working.
    """
    made = firepanda.Timestamp("1000-01-01 12:00:00").replace(nanosecond=5)
    assert made.nanosecond == 5
    assert made.year == 1000


@needs_pandas
@pytest.mark.parametrize("text", MOMENTS)
def test_to_julian_date_gives_the_same_number_in_a_plain_float(
    firepanda: ModuleType, text: str
) -> None:
    """The number matches and the type does not, which is the same call as `dtype`.

    pandas hands back a `numpy.float64` here. firepanda does not depend on numpy
    and a value that changes type depending on what else is installed is worse
    than one that is always a float, so it is always a float. They compare equal,
    which is what almost all of the code that reads this actually needs.
    """
    import pandas as pd

    mine = firepanda.Timestamp(text).to_julian_date()
    assert type(mine) is float
    assert mine == pd.Timestamp(text).to_julian_date()


@both
@pytest.mark.parametrize("freq", FREQUENCIES)
@pytest.mark.parametrize("how", ["round", "floor", "ceil"])
def test_the_scalar_and_the_column_round_the_same_way(
    firepanda: ModuleType, how: str, freq: str
) -> None:
    """The test the scalars module docstring names, and the reason it names one.

    There are two frequency parsers in firepanda. The column goes through
    `frequency_period` in `firepanda/kernel/temporal.mojo` and the scalar goes
    through `_period` in `firepanda/_scalars.py`, and a scalar cannot reach the
    first one without building a column to carry one value through it. Two
    parsers for one vocabulary drift, and the only thing that stops them is
    asking both the same question and comparing.

    The rounding rule is the other half. Half to even on an exact tie is easy to
    write as half up by accident, and every frequency here divides at least one
    of the rows exactly in half.
    """
    import pyarrow as pa

    values = [
        dt.datetime(2024, 1, 1, 0, 0, 0),
        dt.datetime(2024, 2, 29, 13, 45, 30, 123456),
        dt.datetime(2023, 12, 31, 23, 59, 59),
        dt.datetime(2024, 7, 4, 12, 0, 0),
        dt.datetime(2024, 7, 4, 18, 0, 0),
    ]
    table = pa.table({"t": pa.array(values, type=pa.timestamp("us"))})
    column = firepanda.from_arrow(table)["t"]

    through_the_kernel = pa.array(getattr(column.dt, how)(freq)).to_pylist()
    through_the_scalar = [
        getattr(firepanda.Timestamp(one), how)(freq).to_pydatetime() for one in values
    ]
    assert through_the_kernel == through_the_scalar, (
        f"{how}({freq}) gave {through_the_kernel} through the column"
        f" and {through_the_scalar} through the scalar"
    )


@both
@pytest.mark.parametrize("freq", ["W", "ME", "nonsense"])
def test_the_scalar_and_the_column_refuse_the_same_frequencies(
    firepanda: ModuleType, freq: str
) -> None:
    """The other half of the agreement, which is what the two of them will not do.

    A parser that accepts more than the other one is as much of a drift as a
    parser that answers differently, and it is the easier one to introduce, since
    adding an alias to one file is a one line change that looks complete.
    """
    import pyarrow as pa

    table = pa.table({"t": pa.array([dt.datetime(2024, 1, 1)], type=pa.timestamp("us"))})
    column = firepanda.from_arrow(table)["t"]

    with pytest.raises(ValueError):
        column.dt.floor(freq)
    with pytest.raises(ValueError):
        firepanda.Timestamp("2024-01-01").floor(freq)


@both
def test_a_moment_out_of_a_column_could_be_one_of_these(firepanda: ModuleType) -> None:
    """What the whole slice is for, stated as the thing it makes possible.

    A temporal column reads back as an integer today, which is filed rather than
    fixed here. This asserts the other end: that the value in the buffer, handed
    to `Timestamp`, gives the moment pandas gives for the same row. When the
    column starts handing back scalars, this is the answer it has to hand back.
    """
    import pandas as pd
    import pyarrow as pa

    values = [dt.datetime(2024, 2, 29, 13, 45, 30, 123456), dt.datetime(2023, 12, 31, 23, 59, 59)]
    table = pa.table({"t": pa.array(values, type=pa.timestamp("us"))})
    column = firepanda.from_arrow(table)["t"]

    for stored, expected in zip(column.tolist(), values, strict=True):
        made = firepanda.Timestamp(stored, unit="us")
        assert made == firepanda.Timestamp(expected)
        assert repr(made) == repr(pd.Timestamp(expected))
