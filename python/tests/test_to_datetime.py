"""`firepanda.to_datetime`, checked against a running pandas.

This is the first module level function in the package that is hand written
rather than generated, and it is the first door into a timestamp column that is
not Arrow. Both of those are why it gets a file of its own rather than a section
of `test_dt.py`.

The comparison against pandas is done on the values and on the dtype string,
because those are the two things a caller sees. It is not done on the type of
the answer, and that is the one difference worth reading before the tests. pandas
hands back a `DatetimeIndex` for a list and a `Series` for a `Series`, and this
hands back a `Series` for both. firepanda's `Index` is a labels object with none
of the calendar members a `DatetimeIndex` carries, so answering one would be a
name that resolves and then has nothing on it. That is asserted here rather than
worked around, because it is a decision and not an accident.

Values come back out through `pyarrow.array` for the same reason `test_dt.py`
gives. `tolist` on a temporal column hands back the stored integer rather than a
`datetime`, since it reads the buffer and the buffer holds microseconds since
the epoch.

The other thing worth knowing before reading is which strings are refused. Only
ISO 8601 is guessed. pandas also guesses `01/02/2026` and decides for itself
which of the two numbers is the month, and a wrong guess there is a column that
is wrong by up to eleven months with nothing anywhere reporting it. Passing
`format` reads anything, so nothing is unreachable, it just has to be said.
"""

from __future__ import annotations

import importlib.util
from types import ModuleType
from typing import Any

import pytest

both = pytest.mark.skipif(
    importlib.util.find_spec("pyarrow") is None or importlib.util.find_spec("pandas") is None,
    reason="pyarrow and pandas are not both installed",
)

needs_pandas = pytest.mark.skipif(
    importlib.util.find_spec("pandas") is None, reason="pandas is not installed"
)


DATES = ["2026-01-01", "2026-01-02", None, "2026-12-31"]
"""Four dates, one of them missing, all at the same resolution.

They have to be at the same resolution because the format is worked out from the
first row that is there and every other row is held to it, which is what pandas
does too and is checked further down.
"""

STAMPS = ["2026-01-02T03:04:05", "2026-01-03T00:00:00", "2026-06-30T23:59:59"]
"""Three instants with a time on them, for the shape the guesser reaches second."""


def read(series: Any) -> list[Any]:
    """Reads a firepanda column out as Python values, through Arrow."""
    import pyarrow as pa

    return list(pa.array(series).to_pylist())


def theirs(values: list[Any], **kwargs: Any) -> Any:
    """The same call in pandas, as a list of Python values."""
    import pandas as pd

    return list(pd.to_datetime(values, **kwargs))


def absent(one: Any) -> bool:
    """Whether a value is missing, counting every spelling of it as one."""
    import pandas as pd

    return one is None or one is pd.NaT or (isinstance(one, float) and one != one)


def like(mine: list[Any], them: list[Any]) -> bool:
    """Whether two lists agree, once every spelling of missing is the same thing."""
    if len(mine) != len(them):
        return False
    return all(
        absent(a) == absent(b) and (absent(a) or a == b) for a, b in zip(mine, them, strict=True)
    )


@both
def test_a_list_of_dates_gives_the_pandas_answer(firepanda: ModuleType) -> None:
    """The plainest call there is, and the one the conformance board makes most."""
    assert like(read(firepanda.to_datetime(DATES)), theirs(DATES))


@both
def test_a_list_of_instants_gives_the_pandas_answer(firepanda: ModuleType) -> None:
    """The same, with a time of day on every row."""
    assert like(read(firepanda.to_datetime(STAMPS)), theirs(STAMPS))


@both
def test_the_resolution_is_microseconds_like_pandas(firepanda: ModuleType) -> None:
    """pandas 3 answers `datetime64[us]` for text, and this is where that is pinned.

    It is worth an assertion of its own because it is not the resolution the
    kernel would pick on its own. A column of whole seconds fits in seconds and
    the parser deliberately does not use that, because a caller who reads two
    columns and lines them up should not have to think about which one had a
    fractional second in it.
    """
    import pandas as pd

    assert firepanda.to_datetime(DATES).dtype == "datetime64[us]"
    assert str(pd.to_datetime(DATES).dtype) == "datetime64[us]"


@both
def test_nine_fraction_digits_promote_the_column_to_nanoseconds(firepanda: ModuleType) -> None:
    """More precision than microseconds hold has to widen the column or be lost."""
    fine = ["2026-01-01T00:00:00.123456789"]
    assert firepanda.to_datetime(fine).dtype == "datetime64[ns]"
    assert like(read(firepanda.to_datetime(fine)), theirs(fine))


@both
@pytest.mark.parametrize("unit", ["s", "ms", "us", "ns"])
def test_whole_numbers_are_counts_from_the_epoch(firepanda: ModuleType, unit: str) -> None:
    """The other door in, and the only one `unit` means anything on."""
    counts = [0, 1, 1_000_000, -1]
    assert like(read(firepanda.to_datetime(counts, unit=unit)), theirs(counts, unit=unit))


@both
def test_a_missing_row_stays_missing(firepanda: ModuleType) -> None:
    """A null in, a null out, without the row moving anything around it."""
    answer = read(firepanda.to_datetime(DATES))
    assert answer[2] is None
    assert answer[0] is not None
    assert answer[3] is not None


@both
def test_the_words_that_mean_missing(firepanda: ModuleType) -> None:
    """Empty, `NaT` and `nan` are missing, and `null` is not.

    pandas is the reason for the odd looking third one. It reads `nan` in a
    column of text as a missing timestamp, which is not a date by any reading,
    and it does not read `null`. Copying the list exactly is the point.
    """
    words = ["2026-01-01", "", "NaT", "nan", "NAN"]
    assert like(read(firepanda.to_datetime(words)), theirs(words))
    with pytest.raises(ValueError):
        firepanda.to_datetime(["2026-01-01", "null"])


@both
def test_coerce_turns_a_bad_row_into_a_missing_one(firepanda: ModuleType) -> None:
    """The second of the two `errors` words, and the only other one implemented."""
    rows = ["2026-01-01", "nope", "2026-02-30"]
    assert like(read(firepanda.to_datetime(rows, errors="coerce")), theirs(rows, errors="coerce"))


@both
def test_raise_is_the_default_and_names_the_row(firepanda: ModuleType) -> None:
    """A row that will not read stops the call rather than quietly going missing."""
    with pytest.raises(ValueError):
        firepanda.to_datetime(["2026-01-01", "nope"])


@needs_pandas
def test_an_impossible_date_is_refused(firepanda: ModuleType) -> None:
    """The thirtieth of February parses as digits and is not a day."""
    with pytest.raises(ValueError):
        firepanda.to_datetime(["2026-02-30"])


@needs_pandas
def test_only_two_words_of_errors_are_accepted(firepanda: ModuleType) -> None:
    """`ignore` was removed from pandas 3, and anything else was never in it."""
    with pytest.raises(ValueError, match="errors"):
        firepanda.to_datetime(["2026-01-01"], errors="ignore")


@both
def test_an_offset_is_kept_in_the_dtype(firepanda: ModuleType) -> None:
    """A row carrying its own offset gives a column that knows which clock it is on.

    The values are compared as stored integers rather than through Arrow, which
    is the one place in this file that happens. `pa.array(...).to_pylist()` on a
    fixed offset column asks pyarrow to build a Python tzinfo and it refuses
    with "The zoneinfo module or pytz package must be installed" even where
    `zoneinfo` imports fine. That is pyarrow's reading of the column and not
    firepanda's writing of it, so the test reads the buffer instead.
    """
    import pandas as pd

    zoned = ["2026-01-01T14:00:00+02:00"]
    assert firepanda.to_datetime(zoned).dtype == "datetime64[us, UTC+02:00]"
    assert str(pd.to_datetime(zoned).dtype) == "datetime64[us, UTC+02:00]"
    assert firepanda.to_datetime(zoned).tolist() == list(pd.to_datetime(zoned).astype("int64"))


@both
def test_mixed_offsets_need_utc(firepanda: ModuleType) -> None:
    """Two offsets in one column have no single clock, so the caller has to pick one.

    pandas raises here as well, and its message says to pass `utc=True` too. The
    instants are the same moment written two ways, so once there is one clock
    the two values are equal, which is what the last line says.
    """
    mixed = ["2026-01-01T14:00:00+02:00", "2026-01-01T10:00:00-02:00"]
    with pytest.raises(ValueError):
        firepanda.to_datetime(mixed)
    answer = firepanda.to_datetime(mixed, utc=True)
    assert firepanda.to_datetime(mixed, utc=True).dtype == "datetime64[us, UTC]"
    assert read(answer)[0] == read(answer)[1]


@both
def test_utc_on_naive_text_says_utc(firepanda: ModuleType) -> None:
    """Rows with no offset are read as UTC rather than moved."""
    import pandas as pd

    assert firepanda.to_datetime(DATES, utc=True).dtype == "datetime64[us, UTC]"
    assert str(pd.to_datetime(DATES, utc=True).dtype) == "datetime64[us, UTC]"


@both
def test_a_format_reads_what_the_guesser_will_not(firepanda: ModuleType) -> None:
    """The escape hatch, and the reason refusing to guess costs nothing."""
    american = ["01/02/2026", "12/31/2026"]
    assert like(
        read(firepanda.to_datetime(american, format="%m/%d/%Y")),
        theirs(american, format="%m/%d/%Y"),
    )


@needs_pandas
def test_the_guesser_refuses_rather_than_picks(firepanda: ModuleType) -> None:
    """The one place this is deliberately stricter than pandas, and why.

    pandas reads `01/02/2026` as the second of January. Somebody who wrote it
    meaning the first of February gets a column that is wrong by a month and no
    warning anywhere. firepanda says it does not recognise the shape and names
    the value, which costs the caller one `format=` and cannot be wrong quietly.
    """
    with pytest.raises(ValueError, match="01/02/2026"):
        firepanda.to_datetime(["01/02/2026"])


@needs_pandas
def test_every_row_is_held_to_the_first_row_format(firepanda: ModuleType) -> None:
    """One format for the column, worked out once, which is what pandas does too.

    pandas raises `unconverted data remains` here. The rule matters more than
    the message: guessing per row is what `format="mixed"` is for in pandas, and
    it is refused by name rather than done silently.
    """
    with pytest.raises(ValueError):
        firepanda.to_datetime(["2026-01-01", "2026-01-02T03:04:05"])


@needs_pandas
@pytest.mark.parametrize(
    ("argument", "value"),
    [
        ("dayfirst", True),
        ("yearfirst", True),
        ("origin", "julian"),
        ("exact", False),
        ("format", "ISO8601"),
        ("format", "mixed"),
    ],
)
def test_a_declared_argument_that_is_not_implemented_says_so(
    firepanda: ModuleType, argument: str, value: Any
) -> None:
    """Six refusals, each naming the argument the caller passed.

    They are declared rather than left out for the reason document 18 section 4
    gives. A caller who passes one gets a message about that argument instead of
    a TypeError about an unexpected keyword, and the day one of them is
    implemented no signature changes.
    """
    with pytest.raises(NotImplementedError, match=argument):
        firepanda.to_datetime(["2026-01-01"], **{argument: value})


@needs_pandas
def test_the_defaults_of_the_refused_arguments_are_accepted(firepanda: ModuleType) -> None:
    """Passing them at their default is not passing them, which is what makes it honest."""
    answer = firepanda.to_datetime(
        ["2026-01-01"], dayfirst=False, yearfirst=False, origin="unix", format=None
    )
    assert answer.dtype == "datetime64[us]"


@needs_pandas
def test_cache_is_accepted_and_changes_nothing(firepanda: ModuleType) -> None:
    """The one argument that is honoured by being ignored.

    pandas caches repeated values to go faster and the answer is the same either
    way, so honouring it means not changing the answer. It is not refused
    because refusing an argument that cannot be observed would fail calls that
    are asking for nothing.
    """
    assert read(firepanda.to_datetime(["2026-01-01"], cache=False)) == read(
        firepanda.to_datetime(["2026-01-01"], cache=True)
    )


@both
def test_a_series_goes_in_and_keeps_its_name(firepanda: ModuleType) -> None:
    """The form the `dt` accessor needs, since a column has a name and a list does not."""
    column = firepanda.Series(DATES, name="when")
    answer = firepanda.to_datetime(column)
    assert answer.name == "when"
    assert like(read(answer), theirs(DATES))


@both
def test_a_column_of_instants_passes_through(firepanda: ModuleType) -> None:
    """Reading a timestamp column again is not an error and not a second parse."""
    once = firepanda.to_datetime(DATES)
    twice = firepanda.to_datetime(once)
    assert read(twice) == read(once)
    assert twice.dtype == once.dtype


@needs_pandas
def test_the_answer_is_a_series_where_pandas_gives_an_index(firepanda: ModuleType) -> None:
    """The one divergence a caller meets on their first line, asserted on purpose.

    firepanda's `Index` is a labels object with none of the calendar members a
    `DatetimeIndex` carries, so answering one would be a name that resolves and
    then has nothing on it. A `Series` has the `dt` accessor, which is what
    almost every use of this reaches for next. See #354.
    """
    import pandas as pd

    assert isinstance(firepanda.to_datetime(DATES), firepanda.Series)
    assert isinstance(pd.to_datetime(DATES), pd.DatetimeIndex)


def test_it_is_exported_from_the_package(firepanda: ModuleType) -> None:
    """A module level name, which is how pandas spells it and how callers write it."""
    assert "to_datetime" in firepanda.__all__
    assert callable(firepanda.to_datetime)


@needs_pandas
def test_the_signature_matches_pandas(firepanda: ModuleType) -> None:
    """Every parameter pandas takes, in the order pandas takes them.

    `test_bindings.py` checks the same function through its parity walk, and
    this is stricter in the one way that matters for a function nothing
    generates. That walk compares the parameters we declare against the pandas
    ones by name and default, which lets a member declare a subset while it is
    being adopted a few arguments at a time. This one declares all ten and says
    so, including the order, because a caller writing `to_datetime(values,
    "coerce")` positionally is reading the order and nothing else.
    """
    import inspect

    import pandas as pd

    mine = list(inspect.signature(firepanda.to_datetime).parameters)
    them = list(inspect.signature(pd.to_datetime).parameters)
    assert mine == them
