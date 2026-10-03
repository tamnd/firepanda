"""Mistakes with dates, times and spans, refused with pandas' words.

A caller who catches one of these by its message should not have to know which
library raised it, so each case asserts the sentence pandas uses.
"""

from __future__ import annotations

import zoneinfo
from types import ModuleType

import pytest

_COUNT_OF_FREQ = (
    r"^Addition/subtraction of integers and integer-arrays with {} is no longer supported\."
    r"  Instead of adding/subtracting `n`, use `n \* obj\.freq`$"
)


def _instants(firepanda: ModuleType) -> object:
    return firepanda.Series(firepanda.to_datetime(["2024-01-01", "2024-01-02"]))


@pytest.mark.parametrize(
    ("freq", "words"),
    [
        ("xx", r"^Invalid frequency: xx\. Failed to parse with error message: ValueError"),
        ("M", r"'M' is no longer supported for offsets\. Please use 'ME' instead\."),
    ],
)
def test_rounding_to_text_that_is_no_frequency(
    firepanda: ModuleType, freq: str, words: str
) -> None:
    """`round`, `floor` and `ceil` read the frequency the way `to_offset` does."""
    column = _instants(firepanda)
    for method in ("round", "floor", "ceil"):
        with pytest.raises(ValueError, match=words):
            getattr(column.dt, method)(freq)
    with pytest.raises(ValueError, match=words):
        firepanda.DatetimeIndex(["2024-01-01"]).round(freq)


def test_rounding_to_a_fixed_frequency_still_rounds(firepanda: ModuleType) -> None:
    """A compound frequency is still read as one count of its finest part."""
    column = firepanda.Series(firepanda.to_datetime(["2024-01-01 10:47"]))
    assert str(column.dt.floor("1h30min").iloc[0]) == "2024-01-01 10:30:00"


def test_units_nobody_has(firepanda: ModuleType) -> None:
    """Each constructor has its own sentence for a unit it does not know."""
    with pytest.raises(ValueError, match=r"^Unrecognized unit xx$"):
        firepanda.Timestamp(1, unit="xx")
    with pytest.raises(ValueError, match=r"^Unrecognized unit xx$"):
        firepanda.Timestamp(1.5, unit="xx")
    with pytest.raises(ValueError, match=r"^invalid unit abbreviation: xx$"):
        firepanda.Timedelta(1, unit="xx")
    with pytest.raises(ValueError, match=r"^Unrecognized unit xx$"):
        firepanda.to_datetime([1], unit="xx")
    with pytest.raises(TypeError, match=r'^Invalid datetime unit in metadata string "\[xx\]"$'):
        firepanda.to_datetime(firepanda.Series([1]), unit="xx")


@pytest.mark.parametrize(
    ("text", "words"),
    [
        ("xx", r"^Unknown datetime string format, unable to parse: xx$"),
        ("2024-01-01x", r"^Unknown datetime string format, unable to parse: 2024-01-01x$"),
        ("2024-99-99", r"^month must be in 1\.\.12, not 99: 2024-99-99$"),
        ("2023-02-29", r"^day 29 must be in range 1\.\.28 for month 2 in year 2023: 2023-02-29$"),
        ("2024-01-01 25:00", r"^hour must be in 0\.\.23, not 25: 2024-01-01 25:00$"),
        ("2024-01-01T24:00", r"^hour must be in 0\.\.23, not 24: 2024-01-01T24:00$"),
        ("2024-01-01 10:61", r"^minute must be in 0\.\.59, not 61: 2024-01-01 10:61$"),
    ],
)
def test_text_a_timestamp_cannot_read(firepanda: ModuleType, text: str, words: str) -> None:
    """The first field out of range is named, and anything else is an unknown format."""
    with pytest.raises(ValueError, match=words) as caught:
        firepanda.Timestamp(text)
    assert type(caught.value).__name__ == "DateParseError"


def test_dt_on_a_column_that_holds_no_instants(firepanda: ModuleType) -> None:
    """The accessor is refused where it is reached, so `hasattr` answers False."""
    with pytest.raises(AttributeError, match=r"^Can only use \.dt accessor with datetimelike"):
        _ = firepanda.Series([1, 2]).dt
    assert not hasattr(firepanda.Series(["a"]), "dt")
    assert hasattr(firepanda.Series([firepanda.Timedelta(1)]), "dt")


@pytest.mark.parametrize(
    ("dtype", "kind"),
    [("int8", "integer"), ("float32", "floating"), ("bool", "boolean"), ("Int64", "integer")],
)
def test_str_on_a_column_of_another_kind(firepanda: ModuleType, dtype: str, kind: str) -> None:
    """The message ends with the kind `infer_dtype` names rather than the dtype."""
    column = firepanda.Series([1, 0]).astype(dtype)
    with pytest.raises(AttributeError, match=rf"with string values, not {kind}$"):
        _ = column.str


def test_whole_numbers_added_to_instants_and_spans(firepanda: ModuleType) -> None:
    """pandas no longer reads a whole number as a count of the frequency."""
    column = _instants(firepanda)
    instants = _COUNT_OF_FREQ.format("DatetimeArray")
    for attempt in (
        lambda: column + 1,
        lambda: 1 - column,
        lambda: column.add(1),
        lambda: column + firepanda.Series([1, 2]),
        lambda: firepanda.Series([1, 2]) + column,
        lambda: firepanda.DataFrame({"a": column}) + 1,
        lambda: firepanda.DatetimeIndex(["2024-01-01"]) - 1,
    ):
        with pytest.raises(TypeError, match=instants):
            attempt()
    with pytest.raises(TypeError, match=_COUNT_OF_FREQ.format("TimedeltaArray")):
        firepanda.Series([firepanda.Timedelta(1, "D")]) - 1
    with pytest.raises(TypeError, match=_COUNT_OF_FREQ.format("Timestamp")):
        firepanda.Timestamp("2024-01-01") + 1


def test_instants_are_never_scaled(firepanda: ModuleType) -> None:
    """Multiplying or dividing instants names the dunder and numpy's array."""
    column = _instants(firepanda)
    with pytest.raises(TypeError, match=r"^cannot perform __mul__ with this index type: Datet"):
        column * 2
    with pytest.raises(TypeError, match=r"^cannot perform __rmul__ with this index type"):
        2 * column
    with pytest.raises(TypeError, match=r"^cannot perform __truediv__ with this index type"):
        column / 2
    with pytest.raises(TypeError, match=r"^unsupported operand type\(s\) for \+: 'DatetimeArray"):
        column + 1.5


def test_spans_still_scale(firepanda: ModuleType) -> None:
    """A span times a number is a span, which the refusals above leave alone."""
    spans = firepanda.Series([firepanda.Timedelta(1, "D")])
    assert str((spans * 2).iloc[0]) == "2 days 00:00:00"


def test_localizing_a_column_that_has_a_zone(firepanda: ModuleType) -> None:
    """The sentence ends where pandas ends it."""
    column = _instants(firepanda).dt.tz_localize("UTC")
    with pytest.raises(TypeError, match=r"^Already tz-aware, use tz_convert to convert\.$"):
        column.dt.tz_localize("UTC")


def test_a_zone_the_system_does_not_know(firepanda: ModuleType) -> None:
    """zoneinfo's own KeyError, which is what pandas lets through."""
    with pytest.raises(zoneinfo.ZoneInfoNotFoundError, match="No time zone found with key Mars"):
        _instants(firepanda).dt.tz_localize("Mars/Base")


@pytest.mark.parametrize(
    ("freq", "inner"),
    [
        ("bogus", "ValueError('Invalid frequency: bogus.')"),
        ("3xs", "ValueError('Invalid frequency: xs.')"),
        ("xx", "KeyError('xx')"),
        ("A-JAN", "KeyError('A'). Did you mean Y-JAN?"),
        ("AS", "KeyError('AS'). Did you mean YS?"),
    ],
)
def test_the_reason_inside_an_unreadable_frequency(
    firepanda: ModuleType, freq: str, inner: str
) -> None:
    """A name ending in a second fails as a count of seconds, and the rest as a lookup."""
    with pytest.raises(ValueError) as caught:
        firepanda.date_range("2024-01-01", periods=2, freq=freq)
    assert inner in str(caught.value)
