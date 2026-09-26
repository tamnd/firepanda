"""Reading every row of text with its own format, for `format="ISO8601"` and `"mixed"`.

The core reads a column of text against one format, worked out from the first
row or given by the caller. These two spellings ask for the format to be worked
out again on every row, so the rows are read here, one distinct text at a time,
into their parts. The parts are then written back as text in one shape for the
whole column, which the core reads in a single pass, so the instants, the unit
and the zone come out of the same code as every other `to_datetime`.

The rules were measured against pandas 3.0.

- `ISO8601` reads the shapes of ISO 8601 pandas reads: a year alone, a year
  and month, a full date with one or two digit month and day, or eight digits;
  then a `T` or one space and a time with or without colons; then `Z` or an
  offset, which may follow a space. Anything else, including a date that does
  not exist, is `Time data ... is not ISO8601 format`.
- `mixed` also reads what dateutil reads for the common shapes: dates with
  slashes, hyphens or dots, month first unless the first number cannot be a
  month; month names and weekday names; a clock with `AM` or `PM`; a two digit
  year in the century nearest today; and a time alone, which is today.
- Leading spaces are dropped, and trailing ones too under `mixed`. The empty
  text, `NaT` and `nan` in any case are missing.
- The unit is microseconds, or nanoseconds when a row has more than six digits
  of fraction.
- Rows at more than one offset, or some at an offset and some at none, raise
  `Mixed timezones detected` unless `utc` is set.
"""

from __future__ import annotations

import datetime
import re
from typing import Any

from .errors import InvalidArgumentError

__all__ = ["DateParseError", "rows_as_text"]


class DateParseError(InvalidArgumentError):
    """Text `format="mixed"` cannot read as an instant, with pandas' own type name."""


_MISSING = frozenset({"", "nat", "nan"})

_ISO = re.compile(
    r"(\d{4})(?:-(\d{1,2})(?:-(\d{1,2}))?|(\d{2})(\d{2}))?"
    r"(?:[T ](\d{2})(?::?(\d{2})(?::?(\d{2})(?:[.,](\d+))?)?)?)?"
    r"(?: ?(Z|[+-]\d{2}(?::?\d{2})?))?"
)

_HINT = (
    " You might want to try:\n"
    "    - passing `format` if your strings have a consistent format;\n"
    "    - passing `format='ISO8601'` if your strings are all ISO8601 but not"
    " necessarily in exactly the same format;\n"
    "    - passing `format='mixed'`, and the format will be inferred for each element"
    " individually. You might want to use `dayfirst` alongside this."
)

_MIXED_ZONES = (
    "Mixed timezones detected. Pass utc=True in to_datetime or tz='UTC' in"
    " DatetimeIndex to convert to a common timezone."
)

_MONTHS = {
    name: number
    for number, names in enumerate(
        (
            ("jan", "january"),
            ("feb", "february"),
            ("mar", "march"),
            ("apr", "april"),
            ("may",),
            ("jun", "june"),
            ("jul", "july"),
            ("aug", "august"),
            ("sep", "sept", "september"),
            ("oct", "october"),
            ("nov", "november"),
            ("dec", "december"),
        ),
        start=1,
    )
    for name in names
}

_WEEKDAYS = frozenset(
    {
        *("mon", "tue", "tues", "wed", "thu", "thur", "thurs", "fri", "sat", "sun"),
        *("monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"),
    }
)

_UTC_WORDS = frozenset({"utc", "gmt", "z"})

_CLOCK = re.compile(
    r"(?<![\d.])(\d{1,2}):(\d{2})(?::(\d{2})(?:\.(\d+))?)?(?:\s*([ap])\.?m\.?)?(?![\w:])",
    re.IGNORECASE,
)
_HOUR_ONLY = re.compile(r"(?<![\d.:])(\d{1,2})\s*([ap])\.?m\.?(?!\w)", re.IGNORECASE)
_NUMBERS = re.compile(r"\d+(?:([/.-])\d+(?:\1\d+)?)?")


def missing(value: Any) -> bool:
    """Whether a row is missing: None, NaN, or text pandas reads as missing."""
    if value is None or (isinstance(value, float) and value != value):
        return True
    return isinstance(value, str) and value.strip().lower() in _MISSING


def _checked(parts: tuple[Any, ...]) -> tuple[Any, ...]:
    """The parts, after Python's datetime has checked every field is in range."""
    datetime.datetime(*parts[:6])
    return parts


def _iso_parts(text: str) -> tuple[Any, ...] | None:
    """A row's parts under ISO 8601 as pandas reads it, or None when it is not ISO 8601.

    The parts are the year, month, day, hour, minute and second, the fraction's
    digits as text, and the offset in minutes, or None for a row at no offset.
    """
    found = _ISO.fullmatch(text)
    if found is None:
        return None
    try:
        return _checked(_iso_fields(found))
    except ValueError:
        return None


def _offset(zone: str | None) -> int | None:
    """An offset written as `Z`, `+01`, `+0100` or `+01:00`, in minutes."""
    if zone is None:
        return None
    if zone == "Z":
        return 0
    digits = zone[1:].replace(":", "")
    minutes = int(digits[:2]) * 60 + int(digits[2:] or 0)
    return -minutes if zone[0] == "-" else minutes


def _century(year: int, today: datetime.date) -> int:
    """A two digit year in the century that puts it nearest today, as dateutil does."""
    year += today.year // 100 * 100
    if year >= today.year + 50:
        return year - 100
    if year < today.year - 50:
        return year + 100
    return year


def _unreadable(text: str) -> DateParseError:
    return DateParseError(f"Unknown datetime string format, unable to parse: {text}")


def _loose_parts(text: str, today: datetime.date) -> tuple[Any, ...]:
    """A row's parts the way dateutil reads them, for `format="mixed"`.

    Raises:
        DateParseError: For text with no reading, and for a field out of range.
        ValueError: For a zone written as a name other than UTC.
    """
    folded = " ".join(text.split())
    iso = _ISO.fullmatch(folded[:10] + folded[10:].upper()) if folded[:4].isdigit() else None
    rest = folded
    zone: int | None = None
    words = folded.split(" ")
    if len(words) > 1 and words[-1].lower() in _UTC_WORDS:
        zone, rest = 0, " ".join(words[:-1])
        iso = _ISO.fullmatch(rest.upper()) if rest[:4].isdigit() else None
    elif len(words) > 1 and re.fullmatch(r"[A-Z]{3,5}", words[-1]):
        raise InvalidArgumentError(
            f'Parsed string "{text}" included an un-recognized timezone "{words[-1]}".'
        )
    if iso is not None:
        try:
            parts = _checked(_iso_fields(iso))
        except ValueError as error:
            raise DateParseError(f"{error}: {text}") from None
        return parts if zone is None else (*parts[:7], zone)
    return _fielded(text, rest, zone, today)


def _iso_fields(found: re.Match[str]) -> tuple[Any, ...]:
    """The fields of an ISO 8601 match, before the range check."""
    year, month, day, packed_month, packed_day, hour, minute, second, fraction, zone = (
        found.groups()
    )
    return (
        int(year),
        int(month or packed_month or 1),
        int(day or packed_day or 1),
        int(hour or 0),
        int(minute or 0),
        int(second or 0),
        fraction or "",
        _offset(zone),
    )


def _fielded(text: str, rest: str, zone: int | None, today: datetime.date) -> tuple[Any, ...]:
    """The parts of a row that is not ISO 8601, read field by field."""
    hour = minute = second = 0
    fraction = ""
    clock = _CLOCK.search(rest) or _HOUR_ONLY.search(rest)
    if clock is not None:
        fields = clock.groups()
        hour = int(fields[0])
        if len(fields) == 2:
            meridiem = fields[1]
        else:
            minute, second, fraction = int(fields[1]), int(fields[2] or 0), fields[3] or ""
            meridiem = fields[4]
        if meridiem is not None:
            if not 1 <= hour <= 12:
                raise _unreadable(text)
            hour = hour % 12 + (12 if meridiem.lower() == "p" else 0)
        rest = rest[: clock.start()] + " " + rest[clock.end() :]
    month_named: int | None = None
    numbers: list[str] = []
    separated: list[str] | None = None
    for word in rest.replace(",", " ").split():
        lowered = word.lower().rstrip(".")
        if lowered in _MONTHS and month_named is None:
            month_named = _MONTHS[lowered]
        elif lowered in _WEEKDAYS:
            continue
        elif (number := _NUMBERS.fullmatch(word)) is not None and number.group(1):
            if separated is not None:
                raise _unreadable(text)
            separated = re.split(r"[/.-]", word)
        elif word.isdigit():
            numbers.append(word)
        else:
            raise _unreadable(text)
    if separated is not None:
        if numbers or month_named is not None:
            raise _unreadable(text)
        year, month, day = _dated(separated, today, text)
    elif month_named is not None:
        year, month, day = _named(month_named, numbers, today, text)
    elif len(numbers) == 1 and len(numbers[0]) in (6, 8):
        digits = numbers[0]
        cut = len(digits) - 4
        year = int(digits[:cut]) if cut == 4 else _century(int(digits[:cut]), today)
        month, day = int(digits[cut : cut + 2]), int(digits[cut + 2 :])
    elif not numbers and clock is not None:
        year, month, day = today.year, today.month, today.day
    else:
        raise _unreadable(text)
    parts = (year, month, day, hour, minute, second, fraction, zone)
    try:
        return _checked(parts)
    except ValueError as error:
        raise DateParseError(f"{error}: {text}") from None


def _dated(fields: list[str], today: datetime.date, text: str) -> tuple[int, int, int]:
    """A year, month and day from numbers written with separators, month first."""
    if len(fields) == 3:
        if len(fields[0]) == 4:
            year, month, day = (int(field) for field in fields)
            return year, month, day
        first, second, last = (int(field) for field in fields)
        year = last if len(fields[2]) == 4 else _century(last, today)
        if first > 12 and second <= 12:
            first, second = second, first
        return year, first, second
    first, second = fields
    if len(second) == 4 and len(first) <= 2:
        return int(second), int(first), 1
    if len(first) == 4 and len(second) <= 2:
        return int(first), int(second), 1
    if len(first) <= 2 and len(second) <= 2:
        return today.year, int(first), int(second)
    raise _unreadable(text)


def _named(month: int, numbers: list[str], today: datetime.date, text: str) -> tuple[int, int, int]:
    """A year and day from the numbers written beside a month's name."""
    year: int | None = None
    day: int | None = None
    for number in numbers:
        if len(number) == 4 or int(number) > 31 or day is not None:
            if year is not None:
                raise _unreadable(text)
            year = int(number) if len(number) == 4 else _century(int(number), today)
        else:
            day = int(number)
    return today.year if year is None else year, month, 1 if day is None else day


def _shift(parts: tuple[Any, ...]) -> tuple[Any, ...]:
    """A row at an offset moved to UTC, with no offset, fraction kept."""
    moment = datetime.datetime(*parts[:6]) - datetime.timedelta(minutes=parts[7])
    fields = moment.timetuple()[:6]
    return (*fields, parts[6], None)


def rows_as_text(values: list[Any], mixed: bool, coerce: bool, utc: bool) -> list[str | None]:
    """Every row read with its own format, written back in the one shape the core reads.

    Args:
        values: The rows, text or missing.
        mixed: True for `format="mixed"` and False for `format="ISO8601"`.
        coerce: Whether a row that will not read is missing rather than an error.
        utc: Whether rows at different offsets are read against UTC.

    Returns:
        One text per row, None where the row is missing, all in one shape: the
        date, a `T`, the time with six or nine digits of fraction, and the
        offset when the column has one.

    Raises:
        ValueError: For a row that will not read under `ISO8601`, and for rows at
            more than one offset without `utc`.
        DateParseError: For a row that will not read under `mixed`.
    """
    today = datetime.date.today()
    read: dict[str, tuple[Any, ...] | None] = {}
    rows: list[tuple[Any, ...] | None] = []
    for value in values:
        if missing(value):
            rows.append(None)
            continue
        text = value.strip() if mixed else value.lstrip()
        if text not in read:
            if mixed:
                try:
                    read[text] = _loose_parts(text, today)
                except ValueError:
                    if not coerce:
                        raise
                    read[text] = None
            else:
                read[text] = _iso_parts(text)
                if read[text] is None and not coerce:
                    raise InvalidArgumentError(f"Time data {value} is not ISO8601 format.{_HINT}")
        rows.append(read[text])
    present = [row for row in rows if row is not None]
    zones = {row[7] for row in present}
    if len(zones) > 1 and not utc:
        raise InvalidArgumentError(_MIXED_ZONES)
    if utc:
        rows = [row if row is None or row[7] is None else _shift(row) for row in rows]
    digits = 9 if any(len(row[6]) > 6 for row in present) else 6
    zone = None if utc or not zones else zones.pop()
    written = "" if zone is None else "Z" if zone == 0 else _written_offset(zone)
    texts: list[str | None] = []
    for row in rows:
        if row is None:
            texts.append(None)
            continue
        year, month, day, hour, minute, second, fraction = row[:7]
        texts.append(
            f"{year:04d}-{month:02d}-{day:02d}T{hour:02d}:{minute:02d}:{second:02d}"
            f".{fraction[:digits]:0<{digits}}{written}"
        )
    return texts


def _written_offset(minutes: int) -> str:
    sign = "-" if minutes < 0 else "+"
    hours, rest = divmod(abs(minutes), 60)
    return f"{sign}{hours:02d}:{rest:02d}"
