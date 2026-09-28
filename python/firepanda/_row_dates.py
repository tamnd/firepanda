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

__all__ = ["DateParseError", "rows_as_text", "written"]


class DateParseError(InvalidArgumentError):
    """Text `format="mixed"` cannot read as an instant, with pandas' own type name."""


_MISSING = frozenset({"", "nat", "nan"})

_ISO = re.compile(
    r"(\d{4})(?:-(\d{1,2})(?:-(\d{1,2}))?|(\d{2})(\d{2}))?"
    r"(?:[T ](\d{2})(?::?(\d{2})(?::?(\d{2})(?:[.,](\d+))?)?)?)?"
    r"(?: ?(Z|[+-]\d{2}(?::?\d{2})?))?"
)
# pandas' own ISO 8601 reader, stricter than dateutil in one way and looser in
# another: the date may be split by a slash, a dot, a space or a backslash as
# long as both splits are the same, and the fraction takes a dot and no comma.
_ISO_STRICT = re.compile(
    r"(\d{4})(?:([-/\\. ])(\d{1,2})(?:\2(\d{1,2}))?|(\d{2})(\d{2}))?"
    r"(?:[T ](\d{2})(?::?(\d{2})(?::?(\d{2})(?:\.(\d+))?)?)?)?"
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
_JOINED = re.compile(r"\w+([/.-])\w+(?:\1\w+)?")


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
    found = _ISO_STRICT.fullmatch(text)
    if found is None:
        return None
    groups = found.groups()
    try:
        return _checked(_iso_fields((groups[0], *groups[2:])))
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


def _loose_parts(
    text: str, today: datetime.date, dayfirst: bool = False, yearfirst: bool = False
) -> tuple[Any, ...]:
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
        fields = _iso_fields(iso.groups())
        if dayfirst and (iso.group(3) or iso.group(5)) and fields[2] <= 12:
            # dateutil reads a year first and two numbers as year, day, month
            # when asked for the day first, and pandas asks it for every row.
            fields = (fields[0], fields[2], fields[1], *fields[3:])
        try:
            parts = _checked(fields)
        except ValueError as error:
            raise DateParseError(f"{error}: {text}") from None
        return parts if zone is None else (*parts[:7], zone)
    return _fielded(text, rest, zone, today, dayfirst, yearfirst)


def _iso_fields(groups: tuple[Any, ...]) -> tuple[Any, ...]:
    """The fields of an ISO 8601 match's groups, before the range check."""
    year, month, day, packed_month, packed_day, hour, minute, second, fraction, zone = groups
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


def _fielded(
    text: str,
    rest: str,
    zone: int | None,
    today: datetime.date,
    dayfirst: bool = False,
    yearfirst: bool = False,
) -> tuple[Any, ...]:
    """The parts of a row that is not ISO 8601, read field by field as dateutil does."""
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
    ymd = _Ymd(text)
    for word in rest.replace(",", " ").split():
        lowered = word.lower().rstrip(".")
        if lowered in _MONTHS:
            ymd.add(str(_MONTHS[lowered]), "M")
        elif lowered in _WEEKDAYS:
            continue
        elif clock is not None and zone is None and re.fullmatch(r"[+-]\d{2}:?\d{2}", word):
            zone = _offset(word)
        elif _JOINED.fullmatch(word) is not None:
            for piece in re.split(r"[/.-]", word):
                name = piece.lower()
                if name in _MONTHS:
                    ymd.add(str(_MONTHS[name]), "M")
                elif piece.isdigit():
                    ymd.add(piece)
                else:
                    raise _unreadable(text)
        elif word.isdigit() and len(word) == 8:
            ymd.add(word[:4], "Y")
            ymd.add(word[4:6])
            ymd.add(word[6:])
        elif word.isdigit() and len(word) == 6 and not ymd.values:
            ymd.add(word[:2])
            ymd.add(word[2:4])
            ymd.add(word[4:])
        elif word.isdigit() and len(word) <= 4:
            ymd.add(word)
        else:
            raise _unreadable(text)
    if not ymd.values:
        if clock is None:
            raise _unreadable(text)
        year, month, day = today.year, today.month, today.day
    else:
        found = ymd.resolved(dayfirst, yearfirst)
        year = 1 if found[0] is None else found[0]
        if found[0] is not None and found[0] < 100 and not ymd.century:
            year = _century(found[0], today)
        month = 1 if found[1] is None else found[1]
        day = 1 if found[2] is None else found[2]
    parts = (year, month, day, hour, minute, second, fraction, zone)
    try:
        return _checked(parts)
    except ValueError as error:
        raise DateParseError(f"{error}: {text}") from None


class _Ymd:
    """The numbers of a date in the order written, as dateutil's `_ymd` holds them."""

    def __init__(self, text: str) -> None:
        self.text = text
        self.values: list[int] = []
        self.places: dict[str, int] = {}
        self.century = False

    def add(self, written: str, label: str | None = None) -> None:
        """One number, labelled a year when it has more than two digits."""
        if len(written) > 2 and label is None:
            label = "Y"
        if len(written) > 2:
            self.century = True
        if label is not None and label in self.places:
            raise _unreadable(self.text)
        self.values.append(int(written))
        if label is not None:
            self.places[label] = len(self.values) - 1

    def resolved(self, dayfirst: bool, yearfirst: bool) -> tuple[Any, Any, Any]:
        """The year, month and day, any of them None, by dateutil's `resolve_ymd`."""
        values, places = self.values, self.places
        count = len(values)
        if count > 3:
            raise _unreadable(self.text)
        if count == len(places) or (count == 3 and len(places) == 2):
            places = dict(places)
            if count == 3 and len(places) == 2:
                left = next(i for i in range(3) if i not in places.values())
                places[next(key for key in "YMD" if key not in places)] = left
            return tuple(values[places[key]] if key in places else None for key in "YMD")
        month_at = places.get("M")
        if count == 1 or (month_at is not None and count == 2):
            month = values[month_at] if month_at is not None else None
            other = values[month_at - 1] if month_at is not None else values[0]
            if count > 1 or month_at is None:
                return (other, month, None) if other > 31 else (None, month, other)
            return None, month, None
        first, second = values[0], values[1]
        if count == 2:
            if first > 31:
                return first, second, None
            if second > 31:
                return second, first, None
            if dayfirst and second <= 12:
                return None, second, first
            return None, first, second
        last = values[2]
        if month_at == 0:
            return (second, first, last) if second > 31 else (last, first, second)
        if month_at == 1:
            if first > 31 or (yearfirst and last <= 31):
                return first, second, last
            return last, second, first
        if month_at == 2:
            return (second, last, first) if second > 31 else (first, last, second)
        if first > 31 or places.get("Y") == 0 or (yearfirst and second <= 12 and last <= 31):
            if dayfirst and last <= 12:
                return first, last, second
            return first, second, last
        if first > 12 or (dayfirst and second <= 12):
            return last, second, first
        return last, first, second


def _shift(parts: tuple[Any, ...]) -> tuple[Any, ...]:
    """A row at an offset moved to UTC, with no offset, fraction kept."""
    moment = datetime.datetime(*parts[:6]) - datetime.timedelta(minutes=parts[7])
    fields = moment.timetuple()[:6]
    return (*fields, parts[6], None)


def rows_as_text(
    values: list[Any],
    mixed: bool,
    coerce: bool,
    utc: bool,
    dayfirst: bool = False,
    yearfirst: bool = False,
) -> list[str | None]:
    """Every row read with its own format, written back in the one shape the core reads.

    Args:
        values: The rows, text or missing.
        mixed: True for `format="mixed"` and False for `format="ISO8601"`.
        coerce: Whether a row that will not read is missing rather than an error.
        utc: Whether rows at different offsets are read against UTC.
        dayfirst: Whether `format="mixed"` reads two small numbers day first.
        yearfirst: Whether `format="mixed"` reads three small numbers year first.

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
                    read[text] = _loose_parts(text, today, dayfirst, yearfirst)
                except ValueError:
                    if not coerce:
                        raise
                    read[text] = None
            else:
                read[text] = _iso_parts(text)
                if read[text] is None and not coerce:
                    raise InvalidArgumentError(f"Time data {value} is not ISO8601 format.{_HINT}")
        rows.append(read[text])
    return written(rows, utc)


def written(rows: list[tuple[Any, ...] | None], utc: bool) -> list[str | None]:
    """Rows read into their parts, written back in the one shape the core reads.

    Args:
        rows: The parts of every row, None where the row is missing.
        utc: Whether rows at different offsets are read against UTC.
        dayfirst: Whether `format="mixed"` reads two small numbers day first.
        yearfirst: Whether `format="mixed"` reads three small numbers year first.

    Returns:
        One text per row, None where the row is missing.

    Raises:
        ValueError: For rows at more than one offset without `utc`.
    """
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
