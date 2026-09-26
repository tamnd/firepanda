"""Reading a column of text against one format, the way pandas reads it.

The core reads text against a format too, and it reads the common case in one
pass. It guesses ISO 8601 and nothing else, and it has no month names and no
twelve hour clock, so a column it cannot read comes here, where the rows are
read the way pandas reads them and written back in one shape for the core.

The rules were measured against pandas 3.0.

- With no format, pandas guesses one from the first value that is not None,
  with dateutil's lexer and dateutil's reading of that value. The guess holds
  every other row. A first value it cannot guess from, which includes `NaT`
  and the empty text, sends the whole column to dateutil one row at a time,
  which is `format="mixed"`.
- A row is read with the regular expressions of Python's `strptime`, with
  pandas' stricter ones for the day, the month, the hour, the minute and the
  second, nine digits of fraction, and its own offset. Letters match in any
  case and any run of spaces in the format matches any run of spaces.
- A row that does not match is `time data "..." doesn't match format "..."`,
  a row with text left over is `unconverted data remains when parsing with
  format "...": "..."`, and a day past the end of its month says which month
  and year. Each of the three ends with pandas' three suggestions.
- `now` and `today` are read as the moment they are read.
- A zone name is read as the offset it has at that reading, so a column of
  one zone name across a daylight saving change is at two offsets.
"""

from __future__ import annotations

import calendar
import datetime
import re
import zoneinfo
from functools import cache
from typing import Any

from ._row_dates import _HINT, _loose_parts, missing
from .errors import InvalidArgumentError

__all__ = ["guess", "readable", "rows_by_format"]

_WEEKDAY_NAMES = list(calendar.day_name)
_WEEKDAY_ABBREVIATIONS = list(calendar.day_abbr)
_MONTH_NAMES = list(calendar.month_name)[1:]
_MONTH_ABBREVIATIONS = list(calendar.month_abbr)[1:]


def _alternatives(words: list[str], name: str) -> str:
    """Words as one group, longest first so a short word does not win over a long one."""
    ordered = sorted((word.lower() for word in words if word), key=len, reverse=True)
    return f"(?P<{name}>{'|'.join(re.escape(word) for word in ordered)})"


_DIRECTIVES = {
    "Y": r"(?P<Y>\d\d\d\d)",
    "y": r"(?P<y>\d\d)",
    "m": r"(?P<m>1[0-2]|0[1-9]|[1-9])",
    "d": r"(?P<d>3[01]|[12]\d|0[1-9]|[1-9])",
    "H": r"(?P<H>2[0-3]|[0-1]\d|\d)",
    "I": r"(?P<I>1[0-2]|0[1-9]|[1-9])",
    "M": r"(?P<M>[0-5]\d|\d)",
    "S": r"(?P<S>6[0-1]|[0-5]\d|\d)",
    "f": r"(?P<f>[0-9]{1,9})",
    "j": r"(?P<j>36[0-6]|3[0-5]\d|[12]\d\d|0[1-9]\d|00[1-9]|[1-9]\d|0[1-9]|[1-9])",
    "z": r"(?P<z>[+-]\d\d:?[0-5]\d(:?[0-5]\d(\.\d{1,6})?)?|(?-i:Z))",
    "a": _alternatives(_WEEKDAY_ABBREVIATIONS, "a"),
    "A": _alternatives(_WEEKDAY_NAMES, "A"),
    "b": _alternatives(_MONTH_ABBREVIATIONS, "b"),
    "B": _alternatives(_MONTH_NAMES, "B"),
    "p": _alternatives(["am", "pm"], "p"),
    "%": "%",
}


@cache
def _zone_names() -> str:
    return _alternatives([*zoneinfo.available_timezones(), "UTC", "GMT"], "Z")


def readable(fmt: str) -> bool:
    """Whether every directive in a format is one this module reads."""
    directives = re.findall(r"%(.)", fmt)
    return all(one in _DIRECTIVES or one == "Z" for one in directives)


@cache
def _pattern(fmt: str) -> re.Pattern[str]:
    """A format as the regular expression Python's `strptime` builds, with pandas' overrides."""
    escaped = re.sub(r"([\\.^$*+?(){}\[\]|])", r"\\\1", fmt)
    escaped = re.sub(r"\s+", r"\\s+", escaped)
    out = []
    rest = escaped
    while "%" in rest:
        at = rest.index("%")
        directive = rest[at + 1]
        out.append(rest[:at])
        if directive == "f" and _iso(fmt):
            out.append(r"(?P<f>[0-9]+)")
        else:
            out.append(_zone_names() if directive == "Z" else _DIRECTIVES[directive])
        rest = rest[at + 2 :]
    out.append(rest)
    return re.compile("".join(out), re.IGNORECASE)


def _iso(fmt: str) -> bool:
    """Whether pandas reads a format with its ISO 8601 parser first, which is `format_is_iso`.

    That parser takes a fraction of any length and keeps nine digits, where the
    regular expression stops at nine, and it is what pandas tries first for a
    format that begins the way an ISO 8601 reading does.
    """
    for date in (" ", "/", "\\", "-", ".", ""):
        for time in (" ", "T"):
            for tail in ("", "%z", ".%f", ".%f%z"):
                whole = f"%Y{date}%m{date}%d{time}%H:%M:%S{tail}"
                if whole.startswith(fmt) and fmt != "%Y%m":
                    return True
    return False


def _mistake(words: str) -> InvalidArgumentError:
    return InvalidArgumentError(f"{words}.{_HINT}")


def _now() -> tuple[Any, ...]:
    moment = datetime.datetime.now()
    return (*moment.timetuple()[:6], f"{moment.microsecond:06d}", None)


def _parts(value: str, fmt: str) -> tuple[Any, ...]:
    """One row's parts under a format, or pandas' error for a row that does not read."""
    if value in ("now", "today"):
        return _now()
    found = _pattern(fmt).match(value)
    if found is None:
        raise _mistake(f'time data "{value}" doesn\'t match format "{fmt}"')
    if found.end() != len(value):
        raise _mistake(
            f'unconverted data remains when parsing with format "{fmt}": "{value[found.end() :]}"'
        )
    fields = {key: text for key, text in found.groupdict().items() if text is not None}
    year, month, day, hour, minute, second = 1900, 1, 1, 0, 0, 0
    fraction = ""
    zone = None
    if "Y" in fields:
        year = int(fields["Y"])
    elif "y" in fields:
        year = int(fields["y"])
        year += 2000 if year <= 68 else 1900
    if "m" in fields:
        month = int(fields["m"])
    elif "B" in fields:
        month = [name.lower() for name in _MONTH_NAMES].index(fields["B"].lower()) + 1
    elif "b" in fields:
        month = [name.lower() for name in _MONTH_ABBREVIATIONS].index(fields["b"].lower()) + 1
    if "d" in fields:
        day = int(fields["d"])
    if "H" in fields:
        hour = int(fields["H"])
    elif "I" in fields:
        hour = int(fields["I"]) % 12
        if fields.get("p", "").lower() == "pm":
            hour += 12
    minute = int(fields.get("M", 0))
    second = int(fields.get("S", 0))
    fraction = fields.get("f", "")[:9]
    if "j" in fields and not {"m", "B", "b", "d"} & fields.keys():
        ordinal = datetime.date(year, 1, 1) + datetime.timedelta(days=int(fields["j"]) - 1)
        month, day = ordinal.month, ordinal.day
    last = calendar.monthrange(year, month)[1]
    if day > last:
        raise _mistake(f"day {day} must be in range 1..{last} for month {month} in year {year}")
    moment = datetime.datetime(year, month, day, hour, minute) + datetime.timedelta(seconds=second)
    if "z" in fields:
        zone = _minutes(fields["z"])
    elif "Z" in fields:
        zone = _named_offset(fields["Z"], moment)
    return (*moment.timetuple()[:6], fraction, zone)


def _minutes(written: str) -> int:
    """An offset as `%z` writes it, in minutes, seconds dropped."""
    if written == "Z":
        return 0
    digits = written[1:].replace(":", "")
    minutes = int(digits[:2]) * 60 + int(digits[2:4])
    return -minutes if written[0] == "-" else minutes


def _named_offset(name: str, moment: datetime.datetime) -> int:
    """The offset a zone name has at a reading on its clock, in minutes."""
    if name.upper() in ("UTC", "GMT"):
        return 0
    zone = zoneinfo.ZoneInfo(next(key for key in _zone_keys() if key.lower() == name.lower()))
    offset = moment.replace(tzinfo=zone).utcoffset()
    return int(offset.total_seconds()) // 60 if offset is not None else 0


@cache
def _zone_keys() -> frozenset[str]:
    return frozenset(zoneinfo.available_timezones())


def rows_by_format(values: list[Any], fmt: str, coerce: bool) -> list[tuple[Any, ...] | None]:
    """Every row read against one format, the way pandas reads it.

    Args:
        values: The rows, text or missing.
        fmt: The format, given or guessed.
        coerce: Whether a row that will not read is missing rather than an error.

    Returns:
        The parts of every row, None where it is missing or did not read.

    Raises:
        InvalidArgumentError: pandas' error for the first row that does not read.
    """
    read: dict[str, tuple[Any, ...] | None] = {}
    rows: list[tuple[Any, ...] | None] = []
    for value in values:
        if missing(value):
            rows.append(None)
            continue
        if value not in read:
            try:
                read[value] = _parts(value, fmt)
            except InvalidArgumentError:
                if not coerce:
                    raise
                read[value] = None
        rows.append(read[value])
    return rows


def _tokens(text: str) -> list[str]:
    """Text split the way dateutil's lexer splits it.

    Runs of letters and runs of digits are tokens, a digit run may carry one
    decimal point or comma, every space is a token of one space, and anything
    else is a token of its own. A run with more than one point, or letters and
    points together, is split again at the points.
    """
    tokens: list[str] = []
    at = 0
    while at < len(text):
        char = text[at]
        if char.isspace():
            tokens.append(" ")
            at += 1
            continue
        if not (char.isalpha() or char.isdigit()):
            tokens.append(char)
            at += 1
            continue
        state = "a" if char.isalpha() else "0"
        token = char
        seen_letters = False
        at += 1
        while at < len(text):
            char = text[at]
            if state == "a":
                seen_letters = True
                if char.isalpha():
                    token += char
                elif char == ".":
                    token += char
                    state = "a."
                else:
                    break
            elif state == "0":
                if char.isdigit():
                    token += char
                elif char == "." or (char == "," and len(token) >= 2):
                    token += char
                    state = "0."
                else:
                    break
            elif state == "a.":
                seen_letters = True
                if char == "." or char.isalpha():
                    token += char
                elif char.isdigit() and token[-1] == ".":
                    token += char
                    state = "0."
                else:
                    break
            else:
                if char == "." or char.isdigit():
                    token += char
                elif char.isalpha() and token[-1] == ".":
                    token += char
                    state = "a."
                else:
                    break
            at += 1
        if state in ("a.", "0.") and (seen_letters or token.count(".") > 1 or token[-1] in ".,"):
            pieces = re.split(r"([.,])", token)
            tokens.append(pieces[0])
            tokens.extend(piece for piece in pieces[1:] if piece)
            continue
        if state == "0." and "." not in token:
            token = token.replace(",", ".")
        tokens.append(token)
    return tokens


def _filled(token: str, padding: int) -> str:
    """A token with the zeros dateutil's reading would print, as pandas pads it."""
    if re.search(r"\d+\.\d+", token) is None:
        return token.zfill(padding)
    seconds, fraction = token.split(".")
    return f"{int(seconds):02d}.{fraction.ljust(9, '0')[:6]}"


_GUESSES: list[tuple[tuple[str, ...], str, int]] = [
    (("year", "month", "day", "hour", "minute", "second"), "%Y%m%d%H%M%S", 0),
    (("year", "month", "day", "hour", "minute"), "%Y%m%d%H%M", 0),
    (("year", "month", "day", "hour"), "%Y%m%d%H", 0),
    (("year", "month", "day"), "%Y%m%d", 0),
    (("hour", "minute", "second"), "%H%M%S", 0),
    (("hour", "minute"), "%H%M", 0),
    (("year",), "%Y", 0),
    (("month",), "%B", 0),
    (("month",), "%b", 0),
    (("month",), "%m", 2),
    (("day",), "%d", 2),
    (("hour",), "%H", 2),
    (("minute",), "%M", 2),
    (("second",), "%S", 2),
    (("second", "microsecond"), "%S.%f", 0),
    (("tzinfo",), "%z", 0),
    (("tzinfo",), "%Z", 0),
    (("day_of_week",), "%a", 0),
    (("day_of_week",), "%A", 0),
    (("meridiem",), "%p", 0),
]


def _printed(moment: datetime.datetime, zone: int | None, fmt: str) -> str:
    """strftime, with the zone dateutil would have attached."""
    if fmt == "%Z":
        return "UTC" if zone == 0 else ""
    if fmt == "%z":
        sign = "-" if zone is not None and zone < 0 else "+"
        hours, minutes = divmod(abs(zone or 0), 60)
        return f"{sign}{hours:02d}{minutes:02d}"
    return moment.strftime(fmt)


def guess(text: str) -> str | None:
    """The format pandas guesses from one value, or None when it cannot guess one.

    This is pandas' `guess_datetime_format`. The value is read the way dateutil
    reads it, split into dateutil's tokens, and each token is matched against
    each part of the reading printed in each directive, in pandas' order of
    preference. The guess has to cover a year, a month and a day, except for a
    year alone and a year and month with a hyphen, and has to read the value
    back to the same text.
    """
    today = datetime.date.today()
    try:
        parts = _loose_parts(text.strip(), today)
    except ValueError:
        return None
    year, month, day, hour, minute, second, fraction, zone = parts
    micro = int((fraction or "0")[:6].ljust(6, "0"))
    moment = datetime.datetime(year, month, day, hour, minute, second, micro)
    tokens = _tokens(text)
    if zone is not None:
        index = None
        if tokens and tokens[-1] == "Z":
            index = -1
        elif len(tokens) > 1 and tokens[-2] in ("+", "-"):
            index = -2
        elif len(tokens) > 3 and tokens[-4] in ("+", "-"):
            index = -4
        if index is not None:
            tokens[index] = _printed(moment, zone, "%z")
            tokens = tokens[: index + 1 or None]
    formats: list[str | None] = [None] * len(tokens)
    found: set[str] = set()
    for attrs, directive, padding in _GUESSES:
        if set(attrs) & found:
            continue
        if zone is None and directive in ("%z", "%Z"):
            continue
        if directive in ("%p",) and not re.search(r"[ap]\.?m\b", text, re.IGNORECASE):
            continue
        printed = _printed(moment, zone, directive)
        for i, token in enumerate(tokens):
            filled = _filled(token, padding)
            if formats[i] is None and filled == printed:
                formats[i] = directive
                tokens[i] = filled
                found.update(attrs)
                break
    if (
        len({"year", "month", "day"} & found) != 3
        and formats != ["%Y"]
        and not (formats == ["%Y", None, "%m"] and tokens[1] == "-")
    ):
        return None
    out: list[str] = []
    for token, directive in zip(tokens, formats, strict=True):
        if directive is not None:
            out.append(directive)
            continue
        try:
            float(token)
            return None
        except ValueError:
            out.append(token)
    if "%p" in out and "%H" in out:
        out[out.index("%H")] = "%I"
    fmt = "".join(out)
    try:
        _parts(text, fmt)
    except (InvalidArgumentError, ValueError):
        return None
    rebuilt = "".join(tokens)
    printed = "".join(
        _printed(moment, zone, piece) if piece.startswith("%") else piece for piece in _split(fmt)
    )
    return fmt if printed == rebuilt else None


def _split(fmt: str) -> list[str]:
    """A format as its directives and the literal text between them."""
    return [piece for piece in re.split(r"(%.)", fmt) if piece]
