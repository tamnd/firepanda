"""Row labels made of several levels, held by the extension as one text label per row.

The extension holds one column of row labels, of one type. A frame or a column
labelled by a `MultiIndex` keeps its labels there all the same, by writing
each tuple as one piece of text that sorts the way the tuple sorts, so that
alignment, sorting, filtering and every other operation that carries labels
along carries the tuples along too. Document 93 of the compat notes describes
the design and this module is the writing and the reading.

Every written label starts with U+001E. Each value is one character naming its
kind followed by the value, and ends with U+0001, which sorts below every
character a value can hold:

- `s` is text, with U+0000, U+0001 and U+0002 escaped behind U+0002.
- `i` is a whole number as sixteen hex digits of the number plus 2 to the 63.
- `f` is a float as sixteen hex digits of its bits, made to sort as the float.
- `b` is a bool, `0` or `1`.
- `t` is an instant, its nanoseconds as `i` has them, then its unit and zone.
- `d` is a span, its nanoseconds as `i` has them, then its unit.
- `v` is an interval of numbers, each end as `i` or `f` has it behind its
  kind letter, then the letter of the side it is closed on.
- `~` is a gap, which sorts after every other kind.

The name of the labels is U+001E followed by the level names as JSON.
"""

from __future__ import annotations

import datetime as dt
import json
import math
import struct
from typing import Any

MARK = "\x1e"
_END = "\x01"
_ESCAPES = {"\x00": "\x02\x02", "\x01": "\x02\x03", "\x02": "\x02\x04"}
_UNESCAPES = {"\x02": "\x00", "\x03": "\x01", "\x04": "\x02"}
_SHIFT = 2**63
_MASK = 2**64 - 1


def _whole(value: int) -> str:
    """A whole number of the int64 range as sixteen hex digits that sort as it does."""
    if not -_SHIFT <= value < _SHIFT:
        raise OverflowError(f"a level value {value} does not fit in 64 bits")
    return format(value + _SHIFT, "016x")


def _float(value: float) -> str:
    """A float as sixteen hex digits of its bits, flipped so that they sort as it does."""
    (bits,) = struct.unpack(">Q", struct.pack(">d", value + 0.0))
    bits = (~bits & _MASK) if bits >> 63 else bits | _SHIFT
    return format(bits, "016x")


def _unfloat(digits: str) -> float:
    bits = int(digits, 16)
    bits = bits & ~_SHIFT & _MASK if bits >> 63 else ~bits & _MASK
    (value,) = struct.unpack(">d", struct.pack(">Q", bits))
    return value


def _gap(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, float):
        return math.isnan(value)
    return type(value).__name__ in ("NaTType", "NAType")


def _value(value: Any) -> str:
    """One value of a tuple, written with its kind and its end mark."""
    from ._scalars import Timedelta, Timestamp

    if _gap(value):
        return "~" + _END
    if hasattr(value, "item") and type(value).__module__ == "numpy":
        value = value.item()
    if isinstance(value, bool):
        return "b" + ("1" if value else "0") + _END
    if isinstance(value, int):
        return "i" + _whole(value) + _END
    if isinstance(value, float):
        return "f" + _float(value) + _END
    if isinstance(value, str):
        return "s" + "".join(_ESCAPES.get(c, c) for c in value) + _END
    if isinstance(value, dt.datetime):
        moment = Timestamp(value)
        zone = "" if moment.tz is None else str(moment.tz)
        return "t" + _whole(moment.value) + moment.unit + "|" + zone + _END
    if isinstance(value, dt.timedelta):
        span = Timedelta(value)
        return "d" + _whole(span.value) + span.unit + _END
    if type(value).__name__ == "Interval" and _numbers(value.left, value.right):
        ends = "".join(_end(end) for end in (value.left, value.right))
        return "v" + ends + _CLOSED[value.closed] + _END
    raise NotImplementedError(
        f"a level value of type {type(value).__name__} cannot label a row yet, because"
        " only text, numbers, bools, instants, spans and intervals of numbers are"
        " written into row labels"
    )


_CLOSED = {"right": "r", "left": "l", "both": "b", "neither": "n"}


def _numbers(*values: Any) -> bool:
    """Whether every value is a plain int or float, the ends an interval label can hold."""
    return all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in values)


def _end(value: Any) -> str:
    """One end of an interval as its kind letter and sixteen hex digits."""
    return "i" + _whole(value) if isinstance(value, int) else "f" + _float(value)


def _unend(part: str) -> Any:
    """One end of an interval back as the number it was."""
    return int(part[1:], 16) - _SHIFT if part[0] == "i" else _unfloat(part[1:])


def _read(part: str) -> Any:
    """One written value back as the value it was."""
    from ._scalars import Timedelta, Timestamp

    kind, rest = part[0], part[1:]
    if kind == "s":
        out, at = [], 0
        while at < len(rest):
            if rest[at] == "\x02":
                out.append(_UNESCAPES[rest[at + 1]])
                at += 2
            else:
                out.append(rest[at])
                at += 1
        return "".join(out)
    if kind == "i":
        return int(rest, 16) - _SHIFT
    if kind == "f":
        return _unfloat(rest)
    if kind == "b":
        return rest == "1"
    if kind == "t":
        nanos = int(rest[:16], 16) - _SHIFT
        unit, zone = rest[16:].split("|", 1)
        return Timestamp(nanos, unit="ns", tz=zone or None).as_unit(unit)
    if kind == "d":
        return Timedelta(int(rest[:16], 16) - _SHIFT, unit="ns").as_unit(rest[16:])
    if kind == "v":
        from ._interval import Interval

        closed = next(word for word, letter in _CLOSED.items() if letter == rest[34])
        return Interval(_unend(rest[:17]), _unend(rest[17:34]), closed)
    return math.nan


def written(row: Any) -> str:
    """A tuple as the one text label that holds it."""
    return MARK + "".join(_value(value) for value in row)


def read(label: Any) -> tuple[Any, ...]:
    """A written label back as its tuple."""
    return tuple(_read(part) for part in label[1:].split(_END)[:-1])


def is_written(label: Any) -> bool:
    """Whether a label is a written tuple."""
    return isinstance(label, str) and label.startswith(MARK)


def named(names: list[Any]) -> str:
    """The level names as the one name of the written labels."""
    return MARK + json.dumps(list(names), default=str)


def names_of(name: Any, count: int) -> list[Any]:
    """The level names out of the name of written labels, or none when it was lost."""
    if is_written(name):
        try:
            names = json.loads(name[1:])
        except ValueError:
            names = None
        if isinstance(names, list) and len(names) == count:
            return names
    return [None] * count


def labels_of(multi: Any) -> Any:
    """A `MultiIndex` as the text index that holds it in the extension."""
    from ._frame import Index

    return Index([written(row) for row in multi], name=named(multi.names))


def rows_of(columns: list[list[Any]], names: list[Any]) -> Any:
    """One list of values per level as the text index that holds their tuples."""
    from ._frame import Index

    return Index([written(row) for row in zip(*columns, strict=True)], name=named(names))


def multi_of(labels: Any) -> Any:
    """Text labels read back as the `MultiIndex` they hold, or None when they hold none."""
    from ._multi import MultiIndex

    if str(labels.dtype) not in ("string", "str"):
        return None
    if not is_written(labels.name) and (len(labels) == 0 or not is_written(labels[0])):
        return None
    rows = [read(label) for label in labels.tolist()]
    if rows:
        count = len(rows[0])
    else:
        try:
            count = len(json.loads(labels.name[1:]))
        except (TypeError, ValueError):
            count = 0
    columns = [list(level) for level in zip(*rows, strict=True)] if rows else [[]] * count
    return MultiIndex._of_columns(columns, names_of(labels.name, count))
