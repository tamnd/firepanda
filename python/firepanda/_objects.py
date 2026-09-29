"""Columns of any Python value, pandas' object dtype, held by the extension as text.

The extension has no column that holds any value at all, and it has a text
column that every operation moving rows around carries unchanged. So an object
column is a text column whose cells are written values, and the values are
read back on the way out. Document 95 of the compat notes describes the design
and this module is the writing and the reading.

A written cell starts with U+001C, then holds one value with its kind:

- `s` is text, escaped as `_levels` escapes it.
- `i`, `f` and `b` are a whole number, a float and a bool, written as `_levels`
  writes them.
- `n` is None inside a list or a tuple, where it is a value rather than a gap.
- `[` and `(` are a list and a tuple, their values one after another and then
  `]` or `)`.
- `t` and `d` are firepanda's own instant and span, written as `_levels` writes
  them.
- `p` is anything else, pickled and written in base64, so a dict, a set, a time
  or an instant comes back as the object it was.

Each value ends with U+0001. A gap is a gap of the text column, not a cell.

pandas spells a gap in an object column the way the values got there. A list
somebody wrote keeps its None, and a column cast from floats, text or moments,
or a row read across typed columns, has NaN or NaT. So a cell may carry one
more letter between the mark and the kind, `N` when the column's gaps read as
NaN and `T` when they read as NaT, and every cell of a column carries the same
one. Moving rows carries the letter with the cells, and the gaps stay gaps of
the text column, so what counts as missing is still the extension's answer.
"""

from __future__ import annotations

import base64
import math
import pickle
from typing import Any

from ._levels import _END, _ESCAPES, _float, _read, _unfloat, _value

MARK = "\x1c"
_CLOSE = {"[": "]", "(": ")"}
_KINDS = frozenset("sibftdnp[(")
_SPELLINGS = frozenset("NT")


def _written(value: Any) -> str:
    """One value with its kind and its end mark."""
    kind = type(value)
    if kind is str:
        return "s" + "".join(_ESCAPES.get(c, c) for c in value) + _END
    if kind is bool:
        return "b" + ("1" if value else "0") + _END
    if kind is int and -(2**63) <= value < 2**63:
        return "i" + format(value + 2**63, "016x") + _END
    if kind is float:
        return "f" + _float(value) + _END
    if value is None:
        return "n" + _END
    if kind is list or kind is tuple:
        opening = "[" if kind is list else "("
        return opening + "".join(_written(item) for item in value) + _CLOSE[opening] + _END
    if hasattr(value, "item") and type(value).__module__ == "numpy" and value.ndim == 0:
        return _written(value.item())
    if kind.__name__ in ("Timestamp", "Timedelta") and kind.__module__ == "firepanda._scalars":
        # Written as a row label writes one, since pickling one goes through datetime's.
        return _value(value)
    return "p" + base64.b64encode(pickle.dumps(value, protocol=4)).decode("ascii") + _END


def _parsed(text: str, at: int) -> tuple[Any, int]:
    """The value written at `at` and where the next one starts."""
    kind = text[at]
    if kind in _CLOSE:
        items = []
        at += 1
        while text[at] != _CLOSE[kind]:
            item, at = _parsed(text, at)
            items.append(item)
        # The closing bracket and its end mark.
        at += 2
        return (items if kind == "[" else tuple(items)), at
    end = text.index(_END, at)
    part = text[at + 1 : end]
    if kind == "n":
        value = None
    elif kind == "f":
        value = _unfloat(part)
    elif kind == "p":
        value = pickle.loads(base64.b64decode(part))
    else:
        value = _read(kind + part)
    return value, end + 1


def is_gap(value: Any) -> bool:
    """Whether a value is one pandas counts as missing: None, NaN or NaT."""
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return True
    return type(value).__name__ == "NaTType"


def cell(value: Any, spelling: str = "") -> Any:
    """One value as the text an object column holds, or None for a gap.

    `spelling` is the letter that says how the column's gaps read, `N`, `T` or
    nothing for None.
    """
    if is_gap(value):
        return None
    return MARK + spelling + _written(value)


def cells(values: Any, spelling: str | None = None) -> list[Any]:
    """Every value of a list as the text an object column holds.

    Without a `spelling` the list's first gap decides it, since pandas keeps
    the None, NaN or NaT a list was written with.
    """
    values = list(values)
    if spelling is None:
        first = next((value for value in values if is_gap(value)), None)
        spelling = "" if first is None else "T" if type(first).__name__ == "NaTType" else "N"
    return [cell(value, spelling) for value in values]


def is_cell(text: Any) -> bool:
    """Whether a text is a written cell: the mark, a kind, and the end mark last.

    Text that only starts with U+001C is not taken for one, since that character
    is a separator somebody can have in their data.
    """
    if not isinstance(text, str) or len(text) < 3 or text[0] != MARK or text[-1] != _END:
        return False
    return text[1] in _KINDS or (text[1] in _SPELLINGS and len(text) >= 4 and text[2] in _KINDS)


def value(text: Any) -> Any:
    """The value a written cell holds, or the text itself when it was not written."""
    if is_cell(text):
        return _parsed(text, 2 if text[1] in _SPELLINGS else 1)[0]
    return text


def values(texts: list[Any]) -> list[Any]:
    """Every cell of a list as the value it holds."""
    return [value(text) for text in texts]


def _first(inner: Any) -> Any:
    """An extension column's first value that is not a gap, or None."""
    if inner.dtype() != "string":
        return None
    rows = inner.length()
    if inner.null_count() == rows:
        return None
    # A column reads one value with `cell` and an index with `at`.
    read = inner.cell if hasattr(inner, "cell") else inner.at
    for at in range(rows):
        first = read(at)
        if first is not None:
            return first
    return None


def is_object(inner: Any) -> bool:
    """Whether an extension column holds written values, which is its first value's mark."""
    return is_cell(_first(inner))


def spelling(text: Any) -> str:
    """The letter a written cell carries for its column's gaps, or nothing."""
    return text[1] if is_cell(text) and text[1] in _SPELLINGS else ""


def spelling_of(inner: Any) -> str | None:
    """The letter an object column's cells carry for its gaps, or None for a None gap."""
    return spelling(_first(inner)) or None


def gap_of(inner: Any) -> Any:
    """What a gap of an object column reads as: None, NaN or NaT."""
    letter = spelling(_first(inner))
    if letter == "N":
        return math.nan
    if letter == "T":
        from ._scalars import NaT

        return NaT
    return None


def spelled(values: list[Any], gap: Any) -> list[Any]:
    """Values read out of an object column with each gap spelled as `gap`."""
    if gap is None:
        return values
    return [gap if value is None else value for value in values]
