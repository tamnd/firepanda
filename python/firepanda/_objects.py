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
- `p` is anything else, pickled and written in base64, so a dict, a set, a time
  or an instant comes back as the object it was.

Each value ends with U+0001. A gap is a gap of the text column, not a cell.
"""

from __future__ import annotations

import base64
import math
import pickle
from typing import Any

from ._levels import _END, _ESCAPES, _float, _read, _unfloat

MARK = "\x1c"
_CLOSE = {"[": "]", "(": ")"}
_KINDS = frozenset("sibftdnp[(")


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


def cell(value: Any) -> Any:
    """One value as the text an object column holds, or None for a gap."""
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    return MARK + _written(value)


def cells(values: Any) -> list[Any]:
    """Every value of a list as the text an object column holds."""
    return [cell(value) for value in values]


def is_cell(text: Any) -> bool:
    """Whether a text is a written cell: the mark, a kind, and the end mark last.

    Text that only starts with U+001C is not taken for one, since that character
    is a separator somebody can have in their data.
    """
    return (
        isinstance(text, str)
        and len(text) >= 3
        and text[0] == MARK
        and text[1] in _KINDS
        and text[-1] == _END
    )


def value(text: Any) -> Any:
    """The value a written cell holds, or the text itself when it was not written."""
    if is_cell(text):
        return _parsed(text, 1)[0]
    return text


def values(texts: list[Any]) -> list[Any]:
    """Every cell of a list as the value it holds."""
    return [value(text) for text in texts]


def is_object(inner: Any) -> bool:
    """Whether an extension column holds written values, which is its first value's mark."""
    if inner.dtype() != "string":
        return False
    rows = inner.length()
    if inner.null_count() == rows:
        return False
    # A column reads one value with `cell` and an index with `at`.
    read = inner.cell if hasattr(inner, "cell") else inner.at
    for at in range(rows):
        first = read(at)
        if first is not None:
            return is_cell(first)
    return False
