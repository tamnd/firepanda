"""Column names and series names that are not text, held by the extension as text.

The extension names each column, and each series, with a piece of text. pandas
names them with any value, and a frame whose names are the whole numbers 0, 1,
2 comes out of `transpose`, `pivot`, `get_dummies`, `str.split` and building a
frame from rows, so a name that is not text is written into text on the way in
and read back on the way out. Document 94 of the compat notes describes the
design and this module is the writing and the reading.

A written name starts with U+001D, then holds one value written the way
`_levels` writes one value of a tuple, with its kind and its end mark. A name
that is a tuple is written as `_levels` writes a whole row. Text is never
written, so a frame whose names are all text is held exactly as before.
"""

from __future__ import annotations

from typing import Any

from ._levels import _END, _read, _value, is_written, read, written

MARK = "\x1d"


def held(name: Any) -> Any:
    """A name as the text the extension holds, or the name itself when it is text or None."""
    if name is None or isinstance(name, str):
        return name
    try:
        if isinstance(name, tuple):
            # A series named by a row of a `MultiIndex`, written as `_levels` writes a row.
            return written(name)
        return MARK + _value(name)
    except NotImplementedError:
        raise NotImplementedError(
            f"a name of type {type(name).__name__} is not supported yet, because only"
            " text, numbers, bools, instants and spans are written into names"
        ) from None


def shown(text: Any) -> Any:
    """The name a held text stands for, or the text itself when it was not written."""
    if isinstance(text, str) and text.startswith(MARK) and text.endswith(_END):
        return _read(text[1:-1])
    if is_written(text) and text.endswith(_END):
        return read(text)
    return text


def is_held(text: Any) -> bool:
    """Whether a text from the extension is a written name."""
    return isinstance(text, str) and (text.startswith(MARK) or is_written(text))


def held_all(names: Any) -> list[Any]:
    """Every name of a list as the text the extension holds."""
    return [held(name) for name in names]


def keys_held(keys: Any) -> list[Any]:
    """Every key that could be a name as held text, and the rest, such as arrays, as they are."""
    return [held(key) if isinstance(key, (str, int, float)) else key for key in keys]


def shown_all(texts: list[Any]) -> list[Any]:
    """Every text of a list as the name it stands for."""
    if not any(is_held(text) for text in texts):
        return texts
    return [shown(text) for text in texts]
