"""The members that are not a plain delegation.

`tools/bindings.py` generates a class per bound type out of one table, and every
member in that table is one expression written against `self._inner`. That is the
right shape for the surface it covers and it is not a shape logic fits into. The
note at the top of the generator says so, and asks that the first member needing
real logic go in a hand written base class rather than be smuggled into the table
as a longer expression, because the moment the table carries code it stops being
reviewable as a table.

This is that file. Everything here is inherited by a generated class, so what a
user holds is still the generated one, and the parity tests still walk the table
rather than this.

The rule for what belongs here is narrow on purpose. A member goes here when what
it does depends on its argument, and nowhere else. Anything that is one call with
a different name belongs in the table where it can be checked against pandas.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from . import _firepanda
from .errors import translate

if TYPE_CHECKING:
    from ._frame import DataFrame, Series


def _refuse(name: str, value: object, why: str) -> None:
    """Complains about a constructor argument that is declared and not honoured.

    The pandas signature is declared in full and four of its five parameters are
    not implemented, which is a deliberate choice document 18 section 4 argues
    for: a caller who passes `columns=` gets a message about `columns` rather
    than a TypeError about an unexpected keyword, and the day one of them is
    implemented no signature changes. This is what makes that honest, because a
    declared parameter that is silently ignored is worse than one that is
    missing.

    Args:
        name: The parameter name.
        value: What was passed.
        why: What it would take to support it.

    Raises:
        NotImplementedError: If anything other than None was passed.
    """
    if value is not None:
        raise NotImplementedError(f"{name}= is not supported yet, because {why}")


__all__ = ["DataFrameMixin", "SeriesMixin"]


class DataFrameMixin:
    """The hand written half of `DataFrame`."""

    __slots__ = ("_inner",)
    """The one piece of state, declared here rather than on the generated class.

    It has to be here because the constructor is here, and a class cannot assign
    to a slot it does not own. The generated subclass declares an empty
    `__slots__`, so an instance still has no `__dict__` and there is still
    exactly one place the extension object lives."""

    _inner: _firepanda.DataFrame

    def __init__(
        self,
        data: Any = None,
        index: Any = None,
        columns: Any = None,
        dtype: Any = None,
        copy: bool | None = None,
    ) -> None:
        """Builds a frame from a mapping of column name to values.

        The signature is the pandas one in full and only the first parameter is
        implemented. The other four are refused by name, which is a stronger
        statement than leaving them out: the signature parity test compares five
        parameters against pandas instead of one, and a caller who passes one of
        them is told what is missing rather than that the keyword is unexpected.
        """
        _refuse("index", index, "there is no Index type to put one in yet")
        _refuse("columns", columns, "selecting and reordering on the way in is not written")
        _refuse("dtype", dtype, "casting on the way in needs the cast machinery")
        _refuse("copy", copy, "there is exactly one behaviour and it always copies")
        try:
            self._inner = _firepanda.DataFrame(data)
        except Exception as error:
            raise translate(error) from None

    def __getitem__(self, key: Any) -> DataFrame | Series:
        """One column as a series, or several as a frame.

        This is the most written expression in pandas and it is two operations
        wearing one name, which is why it cannot be a row in the table. A string
        key takes a column out and a list of strings takes a frame out, and the
        difference is the argument rather than the method.

        pandas reads several other kinds of key here, including a boolean mask, a
        slice and a callable. Those are refused rather than approximated, with a
        message that says what is read today, because a key that quietly means
        something else is worse than one that does not work at all.
        """
        from ._frame import DataFrame, Series

        try:
            if isinstance(key, str):
                return Series._wrap(self._inner.column(key))
            if isinstance(key, (list, tuple)) and all(isinstance(k, str) for k in key):
                return DataFrame._wrap(self._inner.select(list(key)))
        except Exception as error:
            raise translate(error) from None
        raise TypeError(
            f"cannot select with a {type(key).__name__}; df[key] reads a column"
            " name or a list of column names"
        )


class SeriesMixin:
    """The hand written half of `Series`."""

    __slots__ = ("_inner",)
    """The one piece of state, for the reason `DataFrameMixin` gives."""

    _inner: _firepanda.Series

    def __init__(
        self,
        data: Any = None,
        index: Any = None,
        dtype: Any = None,
        name: Any = None,
        copy: bool | None = None,
    ) -> None:
        """Builds a series from a sequence of values.

        Same shape as the frame constructor and refusing the same way, with the
        one difference that `name` is honoured, since a series carries its name
        and there is nothing to implement.
        """
        _refuse("index", index, "there is no Index type to put one in yet")
        _refuse("dtype", dtype, "casting on the way in needs the cast machinery")
        _refuse("copy", copy, "there is exactly one behaviour and it always copies")
        try:
            self._inner = _firepanda.Series(data, "" if name is None else str(name))
        except Exception as error:
            raise translate(error) from None
