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

__all__ = ["DataFrameMixin"]


class DataFrameMixin:
    """The hand written half of `DataFrame`."""

    __slots__ = ()

    _inner: _firepanda.DataFrame
    """Declared for the type checker. The generated subclass is what holds it,
    and its `__slots__` is where the storage comes from."""

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
                return Series(self._inner.column(key))
            if isinstance(key, (list, tuple)) and all(isinstance(k, str) for k in key):
                return DataFrame(self._inner.select(list(key)))
        except Exception as error:
            raise translate(error) from None
        raise TypeError(
            f"cannot select with a {type(key).__name__}; df[key] reads a column"
            " name or a list of column names"
        )
