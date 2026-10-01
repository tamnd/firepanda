"""Assigning to an axis or a name, which pandas writes as `df.columns = [...]`.

The generated classes declare `index`, `columns` and `name` as properties with
a getter only, because the generator writes one getter per member. This module
gives each the setter pandas has. An assignment rebinds the extension object
underneath, which is what an `inplace` call does and document 51 argues for:
nothing that shares the old buffers can tell, and a column taken out of the
frame before the assignment keeps its old name, as it does in pandas.

An index read off a frame or a column remembers where it came from, so
`df.index.name = "r"` names the frame's rows and `s.index.name = "k"` names the
column's labels, as they do in pandas. A standalone index is renamed alone.
"""

from __future__ import annotations

import contextlib
from typing import Any

from . import _names
from ._attrs import column_names, hold_columns
from ._frame import DataFrame, Index, Series
from ._multi import MultiIndex
from .errors import InvalidArgumentError

__all__: list[str] = []


def _hashable(value: Any, owner: str) -> None:
    """Refuses a name pandas cannot hash, with pandas' message."""
    try:
        hash(value)
    except TypeError:
        raise TypeError(f"{owner}.name must be a hashable type") from None


def _width(owner: Any, axis: int) -> int:
    """How many labels an axis has, rows or columns."""
    return len(owner.index) if axis == 0 else len(owner.columns)


def _set_axis(owner: Any, labels: Any, axis: int) -> None:
    """Puts new labels on an axis, refusing a count that does not match as pandas does.

    Raises:
        ValueError: For labels of another length than the axis.
    """
    from ._pandas import _list_of_arrays

    if not hasattr(labels, "__len__"):
        labels = list(labels)
    if _list_of_arrays(labels):
        # pandas reads a list of arrays as the levels of a MultiIndex.
        labels = MultiIndex.from_arrays([list(array) for array in labels])
    expected, given = _width(owner, axis), len(labels)
    if expected != given:
        raise ValueError(
            f"Length mismatch: Expected axis has {expected} elements,"
            f" new values have {given} elements"
        )
    made = owner.set_axis(labels, axis=axis)
    owner._inner = made._inner
    owner._row_freq = getattr(made, "_row_freq", None)
    if axis == 1:
        hold_columns(owner, column_names(made))


def _owned(getter: Any, axis: int) -> Any:
    """A getter whose index remembers the frame or column and axis it was read off."""

    def read(self: Any) -> Any:
        made = getter(self)
        with contextlib.suppress(AttributeError):
            made._owner = (self, axis)
        return made

    read.__doc__ = getter.__doc__
    return read


def _name_owner(index: Any, names: list[Any]) -> None:
    """Names the axis of whatever an index was read off, when it was read off one."""
    owner = getattr(index, "_owner", None)
    if owner is None:
        return
    target, axis = owner
    value: Any = names if isinstance(index, MultiIndex) else names[0]
    made = target.rename_axis(value, axis=axis)
    target._inner = made._inner
    if axis == 1:
        hold_columns(target, column_names(made))


def _series_name(self: Any, value: Any) -> None:
    _hashable(value, "Series")
    self._inner = self._inner.relabel(_names.held(value))
    self._typed_name = None if value is None or isinstance(value, str) else value


def _series_index(self: Any, labels: Any) -> None:
    _set_axis(self, labels, 0)


def _frame_index(self: Any, labels: Any) -> None:
    _set_axis(self, labels, 0)


def _frame_columns(self: Any, labels: Any) -> None:
    _set_axis(self, labels, 1)


def _index_name(self: Any, value: Any) -> None:
    _hashable(value, "Index")
    self.rename(value, inplace=True)
    _name_owner(self, [value])


def _index_names(self: Any, names: Any) -> None:
    listed = list(names) if isinstance(names, (list, tuple)) else None
    if listed is None or len(listed) != 1:
        raise InvalidArgumentError(
            f"Length of new names must be 1, got {1 if listed is None else len(listed)}"
        )
    _index_name(self, listed[0])


_multi_names_setter = MultiIndex.names.fset


def _multi_names(self: Any, names: Any) -> None:
    _multi_names_setter(self, names)
    _name_owner(self, list(self._names))


def _frame_setattr(self: Any, name: str, value: Any) -> None:
    """`df.a = [...]` writes column `a` when the frame has one, as pandas does.

    A private name or anything the class defines, such as `index` or `attrs`,
    is set as an attribute. A frame has no `__dict__`, so any other name is
    refused with Python's `AttributeError`.
    """
    if name.startswith("_") or hasattr(type(self), name) or name not in self.columns:
        object.__setattr__(self, name, value)
    else:
        self[name] = value


Series.name = property(Series.name.fget, _series_name, doc=Series.name.__doc__)
Series.index = property(_owned(Series.index.fget, 0), _series_index, doc=Series.index.__doc__)
DataFrame.index = property(
    _owned(DataFrame.index.fget, 0), _frame_index, doc=DataFrame.index.__doc__
)
DataFrame.columns = property(
    _owned(DataFrame.columns.fget, 1), _frame_columns, doc=DataFrame.columns.__doc__
)
Index.name = property(Index.name.fget, _index_name, doc=Index.name.__doc__)
Index.names = property(Index.names.fget, _index_names, doc=Index.names.__doc__)
MultiIndex.names = property(MultiIndex.names.fget, _multi_names, doc=MultiIndex.names.__doc__)
DataFrame.__setattr__ = _frame_setattr
