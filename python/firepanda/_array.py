"""`FirepandaArray`, the answer pandas gives as an array rather than a column.

`Series.unique` in pandas hands back neither a Series nor a list. It hands back
an extension array for text, a categorical or a timestamp column, and a numpy
array for numbers, and `factorize` hands back a numpy array of codes. What those
have in common is that they are values with a type and nothing else: no row
labels and no name. This class is that shape. It holds one column and gives
out its values, its length, its type and its Arrow export, and nothing that
needs a label.

It is one class where pandas has several, and the reason is that the
difference between pandas' classes is how the values are stored, which is
Arrow for every column here. A consumer reading the values through the Arrow
PyCapsule protocol, which is what pyarrow, polars and DuckDB do, gets them
without a copy, as it does for a Series.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ._frame import Series


class FirepandaArray:
    """Values of one type in order, with no row labels and no name."""

    __slots__ = ("_column",)

    def __init__(self, column: Series) -> None:
        self._column = column.reset_index(drop=True).rename(None)

    @property
    def dtype(self) -> Any:
        """The type of the values, spelled the way a column spells it."""
        return self._column.dtype

    @property
    def shape(self) -> tuple[int]:
        """The length, as a tuple of one."""
        return (len(self._column),)

    @property
    def ndim(self) -> int:
        """One, since an array here is always flat."""
        return 1

    @property
    def size(self) -> int:
        """The number of values."""
        return len(self._column)

    def __len__(self) -> int:
        return len(self._column)

    def __iter__(self) -> Iterator[Any]:
        return iter(self._column.tolist())

    def __getitem__(self, key: Any) -> Any:
        """One value for a position, and an array for a slice, a list or a mask."""
        if isinstance(key, int):
            return self._column.iloc[key]
        if isinstance(key, FirepandaArray):
            key = key.tolist()
        return FirepandaArray(self._column.iloc[key])

    def __contains__(self, value: Any) -> bool:
        return value in self._column.tolist()

    def tolist(self) -> list[Any]:
        """The values as Python objects."""
        return self._column.tolist()

    def to_list(self) -> list[Any]:
        """The values as Python objects."""
        return self.tolist()

    def isna(self) -> FirepandaArray:
        """Whether each value is missing."""
        return FirepandaArray(self._column.isna())

    def unique(self) -> FirepandaArray:
        """Each value once, in the order first seen."""
        return self._column.unique()

    def factorize(self, use_na_sentinel: bool = True) -> tuple[FirepandaArray, FirepandaArray]:
        """The code of each value and the values the codes point at."""
        from ._pandas import factorize

        return factorize(self, use_na_sentinel=use_na_sentinel)

    def to_series(self) -> Series:
        """The values as a column labelled 0 to n minus 1."""
        return self._column.copy()

    def __arrow_c_schema__(self) -> object:
        return self._column.__arrow_c_schema__()

    def __arrow_c_array__(self, requested_schema: object | None = None) -> tuple[object, ...]:
        return self._column.__arrow_c_array__(requested_schema)

    def __repr__(self) -> str:
        values = self._column.tolist()
        shown = ", ".join(repr(value) for value in values[:10])
        if len(values) > 10:
            shown += ", ..."
        return f"<FirepandaArray>\n[{shown}]\nLength: {len(values)}, dtype: {self.dtype}"
