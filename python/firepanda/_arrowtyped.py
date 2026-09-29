"""Columns pandas backs with Arrow: `ArrowDtype`, and the `list` and `struct` accessors.

pandas keeps an `ArrowDtype` column as a pyarrow array and answers the list and
struct methods with pyarrow's compute functions. firepanda holds such a column
as an object column whose cells carry the Arrow type, which `_objects` writes,
and runs the same compute functions over the values when a method is asked
for. Document 98 of the compat notes describes the design.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from . import _objects

if TYPE_CHECKING:
    from ._frame import DataFrame, Series


class ArrowDtype(str):
    """The type of a column pandas backs with Arrow, named as pandas names it.

    Equal to its name, such as `int64[pyarrow]` or `list<item: int64>[pyarrow]`,
    and to another `ArrowDtype` of the same Arrow type.
    """

    def __new__(cls, pyarrow_dtype: Any) -> ArrowDtype:
        import pyarrow as pa

        if not isinstance(pyarrow_dtype, pa.DataType):
            raise ValueError(
                f"pyarrow_dtype ({pyarrow_dtype}) must be an instance of a pyarrow.DataType."
                f" Got {type(pyarrow_dtype)} instead."
            )
        self = super().__new__(cls, f"{_type_name(pyarrow_dtype)}[pyarrow]")
        self._pyarrow_dtype = pyarrow_dtype
        return self

    @property
    def pyarrow_dtype(self) -> Any:
        """The Arrow type."""
        return self._pyarrow_dtype

    @property
    def name(self) -> str:
        """The name, which is also what the type prints as."""
        return str.__str__(self)

    def __repr__(self) -> str:
        return self.name

    def __hash__(self) -> int:
        return str.__hash__(self)

    def __eq__(self, other: object) -> bool:
        if isinstance(other, ArrowDtype):
            return self._pyarrow_dtype == other._pyarrow_dtype
        return isinstance(other, str) and str.__eq__(self, other)

    def __ne__(self, other: object) -> bool:
        return not self == other


def _type_name(arrow_type: Any) -> str:
    """An Arrow type's name as pandas prints it, where `large_string` is still `large_string`."""
    return str(arrow_type)


def arrow_type_of(column: Series) -> Any:
    """The Arrow type of a column pandas would back with Arrow, or None."""
    return _objects.arrow_type_of(column._inner)


def arrow_values(inner: Any) -> list[Any]:
    """The values of an extension column, a gap as None, as pyarrow reads them.

    In a column backed by Arrow only a gap is None, and a NaN among floats is a
    value, as it is in pandas. In any other column NaN and NaT are gaps too.
    """
    from ._pandas import _values_of

    values = [_plain(value) for value in _values_of(inner)]
    if _objects.arrow_type_of(inner) is not None:
        return values
    return [None if _objects.is_gap(value) else value for value in values]


def arrow_series(array: Any, index: Any = None, name: Any = None) -> Series:
    """A pyarrow array as a column of its own Arrow type."""
    from ._frame import Series

    cells = _objects.arrow_cells(array.to_pylist(), array.type)
    return Series(cells, dtype="str", index=index, name=name)


def as_arrow(column: Series, dtype: ArrowDtype) -> Series:
    """A column cast to an Arrow type, through pyarrow, which raises what pandas raises."""
    import pyarrow as pa

    array = pa.array(arrow_values(column._inner), type=dtype.pyarrow_dtype)
    return arrow_series(array, column.index, column.name)


def _plain(value: Any) -> Any:
    """A value as pyarrow reads it, so firepanda's own gap is None."""
    return None if type(value).__name__ == "NAType" else value


def _array(column: Series) -> Any:
    """A column of an Arrow type as a pyarrow array."""
    import pyarrow as pa

    return pa.array(arrow_values(column._inner), type=arrow_type_of(column))


def arrow_text(value: Any) -> str:
    """One value of a column backed by Arrow as pandas prints it.

    A list prints as numpy prints the array pandas turns it into, its items
    apart by a space, and everything else prints with `str`.
    """
    if isinstance(value, list):
        return "[" + " ".join(arrow_text(item) for item in value) + "]"
    return str(value)


class ListAccessor:
    """pandas' `Series.list`, over a column of an Arrow list type."""

    def __init__(self, data: Series = None) -> None:  # type: ignore[assignment]
        import pyarrow as pa

        kind = arrow_type_of(data)
        if kind is None or not (pa.types.is_list(kind) or pa.types.is_large_list(kind)):
            raise AttributeError(
                f"Can only use the '.list' accessor with 'list[pyarrow]' dtype, not {data.dtype}."
            )
        self._data = data

    def len(self) -> Series:
        """The length of each list, a gap for a missing one."""
        import pyarrow.compute as pc

        lengths = pc.list_value_length(_array(self._data))
        return arrow_series(lengths, self._data.index, self._data.name)

    def __getitem__(self, key: Any) -> Series:
        """The item at a position of each list, or a slice of each list.

        Raises:
            ArrowInvalid: For a position past the end of any list, as pandas raises.
        """
        import pyarrow.compute as pc

        array = _array(self._data)
        if isinstance(key, int):
            if key < 0:
                raise NotImplementedError("Negative indexing is not supported yet")
            picked = pc.list_element(array, key)
        elif isinstance(key, slice):
            if key.step == 0:
                raise ValueError("slice step cannot be zero.")
            start = 0 if key.start is None else key.start
            picked = pc.list_slice(array, start, key.stop, key.step or 1)
        else:
            raise ValueError(f"key must be an int or slice, got {type(key).__name__}")
        return arrow_series(picked, self._data.index, self._data.name)

    def __iter__(self) -> Any:
        raise TypeError(f"'{type(self).__name__}' object is not iterable")

    def flatten(self) -> Series:
        """Every item of every list, each under the label of the row it came from."""
        import pyarrow.compute as pc

        array = _array(self._data)
        items = pc.list_flatten(array)
        parents = pc.list_parent_indices(array).to_pylist()
        index = self._data.index.take(parents)
        return arrow_series(items, index, self._data.name)


class StructAccessor:
    """pandas' `Series.struct`, over a column of an Arrow struct type."""

    def __init__(self, data: Series = None) -> None:  # type: ignore[assignment]
        import pyarrow as pa

        kind = arrow_type_of(data)
        if kind is None or not pa.types.is_struct(kind):
            raise AttributeError(
                f"Can only use the '.struct' accessor with 'struct[pyarrow]' dtype, not"
                f" {data.dtype}."
            )
        self._data = data
        self._type = kind

    @property
    def dtypes(self) -> Series:
        """The type of each field, labelled by the field's name."""
        from ._frame import Index, Series

        fields = [self._type.field(i) for i in range(self._type.num_fields)]
        kinds = [ArrowDtype(field.type) for field in fields]
        return Series(
            _objects.cells([str(kind) for kind in kinds]),
            dtype="str",
            index=Index([field.name for field in fields]),
        )

    def field(self, name_or_index: Any) -> Series:
        """One field of each struct, found by name, by position or by a path of either.

        The path goes to pyarrow as it is, so a name the struct has no field of
        raises pyarrow's `ArrowInvalid`, as it does in pandas.
        """
        import pyarrow.compute as pc

        path = name_or_index if isinstance(name_or_index, list) else [name_or_index]
        picked = pc.struct_field(_array(self._data), path)
        kind, name = self._type, None
        for step in path:
            place = step if isinstance(step, int) else kind.get_field_index(step)
            name = kind.field(place).name
            kind = kind.field(place).type
        return arrow_series(picked, self._data.index, name)

    def explode(self) -> DataFrame:
        """Each field as a column of a frame, under the column's labels."""
        from ._pandas import concat

        parts = [self.field(i) for i in range(self._type.num_fields)]
        return concat(parts, axis=1)
