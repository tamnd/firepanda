"""`FirepandaArray`, the answer pandas gives as an array rather than a column.

`Series.unique` in pandas hands back neither a Series nor a list. It hands back
an extension array for text, a categorical or a timestamp column, and a numpy
array for numbers, and `factorize` hands back a numpy array of codes. What those
have in common is that they are values with a type and nothing else: no row
labels and no name. This class is that shape. It holds one column and gives
out its values, its length, its type and its Arrow export, and nothing that
needs a label.

The values are stored the same way for every type, as Arrow, so one class
does the work. pandas has a class per kind of storage, and a program reads
those names off `type(x).__name__`, off the repr and off `pd.arrays`, so each
array takes the name of the class pandas would answer for its type when it is
made: `IntegerArray` for a masked integer column, `ArrowStringArray` for text,
`NumpyExtensionArray` for a numpy typed column and so on. Each of those is a
subclass with nothing of its own but how it prints a value. A consumer reading
the values through the Arrow PyCapsule protocol, which is what pyarrow, polars
and DuckDB do, gets them without a copy, as it does for a Series.

`array` is `pandas.array`, which infers the masked type for plain Python
numbers and bools and text for strings, as pandas does, and keeps any other
type the values or `dtype` name.
"""

from __future__ import annotations

import numbers
import re
from collections.abc import Iterator
from typing import TYPE_CHECKING, Any

from .errors import DTypeError, InvalidArgumentError, OutOfBoundsError, UnsupportedError

if TYPE_CHECKING:
    from ._frame import Series


class FirepandaArray:
    """Values of one type in order, with no row labels and no name."""

    __slots__ = ("_column",)

    # Whether pandas keeps these values in a numpy array, whose axes it checks.
    _numpy_backed = False

    def __init__(self, column: Series, *args: Any, **kwargs: Any) -> None:
        if not hasattr(column, "reset_index"):
            # Plain values, as `pd.arrays.NumpyExtensionArray(numpy_array)` passes them.
            from ._frame import Series

            column = Series(list(column) if not hasattr(column, "dtype") else column)
        self._column = column.reset_index(drop=True).rename(None)
        if type(self) is FirepandaArray:
            self.__class__ = _class_of(str(self._column.dtype))

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

    def isna(self) -> Any:
        """Whether each value is missing, as a numpy array of bools as pandas answers it."""
        import numpy

        return numpy.array(self._column.isna().tolist(), dtype=bool)

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

    def isnull(self) -> Any:
        """Whether each value is missing, the same as `isna`."""
        return self.isna()

    def copy(self) -> FirepandaArray:
        """The same values in an array of their own."""
        return self._again(self._column.copy())

    def astype(self, dtype: Any, copy: bool = True) -> FirepandaArray:
        """The values as another type."""
        return FirepandaArray(self._column.astype(dtype))

    def fillna(self, value: Any = None, limit: Any = None, copy: bool = True) -> FirepandaArray:
        """The values with every gap filled by `value`."""
        return FirepandaArray(self._column.fillna(value, limit=limit))

    def dropna(self) -> FirepandaArray:
        """The values that are not missing."""
        return FirepandaArray(self._column.dropna())

    def value_counts(self, dropna: bool = True) -> Series:
        """How often each value appears, most often first."""
        return self._column.value_counts(dropna=dropna)

    def to_numpy(self, dtype: Any = None, copy: bool = False, na_value: Any = None) -> Any:
        """The values as a numpy array."""
        from ._pandas import NO_DEFAULT

        missing = NO_DEFAULT if na_value is None else na_value
        return self._column.to_numpy(dtype=dtype, copy=copy, na_value=missing)

    def __array__(self, dtype: Any = None, copy: Any = None) -> Any:
        return self.to_numpy(dtype=dtype)

    def _reduced(self, name: str, skipna: bool) -> Any:
        return getattr(self._column, name)(skipna=skipna)

    def sum(self, skipna: bool = True) -> Any:
        """The total of the values."""
        return self._reduced("sum", skipna)

    def min(self, skipna: bool = True) -> Any:
        """The smallest value."""
        return self._reduced("min", skipna)

    def max(self, skipna: bool = True) -> Any:
        """The largest value."""
        return self._reduced("max", skipna)

    def mean(self, skipna: bool = True) -> Any:
        """The mean of the values."""
        return self._reduced("mean", skipna)

    def _again(self, column: Series) -> FirepandaArray:
        """An array of the same kind over another column."""
        return FirepandaArray(column)

    def _place(self, name: str, skipna: bool) -> Any:
        import numpy

        if not skipna and bool(self._column.isna().any()):
            raise InvalidArgumentError("Encountered an NA value with skipna=False")
        return numpy.int64(self.argsort(ascending=name == "argmin")[0])

    def argmax(self, skipna: bool = True) -> Any:
        """The position of the largest value."""
        return self._place("argmax", skipna)

    def argmin(self, skipna: bool = True) -> Any:
        """The position of the smallest value."""
        return self._place("argmin", skipna)

    def argsort(
        self,
        *,
        ascending: bool = True,
        kind: str = "quicksort",
        na_position: str = "last",
        **kwargs: Any,
    ) -> Any:
        """The positions that put the values in order, ties kept in place, as a numpy array."""
        import numpy

        ordered = self._column.sort_values(
            ascending=ascending, kind="stable", na_position=na_position
        )
        return numpy.array(ordered.index.tolist(), dtype=numpy.int64)

    def duplicated(self, keep: Any = "first") -> Any:
        """Whether each value repeats one kept, as a numpy array of bools."""
        import numpy

        return numpy.array(self._column.duplicated(keep=keep).tolist(), dtype=bool)

    def isin(self, values: Any) -> Any:
        """Whether each value is among `values`, as a numpy array of bools."""
        import numpy

        if isinstance(values, FirepandaArray):
            values = values.tolist()
        return numpy.array(self._column.isin(values).tolist(), dtype=bool)

    def item(self, index: Any = None) -> Any:
        """The one value of an array of one.

        Raises:
            ValueError: For an array of any other length, in numpy's words.
        """
        if index is None and len(self._column) != 1:
            raise InvalidArgumentError("can only convert an array of size 1 to a Python scalar")
        return self.to_numpy()[0 if index is None else index]

    def _mapped(self, mapper: Any, na_action: Any) -> list[Any]:
        if na_action not in (None, "ignore"):
            raise InvalidArgumentError(
                f"na_action must either be 'ignore' or None, {na_action!r} was passed"
            )
        pick = mapper.get if isinstance(mapper, dict) else mapper
        return [
            value if na_action == "ignore" and _gap(value) else pick(value)
            for value in self._column.tolist()
        ]

    def map(self, mapper: Any, na_action: Any = None) -> Any:
        """`mapper` applied to each value, as a numpy array."""
        import numpy

        text = isinstance(self, ArrowStringArray)
        return numpy.array(self._mapped(mapper, na_action), dtype=object if text else None)

    @property
    def nbytes(self) -> int:
        """The bytes the values take."""
        return int(self._column.nbytes)

    def searchsorted(self, value: Any, side: str = "left", sorter: Any = None) -> Any:
        """Where `value` goes to keep the values in order, as numpy answers it."""
        import numpy

        if isinstance(value, FirepandaArray):
            value = value.tolist()
        return numpy.searchsorted(self.to_numpy(), value, side=side, sorter=sorter)

    def shift(self, periods: int = 1, fill_value: Any = None) -> FirepandaArray:
        """The values moved by `periods` places, the places left a gap or `fill_value`."""
        if fill_value is None:
            return self._again(self._column.shift(periods))
        return self._again(self._column.shift(periods, fill_value=fill_value))

    def view(self, dtype: Any = None) -> Any:
        """The same values, or with `dtype` their numpy bytes read as that type."""
        if dtype is not None:
            return self.to_numpy().view(dtype)
        return self._again(self._column)

    def interpolate(
        self,
        *,
        method: str,
        axis: int,
        index: Any,
        limit: Any,
        limit_direction: str,
        limit_area: Any,
        copy: bool,
        **kwargs: Any,
    ) -> FirepandaArray:
        """The gaps filled along `index`, as a column's `interpolate` fills them."""
        from ._frame import Series

        column = Series(self._column.tolist(), index=index, dtype=self._column.dtype)
        filled = column.interpolate(
            method=method, limit=limit, limit_direction=limit_direction, limit_area=limit_area
        )
        return self._again(filled)

    def _from_values(self, values: list[Any]) -> FirepandaArray:
        """An array of the same type over `values`, a gap for each None."""
        from ._frame import Series

        if str(self._column.dtype) in _NO_GAP and any(_gap(value) for value in values):
            # A numpy number or bool has no gap, so pandas lets the values pick a type.
            return FirepandaArray(Series([float("nan") if _gap(v) else v for v in values]))
        return self._again(Series(values, dtype=str(self._column.dtype)))

    def _picked(self, positions: list[int]) -> FirepandaArray:
        return self._again(self._column.iloc[positions])

    def take(
        self, indexer: Any, *, allow_fill: bool = False, fill_value: Any = None, axis: int = 0
    ) -> FirepandaArray:
        """The values at the positions `indexer` lists, -1 a gap when `allow_fill` is set.

        Raises:
            ValueError: For a position below -1 with `allow_fill`, in pandas' words.
            IndexError: For a position past the values, in pandas' words.
        """
        len(indexer)
        listed = indexer.tolist() if hasattr(indexer, "tolist") else list(indexer)
        positions = [int(at) for at in listed]
        size = len(self._column)
        if allow_fill:
            low = min(positions, default=0)
            if low < -1:
                raise InvalidArgumentError(
                    f"'indices' contains values less than allowed ({low} < -1)"
                )
            if any(at >= size for at in positions):
                raise OutOfBoundsError("indices are out-of-bounds")
            if -1 in positions:
                values = self._column.tolist()
                return self._from_values([fill_value if at < 0 else values[at] for at in positions])
            return self._picked(positions)
        for at in positions:
            if not -size <= at < size:
                if isinstance(self, ArrowStringArray | ArrowExtensionArray):
                    raise OutOfBoundsError("out of bounds value in 'indices'.")
                raise OutOfBoundsError(f"index {at} is out of bounds for axis 0 with size {size}")
        return self._picked([at % size for at in positions])

    def delete(self, loc: Any, axis: int = 0) -> FirepandaArray:
        """The values without the one at `loc`, or without each one at a position it lists.

        Raises:
            IndexError: For a position past the values, in numpy's words.
        """
        import numpy

        try:
            kept = numpy.delete(numpy.arange(len(self._column)), loc)
        except IndexError as error:
            raise OutOfBoundsError(str(error)) from None
        return self._picked(kept.tolist())

    def insert(self, loc: int, item: Any) -> FirepandaArray:
        """The values with `item` put at the position `loc`, which may count from the end.

        Raises:
            IndexError: For a position outside the values and the one past them.
        """
        size = len(self._column)
        if not isinstance(loc, numbers.Integral) or not -size <= loc <= size:
            raise OutOfBoundsError(f"loc must be an integer between {-size} and {size}")
        values = self._column.tolist()
        values.insert(int(loc), item)
        return self._from_values(values)

    def repeat(self, repeats: Any, axis: Any = None) -> FirepandaArray:
        """Each value `repeats` times, or as many times as its own entry in `repeats` says.

        Raises:
            ValueError: For an axis, or for counts that do not fit the values, in
                pandas' and numpy's words.
        """
        import numpy

        if axis is not None and self._numpy_backed:
            # numpy repeats these, and has the one axis there is.
            _axis_checked(axis)
        elif axis is not None:
            raise InvalidArgumentError(
                "the 'axis' parameter is not supported in the pandas implementation of repeat()"
            )
        try:
            kept = numpy.repeat(numpy.arange(len(self._column)), repeats)
        except ValueError as error:
            raise InvalidArgumentError(str(error)) from None
        return self._picked(kept.tolist())

    def equals(self, other: Any) -> bool:
        """Whether `other` is an array of the same type with the same values and gaps."""
        if type(other) is not type(self) or not bool(other.dtype == self.dtype):
            return False
        return bool(self._column.equals(other._column))

    def ravel(self, *args: Any, **kwargs: Any) -> FirepandaArray:
        """The same values, which already lie in one dimension."""
        return self._again(self._column)

    def transpose(self, *axes: int) -> FirepandaArray:
        """The same values, as an array of one dimension is its own transpose.

        Raises:
            AxisError: For an axis past the one there is, on the kinds pandas keeps
                in numpy, in numpy's words.
        """
        if self._numpy_backed:
            for axis in axes[0] if len(axes) == 1 and isinstance(axes[0], tuple | list) else axes:
                _axis_checked(axis)
        return self._again(self._column)

    @property
    def T(self) -> FirepandaArray:
        """The same values, as `transpose` gives them."""
        return self.transpose()

    def _operated(self, other: Any, name: str) -> Any:
        """An operator applied value by value, as the column applies it."""
        if isinstance(other, FirepandaArray):
            other = other._column
        found = getattr(self._column, name)(other)
        return found if found is NotImplemented else FirepandaArray(found)

    def __add__(self, other: Any) -> Any:
        return self._operated(other, "__add__")

    def __radd__(self, other: Any) -> Any:
        return self._operated(other, "__radd__")

    def __sub__(self, other: Any) -> Any:
        return self._operated(other, "__sub__")

    def __rsub__(self, other: Any) -> Any:
        return self._operated(other, "__rsub__")

    def __mul__(self, other: Any) -> Any:
        return self._operated(other, "__mul__")

    def __rmul__(self, other: Any) -> Any:
        return self._operated(other, "__rmul__")

    def __truediv__(self, other: Any) -> Any:
        return self._operated(other, "__truediv__")

    def __rtruediv__(self, other: Any) -> Any:
        return self._operated(other, "__rtruediv__")

    def __floordiv__(self, other: Any) -> Any:
        return self._operated(other, "__floordiv__")

    def __mod__(self, other: Any) -> Any:
        return self._operated(other, "__mod__")

    def __pow__(self, other: Any) -> Any:
        return self._operated(other, "__pow__")

    def __eq__(self, other: object) -> Any:  # type: ignore[override]
        return self._operated(other, "__eq__")

    def __ne__(self, other: object) -> Any:  # type: ignore[override]
        return self._operated(other, "__ne__")

    def __lt__(self, other: Any) -> Any:
        return self._operated(other, "__lt__")

    def __le__(self, other: Any) -> Any:
        return self._operated(other, "__le__")

    def __gt__(self, other: Any) -> Any:
        return self._operated(other, "__gt__")

    def __ge__(self, other: Any) -> Any:
        return self._operated(other, "__ge__")

    __hash__ = None  # type: ignore[assignment]

    def __neg__(self) -> FirepandaArray:
        return FirepandaArray(-self._column)

    def __abs__(self) -> FirepandaArray:
        return FirepandaArray(abs(self._column))

    def _formatter(self, values: list[Any]) -> Any:
        """How the repr prints each of these values."""
        return self._shown

    def _shown(self, value: Any) -> str:
        """One value as the repr prints it, which is how the classes differ."""
        from ._pandas import _pprinted

        return _pprinted(value)

    def __repr__(self) -> str:
        from ._config import get_option
        from ._pandas import _summary

        values = self._column.tolist()
        width = get_option("display.width") or 80
        most = get_option("display.max_seq_items") or len(values)
        body = _summary(values, self._formatter(values), True, "", width, most, indent=False)
        dtype = "str" if str(self.dtype) == "string" and not _masked_text(self) else self.dtype
        return (
            f"<{type(self).__name__}>\n{body.rstrip(', ' + chr(10))}\n"
            f"Length: {len(values)}, dtype: {dtype}"
        )


def _axis_checked(axis: Any, prefix: str | None = None) -> None:
    """Refuse an axis an array of one dimension does not have, in numpy's words."""
    import numpy

    if not -1 <= int(axis) <= 0:
        raise numpy.exceptions.AxisError(int(axis), 1, prefix)


_NO_GAP = frozenset(
    ("int8", "int16", "int32", "int64", "uint8", "uint16", "uint32", "uint64", "bool")
)


class _Shaped:
    """`reshape` and `swapaxes`, which pandas gives the kinds it can lay out in more axes."""

    __slots__ = ()
    _column: Series

    def reshape(self, *args: Any, **kwargs: Any) -> Any:
        """The same values, for a shape of one dimension as long as they are.

        Raises:
            ValueError: For a shape of another size, in numpy's words.
            NotImplementedError: For a shape of more than one dimension.
        """
        import numpy

        try:
            laid = numpy.arange(len(self._column)).reshape(*args, **kwargs)
        except ValueError as error:
            raise InvalidArgumentError(str(error)) from None
        if laid.ndim != 1:
            raise UnsupportedError("firepanda arrays have one dimension, so reshape takes one")
        return self._again(self._column)  # type: ignore[attr-defined]

    def swapaxes(self, axis1: Any, axis2: Any) -> Any:
        """The same values, as swapping the one axis with itself changes nothing.

        Raises:
            AxisError: For an axis past the one there is, in numpy's words.
        """
        _axis_checked(axis1, "axis1")
        _axis_checked(axis2, "axis2")
        return self._again(self._column)  # type: ignore[attr-defined]


class _Truth:
    """`all` and `any`, on the kinds pandas reads as true or false."""

    __slots__ = ()
    _column: Series

    def all(self, *, skipna: bool = True, axis: Any = 0, **kwargs: Any) -> Any:
        """Whether every value is true, a gap unknown when `skipna` is off."""
        return self._column.all(skipna=skipna)

    def any(self, *, skipna: bool = True, axis: Any = 0, **kwargs: Any) -> Any:
        """Whether some value is true, a gap unknown when `skipna` is off."""
        return self._column.any(skipna=skipna)


class _Spread:
    """`prod`, `std` and `var`, on the kinds of number pandas reduces itself."""

    __slots__ = ()
    _column: Series

    def prod(self, *, skipna: bool = True, min_count: int = 0, axis: Any = 0, **kwargs: Any) -> Any:
        """The product of the values."""
        return self._column.prod(skipna=skipna, min_count=min_count)

    def std(self, *, skipna: bool = True, axis: Any = 0, ddof: int = 1, **kwargs: Any) -> Any:
        """The standard deviation of the values."""
        return self._column.std(skipna=skipna, ddof=ddof)

    def var(self, *, skipna: bool = True, axis: Any = 0, ddof: int = 1, **kwargs: Any) -> Any:
        """The variance of the values."""
        return self._column.var(skipna=skipna, ddof=ddof)


class _Middle:
    """`median`, on the kinds pandas keeps in a numpy array."""

    __slots__ = ()
    _column: Series

    def median(self, *, axis: Any = None, skipna: bool = True, **kwargs: Any) -> Any:
        """The middle value."""
        return self._column.median(skipna=skipna)


def _gap(value: Any) -> bool:
    from ._pandas import _missing

    return _missing(value)


class NumpyExtensionArray(_Shaped, _Truth, _Spread, _Middle, FirepandaArray):
    """Values of a numpy type, which is `pandas.arrays.NumpyExtensionArray`."""

    __slots__ = ()
    _numpy_backed = True

    def sem(self, *, axis: Any = None, ddof: int = 1, skipna: bool = True, **kwargs: Any) -> Any:
        """The standard error of the mean."""
        return self._column.sem(skipna=skipna, ddof=ddof)

    def skew(self, *, axis: Any = None, skipna: bool = True, **kwargs: Any) -> Any:
        """The skew of the values."""
        return self._column.skew(skipna=skipna)

    def kurt(self, *, axis: Any = None, skipna: bool = True, **kwargs: Any) -> Any:
        """The kurtosis of the values."""
        return self._column.kurt(skipna=skipna)


class IntegerArray(_Shaped, _Truth, _Spread, FirepandaArray):
    """Whole numbers of a masked type, which is `pandas.arrays.IntegerArray`."""

    __slots__ = ()

    def round(self, decimals: int = 0, *args: Any, **kwargs: Any) -> FirepandaArray:
        """Each value rounded to `decimals` places."""
        return self._again(self._column.round(decimals))

    def isin(self, values: Any) -> Any:
        """Whether each value is among `values`, as a masked array of bools."""
        from ._frame import Series

        return FirepandaArray(Series(super().isin(values).tolist(), dtype="boolean"))

    def map(self, mapper: Any, na_action: Any = None) -> Any:
        """`mapper` applied to each value, as a numpy array with NaN for a gap."""
        import numpy

        values = self._mapped(mapper, na_action)
        if any(_gap(value) for value in values):
            values = [numpy.nan if _gap(value) else value for value in values]
        return numpy.array(values)

    @property
    def nbytes(self) -> int:
        """The bytes of the values and of the mask, as pandas counts them."""
        bits = re.search(r"\d+", str(self.dtype))
        width = int(bits.group()) // 8 if bits else 1
        return len(self._column) * (width + 1)

    def _shown(self, value: Any) -> str:
        return "<NA>" if _gap(value) else str(value)


class FloatingArray(IntegerArray):
    """Floats of a masked type, which is `pandas.arrays.FloatingArray`."""

    __slots__ = ()


class BooleanArray(IntegerArray):
    """Bools of the masked type, which is `pandas.arrays.BooleanArray`."""

    __slots__ = ()


class ArrowStringArray(FirepandaArray):
    """Text, which is `pandas.arrays.ArrowStringArray`."""

    __slots__ = ()

    def _from_values(self, values: list[Any]) -> FirepandaArray:
        """Text over `values`, each one a string or a gap.

        Raises:
            TypeError: For a value of another type, in pandas' words.
        """
        for value in values:
            if not isinstance(value, str) and not _gap(value):
                raise DTypeError(
                    f"Invalid value '{value}' for dtype 'str'. Value should be a string or "
                    f"missing value, got '{type(value).__name__}' instead."
                )
        return super()._from_values(values)

    def _shown(self, value: Any) -> str:
        from ._pandas import _pprinted

        if _gap(value):
            return "<NA>" if _masked_text(self) else "nan"
        return _pprinted(value)


def _masked_text(array: FirepandaArray) -> bool:
    """Whether the array holds pandas' masked `string` type rather than plain text."""
    return type(array.dtype).__name__ == "StringDtype" and str(array.dtype) == "string"


class ArrowExtensionArray(FirepandaArray):
    """Values of an Arrow type, which is `pandas.arrays.ArrowExtensionArray`."""

    __slots__ = ()

    def _shown(self, value: Any) -> str:
        from ._pandas import _pprinted

        return "<NA>" if _gap(value) else _pprinted(value)


def _from_index(value: Any) -> Any:
    """An index's answer as the array gives it: numbers and flags in numpy, other labels as
    an array of their own, a table counted from 0 and a list as a numpy array of objects."""
    from ._frame import DataFrame, Index

    if isinstance(value, DataFrame):
        return value.reset_index(drop=True)
    if isinstance(value, Index):
        kind = str(value.dtype)
        return value.to_numpy() if kind in _NO_GAP or kind.startswith("float") else value.array
    if isinstance(value, list):
        import numpy

        made = numpy.empty(len(value), dtype=object)
        for place, item in enumerate(value):
            made[place] = item
        return made
    return value


def _field(name: str, doc: str) -> property:
    """A field the array reads from the index of its own kind."""
    return property(lambda self: _from_index(getattr(self._index(), name)), doc=doc)


class _Fields:
    """The fields and methods an instant, span or period array shares with its index."""

    __slots__ = ()
    _column: Series

    def _index(self) -> Any:
        raise NotImplementedError

    def _asked(self, name: str, *args: Any, **kwargs: Any) -> Any:
        return _from_index(getattr(self._index(), name)(*args, **kwargs))

    asi8 = _field("asi8", "The values as int64 counts, the least int64 for a gap.")


class _Stepped(_Fields):
    """The frequency fields instants and spans share, which periods hold in their type."""

    __slots__ = ()

    freq = _field("freq", "The step between the values, None when it is not kept.")
    inferred_freq = _field("inferred_freq", "The step the values are evenly spaced by, if any.")


class DatetimeArray(_Stepped, _Shaped, _Middle, FirepandaArray):
    """Instants, which is `pandas.arrays.DatetimeArray`."""

    __slots__ = ()
    _numpy_backed = True

    def _index(self) -> Any:
        from ._datetime import DatetimeIndex

        return DatetimeIndex(self._column)

    year = _field("year", "The year of each instant.")
    month = _field("month", "The month of each instant, 1 for January.")
    day = _field("day", "The day of the month of each instant.")
    hour = _field("hour", "The hour of each instant.")
    minute = _field("minute", "The minute of each instant.")
    second = _field("second", "The second of each instant.")
    microsecond = _field("microsecond", "The microseconds past the second of each instant.")
    nanosecond = _field("nanosecond", "The nanoseconds past the microsecond of each instant.")
    quarter = _field("quarter", "The quarter of the year of each instant.")
    dayofweek = day_of_week = weekday = _field("dayofweek", "The weekday, 0 for Monday.")
    dayofyear = day_of_year = _field("dayofyear", "The day of the year of each instant.")
    days_in_month = daysinmonth = _field("days_in_month", "How many days each month has.")
    is_month_start = _field("is_month_start", "Whether each instant is a month's first day.")
    is_month_end = _field("is_month_end", "Whether each instant is a month's last day.")
    is_quarter_start = _field("is_quarter_start", "Whether each is a quarter's first day.")
    is_quarter_end = _field("is_quarter_end", "Whether each is a quarter's last day.")
    is_year_start = _field("is_year_start", "Whether each instant is a year's first day.")
    is_year_end = _field("is_year_end", "Whether each instant is a year's last day.")
    is_leap_year = _field("is_leap_year", "Whether each instant's year is a leap year.")
    is_normalized = _field("is_normalized", "Whether every instant is at midnight.")
    date = _field("date", "The date of each instant, NaT for a gap.")
    time = _field("time", "The time of day of each instant, NaT for a gap.")
    timetz = _field("timetz", "The time of day with its zone, NaT for a gap.")
    tz = _field("tz", "The time zone, None when the instants have none.")
    tzinfo = _field("tzinfo", "The time zone, the same as `tz`.")
    unit = _field("unit", "The unit the instants are counted in.")
    resolution = _field("resolution", "The finest unit any instant needs.")

    def normalize(self) -> Any:
        """Each instant at the midnight that starts its day."""
        return self._asked("normalize")

    def day_name(self, locale: Any = None) -> Any:
        """The weekday's name for each instant."""
        return self._asked("day_name", locale)

    def month_name(self, locale: Any = None) -> Any:
        """The month's name for each instant."""
        return self._asked("month_name", locale)

    def isocalendar(self) -> Any:
        """The ISO year, week and day of each instant, as a table."""
        return self._asked("isocalendar")

    def floor(self, freq: Any, ambiguous: Any = "raise", nonexistent: Any = "raise") -> Any:
        """Each instant rounded down to a multiple of `freq`."""
        return self._asked("floor", freq, ambiguous=ambiguous, nonexistent=nonexistent)

    def ceil(self, freq: Any, ambiguous: Any = "raise", nonexistent: Any = "raise") -> Any:
        """Each instant rounded up to a multiple of `freq`."""
        return self._asked("ceil", freq, ambiguous=ambiguous, nonexistent=nonexistent)

    def round(self, freq: Any, ambiguous: Any = "raise", nonexistent: Any = "raise") -> Any:
        """Each instant rounded to the nearest multiple of `freq`."""
        return self._asked("round", freq, ambiguous=ambiguous, nonexistent=nonexistent)

    def as_unit(self, unit: str, round_ok: bool = True) -> Any:
        """The instants counted in another unit."""
        return self._asked("as_unit", unit, round_ok=round_ok)

    def to_pydatetime(self) -> Any:
        """The instants as Python datetimes in a numpy array of objects."""
        return self._asked("to_pydatetime")

    def to_period(self, freq: Any = None) -> Any:
        """The period of `freq` each instant falls in."""
        return self._asked("to_period", freq)

    def to_julian_date(self) -> Any:
        """The Julian date of each instant."""
        return self._asked("to_julian_date")

    def strftime(self, date_format: str) -> Any:
        """Each instant written by a format, a gap for NaT."""
        return self._asked("strftime", date_format)

    def tz_localize(self, tz: Any, ambiguous: Any = "raise", nonexistent: Any = "raise") -> Any:
        """The wall times read in a time zone."""
        return self._asked("tz_localize", tz, ambiguous=ambiguous, nonexistent=nonexistent)

    def tz_convert(self, tz: Any) -> Any:
        """The same instants as another time zone's wall times."""
        return self._asked("tz_convert", tz)

    def std(self, *, axis: Any = None, ddof: int = 1, skipna: bool = True, **kwargs: Any) -> Any:
        """The spread of the instants, as a span."""
        return self._index().std(axis=axis, ddof=ddof, skipna=skipna)

    def _shown(self, value: Any) -> str:
        return "'NaT'" if _gap(value) else f"'{value}'"


class TimedeltaArray(_Stepped, _Shaped, _Truth, _Middle, FirepandaArray):
    """Spans, which is `pandas.arrays.TimedeltaArray`."""

    __slots__ = ()
    _numpy_backed = True

    def _index(self) -> Any:
        from ._timedelta import TimedeltaIndex

        return TimedeltaIndex(self._column)

    days = _field("days", "The whole days of each span, rounded toward minus infinity.")
    seconds = _field("seconds", "The seconds past the whole days of each span.")
    microseconds = _field("microseconds", "The microseconds past the second of each span.")
    nanoseconds = _field("nanoseconds", "The nanoseconds past the microsecond of each span.")
    components = _field("components", "Each span's days down to nanoseconds, as a table.")
    unit = _field("unit", "The unit the spans are counted in.")

    def total_seconds(self) -> Any:
        """Each span in seconds, NaN for a gap."""
        return self._asked("total_seconds")

    def to_pytimedelta(self) -> Any:
        """The spans as Python timedeltas in a numpy array of objects."""
        return self._asked("to_pytimedelta")

    def floor(self, freq: Any, ambiguous: Any = "raise", nonexistent: Any = "raise") -> Any:
        """Each span rounded down to a multiple of `freq`."""
        return self._asked("floor", freq)

    def ceil(self, freq: Any, ambiguous: Any = "raise", nonexistent: Any = "raise") -> Any:
        """Each span rounded up to a multiple of `freq`."""
        return self._asked("ceil", freq)

    def round(self, freq: Any, ambiguous: Any = "raise", nonexistent: Any = "raise") -> Any:
        """Each span rounded to the nearest multiple of `freq`."""
        return self._asked("round", freq)

    def as_unit(self, unit: str, round_ok: bool = True) -> Any:
        """The spans counted in another unit."""
        return self._asked("as_unit", unit, round_ok=round_ok)

    def std(self, *, axis: Any = None, ddof: int = 1, skipna: bool = True, **kwargs: Any) -> Any:
        """The spread of the spans."""
        return self._index().std(axis=axis, ddof=ddof, skipna=skipna)

    def _formatter(self, values: list[Any]) -> Any:
        present = [value for value in values if not _gap(value)]
        days = all(v.seconds == v.microseconds == v.nanoseconds == 0 for v in present)

        def shown(value: Any) -> str:
            if _gap(value):
                return "NaT"
            return f"'{value.days} days'" if days else f"'{value}'"

        return shown


class PeriodArray(_Fields, _Shaped, _Middle, FirepandaArray):
    """Periods, which is `pandas.arrays.PeriodArray`."""

    __slots__ = ()
    _numpy_backed = True

    def _index(self) -> Any:
        from ._period_index import PeriodIndex

        return PeriodIndex(self._column)

    year = _field("year", "The year of each period.")
    month = _field("month", "The month of each period, 1 for January.")
    day = _field("day", "The day of the month each period ends on.")
    hour = _field("hour", "The hour of each period.")
    minute = _field("minute", "The minute of each period.")
    second = _field("second", "The second of each period.")
    quarter = _field("quarter", "The quarter of the year of each period.")
    qyear = _field("qyear", "The fiscal year each period's quarter falls in.")
    week = weekofyear = _field("week", "The ISO week of the year of each period.")
    dayofweek = day_of_week = weekday = _field("dayofweek", "The weekday, 0 for Monday.")
    dayofyear = day_of_year = _field("dayofyear", "The day of the year of each period.")
    days_in_month = daysinmonth = _field("days_in_month", "How many days each month has.")
    is_leap_year = _field("is_leap_year", "Whether each period's year is a leap year.")
    start_time = _field("start_time", "The first instant of each period.")
    end_time = _field("end_time", "The last instant of each period.")
    freq = _field("freq", "The frequency, as the offset that steps by it.")
    freqstr = _field("freqstr", "The frequency, as its string.")

    def to_timestamp(self, freq: Any = None, how: str = "start") -> Any:
        """Each period as an instant, at its start or its end."""
        return self._asked("to_timestamp", freq=freq, how=how)

    def asfreq(self, freq: Any = None, how: str = "E") -> Any:
        """Each period as the one of another frequency at its start or its end."""
        return self._asked("asfreq", freq, how=how)

    def strftime(self, date_format: str) -> Any:
        """Each period written by a format, a gap for NaT."""
        return self._asked("strftime", date_format)

    def median(self, *, axis: Any = None, skipna: bool = True, **kwargs: Any) -> Any:
        """The middle period, the earlier of the two middle ones' midpoint for an even count."""
        values = self._column.tolist()
        present = sorted(value for value in values if not _gap(value))
        if not present or (not skipna and len(present) < len(values)):
            from ._pandas import NaT

            return NaT
        low, high = present[(len(present) - 1) // 2], present[len(present) // 2]
        return low + (high.ordinal - low.ordinal) // 2

    def _shown(self, value: Any) -> str:
        return "'NaT'" if _gap(value) else f"'{value}'"


class IntervalArray(FirepandaArray):
    """Intervals, which is `pandas.arrays.IntervalArray`."""

    __slots__ = ()

    @classmethod
    def _of(cls, index: Any) -> IntervalArray:
        from ._frame import Series

        return cls(Series(index))

    @classmethod
    def from_breaks(
        cls, breaks: Any, closed: str = "right", copy: bool = False, dtype: Any = None
    ) -> IntervalArray:
        """Intervals between each break and the next."""
        from ._interval_index import IntervalIndex

        return cls._of(IntervalIndex.from_breaks(breaks, closed=closed, dtype=dtype))

    @classmethod
    def from_arrays(
        cls, left: Any, right: Any, closed: str = "right", copy: bool = False, dtype: Any = None
    ) -> IntervalArray:
        """Intervals from a list of left ends and a list of right ends."""
        from ._interval_index import IntervalIndex

        return cls._of(IntervalIndex.from_arrays(left, right, closed=closed, dtype=dtype))

    @classmethod
    def from_tuples(
        cls, data: Any, closed: str = "right", copy: bool = False, dtype: Any = None
    ) -> IntervalArray:
        """Intervals from pairs of ends, None for a gap."""
        from ._interval_index import IntervalIndex

        return cls._of(IntervalIndex.from_tuples(data, closed=closed, dtype=dtype))

    def _index(self) -> Any:
        from ._interval_index import IntervalIndex

        return IntervalIndex(self._column)

    can_hold_na = True

    @property
    def closed(self) -> str:
        """Which ends each interval holds."""
        return self._index().closed

    @property
    def closed_left(self) -> bool:
        """Whether each interval holds its left end."""
        return self._index().closed_left

    @property
    def closed_right(self) -> bool:
        """Whether each interval holds its right end."""
        return self._index().closed_right

    @property
    def open_left(self) -> bool:
        """Whether each interval leaves out its left end."""
        return self._index().open_left

    @property
    def open_right(self) -> bool:
        """Whether each interval leaves out its right end."""
        return self._index().open_right

    def to_tuples(self, na_tuple: bool = True) -> Any:
        """Each interval as a pair of its ends, a numpy array of objects."""
        import numpy

        pairs = self._index()._tuples(na_tuple)
        found = numpy.empty(len(pairs), dtype=object)
        for at, pair in enumerate(pairs):
            # One at a time, or numpy reads the pairs as a second axis.
            found[at] = pair
        return found

    @property
    def left(self) -> Any:
        """The left ends, as an index."""
        return self._index().left

    @property
    def right(self) -> Any:
        """The right ends, as an index."""
        return self._index().right

    @property
    def mid(self) -> Any:
        """The middle of each interval, as an index."""
        return self._index().mid

    @property
    def length(self) -> Any:
        """Each interval's length, as an index."""
        return self._index().length

    @property
    def is_empty(self) -> Any:
        """Whether each interval holds no point, as numpy bools."""
        return self._index().is_empty

    @property
    def is_non_overlapping_monotonic(self) -> bool:
        """Whether the intervals increase and none overlaps the next."""
        return self._index().is_non_overlapping_monotonic

    def contains(self, other: Any) -> Any:
        """Whether each interval holds the point `other`, as numpy bools."""
        return self._index().contains(other)

    def overlaps(self, other: Any) -> Any:
        """Whether each interval shares a point with the interval `other`, as numpy bools."""
        return self._index().overlaps(other)

    def _from_values(self, values: list[Any]) -> IntervalArray:
        from ._interval_index import IntervalIndex

        return self._of(IntervalIndex(values, closed=self.closed))

    def set_closed(self, closed: str) -> IntervalArray:
        """The same ends, each interval holding the ends `closed` names."""
        return self._of(self._index().set_closed(closed))

    def _shown(self, value: Any) -> str:
        return "nan" if _gap(value) else str(value)


def _class_of(dtype: str) -> type[FirepandaArray]:
    """The class pandas answers for an array of this type."""
    if dtype in _MASKED_INTEGERS:
        return IntegerArray
    if dtype in ("Float32", "Float64"):
        return FloatingArray
    if dtype == "boolean":
        return BooleanArray
    if dtype in ("str", "string"):
        return ArrowStringArray
    if dtype.endswith("[pyarrow]"):
        return ArrowExtensionArray
    if dtype.startswith("datetime64"):
        return DatetimeArray
    if dtype.startswith("timedelta64"):
        return TimedeltaArray
    if dtype.startswith("period["):
        return PeriodArray
    if dtype.startswith("interval"):
        return IntervalArray
    return NumpyExtensionArray


_MASKED_INTEGERS = frozenset(
    ("Int8", "Int16", "Int32", "Int64", "UInt8", "UInt16", "UInt32", "UInt64")
)


def _inferred(values: list[Any]) -> str | None:
    """The type `pandas.array` gives plain values, None to let the column decide.

    Whole numbers and bools become the masked types and text becomes `string`,
    where a column would give numpy's types, and values of mixed kinds become
    objects.
    """
    import numpy

    from ._objects import is_gap

    present = [value for value in values if not is_gap(value)]
    if not values:
        return "Float64"
    if not present:
        return "object"
    kinds = set()
    for value in present:
        if isinstance(value, bool | numpy.bool_):
            kinds.add("bool")
        elif isinstance(value, numbers.Integral):
            kinds.add("int")
        elif isinstance(value, numbers.Real):
            kinds.add("float")
        elif isinstance(value, str):
            kinds.add("str")
        else:
            return None
    if kinds == {"bool"}:
        return "boolean"
    if kinds == {"int"} or (kinds == {"int", "float"} and len(present) < len(values)):
        whole = all(isinstance(v, numbers.Integral) or float(v).is_integer() for v in present)
        if kinds == {"int"} or whole:
            return "Int64"
    if kinds <= {"int", "float"}:
        return "Float64"
    if kinds == {"str"}:
        return "string"
    return "object"


_NUMPY_MASKED = {"b": "boolean", "i": "Int64", "u": "UInt64", "f": "Float64", "U": "string"}
"""The masked type `pandas.array` gives a numpy array of each kind, by its kind letter."""


def array(data: Any, dtype: Any = None, copy: bool = True) -> Any:
    """Values as an array of one type, which is `pandas.array`.

    Raises:
        ValueError: For a scalar, in pandas' words.
    """
    import numpy

    from ._frame import Index, Series
    from .errors import InvalidArgumentError

    if isinstance(data, str | bytes | numbers.Number) or not hasattr(data, "__iter__"):
        raise InvalidArgumentError(f"Cannot pass scalar '{data}' to 'pandas.array'.")
    if dtype is None and isinstance(data, Series | Index | FirepandaArray):
        return FirepandaArray(Series(list(data), dtype=data.dtype))
    if dtype is None and isinstance(data, numpy.ndarray):
        dtype = _NUMPY_MASKED.get(data.dtype.kind, data.dtype if data.dtype.kind != "O" else None)
        data = data.tolist()
    values = list(data)
    if dtype is None:
        dtype = _inferred(values)
    if str(dtype) == "category":
        from ._categorical import Categorical

        return Categorical(values)
    unit = re.fullmatch(r"(datetime64|timedelta64)\[(\w+)\]", str(dtype))
    if unit is not None:
        # A column is not cast to instants by type, so they are read as a
        # to_datetime reads them and then put in the unit asked for.
        from ._pandas import to_datetime, to_timedelta

        read = to_datetime if unit.group(1) == "datetime64" else to_timedelta
        return FirepandaArray(Series(read(values)).dt.as_unit(unit.group(2)))
    return FirepandaArray(Series(values, dtype=dtype))
