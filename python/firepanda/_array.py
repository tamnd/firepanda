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

if TYPE_CHECKING:
    from ._frame import Series


class FirepandaArray:
    """Values of one type in order, with no row labels and no name."""

    __slots__ = ("_column",)

    def __init__(self, column: Series) -> None:
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
        return FirepandaArray(self._column.copy())

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
        dtype = "str" if str(self.dtype) == "string" else self.dtype
        return (
            f"<{type(self).__name__}>\n{body.rstrip(', ' + chr(10))}\n"
            f"Length: {len(values)}, dtype: {dtype}"
        )


def _gap(value: Any) -> bool:
    from ._pandas import _missing

    return _missing(value)


class NumpyExtensionArray(FirepandaArray):
    """Values of a numpy type, which is `pandas.arrays.NumpyExtensionArray`."""

    __slots__ = ()


class IntegerArray(FirepandaArray):
    """Whole numbers of a masked type, which is `pandas.arrays.IntegerArray`."""

    __slots__ = ()

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

    def _shown(self, value: Any) -> str:
        from ._pandas import _pprinted

        return "nan" if _gap(value) else _pprinted(value)


class ArrowExtensionArray(FirepandaArray):
    """Values of an Arrow type, which is `pandas.arrays.ArrowExtensionArray`."""

    __slots__ = ()

    def _shown(self, value: Any) -> str:
        from ._pandas import _pprinted

        return "<NA>" if _gap(value) else _pprinted(value)


class DatetimeArray(FirepandaArray):
    """Instants, which is `pandas.arrays.DatetimeArray`."""

    __slots__ = ()

    def _shown(self, value: Any) -> str:
        return "'NaT'" if _gap(value) else f"'{value}'"


class TimedeltaArray(FirepandaArray):
    """Spans, which is `pandas.arrays.TimedeltaArray`."""

    __slots__ = ()

    def _formatter(self, values: list[Any]) -> Any:
        present = [value for value in values if not _gap(value)]
        days = all(v.seconds == v.microseconds == v.nanoseconds == 0 for v in present)

        def shown(value: Any) -> str:
            if _gap(value):
                return "NaT"
            return f"'{value.days} days'" if days else f"'{value}'"

        return shown


class PeriodArray(FirepandaArray):
    """Periods, which is `pandas.arrays.PeriodArray`."""

    __slots__ = ()

    def _shown(self, value: Any) -> str:
        return "'NaT'" if _gap(value) else f"'{value}'"


class IntervalArray(FirepandaArray):
    """Intervals, which is `pandas.arrays.IntervalArray`."""

    __slots__ = ()

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
