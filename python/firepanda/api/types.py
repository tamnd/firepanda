"""`firepanda.api.types`, the forty five questions other libraries ask about a column.

Nothing in here computes anything. Every name is either a question about a dtype
or a question about a Python object, and the code that asks them is almost never
the code that made the frame. `is_numeric_dtype` is what scikit-learn calls
before it fits, `is_list_like` is what half of pandas' own argument handling
calls on the way in, and `is_scalar` is what decides whether a value is a cell or
a column. A library that answers these wrongly is a library that fails inside
somebody else's stack trace.

### Why it is written by hand

The rest of the pandas surface is generated from the table in `tools/bindings.py`
because it crosses the language boundary and the two halves would drift. Nothing
here crosses anything. These are forty five pure Python functions over Python
objects and dtype names, and the Mojo side has no opinion about any of them, so
generating them would mean inventing a table that describes forty five different
bodies. The table earns its place when the bodies are the same shape. Here they
are not.

### The dtype is a string, and that is a decision already made

`Series.dtype` hands back `'int64'` rather than `numpy.dtype('int64')`, which is
document 13's call and predates this module. Everything here follows it: a dtype
is normalised to its pandas spelling as a string, and the predicates read the
string. That is what makes this work with no numpy in the install, which matters
because firepanda has no dependencies and this is the corner of pandas where
numpy is normally unavoidable.

It also means the four dtype classes below hand back their `name` rather than
wrapping a numpy dtype, and `pandas_dtype` hands back a string for the dtypes
that have no parameters. A program that compares the result against a string
keeps working. A program that compares it against `numpy.dtype` was already
going to notice, because `Series.dtype` told it first.

### numpy, when it is there

Some of these have to answer correctly about a `numpy.float32`, because a caller
who has numpy will hand one over. None of them import numpy. The trick is that a
value the caller is holding cannot be a numpy scalar unless numpy is already
imported, so the numpy scalar types are reached through `sys.modules` and the
lookup costs a dict hit and never pulls numpy into an install that does not have
it. Where that is not enough, the builtin ABCs in `numbers` do the work, because
numpy registers its scalar types with them.

### What is deprecated in pandas is deprecated here

Five of these warn in pandas 3.0. They warn here too, with the same message and
for the same reason, because a program running under `-W error::DeprecationWarning`
has to break in the same place in both libraries. The class is
`DeprecationWarning` rather than `Pandas4Warning`, since that name belongs to
pandas' release schedule and firepanda has no version four to point at.
"""

from __future__ import annotations

import numbers
import re
import sys
import warnings
from datetime import date, time, timedelta
from decimal import Decimal
from typing import Any

from ..errors import DTypeError, UnsupportedError

__all__ = [
    "CategoricalDtype",
    "DatetimeTZDtype",
    "IntervalDtype",
    "PeriodDtype",
    "infer_dtype",
    "is_any_real_numeric_dtype",
    "is_array_like",
    "is_bool",
    "is_bool_dtype",
    "is_categorical_dtype",
    "is_complex",
    "is_complex_dtype",
    "is_datetime64_any_dtype",
    "is_datetime64_dtype",
    "is_datetime64_ns_dtype",
    "is_datetime64tz_dtype",
    "is_dict_like",
    "is_dtype_equal",
    "is_extension_array_dtype",
    "is_file_like",
    "is_float",
    "is_float_dtype",
    "is_hashable",
    "is_int64_dtype",
    "is_integer",
    "is_integer_dtype",
    "is_interval_dtype",
    "is_iterator",
    "is_list_like",
    "is_named_tuple",
    "is_number",
    "is_numeric_dtype",
    "is_object_dtype",
    "is_period_dtype",
    "is_re",
    "is_re_compilable",
    "is_scalar",
    "is_signed_integer_dtype",
    "is_sparse",
    "is_string_dtype",
    "is_timedelta64_dtype",
    "is_timedelta64_ns_dtype",
    "is_unsigned_integer_dtype",
    "pandas_dtype",
    "union_categoricals",
]


# The dtype names, grouped the way the predicates below ask about them. Both
# spellings of each masked type are here, because pandas has two of everything
# in this area: `int64` is the numpy backed one and `Int64` is the nullable one,
# and `is_integer_dtype` says yes to both.
_BOOLS = frozenset({"bool", "bool_", "boolean"})
_SIGNED = frozenset({"int8", "int16", "int32", "int64", "Int8", "Int16", "Int32", "Int64"})
_UNSIGNED = frozenset(
    {"uint8", "uint16", "uint32", "uint64", "UInt8", "UInt16", "UInt32", "UInt64"}
)
_FLOATS = frozenset({"float16", "float32", "float64", "Float32", "Float64"})
_COMPLEX = frozenset({"complex64", "complex128"})
_STRINGS = frozenset({"str", "string", "large_string"})
_BYTES = frozenset({"bytes", "bytes_", "S", "binary", "large_binary"})

# The nested dtypes are matched on their opening rather than by name, because
# firepanda spells them with the element type inside, as `large_list<item:
# int64>`. Matching the bare word would also catch the builtin `list`, which is
# a class a caller passes to mean the object dtype and not a column of lists.
_NESTED = ("list<", "large_list<", "fixed_size_list<", "struct<", "map<")

# The dtypes pandas backs with an extension array rather than with a numpy one.
# This is not a property of the family, which is why it is its own set: `int64`
# is not an extension dtype and `Int64` is, and they are the same family.
_EXTENSIONS = (
    _STRINGS
    | frozenset({"boolean", "category", "Int8", "Int16", "Int32", "Int64"})
    | frozenset({"UInt8", "UInt16", "UInt32", "UInt64", "Float32", "Float64"})
)

# What a bare Python type means as a dtype. pandas accepts these and a surprising
# amount of code passes them, usually as `df.select_dtypes(include=[float])`.
_FROM_TYPE: dict[type, str] = {
    bool: "bool",
    int: "int64",
    float: "float64",
    complex: "complex128",
    str: "str",
    bytes: "bytes",
    object: "object",
}


def _deprecated(name: str, instead: str) -> None:
    """Warns the way pandas warns, so that a strict warning filter fires here too.

    Args:
        name: The function being called.
        instead: The sentence pandas puts after "Use".
    """
    warnings.warn(
        f"{name} is deprecated and will be removed in a future version. {instead}",
        DeprecationWarning,
        stacklevel=3,
    )


def _named(arr_or_dtype: Any) -> str:
    """Reduces anything a caller might pass to the dtype name it stands for.

    The order matters. A string is already a name, a type is looked up before
    anything tries to read attributes off it, and a column is asked for its
    `dtype` before the object itself is inspected, because a firepanda `Series`
    holds a name and is not one. Everything that falls through gets `str`
    applied, which is right for a numpy dtype and harmless for a frame or a list,
    since the result matches nothing and every predicate then says no. That is
    what pandas does with rubbish as well.

    Args:
        arr_or_dtype: A dtype, a string, a type, or something holding a dtype.

    Returns:
        The pandas spelling of the dtype, or an empty string when there is none.
    """
    if arr_or_dtype is None:
        return ""
    if isinstance(arr_or_dtype, str):
        return arr_or_dtype
    if isinstance(arr_or_dtype, type):
        found = _FROM_TYPE.get(arr_or_dtype)
        if found is not None:
            return found
        # `numpy.int64` is a type and its `__name__` is already the dtype name,
        # which is how this reaches numpy's scalar types without importing numpy
        # and without listing them. Any other class is the object dtype, because
        # a column of instances of it is a column of Python objects, and that is
        # the answer pandas gives for `list`, `tuple` and `dict`.
        return arr_or_dtype.__name__ if _sorted(arr_or_dtype.__name__) else "object"
    held = getattr(arr_or_dtype, "dtype", None)
    if held is not None and held is not arr_or_dtype:
        return _named(held)
    name = getattr(arr_or_dtype, "name", None)
    if isinstance(name, str):
        return name
    return str(arr_or_dtype)


def _family(arr_or_dtype: Any) -> str:
    """Sorts a dtype into the one group every predicate below is really asking about.

    Args:
        arr_or_dtype: A dtype, a string, a type, or something holding a dtype.

    Returns:
        One of the family names, or an empty string when nothing recognises it.
    """
    return _sorted(_named(arr_or_dtype))


def _sorted(name: str) -> str:
    """Sorts a dtype name, which is the half of `_family` that takes no object.

    It is split out because `_named` needs it: deciding what a bare class means
    as a dtype comes down to whether its name is one, and calling `_family` there
    would be a loop.

    Args:
        name: A dtype name.

    Returns:
        One of the family names, or an empty string when nothing recognises it.
    """
    core = name.split("[", 1)[0].strip()
    if core.startswith(_NESTED):
        return "nested"
    if core in _BOOLS:
        return "bool"
    if core in _SIGNED:
        return "signed"
    if core in _UNSIGNED:
        return "unsigned"
    if core in _FLOATS:
        return "float"
    if core in _COMPLEX:
        return "complex"
    if core in _STRINGS:
        return "string"
    if core in _BYTES:
        return "bytes"
    if core == "object":
        return "object"
    if core == "category":
        return "category"
    if core == "datetime64":
        return "datetime-tz" if "," in name else "datetime"
    if core == "timedelta64":
        return "timedelta"
    if core in {"date32", "date64"}:
        return "date"
    if core == "period":
        return "period"
    if core == "interval":
        return "interval"
    if core == "null":
        return "null"
    return ""


def _unit(arr_or_dtype: Any) -> str:
    """Reads the resolution out of a datetime or timedelta dtype name.

    Args:
        arr_or_dtype: A dtype, a string, a type, or something holding a dtype.

    Returns:
        The unit, or an empty string when the name carries none.
    """
    name = _named(arr_or_dtype)
    if "[" not in name:
        return ""
    return name.split("[", 1)[1].split(",", 1)[0].rstrip("]").strip()


class CategoricalDtype:
    """A column of a fixed set of values, described rather than held.

    firepanda stores this as an Arrow dictionary and prints it as `category`,
    which is what pandas prints, so two categorical columns holding different
    things have the same dtype in both libraries. That is the reason the
    categories live on this object rather than in the dtype name.

    Args:
        categories: The values, in the order they should be counted, or None to
            say that the column decides.
        ordered: Whether comparing two of them means anything.
    """

    __slots__ = ("_categories", "_ordered")

    def __init__(self, categories: Any = None, ordered: bool = False) -> None:
        self._categories = None if categories is None else list(categories)
        self._ordered = ordered

    @property
    def categories(self) -> Any:
        """The values, as an `Index`, or None when the column decides."""
        if self._categories is None:
            return None
        from .._frame import Index

        return Index(self._categories)

    @property
    def ordered(self) -> bool:
        """Whether comparing two of them means anything."""
        return self._ordered

    @property
    def name(self) -> str:
        """The dtype name, which says nothing about the categories."""
        return "category"

    @property
    def kind(self) -> str:
        """The numpy kind letter pandas reports for this, which is object."""
        return "O"

    def __str__(self) -> str:
        return "category"

    def __repr__(self) -> str:
        return f"CategoricalDtype(categories={self._categories!r}, ordered={self._ordered!r})"

    def __eq__(self, other: object) -> bool:
        # Equal to the bare string, because that is how pandas users write it and
        # `dtype == "category"` is the check in most of the code that cares.
        if isinstance(other, str):
            return other == "category"
        if not isinstance(other, CategoricalDtype):
            return NotImplemented
        return self._categories == other._categories and self._ordered == other._ordered

    def __hash__(self) -> int:
        return hash(("category", self._ordered))


class DatetimeTZDtype:
    """A column of instants that know which zone they were read in.

    Args:
        unit: The resolution, one of `s`, `ms`, `us` or `ns`.
        tz: The zone, which is required, since a datetime dtype without one is
            spelled `datetime64[unit]` and is a different dtype.

    Raises:
        TypeError: If no zone is given.
    """

    __slots__ = ("_tz", "_unit")

    def __init__(self, unit: Any = "ns", tz: Any = None) -> None:
        if tz is None:
            raise DTypeError("A 'tz' is required.")
        self._unit = str(unit)
        self._tz = tz

    @property
    def unit(self) -> str:
        """The resolution."""
        return self._unit

    @property
    def tz(self) -> Any:
        """The zone."""
        return self._tz

    @property
    def name(self) -> str:
        """The dtype name, which carries both the resolution and the zone."""
        return f"datetime64[{self._unit}, {self._tz}]"

    @property
    def kind(self) -> str:
        """The numpy kind letter, which is the same as a naive datetime's."""
        return "M"

    def __str__(self) -> str:
        return self.name

    def __repr__(self) -> str:
        return self.name

    def __eq__(self, other: object) -> bool:
        if isinstance(other, str):
            return other == self.name
        if not isinstance(other, DatetimeTZDtype):
            return NotImplemented
        return self._unit == other._unit and str(self._tz) == str(other._tz)

    def __hash__(self) -> int:
        return hash(self.name)


class IntervalDtype:
    """A column of ranges, described rather than held.

    firepanda has no interval column, so nothing produces this dtype. It exists
    because the predicates have to be able to say no to it, and a program that
    builds one and asks `is_interval_dtype` should get the same yes pandas gives.

    Args:
        subtype: The dtype of the two endpoints.
        closed: Which end is included, one of `left`, `right`, `both` or
            `neither`.
    """

    __slots__ = ("_closed", "_subtype")

    def __init__(self, subtype: Any = None, closed: Any = None) -> None:
        self._subtype = subtype
        self._closed = closed

    @property
    def subtype(self) -> Any:
        """The dtype of the two endpoints."""
        return self._subtype

    @property
    def closed(self) -> Any:
        """Which end is included."""
        return self._closed

    @property
    def name(self) -> str:
        """The dtype name, which is the bare word rather than the printed form.

        This is the one of the four where `name` and `str` disagree, and it
        disagrees in pandas too: `str` carries the endpoints and the closed end
        and `name` is `interval` whatever they are. It reads like an oversight
        and it is load bearing, because `df.select_dtypes("interval")` matches on
        the name and would otherwise match nothing.
        """
        return "interval"

    @property
    def kind(self) -> str:
        """The numpy kind letter pandas reports for this, which is object."""
        return "O"

    def _spelled(self) -> str:
        """The printed form, which is what carries the parameters."""
        if self._subtype is None:
            return "interval"
        if self._closed is None:
            return f"interval[{_named(self._subtype)}]"
        return f"interval[{_named(self._subtype)}, {self._closed}]"

    def __str__(self) -> str:
        return self._spelled()

    def __repr__(self) -> str:
        return self._spelled()

    def __eq__(self, other: object) -> bool:
        if isinstance(other, str):
            return other in {"interval", self._spelled()}
        if not isinstance(other, IntervalDtype):
            return NotImplemented
        return self._spelled() == other._spelled()

    def __hash__(self) -> int:
        return hash(self._spelled())


class PeriodDtype:
    """A column of spans of time, described rather than held.

    firepanda has no period column, and this is here for the reason
    `IntervalDtype` is. `freq` hands back the string it was given rather than an
    offset object, because the offsets are a namespace firepanda does not have
    yet and inventing half of one to fill this in would be worse than saying so.

    Args:
        freq: The length of one period, as a frequency string.
    """

    __slots__ = ("_freq",)

    def __init__(self, freq: Any) -> None:
        self._freq = freq

    @property
    def freq(self) -> Any:
        """The length of one period, as the string it was built from."""
        return self._freq

    @property
    def name(self) -> str:
        """The dtype name, which carries the frequency."""
        return f"period[{self._freq}]"

    @property
    def kind(self) -> str:
        """The numpy kind letter pandas reports for this, which is object."""
        return "O"

    def __str__(self) -> str:
        return self.name

    def __repr__(self) -> str:
        return self.name

    def __eq__(self, other: object) -> bool:
        if isinstance(other, str):
            return other == self.name
        if not isinstance(other, PeriodDtype):
            return NotImplemented
        return str(self._freq) == str(other._freq)

    def __hash__(self) -> int:
        return hash(self.name)


def pandas_dtype(dtype: Any) -> Any:
    """Turns anything that names a dtype into the dtype itself.

    The dtypes that carry no parameters come back as their name, which is what
    `Series.dtype` hands out and what the rest of firepanda compares against. The
    four that carry parameters come back as the classes above, because the name
    alone would lose the categories, the zone, the endpoints or the frequency.

    Args:
        dtype: A dtype, a string, a type, or something holding a dtype.

    Returns:
        The dtype, as a string or as one of the four dtype objects.

    Raises:
        TypeError: If nothing here recognises it.
    """
    if isinstance(dtype, CategoricalDtype | DatetimeTZDtype | IntervalDtype | PeriodDtype):
        return dtype
    family = _family(dtype)
    if family == "":
        raise DTypeError(f"data type {_named(dtype)!r} not understood")
    if family == "category":
        return CategoricalDtype()
    name = _named(dtype)
    if family == "datetime-tz":
        zone = name.split(",", 1)[1].rstrip("]").strip()
        return DatetimeTZDtype(_unit(name), zone)
    if family == "period":
        return PeriodDtype(name.split("[", 1)[1].rstrip("]"))
    if family == "interval":
        inside = name.split("[", 1)[1].rstrip("]") if "[" in name else ""
        if inside == "":
            return IntervalDtype()
        parts = [part.strip() for part in inside.split(",")]
        return IntervalDtype(parts[0], parts[1] if len(parts) > 1 else None)
    return name


def is_dtype_equal(source: Any, target: Any) -> bool:
    """Says whether two things name the same dtype.

    Args:
        source: A dtype, or something naming one.
        target: The other.

    Returns:
        True if they are the same dtype. False if they are not, and False rather
        than an exception if either of them names no dtype at all, which is what
        pandas does.
    """
    try:
        return bool(pandas_dtype(source) == pandas_dtype(target))
    except TypeError:
        return False


def is_bool_dtype(arr_or_dtype: Any) -> bool:
    """Says whether this is a column of true and false.

    Args:
        arr_or_dtype: A dtype, or something holding one.

    Returns:
        True for `bool` and for the nullable `boolean`.
    """
    return _family(arr_or_dtype) == "bool"


def is_integer_dtype(arr_or_dtype: Any) -> bool:
    """Says whether this is a column of whole numbers.

    `bool` is not one of them, here and in pandas, even though it is stored as
    one and `is_numeric_dtype` says yes to it.

    Args:
        arr_or_dtype: A dtype, or something holding one.

    Returns:
        True for every width, signed or unsigned, nullable or not.
    """
    return _family(arr_or_dtype) in {"signed", "unsigned"}


def is_signed_integer_dtype(arr_or_dtype: Any) -> bool:
    """Says whether this is a column of whole numbers that can be negative.

    Args:
        arr_or_dtype: A dtype, or something holding one.

    Returns:
        True for every signed width, nullable or not.
    """
    return _family(arr_or_dtype) == "signed"


def is_unsigned_integer_dtype(arr_or_dtype: Any) -> bool:
    """Says whether this is a column of whole numbers that cannot be negative.

    Args:
        arr_or_dtype: A dtype, or something holding one.

    Returns:
        True for every unsigned width, nullable or not.
    """
    return _family(arr_or_dtype) == "unsigned"


def is_int64_dtype(arr_or_dtype: Any) -> bool:
    """Says whether this is a column of sixty four bit signed whole numbers.

    Args:
        arr_or_dtype: A dtype, or something holding one.

    Returns:
        True for `int64` and for the nullable `Int64`, and not for the other
        widths.
    """
    _deprecated("is_int64_dtype", "Use dtype == np.int64 instead.")
    return _named(arr_or_dtype) in {"int64", "Int64"}


def is_float_dtype(arr_or_dtype: Any) -> bool:
    """Says whether this is a column of numbers with a fractional part.

    Args:
        arr_or_dtype: A dtype, or something holding one.

    Returns:
        True for every float width, nullable or not.
    """
    return _family(arr_or_dtype) == "float"


def is_complex_dtype(arr_or_dtype: Any) -> bool:
    """Says whether this is a column of complex numbers.

    firepanda has no complex column, so this is always False for anything
    firepanda produced. It is not always False for what a caller passes in.

    Args:
        arr_or_dtype: A dtype, or something holding one.

    Returns:
        True for `complex64` and `complex128`.
    """
    return _family(arr_or_dtype) == "complex"


def is_numeric_dtype(arr_or_dtype: Any) -> bool:
    """Says whether arithmetic on this column means anything.

    `bool` counts and so does `complex`, which is the pair of answers that
    surprises people. `is_any_real_numeric_dtype` is the one that excludes both.

    Args:
        arr_or_dtype: A dtype, or something holding one.

    Returns:
        True for booleans, whole numbers, floats and complex numbers.
    """
    return _family(arr_or_dtype) in {"bool", "signed", "unsigned", "float", "complex"}


def is_any_real_numeric_dtype(arr_or_dtype: Any) -> bool:
    """Says whether this column holds numbers that sit on the number line.

    Args:
        arr_or_dtype: A dtype, or something holding one.

    Returns:
        True for whole numbers and floats, and False for booleans and complex
        numbers, which is the difference from `is_numeric_dtype`.
    """
    return _family(arr_or_dtype) in {"signed", "unsigned", "float"}


def is_object_dtype(arr_or_dtype: Any) -> bool:
    """Says whether this column holds arbitrary Python objects.

    firepanda has no object column and never will, since the whole point of the
    Arrow layout is that a column is one type laid out end to end. This is here
    to answer no, and answering no is the useful part: a caller that branches on
    it takes the fast branch.

    Args:
        arr_or_dtype: A dtype, or something holding one.

    Returns:
        True only for `object`.
    """
    return _family(arr_or_dtype) == "object"


def is_string_dtype(arr_or_dtype: Any) -> bool:
    """Says whether this column holds text.

    `object` counts, which is the one that catches people out. pandas said yes to
    it long before there was a string dtype and code was written against that
    answer, so it still says yes. Bytes count too, for the same reason: numpy's
    fixed width bytes dtype is a string dtype to pandas, and firepanda's `binary`
    column is the same thing, so both answer the way the nearest pandas dtype
    answers rather than the way the name reads.

    Args:
        arr_or_dtype: A dtype, or something holding one.

    Returns:
        True for `str`, `string`, `object`, `bytes` and `binary`.
    """
    return _family(arr_or_dtype) in {"string", "object", "bytes"}


def is_datetime64_dtype(arr_or_dtype: Any) -> bool:
    """Says whether this is a column of instants with no zone on them.

    Args:
        arr_or_dtype: A dtype, or something holding one.

    Returns:
        True for a naive datetime column at any resolution, and False for a
        zoned one, which is `is_datetime64_any_dtype`.
    """
    return _family(arr_or_dtype) == "datetime"


def is_datetime64_any_dtype(arr_or_dtype: Any) -> bool:
    """Says whether this is a column of instants, zoned or not.

    Args:
        arr_or_dtype: A dtype, or something holding one.

    Returns:
        True for a datetime column at any resolution, with or without a zone.
    """
    return _family(arr_or_dtype) in {"datetime", "datetime-tz"}


def is_datetime64_ns_dtype(arr_or_dtype: Any) -> bool:
    """Says whether this is a column of instants counted in nanoseconds.

    firepanda reads timestamps at microseconds by default, so this is False for
    most firepanda columns and `is_datetime64_any_dtype` is the question that was
    meant. Code that asks this one is usually code that is about to reach for a
    numpy buffer.

    Args:
        arr_or_dtype: A dtype, or something holding one.

    Returns:
        True for a nanosecond datetime column, zoned or not.
    """
    return is_datetime64_any_dtype(arr_or_dtype) and _unit(arr_or_dtype) == "ns"


def is_datetime64tz_dtype(arr_or_dtype: Any) -> bool:
    """Says whether this is a column of instants that carry a zone.

    Args:
        arr_or_dtype: A dtype, or something holding one.

    Returns:
        True for a zoned datetime column at any resolution.
    """
    _deprecated("is_datetime64tz_dtype", "Check `isinstance(dtype, pd.DatetimeTZDtype)` instead.")
    return _family(arr_or_dtype) == "datetime-tz"


def is_timedelta64_dtype(arr_or_dtype: Any) -> bool:
    """Says whether this is a column of lengths of time.

    Args:
        arr_or_dtype: A dtype, or something holding one.

    Returns:
        True for a timedelta column at any resolution.
    """
    return _family(arr_or_dtype) == "timedelta"


def is_timedelta64_ns_dtype(arr_or_dtype: Any) -> bool:
    """Says whether this is a column of lengths of time counted in nanoseconds.

    Args:
        arr_or_dtype: A dtype, or something holding one.

    Returns:
        True for a nanosecond timedelta column and False for the other
        resolutions.
    """
    return is_timedelta64_dtype(arr_or_dtype) and _unit(arr_or_dtype) == "ns"


def is_categorical_dtype(arr_or_dtype: Any) -> bool:
    """Says whether this is a column drawn from a fixed set of values.

    Args:
        arr_or_dtype: A dtype, or something holding one.

    Returns:
        True for `category`.
    """
    _deprecated("is_categorical_dtype", "Use isinstance(dtype, pd.CategoricalDtype) instead")
    return _family(arr_or_dtype) == "category"


def is_interval_dtype(arr_or_dtype: Any) -> bool:
    """Says whether this is a column of ranges.

    firepanda has no interval column, so this is False for everything firepanda
    produced.

    Args:
        arr_or_dtype: A dtype, or something holding one.

    Returns:
        True for an interval dtype.
    """
    _deprecated("is_interval_dtype", "Use `isinstance(dtype, pd.IntervalDtype)` instead")
    return _family(arr_or_dtype) == "interval"


def is_period_dtype(arr_or_dtype: Any) -> bool:
    """Says whether this is a column of spans of time.

    firepanda has no period column, so this is False for everything firepanda
    produced.

    Args:
        arr_or_dtype: A dtype, or something holding one.

    Returns:
        True for a period dtype.
    """
    _deprecated("is_period_dtype", "Use `isinstance(dtype, pd.PeriodDtype)` instead")
    return _family(arr_or_dtype) == "period"


def is_sparse(arr: Any) -> bool:
    """Says whether this column stores only the values that are not the fill value.

    firepanda has no sparse layout and this is always False. It takes `arr`
    rather than `arr_or_dtype`, which is pandas' own inconsistency and is kept
    because the parameter name is part of the signature.

    Args:
        arr: A column.

    Returns:
        False, always.
    """
    _deprecated("is_sparse", "Check `isinstance(dtype, pd.SparseDtype)` instead.")
    return False


def is_extension_array_dtype(arr_or_dtype: Any) -> bool:
    """Says whether this dtype is backed by something other than a numpy array.

    The reason a caller asks is almost always that it is about to reach for a
    numpy buffer and wants to know whether it can. Every firepanda column is
    backed by Arrow, so the honest answer for firepanda is that none of them are
    numpy arrays, but the answer here is about the dtype rather than about the
    storage, and it is the same answer pandas gives for the same dtype name.

    Args:
        arr_or_dtype: A dtype, or something holding one.

    Returns:
        True for the nullable dtypes, the string dtypes, `category`, a zoned
        datetime, a period and an interval.
    """
    if _named(arr_or_dtype) in _EXTENSIONS:
        return True
    return _family(arr_or_dtype) in {"category", "datetime-tz", "period", "interval", "string"}


def _numpy_kind(obj: object) -> str:
    """Reads the numpy scalar kind letter off a value without importing numpy.

    A value the caller is holding cannot be a numpy scalar unless numpy has
    already been imported, so this looks in `sys.modules` rather than importing.
    In an install with no numpy the lookup misses and every caller falls back to
    the builtin types, which is the whole answer there.

    Args:
        obj: Any value.

    Returns:
        The kind letter, or an empty string when this is not a numpy scalar.
    """
    numpy = sys.modules.get("numpy")
    if numpy is None:
        return ""
    generic = getattr(numpy, "generic", None)
    if generic is None or not isinstance(obj, generic):
        return ""
    kind = getattr(getattr(obj, "dtype", None), "kind", "")
    return kind if isinstance(kind, str) else ""


def is_bool(obj: object) -> bool:
    """Says whether this value is a true or a false.

    Args:
        obj: Any value.

    Returns:
        True for a Python bool and for a numpy one.
    """
    return isinstance(obj, bool) or _numpy_kind(obj) == "b"


def is_integer(obj: object) -> bool:
    """Says whether this value is a whole number.

    A bool is not one, which is the opposite of what `isinstance(True, int)`
    says, and matching pandas here matters because this is what decides whether
    a value is a position.

    Args:
        obj: Any value.

    Returns:
        True for a Python int that is not a bool, and for a numpy integer.
    """
    if isinstance(obj, bool):
        return False
    return isinstance(obj, int) or _numpy_kind(obj) in {"i", "u"}


def is_float(obj: object) -> bool:
    """Says whether this value is a number with a fractional part.

    A `Decimal` is not one and neither is a `Fraction`, even though both are
    real numbers, because this asks about the type rather than about the
    mathematics.

    Args:
        obj: Any value.

    Returns:
        True for a Python float and for a numpy float of any width.
    """
    return isinstance(obj, float) or _numpy_kind(obj) == "f"


def is_complex(obj: object) -> bool:
    """Says whether this value is a complex number.

    Args:
        obj: Any value.

    Returns:
        True for a Python complex and for a numpy one.
    """
    if isinstance(obj, complex) and not isinstance(obj, bool | int | float):
        return True
    return _numpy_kind(obj) == "c"


def is_number(obj: object) -> bool:
    """Says whether this value is a number of any kind.

    A bool is a number here. A numpy bool is not, which is pandas' answer and is
    the one place in this file where the two spellings of the same idea disagree.
    The line is the `numbers` tower: a numpy bool is registered with nothing in
    it and a numpy integer is registered with all of it.

    Args:
        obj: Any value.

    Returns:
        True for anything in the `numbers` tower, which includes `Decimal` and
        `Fraction`.
    """
    return isinstance(obj, numbers.Number)


def is_scalar(val: object) -> bool:
    """Says whether this value is one cell rather than a column of them.

    This is the question that decides whether `df["a"] = x` fills a column or
    assigns one, so it is asked constantly and the edge cases are all real. None
    is a scalar. A tuple is not, even an empty one. Bytes are, and a string is,
    though both are iterable.

    Args:
        val: Any value.

    Returns:
        True for a value that goes in one cell.
    """
    if val is None:
        return True
    if isinstance(val, str | bytes | numbers.Number | date | time | timedelta | Decimal):
        return True
    return _numpy_kind(val) != ""


def is_list_like(obj: object, allow_sets: bool = True) -> bool:
    """Says whether this value is a sequence of things rather than one thing.

    A string is not list like and neither are bytes, which is the special case
    that makes this function exist at all. A type is not list like either, even
    when its instances are, because a class that defines `__iter__` still is not
    a sequence of anything.

    Args:
        obj: Any value.
        allow_sets: Whether a set counts. It is False in the callers that are
            about to put the values in order, since a set has none.

    Returns:
        True for something iterable that is not text and not a class.
    """
    if isinstance(obj, str | bytes | type):
        return False
    if not hasattr(obj, "__iter__"):
        return False
    return allow_sets or not isinstance(obj, frozenset | set)


def is_array_like(obj: object) -> bool:
    """Says whether this value is a column rather than any old sequence.

    A list is not array like. A `Series`, an `Index` and a numpy array are,
    because they carry a dtype, and the dtype is the difference: it is what makes
    the values one type rather than a heap of objects that happen to be in a row.

    Args:
        obj: Any value.

    Returns:
        True for something list like that also has a `dtype`.
    """
    return is_list_like(obj) and hasattr(obj, "dtype")


def is_dict_like(obj: object) -> bool:
    """Says whether this value can be looked up by key.

    A `Series` is dict like, which is deliberate on pandas' part and is why
    passing one where a mapping is expected works.

    Args:
        obj: Any value.

    Returns:
        True for something with `keys`, `__getitem__` and `__contains__` that is
        not a class.
    """
    if isinstance(obj, type):
        return False
    return all(hasattr(obj, name) for name in ("keys", "__getitem__", "__contains__"))


def is_file_like(obj: object) -> bool:
    """Says whether this value is an open file rather than a path to one.

    Args:
        obj: Any value.

    Returns:
        True for something that can be read or written and can be iterated,
        which is what separates a file object from a path or a buffer protocol.
    """
    if not (hasattr(obj, "read") or hasattr(obj, "write")):
        return False
    return hasattr(obj, "__iter__")


def is_iterator(obj: object) -> bool:
    """Says whether this value is a stream that is consumed by reading it.

    The reason this is asked is that an iterator can only be walked once, so
    anything that needs two passes has to make a list first.

    Args:
        obj: Any value.

    Returns:
        True for something with `__next__`.
    """
    return hasattr(obj, "__next__")


def is_hashable(obj: object, allow_slice: bool = True) -> bool:
    """Says whether this value can be a key.

    It hashes the value rather than checking for `__hash__`, because a type can
    declare the method and raise from it, and a list inside a tuple makes the
    tuple unhashable without changing either type.

    Args:
        obj: Any value.
        allow_slice: Whether a slice counts. Slices became hashable in Python
            3.12 and code written before that treats them as labels rather than
            as keys, which is what this is for.

    Returns:
        True if hashing it works.
    """
    if not allow_slice and isinstance(obj, slice):
        return False
    try:
        hash(obj)
    except TypeError:
        return False
    return True


def is_named_tuple(obj: object) -> bool:
    """Says whether this value is a tuple whose positions have names.

    Args:
        obj: Any value.

    Returns:
        True for a tuple carrying `_fields`, which is what `namedtuple` and
        `NamedTuple` both put there.
    """
    return isinstance(obj, tuple) and hasattr(obj, "_fields")


def is_re(obj: object) -> bool:
    """Says whether this value is an already compiled regular expression.

    Args:
        obj: Any value.

    Returns:
        True for a compiled pattern and False for the string it was compiled
        from.
    """
    return isinstance(obj, re.Pattern)


def is_re_compilable(obj: object) -> bool:
    """Says whether this value could be compiled into a regular expression.

    pandas raises here when the value is a string that is not a valid pattern,
    because it catches the wrong exception, and `is_re_compilable("[")` is a
    `PatternError` rather than a False. firepanda answers False, which is what
    the name promises. It is the one behaviour in this file that is deliberately
    not what pandas does. It is not in the divergence registry, because a
    registry entry is checked by running a case and the conformance board has no
    way to run one of these yet, so it is recorded in the specification and in
    `python/tests/test_api_types.py` instead.

    Args:
        obj: Any value.

    Returns:
        True if `re.compile` would accept it.
    """
    try:
        re.compile(obj)  # type: ignore[call-overload]
    except (TypeError, re.error):
        return False
    return True


def infer_dtype(value: object, skipna: bool = True) -> str:
    """Names the kind of thing a sequence holds, by looking at what is in it.

    This is what the constructors use before they decide on a dtype, so the names
    it returns are a vocabulary rather than dtypes: `mixed-integer` is what a
    sequence of whole numbers and something else is called, and there is no dtype
    by that name.

    Args:
        value: A sequence.
        skipna: Whether the missing values are ignored. With it False, a missing
            value is a value of its own and a sequence of numbers and nulls is
            mixed.

    Returns:
        One of the pandas inference names.

    Raises:
        TypeError: If the value is not iterable, since a scalar has nothing to
            infer from.
    """
    if not hasattr(value, "__iter__"):
        raise DTypeError(f"'{type(value).__name__}' object is not iterable")
    values = list(value)
    if skipna:
        values = [item for item in values if not _missing(item)]
    if not values:
        return "empty"
    for name, test in _INFERENCE:
        if all(test(item) for item in values):
            return name
    numeric = [item for item in values if is_integer(item) or is_float(item)]
    if len(numeric) == len(values):
        return "mixed-integer-float"
    if any(is_integer(item) for item in values):
        return "mixed-integer"
    return "mixed"


def _missing(item: object) -> bool:
    """Says whether one value in a sequence is a missing one.

    Args:
        item: One value.

    Returns:
        True for None and for a float that is not a number.
    """
    if item is None:
        return True
    return isinstance(item, float) and item != item


# The inference names, in the order they are tried. Order matters twice: a bool
# is an int in Python so booleans have to be asked about first, and a datetime is
# a date so datetimes have to be asked about before dates.
_INFERENCE: tuple[tuple[str, Any], ...] = (
    ("boolean", is_bool),
    ("integer", is_integer),
    ("floating", is_float),
    ("complex", is_complex),
    ("decimal", lambda item: isinstance(item, Decimal)),
    ("string", lambda item: isinstance(item, str)),
    ("bytes", lambda item: isinstance(item, bytes)),
    ("datetime", lambda item: isinstance(item, date) and type(item) is not date),
    ("date", lambda item: type(item) is date),
    ("time", lambda item: isinstance(item, time)),
    ("timedelta", lambda item: isinstance(item, timedelta)),
)


def union_categoricals(
    to_union: Any, sort_categories: bool = False, ignore_order: bool = False
) -> Any:
    """Puts several categorical columns end to end over the union of their categories.

    Args:
        to_union: The columns.
        sort_categories: Whether the resulting categories come back in order.
        ignore_order: Whether an ordered column may be unioned with one that
            orders its categories differently.

    Raises:
        NotImplementedError: Always. firepanda stores a categorical column as an
            Arrow dictionary and has no `Categorical` object for this to take or
            hand back, so there is nothing here to union yet.
    """
    raise UnsupportedError(
        "union_categoricals is not supported yet, because firepanda has no Categorical"
        " object to union, only a category dtype on a column"
    )
