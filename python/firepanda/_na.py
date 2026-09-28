"""`NA`, pandas' missing value that is neither a float NaN nor None.

`NA` stands for a value that is not known, so almost anything done with it is
not known either: arithmetic and comparisons with a number, text, bytes or
another `NA` answer `NA`. The logical operators follow three-valued logic,
where `NA & False` is False and `NA | True` is True because the answer is the
same whatever the missing value is, and asking whether `NA` is true raises.

The rules were measured on pandas 3.0, including the few that are not
obvious: `NA ** 0` is 1 and `1 ** NA` is 1, a date answers `NA` only for a
comparison or a subtraction, and a span of time also for an addition.
"""

from __future__ import annotations

import datetime
import numbers
from collections.abc import Callable
from typing import Any

_COMPARISONS = ("eq", "ne", "le", "lt", "ge", "gt")

_HASH = 2**61 - 1

_UFUNCS = {
    "subtract": "sub",
    "multiply": "mul",
    "floor_divide": "floordiv",
    "true_divide": "truediv",
    "divide": "truediv",
    "power": "pow",
    "remainder": "mod",
    "equal": "eq",
    "not_equal": "ne",
    "less": "lt",
    "less_equal": "le",
    "greater": "gt",
    "greater_equal": "ge",
    "bitwise_or": "or",
    "bitwise_and": "and",
    "bitwise_xor": "xor",
}
"""numpy's names for the functions that are an operator, where they differ."""

_DISPATCHED = frozenset(
    [
        "add",
        "sub",
        "mul",
        "pow",
        "mod",
        "floordiv",
        "truediv",
        "divmod",
        "matmul",
        "and",
        "or",
        "xor",
        *_COMPARISONS,
    ]
)
"""The numpy functions answered by the operator of the same name, as pandas does."""


def _numpy_scalar(value: Any) -> bool:
    """Whether a value is a numpy array of no dimensions or a numpy flag."""
    kind = type(value)
    if kind.__module__ != "numpy":
        return False
    return kind.__name__ == "bool" or getattr(value, "shape", None) == ()


def _numpy_array(value: Any) -> bool:
    return type(value).__module__ == "numpy" and type(value).__name__ == "ndarray"


def _filled(shape: Any) -> Any:
    """A numpy array of objects of this shape, every one of them `NA`."""
    import numpy

    out = numpy.empty(shape, dtype=object)
    out[:] = NA
    return out


def _propagating(name: str, pair: bool = False) -> Callable[[NAType, Any], Any]:
    """An operator that answers `NA` for a number, text, bytes or `NA` on the other side."""
    comparison = name in _COMPARISONS

    def op(self: NAType, other: Any) -> Any:
        if other is NA or isinstance(other, (str, bytes, numbers.Number)) or _numpy_scalar(other):
            return (NA, NA) if pair else NA
        if _numpy_array(other):
            return (_filled(other.shape), _filled(other.shape)) if pair else _filled(other.shape)
        if comparison and isinstance(other, (datetime.date, datetime.time, datetime.timedelta)):
            return NA
        if isinstance(other, datetime.date) and name in ("sub", "rsub"):
            return NA
        if isinstance(other, datetime.timedelta) and name in ("sub", "rsub", "add", "radd"):
            return NA
        return NotImplemented

    op.__name__ = f"__{name}__"
    return op


class NAType:
    """The type of `NA`, which has one value."""

    _instance: NAType | None = None

    def __new__(cls, *args: Any, **kwargs: Any) -> NAType:
        if NAType._instance is None:
            NAType._instance = object.__new__(cls)
        return NAType._instance

    def __repr__(self) -> str:
        return "<NA>"

    def __format__(self, format_spec: str) -> str:
        try:
            return self.__repr__().__format__(format_spec)
        except ValueError:
            return self.__repr__()

    def __bool__(self) -> bool:
        raise TypeError("boolean value of NA is ambiguous")

    def __hash__(self) -> int:
        return _HASH

    def __reduce__(self) -> str:
        return "NA"

    def __copy__(self) -> NAType:
        return self

    def __deepcopy__(self, memo: Any) -> NAType:
        return self

    def __neg__(self) -> NAType:
        return self

    def __pos__(self) -> NAType:
        return self

    def __abs__(self) -> NAType:
        return self

    def __invert__(self) -> NAType:
        return self

    def __pow__(self, other: Any) -> Any:
        if other is NA:
            return NA
        if isinstance(other, numbers.Number) or _numpy_scalar(other):
            return type(other)(1) if other == 0 else NA
        if _numpy_array(other):
            import numpy

            return numpy.where(other == 0, other.dtype.type(1), NA)
        return NotImplemented

    def __rpow__(self, other: Any) -> Any:
        if other is NA:
            return NA
        if isinstance(other, numbers.Number) or _numpy_scalar(other):
            return other if other == 1 else NA
        if _numpy_array(other):
            import numpy

            return numpy.where(other == 1, other, NA)
        return NotImplemented

    def __and__(self, other: Any) -> Any:
        if other is False:
            return False
        if other is True or other is NA:
            return NA
        return NotImplemented

    __rand__ = __and__

    def __or__(self, other: Any) -> Any:
        if other is True:
            return True
        if other is False or other is NA:
            return NA
        return NotImplemented

    __ror__ = __or__

    def __xor__(self, other: Any) -> Any:
        if other is False or other is True or other is NA:
            return NA
        return NotImplemented

    __rxor__ = __xor__

    def __array_ufunc__(self, ufunc: Any, method: str, *inputs: Any, **kwargs: Any) -> Any:
        """A numpy function over `NA`, which answers `NA` in the shape of the other input."""
        import numpy

        for each in inputs:
            if not isinstance(each, (NAType, numpy.ndarray, numbers.Number, str, numpy.bool_)):
                return NotImplemented
        if method != "__call__":
            raise ValueError(f"ufunc method '{method}' not supported for NA")
        name = _UFUNCS.get(ufunc.__name__, ufunc.__name__)
        if len(inputs) == 2 and not kwargs and name in _DISPATCHED:
            reflected = inputs[1] is self
            other = inputs[0] if reflected else inputs[1]
            flipped = reflected and name not in _COMPARISONS
            answer = getattr(self, f"__{'r' if flipped else ''}{name}__")(other)
            if answer is not NotImplemented:
                return answer
        if ufunc.nout > 1:
            return (NA,) * ufunc.nout
        at = next(place for place, each in enumerate(inputs) if each is NA)
        result = numpy.broadcast_arrays(*inputs)[at]
        return result.item() if result.ndim == 0 else result


for _name in (
    "add",
    "radd",
    "sub",
    "rsub",
    "mul",
    "rmul",
    "matmul",
    "rmatmul",
    "truediv",
    "rtruediv",
    "floordiv",
    "rfloordiv",
    "mod",
    "rmod",
    *_COMPARISONS,
):
    setattr(NAType, f"__{_name}__", _propagating(_name))
NAType.__divmod__ = _propagating("divmod", pair=True)  # type: ignore[attr-defined]
NAType.__rdivmod__ = _propagating("rdivmod", pair=True)  # type: ignore[attr-defined]
del _name

NA = NAType()
"""The missing value that is neither a float NaN nor None."""


class _IndexSlice:
    """`IndexSlice`, which hands back whatever is put between its brackets.

    `IndexSlice[1:3, "a"]` is `(slice(1, 3), "a")`, a shorter way to write
    slices for `loc`.
    """

    def __getitem__(self, arg: Any) -> Any:
        return arg


IndexSlice = _IndexSlice()
