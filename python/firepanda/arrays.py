"""`firepanda.arrays`, the array classes pandas names in `pandas.arrays`.

Each is the class an array of that type answers to, so `isinstance` against
them works the way it does in pandas. They are built by `array`, `Series.array`
and the methods that answer an array, rather than directly, since they all hold
their values the same way. `SparseArray` and `StringArray` are absent, because
nothing here stores values sparsely or as Python objects.
"""

from ._array import (
    ArrowExtensionArray,
    ArrowStringArray,
    BooleanArray,
    DatetimeArray,
    FloatingArray,
    IntegerArray,
    IntervalArray,
    NumpyExtensionArray,
    PeriodArray,
    TimedeltaArray,
)
from ._categorical import Categorical

__all__ = [
    "ArrowExtensionArray",
    "ArrowStringArray",
    "BooleanArray",
    "Categorical",
    "DatetimeArray",
    "FloatingArray",
    "IntegerArray",
    "IntervalArray",
    "NumpyExtensionArray",
    "PeriodArray",
    "TimedeltaArray",
]
