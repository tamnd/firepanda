"""The core names, the same objects `firepanda` exports at the top."""

from __future__ import annotations

from typing import Any as _Any

from .._array import array
from .._arrowtyped import ArrowDtype
from .._attrs import Flags
from .._categorical import Categorical, CategoricalDtype
from .._category_index import CategoricalIndex
from .._config import set_eng_float_format
from .._date_range import bdate_range, date_range
from .._datetime import DatetimeIndex
from .._dtypes import DatetimeTZDtype, StringDtype
from .._frame import DataFrame, Index, Series
from .._interval import Interval, IntervalDtype, IntervalIndex, interval_range
from .._masked import (
    BooleanDtype,
    Float32Dtype,
    Float64Dtype,
    Int8Dtype,
    Int16Dtype,
    Int32Dtype,
    Int64Dtype,
    UInt8Dtype,
    UInt16Dtype,
    UInt32Dtype,
    UInt64Dtype,
)
from .._multi import MultiIndex
from .._na import NA, IndexSlice
from .._pandas import (
    Grouper,
    NamedAgg,
    factorize,
    isna,
    isnull,
    notna,
    notnull,
    to_timedelta,
    unique,
)
from .._period import Period, PeriodDtype
from .._period_index import PeriodIndex, period_range
from .._range_index import RangeIndex
from .._scalars import NaT, Timedelta, Timestamp
from .._timedelta import TimedeltaIndex, timedelta_range
from ..offsets import DateOffset

_WRAPPED = ("to_datetime", "to_numeric")
"""The names `_attrs.install` wraps on the package after this module loads."""


def __getattr__(name: str) -> _Any:
    if name in _WRAPPED:
        import firepanda

        return getattr(firepanda, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted([*globals(), *_WRAPPED])


__all__ = [
    "NA",
    "ArrowDtype",
    "BooleanDtype",
    "Categorical",
    "CategoricalDtype",
    "CategoricalIndex",
    "DataFrame",
    "DateOffset",
    "DatetimeIndex",
    "DatetimeTZDtype",
    "Flags",
    "Float32Dtype",
    "Float64Dtype",
    "Grouper",
    "Index",
    "IndexSlice",
    "Int8Dtype",
    "Int16Dtype",
    "Int32Dtype",
    "Int64Dtype",
    "Interval",
    "IntervalDtype",
    "IntervalIndex",
    "MultiIndex",
    "NaT",
    "NamedAgg",
    "Period",
    "PeriodDtype",
    "PeriodIndex",
    "RangeIndex",
    "Series",
    "StringDtype",
    "Timedelta",
    "TimedeltaIndex",
    "Timestamp",
    "UInt8Dtype",
    "UInt16Dtype",
    "UInt32Dtype",
    "UInt64Dtype",
    "array",
    "bdate_range",
    "date_range",
    "factorize",
    "interval_range",
    "isna",
    "isnull",
    "notna",
    "notnull",
    "period_range",
    "set_eng_float_format",
    "timedelta_range",
    "to_datetime",  # noqa: F822
    "to_numeric",  # noqa: F822
    "to_timedelta",
    "unique",
]
