"""`PeriodDtype` against pandas, compared by repr."""

from __future__ import annotations

import pickle
import warnings
from collections.abc import Callable
from types import ModuleType
from typing import Any

import pytest

import firepanda as fp

pd = pytest.importorskip("pandas")

FREQS = ["D", "M", "2M", "Q-JAN", "Y", "W-WED", "B", "h", "ns", "period[D]", "Period[M]", " M"]
FREQS += ["-1D", "0D", "2h30min", "90min", "ME", "x", "period[x]"]


def made(lib: ModuleType, freq: Any) -> Any:
    kind = lib.PeriodDtype(freq)
    return kind, kind.name, str(kind), kind.freq


def attributes(lib: ModuleType, freq: str) -> Any:
    kind = lib.PeriodDtype(freq)
    return (
        kind.kind,
        kind.type.__name__,
        kind.na_value,
        kind.str,
        kind.base,
        kind.num,
        kind.itemsize,
        kind.shape,
        kind.names,
        kind.subdtype,
        kind.isbuiltin,
        kind.isnative,
    )


CASES: dict[str, Callable[[ModuleType], Any]] = {
    "from-offset": lambda lib: made(lib, lib.offsets.MonthEnd()),
    "from-two-days": lambda lib: made(lib, lib.offsets.Day(2)),
    "from-itself": lambda lib: made(lib, lib.PeriodDtype("M")),
    "from-none": lambda lib: lib.PeriodDtype(None),
    "from-number": lambda lib: lib.PeriodDtype(3),
    "from-span": lambda lib: lib.PeriodDtype(lib.Timedelta("1D")),
    "backwards-freq": lambda lib: lib.PeriodDtype("-2M").freq,
    "weekly-freq": lambda lib: lib.PeriodDtype("W").freq,
    "month": lambda lib: attributes(lib, "M"),
    "nanosecond": lambda lib: attributes(lib, "ns"),
    "type": lambda lib: lib.PeriodDtype.type.__name__,
    "equal-to-text": lambda lib: [
        lib.PeriodDtype("Y") == text
        for text in ("period[Y]", "period[Y-DEC]", "Period[Y-DEC]", "period[A]", "Y-DEC")
    ],
    "equal-to-dtype": lambda lib: (
        lib.PeriodDtype("M") == lib.PeriodDtype("M"),
        lib.PeriodDtype("M") == lib.PeriodDtype("2M"),
        lib.PeriodDtype("Y") == lib.PeriodDtype("12M"),
        lib.PeriodDtype("M") != "period[M]",
        lib.PeriodDtype("M") == "int64",
        lib.PeriodDtype("M") == None,  # noqa: E711
    ),
    "hash": lambda lib: hash(lib.PeriodDtype("M")) == hash(lib.PeriodDtype("M")),
    "in-list": lambda lib: lib.PeriodDtype("M") in ["period[M]"],
    "pickled": lambda lib: pickle.loads(pickle.dumps(lib.PeriodDtype("2M"))),
    "construct": lambda lib: lib.PeriodDtype.construct_from_string("period[2M]"),
    "construct-capital": lambda lib: lib.PeriodDtype.construct_from_string("Period[M]"),
    "construct-backwards": lambda lib: lib.PeriodDtype.construct_from_string("period[-1D]"),
    "construct-bare": lambda lib: lib.PeriodDtype.construct_from_string("M"),
    "construct-bad-freq": lambda lib: lib.PeriodDtype.construct_from_string("period[x]"),
    "construct-renamed": lambda lib: lib.PeriodDtype.construct_from_string("period[ME]"),
    "construct-unclosed": lambda lib: lib.PeriodDtype.construct_from_string("period[M"),
    "construct-number": lambda lib: lib.PeriodDtype.construct_from_string(5),
    "is-dtype": lambda lib: [
        lib.PeriodDtype.is_dtype(value)
        for value in ("period[M]", "M", "period[x]", None, 3, lib.PeriodDtype("M"))
    ],
    "pandas-dtype": lambda lib: lib.api.types.pandas_dtype("period[2M]"),
    "pandas-dtype-capital": lambda lib: lib.api.types.pandas_dtype("Period[M]"),
    "pandas-dtype-bad": lambda lib: lib.api.types.pandas_dtype("period[x]"),
    "pandas-dtype-given": lambda lib: lib.api.types.pandas_dtype(lib.PeriodDtype("M")),
    "dtype-equal": lambda lib: lib.api.types.is_dtype_equal(lib.PeriodDtype("M"), "period[M]"),
    "not-object": lambda lib: lib.api.types.is_object_dtype(lib.PeriodDtype("M")),
    "is-period": lambda lib: [
        lib.api.types.is_period_dtype(value)
        for value in ("period[M]", "period[x]", lib.PeriodDtype("M"), "M")
    ],
    "is-extension": lambda lib: lib.api.types.is_extension_array_dtype(lib.PeriodDtype("M")),
}


def mistake(error: Exception) -> str:
    kind = next(kind.__name__ for kind in type(error).__mro__ if kind.__module__ == "builtins")
    return f"{kind}: {error}"


def outcome(build: Callable[[], Any]) -> str:
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return repr(build())
    except Exception as error:
        return mistake(error)


@pytest.mark.parametrize("case", CASES.values(), ids=CASES.keys())
def test_answers_as_pandas(case: Callable[[ModuleType], Any]) -> None:
    assert outcome(lambda: case(fp)) == outcome(lambda: case(pd))


@pytest.mark.parametrize("freq", FREQS)
def test_made_as_in_pandas(freq: str) -> None:
    assert outcome(lambda: made(fp, freq)) == outcome(lambda: made(pd, freq))
