"""`StringDtype`, `DatetimeTZDtype`, `dt.timetz` and a group by's key columns, against pandas."""

from __future__ import annotations

import datetime
from collections.abc import Callable
from types import ModuleType
from typing import Any

import pytest

import firepanda as fp

pd = pytest.importorskip("pandas")
np = pytest.importorskip("numpy")


def utc(lib: ModuleType) -> Any:
    return lib.DatetimeTZDtype(tz="UTC")


def keyed(lib: ModuleType) -> Any:
    return lib.DataFrame({"a": [1, 1, 2], "b": [1, 2, 3], "c": ["x", "y", "x"]})


def clock(lib: ModuleType, zone: str | None) -> Any:
    column = lib.Series(lib.to_datetime(["2026-01-01 10:30", None]))
    return column.dt.tz_localize(zone) if zone else column


CASES: dict[str, Callable[[ModuleType], Any]] = {
    "string": lambda lib: lib.StringDtype(),
    "string-text": lambda lib: str(lib.StringDtype()),
    "string-nan": lambda lib: lib.StringDtype(na_value=np.nan),
    "string-python": lambda lib: lib.StringDtype("python"),
    "string-attributes": lambda lib: (
        lib.StringDtype().storage,
        lib.StringDtype().na_value,
        lib.StringDtype().name,
        lib.StringDtype().kind,
        lib.StringDtype().type,
    ),
    "string-equality": lambda lib: (
        lib.StringDtype() == "string",
        lib.StringDtype() == "str",
        lib.StringDtype(na_value=np.nan) == "str",
        lib.StringDtype(na_value=np.nan) == "string",
        lib.StringDtype("python") == lib.StringDtype(),
        lib.StringDtype() == "string[pyarrow]",
    ),
    "string-column": lambda lib: lib.Series(["a"]).dtype == lib.StringDtype(na_value=np.nan),
    "string-storage": lambda lib: lib.StringDtype("x"),
    "string-na": lambda lib: lib.StringDtype(na_value=0),
    "zoned": lambda lib: utc(lib),
    "zoned-tokyo": lambda lib: lib.DatetimeTZDtype("ms", "Asia/Tokyo"),
    "zoned-attributes": lambda lib: (
        utc(lib).unit,
        utc(lib).tz,
        utc(lib).kind,
        utc(lib).base,
        utc(lib).str,
        utc(lib).na_value,
        lib.DatetimeTZDtype("s", "Asia/Tokyo").tz,
    ),
    "zoned-equality": lambda lib: (
        utc(lib) == "datetime64[ns, UTC]",
        utc(lib) == lib.DatetimeTZDtype("ns", datetime.UTC),
        utc(lib) == "datetime64[us, UTC]",
    ),
    "zoned-hash": lambda lib: {utc(lib): 1}["datetime64[ns, UTC]"],
    "zoned-no-zone": lambda lib: lib.DatetimeTZDtype(),
    "zoned-day": lambda lib: lib.DatetimeTZDtype("D", "UTC"),
    "zoned-alias": lambda lib: lib.DatetimeTZDtype("datetime64[ns, UTC]"),
    "zoned-from-text": lambda lib: lib.DatetimeTZDtype.construct_from_string(
        "datetime64[ms, Asia/Tokyo]"
    ),
    "zoned-from-bad-text": lambda lib: lib.DatetimeTZDtype.construct_from_string("x"),
    "zoned-column": lambda lib: clock(lib, "UTC").dt.as_unit("ns").dtype == utc(lib),
    "timetz": lambda lib: clock(lib, None).dt.timetz,
    "timetz-zoned": lambda lib: clock(lib, "Asia/Tokyo").dt.timetz,
    "key-selected": lambda lib: keyed(lib).groupby("a")["a"].sum(),
    "key-counted": lambda lib: keyed(lib).groupby("a")["a"].count(),
    "key-of-two": lambda lib: keyed(lib).groupby(["a", "c"])["a"].max(),
    "key-in-a-list": lambda lib: keyed(lib).groupby("a")[["a", "b"]].sum(),
    "key-transform": lambda lib: keyed(lib).groupby("a")["a"].transform("sum"),
    "key-attribute": lambda lib: keyed(lib).groupby("a").a.agg(["sum", "max"]),
    "key-groups": lambda lib: list(keyed(lib).groupby("a")["a"]),
    "outside-key-names": lambda lib: keyed(lib).groupby([lib.Series([5, 5, 6]), "c"])["b"].sum(),
    "columns-in-dir": lambda lib: [n for n in dir(keyed(lib).groupby("a")) if len(n) == 1],
}


def mistake(error: Exception) -> str:
    """A mistake as its builtin class and message, since each library raises its own subclass."""
    kind = next(kind.__name__ for kind in type(error).__mro__ if kind.__module__ == "builtins")
    return f"{kind}: {error}"


def outcome(build: Callable[[], Any]) -> str:
    try:
        return repr(build())
    except Exception as error:
        return mistake(error)


@pytest.mark.parametrize("case", CASES.values(), ids=CASES.keys())
def test_answers_as_pandas(case: Callable[[ModuleType], Any]) -> None:
    assert outcome(lambda: case(fp)) == outcome(lambda: case(pd))
