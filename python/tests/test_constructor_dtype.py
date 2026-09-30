"""`dtype=` on the Index, Series and DataFrame constructors, against pandas.

An index reads its labels the way a series of the same type reads them, text
asked for as instants is parsed rather than cast, and a list that does not fit
a whole number type is refused rather than truncated or wrapped.
"""

from __future__ import annotations

from collections.abc import Callable
from types import ModuleType
from typing import Any

import pytest

import firepanda as fp

pd = pytest.importorskip("pandas")

CASES: dict[str, Callable[[ModuleType], Any]] = {
    "index-int64": lambda lib: lib.Index([1, 2], dtype="int64"),
    "index-float64": lambda lib: lib.Index([1, 2], dtype="float64"),
    "index-int32": lambda lib: lib.Index([1, 2], dtype="int32"),
    "index-uint8": lambda lib: lib.Index([1, 2], dtype="uint8"),
    "index-str": lambda lib: lib.Index([1, 2], dtype="str"),
    "index-bool": lambda lib: lib.Index([1, 0], dtype="bool"),
    "index-dates": lambda lib: lib.Index(["2026-01-01", "2026-02-01"], dtype="datetime64[ns]"),
    "index-dates-s": lambda lib: lib.Index(["2026-01-01"], dtype="datetime64[s]"),
    "index-spans": lambda lib: lib.Index(["1D", "2h"], dtype="timedelta64[ns]"),
    "index-named": lambda lib: lib.Index([1, 2], dtype="float64", name="n"),
    "index-of-index": lambda lib: lib.Index(lib.Index([1, 2]), dtype="float64"),
    "index-of-series": lambda lib: lib.Index(lib.Series([1, 2], name="s"), dtype="float64"),
    "index-copy": lambda lib: lib.Index([1, 2], copy=True),
    "index-fraction": lambda lib: lib.Index([1.5], dtype="int64"),
    "series-dates": lambda lib: lib.Series(["2026-01-01", None], dtype="datetime64[ns]"),
    "series-dates-ms": lambda lib: lib.Series(["2026-01-01 10:00"], dtype="datetime64[ms]"),
    "series-dates-tz": lambda lib: lib.Series(["2026-01-01"], dtype="datetime64[ns, UTC]"),
    "series-dates-mixed": lambda lib: lib.Series(
        ["2026-01-01", None, "2026-01-03 10:00"], dtype="datetime64[ms]"
    ),
    "series-dates-words": lambda lib: lib.Series(
        ["Jan 2 2026", "2026-01-03"], dtype="datetime64[s]"
    ),
    "index-dates-mixed": lambda lib: lib.Index(
        ["2026-01-01", "2026-01-03 10:00"], dtype="datetime64[s]"
    ),
    "series-spans": lambda lib: lib.Series(["1D", None], dtype="timedelta64[ns]"),
    "series-spans-s": lambda lib: lib.Series(["90s"], dtype="timedelta64[s]"),
    "series-fraction": lambda lib: lib.Series([1.7, 2], dtype="int64"),
    "series-fraction-tuple": lambda lib: lib.Series((1.5,), dtype="int8"),
    "series-whole-floats": lambda lib: lib.Series([1.0, 2.0], dtype="uint8"),
    "series-none": lambda lib: lib.Series([1.0, None], dtype="int64"),
    "series-nan": lambda lib: lib.Series([1.0, float("nan")], dtype="int64"),
    "series-inf": lambda lib: lib.Series([1.0, float("inf")], dtype="int32"),
    "series-negative-unsigned": lambda lib: lib.Series([-1], dtype="uint8"),
    "series-too-big": lambda lib: lib.Series([300.0], dtype="uint8"),
    "series-flags": lambda lib: lib.Series([True, False], dtype="int64"),
    "series-text": lambda lib: lib.Series(["1", "2"], dtype="int64"),
    "series-cast-truncates": lambda lib: lib.Series(lib.Series([1.7]), dtype="int64"),
    "frame-fraction": lambda lib: lib.DataFrame({"a": [1.5]}, dtype="int64"),
    "frame-whole": lambda lib: lib.DataFrame({"a": [1.0], "b": [2]}, dtype="int64"),
    "frame-dates": lambda lib: lib.DataFrame(
        {"a": ["2026-01-01", "2026-01-02"]}, dtype="datetime64[ns]"
    ),
    "frame-dates-rows": lambda lib: (
        lib.DataFrame([["2026-01-01"]], columns=["a"], dtype="datetime64[s]").dtypes
    ),
    "frame-spans": lambda lib: lib.DataFrame({"a": ["1D", "2h"]}, dtype="timedelta64[ns]"),
    "frame-counts-as-dates": lambda lib: lib.DataFrame({"a": [1, 2]}, dtype="datetime64[ns]"),
    "frame-objects": lambda lib: lib.DataFrame({"a": [1, "x"]}, dtype=object).dtypes,
    "frame-objects-ints": lambda lib: lib.DataFrame({"a": [1, 2]}, dtype=object),
    "astype-text-dates": lambda lib: lib.Series(["2026-01-01", None]).astype("datetime64[ns]"),
    "astype-text-dates-ms": lambda lib: lib.Series(["2026-01-01 10:00:00.5"]).astype(
        "datetime64[ms]"
    ),
    "astype-text-mixed": lambda lib: lib.Series(["2026-01-01", None, "2026-03-04 10:00"]).astype(
        "datetime64[ms]"
    ),
    "astype-text-words": lambda lib: lib.Series(["Jan 2 2026", "2026-01-03"]).astype(
        "datetime64[s]"
    ),
    "astype-text-named": lambda lib: lib.Series(["2026-01-01"], index=[5], name="d").astype(
        "datetime64[ns]"
    ),
    "astype-text-spans": lambda lib: lib.Series(["1D", "2h", None]).astype("timedelta64[s]"),
    "astype-text-bad": lambda lib: lib.Series(["nope"]).astype("datetime64[ns]"),
    "astype-frame-dates": lambda lib: (
        lib.DataFrame({"a": ["2026-01-01"], "b": [1]}).astype({"a": "datetime64[ns]"}).dtypes
    ),
    "astype-frame-objects": lambda lib: (
        lib.DataFrame({"a": [1, 2], "b": ["x", "y"]}).astype(object).dtypes
    ),
    "mask-none-flag": lambda lib: lib.Series([1, 2, 3]).mask([True, None, False]),
    "where-none-flag": lambda lib: lib.Series([1, 2, 3]).where([True, None, False]),
}


def outcome(build: Callable[[], Any]) -> str:
    try:
        found = build()
    except Exception as error:
        kind = next(k.__name__ for k in type(error).__mro__ if k.__module__ == "builtins")
        return f"{kind}: {error}"
    return f"{type(found).__name__} {found!r}"


@pytest.mark.parametrize("case", CASES.values(), ids=CASES.keys())
def test_answers_as_pandas(case: Callable[[ModuleType], Any]) -> None:
    assert outcome(lambda: case(fp)) == outcome(lambda: case(pd))


def test_an_index_of_a_category_prints_as_pandas_prints() -> None:
    """The class is still `Index`, since there is no `CategoricalIndex` class yet."""
    made = [lib.Index(["a", "b", "a"], dtype="category") for lib in (fp, pd)]
    assert repr(made[0]) == repr(made[1])
