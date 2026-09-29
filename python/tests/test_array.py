"""`array` and the classes of `arrays` against pandas, compared by class name and repr."""

from __future__ import annotations

from collections.abc import Callable
from types import ModuleType
from typing import Any

import numpy as np
import pytest

import firepanda as fp

pd = pytest.importorskip("pandas")

CASES: dict[str, Callable[[ModuleType], Any]] = {
    "int": lambda lib: lib.array([1, 2, None]),
    "int-plain": lambda lib: lib.array([1, 2]),
    "float": lambda lib: lib.array([1.5, None]),
    "float-nan": lambda lib: lib.array([1.5, float("nan")]),
    "bool": lambda lib: lib.array([True, None]),
    "mixed": lambda lib: lib.array(["a", 1]),
    "dates": lambda lib: lib.array([lib.Timestamp("2026-01-01"), None]),
    "spans": lambda lib: lib.array([lib.Timedelta("1D"), None]),
    "periods": lambda lib: lib.array([lib.Period("2026-01", "M"), None]),
    "dtype-int64": lambda lib: lib.array([1, 2], dtype="int64"),
    "dtype-Int8": lambda lib: lib.array([1, 2], dtype="Int8"),
    "dtype-str": lambda lib: lib.array([1, 2], dtype="str"),
    "dtype-cat": lambda lib: lib.array(["a", "b", "a"], dtype="category"),
    "dtype-float32": lambda lib: lib.array([1, 2], dtype="float32"),
    "np": lambda lib: lib.array(np.array([1, 2])),
    "np-f": lambda lib: lib.array(np.array([1.0, 2.0])),
    "series": lambda lib: lib.array(lib.Series([1, 2])),
    "scalar": lambda lib: lib.array(1),
    "len": lambda lib: len(lib.array([1, 2, None])),
    "dtype-of": lambda lib: lib.array([1, 2, None]).dtype,
    "getitem": lambda lib: lib.array([1, 2, None])[2],
    "slice": lambda lib: lib.array([1, 2, None])[1:],
    "tolist": lambda lib: lib.array([1, 2, None]).tolist(),
    "to_numpy": lambda lib: lib.array([1.5, 2.5]).to_numpy(),
    "series-of": lambda lib: lib.Series(lib.array([1, 2, None])),
    "isna": lambda lib: lib.array([1, None]).isna(),
    "add": lambda lib: lib.array([1, 2, None]) + 1,
    "eq": lambda lib: lib.array([1, 2, None]) == 1,
    "isinstance": lambda lib: isinstance(lib.array([1, 2]), lib.arrays.IntegerArray),
    "ser-array": lambda lib: lib.Series([1, 2]).array,
    "ser-array-str": lambda lib: lib.Series(["a", "b"]).array,
    "long": lambda lib: lib.array(list(range(100))),
    "int-nan": lambda lib: lib.array([1, float("nan")]),
    "bool-int": lambda lib: lib.array([True, 1]),
    "np-bool": lambda lib: lib.array(np.array([True, False])),
    "np-dt": lambda lib: lib.array(np.array(["2026-01-01"], dtype="datetime64[s]")),
    "np-obj": lambda lib: lib.array(np.array([1, "a"], dtype=object)),
    "float64-dtype": lambda lib: lib.array([1, 2], dtype="Float64"),
    "intervals": lambda lib: lib.array([lib.Interval(0, 1), lib.Interval(1, 2)]),
    "intervals-gap": lambda lib: lib.array([lib.Interval(0, 1), None]),
    "index": lambda lib: lib.array(lib.Index([1, 2])),
    "tuple": lambda lib: lib.array((1, 2)),
    "str-scalar": lambda lib: lib.array("abc"),
    "td-mixed": lambda lib: lib.array([lib.Timedelta("1D"), lib.Timedelta("2h")]),
    "tz": lambda lib: lib.array([lib.Timestamp("2026-01-01", tz="UTC")]),
    "dates-only": lambda lib: lib.array([lib.Timestamp("2026-01-01 10:00")]),
    "ne": lambda lib: lib.array([1.5, None]) != 1.5,
    "neg": lambda lib: -lib.array([1, None]),
    "fillna": lambda lib: lib.array([1, None]).fillna(0),
    "astype": lambda lib: lib.array([1, None]).astype("Float64"),
    "copy": lambda lib: lib.array([1, None]).copy(),
    "dropna": lambda lib: lib.array([1, None]).dropna(),
    "np-asarray": lambda lib: np.asarray(lib.array([1.5, 2.5])),
    "to_numpy-int": lambda lib: lib.array([1, None]).to_numpy(),
    "value_counts": lambda lib: lib.array([1, 1, None]).value_counts(),
    "unique": lambda lib: lib.array([1, 1, None]).unique(),
    "dtype-object": lambda lib: lib.array([1, 2], dtype=object),
    "dtype-bool": lambda lib: lib.array([True, False], dtype="bool"),
    "dtype-dt": lambda lib: lib.array(["2026-01-01"], dtype="datetime64[ns]"),
    "dtype-period": lambda lib: lib.array(["2026-01"], dtype="period[M]"),
    "sum-int": lambda lib: int(lib.array([1, 2, None]).sum()),
    "min-int": lambda lib: int(lib.array([3, 1, None]).min()),
    "mean-int": lambda lib: float(lib.array([3, 1, None]).mean()),
    "first": lambda lib: int(lib.array([1, 2, None])[0]),
}


def mistake(error: Exception) -> str:
    kind = next(kind.__name__ for kind in type(error).__mro__ if kind.__module__ == "builtins")
    return f"{kind}: {error}"


def outcome(build: Callable[[], Any]) -> str:
    try:
        found = build()
    except Exception as error:
        return mistake(error)
    return f"{type(found).__name__} {found!r}"


@pytest.mark.parametrize("case", CASES.values(), ids=CASES.keys())
def test_answers_as_pandas(case: Callable[[ModuleType], Any]) -> None:
    assert outcome(lambda: case(fp)) == outcome(lambda: case(pd))


def test_the_array_classes_are_pandas_classes_but_two() -> None:
    theirs = {name for name in dir(pd.arrays) if not name.startswith("_")}
    ours = {name for name in dir(fp.arrays) if not name.startswith("_")}
    assert theirs - ours == {"SparseArray", "StringArray"}
    assert ours <= theirs


def test_text_arrays_print_as_a_text_column_prints() -> None:
    assert repr(fp.array(["a", None])) == "<ArrowStringArray>\n['a', nan]\nLength: 2, dtype: str"
