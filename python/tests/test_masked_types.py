"""pandas' masked types, `Int64`, `Float64`, `boolean` and the rest, against pandas.

firepanda holds a masked column as an object column whose cells carry the
type's name, and computes over the lower case column of the same width, which
is document 99. Each case runs in both libraries and the answers are compared
by their repr, or for a mistake by its class name. Scalars are compared by
value, since firepanda answers a Python number where pandas answers numpy's.
"""

from __future__ import annotations

from collections.abc import Callable
from types import ModuleType
from typing import Any

import pytest

import firepanda as fp

pd = pytest.importorskip("pandas")


def ints(lib: ModuleType) -> Any:
    return lib.Series([1, None, 3], dtype="Int64")


def flags(lib: ModuleType) -> Any:
    return lib.Series([True, None, False], dtype="boolean")


def floats(lib: ModuleType) -> Any:
    return lib.Series([1.5, None, 2.25], dtype="Float64")


CASES: dict[str, Callable[[ModuleType], Any]] = {
    "astype": lambda lib: lib.Series([1, None, 3]).astype("Int64"),
    "constructor": ints,
    "dtype-object": lambda lib: lib.Series([1, 2]).astype(lib.Int64Dtype()),
    "dtype": lambda lib: ints(lib).dtype,
    "dtype-name": lambda lib: str(ints(lib).dtype),
    "dtype-classes": lambda lib: [lib.Float64Dtype(), lib.BooleanDtype(), lib.UInt8Dtype()],
    "floats": floats,
    "flags": flags,
    "narrow": lambda lib: lib.Series([1, None], dtype="UInt8"),
    "na-in-list": lambda lib: lib.Series([1, lib.NA], dtype="Int64"),
    "nan-to-float": lambda lib: lib.Series([1.0, float("nan")]).astype("Float64"),
    "fraction-to-int": lambda lib: lib.Series([1.5]).astype("Int64"),
    "cell": lambda lib: ints(lib).iloc[1],
    "tolist": lambda lib: ints(lib).tolist(),
    "isna": lambda lib: ints(lib).isna(),
    "notna": lambda lib: ints(lib).notna(),
    "fillna": lambda lib: ints(lib).fillna(0),
    "dropna": lambda lib: ints(lib).dropna(),
    "add": lambda lib: ints(lib) + 1,
    "add-float": lambda lib: ints(lib) + 1.5,
    "add-series": lambda lib: ints(lib) + lib.Series([1, 2, 3]),
    "add-masked": lambda lib: ints(lib) + ints(lib),
    "add-named": lambda lib: ints(lib).add(1),
    "true-div": lambda lib: ints(lib) / 2,
    "floor-div": lambda lib: ints(lib) // 2,
    "power": lambda lib: ints(lib) ** 2,
    "float-arithmetic": lambda lib: floats(lib) * 2,
    "equal": lambda lib: ints(lib) == 1,
    "not-equal": lambda lib: ints(lib) != 1,
    "greater": lambda lib: ints(lib) > 1,
    "compare-series": lambda lib: ints(lib) < lib.Series([2, 2, 2]),
    "and": lambda lib: flags(lib) & True,
    "and-false": lambda lib: flags(lib) & False,
    "or": lambda lib: flags(lib) | True,
    "or-false": lambda lib: flags(lib) | False,
    "xor": lambda lib: flags(lib) ^ True,
    "and-masked": lambda lib: flags(lib) & flags(lib),
    "invert": lambda lib: ~flags(lib),
    "negate": lambda lib: -ints(lib),
    "abs": lambda lib: (-ints(lib)).abs(),
    "cumsum": lambda lib: ints(lib).cumsum(),
    "shift": lambda lib: ints(lib).shift(),
    "diff": lambda lib: lib.Series([1, 4, 9], dtype="Int64").diff(),
    "where": lambda lib: ints(lib).where(ints(lib) > 1),
    "sort": lambda lib: ints(lib).sort_values(),
    "value-counts": lambda lib: ints(lib).value_counts(),
    "describe": lambda lib: ints(lib).describe(),
    "to-numpy": lambda lib: ints(lib).to_numpy(),
    "astype-float": lambda lib: ints(lib).astype("float64"),
    "astype-int-with-gap": lambda lib: ints(lib).astype("int64"),
    "astype-int": lambda lib: lib.Series([1, 2], dtype="Int64").astype("int64"),
    "frame": lambda lib: lib.DataFrame({"a": [1, None], "b": [0.5, 1.0]}).astype("Int64"),
    "frame-dict": lambda lib: lib.DataFrame({"a": [1, None], "b": [0.5, 1.0]}).astype(
        {"a": "Int64"}
    ),
    "frame-dtypes": lambda lib: [str(t) for t in lib.DataFrame({"a": [1]}).astype("Int64").dtypes],
    "convert": lambda lib: lib.DataFrame(
        {"a": [1, 2], "b": [1.5, None], "c": [1.0, None], "d": [True, False]}
    ).convert_dtypes(),
    "convert-dtypes": lambda lib: [
        str(t)
        for t in lib.DataFrame(
            {"a": [1, 2], "b": [1.5, None], "c": [1.0, None], "d": [True, False]}
        )
        .convert_dtypes()
        .dtypes
    ],
    "convert-series": lambda lib: lib.Series([1.0, 2.0]).convert_dtypes(),
    "convert-no-integer": lambda lib: lib.Series([1.0, 2.0]).convert_dtypes(convert_integer=False),
}

SCALARS: dict[str, Callable[[ModuleType], Any]] = {
    "sum": lambda lib: ints(lib).sum(),
    "sum-keeping-gaps": lambda lib: ints(lib).sum(skipna=False),
    "mean": lambda lib: ints(lib).mean(),
    "max": lambda lib: ints(lib).max(),
    "std": lambda lib: ints(lib).std(),
    "any": lambda lib: lib.Series([None, False], dtype="boolean").any(),
    "all": lambda lib: flags(lib).all(),
    "flag-sum": lambda lib: flags(lib).sum(),
    "equal-names": lambda lib: lib.Int64Dtype() == "Int64",
}


def mistake(error: Exception) -> str:
    """The builtin class a mistake is, since each library raises its own subclass of it."""
    return next(kind.__name__ for kind in type(error).__mro__ if kind.__module__ == "builtins")


def outcome(build: Callable[[], Any]) -> str:
    try:
        return repr(build())
    except Exception as error:
        return mistake(error)


def value(build: Callable[[], Any]) -> Any:
    try:
        answer = build()
    except Exception as error:
        return mistake(error)
    if type(answer).__name__ == "NAType":
        return "NA"
    return answer.item() if hasattr(answer, "item") else answer


@pytest.mark.parametrize("case", CASES.values(), ids=CASES.keys())
def test_masked_types_answer_as_pandas(case: Callable[[ModuleType], Any]) -> None:
    assert outcome(lambda: case(fp)) == outcome(lambda: case(pd))


@pytest.mark.parametrize("case", SCALARS.values(), ids=SCALARS.keys())
def test_masked_reductions_answer_as_pandas(case: Callable[[ModuleType], Any]) -> None:
    assert value(lambda: case(fp)) == pytest.approx(value(lambda: case(pd)))


def test_the_type_survives_moving_rows() -> None:
    column = ints(fp)
    assert str(column.iloc[[2, 0]].dtype) == "Int64"
    assert str(column.sort_index(ascending=False).dtype) == "Int64"
    assert str(fp.concat([column, column]).dtype) == "Int64"
