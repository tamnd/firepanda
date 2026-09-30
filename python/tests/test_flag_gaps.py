"""Flags beside a gap are held as objects, as pandas holds them, compared by repr."""

from __future__ import annotations

import math
from collections.abc import Callable
from types import ModuleType
from typing import Any

import pytest

import firepanda as fp

pd = pytest.importorskip("pandas")

CASES: dict[str, Callable[[ModuleType], Any]] = {
    "series": lambda lib: lib.Series([True, False, None]),
    "nan": lambda lib: lib.Series([True, math.nan]),
    "tuple": lambda lib: lib.Series((False, None)),
    "frame": lambda lib: lib.DataFrame({"b": [True, None], "x": [1, 2]}),
    "frame-dtypes": lambda lib: lib.DataFrame({"b": [True, None], "x": [1, 2]}).dtypes,
    "no-gap": lambda lib: lib.Series([True, False]),
    "masked": lambda lib: lib.Series([True, None], dtype="boolean"),
    "to-masked": lambda lib: lib.Series([True, False, None]).astype("boolean"),
    "convert": lambda lib: lib.Series([True, False, None]).convert_dtypes(),
    "convert-off": lambda lib: lib.Series([True, None]).convert_dtypes(convert_boolean=False),
    "isna": lambda lib: lib.Series([True, None]).isna(),
    "fill": lambda lib: lib.Series([True, None]).fillna(False),
    "fill-number": lambda lib: lib.Series([True, None]).fillna(1),
    "tolist": lambda lib: lib.Series([True, None]).tolist(),
    "where": lambda lib: lib.Series([1, 2, 3]).where([True, None, True], 0),
}


def outcome(build: Callable[[], Any]) -> str:
    try:
        return repr(build())
    except Exception as error:
        return f"{type(error).__name__}: {error}"


@pytest.mark.parametrize("case", CASES.values(), ids=CASES.keys())
def test_answers_as_pandas(case: Callable[[ModuleType], Any]) -> None:
    assert outcome(lambda: case(fp)) == outcome(lambda: case(pd))


def test_a_list_of_flags_with_a_gap_is_an_object_column() -> None:
    assert str(fp.Series([True, None]).dtype) == "object"
    assert str(fp.DataFrame({"b": [False, None]})["b"].dtype) == "object"
