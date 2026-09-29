"""`wide_to_long` against pandas, each case run in both libraries and compared by repr."""

from __future__ import annotations

from collections.abc import Callable
from types import ModuleType
from typing import Any

import pytest

import firepanda as fp

pd = pytest.importorskip("pandas")


def years(lib: ModuleType) -> Any:
    return lib.DataFrame(
        {
            "A1970": ["a", "b", "c"],
            "A1980": ["d", "e", "f"],
            "B1970": [2.5, 1.2, 0.7],
            "B1980": [3.2, 1.3, 0.1],
            "X": [0.1, 0.2, 0.3],
            "id": [0, 1, 2],
        }
    )


def families(lib: ModuleType) -> Any:
    return lib.DataFrame(
        {
            "famid": [1, 1, 2, 2],
            "birth": [1, 2, 1, 2],
            "ht1": [2.8, 2.9, 2.2, 2.0],
            "ht2": [3.4, 3.8, 2.9, 3.2],
        }
    )


CASES: dict[str, Callable[[ModuleType], Any]] = {
    "two-stubs": lambda lib: lib.wide_to_long(years(lib), ["A", "B"], i="id", j="year"),
    "two-ids": lambda lib: lib.wide_to_long(families(lib), "ht", i=["famid", "birth"], j="age"),
    "separator": lambda lib: lib.wide_to_long(
        lib.DataFrame({"id": [1, 2], "x_a": [1, 2], "x_b": [3, 4]}),
        "x",
        i="id",
        j="kind",
        sep="_",
        suffix=r"\w+",
    ),
    "text-suffix": lambda lib: lib.wide_to_long(
        lib.DataFrame({"id": [1], "vone": [1], "vtwo": [2]}), "v", i="id", j="n", suffix=r"\w+"
    ),
    "stub-is-a-column": lambda lib: lib.wide_to_long(
        lib.DataFrame({"id": [1], "A": [1], "A1": [2]}), "A", i="id", j="n"
    ),
    "ids-repeat": lambda lib: lib.wide_to_long(
        lib.DataFrame({"id": [1, 1], "A1": [1, 2]}), "A", i="id", j="n"
    ),
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
def test_wide_to_long_answers_as_pandas(case: Callable[[ModuleType], Any]) -> None:
    assert outcome(lambda: case(fp)) == outcome(lambda: case(pd))
