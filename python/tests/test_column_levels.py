"""Columns named by tuples, pandas' MultiIndex columns, compared with pandas.

Each case runs in both libraries and the answers are compared by their repr,
or for a mistake by its class name.
"""

from __future__ import annotations

from collections.abc import Callable
from types import ModuleType
from typing import Any

import pytest

import firepanda as fp

pd = pytest.importorskip("pandas")


def grouped(lib: ModuleType) -> Any:
    return lib.DataFrame({("a", "x"): [1, 2], ("a", "y"): [3.5, 4], ("b", "x"): ["p", "q"]})


def deep(lib: ModuleType) -> Any:
    return lib.DataFrame({("a", "x", 1): [1], ("a", "x", 2): [2], ("a", "y", 1): [3]})


def flat_beside(lib: ModuleType) -> Any:
    return lib.DataFrame({("a", ""): [1], ("b", "x"): [2], ("b", "y"): [3]})


def numbered(lib: ModuleType) -> Any:
    return lib.DataFrame({(1, "x"): [1], (1, "y"): [2], (2, "x"): [3]})


CASES: dict[str, Callable[[ModuleType], Any]] = {
    "repr": grouped,
    "repr-wide-label": lambda lib: lib.DataFrame(
        {("long", "x"): [1], ("long", "y"): [2], ("z", "w"): [3]}
    ),
    "repr-flat-beside": flat_beside,
    "repr-three-levels": deep,
    "repr-numbered": numbered,
    "repr-named-rows": lambda lib: grouped(lib).rename_axis("k"),
    "head": lambda lib: grouped(lib).head(1),
    "columns": lambda lib: grouped(lib).columns,
    "columns-tolist": lambda lib: grouped(lib).columns.tolist(),
    "nlevels": lambda lib: grouped(lib).columns.nlevels,
    "tuple-key": lambda lib: grouped(lib)[("a", "x")],
    "first-level": lambda lib: grouped(lib)["a"],
    "first-level-series": lambda lib: flat_beside(lib)["a"],
    "first-level-three": lambda lib: deep(lib)["a"],
    "first-level-number": lambda lib: numbered(lib)[1],
    "list-of-tuples": lambda lib: grouped(lib)[[("a", "x"), ("b", "x")]],
    "iterate": lambda lib: list(grouped(lib)),
    "contains": lambda lib: ("a", "x") in grouped(lib),
    "sum": lambda lib: deep(lib).sum(),
    "to-dict": lambda lib: grouped(lib).to_dict(),
    "concat": lambda lib: lib.concat([grouped(lib), grouped(lib)]),
    "from-multiindex": lambda lib: lib.DataFrame(
        [[1, 2], [3, 4]],
        columns=lib.MultiIndex.from_tuples([("a", "x"), ("b", "y")]),
        index=["r", "s"],
    ),
    "to-csv": lambda lib: grouped(lib).to_csv(),
    "to-csv-no-index": lambda lib: grouped(lib).to_csv(index=False),
    "to-csv-named-rows": lambda lib: grouped(lib).rename_axis("k").to_csv(),
    "to-csv-index-label": lambda lib: grouped(lib).to_csv(index_label="r"),
}


@pytest.mark.parametrize("case", CASES.values(), ids=CASES.keys())
def test_column_levels_answer_as_pandas(case: Callable[[ModuleType], Any]) -> None:
    assert repr(case(fp)) == repr(case(pd))


MISSING: dict[str, Callable[[ModuleType], Any]] = {
    "first-level": lambda lib: grouped(lib)["z"],
    "tuple": lambda lib: grouped(lib)[("a", "z")],
}


@pytest.mark.parametrize("case", MISSING.values(), ids=MISSING.keys())
def test_a_missing_label_is_a_key_error(case: Callable[[ModuleType], Any]) -> None:
    with pytest.raises(KeyError):
        case(fp)
    with pytest.raises(KeyError):
        case(pd)
