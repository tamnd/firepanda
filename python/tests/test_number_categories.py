"""Categories of numbers and flags, compared with pandas.

pandas keeps a category column's categories in the values' own type, so a
column of whole numbers has `int64` categories. firepanda writes each value as
an object cell and reads the categories back as values, which is document 97.
Each case runs in both libraries and the answers are compared by their repr, or
for a mistake by its class name.
"""

from __future__ import annotations

from collections.abc import Callable
from types import ModuleType
from typing import Any

import pytest

import firepanda as fp

pd = pytest.importorskip("pandas")


def ints(lib: ModuleType) -> Any:
    return lib.Series([3, 1, 3, 10]).astype("category")


def keyed(lib: ModuleType) -> Any:
    return lib.DataFrame({"k": lib.Series([3, 1, 3]).astype("category"), "v": [1, 2, 3]})


CASES: dict[str, Callable[[ModuleType], Any]] = {
    "ints": ints,
    "floats-with-gap": lambda lib: lib.Series([1.0, 1.25, None]).astype("category"),
    "ints-with-gap": lambda lib: lib.Series([3, 1, None]).astype("category"),
    "bools": lambda lib: lib.Series([True, False]).astype("category"),
    "many": lambda lib: lib.Series(list(range(12))).astype("category"),
    "frame": lambda lib: lib.DataFrame(
        {"a": lib.Series([1.0, 1.25, None]).astype("category"), "b": [3, 10, 3]}
    ).astype({"b": "category"}),
    "constructor": lambda lib: lib.Series([3, 1, 3], dtype="category"),
    "constructor-dtype": lambda lib: lib.Series([1.5, 2.5], dtype="category").dtype,
    "categorical": lambda lib: lib.Categorical([3, 1, 3]),
    "decided": lambda lib: lib.Series([3, 1, 3]).astype(lib.CategoricalDtype([1, 3, 5])),
    "categories": lambda lib: ints(lib).cat.categories,
    "codes": lambda lib: ints(lib).cat.codes,
    "tolist": lambda lib: ints(lib).tolist(),
    "add": lambda lib: ints(lib).cat.add_categories([7]),
    "rename": lambda lib: ints(lib).cat.rename_categories({1: 100}),
    "remove": lambda lib: ints(lib).cat.remove_categories(3),
    "reorder": lambda lib: ints(lib).cat.reorder_categories([10, 3, 1], ordered=True),
    "set": lambda lib: ints(lib).cat.set_categories([1, 2]),
    "equal": lambda lib: ints(lib) == 3,
    "not-equal": lambda lib: ints(lib) != 3,
    "ordered-greater": lambda lib: ints(lib).cat.as_ordered() > 1,
    "unordered-greater": lambda lib: ints(lib) > 1,
    "isin": lambda lib: ints(lib).isin([3]),
    "unique": lambda lib: ints(lib).unique(),
    "values": lambda lib: ints(lib).values,
    "to-numpy": lambda lib: ints(lib).to_numpy(),
    "astype-int": lambda lib: ints(lib).astype("int64"),
    "astype-str": lambda lib: ints(lib).astype(str),
    "value-counts": lambda lib: ints(lib).value_counts(),
    "group-observed": lambda lib: keyed(lib).groupby("k", observed=True).sum(),
    "group-unobserved": lambda lib: keyed(lib).groupby("k", observed=False)["v"].sum(),
    "sort": lambda lib: ints(lib).sort_values(ascending=False),
    "drop-duplicates": lambda lib: ints(lib).drop_duplicates(),
    "text-drop-duplicates": lambda lib: (
        lib.Series(["b", "a", "b"]).astype("category").drop_duplicates()
    ),
    "concat": lambda lib: lib.concat([ints(lib), ints(lib)]),
    "fillna": lambda lib: lib.Series([3, 1, None]).astype("category").fillna(1),
    "text-dtype": lambda lib: lib.CategoricalDtype(["a", "b"]),
}


def outcome(build: Callable[[], Any]) -> str:
    try:
        return repr(build())
    except Exception as error:
        return type(error).__name__


@pytest.mark.parametrize("case", CASES.values(), ids=CASES.keys())
def test_number_categories_answer_as_pandas(case: Callable[[ModuleType], Any]) -> None:
    assert outcome(lambda: case(fp)) == outcome(lambda: case(pd))


def test_the_categories_are_values_of_their_own_type() -> None:
    categories = ints(fp).cat.categories
    assert categories.tolist() == [1, 3, 10]
    assert str(categories.dtype) == "int64"
