"""An index of objects against pandas, compared by what each answers or raises."""

from __future__ import annotations

from collections.abc import Callable
from types import ModuleType
from typing import Any

import pytest

import firepanda as fp

pd = pytest.importorskip("pandas")


def mixed(lib: ModuleType) -> Any:
    return lib.Index(["a", 1, None, 2.5])


def numbers(lib: ModuleType) -> Any:
    return lib.Index([3, 1, 2.5], dtype=object)


CASES: dict[str, Callable[[ModuleType], Any]] = {
    "repr": mixed,
    "tolist": lambda lib: mixed(lib).tolist(),
    "one": lambda lib: mixed(lib)[1],
    "slice": lambda lib: mixed(lib)[1:],
    "get-loc": lambda lib: mixed(lib).get_loc(1),
    "get-loc-text": lambda lib: mixed(lib).get_loc("a"),
    "get-loc-twin": lambda lib: lib.Index([1, "x"]).get_loc(1.0),
    "get-loc-missing": lambda lib: mixed(lib).get_loc("zz"),
    "contains": lambda lib: 1 in mixed(lib),
    "contains-not": lambda lib: "b" in mixed(lib),
    "eq": lambda lib: [bool(each) for each in mixed(lib) == 1],
    "ne": lambda lib: [bool(each) for each in lib.Index([1, "x"]) != "x"],
    "isin": lambda lib: [bool(each) for each in mixed(lib).isin([1, "a"])],
    "len": lambda lib: len(mixed(lib)),
    "named": lambda lib: lib.Index(["a", 1], name="k"),
    "inferred": lambda lib: mixed(lib).inferred_type,
    "inferred-text": lambda lib: lib.Index(["a", None], dtype=object).inferred_type,
    "flags-gap": lambda lib: lib.Index([True, None]),
    "dtype-object": lambda lib: lib.Index(["a", "b", None], dtype=object),
    "dtype-object-numbers": numbers,
    "unique": lambda lib: lib.Index([1, "b", 1]).unique(),
    "value-counts": lambda lib: lib.Index([1, "b", 1]).value_counts(),
    "take": lambda lib: mixed(lib).take([0, 2]),
    "append": lambda lib: mixed(lib).append(lib.Index(["z"])),
    "rename": lambda lib: mixed(lib).rename("n"),
    "to-series": lambda lib: mixed(lib).to_series(),
    "to-frame": lambda lib: mixed(lib).to_frame(),
    "fillna": lambda lib: mixed(lib).fillna("q"),
    "dropna": lambda lib: mixed(lib).dropna(),
    "astype-str": lambda lib: lib.Index(["a", 1]).astype(str),
    "map": lambda lib: lib.Index(["a", 1]).map(str),
    "nunique": lambda lib: lib.Index(["a", 1]).nunique(),
    "hasnans": lambda lib: mixed(lib).hasnans,
    "equals": lambda lib: lib.Index(["a", 1]).equals(lib.Index(["a", 1])),
    "iterate": lambda lib: list(lib.Index(["a", 1])),
    "sort": lambda lib: lib.Index([3, 1, 2.5, None], dtype=object).sort_values(),
    "sort-down": lambda lib: numbers(lib).sort_values(ascending=False),
    "sort-mix": lambda lib: lib.Index([3, "b"]).sort_values(),
    "series": lambda lib: lib.Series([10, 20, 30, 40], index=mixed(lib)),
    "series-loc": lambda lib: lib.Series([10, 20, 30, 40], index=mixed(lib)).loc[1],
    "series-item": lambda lib: lib.Series([10, 20, 30, 40], index=mixed(lib))["a"],
    "series-index": lambda lib: lib.Series([1, 2, 3], index=numbers(lib)).index,
    "series-sort-index": lambda lib: lib.Series([1, 2, 3], index=numbers(lib)).sort_index(),
    "series-sort-mix": lambda lib: lib.Series([1, 2, 3], index=lib.Index([3, 1, "a"])).sort_index(),
    "frame": lambda lib: lib.DataFrame({"x": [1, 2, 3, 4]}, index=mixed(lib)),
    "frame-sort-index": lambda lib: lib.DataFrame({"x": [1, 2, 3]}, index=numbers(lib)).sort_index(
        ascending=False
    ),
    "frame-loc": lambda lib: lib.DataFrame({"x": [1, 2]}, index=lib.Index(["a", 1])).loc[[1]],
    "reset-index": lambda lib: lib.Series([1, 2], index=mixed(lib)[:2]).reset_index(),
    "reindex": lambda lib: lib.Series([1, 2], index=lib.Index(["a", 1])).reindex([1, "a", 3]),
    "concat": lambda lib: lib.concat([lib.Series([1], index=["a"]), lib.Series([2], index=[1])]),
    "series-object-numbers": lambda lib: lib.Series([3, 1, 2.5], dtype=object),
    "series-object-sorted": lambda lib: lib.Series([3, 1, 2.5], dtype=object).sort_values(),
}


def mistake(error: Exception) -> str:
    kind = next(kind.__name__ for kind in type(error).__mro__ if kind.__module__ == "builtins")
    return f"{kind}: {error}"


def outcome(build: Callable[[], Any]) -> str:
    try:
        found = build()
    except Exception as error:
        return mistake(error)
    if type(found).__module__.startswith("numpy"):
        found = found.item()
    return repr(found)


@pytest.mark.parametrize("case", CASES.values(), ids=CASES.keys())
def test_answers_as_pandas(case: Callable[[ModuleType], Any]) -> None:
    assert outcome(lambda: case(fp)) == outcome(lambda: case(pd))


def test_the_class_is_called_index() -> None:
    assert type(fp.Index(["a", 1])).__name__ == "Index"
    assert isinstance(fp.Index(["a", 1]), fp.Index)
    assert fp.Index(["a", 1]).dtype == "object"
