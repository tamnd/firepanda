"""An index of a masked type against pandas, compared by what each answers or raises."""

from __future__ import annotations

from collections.abc import Callable
from types import ModuleType
from typing import Any

import pytest

import firepanda as fp

pd = pytest.importorskip("pandas")


def whole(lib: ModuleType) -> Any:
    return lib.Index([3, None, 1], dtype="Int64")


def keyed(lib: ModuleType) -> Any:
    return lib.DataFrame({"k": lib.Series([3, None, 1], dtype="Int64"), "v": [1, 2, 3]})


CASES: dict[str, Callable[[ModuleType], Any]] = {
    "repr": whole,
    "dtype": lambda lib: whole(lib).dtype,
    "from-series": lambda lib: lib.Index(lib.Series([1, None, 3], dtype="Int64")),
    "floats": lambda lib: lib.Index([1.5, None], dtype="Float64"),
    "flags": lambda lib: lib.Index([True, None], dtype="boolean"),
    "text": lambda lib: lib.Index(["a", None], dtype="string"),
    "text-get-loc": lambda lib: lib.Index(["a", None], dtype="string").get_loc("a"),
    "named": lambda lib: lib.Index([1, None], dtype="Int64", name="n"),
    "tolist": lambda lib: whole(lib).tolist(),
    "get-loc": lambda lib: whole(lib).get_loc(1),
    "get-loc-float": lambda lib: whole(lib).get_loc(1.0),
    "get-loc-missing": lambda lib: whole(lib).get_loc(2),
    "contains": lambda lib: 3 in whole(lib),
    "contains-not": lambda lib: 2 in whole(lib),
    "eq": lambda lib: whole(lib) == 3,
    "isin": lambda lib: whole(lib).isin([1, 3]),
    "map-floats": lambda lib: whole(lib).map(lambda x: x * 1.5),
    "map-same-width": lambda lib: lib.Index([1, 2], dtype="Int32").map(lambda x: x + 1),
    "map-float32": lambda lib: lib.Index([1.5, 2], dtype="Float32").map(lambda x: x + 1),
    "map-flags": lambda lib: lib.Index([True, False], dtype="boolean").map(lambda x: not x),
    "map-ignore": lambda lib: lib.Index([1, None], dtype="Int64").map(
        lambda x: x, na_action="ignore"
    ),
    "map-text": lambda lib: lib.Index([1, 2], dtype="Int64").map(lambda x: "a"),
    "map-dict": lambda lib: lib.Index([1, 2], dtype="Int64").map({1: 1.5, 2: 2}),
    "map-string": lambda lib: lib.Index(["a", None], dtype="string").map(lambda x: x),
    "isna": lambda lib: [bool(each) for each in whole(lib).isna()],
    "add": lambda lib: whole(lib) + 1,
    "max": lambda lib: whole(lib).max(),
    "astype-float": lambda lib: whole(lib).astype("float64"),
    "astype-masked": lambda lib: whole(lib).astype("Float64"),
    "sort": lambda lib: whole(lib).sort_values(),
    "unique": lambda lib: whole(lib).unique(),
    "take": lambda lib: whole(lib).take([0, 2]),
    "slice": lambda lib: whole(lib)[1:],
    "fillna": lambda lib: whole(lib).fillna(0),
    "dropna": lambda lib: whole(lib).dropna(),
    "to-series": lambda lib: whole(lib).to_series(),
    "to-frame": lambda lib: whole(lib).to_frame(),
    "value-counts": lambda lib: whole(lib).value_counts(),
    "series": lambda lib: lib.Series([1, 2, 3], index=whole(lib)),
    "series-loc": lambda lib: lib.Series([1, 2, 3], index=whole(lib)).loc[3],
    "series-loc-list": lambda lib: lib.Series([1, 2, 3], index=whole(lib)).loc[[1, 3]],
    "series-sort-index": lambda lib: lib.Series([1, 2, 3], index=whole(lib)).sort_index(),
    "series-reset-index": lambda lib: lib.Series([1, 2, 3], index=whole(lib)).reset_index(),
    "series-reindex": lambda lib: lib.Series([1, 2, 3], index=whole(lib)).reindex([1, 2]),
    "frame": lambda lib: lib.DataFrame({"v": [1, 2, 3]}, index=whole(lib)),
    "set-index": lambda lib: keyed(lib).set_index("k"),
    "set-index-back": lambda lib: keyed(lib).set_index("k").reset_index().dtypes,
    "concat": lambda lib: lib.concat(
        [
            lib.Series([1], index=lib.Index([1], dtype="Int64")),
            lib.Series([2], index=lib.Index([2], dtype="Int64")),
        ]
    ),
    "frame-masked-head": lambda lib: lib.DataFrame(
        {"index": lib.Series([3, None, 1], dtype="Int64"), "v": [1, 2, 3]}
    ),
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
    made = fp.Index([1, None], dtype="Int64")
    assert type(made).__name__ == "Index"
    assert isinstance(made, fp.Index)
    assert str(made.dtype) == "Int64"


def grouped(lib: ModuleType) -> Any:
    return lib.DataFrame(
        {
            "k": lib.Series([1, 1, None, 2], dtype="Int64"),
            "j": ["a", "b", "a", "a"],
            "v": [1, 2, 3, 4],
        }
    )


KEYED: dict[str, Callable[[ModuleType], Any]] = {
    "mean": lambda lib: grouped(lib).groupby("k").v.mean(),
    "size-gaps": lambda lib: grouped(lib).groupby("k", dropna=False).size(),
    "as-column": lambda lib: grouped(lib).groupby("k", as_index=False).v.sum(),
    "agg": lambda lib: grouped(lib).groupby("k").agg({"v": "max"}),
    "first": lambda lib: grouped(lib).groupby("k").first(),
    "nunique": lambda lib: grouped(lib).groupby("k").nunique(),
    "get-group": lambda lib: grouped(lib).groupby("k").get_group(1),
    "count": lambda lib: grouped(lib).groupby("k").count(),
    "two-keys": lambda lib: grouped(lib).groupby(["k", "j"]).v.sum(),
    "pivot": lambda lib: grouped(lib).pivot_table(index="k", values="v", aggfunc="sum"),
    "counts": lambda lib: lib.Series([5, 6, 5, None], dtype="Int64").value_counts(),
    "counts-gaps": lambda lib: lib.Series([5, 6, 5, None], dtype="Int64").value_counts(
        dropna=False
    ),
    "counts-share": lambda lib: lib.Series([1.5, 1.5, 2.0], dtype="Float64").value_counts(
        normalize=True
    ),
    "counts-flags": lambda lib: lib.Series([True, None, True], dtype="boolean").value_counts(),
}


@pytest.mark.parametrize("case", KEYED.values(), ids=KEYED.keys())
def test_a_masked_key_labels_the_answer_in_its_type(case: Callable[[ModuleType], Any]) -> None:
    assert outcome(lambda: case(fp)) == outcome(lambda: case(pd))
