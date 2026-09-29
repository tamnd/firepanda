"""`DataFrame.compare`, `stack` over column levels and `droplevel` on the columns.

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


def mine(lib: ModuleType) -> Any:
    return lib.DataFrame(
        {"x": [1, 2, 3], "y": [1.5, None, 3.5], "s": ["p", "q", "r"]}, index=["r0", "r1", "r2"]
    )


def theirs(lib: ModuleType) -> Any:
    return mine(lib).assign(x=[1, 5, 3], s=["p", "q", "z"])


def levelled(lib: ModuleType) -> Any:
    return lib.DataFrame(
        {("a", "x"): [1, 2], ("a", "y"): [3, 4], ("b", "x"): [5, 6]}, index=["r", "s"]
    )


def unordered(lib: ModuleType) -> Any:
    return lib.DataFrame({("b", "y"): [1], ("a", "x"): [2], ("a", "y"): [3]})


CASES: dict[str, Callable[[ModuleType], Any]] = {
    "compare": lambda lib: mine(lib).compare(theirs(lib)),
    "compare-keep-shape": lambda lib: mine(lib).compare(theirs(lib), keep_shape=True),
    "compare-keep-equal": lambda lib: mine(lib).compare(theirs(lib), keep_equal=True),
    "compare-keep-both": lambda lib: mine(lib).compare(
        theirs(lib), keep_shape=True, keep_equal=True
    ),
    "compare-rows": lambda lib: mine(lib).compare(theirs(lib), align_axis=0),
    "compare-rows-index": lambda lib: mine(lib).compare(theirs(lib), align_axis=0).index,
    "compare-names": lambda lib: mine(lib).compare(theirs(lib), result_names=("L", "R")),
    "compare-same": lambda lib: mine(lib).compare(mine(lib)),
    "compare-columns": lambda lib: mine(lib).compare(theirs(lib)).columns,
    "multiindex-repr": lambda lib: lib.MultiIndex.from_tuples([(1, "a"), (10, "bb"), (100, "c")]),
    "stack": lambda lib: levelled(lib).stack(),
    "stack-first": lambda lib: levelled(lib).stack(0),
    "stack-order": lambda lib: unordered(lib).stack(),
    "stack-order-first": lambda lib: unordered(lib).stack(0),
    "stack-three": lambda lib: lib.DataFrame({("a", "x", 1): [1], ("a", "y", 2): [2]}).stack(),
    "stack-all": lambda lib: levelled(lib).stack([0, 1]),
    "stack-named-rows": lambda lib: levelled(lib).rename_axis("q").stack().index,
    "stack-mixed": lambda lib: lib.DataFrame({("a", "x"): ["p"], ("a", "y"): [1.5]}).stack(0),
    "droplevel": lambda lib: lib.DataFrame({("a", "x"): [1], ("b", "y"): [2]}).droplevel(0, axis=1),
    "swaplevel": lambda lib: levelled(lib).swaplevel(axis=1),
}

MISTAKES: dict[str, Callable[[ModuleType], Any]] = {
    "compare-labels": lambda lib: mine(lib).compare(theirs(lib).iloc[:2]),
    "compare-names-list": lambda lib: mine(lib).compare(theirs(lib), result_names=["L", "R"]),
    "stack-level": lambda lib: levelled(lib).stack(2),
}


@pytest.mark.parametrize("case", CASES.values(), ids=CASES.keys())
def test_answers_are_pandas_answers(case: Callable[[ModuleType], Any]) -> None:
    assert repr(case(fp)) == repr(case(pd))


@pytest.mark.parametrize("case", MISTAKES.values(), ids=MISTAKES.keys())
def test_mistakes_are_pandas_mistakes(case: Callable[[ModuleType], Any]) -> None:
    with pytest.raises(Exception) as ours:
        case(fp)
    with pytest.raises(Exception) as yours:
        case(pd)
    assert isinstance(ours.value, type(yours.value))
