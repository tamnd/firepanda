"""`DataFrame.unstack`, `Series.from_arrow` and `to_markdown` against pandas, compared by repr."""

from __future__ import annotations

from collections.abc import Callable
from types import ModuleType
from typing import Any

import pytest

import firepanda as fp

pd = pytest.importorskip("pandas")
pa = pytest.importorskip("pyarrow")


def stacked(lib: ModuleType) -> Any:
    labels = lib.MultiIndex.from_tuples([("a", "x"), ("a", "y"), ("b", "x")], names=["r", "s"])
    return lib.DataFrame({"v": [1, 2, 3], "w": [4.0, 5.0, 6.0]}, index=labels)


def flat(lib: ModuleType) -> Any:
    return lib.DataFrame({"v": [1, 2], "w": [3, 4]}, index=lib.Index(["p", "q"], name="k"))


def cells(frame: Any) -> str:
    """The frame's labels and values, leaving out the name of the column axis firepanda lacks."""
    return repr((list(frame.columns), frame.index.tolist(), frame.to_dict()))


CASES: dict[str, Callable[[ModuleType], Any]] = {
    "last-level": lambda lib: cells(stacked(lib).unstack()),
    "first-level": lambda lib: cells(stacked(lib).unstack(0)),
    "by-name": lambda lib: cells(stacked(lib).unstack("r")),
    "fill-value": lambda lib: cells(stacked(lib).unstack(fill_value=0)),
    "row-names": lambda lib: stacked(lib).unstack().index.name,
    "flat": lambda lib: flat(lib).unstack(),
    "flat-mixed": lambda lib: lib.DataFrame({"v": [1, 2], "w": [3.5, 4]}).unstack(),
    "flat-names": lambda lib: list(flat(lib).unstack().index.names),
    "duplicates": lambda lib: lib.DataFrame(
        {"v": [1, 2]}, index=lib.MultiIndex.from_tuples([("a", "x"), ("a", "x")])
    ).unstack(),
    "arrow-array": lambda lib: lib.Series.from_arrow(pa.array([1, None, 3])),
    "arrow-chunked": lambda lib: lib.Series.from_arrow(pa.chunked_array([["a"], ["b", None]])),
    "arrow-floats": lambda lib: lib.Series.from_arrow(pa.array([1.5, None])),
    "arrow-list": lambda lib: lib.Series.from_arrow([1, 2]),
    "markdown": lambda lib: flat(lib).to_markdown(),
    "markdown-series": lambda lib: flat(lib)["v"].to_markdown(),
    "markdown-no-index": lambda lib: flat(lib).to_markdown(index=False),
    "markdown-showindex": lambda lib: flat(lib).to_markdown(showindex=False),
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
def test_answers_as_pandas(case: Callable[[ModuleType], Any]) -> None:
    assert outcome(lambda: case(fp)) == outcome(lambda: case(pd))
