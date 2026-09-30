"""`to_string(sparsify=)` and `display.multi_sparse`, which blank repeated level labels.

pandas leaves a label on a level blank when the row above, or the column to
the left, has the same labels on it and on every level before it. With
`sparsify=False`, or the option off, every label prints whole. Each test runs
the same code on both libraries and compares what they print.
"""

from __future__ import annotations

from typing import Any

import pandas as pd
import pytest


def levels(lib: Any) -> Any:
    return lib.MultiIndex.from_tuples([("a", 1), ("a", 2), ("b", 1)], names=["k", "n"])


def rows(lib: Any) -> Any:
    return lib.DataFrame({"v": [1, 2, 3]}, index=levels(lib))


def columns(lib: Any) -> Any:
    return lib.DataFrame([[1, 2, 3]], columns=levels(lib))


def unsparse(lib: Any, make: Any) -> str:
    with lib.option_context("display.multi_sparse", False):
        return make()


BUILDS = {
    "rows off": lambda lib: rows(lib).to_string(sparsify=False),
    "rows on": lambda lib: rows(lib).to_string(sparsify=True),
    "columns off": lambda lib: columns(lib).to_string(sparsify=False),
    "columns default": lambda lib: columns(lib).to_string(),
    "option rows": lambda lib: unsparse(lib, lambda: rows(lib).to_string()),
    "option columns": lambda lib: unsparse(lib, lambda: columns(lib).to_string()),
    "option repr": lambda lib: unsparse(lib, lambda: repr(rows(lib))),
    "option series": lambda lib: unsparse(lib, lambda: repr(rows(lib)["v"])),
}


@pytest.mark.parametrize("make", BUILDS.values(), ids=BUILDS.keys())
def test_sparsify_is_pandas(firepanda: Any, make: Any) -> None:
    assert make(firepanda) == make(pd)
