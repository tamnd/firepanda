"""Assigning to an axis or a name, `df.columns = [...]` and `s.name = "w"`, against pandas."""

from __future__ import annotations

import copy
from types import ModuleType
from typing import Any

import pandas as pd
import pytest


def _series_index(module: Any, s: Any, f: Any) -> Any:
    s.index = ["x", "y", "z"]
    return s


def _series_index_named(module: Any, s: Any, f: Any) -> Any:
    s.index = module.Index([10, 20, 30], name="k")
    return s


def _series_name(module: Any, s: Any, f: Any) -> Any:
    s.name = "w"
    return s


def _series_name_number(module: Any, s: Any, f: Any) -> Any:
    s.name = 7
    return s.name


def _series_index_name(module: Any, s: Any, f: Any) -> Any:
    s.index.name = "k"
    return s


def _frame_index(module: Any, s: Any, f: Any) -> Any:
    f.index = ["x", "y"]
    return f


def _frame_columns(module: Any, s: Any, f: Any) -> Any:
    f.columns = ["c", "d"]
    return f


def _frame_columns_index(module: Any, s: Any, f: Any) -> Any:
    f.columns = module.Index(["c", "d"])
    return f.dtypes


def _frame_columns_multi(module: Any, s: Any, f: Any) -> Any:
    f.columns = module.MultiIndex.from_tuples([("a", 1), ("a", 2)])
    return f


def _frame_index_name(module: Any, s: Any, f: Any) -> Any:
    f.index.name = "r"
    return f


def _frame_index_names(module: Any, s: Any, f: Any) -> Any:
    f.index.names = ["r"]
    return f


def _multi_names(module: Any, s: Any, f: Any) -> Any:
    g = f.set_index(["a", "b"])
    g.index.names = ["x", "y"]
    return g


def _taken_before(module: Any, s: Any, f: Any) -> Any:
    column = f["a"]
    f.columns = ["c", "d"]
    return column


def _standalone(module: Any, s: Any, f: Any) -> Any:
    index = module.Index([1, 2])
    index.name = "k"
    return index


CASES = [
    _series_index,
    _series_index_named,
    _series_name,
    _series_name_number,
    _series_index_name,
    _frame_index,
    _frame_columns,
    _frame_columns_index,
    _frame_columns_multi,
    _frame_index_name,
    _frame_index_names,
    _multi_names,
    _taken_before,
    _standalone,
]


def _run(module: Any, case: Any) -> str:
    s = module.Series([1, 2, 3], name="v")
    f = module.DataFrame({"a": [1, 2], "b": [3, 4]})
    return repr(case(module, s, f))


@pytest.mark.parametrize("case", CASES, ids=[case.__name__.strip("_") for case in CASES])
def test_an_assignment_answers_as_pandas(firepanda: ModuleType, case: Any) -> None:
    """Each assignment leaves the object printing as pandas prints it."""
    assert _run(firepanda, case) == _run(pd, case)


@pytest.mark.parametrize(
    "assign",
    [
        lambda s, f: setattr(s, "index", [1, 2]),
        lambda s, f: setattr(f, "columns", ["c"]),
        lambda s, f: setattr(f, "index", [1, 2, 3]),
    ],
    ids=["series-index", "frame-columns", "frame-index"],
)
def test_a_length_that_does_not_match_is_refused(firepanda: ModuleType, assign: Any) -> None:
    """pandas refuses labels of another length with a `ValueError` that says both lengths."""
    s = firepanda.Series([1, 2, 3])
    f = firepanda.DataFrame({"a": [1, 2], "b": [3, 4]})
    with pytest.raises(ValueError, match="Length mismatch"):
        assign(s, f)


def test_a_name_that_cannot_be_hashed_is_refused(firepanda: ModuleType) -> None:
    """A list is not a name, for a column or for an index."""
    with pytest.raises(TypeError, match=r"Series\.name must be a hashable type"):
        firepanda.Series([1]).name = ["a"]
    with pytest.raises(TypeError, match=r"Index\.name must be a hashable type"):
        firepanda.Index([1]).name = ["a"]


def test_a_copy_of_an_index_is_renamed_alone(firepanda: ModuleType) -> None:
    """A copy of an index read off a frame does not reach back into the frame."""
    f = firepanda.DataFrame({"a": [1, 2], "b": [3, 4]}).set_index(["a", "b"])
    taken = copy.copy(f.index)
    taken.names = ["x", "y"]
    assert list(f.index.names) == ["a", "b"]
