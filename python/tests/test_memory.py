"""How much memory a frame and a column are using, counted the way pandas counts it.

Three members, `Series.nbytes` and `memory_usage` on both classes, answer the number pandas
answers. That number is a count of the numpy or Arrow arrays pandas would hold the data in rather
than of the buffers firepanda holds it in, so a column of three int64 values is 24 bytes, a
column of text is eight bytes of offset a row plus its UTF-8 bytes, and a `RangeIndex` is the
size of the Python range and its three numbers. A program that adds the numbers up and compares
them with what pandas said gets the same total here.
"""

from __future__ import annotations

from types import ModuleType
from typing import Any

import pytest

pd = pytest.importorskip("pandas")

DATA: dict[str, list[Any]] = {"k": ["p", "qq", "rrr"], "v": [1, 2, 3], "f": [1.5, 2.5, 3.5]}
"""Text, integers and floats, so the three array shapes are all present."""

COLUMNS: list[tuple[list[Any], Any]] = [
    ([1, 2, 3], None),
    ([1.5, None], None),
    (["a", None, "bc"], None),
    (["x", "yy"], None),
    (["é", "ab"], None),
    ([True, False], None),
    ([1, None], "Int64"),
    ([1.0], "Float32"),
    ([True, None], "boolean"),
    (["x", "y", "x"], "category"),
    ([1, 2, 1], "category"),
    (list(range(200)), "category"),
    (["a", 1, None], object),
    ([1, 2], "int8"),
    ([1], "uint16"),
    ([], "float64"),
    ([], "str"),
]
"""A column of each kind whose count follows a different rule."""


@pytest.mark.parametrize(("values", "dtype"), COLUMNS)
def test_a_column_weighs_what_pandas_says(
    firepanda: ModuleType, values: list[Any], dtype: Any
) -> None:
    """Width times length, a mask byte for a masked type, and text and categories by their parts."""
    ours = firepanda.Series(values, dtype=dtype)
    theirs = pd.Series(values, dtype=dtype)
    assert ours.nbytes == theirs.nbytes
    assert ours.memory_usage(index=False) == theirs.memory_usage(index=False)
    assert ours.memory_usage(index=False, deep=True) == theirs.memory_usage(index=False, deep=True)


def test_a_column_counts_its_index_by_default_and_nbytes_does_not(firepanda: ModuleType) -> None:
    """`memory_usage()` includes the labels and `nbytes` never does, which is pandas' rule."""
    column = firepanda.DataFrame(DATA)["v"]
    assert column.memory_usage() == column.nbytes + column.index.nbytes
    assert column.memory_usage() == pd.DataFrame(DATA)["v"].memory_usage()


@pytest.mark.parametrize(
    "make",
    [
        lambda m: m.RangeIndex(5),
        lambda m: m.RangeIndex(0, 2**40, 2**39),
        lambda m: m.RangeIndex(1, 9, 2),
        lambda m: m.Index([1.5, 2.0]),
        lambda m: m.Index(["a", "b"]),
        lambda m: m.DataFrame(DATA).index,
    ],
)
def test_an_index_weighs_what_pandas_says(firepanda: ModuleType, make: Any) -> None:
    """A range is the Python range and its three numbers, and stored labels are a column."""
    ours, theirs = make(firepanda), make(pd)
    assert ours.nbytes == theirs.nbytes
    assert ours.memory_usage() == theirs.memory_usage()


@pytest.mark.parametrize(
    "make",
    [
        lambda m: m.DataFrame(DATA),
        lambda m: m.DataFrame(DATA).set_index("k"),
        lambda m: m.DataFrame({"a": [1, 2], "c": ["x", "yy"]}, index=["r", "s"]),
        lambda m: m.DataFrame({}),
    ],
)
@pytest.mark.parametrize("flags", [{}, {"deep": True}, {"index": False}])
def test_a_frame_weighs_what_pandas_says(firepanda: ModuleType, make: Any, flags: Any) -> None:
    """One row a column, the index first under the name `Index`, each the column's own count."""
    ours = make(firepanda).memory_usage(**flags)
    theirs = make(pd).memory_usage(**flags)
    assert list(ours.index) == list(theirs.index)
    assert ours.tolist() == theirs.tolist()
    assert ours.name is None
    assert ours.index.name is None


def test_deep_measures_the_objects_an_object_column_holds(firepanda: ModuleType) -> None:
    """A pointer a row when shallow, and the size of each object on top when deep."""
    values = ["a", 1, None]
    ours = firepanda.Series(values, dtype=object)
    theirs = pd.Series(values, dtype=object)
    assert ours.memory_usage(index=False) == 24
    assert ours.memory_usage(index=False, deep=True) == theirs.memory_usage(index=False, deep=True)


def test_nbytes_cannot_be_set(firepanda: ModuleType) -> None:
    """A property with no setter, so a typo is caught rather than kept."""
    with pytest.raises(AttributeError):
        firepanda.DataFrame(DATA)["v"].nbytes = 4  # type: ignore[misc]


@pytest.mark.parametrize("flag", [1, 0, None, "", "no", 2.5])
def test_a_flag_that_is_not_a_boolean_is_read_for_its_truth(
    firepanda: ModuleType, flag: Any
) -> None:
    """pandas reads these two flags for their truth rather than checking them, so this does too."""
    frame = firepanda.DataFrame(DATA)
    theirs = pd.DataFrame(DATA)
    assert frame.memory_usage(index=flag).tolist() == theirs.memory_usage(index=flag).tolist()
    assert frame["v"].memory_usage(index=flag) == theirs["v"].memory_usage(index=flag)
    assert frame.memory_usage(deep=flag).tolist() == theirs.memory_usage(deep=flag).tolist()
