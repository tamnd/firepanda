"""Keys and flags pandas refuses, refused with pandas' words, and the ones it reads.

`iloc` and `iat` take whole numbers and nothing that only looks like one, a
group by takes a Boolean `numeric_only` and nothing else, and a frame reads any
`numeric_only` by its truth, as pandas does.
"""

from __future__ import annotations

from types import ModuleType

import pytest

_KINDS = r"^Location based indexing can only have \[integer, integer slice"


def _frame(firepanda: ModuleType) -> object:
    return firepanda.DataFrame({"a": [1, 2, 3], "b": [3.0, 4.0, 5.0]})


@pytest.mark.parametrize("key", [1.5, 1.0, "a", None])
def test_iloc_takes_no_key_that_is_not_a_whole_number(firepanda: ModuleType, key: object) -> None:
    """A float is refused even when it is whole, which is pandas 3's rule."""
    frame = _frame(firepanda)
    words = r"^Cannot index by location index with a non-integer key$"
    with pytest.raises(TypeError, match=words):
        frame.iloc[key]
    with pytest.raises(TypeError, match=words):
        frame["a"].iloc[key]


def test_iloc_pairs_name_the_kinds_it_takes(firepanda: ModuleType) -> None:
    """A pair with a half that is no position lists what `iloc` reads."""
    frame = _frame(firepanda)
    for key in ((1.5, 0), (0, "a")):
        with pytest.raises(ValueError, match=_KINDS):
            frame.iloc[key]
    assert frame.iloc[1, 0] == 2


def test_iat_takes_whole_numbers(firepanda: ModuleType) -> None:
    """Both halves of the pair, and the one key of a column, must be integers."""
    frame = _frame(firepanda)
    words = r"^iAt based indexing can only have integer indexers$"
    with pytest.raises(ValueError, match=words):
        frame.iat[1.5, 0]
    with pytest.raises(ValueError, match=words):
        frame["a"].iat["a"]
    assert frame.iat[2, 1] == 5.0


def test_head_and_tail_take_a_count(firepanda: ModuleType) -> None:
    """The slice pandas makes with `n` names a RangeIndex whatever the labels are."""
    frame = _frame(firepanda)
    with pytest.raises(TypeError, match=r"RangeIndex with these indexers \[2\] of type str$"):
        frame.head("2")
    with pytest.raises(TypeError, match=r"these indexers \[1\.5\] of type float$"):
        frame.set_index("a").head(1.5)
    with pytest.raises(TypeError, match=r"these indexers \[2\] of type str$"):
        frame["a"].head("2")
    with pytest.raises(TypeError, match=r"^bad operand type for unary -: 'str'$"):
        frame.tail("2")
    assert len(frame.tail(2)) == 2


@pytest.mark.parametrize("method", ["sum", "mean", "max", "std", "first"])
def test_group_by_numeric_only_is_a_boolean(firepanda: ModuleType, method: str) -> None:
    """A group by checks the flag where a frame only asks whether it is true."""
    grouped = _frame(firepanda).groupby("a")
    with pytest.raises(ValueError, match=r"^numeric_only accepts only Boolean values$"):
        getattr(grouped, method)(numeric_only="x")


def test_frame_numeric_only_is_read_by_its_truth(firepanda: ModuleType) -> None:
    """Any truthy value keeps the numeric columns, as pandas reads it."""
    frame = firepanda.DataFrame({"a": [1, 2], "s": ["x", "y"]})
    assert list(frame.sum(numeric_only="x").index) == ["a"]
    assert list(frame.rolling(2).sum(numeric_only=1).columns) == ["a"]
