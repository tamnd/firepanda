"""Index.insert of another kind, MultiIndex.to_flat_index, and slice_locs bound types."""

import numpy as np
import pytest

import firepanda as pd


def test_a_text_label_makes_an_index_of_objects():
    out = pd.Index([1, 2], name="k").insert(1, "x")
    assert list(out) == [1, "x", 2]
    assert str(out.dtype) == "object"
    assert out.name == "k"


def test_a_negative_position_counts_from_the_end():
    out = pd.Index([1, 2]).insert(-1, "z")
    assert list(out) == [1, "z", 2]


def test_a_float_among_whole_numbers_makes_floats():
    out = pd.Index([1, 2]).insert(1, 2.5)
    assert list(out) == [1.0, 2.5, 2.0]
    assert str(out.dtype) == "float64"


def test_a_position_past_the_end_is_an_index_error():
    with pytest.raises(IndexError, match="index 9 is out of bounds for axis 0 with size 2"):
        pd.Index([1, 2]).insert(9, "x")


def test_the_flat_index_holds_tuples():
    index = pd.MultiIndex.from_tuples([(1, "a"), (2, "b")], names=["x", "y"])
    out = index.to_flat_index()
    assert not isinstance(out, pd.MultiIndex)
    assert list(out) == [(1, "a"), (2, "b")]
    assert str(out.dtype) == "object"


def test_a_searched_bound_is_a_numpy_integer():
    first, last = pd.Index([1, 3, 5]).slice_locs(2, 4)
    assert (first, last) == (1, 2)
    assert isinstance(first, np.int64)
    assert isinstance(last, np.int64)


def test_a_held_bound_is_a_plain_integer():
    first, last = pd.Index([1, 3, 5]).slice_locs(3)
    assert (first, last) == (1, 3)
    assert type(first) is int
    assert type(last) is int


def test_a_backward_slice_searches_both_bounds():
    first, last = pd.Index([1, 3, 5]).slice_locs(4, 2, step=-1)
    assert (first, last) == (1, 0)
    assert isinstance(first, np.int64)
