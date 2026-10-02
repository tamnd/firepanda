"""Index labels as numpy scalars, set operations named for this side, putmask of other kinds."""

import math

import numpy as np

import firepanda as pd


def test_a_label_out_of_a_number_index_is_a_numpy_scalar():
    assert isinstance(pd.Index([1, 3, 5])[1], np.int64)
    assert isinstance(pd.Index([1.5, 2.0])[0], np.float64)
    assert isinstance(pd.Index([True, False])[0], np.bool_)


def test_a_label_out_of_a_range_or_text_stays_plain():
    assert type(pd.RangeIndex(3)[1]) is int
    assert pd.Index(["a"])[0] == "a"


def test_asof_answers_a_numpy_scalar():
    found = pd.Index([1, 3, 5]).asof(4)
    assert isinstance(found, np.int64)
    assert found == 3
    assert math.isnan(pd.Index([1, 3, 5]).asof(0))


def test_a_list_keeps_the_name():
    index = pd.Index([3, 1, 2], name="k")
    assert index.difference([1], sort=False).name == "k"
    assert index.symmetric_difference([2, 5]).name == "k"
    assert index.union([0], sort=False).name == "k"


def test_another_index_with_another_name_takes_it_away():
    index = pd.Index([1, 2], name="k")
    assert index.symmetric_difference(pd.Index([2, 5], name="j")).name is None


def test_a_column_is_read_as_its_values():
    out = pd.Index([1, 2], name="k").union(pd.Series([2, 5], name="j"))
    assert list(out) == [1, 2, 5]
    assert out.name == "k"


def test_intersection_sorts_unless_the_sides_match():
    index = pd.Index([3, 1, 2], name="k")
    assert list(index.intersection([2, 3, 9], sort=None)) == [2, 3]
    assert list(index.intersection(pd.Index([3, 1, 2]), sort=None)) == [3, 1, 2]
    assert list(index.intersection([2, 3, 9])) == [3, 2]


def test_putmask_of_another_kind_reads_the_labels_again():
    floats = pd.Index([1, 2, 3]).putmask([False, True, False], 2.5)
    assert list(floats) == [1.0, 2.5, 3.0]
    assert str(floats.dtype) == "float64"
    mixed = pd.Index([3, 1, 2], name="k").putmask([True, False, False], "x")
    assert list(mixed) == ["x", 1, 2]
    assert str(mixed.dtype) == "object"
    assert mixed.name == "k"
