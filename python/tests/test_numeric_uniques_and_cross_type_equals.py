"""Numeric uniques, label values and masks as numpy arrays, and equals across number types."""

import numpy as np

import firepanda as pd


def test_unique_of_numbers_is_a_numpy_array():
    out = pd.Series([2, 1, 3, 3]).unique()
    assert isinstance(out, np.ndarray)
    assert out.dtype == np.int64
    assert out.tolist() == [2, 1, 3]


def test_unique_of_floats_keeps_one_nan():
    out = pd.Series([1.5, None, 1.5]).unique()
    assert isinstance(out, np.ndarray)
    assert out[0] == 1.5
    assert np.isnan(out[1])


def test_unique_of_flags_and_top_level_unique():
    assert pd.Series([True, False, True]).unique().tolist() == [True, False]
    assert isinstance(pd.unique(pd.Series([2, 1, 2])), np.ndarray)


def test_unique_of_text_is_not_numpy():
    assert not isinstance(pd.Series(["a", "b", "a"]).unique(), np.ndarray)


def test_get_loc_mask_is_numpy():
    out = pd.Index(list("abab")).get_loc("b")
    assert isinstance(out, np.ndarray)
    assert out.tolist() == [False, True, False, True]
    assert pd.Index(list("aabb")).get_loc("b") == slice(2, 4)


def test_index_values_of_numbers_is_numpy():
    out = pd.Index([1, 2]).values
    assert isinstance(out, np.ndarray)
    assert out.tolist() == [1, 2]
    assert pd.Index(["a", "b"]).values == ["a", "b"]


def test_equals_across_number_types():
    left = pd.Index([1, 2])
    assert left.equals(pd.Index([1, 2], dtype="uint64"))
    assert left.equals(pd.Index([1.0, 2.0]))
    assert left.equals(pd.Index([1, 2], dtype="object"))
    assert not left.equals(pd.Index([1.0, 2.5]))
    assert not left.equals(pd.Index(["1", "2"]))
    assert not left.equals(pd.Index([1, 2, 3], dtype="uint64"))
