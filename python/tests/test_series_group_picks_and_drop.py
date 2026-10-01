"""SeriesGroupBy picks and flags, TimedeltaIndex reductions, drop, equals and combine."""

import numpy as np

import firepanda as fp


def _grouped():
    return fp.Series([3, 1, 2, 5, 4], index=list("abcde")).groupby([1, 1, 1, 2, 2])


def test_series_group_largest_and_smallest_keep_key_and_label():
    top = _grouped().nlargest(2)
    assert top.tolist() == [3, 2, 5, 4]
    assert top.index.tolist() == [(1, "a"), (1, "c"), (2, "d"), (2, "e")]
    assert _grouped().nsmallest(1).tolist() == [1, 4]


def test_series_group_unique_and_monotonic_flags():
    groups = fp.Series([1, 1, 2, 3, 2]).groupby(["x", "x", "x", "y", "y"])
    assert groups.unique().tolist() == [[1, 2], [3, 2]]
    assert groups.is_monotonic_increasing.tolist() == [True, False]
    assert groups.is_monotonic_decreasing.tolist() == [False, True]


def test_timedelta_index_reductions():
    index = fp.to_timedelta(["1s", "2s", "6s"])
    assert index.sum() == fp.Timedelta("9s")
    assert index.mean() == fp.Timedelta("3s")
    assert index.median() == fp.Timedelta("2s")
    assert index.std() == fp.Timedelta("2.645751311s")


def test_drop_takes_a_tuple_as_one_multiindex_key():
    index = fp.MultiIndex.from_tuples([("a", 1), ("a", 2), ("b", 1)])
    frame = fp.DataFrame({"v": [1, 2, 3]}, index=index)
    assert frame.drop(("a", 2))["v"].tolist() == [1, 3]
    assert fp.Series([1, 2, 3], index=index).drop(("b", 1)).tolist() == [1, 2]
    assert frame.drop(("z", 9), errors="ignore")["v"].tolist() == [1, 2, 3]


def test_equals_across_label_kinds():
    assert fp.DataFrame({1: [10], 2: [20]}).equals(fp.DataFrame({1.0: [10], 2.0: [20]}))
    assert not fp.DataFrame({1: [10]}).equals(fp.DataFrame({1.0: [11]}))


def test_combine_with_a_numpy_function():
    left = fp.DataFrame({"A": [5, 0], "B": [2, 4]})
    right = fp.DataFrame({"A": [1, 1], "B": [3, 3]})
    out = left.combine(right, np.minimum)
    assert out["A"].tolist() == [1, 0]
    assert out["B"].tolist() == [2, 3]
