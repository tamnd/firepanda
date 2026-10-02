"""RangeIndex methods answer with a RangeIndex wherever pandas does."""

import firepanda as pd


def _bounds(index):
    assert type(index) is pd.RangeIndex
    return (index.start, index.stop, index.step)


def test_slice_keeps_range():
    index = pd.RangeIndex(0, 20, 2, name="n")
    assert _bounds(index[2:8]) == (4, 16, 2)
    assert index[2:8].name == "n"
    assert _bounds(index[::-1]) == (18, -2, -2)


def test_take_with_even_step_is_range():
    index = pd.RangeIndex(10)
    assert _bounds(index[[1, 3, 5]]) == (1, 7, 2)
    assert _bounds(index.take([4])) == (4, 5, 1)


def test_take_without_even_step_is_plain():
    index = pd.RangeIndex(10)
    answer = index[[1, 2, 5]]
    assert type(answer) is pd.Index
    assert answer.tolist() == [1, 2, 5]


def test_delete_insert_and_drop():
    index = pd.RangeIndex(5)
    assert _bounds(index.delete(0)) == (1, 5, 1)
    assert type(index.delete(2)) is pd.Index
    assert _bounds(index.insert(5, 5)) == (0, 6, 1)
    assert _bounds(index.drop([4])) == (0, 4, 1)


def test_sort_values_keeps_range():
    index = pd.RangeIndex(0, 10, 2)
    assert _bounds(index.sort_values(ascending=False)) == (8, -2, -2)


def test_append_keeps_name():
    answer = pd.RangeIndex(3, name="n").append(pd.Index([3, 4]))
    assert _bounds(answer) == (0, 5, 1)


def test_union_and_intersection():
    assert _bounds(pd.RangeIndex(0, 5).union(pd.RangeIndex(5, 9))) == (0, 9, 1)
    met = pd.RangeIndex(0, 10, 2).intersection(pd.RangeIndex(0, 10, 3))
    assert _bounds(met) == (0, 10, 6)
    met = pd.RangeIndex(0, 20, 4).intersection(pd.RangeIndex(2, 30, 6))
    assert _bounds(met) == (8, 20, 12)
    assert _bounds(pd.RangeIndex(0, 10, 2).intersection(pd.RangeIndex(1, 10, 2))) == (0, 0, 1)


def test_arithmetic_keeps_range():
    index = pd.RangeIndex(0, 5)
    assert _bounds(index * 3) == (0, 15, 3)
    assert _bounds(index + 2) == (2, 7, 1)
    assert _bounds(-index) == (0, -5, -1)
    assert type(index * 1.5) is pd.Index


def test_min_max_are_python_ints():
    index = pd.RangeIndex(3, 30, 3)
    assert type(index.min()) is int and index.min() == 3
    assert type(index.max()) is int and index.max() == 27
    assert index.argmax() == 8


def test_repeat_is_plain():
    answer = pd.RangeIndex(2).repeat(2)
    assert type(answer) is pd.Index
    assert answer.tolist() == [0, 0, 1, 1]
