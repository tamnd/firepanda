"""Row labels that count up from zero are a RangeIndex, as they are in pandas."""

import firepanda as pd


def test_default_frame_index_is_range():
    frame = pd.DataFrame({"a": [1, 2, 3]})
    assert isinstance(frame.index, pd.RangeIndex)
    assert isinstance(frame["a"].index, pd.RangeIndex)


def test_range_bounds_of_core_range():
    index = pd.DataFrame({"a": [1, 2, 3]}).index
    assert (index.start, index.stop, index.step) == (0, 3, 1)


def test_reduction_over_range_columns():
    frame = pd.DataFrame([[1, 2, 3], [4, 5, 6]])
    assert isinstance(frame.sum().index, pd.RangeIndex)
    assert isinstance(frame.dtypes.index, pd.RangeIndex)


def test_reduction_over_plain_columns():
    frame = pd.DataFrame([[1, 2, 3], [4, 5, 6]], columns=[0, 1, 2])
    assert type(frame.sum().index) is pd.Index
    assert type(frame.dtypes.index) is pd.Index
    assert frame.sum().index.tolist() == [0, 1, 2]
