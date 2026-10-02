"""Whole number columns a caller hands over stay a plain index, as pandas keeps them."""

import io
import pickle

import numpy as np

import firepanda as pd


def kind(frame):
    return type(frame.columns).__name__


def plain():
    return pd.DataFrame([[1, 2, 3], [4, 5, 6]], columns=[0, 1, 2])


def test_columns_written_out_are_a_plain_index():
    assert kind(plain()) == "Index"
    assert kind(pd.DataFrame([[1, 2, 3]], columns=[10, 20, 30])) == "Index"


def test_columns_pandas_makes_are_a_range():
    assert kind(pd.DataFrame([[1, 2]])) == "RangeIndex"
    assert kind(pd.DataFrame({0: [1], 1: [2]})) == "RangeIndex"
    assert kind(pd.DataFrame([[1, 2]], columns=range(2))) == "RangeIndex"


def test_set_axis_rename_and_assignment_make_a_plain_index():
    frame = pd.DataFrame([[1, 2]])
    assert kind(frame.set_axis([0, 1], axis=1)) == "Index"
    assert kind(frame.set_axis(range(2), axis=1)) == "RangeIndex"
    assert kind(frame.rename(columns={0: 5, 1: 6})) == "Index"
    other = frame.copy()
    other.columns = [0, 1]
    assert kind(other) == "Index"


def test_transpose_takes_the_row_index_as_it_is():
    frame = pd.DataFrame({"a": [3, 1, 2]}, index=pd.Index([10, 20, 30], name="n"))
    columns = frame.T.columns
    assert type(columns).__name__ == "Index"
    assert columns.name == "n"
    assert kind(pd.DataFrame({"a": [1, 2]}).T) == "RangeIndex"
    assert kind(pd.DataFrame({"a": [1, 2]}, index=[0, 1]).T) == "Index"


def test_the_mark_is_carried_through_methods():
    frame = plain()
    for made in (
        frame.copy(),
        frame.head(1),
        frame[[0, 1]],
        frame.sort_values(0),
        frame + 1,
        frame.T.T,
        pd.concat([frame, frame]),
    ):
        assert kind(made) == "Index"


def test_the_mark_survives_pickle():
    frame = pd.read_pickle(io.BytesIO(pickle.dumps(plain())))
    assert kind(frame) == "Index"


def test_the_first_and_last_valid_labels_are_numpy_scalars():
    frame = pd.DataFrame({"b": [1.5, None, 2.5]}, index=[10, 20, 30])
    assert isinstance(frame["b"].first_valid_index(), np.int64)
    assert frame.last_valid_index() == 30
    assert isinstance(frame.last_valid_index(), np.int64)
    assert type(pd.Series([1.0, 2.0]).first_valid_index()) is int
    assert pd.Series([None, None], dtype="float64").first_valid_index() is None


def test_text_columns_are_not_marked():
    frame = pd.DataFrame([[1, 2]], columns=["a", "b"])
    assert not getattr(frame, "_plain_columns", False)
