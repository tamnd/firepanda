"""`get` and `squeeze` on a frame and on a series, against pandas.

Two small methods with one thing in common, which is that what they answer
depends on the data rather than on the call. `get` is square brackets with the
failure turned into a value, and `squeeze` is three shapes of answer out of one
name. Neither of them can be a row in the generated table for that reason, and
both of them are the kind of method that is easy to write slightly wrong, so
the answers below are compared against pandas rather than asserted.
"""

from __future__ import annotations

import pandas as pd
import pytest


def rows(module):
    """Three rows and two columns, which is neither axis short enough to drop."""
    return module.DataFrame({"a": [1, 2, 3], "b": [4.5, 5.5, 6.5]})


def one_column(module):
    """Three rows and one column, which squeezes to the column."""
    return module.DataFrame({"a": [1, 2, 3]})


def one_cell(module):
    """One row and one column, which squeezes all the way to the value."""
    return module.DataFrame({"a": [7]})


def test_get_answers_the_column_when_there_is_one(firepanda):
    got = rows(firepanda).get("a")
    assert got.tolist() == rows(pd).get("a").tolist() == [1, 2, 3]
    assert got.name == "a"


def test_get_answers_the_default_when_there_is_not(firepanda):
    assert rows(firepanda).get("z", "missing") == rows(pd).get("z", "missing")
    assert rows(firepanda).get("z", "missing") == "missing"


def test_get_answers_none_when_no_default_was_written(firepanda):
    assert rows(firepanda).get("z") is None
    assert rows(pd).get("z") is None


def test_get_takes_a_list_and_answers_a_frame(firepanda):
    got = rows(firepanda).get(["a", "b"])
    assert list(got.columns) == list(rows(pd).get(["a", "b"]).columns) == ["a", "b"]


def test_get_answers_the_default_when_one_name_of_a_list_is_missing(firepanda):
    assert rows(firepanda).get(["a", "z"], "m") == rows(pd).get(["a", "z"], "m")


def test_get_answers_the_default_for_a_key_that_is_not_a_name_at_all(firepanda):
    # pandas reads this as a column it does not have and this reads it as a key
    # square brackets do not take. Both answer the default, which is the only
    # thing a caller of `get` is in a position to see.
    assert rows(firepanda).get(0, "m") == rows(pd).get(0, "m") == "m"


def test_get_on_a_series_reads_a_label(firepanda):
    series = firepanda.Series([10, 20, 30], name="v")
    theirs = pd.Series([10, 20, 30], name="v")
    assert series.get(1) == theirs.get(1) == 20
    assert series.get(9, "d") == theirs.get(9, "d") == "d"
    assert series.get(9) is None


def test_get_on_a_series_reads_a_list_of_labels(firepanda):
    series = firepanda.Series([10, 20, 30], name="v")
    assert series.get([1, 2]).tolist() == pd.Series([10, 20, 30]).get([1, 2]).tolist()


def test_squeeze_takes_the_column_axis_off_a_frame_of_one_column(firepanda):
    got = one_column(firepanda).squeeze()
    want = one_column(pd).squeeze()
    assert got.tolist() == want.tolist() == [1, 2, 3]
    assert got.name == want.name == "a"


def test_squeeze_takes_both_axes_off_a_frame_of_one_cell(firepanda):
    assert one_cell(firepanda).squeeze() == one_cell(pd).squeeze() == 7


def test_squeeze_leaves_a_frame_that_has_nothing_to_drop_alone(firepanda):
    got = rows(firepanda).squeeze()
    assert list(got.columns) == ["a", "b"]
    assert got.shape == rows(pd).squeeze().shape == (3, 2)


def test_squeeze_answers_something_other_than_what_it_was_given(firepanda):
    # pandas hands back a new object even when there was nothing to drop, so a
    # caller who checks identity is checking something real and this agrees.
    frame = rows(firepanda)
    assert frame.squeeze() is not frame
    theirs = rows(pd)
    assert theirs.squeeze() is not theirs


def test_an_axis_names_which_of_the_two_may_be_dropped(firepanda):
    # One row and one column, so naming an axis picks which one goes and the
    # answer is a series rather than the value. Only one of the two is here,
    # because the other one reads the row across the columns.
    frame = one_cell(firepanda)
    assert frame.squeeze(axis=1).tolist() == one_cell(pd).squeeze(axis=1).tolist() == [7]
    assert list(frame.squeeze(axis=1).index) == [0]


def test_an_axis_spelled_as_a_word_means_the_same_thing(firepanda):
    frame = one_column(firepanda)
    assert frame.squeeze(axis="columns").tolist() == [1, 2, 3]
    assert frame.squeeze(axis="index").shape == (3, 1)
    assert frame.squeeze(axis="rows").shape == (3, 1)


def test_an_axis_a_frame_does_not_have_is_refused_in_pandas_words(firepanda):
    with pytest.raises(ValueError, match="No axis named 2 for object type DataFrame"):
        rows(firepanda).squeeze(axis=2)
    with pytest.raises(ValueError, match="No axis named 2 for object type DataFrame"):
        rows(pd).squeeze(axis=2)


def test_squeeze_on_a_series_of_one_row_answers_the_value(firepanda):
    series = firepanda.Series([10], name="v")
    assert series.squeeze() == pd.Series([10], name="v").squeeze() == 10


def test_squeeze_on_a_longer_series_answers_the_series(firepanda):
    series = firepanda.Series([10, 20, 30], name="v")
    got = series.squeeze()
    assert got.tolist() == [10, 20, 30]
    assert got.name == "v"
    assert got is not series


def test_an_axis_a_series_does_not_have_is_refused_in_pandas_words(firepanda):
    with pytest.raises(ValueError, match="No axis named 1 for object type Series"):
        firepanda.Series([10, 20], name="v").squeeze(axis=1)
    with pytest.raises(ValueError, match="No axis named 1 for object type Series"):
        pd.Series([10, 20], name="v").squeeze(axis=1)


def test_the_axis_a_series_does_have_is_taken(firepanda):
    series = firepanda.Series([10], name="v")
    assert series.squeeze(axis=0) == 10
    assert series.squeeze(axis="index") == 10


def test_a_row_read_across_the_columns_says_what_it_cannot_do(firepanda):
    # pandas answers the row as a series here. A row across columns of
    # different types needs a type of its own and pandas' rule for finding one
    # ends at the object dtype, which this library does not have. It is also
    # named after the row label, which pandas carries as an int and a series
    # here carries as a string, so the refusal covers the one column shape too
    # even though that one has a type it could have used.
    with pytest.raises(NotImplementedError, match="drops the row axis"):
        rows(firepanda).head(1).squeeze()
    assert rows(pd).head(1).squeeze().tolist() == [1.0, 4.5]
    with pytest.raises(NotImplementedError, match="drops the row axis"):
        one_cell(firepanda).squeeze(axis=0)
    assert one_cell(pd).squeeze(axis=0).tolist() == [7]


def test_squeeze_keeps_the_labels_the_frame_had(firepanda):
    frame = firepanda.DataFrame({"key": [4, 3, 2], "v": [10, 20, 30]})
    got = frame.set_index("key").squeeze()
    assert list(got.index) == [4, 3, 2]
    assert got.tolist() == [10, 20, 30]


def test_get_on_a_frame_whose_labels_are_not_the_positions(firepanda):
    frame = firepanda.DataFrame({"key": [4, 3, 2], "v": [10, 20, 30]})
    series = frame.set_index("key")["v"]
    assert series.get(3) == 20
    assert series.get(0, "d") == "d"
