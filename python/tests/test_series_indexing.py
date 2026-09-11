"""`loc`, `iloc`, `at`, `iat` and square brackets on a series, against pandas.

The frame's accessors decide between three shapes of answer because a frame has
two axes. A series has one, so the only decision left is whether the key names
one row or a set of them, and most of this file is about the places where that
is less obvious than it sounds.

The one that is worth reading before the rest is `__getitem__`. `s[2]` is the
label two and `s[2:5]` is the rows two to five counting from the front, on the
same series, in the same session. Nobody would design that. It is what pandas
does and it is written down in code everywhere, so the tests below ask pandas
the same questions and compare, rather than asserting what seems reasonable.
"""

from __future__ import annotations

import pandas as pd
import pytest


def counted(firepanda):
    """Five rows under a plain range index, where label and position agree."""
    return firepanda.Series([10, 20, 30, 40, 50], name="v")


def theirs():
    """The same five rows in pandas."""
    return pd.Series([10, 20, 30, 40, 50], name="v")


def shuffled(module):
    """Five rows labelled four down to zero, where they disagree."""
    frame = module.DataFrame({"key": [4, 3, 2, 1, 0], "v": [10, 20, 30, 40, 50]})
    return frame.set_index("key")["v"]


def worded(module):
    """Three rows labelled by strings, where a number can only be a position."""
    frame = module.DataFrame({"k": ["a", "b", "c"], "v": [10, 20, 30]})
    return frame.set_index("k")["v"]


def test_a_position_reads_the_row_at_that_position(firepanda):
    assert counted(firepanda).iloc[0] == theirs().iloc[0]
    assert shuffled(firepanda).iloc[3] == shuffled(pd).iloc[3]


def test_a_negative_position_counts_from_the_end(firepanda):
    assert counted(firepanda).iloc[-1] == 50
    assert counted(firepanda).iat[-1] == 50


def test_a_slice_of_positions_leaves_out_the_row_it_stops_at(firepanda):
    got = counted(firepanda).iloc[1:3]
    want = theirs().iloc[1:3]
    assert got.tolist() == want.tolist() == [20, 30]
    assert list(got.index) == list(want.index) == [1, 2]


def test_a_slice_of_positions_keeps_its_step(firepanda):
    got = counted(firepanda).iloc[::2]
    assert got.tolist() == theirs().iloc[::2].tolist() == [10, 30, 50]


def test_a_slice_of_positions_counts_its_ends_from_the_end(firepanda):
    got = counted(firepanda).iloc[-2:]
    assert got.tolist() == theirs().iloc[-2:].tolist() == [40, 50]


def test_a_repeated_position_gives_a_repeated_row(firepanda):
    got = counted(firepanda).iloc[[0, 0, 1]]
    want = theirs().iloc[[0, 0, 1]]
    assert got.tolist() == want.tolist() == [10, 10, 20]
    assert list(got.index) == list(want.index) == [0, 0, 1]


def test_a_run_of_booleans_is_a_mask_even_under_iloc(firepanda):
    # `iloc` is about positions and a mask is not one, and pandas takes it
    # anyway and reads it as the positions it is true at.
    got = counted(firepanda).iloc[[True, False, True, False, True]]
    assert got.tolist() == [10, 30, 50]


def test_a_column_of_booleans_is_a_mask_under_either_accessor(firepanda):
    series = counted(firepanda)
    assert series.iloc[series > 25].tolist() == [30, 40, 50]
    assert series.loc[series > 25].tolist() == [30, 40, 50]


def test_a_label_reads_the_row_with_that_label(firepanda):
    assert shuffled(firepanda).loc[3] == shuffled(pd).loc[3] == 20
    assert worded(firepanda).loc["b"] == 20


def test_a_slice_of_labels_includes_the_one_it_stops_at(firepanda):
    got = counted(firepanda).loc[1:3]
    want = theirs().loc[1:3]
    assert got.tolist() == want.tolist() == [20, 30, 40]


def test_a_slice_of_labels_runs_in_the_order_the_labels_are_in(firepanda):
    # Labels four down to zero, so three to one is three rows and not none.
    got = shuffled(firepanda).loc[3:1]
    want = shuffled(pd).loc[3:1]
    assert got.tolist() == want.tolist() == [20, 30, 40]


def test_a_slice_of_labels_may_stop_past_the_end(firepanda):
    got = worded(firepanda).loc["a":"z"]
    assert got.tolist() == worded(pd).loc["a":"z"].tolist() == [10, 20, 30]


def test_a_list_of_labels_comes_back_in_the_order_it_was_asked_for(firepanda):
    got = shuffled(firepanda).loc[[4, 0]]
    want = shuffled(pd).loc[[4, 0]]
    assert got.tolist() == want.tolist() == [10, 50]
    assert list(got.index) == list(want.index) == [4, 0]


def test_square_brackets_read_one_key_as_a_label(firepanda):
    assert shuffled(firepanda)[3] == shuffled(pd)[3] == 20
    assert worded(firepanda)["b"] == 20


def test_square_brackets_read_a_slice_of_numbers_as_positions(firepanda):
    # The whole difficulty of this method. The labels are four down to zero and
    # this answers the second and third rows rather than the labels one and two.
    got = shuffled(firepanda)[1:3]
    want = shuffled(pd)[1:3]
    assert got.tolist() == want.tolist() == [20, 30]
    assert list(got.index) == list(want.index) == [3, 2]


def test_square_brackets_read_a_slice_of_anything_else_as_labels(firepanda):
    got = worded(firepanda)["a":"b"]
    assert got.tolist() == worded(pd)["a":"b"].tolist() == [10, 20]


def test_a_number_in_square_brackets_is_a_label_even_on_a_string_index(firepanda):
    with pytest.raises(KeyError):
        worded(firepanda)[0]
    with pytest.raises(KeyError):
        worded(pd)[0]


def test_a_slice_of_numbers_is_positions_even_on_a_string_index(firepanda):
    got = worded(firepanda)[0:2]
    assert got.tolist() == worded(pd)[0:2].tolist() == [10, 20]


def test_square_brackets_take_a_mask(firepanda):
    series = counted(firepanda)
    assert series[series > 25].tolist() == [30, 40, 50]


def test_square_brackets_take_a_list_of_labels(firepanda):
    got = shuffled(firepanda)[[4, 0]]
    assert got.tolist() == shuffled(pd)[[4, 0]].tolist() == [10, 50]


def test_the_name_and_the_index_name_both_survive_a_selection(firepanda):
    got = shuffled(firepanda).iloc[1:3]
    assert got.name == "v"
    assert got.index.name == "key"


def test_one_coordinate_by_label_and_one_by_position(firepanda):
    assert shuffled(firepanda).at[0] == shuffled(pd).at[0] == 50
    assert shuffled(firepanda).iat[0] == shuffled(pd).iat[0] == 10


def test_a_position_past_the_end_says_so_in_pandas_words(firepanda):
    with pytest.raises(IndexError, match="single positional indexer is out-of-bounds"):
        counted(firepanda).iloc[9]
    with pytest.raises(IndexError, match="single positional indexer is out-of-bounds"):
        theirs().iloc[9]


def test_at_and_iloc_are_told_different_things_about_the_same_mistake(firepanda):
    # pandas says one sentence from `iloc` and another from `iat`, so both are
    # raised by the caller that pandas raises them from.
    with pytest.raises(IndexError, match="out of bounds for axis 0 with size 5"):
        counted(firepanda).iat[9]
    with pytest.raises(IndexError, match="out of bounds for axis 0 with size 5"):
        theirs().iat[9]


def test_one_position_of_a_list_past_the_end_stops_the_whole_gather(firepanda):
    with pytest.raises(IndexError, match="positional indexers are out-of-bounds"):
        counted(firepanda).iloc[[0, 9]]


def test_a_label_the_series_does_not_have_is_the_label_and_nothing_else(firepanda):
    with pytest.raises(KeyError) as raised:
        counted(firepanda).loc[9]
    assert raised.value.args[0] == 9
    with pytest.raises(KeyError):
        counted(firepanda).at[9]


def test_a_list_naming_a_label_that_is_missing_names_it_back(firepanda):
    with pytest.raises(KeyError, match=r"\[9\] not in index"):
        counted(firepanda).loc[[1, 9]]


def test_a_second_axis_is_refused_since_there_is_not_one(firepanda):
    with pytest.raises(ValueError, match="Too many indexers"):
        counted(firepanda).iloc[(0, 1)]
    with pytest.raises(ValueError, match="Too many indexers"):
        counted(firepanda).loc[(0, 1)]


def test_a_boolean_key_of_the_wrong_length_is_refused(firepanda):
    with pytest.raises(IndexError, match="Boolean index has wrong length"):
        counted(firepanda).iloc[[True, False]]


def test_a_missing_value_comes_back_as_none(firepanda):
    series = firepanda.Series([1.5, None, 3.5], name="v")
    assert series.iloc[1] is None
    assert series.iat[1] is None
    assert series.loc[1] is None


def test_a_selection_off_a_frame_column_still_has_the_frames_labels(firepanda):
    frame = firepanda.DataFrame({"key": [4, 3, 2, 1, 0], "v": [10, 20, 30, 40, 50]})
    got = frame.set_index("key")["v"].loc[[3, 1]]
    assert list(got.index) == [3, 1]
    assert got.tolist() == [20, 40]


def test_the_frames_accessors_still_answer_what_they_did(firepanda):
    # The two key readers moved out of the frame's accessor classes to be
    # shared with the series, and this is the check that the move was a move.
    frame = firepanda.DataFrame({"a": [1, 2, 3], "b": [4.5, 5.5, 6.5]})
    assert frame.iloc[0, 0] == 1
    assert frame.iloc[1:3]["a"].tolist() == [2, 3]
    assert frame.loc[1:2]["a"].tolist() == [2, 3]
    assert frame.at[0, "a"] == 1
    assert frame.iat[2, 1] == 6.5
