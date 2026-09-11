"""`loc`, `iloc`, `at`, `iat` and `take` on a frame, against pandas.

Every case here runs both libraries and compares, because the whole of this
work is a rule about which of two mental models a key belongs to and a test
that only asserts a shape would pass under either. The frames are built with a
column whose order disagrees with the index's order, since on a frame that has
never been sorted a label and a position are the same number and a test written
on one of those is testing nothing.
"""

from __future__ import annotations

import pandas as pd
import pyarrow as pa
import pytest

ROWS = {
    "k": [30, 10, 20, 10, 50, 40],
    "v": [1, 2, 3, 4, 5, 6],
    "w": [1.5, 2.5, 3.5, 4.5, 5.5, 6.5],
}


def made(firepanda):
    """The frame under test, with its labels no longer in value order."""
    return firepanda.DataFrame(dict(ROWS))


def theirs():
    """The same frame in pandas."""
    return pd.DataFrame(dict(ROWS))


def same(mine, them):
    """Asserts that two frames agree about labels, columns and values."""
    assert list(mine.index) == list(them.index)
    assert list(mine.columns) == list(them.columns)
    ours = pa.table(mine).to_pandas()
    ours.index = them.index
    pd.testing.assert_frame_equal(ours, them, check_dtype=False)


def column(mine, them):
    """Asserts that two columns agree about name, labels and values."""
    assert mine.name == them.name
    assert list(mine.index) == list(them.index)
    assert mine.tolist() == them.tolist()


def test_iloc_reads_one_value_by_a_pair_of_positions(firepanda):
    assert made(firepanda).iloc[0, 0] == theirs().iloc[0, 0]
    assert made(firepanda).iloc[-1, 2] == theirs().iloc[-1, 2]


def test_iloc_slices_rows_without_the_row_it_stops_at(firepanda):
    same(made(firepanda).iloc[2:5], theirs().iloc[2:5])


def test_iloc_slices_from_the_end(firepanda):
    same(made(firepanda).iloc[-3:], theirs().iloc[-3:])


def test_iloc_walks_a_slice_with_a_step(firepanda):
    same(made(firepanda).iloc[::2], theirs().iloc[::2])


def test_iloc_gathers_a_list_of_positions_repeats_and_all(firepanda):
    same(made(firepanda).iloc[[0, 0, 1]], theirs().iloc[[0, 0, 1]])


def test_iloc_counts_a_negative_position_from_the_end(firepanda):
    same(made(firepanda).iloc[[-1, -2]], theirs().iloc[[-1, -2]])


def test_iloc_takes_one_column_as_a_series(firepanda):
    column(made(firepanda).iloc[:, 1], theirs().iloc[:, 1])


def test_iloc_takes_both_axes_at_once(firepanda):
    same(made(firepanda).iloc[1:5, 1:3], theirs().iloc[1:5, 1:3])


def test_iloc_takes_a_list_of_columns(firepanda):
    same(made(firepanda).iloc[:, [2, 0]], theirs().iloc[:, [2, 0]])


def test_iloc_refuses_a_position_past_the_end(firepanda):
    with pytest.raises(IndexError):
        made(firepanda).iloc[[0, 99]]


def test_iloc_refuses_a_column_past_the_end(firepanda):
    with pytest.raises(IndexError):
        made(firepanda).iloc[0, 9]


def test_iloc_reads_a_boolean_list_as_a_mask(firepanda):
    keep = [True, False, True, False, True, False]
    same(made(firepanda).iloc[keep], theirs().iloc[keep])


def test_loc_slices_rows_including_the_label_it_stops_at(firepanda):
    same(made(firepanda).loc[2:5], theirs().loc[2:5])
    assert len(made(firepanda).loc[2:5]) == 4


def test_loc_takes_a_list_of_labels_in_the_order_asked_for(firepanda):
    sorted_mine = made(firepanda).set_index("k").sort_index()
    sorted_them = theirs().set_index("k").sort_index()
    same(sorted_mine.loc[[50, 10]], sorted_them.loc[[50, 10]])


def test_loc_gives_back_every_row_a_repeated_label_has(firepanda):
    mine = made(firepanda).set_index("k").sort_index()
    them = theirs().set_index("k").sort_index()
    same(mine.loc[[10]], them.loc[[10]])
    assert len(mine.loc[[10]]) == 2


def test_loc_complains_about_a_label_that_is_not_there(firepanda):
    with pytest.raises(KeyError):
        made(firepanda).set_index("k").loc[[10, 99]]


def test_loc_keeps_the_rows_a_column_of_booleans_is_true_at(firepanda):
    mine = made(firepanda)
    them = theirs()
    same(mine.loc[mine["v"] > 3], them.loc[them["v"] > 3])


def test_loc_takes_a_mask_and_a_list_of_columns_together(firepanda):
    mine = made(firepanda)
    them = theirs()
    same(mine.loc[mine["v"] > 3, ["w", "k"]], them.loc[them["v"] > 3, ["w", "k"]])


def test_loc_takes_one_column_as_a_series(firepanda):
    column(made(firepanda).loc[:, "w"], theirs().loc[:, "w"])


def test_loc_slices_the_columns_including_the_one_it_stops_at(firepanda):
    same(made(firepanda).loc[:, "k":"v"], theirs().loc[:, "k":"v"])


def test_loc_refuses_a_column_that_is_not_there(firepanda):
    with pytest.raises(KeyError):
        made(firepanda).loc[:, "nope"]


def test_a_mask_of_the_wrong_length_is_an_index_error(firepanda):
    # pandas raises IndexError here and not ValueError, and it says both
    # lengths, which is the only reason this mistake is ever quick to fix.
    for read in (lambda df: df.loc[[True, False]], lambda df: df.iloc[[True, False]]):
        with pytest.raises(IndexError) as raised:
            read(made(firepanda))
        assert "Boolean index has wrong length: 2 instead of 6" in str(raised.value)


def test_a_missing_row_label_names_the_label_and_nothing_else(firepanda):
    with pytest.raises(KeyError) as raised:
        made(firepanda).set_index("k").loc[99]
    assert str(raised.value) == "99"


def test_at_and_iat_read_one_value(firepanda):
    mine = made(firepanda).set_index("k")
    them = theirs().set_index("k")
    assert mine.at[20, "v"] == them.at[20, "v"]
    assert mine.iat[0, 1] == them.iat[0, 1]


def test_at_complains_about_a_label_that_is_not_there(firepanda):
    with pytest.raises(KeyError):
        made(firepanda).set_index("k").at[99, "v"]


def test_iat_complains_about_a_position_past_the_end(firepanda):
    with pytest.raises(IndexError):
        made(firepanda).iat[99, 0]


def test_take_gathers_rows_and_carries_their_labels(firepanda):
    same(made(firepanda).take([2, 0, 1]), theirs().take([2, 0, 1]))


def test_take_counts_a_negative_position_from_the_end(firepanda):
    same(made(firepanda).take([-1, -2]), theirs().take([-1, -2]))


def test_take_along_the_columns_picks_columns(firepanda):
    same(made(firepanda).take([2, 0], axis=1), theirs().take([2, 0], axis=1))


def test_take_refuses_a_position_past_the_end(firepanda):
    with pytest.raises(IndexError):
        made(firepanda).take([99])


def test_take_refuses_a_keyword_pandas_only_accepts_to_ignore(firepanda):
    with pytest.raises(NotImplementedError):
        made(firepanda).take([0], mode="clip")


def test_one_row_across_several_columns_is_not_written_yet(firepanda):
    with pytest.raises(NotImplementedError):
        made(firepanda).iloc[0]


def test_more_than_two_axes_is_refused(firepanda):
    with pytest.raises(ValueError):
        made(firepanda).iloc[0, 0, 0]


def test_the_selection_survives_a_sort_that_moved_the_labels(firepanda):
    mine = made(firepanda).set_index("k").sort_index()
    them = theirs().set_index("k").sort_index()
    same(mine.iloc[0:2], them.iloc[0:2])
    same(mine.loc[10:20], them.loc[10:20])
