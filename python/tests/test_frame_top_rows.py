"""`nlargest` and `nsmallest` on a frame, against pandas.

These two pick rows and sort the ones they picked, so an answer can be wrong in
two ways that look alike from a distance: it can keep the wrong rows, or it can
keep the right ones and hand them back in the wrong order. Every case here
compares both, and most of them compare against pandas rather than against a
written expectation, because the rule being checked is pandas' rule.

The frames are small and the ranking column has ties in it on purpose. A frame
of distinct values would pass with any tie rule at all, including none, which
is exactly the part of this that is worth checking.
"""

from __future__ import annotations

import pandas as pd
import pyarrow as pa
import pytest

ROWS = {
    "key": [3, 1, 3, 2, 1, 3, 2, 1],
    "value": [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0],
}


def made(firepanda):
    """The frame under test."""
    return firepanda.DataFrame(dict(ROWS))


def theirs():
    """The same frame in pandas."""
    return pd.DataFrame(dict(ROWS))


def same_frame(mine, them):
    """Asserts that two frames agree about labels, columns and values."""
    assert list(mine.index) == list(them.index)
    assert list(mine.columns) == list(them.columns)
    ours = pa.table(mine).to_pandas()
    ours.index = them.index
    pd.testing.assert_frame_equal(ours, them, check_dtype=False)


def test_the_largest_rows_come_back_sorted(firepanda):
    same_frame(made(firepanda).nlargest(3, "value"), theirs().nlargest(3, "value"))


def test_the_smallest_rows_come_back_sorted(firepanda):
    same_frame(made(firepanda).nsmallest(3, "value"), theirs().nsmallest(3, "value"))


def test_a_tie_keeps_the_row_that_came_first(firepanda):
    same_frame(made(firepanda).nlargest(4, "key"), theirs().nlargest(4, "key"))


def test_keeping_the_last_of_a_tie_reverses_the_tied_rows(firepanda):
    # The membership and the ordering both change here. pandas hands the tied
    # rows back last first, which falls out of how it reads the column, and a
    # library that got the membership right and the order wrong would look
    # correct in every test that only counted rows.
    same_frame(
        made(firepanda).nlargest(4, "key", keep="last"),
        theirs().nlargest(4, "key", keep="last"),
    )


def test_keeping_the_last_reads_the_small_end_the_same_way(firepanda):
    same_frame(
        made(firepanda).nsmallest(3, "key", keep="last"),
        theirs().nsmallest(3, "key", keep="last"),
    )


def test_the_kept_rows_carry_the_labels_they_had(firepanda):
    # The labels say which rows of the input these were, and they are the only
    # thing in the answer that does, since the frame comes back reordered.
    assert list(made(firepanda).nlargest(3, "value").index) == [7, 6, 5]


def test_a_column_named_in_a_list_is_the_same_as_a_bare_one(firepanda):
    same_frame(
        made(firepanda).nlargest(3, ["value"]),
        theirs().nlargest(3, ["value"]),
    )


def test_a_column_written_twice_is_read_once(firepanda):
    same_frame(
        made(firepanda).nlargest(3, ["key", "key"]),
        theirs().nlargest(3, ["key", "key"]),
    )


def test_asking_for_more_rows_than_there_are_gives_all_of_them(firepanda):
    same_frame(made(firepanda).nlargest(100, "value"), theirs().nlargest(100, "value"))


def test_asking_for_none_gives_an_empty_frame(firepanda):
    same_frame(made(firepanda).nlargest(0, "value"), theirs().nlargest(0, "value"))


def test_asking_for_fewer_than_none_also_gives_an_empty_frame(firepanda):
    # pandas answers an empty frame rather than raising, so a count of zero and
    # a count of minus one are the same request and neither of them is a
    # mistake worth reporting.
    same_frame(made(firepanda).nlargest(-2, "value"), theirs().nlargest(-2, "value"))


def test_no_column_at_all_gives_an_empty_frame_with_the_columns_kept(firepanda):
    same_frame(made(firepanda).nlargest(3, []), theirs().nlargest(3, []))


def test_a_missing_value_ranks_last_and_is_still_a_row(firepanda):
    # pandas pads the answer with the missing values once the present ones run
    # out, rather than answering short, so asking for four rows out of a column
    # with three present values gets four. The kernel underneath never keeps a
    # missing value, so this is the layer that has to know.
    rows = {"value": [1.0, None, 9.0, None, 5.0]}
    for n in (2, 3, 4, 5, 9):
        same_frame(
            firepanda.DataFrame(dict(rows)).nlargest(n, "value"),
            pd.DataFrame(dict(rows)).nlargest(n, "value"),
        )


def test_the_padding_walks_forwards_under_both_tie_rules(firepanda):
    # The rest of the answer reverses under keep="last" and the padding does
    # not, which is what pandas does and is not what falls out of the reversal.
    rows = {"value": [1.0, None, 9.0, None, 5.0]}
    same_frame(
        firepanda.DataFrame(dict(rows)).nlargest(5, "value", keep="last"),
        pd.DataFrame(dict(rows)).nlargest(5, "value", keep="last"),
    )


def test_a_column_of_nothing_but_missing_values_still_answers_rows(firepanda):
    # pandas refuses this, because a column of nothing but `None` is an object
    # column there and object columns cannot be ranked. Here it is a column of
    # floats that are all absent, which can be ranked and which every value
    # loses, so the answer is the padding and nothing else.
    rows = {"value": [None, None, None]}
    mine = firepanda.DataFrame(dict(rows)).nsmallest(2, "value")
    assert list(mine.index) == [0, 1]
    with pytest.raises(TypeError):
        pd.DataFrame(dict(rows)).nsmallest(2, "value")


def test_a_frame_with_no_rows_answers_one(firepanda):
    mine = firepanda.DataFrame({"value": []}).nlargest(3, "value")
    assert len(mine) == 0
    assert list(mine.columns) == ["value"]


def test_a_count_that_is_not_whole_is_refused(firepanda):
    with pytest.raises(TypeError) as raised:
        made(firepanda).nlargest(2.5, "value")
    assert "integer" in str(raised.value)
    with pytest.raises(TypeError):
        theirs().nlargest(2.5, "value")


def test_a_rule_that_is_not_one_of_the_three_is_refused(firepanda):
    with pytest.raises(ValueError) as raised:
        made(firepanda).nlargest(2, "value", keep="oldest")
    assert 'keep must be either "first", "last" or "all"' in str(raised.value)
    with pytest.raises(ValueError):
        theirs().nlargest(2, "value", keep="oldest")


def test_keeping_all_of_a_tie_is_refused_rather_than_answered_wrongly(firepanda):
    # pandas answers more than n rows for this, and an answer whose height
    # depends on the values in the column is a different piece of work from the
    # fixed table of slots underneath. Saying so is better than quietly
    # answering the keep first rows and looking right until there is a tie.
    with pytest.raises(NotImplementedError) as raised:
        made(firepanda).nlargest(2, "key", keep="all")
    assert "keep='all'" in str(raised.value)
    assert len(theirs().nlargest(2, "key", keep="all")) == 3


def test_ranking_by_two_columns_is_refused(firepanda):
    with pytest.raises(NotImplementedError) as raised:
        made(firepanda).nlargest(2, ["key", "value"])
    assert "one column" in str(raised.value)


def test_a_column_that_is_not_there_is_refused(firepanda):
    with pytest.raises(KeyError):
        made(firepanda).nlargest(2, "nope")
    with pytest.raises(KeyError):
        theirs().nlargest(2, "nope")


def test_a_column_that_cannot_be_ranked_is_refused(firepanda):
    rows = {"word": ["a", "b", "c"]}
    with pytest.raises(TypeError):
        firepanda.DataFrame(dict(rows)).nlargest(2, "word")
    with pytest.raises(TypeError):
        pd.DataFrame(dict(rows)).nlargest(2, "word")


def test_the_two_ends_of_a_column_are_the_same_question(firepanda):
    # What makes these one piece of work rather than two. The smallest three of
    # a column are the largest three of it read from the other end, so a frame
    # where that does not hold has one of the two wrong.
    df = made(firepanda)
    high = list(df.nlargest(8, "value").index)
    low = list(df.nsmallest(8, "value").index)
    assert high == list(reversed(low))
