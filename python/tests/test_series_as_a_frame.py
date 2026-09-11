"""The series members that are a frame underneath, against pandas.

`to_frame` is the door and the rest of this file walks through it. A frame in
this library already knows how to drop repeated rows, take rows by position,
sort by label, truncate, and pick the largest few, and a column knew none of
those. What is worth testing is that the column goes in and comes back out
under its own name with its own labels, and that what the frame did to the rows
is what pandas does to the values.
"""

from __future__ import annotations

import pandas as pd
import pytest


def repeats(module):
    """Six values with repeats in them, under a name and on plain labels."""
    return module.Series([3, 1, 3, 2, 1, 4], name="v")


def shuffled(module):
    """Four values whose labels are not in order, which is what sorting is for.

    Through a frame rather than `Series(values, index=labels)`, because putting
    labels on a series as it is built is not written yet and this is the
    spelling both libraries take.
    """
    return module.DataFrame({"k": [2, 0, 3, 1], "v": [10, 20, 30, 40]}).set_index("k")["v"]


def test_to_frame_keeps_the_name_the_values_and_the_labels(firepanda):
    got = repeats(firepanda).to_frame()
    want = repeats(pd).to_frame()
    assert list(got.columns) == list(want.columns) == ["v"]
    assert got["v"].tolist() == want["v"].tolist()
    assert list(got.index) == list(want.index)


def test_to_frame_takes_a_name_of_the_callers_choosing(firepanda):
    got = repeats(firepanda).to_frame("other")
    want = repeats(pd).to_frame("other")
    assert list(got.columns) == list(want.columns) == ["other"]
    assert got["other"].tolist() == want["other"].tolist()


def test_an_unnamed_column_is_called_zero(firepanda):
    got = firepanda.Series([1, 2]).to_frame()
    # pandas calls it the integer zero and a column name here is a string, so
    # it is the text of it. That is the only difference between the two.
    assert list(got.columns) == ["0"]
    assert [str(one) for one in pd.Series([1, 2]).to_frame().columns] == ["0"]


def test_to_frame_keeps_labels_that_are_not_a_range(firepanda):
    got = shuffled(firepanda).to_frame()
    want = shuffled(pd).to_frame()
    assert list(got.index) == list(want.index) == [2, 0, 3, 1]


def test_duplicated_marks_the_repeats_and_keeps_the_name(firepanda):
    got = repeats(firepanda).duplicated()
    want = repeats(pd).duplicated()
    assert got.tolist() == want.tolist() == [False, False, True, False, True, False]
    assert got.name == want.name == "v"


@pytest.mark.parametrize("keep", ["first", "last", False])
def test_duplicated_settles_the_ties_the_way_it_is_told_to(firepanda, keep):
    got = repeats(firepanda).duplicated(keep=keep)
    want = repeats(pd).duplicated(keep=keep)
    assert got.tolist() == want.tolist()


@pytest.mark.parametrize("keep", ["first", "last", False])
def test_drop_duplicates_removes_them_and_keeps_the_labels(firepanda, keep):
    got = repeats(firepanda).drop_duplicates(keep=keep)
    want = repeats(pd).drop_duplicates(keep=keep)
    assert got.tolist() == want.tolist()
    assert list(got.index) == list(want.index)
    assert got.name == want.name == "v"


def test_drop_duplicates_can_number_the_rows_again(firepanda):
    got = repeats(firepanda).drop_duplicates(ignore_index=True)
    want = repeats(pd).drop_duplicates(ignore_index=True)
    assert got.tolist() == want.tolist()
    assert list(got.index) == list(want.index) == [0, 1, 2, 3]


def test_take_reads_positions_and_not_labels(firepanda):
    got = shuffled(firepanda).take([3, 0])
    want = shuffled(pd).take([3, 0])
    assert got.tolist() == want.tolist() == [40, 10]
    assert list(got.index) == list(want.index) == [1, 2]


def test_sort_index_puts_the_rows_in_the_order_of_their_labels(firepanda):
    got = shuffled(firepanda).sort_index()
    want = shuffled(pd).sort_index()
    assert got.tolist() == want.tolist() == [20, 40, 10, 30]
    assert list(got.index) == list(want.index) == [0, 1, 2, 3]


def test_sort_index_can_run_the_other_way(firepanda):
    got = shuffled(firepanda).sort_index(ascending=False)
    want = shuffled(pd).sort_index(ascending=False)
    assert got.tolist() == want.tolist()
    assert list(got.index) == list(want.index) == [3, 2, 1, 0]


def test_truncate_keeps_both_ends_of_the_range_it_is_given(firepanda):
    got = shuffled(firepanda).sort_index().truncate(1, 2)
    want = shuffled(pd).sort_index().truncate(1, 2)
    assert got.tolist() == want.tolist() == [40, 10]
    assert list(got.index) == list(want.index) == [1, 2]


def test_nlargest_and_nsmallest_read_the_ends_in_order(firepanda):
    assert repeats(firepanda).nlargest(2).tolist() == repeats(pd).nlargest(2).tolist()
    assert repeats(firepanda).nsmallest(2).tolist() == repeats(pd).nsmallest(2).tolist()
    got = repeats(firepanda).nlargest(2)
    assert list(got.index) == list(repeats(pd).nlargest(2).index)
    assert got.name == "v"


def test_reset_index_numbering_again_leaves_a_column(firepanda):
    got = shuffled(firepanda).reset_index(drop=True)
    want = shuffled(pd).reset_index(drop=True)
    assert got.tolist() == want.tolist() == [10, 20, 30, 40]
    assert list(got.index) == list(want.index) == [0, 1, 2, 3]
    assert got.name == want.name == "v"


def test_reset_index_keeping_the_labels_makes_a_frame_of_two(firepanda):
    got = shuffled(firepanda).reset_index()
    want = shuffled(pd).reset_index()
    assert list(got.columns) == list(want.columns) == ["k", "v"]
    assert got["k"].tolist() == want["k"].tolist() == [2, 0, 3, 1]
    assert got["v"].tolist() == want["v"].tolist() == [10, 20, 30, 40]


def test_reset_index_takes_a_name_for_the_values(firepanda):
    got = shuffled(firepanda).reset_index(name="q")
    want = shuffled(pd).reset_index(name="q")
    assert list(got.columns) == list(want.columns) == ["k", "q"]


def test_the_door_is_open_on_a_column_of_instants_too(firepanda):
    import datetime

    import pyarrow as pa

    stamps = [
        datetime.datetime(2024, 1, 2),
        datetime.datetime(2024, 1, 1),
        datetime.datetime(2024, 1, 2),
    ]
    table = pa.table({"t": pa.array(stamps, type=pa.timestamp("us"))})
    made = firepanda.from_arrow(table)["t"]
    kept = made.drop_duplicates()
    assert kept.dtype == made.dtype
    assert len(kept) == 2
    assert list(pa.array(kept)) == list(pa.array(made))[:2]
