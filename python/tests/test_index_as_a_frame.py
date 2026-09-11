"""The index members that are a frame underneath, against pandas.

Two doors end to end. The labels become a column and the column becomes a
frame, and what that buys is `to_frame`, `duplicated` and `drop_duplicates` on
the index without any of the three being written a second time.
"""

from __future__ import annotations

import pandas as pd
import pytest


def repeats(module):
    """Six labels with repeats in them, under a name."""
    return module.Index([3, 1, 3, 2, 1, 4], name="k")


def test_to_frame_names_the_column_after_the_index(firepanda):
    got = repeats(firepanda).to_frame()
    want = repeats(pd).to_frame()
    assert list(got.columns) == list(want.columns) == ["k"]
    assert got["k"].tolist() == want["k"].tolist() == [3, 1, 3, 2, 1, 4]


def test_to_frame_keeps_the_labels_as_labels_unless_told_not_to(firepanda):
    assert list(repeats(firepanda).to_frame().index) == [3, 1, 3, 2, 1, 4]
    assert list(repeats(pd).to_frame().index) == [3, 1, 3, 2, 1, 4]
    got = repeats(firepanda).to_frame(index=False)
    want = repeats(pd).to_frame(index=False)
    assert list(got.index) == list(want.index) == [0, 1, 2, 3, 4, 5]
    assert got["k"].tolist() == want["k"].tolist()


def test_to_frame_takes_a_name_of_the_callers_choosing(firepanda):
    got = repeats(firepanda).to_frame(name="other")
    want = repeats(pd).to_frame(name="other")
    assert list(got.columns) == list(want.columns) == ["other"]


def test_an_unnamed_index_names_its_column_zero(firepanda):
    got = firepanda.Index([1, 2]).to_frame()
    # The integer zero in pandas and the text of it here, which is the same
    # difference `Series.to_frame` has and for the same reason.
    assert list(got.columns) == ["0"]
    assert [str(one) for one in pd.Index([1, 2]).to_frame().columns] == ["0"]


def test_duplicated_answers_a_list_of_bools(firepanda):
    got = repeats(firepanda).duplicated()
    want = repeats(pd).duplicated()
    assert got == list(want) == [False, False, True, False, True, False]


@pytest.mark.parametrize("keep", ["first", "last", False])
def test_duplicated_settles_the_ties_the_way_it_is_told_to(firepanda, keep):
    assert repeats(firepanda).duplicated(keep=keep) == list(repeats(pd).duplicated(keep=keep))


@pytest.mark.parametrize("keep", ["first", "last", False])
def test_drop_duplicates_removes_them_and_keeps_the_name(firepanda, keep):
    got = repeats(firepanda).drop_duplicates(keep=keep)
    want = repeats(pd).drop_duplicates(keep=keep)
    assert got.tolist() == list(want)
    assert got.name == want.name == "k"


def test_drop_duplicates_hands_back_the_class_it_was_given(firepanda):
    made = firepanda.DatetimeIndex(["2024-01-02", "2024-01-01", "2024-01-02"], name="t")
    kept = made.drop_duplicates()
    assert isinstance(kept, firepanda.DatetimeIndex)
    assert kept.name == "t"
    assert len(kept) == 2
