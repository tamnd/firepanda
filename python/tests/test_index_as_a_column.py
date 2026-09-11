"""The index members that are a column underneath, against pandas.

`to_series` is the door and the rest of this file walks through it. An index
that can become a column gets the column's reductions and the column's
transforms without a second copy of any of them, so what is worth testing is
that the door hands over the right labels, the right names and the right shape,
and that the members built on it answer what pandas answers.
"""

from __future__ import annotations

import pandas as pd
import pytest


def whole(module):
    """Four whole numbers under a name, with nothing missing."""
    return module.Index([3, 1, 4, 2], name="k")


def gappy(module):
    """Four labels with one of them missing, which is the interesting case."""
    return module.Index([3, 1, None, 2], name="k")


def test_to_series_carries_the_labels_as_values_and_as_labels(firepanda):
    got = whole(firepanda).to_series()
    want = whole(pd).to_series()
    assert got.tolist() == want.tolist() == [3, 1, 4, 2]
    assert list(got.index) == list(want.index) == [3, 1, 4, 2]


def test_to_series_takes_the_name_off_the_index(firepanda):
    assert whole(firepanda).to_series().name == whole(pd).to_series().name == "k"


def test_an_unnamed_index_gives_a_column_named_with_the_empty_string(firepanda):
    # pandas leaves the series unnamed and a series here is named by a string,
    # so the name that means unnamed over there is the empty string over here.
    # Document 21 has the divergence, and the values are the same either way.
    got = firepanda.Index([1, 2]).to_series()
    assert got.name == ""
    assert pd.Index([1, 2]).to_series().name is None
    assert got.tolist() == pd.Index([1, 2]).to_series().tolist() == [1, 2]


def test_a_name_in_the_call_wins_over_the_index_name(firepanda):
    got = whole(firepanda).to_series(name="v")
    assert got.name == whole(pd).to_series(name="v").name == "v"


def test_labels_in_the_call_win_over_the_index(firepanda):
    labels = firepanda.Index([10, 20, 30, 40])
    got = whole(firepanda).to_series(index=labels)
    want = whole(pd).to_series(index=pd.Index([10, 20, 30, 40]))
    assert list(got.index) == list(want.index) == [10, 20, 30, 40]
    assert got.tolist() == want.tolist() == [3, 1, 4, 2]


def test_labels_may_be_written_as_a_plain_list(firepanda):
    got = whole(firepanda).to_series(index=[10, 20, 30, 40])
    assert list(got.index) == [10, 20, 30, 40]


def test_labels_of_the_wrong_length_are_refused(firepanda):
    with pytest.raises(ValueError, match="no row for each of them"):
        whole(firepanda).to_series(index=[1, 2])


def test_isna_finds_the_missing_label(firepanda):
    got = gappy(firepanda).isna()
    assert got == list(gappy(pd).isna()) == [False, False, True, False]
    assert gappy(firepanda).isnull() == got


def test_notna_is_isna_turned_over(firepanda):
    got = gappy(firepanda).notna()
    assert got == list(gappy(pd).notna()) == [True, True, False, True]
    assert gappy(firepanda).notnull() == got


def test_an_index_with_nothing_missing_says_so(firepanda):
    assert whole(firepanda).isna() == [False] * 4
    assert whole(firepanda).notna() == [True] * 4


def test_a_range_has_no_missing_label(firepanda):
    frame = firepanda.DataFrame({"a": [1, 2, 3]})
    assert frame.index.isna() == [False, False, False]


def test_dropna_takes_the_missing_label_out(firepanda):
    got = gappy(firepanda).dropna()
    assert got.tolist() == gappy(pd).dropna().tolist() == [3, 1, 2]


def test_dropna_keeps_the_name(firepanda):
    assert gappy(firepanda).dropna().name == gappy(pd).dropna().name == "k"


def test_dropna_on_an_index_with_nothing_missing_changes_nothing(firepanda):
    assert whole(firepanda).dropna().tolist() == [3, 1, 4, 2]


def test_the_two_words_dropna_takes_mean_the_same_thing_here(firepanda):
    # A flat label is one value, so there is nothing for `any` and `all` to
    # disagree about. The parameter is here because a MultiIndex row is several.
    assert gappy(firepanda).dropna(how="all").tolist() == gappy(pd).dropna(how="all").tolist()


def test_a_third_word_is_not_a_way_to_drop(firepanda):
    with pytest.raises(ValueError, match="invalid how option: some"):
        gappy(firepanda).dropna(how="some")
    with pytest.raises(ValueError, match="invalid how option: some"):
        gappy(pd).dropna(how="some")


def test_dropna_hands_back_the_class_it_was_given(firepanda):
    index = firepanda.DatetimeIndex(["2024-01-02", "2024-03-04"])
    assert isinstance(index.dropna(), firepanda.DatetimeIndex)
    assert isinstance(pd.DatetimeIndex(["2024-01-02", "2024-03-04"]).dropna(), pd.DatetimeIndex)


def test_min_and_max_read_the_ends(firepanda):
    assert whole(firepanda).min() == whole(pd).min() == 1
    assert whole(firepanda).max() == whole(pd).max() == 4


def test_min_and_max_step_over_a_missing_label(firepanda):
    assert gappy(firepanda).min() == gappy(pd).min() == 1
    assert gappy(firepanda).max() == gappy(pd).max() == 3


def test_the_axis_an_index_has_is_the_only_one_it_takes(firepanda):
    assert whole(firepanda).min(axis=0) == 1
    assert whole(firepanda).min(axis=-1) == 1
    with pytest.raises(ValueError, match="must be fewer than the number of dimensions"):
        whole(firepanda).max(axis=1)
    with pytest.raises(ValueError, match="must be fewer than the number of dimensions"):
        whole(pd).max(axis=1)


def test_the_numpy_arguments_are_refused_rather_than_dropped(firepanda):
    with pytest.raises(NotImplementedError, match="numpy compatibility arguments"):
        whole(firepanda).min(None, True, "something")


def test_min_and_max_read_the_ends_of_words_too(firepanda):
    index = firepanda.Index(["pear", "apple", "quince"])
    theirs = pd.Index(["pear", "apple", "quince"])
    assert index.min() == theirs.min() == "apple"
    assert index.max() == theirs.max() == "quince"


def test_an_index_with_no_labels_has_no_smallest_one(firepanda):
    got = firepanda.Index([]).min()
    assert got != got
    want = pd.Index([]).min()
    assert want != want


def test_nunique_counts_the_distinct_labels(firepanda):
    index = firepanda.Index([3, 1, 3, 2])
    theirs = pd.Index([3, 1, 3, 2])
    assert index.nunique() == theirs.nunique() == 3


def test_a_missing_label_is_a_distinct_one_when_it_is_asked_to_be(firepanda):
    assert gappy(firepanda).nunique() == gappy(pd).nunique() == 3
    assert gappy(firepanda).nunique(dropna=False) == gappy(pd).nunique(dropna=False) == 4


def test_the_door_is_open_on_a_datetime_index_too(firepanda):
    # The values are compared against the index's own labels rather than
    # against pandas', because an instant lists as the whole number it is
    # stored as on both sides of this door and as a Timestamp in pandas. That
    # is issue #348 and document 33 has it. Nothing about it is new here, and
    # what this is checking is that the type survives the trip.
    index = firepanda.DatetimeIndex(["2024-03-04", "2024-01-02"])
    theirs = pd.DatetimeIndex(["2024-03-04", "2024-01-02"])
    assert index.to_series().dtype == index.dtype == "datetime64[us]"
    assert index.to_series().tolist() == index.tolist()
    assert index.nunique() == theirs.nunique() == 2
    assert index.isna() == [False, False]
