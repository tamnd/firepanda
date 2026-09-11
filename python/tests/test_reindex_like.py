"""`reindex_like` on a frame and on a series, against pandas.

`reindex` with the labels read off something else rather than written out, which
sounds like a convenience and is not quite one. The thing it does that a caller
cannot do for themselves is carry the other object's index name across, because
taking the labels out and passing them as a list leaves the answer under the name
it already had.

The frame version does both axes and therefore needs something with two of them,
which is why pandas refuses a series there and accepts one here. That asymmetry
is the only real difference between the two methods and most of these tests are
about it.
"""

from __future__ import annotations

import pandas as pd
import pytest


def mine(firepanda):
    """A frame labelled ten, twenty and thirty, with three columns."""
    return firepanda.DataFrame(
        {"key": [10, 20, 30], "count": [1, 2, 3], "word": ["red", "green", "blue"]}
    ).set_index("key")


def shape(firepanda):
    """A frame labelled twenty, thirty and forty, holding one column the frame
    above has and one it does not."""
    return firepanda.DataFrame(
        {"other": [20, 30, 40], "count": [0, 0, 0], "extra": [0, 0, 0]}
    ).set_index("other")


def theirs():
    """The same pair in pandas."""
    first = pd.DataFrame(
        {"key": [10, 20, 30], "count": [1, 2, 3], "word": ["red", "green", "blue"]}
    ).set_index("key")
    second = pd.DataFrame(
        {"other": [20, 30, 40], "count": [0, 0, 0], "extra": [0, 0, 0]}
    ).set_index("other")
    return first, second


def test_the_labels_and_the_columns_both_come_from_the_other_frame(firepanda):
    got = mine(firepanda).reindex_like(shape(firepanda))
    first, second = theirs()
    want = first.reindex_like(second)
    assert list(got.index) == list(want.index)
    assert list(got.columns) == list(want.columns)


def test_the_answer_is_labelled_the_way_the_other_frame_is(firepanda):
    # The reason this is a method rather than two keyword arguments. A caller
    # who took the labels out and passed them as a list would get the answer
    # back under the name it already had.
    got = mine(firepanda).reindex_like(shape(firepanda))
    first, second = theirs()
    assert got.index.name == "other"
    assert first.reindex_like(second).index.name == "other"


def test_a_row_the_other_frame_has_and_this_one_does_not_goes_missing(firepanda):
    got = mine(firepanda).reindex_like(shape(firepanda))
    assert got["count"].tolist()[:2] == [2, 3]
    assert pd.isna(got["count"].tolist()[2])


def test_a_column_that_loses_a_row_widens_the_way_it_does_everywhere(firepanda):
    # There is no fill_value here, pandas does not offer one, so the widening
    # rule has nothing to stop it.
    got = mine(firepanda).reindex_like(shape(firepanda))
    assert got["count"].dtype == "float64"
    first, second = theirs()
    assert first.reindex_like(second)["count"].dtype == "float64"


def test_a_column_the_other_frame_has_and_this_one_does_not_is_made(firepanda):
    got = mine(firepanda).reindex_like(shape(firepanda))
    assert got["extra"].dtype == "float64"
    assert all(pd.isna(one) for one in got["extra"].tolist())


def test_a_column_this_frame_has_and_the_other_does_not_is_dropped(firepanda):
    got = mine(firepanda).reindex_like(shape(firepanda))
    assert "word" not in list(got.columns)


def test_a_frame_shaped_like_itself_comes_back_as_it_was(firepanda):
    got = mine(firepanda).reindex_like(mine(firepanda))
    assert list(got.index) == [10, 20, 30]
    assert got["count"].tolist() == [1, 2, 3]
    assert list(got.columns) == ["count", "word"]


def test_a_frame_cannot_be_shaped_like_a_series(firepanda):
    # pandas refuses this and the sentence is theirs: a series has no axis
    # named columns, and the frame's version of the method reindexes both.
    with pytest.raises(ValueError, match="No axis named columns"):
        mine(firepanda).reindex_like(mine(firepanda)["count"])
    first, _ = theirs()
    with pytest.raises(ValueError, match="No axis named columns"):
        first.reindex_like(first["count"])


def test_a_series_takes_the_labels_of_a_series(firepanda):
    got = mine(firepanda)["count"].reindex_like(shape(firepanda)["count"])
    first, second = theirs()
    want = first["count"].reindex_like(second["count"])
    assert list(got.index) == list(want.index)
    assert got.tolist()[:2] == [2, 3]
    assert pd.isna(got.tolist()[2])


def test_a_series_takes_the_labels_of_a_frame(firepanda):
    # A series has one axis, so there is nothing to disagree about and a frame
    # is a perfectly good thing to take labels from.
    got = mine(firepanda)["count"].reindex_like(shape(firepanda))
    assert list(got.index) == [20, 30, 40]


def test_a_series_keeps_its_own_name_and_takes_the_other_index_name(firepanda):
    got = mine(firepanda)["count"].reindex_like(shape(firepanda))
    assert got.name == "count"
    assert got.index.name == "other"
    first, second = theirs()
    want = first["count"].reindex_like(second)
    assert want.name == "count"
    assert want.index.name == "other"


def test_filling_from_the_row_beside_it_is_not_done_here(firepanda):
    with pytest.raises(NotImplementedError, match="method"):
        mine(firepanda).reindex_like(shape(firepanda), method="ffill")
    with pytest.raises(NotImplementedError, match="method"):
        mine(firepanda)["count"].reindex_like(shape(firepanda), method="ffill")


def test_a_limit_without_a_method_is_refused_in_pandas_words(firepanda):
    with pytest.raises(ValueError, match="only valid if doing pad"):
        mine(firepanda).reindex_like(shape(firepanda), limit=1)
    with pytest.raises(ValueError, match="only valid if doing pad"):
        mine(firepanda)["count"].reindex_like(shape(firepanda), tolerance=1)


def test_copy_is_taken_and_ignored(firepanda):
    got = mine(firepanda).reindex_like(shape(firepanda), copy=True)
    assert list(got.index) == [20, 30, 40]


def test_a_repeated_label_in_this_frame_is_refused(firepanda):
    twice = firepanda.DataFrame({"key": [10, 10], "count": [1, 2]}).set_index("key")
    with pytest.raises(ValueError, match="cannot reindex on an axis with duplicate labels"):
        twice.reindex_like(shape(firepanda))
    with pytest.raises(ValueError, match="cannot reindex on an axis with duplicate labels"):
        twice["count"].reindex_like(shape(firepanda))


def test_something_with_no_labels_on_it_is_refused(firepanda):
    with pytest.raises(TypeError):
        mine(firepanda)["count"].reindex_like([10, 20])
