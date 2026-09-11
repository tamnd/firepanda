"""`reindex` on a series, against pandas.

The row half of the frame's method with one column under it, so most of these
tests are the frame's tests asked again. That is on purpose. A second
implementation of a rule is a second place for the rule to be wrong, and the two
rules most worth asking twice are that an integer column widens when a row goes
missing and that a fill value belongs to the rows the lookup did not find rather
than to every hole in the answer.

What is different from the frame is the parameter list. There is no `columns`,
there is no `labels` to route, and `axis` is taken and ignored rather than
deciding anything, since a series has one axis and naming it is not a choice.
"""

from __future__ import annotations

import pandas as pd
import pytest

VALUES = [1, 2, 3]
LABELS = [10, 20, 30]


def made(firepanda):
    """The series under test, labelled ten, twenty and thirty."""
    frame = firepanda.DataFrame({"key": LABELS, "count": VALUES}).set_index("key")
    return frame["count"]


def theirs():
    """The same series in pandas."""
    return pd.DataFrame({"key": LABELS, "count": VALUES}).set_index("key")["count"]


def same_series(mine, them):
    """Asserts that two series agree about labels, values and name."""
    assert list(mine.index) == list(them.index)
    assert mine.name == them.name
    ours = mine.tolist()
    assert len(ours) == len(them)
    for got, want in zip(ours, list(them), strict=True):
        if pd.isna(want):
            assert got is None or pd.isna(got)
        else:
            assert got == want


def test_a_label_the_series_has_brings_its_own_row(firepanda):
    same_series(made(firepanda).reindex([30, 10]), theirs().reindex([30, 10]))


def test_a_label_the_series_does_not_have_gives_a_missing_row(firepanda):
    same_series(made(firepanda).reindex([10, 99]), theirs().reindex([10, 99]))


def test_an_integer_series_widens_when_a_row_goes_missing(firepanda):
    # The rule that makes this operation change a type at all. pandas has one
    # missing value for a number and it is a NaN, so a column with nowhere to
    # put one becomes a column that has somewhere.
    assert made(firepanda).reindex([10, 20]).dtype == "int64"
    assert made(firepanda).reindex([10, 99]).dtype == "float64"


def test_a_fill_value_keeps_the_series_as_it_was(firepanda):
    got = made(firepanda).reindex([10, 99], fill_value=0)
    assert got.dtype == "int64"
    assert got.tolist() == [1, 0]


def test_a_fill_value_leaves_a_hole_that_was_already_there_alone(firepanda):
    # The fill belongs to the rows the lookup did not find, so a null the series
    # already held stays a null. Writing this as a fill over the answer would
    # have covered both, which is why it is not written that way.
    holed = firepanda.DataFrame({"key": [10, 20], "count": [1.0, None]}).set_index("key")
    got = holed["count"].reindex([20, 99], fill_value=7.0)
    assert pd.isna(got.tolist()[0])
    assert got.tolist()[1] == 7.0


def test_a_fill_value_of_nan_is_the_same_as_no_fill_value(firepanda):
    # pandas' own default here is None rather than the NaN the frame's method
    # defaults to, and both of them mean leave the row missing.
    got = made(firepanda).reindex([10, 99], fill_value=float("nan"))
    assert got.dtype == "float64"
    assert pd.isna(got.tolist()[1])


def test_a_label_asked_for_twice_brings_its_row_twice(firepanda):
    same_series(made(firepanda).reindex([10, 10, 20]), theirs().reindex([10, 10, 20]))


def test_asking_for_no_labels_gives_a_series_of_no_rows(firepanda):
    got = made(firepanda).reindex([])
    assert len(got) == 0
    assert got.dtype == "int64"


def test_the_labels_keep_the_name_the_index_had(firepanda):
    assert made(firepanda).reindex([30, 10]).index.name == "key"


def test_the_series_keeps_the_name_it_had(firepanda):
    assert made(firepanda).reindex([10, 99]).name == "count"


def test_reindexing_with_nothing_is_a_copy(firepanda):
    same_series(made(firepanda).reindex(), theirs().reindex())


def test_a_repeated_label_in_the_series_is_refused(firepanda):
    twice = firepanda.DataFrame({"key": [10, 10], "count": [1, 2]}).set_index("key")
    with pytest.raises(ValueError, match="cannot reindex on an axis with duplicate labels"):
        twice["count"].reindex([10])


def test_a_word_cannot_fill_a_series_of_numbers(firepanda):
    with pytest.raises(TypeError):
        made(firepanda).reindex([10, 99], fill_value="nothing")


def test_the_axis_is_taken_and_ignored(firepanda):
    # A series has one axis, so naming it decides nothing, and pandas accepts
    # any value here without checking it against the axes it has.
    same_series(made(firepanda).reindex([30], axis=0), theirs().reindex([30], axis=0))
    same_series(made(firepanda).reindex([30], axis=1), theirs().reindex([30], axis=1))


def test_copy_and_level_are_taken_and_ignored(firepanda):
    same_series(
        made(firepanda).reindex([30, 10], copy=True, level=0),
        theirs().reindex([30, 10], level=0),
    )


def test_filling_from_the_row_beside_it_is_not_done_here(firepanda):
    with pytest.raises(NotImplementedError, match="method"):
        made(firepanda).reindex([10, 99], method="ffill")


def test_a_limit_without_a_method_is_refused_in_pandas_words(firepanda):
    with pytest.raises(ValueError, match="only valid if doing pad"):
        made(firepanda).reindex([10], limit=1)
    with pytest.raises(ValueError, match="only valid if doing pad"):
        made(firepanda).reindex([10], tolerance=1)


def test_a_label_of_the_wrong_type_is_refused(firepanda):
    # pandas answers a series of nothing but missing rows here, on the grounds
    # that no integer label equals a string one. The lookup puts both sets of
    # labels in one column to compare them and there is no column that holds
    # both, so it refuses rather than inventing a rule about which types are
    # comparable with which. Document 40 section 7 has the price of that.
    with pytest.raises(TypeError):
        made(firepanda).reindex(["10"])
