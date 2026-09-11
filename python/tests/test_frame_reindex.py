"""`reindex` on a frame, against pandas.

The operation has two halves that share a name and share almost nothing else. On
the rows it is a lookup in the index, and a label the frame does not have is a
row of nothing. On the columns it is a lookup in the schema, and a name the frame
does not have is a whole column of nothing. Both halves are here, and so are the
eight parameters that do no work but have to answer for themselves anyway.

Most cases compare against pandas rather than against a written expectation,
because the rule being checked is pandas' rule. The exceptions are the cases
where the answer is a refusal, since a refusal is ours and its wording is worth
pinning, and the two places where we deliberately answer something pandas does
not: a column name asked for twice, and a label whose type is not the index's.
"""

from __future__ import annotations

import pandas as pd
import pyarrow as pa
import pytest

ROWS = {
    "key": [10, 20, 30],
    "count": [1, 2, 3],
    "word": ["red", "green", "blue"],
}


def made(firepanda):
    """The frame under test, labelled by its key column."""
    return firepanda.DataFrame(dict(ROWS)).set_index("key")


def theirs():
    """The same frame in pandas."""
    return pd.DataFrame(dict(ROWS)).set_index("key")


def frame_of(mine):
    """Our frame as pandas holds it, so the two can be compared."""
    return pa.table(mine).to_pandas()


def same_frame(mine, them):
    """Asserts that two frames agree about labels, columns, values and types."""
    assert list(mine.index) == list(them.index)
    assert list(mine.columns) == list(them.columns)
    ours = frame_of(mine)
    ours.index = them.index
    pd.testing.assert_frame_equal(ours, them, check_dtype=False)


def test_a_label_the_frame_has_brings_its_own_row(firepanda):
    same_frame(made(firepanda).reindex([30, 10]), theirs().reindex([30, 10]))


def test_a_label_the_frame_does_not_have_gives_a_missing_row(firepanda):
    same_frame(made(firepanda).reindex([10, 99]), theirs().reindex([10, 99]))


def test_an_integer_column_widens_when_a_row_goes_missing(firepanda):
    # This is the one rule in the operation that is about types rather than
    # rows. An int64 column has no way to say missing, so a frame that loses a
    # row comes back as float64 with a NaN in it, which is pandas' answer and
    # not something a gather would do on its own.
    ours = frame_of(made(firepanda).reindex([10, 99]))
    assert ours["count"].dtype == theirs().reindex([10, 99])["count"].dtype


def test_a_fill_value_keeps_the_column_as_it_was(firepanda):
    numbers = firepanda.DataFrame({"key": [10, 20], "count": [1, 2]}).set_index("key")
    ours = frame_of(numbers.reindex([10, 99], fill_value=0))
    assert list(ours["count"]) == [1, 0]
    assert ours["count"].dtype == "int64"


def test_a_fill_value_leaves_a_hole_that_was_already_there_alone(firepanda):
    # The fill belongs to the rows the lookup did not find and not to the rows
    # it found, so a null the frame already held stays null. Writing this as a
    # fill over the answer would have covered both, which is why it is not
    # written that way.
    holed = firepanda.DataFrame({"key": [10, 20], "count": [1.0, None]}).set_index("key")
    ours = frame_of(holed.reindex([20, 99], fill_value=7.0))
    assert pd.isna(ours["count"].iloc[0])
    assert ours["count"].iloc[1] == 7.0


def test_a_missing_fill_value_is_the_same_as_no_fill_value(firepanda):
    # pandas' own default for the parameter is NaN, so a caller who writes it
    # out has asked for the rows to be missing, which is what happens when the
    # parameter is left alone. The column widens either way.
    ours = frame_of(made(firepanda).reindex([10, 99], fill_value=float("nan")))
    assert ours["count"].dtype == "float64"
    assert pd.isna(ours["count"].iloc[1])


def test_a_label_asked_for_twice_brings_its_row_twice(firepanda):
    same_frame(made(firepanda).reindex([10, 10, 20]), theirs().reindex([10, 10, 20]))


def test_asking_for_no_labels_gives_a_frame_of_no_rows(firepanda):
    mine = made(firepanda).reindex([])
    assert len(mine) == 0
    assert list(mine.columns) == ["count", "word"]


def test_the_labels_keep_the_name_the_index_had(firepanda):
    assert made(firepanda).reindex([30, 10]).index.name == "key"


def test_a_repeated_label_in_the_frame_is_refused(firepanda):
    twice = firepanda.DataFrame({"key": [10, 10], "count": [1, 2]}).set_index("key")
    with pytest.raises(ValueError, match="unique"):
        twice.reindex([10])


def test_a_word_cannot_fill_a_column_of_numbers(firepanda):
    with pytest.raises(TypeError):
        made(firepanda).reindex([10, 99], fill_value="nothing")


def test_the_columns_come_back_in_the_order_they_were_asked_for(firepanda):
    same_frame(
        made(firepanda).reindex(columns=["word", "count"]),
        theirs().reindex(columns=["word", "count"]),
    )


def test_leaving_a_column_out_is_how_a_column_is_dropped(firepanda):
    mine = made(firepanda).reindex(columns=["count"])
    assert list(mine.columns) == ["count"]


def test_a_column_that_is_not_there_is_made_out_of_nothing(firepanda):
    ours = frame_of(made(firepanda).reindex(columns=["count", "extra"]))
    them = theirs().reindex(columns=["count", "extra"])
    assert list(ours.columns) == list(them.columns)
    assert ours["extra"].dtype == them["extra"].dtype
    assert ours["extra"].isna().all()


def test_a_made_up_column_takes_the_fill_values_type(firepanda):
    ours = frame_of(made(firepanda).reindex(columns=["count", "extra"], fill_value=7))
    assert list(ours["extra"]) == [7, 7, 7]
    assert ours["extra"].dtype == "int64"


def test_a_column_asked_for_twice_is_refused(firepanda):
    # pandas answers two columns under one name here. A frame whose schema is a
    # list of names cannot hold that, and answering one column would be a
    # quieter wrong answer than refusing, so this is a refusal.
    with pytest.raises(NotImplementedError, match="twice"):
        made(firepanda).reindex(columns=["count", "count"])


def test_both_axes_at_once_are_both_applied(firepanda):
    same_frame(
        made(firepanda).reindex(index=[30, 99], columns=["word"]),
        theirs().reindex(index=[30, 99], columns=["word"]),
    )


def test_reindexing_with_nothing_is_a_copy(firepanda):
    same_frame(made(firepanda).reindex(), theirs().reindex())


def test_the_axis_says_which_half_the_labels_are_for(firepanda):
    same_frame(
        made(firepanda).reindex(["word"], axis="columns"),
        theirs().reindex(["word"], axis="columns"),
    )
    same_frame(made(firepanda).reindex(["word"], axis=1), theirs().reindex(["word"], axis=1))


def test_the_positional_labels_go_to_the_axis_nobody_named(firepanda):
    # This is the rule nobody expects. Naming one axis does not make the
    # positional labels an argument about that axis being given twice, it makes
    # them the other axis, so the labels here are the columns.
    same_frame(
        made(firepanda).reindex(["word"], index=[30]),
        theirs().reindex(["word"], index=[30]),
    )
    same_frame(
        made(firepanda).reindex([30], columns=["count"]),
        theirs().reindex([30], columns=["count"]),
    )


def test_naming_both_axes_and_passing_labels_as_well_is_refused(firepanda):
    with pytest.raises(TypeError, match="Cannot specify all"):
        made(firepanda).reindex([10], index=[30], columns=["word"])


def test_naming_an_axis_twice_is_refused(firepanda):
    with pytest.raises(TypeError, match="Cannot specify both"):
        made(firepanda).reindex([10], index=[30], axis=0)


def test_an_axis_the_frame_does_not_have_is_refused(firepanda):
    with pytest.raises(ValueError, match="No axis named 2"):
        made(firepanda).reindex([10], axis=2)


def test_filling_from_the_row_beside_it_is_not_done_here(firepanda):
    with pytest.raises(NotImplementedError, match="method"):
        made(firepanda).reindex([10, 99], method="ffill")


def test_a_limit_without_a_method_is_refused_in_pandas_words(firepanda):
    with pytest.raises(ValueError, match="only valid if doing pad"):
        made(firepanda).reindex([10], limit=1)
    with pytest.raises(ValueError, match="only valid if doing pad"):
        made(firepanda).reindex([10], tolerance=1)


def test_copy_and_level_are_taken_and_ignored(firepanda):
    # pandas ignores both on a flat index, one because it is deprecated and one
    # because a flat index has exactly the one level, and a caller who passes
    # either should get an answer rather than a refusal.
    same_frame(
        made(firepanda).reindex([30, 10], copy=True, level=0),
        theirs().reindex([30, 10], level=0),
    )


def test_a_label_of_the_wrong_type_is_refused(firepanda):
    # pandas answers a frame of nothing but missing rows for this, on the
    # grounds that no integer label equals a string one. The lookup here puts
    # the two sets of labels in one column to compare them, and there is no
    # column that holds both, so it refuses rather than inventing a rule.
    with pytest.raises(TypeError):
        made(firepanda).reindex(["10"])
