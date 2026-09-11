"""`duplicated` and `drop_duplicates` on a frame, against pandas.

The two are one question asked twice. `duplicated` hands back the mask and
`drop_duplicates` hands back the frame the mask selects, so a test that passes
for one and fails for the other means they disagree about which row of a
repeated key is the one that is not a repeat, which is the only interesting
thing either of them decides.

The frames here are small and written out, because what is being checked is a
rule and not a volume. Every case runs both libraries on the same values and
compares, since the rule is pandas' and a hand written expectation would be a
second opinion about it rather than a check of it.
"""

from __future__ import annotations

import pandas as pd
import pyarrow as pa
import pytest

ROWS = {
    "key": ["a", "b", "a", "c", "b", "a"],
    "grp": [1, 1, 1, 2, 2, 3],
    "val": [10, 20, 30, 40, 50, 60],
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


def same_mask(mine, them):
    """Asserts that two masks agree about the rows and about the labels."""
    assert mine.tolist() == them.tolist()
    assert list(mine.index) == list(them.index)
    assert mine.dtype == "bool"


def test_the_mask_marks_every_repeat_after_the_first(firepanda):
    same_mask(made(firepanda).duplicated(), theirs().duplicated())


def test_the_mask_reads_one_column_when_it_is_given_one(firepanda):
    same_mask(made(firepanda).duplicated("key"), theirs().duplicated("key"))


def test_the_mask_reads_several_columns_as_one_key(firepanda):
    wanted = ["key", "grp"]
    same_mask(made(firepanda).duplicated(wanted), theirs().duplicated(wanted))


def test_keeping_the_last_marks_everything_before_it(firepanda):
    same_mask(
        made(firepanda).duplicated("key", keep="last"),
        theirs().duplicated("key", keep="last"),
    )


def test_keeping_none_marks_the_original_too(firepanda):
    same_mask(
        made(firepanda).duplicated("key", keep=False),
        theirs().duplicated("key", keep=False),
    )


def test_a_missing_value_repeats_another_missing_value(firepanda):
    # pandas compares a missing value against a missing value and answers that
    # they are the same row, which is the one place in the library where two
    # nulls are equal to each other. A frame of keys where half are absent is
    # the case that separates a correct reading of that from a lucky one.
    rows = {"key": [1.0, None, 2.0, None, 1.0]}
    same_mask(firepanda.DataFrame(dict(rows)).duplicated(), pd.DataFrame(dict(rows)).duplicated())


def test_the_mask_is_as_tall_as_the_frame(firepanda):
    df = made(firepanda)
    assert len(df.duplicated()) == len(df)


def test_the_mask_has_no_name(firepanda):
    # pandas leaves it unnamed, because the answer is about the rows rather than
    # a column of the frame. A mask that carried a name would read like a column
    # somebody had added.
    assert made(firepanda).duplicated().name == ""
    assert theirs().duplicated().name is None


def test_the_drop_keeps_the_first_appearance_of_each_key(firepanda):
    same_frame(made(firepanda).drop_duplicates("key"), theirs().drop_duplicates("key"))


def test_the_drop_with_no_subset_reads_every_column(firepanda):
    same_frame(made(firepanda).drop_duplicates(), theirs().drop_duplicates())


def test_the_drop_keeping_the_last_takes_the_other_end_of_each_group(firepanda):
    same_frame(
        made(firepanda).drop_duplicates("key", keep="last"),
        theirs().drop_duplicates("key", keep="last"),
    )


def test_the_drop_keeping_none_leaves_the_rows_that_never_repeated(firepanda):
    same_frame(
        made(firepanda).drop_duplicates("key", keep=False),
        theirs().drop_duplicates("key", keep=False),
    )


def test_the_drop_keeps_the_labels_the_surviving_rows_had(firepanda):
    # The labels of what is left have gaps in them wherever a row went, which is
    # the thing `ignore_index` exists to undo and is worth asserting on its own
    # so that a frame which quietly renumbered would fail here rather than only
    # in the case that asks for the renumbering.
    assert list(made(firepanda).drop_duplicates("key").index) == [0, 1, 3]


def test_the_drop_can_number_the_rows_again(firepanda):
    same_frame(
        made(firepanda).drop_duplicates("key", ignore_index=True),
        theirs().drop_duplicates("key", ignore_index=True),
    )


def test_the_drop_and_the_mask_answer_the_same_rows(firepanda):
    # What makes them one piece of work rather than two. If the mask said one
    # thing and the drop did another, one of them would be wrong and neither
    # test above would say which.
    for rule in ("first", "last", False):
        df = made(firepanda)
        marked = df.duplicated("key", keep=rule).tolist()
        kept = df.drop_duplicates("key", keep=rule)
        assert list(kept.index) == [i for i, one in enumerate(marked) if not one], rule


def test_a_column_written_twice_is_read_once(firepanda):
    # pandas accepts the repeat and answers as if it were written once. The core
    # refuses it, on the grounds that a caller who wrote it meant something else,
    # so this is the layer where the two rules meet.
    same_mask(
        made(firepanda).duplicated(["key", "key"]),
        theirs().duplicated(["key", "key"]),
    )


def test_a_frame_with_nothing_in_it_answers_an_empty_mask(firepanda):
    mine = firepanda.DataFrame({}).duplicated()
    assert len(mine) == 0
    assert mine.dtype == "bool"
    assert len(pd.DataFrame({}).duplicated()) == 0


def test_a_frame_with_nothing_in_it_drops_nothing(firepanda):
    assert len(firepanda.DataFrame({}).drop_duplicates()) == 0


def test_a_rule_that_is_not_one_of_the_three_is_refused(firepanda):
    with pytest.raises(ValueError) as raised:
        made(firepanda).duplicated("key", keep="all")
    assert "keep must be either 'first', 'last' or False" in str(raised.value)
    with pytest.raises(ValueError):
        made(firepanda).drop_duplicates("key", keep="all")


def test_a_rule_written_as_a_zero_is_read_as_the_third_one(firepanda):
    # pandas checks `keep` by asking whether it is in a tuple holding the two
    # words and `False`, and `0 == False` in Python, so a zero is in that tuple
    # and pandas reads it as the third rule. A layer that checked with `is False`
    # would be tidier and would refuse a call its pandas accepts, so this case
    # holds the accident rather than the intent.
    same_mask(made(firepanda).duplicated("key", keep=0), theirs().duplicated("key", keep=0))
    with pytest.raises(ValueError):
        made(firepanda).duplicated("key", keep=True)
    with pytest.raises(ValueError):
        theirs().duplicated("key", keep=True)


def test_a_column_that_is_not_there_is_refused(firepanda):
    with pytest.raises(KeyError):
        made(firepanda).duplicated("nope")
    with pytest.raises(KeyError):
        made(firepanda).drop_duplicates("nope")


def test_dropping_in_place_is_refused_rather_than_ignored(firepanda):
    with pytest.raises(NotImplementedError) as raised:
        made(firepanda).drop_duplicates("key", inplace=True)
    assert "inplace" in str(raised.value)
