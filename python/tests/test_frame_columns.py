"""`filter`, `select_dtypes` and `truncate` on a frame, against pandas.

The three are one piece of work: each of them answers a set of labels computed
from a rule rather than written out, and each of them then does the narrowing
every other selection here already does. What is worth testing is the rule, so
every case runs both libraries and compares the labels that came back.

The frame has a column of every type firepanda has a name for, because
`select_dtypes` is about the type tree and a frame of two numeric columns
cannot tell a correct reading of that tree from a lucky one.
"""

from __future__ import annotations

import datetime as dt

import pandas as pd
import pyarrow as pa
import pytest

ROWS = {
    "alpha": [1, 2, 3],
    "beta": [1.5, 2.5, 3.5],
    "gamma": ["x", "y", "z"],
    "delta": [True, False, True],
}


def made(firepanda):
    """The frame under test."""
    return firepanda.DataFrame(dict(ROWS))


def theirs():
    """The same frame in pandas."""
    return pd.DataFrame(dict(ROWS))


def typed(firepanda):
    """A frame carrying one column of each type, for the type tree."""
    table = pa.table(
        {
            "i64": pa.array([1, 2, 3], type=pa.int64()),
            "i32": pa.array([1, 2, 3], type=pa.int32()),
            "u8": pa.array([1, 2, 3], type=pa.uint8()),
            "f64": pa.array([1.5, 2.5, 3.5], type=pa.float64()),
            "f32": pa.array([1.5, 2.5, 3.5], type=pa.float32()),
            "flag": pa.array([True, False, True], type=pa.bool_()),
            "text": pa.array(["x", "y", "z"], type=pa.large_string()),
            "when": pa.array([dt.datetime(2020, 1, 1)] * 3, type=pa.timestamp("ns")),
            "span": pa.array([dt.timedelta(seconds=1)] * 3, type=pa.duration("ns")),
        }
    )
    return firepanda.from_arrow(table), table.to_pandas()


def same(mine, them):
    """Asserts that two frames agree about labels, columns and values."""
    assert list(mine.index) == list(them.index)
    assert list(mine.columns) == list(them.columns)
    ours = pa.table(mine).to_pandas()
    ours.index = them.index
    pd.testing.assert_frame_equal(ours, them, check_dtype=False)


def test_filter_keeps_the_columns_a_substring_is_in(firepanda):
    same(made(firepanda).filter(like="a"), theirs().filter(like="a"))


def test_filter_keeps_the_columns_a_pattern_matches(firepanda):
    same(made(firepanda).filter(regex="^.e"), theirs().filter(regex="^.e"))


def test_filter_takes_items_in_the_order_they_were_written(firepanda):
    wanted = ["gamma", "alpha"]
    same(made(firepanda).filter(items=wanted), theirs().filter(items=wanted))


def test_filter_drops_an_item_that_is_not_there_rather_than_complaining(firepanda):
    wanted = ["beta", "nope", "alpha"]
    same(made(firepanda).filter(items=wanted), theirs().filter(items=wanted))


def test_filter_along_the_rows_reads_the_labels(firepanda):
    mine = made(firepanda).set_index("gamma")
    them = theirs().set_index("gamma")
    same(mine.filter(like="y", axis=0), them.filter(like="y", axis=0))
    same(mine.filter(items=["z", "x"], axis=0), them.filter(items=["z", "x"], axis=0))


def test_filter_refuses_two_rules_at_once(firepanda):
    with pytest.raises(TypeError) as raised:
        made(firepanda).filter(items=["alpha"], like="a")
    assert "mutually exclusive" in str(raised.value)


def test_filter_refuses_no_rule_at_all(firepanda):
    with pytest.raises(TypeError) as raised:
        made(firepanda).filter()
    assert "Must pass either" in str(raised.value)


def test_select_dtypes_takes_the_numbers_and_not_the_booleans(firepanda):
    mine, them = typed(firepanda)
    assert list(mine.select_dtypes(include="number").columns) == list(
        them.select_dtypes(include="number").columns
    )


def test_select_dtypes_reads_int_and_integer_as_different_words(firepanda):
    # numpy resolves a bare int to int64 and pandas widens it back out to the
    # two signed widths by hand, so `int` is narrower than `integer` and both
    # are narrower than `number`. Getting this wrong is invisible on a frame
    # that has only int64 in it.
    mine, them = typed(firepanda)
    for word in ("int", "integer", "signedinteger", "unsignedinteger", "float", "floating"):
        assert list(mine.select_dtypes(include=word).columns) == list(
            them.select_dtypes(include=word).columns
        ), word


def test_select_dtypes_counts_a_span_as_a_number(firepanda):
    # numpy makes timedelta64 a kind of signed integer and pandas inherits it.
    # It is surprising and it is what a caller's pandas does.
    mine, them = typed(firepanda)
    assert "span" in list(them.select_dtypes(include="number").columns)
    assert "span" in list(mine.select_dtypes(include="number").columns)


def test_select_dtypes_holds_back_what_exclude_names(firepanda):
    mine, them = typed(firepanda)
    assert list(mine.select_dtypes(exclude="number").columns) == list(
        them.select_dtypes(exclude="number").columns
    )
    assert list(mine.select_dtypes(include="number", exclude="floating").columns) == list(
        them.select_dtypes(include="number", exclude="floating").columns
    )


def test_select_dtypes_takes_a_list_of_words(firepanda):
    mine, them = typed(firepanda)
    assert list(mine.select_dtypes(include=["bool", "float64"]).columns) == list(
        them.select_dtypes(include=["bool", "float64"]).columns
    )


def test_select_dtypes_wants_at_least_one_side(firepanda):
    with pytest.raises(ValueError) as raised:
        made(firepanda).select_dtypes()
    assert "at least one of include or exclude must be nonempty" in str(raised.value)


def test_select_dtypes_refuses_a_word_on_both_sides(firepanda):
    with pytest.raises(ValueError) as raised:
        made(firepanda).select_dtypes(include="int64", exclude="int64")
    assert "include and exclude overlap on" in str(raised.value)


def test_select_dtypes_says_so_when_the_type_is_one_firepanda_has_not_got(firepanda):
    with pytest.raises(NotImplementedError):
        made(firepanda).select_dtypes(include="object")


def test_truncate_keeps_both_of_the_labels_it_was_given(firepanda):
    mine = firepanda.DataFrame({"v": list(range(10))})
    them = pd.DataFrame({"v": list(range(10))})
    same(mine.truncate(before=3, after=6), them.truncate(before=3, after=6))
    assert len(mine.truncate(before=3, after=6)) == 4


def test_truncate_takes_one_end_at_a_time(firepanda):
    mine = firepanda.DataFrame({"v": list(range(10))})
    them = pd.DataFrame({"v": list(range(10))})
    same(mine.truncate(before=7), them.truncate(before=7))
    same(mine.truncate(after=2), them.truncate(after=2))


def test_truncate_reads_a_falling_index_the_other_way_round(firepanda):
    mine = made(firepanda).set_index("alpha").sort_index(ascending=False)
    them = theirs().set_index("alpha").sort_index(ascending=False)
    same(mine.truncate(before=1, after=2), them.truncate(before=1, after=2))


def test_truncate_along_the_columns_cuts_the_names(firepanda):
    mine = made(firepanda).filter(items=sorted(ROWS))
    them = theirs().filter(items=sorted(ROWS))
    same(
        mine.truncate(before="beta", after="delta", axis=1),
        them.truncate(before="beta", after="delta", axis=1),
    )


def test_truncate_refuses_an_index_that_is_not_sorted(firepanda):
    # Sorted here means monotonic in either direction, so the frame has to be
    # genuinely shuffled rather than merely descending.
    rows = {"k": [3, 1, 2], "v": [1, 2, 3]}
    with pytest.raises(ValueError) as raised:
        firepanda.DataFrame(dict(rows)).set_index("k").truncate(before=1, after=2)
    assert "truncate requires a sorted index" in str(raised.value)
    with pytest.raises(ValueError):
        pd.DataFrame(dict(rows)).set_index("k").truncate(before=1, after=2)


def test_truncate_refuses_a_reversed_pair(firepanda):
    with pytest.raises(ValueError) as raised:
        firepanda.DataFrame({"v": list(range(10))}).truncate(before=6, after=3)
    assert "Truncate: 3 must be after 6" in str(raised.value)


def test_truncate_refuses_the_copy_keyword(firepanda):
    with pytest.raises(NotImplementedError):
        firepanda.DataFrame({"v": [1, 2]}).truncate(before=0, copy=True)
