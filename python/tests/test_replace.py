"""Swapping some values for others, which is a comparison and a pick per pair.

`replace` belongs with `where` and `clip` rather than with the arithmetic,
because it is made of the same two things. Every pair is a condition, which is
the rows holding the value being replaced, and an other side, which is the value
going in, so the whole method is document 48 run once per pair with the
condition written for the caller.

Three rules are worth reading before the tests. Every pair is judged against the
column as it arrived rather than against what the pair before it made, so a
value that has just been put in is not replaced again and two values can be
swapped for each other in one call. A value the column cannot hold matches no
rows rather than being refused, which is why replacing a word in a column of
numbers is quietly nothing. And a pair whose replacement the column cannot hold
is refused, because this library does not widen a column to make room, which is
the one rule that separates these answers from pandas'.

The mapping rules on a frame are the ones people get wrong, and they are section
5 of document 50. A mapping arriving on its own is a mapping of values unless
every value in it is a mapping of its own, and a mapping arriving beside a value
is a mapping of column names. That is why `df.replace({"a": 2})` does nothing at
all to a frame with a column called `a`.
"""

from __future__ import annotations

import pandas as pd
import pytest


def frame(module):
    """Two columns of numbers that share a value, so one pair reaches both."""
    return module.DataFrame({"a": [1, 2, 3], "b": [2, 3, 4]})


def labelled(module, labels, values):
    """A column carrying labels, built the way each library builds one."""
    if module is pd:
        return pd.Series(values, index=labels)
    return module.DataFrame({"k": labels, "v": values}).set_index("k")["v"]


def cells(made):
    """Every column of a frame as a list, for comparing two frames by value."""
    return {str(name): made[name].tolist() for name in made.columns}


def test_one_value_is_swapped_for_another(firepanda):
    assert firepanda.Series([1, 2, 3, 2]).replace(2, 9).tolist() == [1, 9, 3, 9]


def test_a_value_nothing_holds_is_quietly_nothing(firepanda):
    assert firepanda.Series([1, 2, 3]).replace(7, 9).tolist() == [1, 2, 3]


def test_a_run_of_values_is_read_against_a_run_of_replacements(firepanda):
    made = firepanda.Series([1, 2, 3, 2])
    assert made.replace([1, 2], [7, 8]).tolist() == [7, 8, 3, 8]


def test_a_run_of_values_all_going_to_one_place(firepanda):
    assert firepanda.Series([1, 2, 3]).replace([1, 2], 0).tolist() == [0, 0, 3]


def test_a_mapping_is_read_as_pairs(firepanda):
    made = firepanda.Series([1, 2, 3])
    assert made.replace({1: 7, 3: 8}).tolist() == [7, 2, 8]


def test_two_values_can_be_swapped_for_each_other(firepanda):
    """Every pair is judged against the column as it arrived, so nothing cascades."""
    made = firepanda.Series([1, 2, 3, 2])
    assert made.replace([1, 2], [2, 1]).tolist() == [2, 1, 3, 1]


def test_a_value_that_has_just_been_put_in_is_not_replaced_again(firepanda):
    made = firepanda.Series([1, 2, 3, 2])
    assert made.replace([1, 2], [2, 3]).tolist() == [2, 3, 3, 3]


def test_the_last_of_two_pairs_that_both_match_wins(firepanda):
    assert firepanda.Series([1, 2]).replace([1, 1], [7, 8]).tolist() == [8, 2]


def test_a_missing_value_is_named_by_nothing(firepanda):
    """Which makes this call `fillna`, and pandas reads it the same way."""
    made = firepanda.Series([1.0, None, 3.0])
    assert made.replace(float("nan"), 0.0).tolist() == [1.0, 0.0, 3.0]


def test_a_missing_value_is_left_alone_by_every_other_pair(firepanda):
    made = firepanda.Series([1.0, None, 3.0])
    assert made.replace(1.0, 0.0).tolist() == [0.0, None, 3.0]


def test_nothing_to_replace_is_nothing_done(firepanda):
    assert firepanda.Series([1, 2, 3]).replace([], 0).tolist() == [1, 2, 3]
    assert firepanda.Series([1, 2, 3]).replace({}).tolist() == [1, 2, 3]


def test_a_value_of_a_type_the_column_cannot_hold_matches_no_rows(firepanda):
    """Not a refusal, because nothing was asked of the column it cannot do."""
    assert firepanda.Series([1, 2, 3]).replace("zz", 9).tolist() == [1, 2, 3]


def test_the_labels_are_carried_through(firepanda):
    made = labelled(firepanda, ["x", "y", "z"], [1, 2, 3])
    assert made.replace(2, 9).tolist() == [1, 9, 3]
    assert list(made.replace(2, 9).index) == ["x", "y", "z"]


def test_a_column_of_words_is_replaced_by_words(firepanda):
    assert firepanda.Series(["a", "b"]).replace("a", "c").tolist() == ["c", "b"]


def test_a_column_of_flags_is_replaced_by_flags(firepanda):
    assert firepanda.Series([True, False]).replace(True, False).tolist() == [False, False]


def test_a_column_written_as_pairs_is_a_mapping(firepanda):
    """A column carries a label against every row, so pandas reads one as a mapping."""
    made = firepanda.DataFrame({"k": [1, 2], "v": [7, 8]}).set_index("k")["v"]
    assert firepanda.Series([1, 2, 3]).replace(made).tolist() == [7, 8, 3]


def test_the_original_is_untouched(firepanda):
    made = firepanda.Series([1, 2, 3])
    made.replace(2, 9)
    assert made.tolist() == [1, 2, 3]


def test_a_value_the_column_cannot_hold_is_refused(firepanda):
    """pandas widens the column here and this library does not, which is document 50."""
    with pytest.raises(firepanda.errors.DTypeError):
        firepanda.Series([1, 2, 3]).replace(2, 2.5)


def test_a_new_category_is_refused(firepanda):
    made = firepanda.Series(["a", "b"], dtype="category")
    assert made.replace("a", "b").tolist() == ["b", "b"]
    with pytest.raises(firepanda.errors.DTypeError):
        made.replace("a", "z")


def test_a_frame_is_the_column_once_per_column(firepanda):
    made = frame(firepanda).replace(2, 9)
    assert cells(made) == {"a": [1, 9, 3], "b": [9, 3, 4]}


def test_a_frame_reads_a_nested_mapping_by_column(firepanda):
    made = frame(firepanda).replace({"a": {2: 99}})
    assert cells(made) == {"a": [1, 99, 3], "b": [2, 3, 4]}


def test_a_frame_reads_a_mapping_beside_a_value_by_column(firepanda):
    made = frame(firepanda).replace({"a": 2}, 5)
    assert cells(made) == {"a": [1, 5, 3], "b": [2, 3, 4]}


def test_a_frame_reads_a_mapping_on_its_own_as_values(firepanda):
    """Which is why naming a column here does nothing at all."""
    assert cells(frame(firepanda).replace({2: 7})) == {"a": [1, 7, 3], "b": [7, 3, 4]}
    assert cells(frame(firepanda).replace({"a": 2})) == {"a": [1, 2, 3], "b": [2, 3, 4]}


def test_a_frame_reads_a_mapping_as_the_value_by_column(firepanda):
    made = frame(firepanda).replace(2, {"a": 7, "b": 8})
    assert cells(made) == {"a": [1, 7, 3], "b": [8, 3, 4]}


def test_a_frame_ignores_a_name_it_does_not_carry(firepanda):
    assert cells(frame(firepanda).replace({"zz": 2}, 5)) == {"a": [1, 2, 3], "b": [2, 3, 4]}


def test_a_frame_leaves_the_columns_a_pair_cannot_reach_alone(firepanda):
    made = firepanda.DataFrame({"a": [1, 2], "b": ["x", "y"]}).replace(2, 9)
    assert cells(made) == {"a": [1, 9], "b": ["x", "y"]}


def test_inplace_settles_and_hands_the_object_back(firepanda):
    """`replace` is on the side of the split that answers the object."""
    column = firepanda.Series([1, 2])
    assert column.replace(2, 9, inplace=True) is column
    assert column.tolist() == [1, 9]
    made = frame(firepanda)
    assert made.replace(2, 9, inplace=True) is made
    assert cells(made) == {"a": [1, 9, 3], "b": [9, 3, 4]}


def test_a_pattern_is_refused(firepanda):
    with pytest.raises(NotImplementedError):
        firepanda.Series(["a"]).replace("a", "b", regex=True)


def test_both_libraries_answer_the_same_things(firepanda):
    questions = [
        lambda m: m.Series([1, 2, 3, 2]).replace(2, 9).tolist(),
        lambda m: m.Series([1, 2, 3, 2]).replace([1, 2], [2, 1]).tolist(),
        lambda m: m.Series([1, 2, 3, 2]).replace([1, 2], [2, 3]).tolist(),
        lambda m: m.Series([1, 2, 3]).replace({1: 7, 3: 8}).tolist(),
        lambda m: m.Series([1, 2, 3]).replace([1, 2], 0).tolist(),
        lambda m: m.Series([1, 2, 3]).replace("zz", 9).tolist(),
        lambda m: m.Series([1, 2, 3]).replace([], 0).tolist(),
        lambda m: m.Series([1.0, 2.0]).replace(float("nan"), 0.0).tolist(),
        lambda m: m.Series(["a", "b"]).replace({"a": "c", "b": "d"}).tolist(),
        lambda m: m.Series([True, False]).replace(True, False).tolist(),
        lambda m: labelled(m, ["x", "y"], [1, 2]).replace(2, 9).tolist(),
        lambda m: cells(frame(m).replace(2, 9)),
        lambda m: cells(frame(m).replace({"a": {2: 99}})),
        lambda m: cells(frame(m).replace({"a": 2}, 5)),
        lambda m: cells(frame(m).replace({2: 7})),
        lambda m: cells(frame(m).replace({"a": 2})),
        lambda m: cells(frame(m).replace(2, {"a": 7, "b": 8})),
        lambda m: cells(frame(m).replace({"zz": 2}, 5)),
        lambda m: cells(frame(m).replace([2, 3], [7, 8])),
    ]
    for question in questions:
        assert question(firepanda) == question(pd), question


def test_both_libraries_refuse_the_same_shapes(firepanda):
    questions = [
        (ValueError, lambda m: m.Series([1, 2, 3]).replace()),
        (ValueError, lambda m: m.Series([1, 2, 3]).replace({1: 2}, 3)),
        (ValueError, lambda m: m.Series([1, 2, 3]).replace(1, {"a": 2})),
        (ValueError, lambda m: m.Series([1, 2, 3]).replace([1, 2], [3])),
        (TypeError, lambda m: m.Series([1, 2, 3]).replace(None, 5)),
        (ValueError, lambda m: m.DataFrame({"a": [1, 2]}).replace()),
        (ValueError, lambda m: m.DataFrame({"a": [1, 2]}).replace([2, 3], {"a": 0})),
        (TypeError, lambda m: m.DataFrame({"a": [1, 2]}).replace({"a": [1, 2]}, [7, 8])),
        (ValueError, lambda m: m.DataFrame({"a": [1, 2]}).replace({"a": {2: 9}}, 5)),
    ]
    for refused, question in questions:
        with pytest.raises(refused) as mine:
            question(firepanda)
        with pytest.raises(refused) as theirs:
            question(pd)
        assert str(mine.value) == str(theirs.value)
