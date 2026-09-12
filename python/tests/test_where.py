"""Choosing each row against a condition, and the other side it falls back to.

`where` keeps the rows a condition says are true and takes the rest from
somewhere else, `mask` is the same thing with the condition turned over, and the
interesting half of both of them is the somewhere else. It can be a value, a
column lined up by label, a run of values read by position, a whole frame lined
up on both axes, or nothing at all, and nothing at all is the one shape `fillna`
never had to answer.

Two rules are worth reading before the tests. A row the condition says nothing
about is not kept, by either method, which is why `mask` turns the condition over
first and reads its nulls as falses second. And a column that keeps all of its
rows is answered untouched without the other side being looked at, which is not
a shortcut but the rule that lets `s.where(everything, "a word")` on a column of
numbers be the column in both libraries.

Where pandas and this part company is the type. pandas widens a column when it
puts a missing value in one, to float64 for integers and to object for words,
because a numpy dtype has nowhere to put one. A column here has somewhere, the
way pandas' own `Int64` does, so the column keeps its type and holds a null.
Document 48 argues it out.
"""

from __future__ import annotations

import pandas as pd
import pytest


def frame(module):
    """Two columns of numbers, complete, so the condition is the only story."""
    return module.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})


def flags(module):
    """A frame of flags shaped like `frame`, false in three of its six cells."""
    return module.DataFrame({"a": [True, False, True], "b": [False, True, False]})


def labelled(module, labels, values):
    """A column carrying labels, built the way each library builds one."""
    if module is pd:
        return pd.Series(values, index=labels)
    return module.DataFrame({"k": labels, "v": values}).set_index("k")["v"]


def cells(made):
    """Every column of a frame as a list, for comparing two frames by value."""
    return {name: made[name].tolist() for name in made.columns}


def gappy(module, values):
    """A condition with a row saying nothing, built the way each library can.

    pandas can only answer this one in a nullable type. A list with a None in it
    becomes an object array over there, and `mask` inverts the condition with a
    `~`, which an object array of bools and a None cannot survive. The nullable
    `boolean` is the type whose behaviour this library's plain bool copies.
    """
    if module is pd:
        return pd.Series(values, dtype="boolean")
    return module.Series(values)


def test_a_condition_keeps_the_rows_it_says_are_true(firepanda):
    made = firepanda.Series([1, 2, 3, 4])
    assert made.where([True, False, True, False], 0).tolist() == [1, 0, 3, 0]


def test_a_mask_keeps_the_rows_the_condition_says_are_false(firepanda):
    made = firepanda.Series([1, 2, 3, 4])
    assert made.mask([True, False, True, False], 0).tolist() == [0, 2, 0, 4]


def test_no_other_side_leaves_those_rows_holding_nothing(firepanda):
    """And holding nothing in int64, which is the whole argument of the slice."""
    answered = firepanda.Series([1, 2, 3, 4]).where([True, False, True, False])
    assert answered.tolist() == [1, None, 3, None]
    assert answered.dtype == "int64"


def test_a_row_the_condition_says_nothing_about_is_not_kept(firepanda):
    made = firepanda.Series([1, 2, 3])
    assert made.where([True, None, True], 0).tolist() == [1, 0, 3]


def test_a_row_the_condition_says_nothing_about_is_not_kept_by_a_mask_either(firepanda):
    """The reason the condition is turned over before its nulls are read.

    A null read as a false and then turned over would be a true, and the row
    would be kept by `mask` where pandas replaces it. Both methods replace a row
    the condition says nothing in, which reads oddly until you notice that it is
    the same rule twice rather than one rule and its opposite.
    """
    made = firepanda.Series([1, 2, 3])
    assert made.mask([True, None, True], 0).tolist() == [0, 0, 0]


def test_a_condition_that_keeps_everything_leaves_the_column_alone(firepanda):
    """So the other side is never looked at, let alone asked to be an int."""
    made = firepanda.Series([1, 2, 3])
    assert made.where([True, True, True], "a word").tolist() == [1, 2, 3]


def test_a_condition_of_ones_and_zeros_is_refused(firepanda):
    """The obvious thing to write, and the one pandas will not take."""
    made = firepanda.Series([1, 2, 3])
    with pytest.raises(TypeError, match="Boolean array expected for the condition"):
        made.where([1, 0, 1])


def test_a_condition_with_no_labels_has_to_be_exactly_as_long(firepanda):
    made = firepanda.Series([1, 2, 3])
    with pytest.raises(ValueError, match="Array conditional must be same shape as self"):
        made.where([True, False])


def test_a_condition_with_labels_is_lined_up_against_the_rows(firepanda):
    made = labelled(firepanda, ["a", "b", "c"], [1, 2, 3])
    given = labelled(firepanda, ["c", "a"], [True, True])
    assert made.where(given, 0).tolist() == [1, 0, 3]


def test_a_label_the_condition_does_not_carry_is_a_false(firepanda):
    """A reindex that misses normally leaves a gap. Here it leaves a decision."""
    made = labelled(firepanda, ["a", "b"], [1, 2])
    assert made.where(labelled(firepanda, ["a"], [True]), 0).tolist() == [1, 0]


def test_the_other_side_is_lined_up_by_label_as_well(firepanda):
    made = labelled(firepanda, ["a", "b", "c"], [1, 2, 3])
    given = labelled(firepanda, ["b", "c"], [8, 9])
    assert made.where([True, False, False], given).tolist() == [1, 8, 9]


def test_a_label_the_other_side_does_not_carry_leaves_the_row_missing(firepanda):
    made = labelled(firepanda, ["a", "b"], [1, 2])
    assert made.where([True, False], labelled(firepanda, ["a"], [9])).tolist() == [1, None]


def test_a_row_of_the_other_side_that_holds_nothing_leaves_the_row_missing(firepanda):
    made = firepanda.Series([1, 2, 3])
    given = firepanda.Series([9, None, 9])
    assert made.where([True, False, False], given).tolist() == [1, None, 9]


def test_a_value_the_column_cannot_hold_is_refused_rather_than_widened(firepanda):
    made = firepanda.Series([1, 2, 3])
    with pytest.raises(TypeError, match="Invalid value 'x' for dtype 'int64'"):
        made.where([True, False, True], "x")


def test_a_run_of_values_with_no_labels_is_read_by_position(firepanda):
    made = firepanda.Series([1, 2, 3])
    assert made.where([True, False, False], [7, 8, 9]).tolist() == [1, 8, 9]


def test_a_run_of_values_of_the_wrong_length_is_refused(firepanda):
    made = firepanda.Series([1, 2, 3])
    with pytest.raises(ValueError, match="cannot reshape array of size 2"):
        made.where([True, False, False], [7, 8])


def test_a_callable_condition_is_handed_the_column(firepanda):
    made = firepanda.Series([1, 2, 3])
    assert made.where(lambda c: c > 1, 0).tolist() == [0, 2, 3]


def test_a_callable_other_side_is_handed_the_column_too(firepanda):
    made = firepanda.Series([1, 2, 3])
    assert made.where([True, False, True], lambda c: c * 10).tolist() == [1, 20, 3]


def test_an_existing_category_is_one_a_category_column_can_hold(firepanda):
    made = firepanda.Series(["a", "b", "c"]).astype("category")
    answered = made.where([True, False, True], "a")
    assert answered.tolist() == ["a", "a", "c"]
    assert answered.dtype == "category"


def test_a_new_category_is_refused_the_way_setting_one_is(firepanda):
    made = firepanda.Series(["a", "b"]).astype("category")
    with pytest.raises(TypeError, match="Cannot setitem on a Categorical"):
        made.where([True, False], "z")


def test_a_category_column_with_no_other_side_holds_nothing_there(firepanda):
    made = firepanda.Series(["a", "b"]).astype("category")
    answered = made.where([True, False])
    assert answered.tolist() == ["a", None]
    assert answered.dtype == "category"


def test_a_column_keeps_its_labels_and_its_name(firepanda):
    made = labelled(firepanda, ["a", "b"], [1, 2])
    answered = made.where([True, False], 0)
    assert answered.index.tolist() == ["a", "b"]
    assert answered.name == "v"


def test_a_frame_cannot_be_the_other_side_of_a_column(firepanda):
    made = firepanda.Series([1, 2, 3])
    with pytest.raises(NotImplementedError, match="higher dimensional"):
        made.where([True, False, True], frame(firepanda))


def test_a_frame_cannot_be_the_condition_of_a_column_either(firepanda):
    made = firepanda.Series([1, 2, 3])
    with pytest.raises(ValueError, match="Must specify axis=0 or 1"):
        made.where(flags(firepanda))


def test_a_column_has_no_second_axis_to_be_read_along(firepanda):
    made = firepanda.Series([1, 2, 3])
    with pytest.raises(ValueError, match="No axis named 1 for object type Series"):
        made.where([True, False, True], 0, axis=1)


def test_a_column_settles_in_place_and_refuses_a_level(firepanda):
    made = firepanda.Series([1, 2, 3])
    assert made.where([True, False, True], 0, inplace=True) is made
    assert made.tolist() == [1, 0, 3]
    with pytest.raises(NotImplementedError, match="level"):
        firepanda.Series([1, 2, 3]).where([True, False, True], 0, level=0)


def test_a_frame_condition_lines_up_on_both_axes(firepanda):
    answered = frame(firepanda).where(flags(firepanda), 0)
    assert cells(answered) == {"a": [1, 0, 3], "b": [0, 5, 0]}


def test_a_column_the_frame_condition_does_not_carry_keeps_nothing(firepanda):
    answered = frame(firepanda).where(firepanda.DataFrame({"a": [True, True, True]}), 0)
    assert cells(answered) == {"a": [1, 2, 3], "b": [0, 0, 0]}


def test_a_condition_of_one_column_is_read_down_the_rows(firepanda):
    answered = frame(firepanda).where(firepanda.Series([True, False, True]), 0)
    assert cells(answered) == {"a": [1, 0, 3], "b": [4, 0, 6]}


def test_a_condition_of_one_column_is_read_across_them_when_told_to_be(firepanda):
    """The shape pandas cannot run at all, which document 48 section 7 reports."""
    given = labelled(firepanda, ["a", "b"], [True, False])
    answered = frame(firepanda).where(given, 0, axis=1)
    assert cells(answered) == {"a": [1, 2, 3], "b": [0, 0, 0]}


def test_a_two_dimensional_condition_is_read_by_position(firepanda):
    answered = frame(firepanda).where([[True, False], [True, False], [True, False]], 0)
    assert cells(answered) == {"a": [1, 2, 3], "b": [0, 0, 0]}


def test_a_one_dimensional_condition_on_a_frame_is_refused(firepanda):
    """Even though it is exactly as tall as the frame, which pandas also does."""
    with pytest.raises(ValueError, match="Array conditional must be same shape as self"):
        frame(firepanda).where([True, False, True])


def test_a_run_of_values_on_a_frame_is_one_value_per_column(firepanda):
    """Numpy's broadcasting rule, and not the reading that looks right."""
    answered = frame(firepanda).where(flags(firepanda), [10, 20])
    assert cells(answered) == {"a": [1, 10, 3], "b": [20, 5, 20]}


def test_a_two_dimensional_run_of_values_is_one_value_per_cell(firepanda):
    answered = frame(firepanda).where(flags(firepanda), [[1, 2], [3, 4], [5, 6]])
    assert cells(answered) == {"a": [1, 3, 3], "b": [2, 5, 6]}


def test_a_run_of_values_that_is_not_the_frames_shape_is_refused(firepanda):
    with pytest.raises(ValueError, match="cannot reshape array of size 3 into shape"):
        frame(firepanda).where(flags(firepanda), [1, 2, 3])


def test_one_column_as_the_other_side_has_to_be_told_which_way_to_read(firepanda):
    with pytest.raises(ValueError, match="Must specify axis=0 or 1"):
        frame(firepanda).where(flags(firepanda), firepanda.Series([7, 8, 9]))


def test_one_column_read_down_the_rows_fills_every_column_from_it(firepanda):
    answered = frame(firepanda).where(flags(firepanda), firepanda.Series([7, 8, 9]), axis=0)
    assert cells(answered) == {"a": [1, 8, 3], "b": [7, 5, 9]}


def test_one_column_read_across_them_is_a_value_for_each(firepanda):
    given = labelled(firepanda, ["a", "b"], [7, 8])
    answered = frame(firepanda).where(flags(firepanda), given, axis=1)
    assert cells(answered) == {"a": [1, 7, 3], "b": [8, 5, 8]}


def test_a_frame_as_the_other_side_lines_up_on_both_axes(firepanda):
    given = firepanda.DataFrame({"a": [0, 0, 0]})
    answered = frame(firepanda).where(flags(firepanda), given)
    assert cells(answered) == {"a": [1, 0, 3], "b": [None, 5, None]}


def test_a_mapping_is_one_object_rather_than_a_value_per_column(firepanda):
    """Which is the opposite of what the same argument means to `fillna`, and
    which pandas answers by putting the mapping itself in every cell."""
    with pytest.raises(ValueError, match="cannot reshape array of size 1"):
        frame(firepanda).where(flags(firepanda), {"a": 9})


def test_a_frame_settles_in_place_and_is_handed_back(firepanda):
    made = frame(firepanda)
    assert made.where(flags(firepanda), 0, inplace=True) is made


def test_the_original_is_untouched(firepanda):
    made = frame(firepanda)
    made.where(flags(firepanda), 0)
    made["a"].where([False, False, False], 0)
    assert cells(made) == {"a": [1, 2, 3], "b": [4, 5, 6]}


def test_both_libraries_answer_the_same_things(firepanda):
    """The questions of this file asked of pandas as well, side by side.

    The ones left out are the ones where pandas widens, since a column of
    integers that pandas has made into floats holds the same numbers under
    different names and the lists do not compare. Those are asked here as a
    pattern of what is missing rather than as values.
    """
    questions = [
        lambda m: m.DataFrame({"a": [1, 2, 3]})["a"].where([True, False, True], 0).tolist(),
        lambda m: m.DataFrame({"a": [1, 2, 3]})["a"].mask([True, False, True], 0).tolist(),
        lambda m: m.DataFrame({"a": [1, 2, 3]})["a"].where([True, None, True], 0).tolist(),
        lambda m: m.Series([1, 2, 3]).where(gappy(m, [True, None, True]), 0).tolist(),
        lambda m: m.Series([1, 2, 3]).mask(gappy(m, [True, None, True]), 0).tolist(),
        lambda m: m.DataFrame({"a": [1, 2, 3]})["a"].where(lambda c: c > 1, 0).tolist(),
        lambda m: m.DataFrame({"a": [1, 2, 3]})["a"].where([True] * 3, "a word").tolist(),
        lambda m: m.DataFrame({"a": [1, 2, 3]})["a"].where([True, False, True]).isna().tolist(),
        lambda m: labelled(m, ["a", "b"], [1, 2]).where(labelled(m, ["a"], [True]), 0).tolist(),
        lambda m: labelled(m, ["a", "b"], [1, 2]).where([True, False], 0).index.tolist(),
        lambda m: cells(frame(m).where(flags(m), 0)),
        lambda m: cells(frame(m).mask(flags(m), 0)),
        lambda m: cells(frame(m).where(m.Series([True, False, True]), 0)),
        lambda m: cells(frame(m).where([[True, False]] * 3, 0)),
        lambda m: cells(frame(m).where(flags(m), [10, 20])),
        lambda m: cells(frame(m).where(flags(m), [[1, 2], [3, 4], [5, 6]])),
        lambda m: cells(frame(m).where(flags(m), m.Series([7, 8, 9]), axis=0)),
        lambda m: cells(frame(m).where(flags(m), labelled(m, ["a", "b"], [7, 8]), axis=1)),
        lambda m: cells(frame(m).where(m.DataFrame({"a": [True] * 3}), 0)),
        lambda m: frame(m).where(flags(m), 0).shape,
    ]
    for ask in questions:
        assert ask(firepanda) == ask(pd)


def test_both_libraries_refuse_the_same_shapes(firepanda):
    """The sentences are pandas' own, so the refusals are checked against it."""
    refusals = [
        lambda m: m.DataFrame({"a": [1, 2, 3]})["a"].where([True, False]),
        lambda m: m.DataFrame({"a": [1, 2, 3]})["a"].where([1, 0, 1]),
        lambda m: m.DataFrame({"a": [1, 2, 3]})["a"].where([True, False, True], [7, 8]),
        lambda m: frame(m).where([True, False, True]),
        lambda m: frame(m).where(flags(m), [1, 2, 3]),
        lambda m: frame(m).where(flags(m), m.Series([7, 8, 9])),
    ]
    for ask in refusals:
        for module in (firepanda, pd):
            with pytest.raises((TypeError, ValueError)):
                ask(module)
