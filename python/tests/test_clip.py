"""Holding every value between two bounds, which is two comparisons and two picks.

`clip` belongs with `where` rather than with the arithmetic, because that is
what it is made of. A value below the floor is replaced by the floor, a value
above the ceiling is replaced by the ceiling, and everything else is kept, which
is two conditions and two other sides in the shape document 48 already built.

Three rules are worth reading before the tests. A value that is missing is
neither above nor below anything, so it is left alone, and that falls out of a
comparison answering nothing rather than out of a rule anybody wrote. A bound
that is a NaN is no bound at all, and so is a row of a bound that holds nothing,
which is how pandas reads numpy here. And a label a bound does not carry is not
the same as a row it carries nothing in: the first loses its value and the
second keeps it, which is section 4 of document 49 and is the only part of this
that surprises people.

The type rules are `where`'s. pandas widens a column when it puts a value of
another type in one and this does not, so a column of whole numbers bounded by a
fraction is a column of floats over there and a refusal here.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


def frame(module):
    """Two columns of numbers, wide enough that a bound reaches into both."""
    return module.DataFrame({"a": [1, 5, 10], "b": [0, 50, -5]})


def labelled(module, labels, values):
    """A column carrying labels, built the way each library builds one."""
    if module is pd:
        return pd.Series(values, index=labels)
    return module.DataFrame({"k": labels, "v": values}).set_index("k")["v"]


def cells(made):
    """Every column of a frame as a list, for comparing two frames by value."""
    return {str(name): made[name].tolist() for name in made.columns}


def test_a_floor_and_a_ceiling_hold_the_values_between_them(firepanda):
    assert firepanda.Series([1, 5, 10]).clip(2, 8).tolist() == [2, 5, 8]


def test_a_floor_on_its_own_lifts_the_values_under_it(firepanda):
    assert firepanda.Series([1, 5, 10]).clip(6).tolist() == [6, 6, 10]


def test_a_ceiling_on_its_own_lowers_the_values_over_it(firepanda):
    assert firepanda.Series([1, 5, 10]).clip(upper=6).tolist() == [1, 5, 6]


def test_no_bounds_at_all_is_the_column(firepanda):
    assert firepanda.Series([1, 5, 10]).clip().tolist() == [1, 5, 10]


def test_two_bounds_the_wrong_way_round_are_put_in_order(firepanda):
    """pandas' oldest correction in this method, and it is only made for values."""
    assert firepanda.Series([1, 5, 10]).clip(8, 2).tolist() == [2, 5, 8]


def test_two_bounds_that_carry_rows_are_not_put_in_order(firepanda):
    """The same pair written as runs of values, where the floor runs first."""
    made = firepanda.Series([1, 5, 10])
    assert made.clip([8, 8, 8], [2, 2, 2]).tolist() == [8, 2, 2]


def test_a_bound_that_is_a_nan_is_no_bound(firepanda):
    assert firepanda.Series([1, 5, 10]).clip(float("nan"), 8).tolist() == [1, 5, 8]


def test_a_value_that_is_missing_is_not_clipped(firepanda):
    """The rule nobody had to write, since a comparison answers nothing here."""
    made = firepanda.Series([1.0, None, 10.0])
    assert made.clip(2, 8).tolist() == [2.0, None, 8.0]


def test_a_nan_is_not_clipped_either(firepanda):
    """A NaN is missing to `isna` and so it is missing to this, as it is to a fill."""
    answered = firepanda.Series([1.0, float("nan"), 10.0]).clip(2, 8).tolist()
    assert answered[0] == 2.0
    assert answered[1] != answered[1]
    assert answered[2] == 8.0


def test_a_column_no_bound_reaches_keeps_its_type(firepanda):
    answered = firepanda.Series([1, 5, 10]).clip(0, 20)
    assert answered.tolist() == [1, 5, 10]
    assert answered.dtype == "int64"


def test_a_fraction_against_a_column_of_whole_numbers_is_refused(firepanda):
    """Where pandas widens to float64. Document 49 section 5 argues it."""
    with pytest.raises(firepanda.errors.DTypeError):
        firepanda.Series([1, 5, 10]).clip(2.5, 8.5)


def test_a_fraction_that_reaches_nothing_is_never_looked_at(firepanda):
    """The same bound on a column it does not reach, which is not an error."""
    assert firepanda.Series([1, 5, 10]).clip(0.5, 20.5).tolist() == [1, 5, 10]


def test_a_run_of_values_is_read_by_position(firepanda):
    assert firepanda.Series([1, 5, 10]).clip([2, 2, 2]).tolist() == [2, 5, 10]


def test_a_run_of_values_has_to_be_as_tall(firepanda):
    with pytest.raises(firepanda.errors.InvalidArgumentError):
        firepanda.Series([1, 5, 10]).clip([2, 2])


def test_a_row_of_a_bound_that_holds_nothing_is_no_bound_for_that_row(firepanda):
    made = firepanda.Series([1.0, 5.0, 10.0])
    assert made.clip(upper=[None, 2.0, None]).tolist() == [1.0, 2.0, 10.0]


def test_a_bound_with_nothing_in_it_at_all_is_no_bound(firepanda):
    made = firepanda.Series([1.0, 5.0, 10.0])
    assert made.clip([None, None, None]).tolist() == [1.0, 5.0, 10.0]


def test_a_bound_that_carries_labels_is_lined_up_on_them(firepanda):
    made = labelled(firepanda, ["a", "b", "c"], [1, 5, 10])
    bound = labelled(firepanda, ["a", "b", "c"], [0, 6, 20])
    assert made.clip(bound).tolist() == [1, 6, 20]


def test_a_label_the_bound_does_not_carry_loses_its_value(firepanda):
    """Which is not the same as a row it carries nothing in, one test above."""
    made = labelled(firepanda, ["a", "b", "c"], [1, 5, 10])
    bound = labelled(firepanda, ["a", "c"], [9, 9])
    assert made.clip(bound).tolist() == [9, None, 10]


def test_labels_the_column_does_not_have_are_ignored(firepanda):
    made = labelled(firepanda, ["a", "b"], [1, 5])
    bound = labelled(firepanda, ["a", "b", "z"], [9, 9, 9])
    assert made.clip(bound).tolist() == [9, 9]


def test_a_mapping_is_read_by_label_and_says_nothing_about_the_rest(firepanda):
    made = labelled(firepanda, ["a", "b", "c"], [1, 5, 10])
    assert made.clip({"a": 9}).tolist() == [9, 5, 10]


def test_the_labels_and_the_name_are_kept(firepanda):
    made = labelled(firepanda, ["x", "y", "z"], [1, 5, 10])
    answered = made.clip(2, 8)
    assert answered.index.tolist() == ["x", "y", "z"]
    assert answered.name == "v"


def test_words_are_bounded_by_words(firepanda):
    assert firepanda.Series(["a", "m", "z"]).clip("b", "y").tolist() == ["b", "m", "y"]


def test_a_category_takes_a_category_it_already_has(firepanda):
    made = firepanda.Series(["a", "b", "c"]).astype("category")
    ordered = made.cat.set_categories(["a", "b", "c"], ordered=True)
    assert ordered.clip("b", "c").tolist() == ["b", "b", "c"]


def test_a_column_refuses_inplace(firepanda):
    with pytest.raises(NotImplementedError):
        firepanda.Series([1, 5, 10]).clip(2, 8, inplace=True)


def test_numpys_out_is_taken_empty_and_refused_full(firepanda):
    made = firepanda.Series([1, 5, 10])
    assert made.clip(2, 8, out=None).tolist() == [2, 5, 8]
    with pytest.raises(firepanda.errors.InvalidArgumentError):
        made.clip(2, 8, out=[0, 0, 0])


def test_a_keyword_neither_library_has_is_refused(firepanda):
    with pytest.raises(TypeError):
        firepanda.Series([1, 5, 10]).clip(2, 8, nonsense=1)


def test_a_frame_bounds_every_column_by_the_same_value(firepanda):
    assert cells(frame(firepanda).clip(2, 8)) == {"a": [2, 5, 8], "b": [2, 8, 2]}


def test_a_run_of_values_on_a_frame_is_one_value_per_column(firepanda):
    assert cells(frame(firepanda).clip([2, 0])) == {"a": [2, 5, 10], "b": [0, 50, 0]}


def test_the_same_run_read_down_the_rows_is_one_value_per_row(firepanda):
    answered = frame(firepanda).clip([2, 3, 4], axis=0)
    assert cells(answered) == {"a": [2, 5, 10], "b": [2, 50, 4]}


def test_a_two_dimensional_run_is_one_value_per_cell(firepanda):
    answered = frame(firepanda).clip(np.array([[2, 2], [2, 2], [2, 2]]))
    assert cells(answered) == {"a": [2, 5, 10], "b": [2, 50, 2]}


def test_a_grid_written_as_lists_is_read_the_same_way(firepanda):
    """pandas takes the grid only as an array, which document 49 section 7 argues."""
    answered = frame(firepanda).clip([[2, 2], [2, 2], [2, 2]])
    assert cells(answered) == {"a": [2, 5, 10], "b": [2, 50, 2]}


def test_a_mapping_on_a_frame_is_read_by_column_name(firepanda):
    answered = frame(firepanda).clip({"b": 0, "a": 2})
    assert cells(answered) == {"a": [2, 5, 10], "b": [0, 50, 0]}


def test_one_column_as_a_bound_has_to_be_told_which_way_to_read(firepanda):
    with pytest.raises(firepanda.errors.InvalidArgumentError):
        frame(firepanda).clip(firepanda.Series([2, 2, 2]))


def test_one_column_read_across_them_is_a_bound_for_each(firepanda):
    bound = labelled(firepanda, ["a", "b"], [2, 10])
    answered = frame(firepanda).clip(bound, axis=1)
    assert cells(answered) == {"a": [2, 5, 10], "b": [10, 50, 10]}


def test_a_column_the_bound_does_not_reach_loses_every_row(firepanda):
    answered = frame(firepanda).clip(firepanda.DataFrame({"a": [2, 2, 2]}))
    assert cells(answered) == {"a": [2, 5, 10], "b": [None, None, None]}


def test_a_frame_of_bounds_lines_up_on_both_axes(firepanda):
    bounds = firepanda.DataFrame({"a": [2, 2, 2], "b": [0, 0, 0]})
    answered = frame(firepanda).clip(bounds)
    assert cells(answered) == {"a": [2, 5, 10], "b": [0, 50, 0]}


def test_a_frame_refuses_inplace(firepanda):
    with pytest.raises(NotImplementedError):
        frame(firepanda).clip(2, 8, inplace=True)


def test_the_original_is_untouched(firepanda):
    made = frame(firepanda)
    made.clip(2, 8)
    made["a"].clip(2, 8)
    assert cells(made) == {"a": [1, 5, 10], "b": [0, 50, -5]}


def test_both_libraries_answer_the_same_things(firepanda):
    """The questions of this file asked of pandas as well, side by side.

    The ones left out are the ones where pandas widens, since a column of whole
    numbers that pandas has made into floats holds the same numbers under
    different names and the lists do not compare.
    """
    questions = [
        lambda m: m.Series([1, 5, 10]).clip(2, 8).tolist(),
        lambda m: m.Series([1, 5, 10]).clip(8, 2).tolist(),
        lambda m: m.Series([1, 5, 10]).clip(6).tolist(),
        lambda m: m.Series([1, 5, 10]).clip(upper=6).tolist(),
        lambda m: m.Series([1, 5, 10]).clip().tolist(),
        lambda m: m.Series([1, 5, 10]).clip(float("nan"), 8).tolist(),
        lambda m: m.Series([1, 5, 10]).clip(0, 20).tolist(),
        lambda m: m.Series([1, 5, 10]).clip(0.5, 20.5).tolist(),
        lambda m: m.Series([1, 5, 10]).clip([8, 8, 8], [2, 2, 2]).tolist(),
        lambda m: m.Series([1, 5, 10]).clip([2, 2, 2]).tolist(),
        lambda m: m.Series([1, 5, 10]).clip(2, [4, 4, 4]).tolist(),
        lambda m: m.Series([1.0, 5.0, 10.0]).clip(upper=[None, 2.0, None]).tolist(),
        lambda m: m.Series([1.0, 5.0, 10.0]).clip([None, None, None]).tolist(),
        lambda m: m.Series([1.0, float("nan"), 10.0]).clip(2, 8).isna().tolist(),
        lambda m: m.Series(["a", "m", "z"]).clip("b", "y").tolist(),
        lambda m: m.Series([True, False]).clip(False, True).tolist(),
        lambda m: m.Series([], dtype="float64").clip(1, 2).tolist(),
        lambda m: m.Series([1, 5, 10]).clip(2, 8, out=None).tolist(),
        lambda m: labelled(m, ["a", "b", "c"], [1, 5, 10]).clip({"a": 9}).tolist(),
        lambda m: labelled(m, ["a", "b", "c"], [1, 5, 10]).clip(2, 8).index.tolist(),
        lambda m: (
            labelled(m, ["a", "b"], [1, 5]).clip(labelled(m, ["a", "b", "z"], [9] * 3)).tolist()
        ),
        lambda m: cells(frame(m).clip(2, 8)),
        lambda m: cells(frame(m).clip()),
        lambda m: cells(frame(m).clip([2, 0])),
        lambda m: cells(frame(m).clip([2, 0], axis=1)),
        lambda m: cells(frame(m).clip([2, 3, 4], axis=0)),
        lambda m: cells(frame(m).clip(np.array([[2, 2], [2, 2], [2, 2]]))),
        lambda m: cells(frame(m).clip({"b": 0, "a": 2})),
        lambda m: cells(frame(m).clip(labelled(m, ["a", "b"], [2, 10]), axis=1)),
        lambda m: cells(frame(m).clip(m.Series([2, 3, 4]), axis=0)),
        lambda m: cells(frame(m).clip(m.DataFrame({"a": [2, 2, 2], "b": [0, 0, 0]}))),
        lambda m: cells(frame(m).clip(2, 8, axis=0)),
    ]
    for ask in questions:
        assert ask(firepanda) == ask(pd)


def test_both_libraries_refuse_the_same_shapes(firepanda):
    """The sentences are pandas' own, so the refusals are checked against it."""
    refusals = [
        lambda m: m.Series([1, 5, 10]).clip([2, 2]),
        lambda m: m.Series([1, 5, 10]).clip(2, 8, nonsense=1),
        lambda m: m.Series([1, 5, 10]).clip(2, 8, out=[0, 0, 0]),
        lambda m: frame(m).clip([2, 3, 4]),
        lambda m: frame(m).clip([1, 2], axis=0),
        lambda m: frame(m).clip({"a": 2}),
        lambda m: frame(m).clip(m.Series([2, 2, 2])),
        lambda m: frame(m).clip([[2, 2, 2], [2, 2, 2]]),
    ]
    for ask in refusals:
        for module in (firepanda, pd):
            with pytest.raises((TypeError, ValueError)):
                ask(module)
