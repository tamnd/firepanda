"""Filling the missing rows of a column with a value that was named.

The order here follows the two questions the method has to answer before it can
do anything. The first is which columns were named, since a scalar names all of
them and a dict names some, and the second is whether the column can hold what
was offered, since a column here is typed and stays typed. Everything after that
is one call to a coalesce that has been in the core the whole time.

The last third of the file is the part worth reading. A column with nothing
missing is untouched whatever the value is, which is pandas' rule and not a
shortcut, and it is the reason `df.fillna(0)` on a frame with a complete text
column is fine here and the same call on the same frame with one gap in that
column is not. Document 47 argues that the refusal is the honest answer, since
pandas widens to object there and this library has no type holding a number
beside text.
"""

from __future__ import annotations

import pandas as pd
import pytest


def frame(module):
    """Three columns of three types, each missing its middle row."""
    return module.DataFrame({"i": [1, None, 3], "f": [1.5, None, 3.5], "s": ["a", None, "c"]})


def labelled(module, labels, values):
    """A column carrying labels, built the way each library builds one."""
    if module is pd:
        return pd.Series(values, index=labels)
    return module.DataFrame({"k": labels, "v": values}).set_index("k")["v"]


def test_a_value_fills_the_column_it_was_offered_for(firepanda):
    assert frame(firepanda).fillna({"f": 0.0})["f"].tolist() == [1.5, 0.0, 3.5]


def test_the_columns_that_were_not_named_are_left_alone(firepanda):
    answered = frame(firepanda).fillna({"f": 0.0})
    assert answered["i"].tolist() == [1, None, 3]
    assert answered["s"].tolist() == ["a", None, "c"]


def test_a_dict_fills_several_columns_in_one_call(firepanda):
    answered = frame(firepanda).fillna({"i": 9, "s": "z"})
    assert answered["i"].tolist() == [1, 9, 3]
    assert answered["s"].tolist() == ["a", "z", "c"]
    assert answered["f"].tolist() == [1.5, None, 3.5]


def test_a_key_the_frame_does_not_have_is_dropped_rather_than_refused(firepanda):
    """The opposite of what `drop` does, because this was offered a value."""
    assert frame(firepanda).fillna({"nope": 1})["i"].tolist() == [1, None, 3]


def test_one_value_fills_every_column_that_can_hold_it(firepanda):
    made = firepanda.DataFrame({"a": [1.0, None], "b": [None, 2.0]})
    answered = made.fillna(0.0)
    assert answered["a"].tolist() == [1.0, 0.0]
    assert answered["b"].tolist() == [0.0, 2.0]


def test_a_none_means_there_is_nothing_to_do(firepanda):
    assert frame(firepanda).fillna(None)["i"].tolist() == [1, None, 3]


def test_the_column_keeps_its_type_and_the_frame_its_shape(firepanda):
    answered = frame(firepanda).fillna({"i": 0})
    assert answered["i"].dtype == "int64"
    assert answered.shape == (3, 3)


def test_the_row_labels_and_their_name_come_through(firepanda):
    made = frame(firepanda).set_index("i")
    answered = made.fillna({"f": 0.0})
    assert answered.index.name == "i"
    assert list(answered.index) == list(made.index)


def test_a_whole_number_written_as_a_float_goes_into_a_column_of_whole_numbers(firepanda):
    """The one place the two kinds of number meet, which pandas allows too."""
    assert frame(firepanda).fillna({"i": 2.0})["i"].tolist() == [1, 2, 3]


def test_a_whole_number_goes_into_a_column_of_fractions(firepanda):
    assert frame(firepanda).fillna({"f": 2})["f"].tolist() == [1.5, 2.0, 3.5]


def test_text_fills_a_column_of_text(firepanda):
    assert frame(firepanda).fillna({"s": "z"})["s"].tolist() == ["a", "z", "c"]


def test_a_column_fills_its_own_rows(firepanda):
    made = firepanda.Series([1.0, None, 3.0])
    assert made.fillna(0.0).tolist() == [1.0, 0.0, 3.0]


def test_a_column_keeps_its_name_and_its_labels(firepanda):
    made = frame(firepanda).set_index("i")["f"]
    answered = made.fillna(0.0)
    assert answered.name == "f"
    assert list(answered.index) == list(made.index)


def test_a_column_of_true_and_false_takes_one(firepanda):
    assert firepanda.Series([True, None, False]).fillna(True).tolist() == [
        True,
        True,
        False,
    ]


def test_true_is_not_a_number_here_and_is_not_one_in_pandas_either(firepanda):
    with pytest.raises(TypeError, match="Invalid value 'True' for dtype 'int64'"):
        frame(firepanda).fillna({"i": True})


def test_a_number_is_not_a_truth_value(firepanda):
    with pytest.raises(TypeError, match="for dtype 'bool'"):
        firepanda.Series([True, None]).fillna(1)


def test_text_does_not_go_into_a_column_of_numbers(firepanda):
    with pytest.raises(TypeError, match="Invalid value 'x' for dtype 'int64'"):
        frame(firepanda).fillna({"i": "x"})


def test_a_fraction_does_not_go_into_a_column_of_whole_numbers(firepanda):
    """A cast exists and throws the fraction away, which is not what was asked."""
    with pytest.raises(TypeError, match=r"Invalid value '0\.5' for dtype 'int64'"):
        frame(firepanda).fillna({"i": 0.5})


def test_a_number_does_not_go_into_a_column_of_text(firepanda):
    """pandas widens to object here and there is no type for that here."""
    with pytest.raises(TypeError, match="for dtype 'string'"):
        frame(firepanda).fillna({"s": 0})


def test_a_nan_is_missing_and_is_filled(firepanda):
    """pandas has no other way to spell missing in a float, so nor has this."""
    made = firepanda.Series([1.0, float("nan"), None])
    assert made.fillna(0.0).tolist() == [1.0, 0.0, 0.0]


def test_a_nan_in_a_frame_is_filled_too(firepanda):
    made = firepanda.DataFrame({"v": [1.0, float("nan"), None]})
    assert made.fillna(0.0)["v"].tolist() == [1.0, 0.0, 0.0]


def test_a_column_whose_only_gap_is_a_nan_is_not_a_clean_column(firepanda):
    """The count of what is missing is the count `isna` would agree with."""
    made = firepanda.Series([1.0, float("nan")])
    assert made.fillna(0.0).tolist() == [1.0, 0.0]


def test_a_category_column_takes_one_of_its_own_categories(firepanda):
    made = firepanda.Series(["b", "a", None, "c"]).astype("category")
    answered = made.fillna("a")
    assert answered.tolist() == ["b", "a", "a", "c"]
    assert answered.dtype == "category"
    assert list(answered.cat.categories) == ["a", "b", "c"]


def test_a_category_column_keeps_its_order_through_a_fill(firepanda):
    made = firepanda.Series(["b", "a", None]).astype("category").cat.as_ordered()
    assert made.fillna("a").cat.ordered is True


def test_a_category_column_in_a_frame_is_filled_the_same_way(firepanda):
    made = firepanda.DataFrame({"c": ["b", None]}).astype({"c": "category"})
    assert made.fillna({"c": "b"})["c"].tolist() == ["b", "b"]


def test_a_value_that_is_not_a_category_is_refused_in_pandas_words(firepanda):
    """A code is a position in a list, so a value off the list has no code."""
    made = firepanda.Series(["b", "a", None]).astype("category")
    with pytest.raises(TypeError, match="Cannot setitem on a Categorical"):
        made.fillna("nope")


def test_a_column_with_nothing_missing_is_untouched_whatever_was_offered(firepanda):
    made = firepanda.DataFrame({"i": [1, None, 3], "s": ["a", "b", "c"]})
    answered = made.fillna(0)
    assert answered["i"].tolist() == [1, 0, 3]
    assert answered["s"].tolist() == ["a", "b", "c"]


def test_a_nan_is_the_gap_that_the_validity_bits_do_not_show(firepanda):
    """A column the bits call complete, with a NaN in it, is filled anyway."""
    made = firepanda.DataFrame({"f": [float("nan"), 1.0, 2.0]})
    assert made.fillna(9.0)["f"].tolist() == [9.0, 1.0, 2.0]
    assert made["f"].fillna(9.0).tolist() == [9.0, 1.0, 2.0]


def test_a_complete_column_on_its_own_is_untouched_too(firepanda):
    assert firepanda.Series([1.0, 2.0]).fillna("x").tolist() == [1.0, 2.0]


def test_a_limit_is_refused_and_a_bad_one_is_refused_first(firepanda):
    with pytest.raises(NotImplementedError, match="limit="):
        frame(firepanda).fillna({"i": 0}, limit=1)
    with pytest.raises(ValueError, match="Limit must be greater than 0"):
        frame(firepanda).fillna({"i": 0}, limit=0)
    with pytest.raises(ValueError, match="Limit must be an integer"):
        frame(firepanda).fillna({"i": 0}, limit=1.5)


def test_inplace_settles_on_both_and_hands_the_object_back(firepanda):
    """`fillna` is on the side of the split that answers the object."""
    made = frame(firepanda)
    assert made.fillna({"i": 0}, inplace=True) is made
    assert made["i"].tolist() == [1, 0, 3]
    column = firepanda.Series([1.0, None])
    assert column.fillna(0.0, inplace=True) is column
    assert column.tolist() == [1.0, 0.0]


def test_an_axis_the_object_does_not_have_says_so(firepanda):
    with pytest.raises(ValueError, match="No axis named 2"):
        frame(firepanda).fillna({"i": 0}, axis=2)
    with pytest.raises(ValueError, match="No axis named 1 for object type Series"):
        firepanda.Series([1.0, None]).fillna(0.0, axis=1)


def test_the_second_axis_of_a_frame_is_the_same_answer(firepanda):
    """One value per column, so there is nothing for the axis to choose."""
    assert frame(firepanda).fillna({"i": 0}, axis=1)["i"].tolist() == [1, 0, 3]


def test_a_dict_on_a_column_maps_row_labels(firepanda):
    """The one shape that means something different on a column than on a frame."""
    assert firepanda.Series([1.0, None, None]).fillna({1: 5.0}).tolist() == [1.0, 5.0, None]


def test_a_dict_key_the_column_does_not_have_fills_nothing(firepanda):
    assert firepanda.Series([1.0, None]).fillna({7: 5.0}).tolist() == [1.0, None]


def test_a_dict_with_nothing_in_it_is_nothing_to_do(firepanda):
    assert firepanda.Series([1.0, None]).fillna({}).tolist() == [1.0, None]


def test_a_column_is_lined_up_by_label_and_not_by_position(firepanda):
    made = labelled(firepanda, ["a", "b", "c"], [1.0, None, None])
    other = labelled(firepanda, ["c", "b", "a"], [7.0, 8.0, 9.0])
    assert made.fillna(other).tolist() == [1.0, 8.0, 7.0]


def test_a_label_the_fallback_does_not_have_stays_missing(firepanda):
    made = labelled(firepanda, ["a", "b", "c"], [1.0, None, None])
    other = labelled(firepanda, ["z", "b"], [7.0, 8.0])
    assert made.fillna(other).tolist() == [1.0, 8.0, None]


def test_a_fallback_row_with_nothing_in_it_fills_nothing(firepanda):
    made = labelled(firepanda, ["a", "b"], [1.0, None])
    assert made.fillna(labelled(firepanda, ["a", "b"], [None, None])).tolist() == [1.0, None]


def test_a_column_handed_to_a_frame_names_columns_rather_than_rows(firepanda):
    """The shape worth reading twice, and pandas reads it the same way."""
    answered = frame(firepanda).fillna(labelled(firepanda, ["i", "f"], [9.0, 0.0]))
    assert answered["i"].tolist() == [1, 9, 3]
    assert answered["f"].tolist() == [1.5, 0.0, 3.5]
    assert answered["s"].tolist() == ["a", None, "c"]


def test_a_frame_handed_to_a_frame_lines_up_on_both_axes(firepanda):
    made = firepanda.DataFrame({"a": [1.0, None], "b": [None, 2.0]})
    answered = made.fillna(firepanda.DataFrame({"a": [0.0, 0.0]}))
    assert answered["a"].tolist() == [1.0, 0.0]
    assert answered["b"].tolist() == [None, 2.0]


def test_a_frame_handed_to_a_column_is_refused_in_pandas_words(firepanda):
    with pytest.raises(TypeError, match="must be a scalar, dict or Series"):
        firepanda.Series([1.0, None]).fillna(firepanda.DataFrame({"a": [1.0, 2.0]}))


def test_a_fallback_of_whole_numbers_fills_a_column_of_fractions(firepanda):
    made = labelled(firepanda, ["a", "b"], [1.5, None])
    assert made.fillna(labelled(firepanda, ["a", "b"], [9, 9])).tolist() == [1.5, 9.0]


def test_a_fraction_cannot_go_into_a_column_of_whole_numbers(firepanda):
    made = labelled(firepanda, ["a", "b"], [1, None])
    with pytest.raises(TypeError, match="cannot safely cast non-equivalent"):
        made.fillna(labelled(firepanda, ["a", "b"], [2.0, 2.5]))


def test_a_fraction_in_a_row_nobody_reads_is_nobody_s_business(firepanda):
    """pandas judges a fallback by the rows it takes out of it, and so does this."""
    made = labelled(firepanda, ["a", "b"], [1, None])
    assert made.fillna(labelled(firepanda, ["a", "b"], [2.5, 2.0])).tolist() == [1, 2]


def test_a_word_that_is_not_a_number_says_what_it_could_not_read(firepanda):
    made = labelled(firepanda, ["a", "b"], [1, None])
    with pytest.raises(ValueError, match="invalid literal for int"):
        made.fillna(labelled(firepanda, ["a", "b"], ["2", "x"]))


def test_a_column_of_numbers_cannot_fill_a_column_of_words(firepanda):
    """The same refusal the scalar half makes, for the same missing type."""
    made = labelled(firepanda, ["a", "b"], ["x", None])
    with pytest.raises(TypeError, match="Invalid fill column of dtype 'int64'"):
        made.fillna(labelled(firepanda, ["a", "b"], [1, 2]))


def test_a_category_column_takes_words_from_its_own_list(firepanda):
    made = firepanda.Series(["b", None, "c"]).astype("category")
    answered = made.fillna(firepanda.Series(["c", "c", "c"]))
    assert answered.tolist() == ["b", "c", "c"]
    assert answered.dtype == "category"


def test_a_word_off_the_list_in_a_row_that_is_read_is_refused(firepanda):
    made = firepanda.Series(["b", None, "c"]).astype("category")
    with pytest.raises(TypeError, match="Cannot setitem on a Categorical"):
        made.fillna(firepanda.Series(["c", "nope", "c"]))


def test_a_category_fallback_has_to_carry_the_same_list(firepanda):
    made = firepanda.Series(["b", None]).astype("category")
    other = firepanda.Series(["x", "x"]).astype("category")
    with pytest.raises(TypeError, match="without identical categories"):
        made.fillna(other)


def test_a_fallback_whose_labels_repeat_has_no_one_row_to_answer_with(firepanda):
    """pandas answers with the second of them. The alignment here refuses, in
    the sentence pandas itself gives when a label in two rows is looked up."""
    made = labelled(firepanda, ["a", "b"], [1.0, None])
    with pytest.raises(ValueError, match="cannot reindex on an axis with duplicate labels"):
        made.fillna(labelled(firepanda, ["a", "a"], [9.0, 9.0]))


def test_the_original_is_untouched(firepanda):
    made = frame(firepanda)
    made.fillna({"i": 0, "f": 0.0, "s": "z"})
    assert made["i"].tolist() == [1, None, 3]
    assert made["s"].tolist() == ["a", None, "c"]


def test_both_libraries_answer_the_same_things(firepanda):
    """The questions of this file asked of pandas as well, side by side."""
    questions = [
        lambda d: d.fillna({"f": 0.0})["f"].tolist(),
        lambda d: d.fillna({"i": 9, "s": "z"})["s"].tolist(),
        lambda d: d.fillna({"nope": 1, "s": "z"})["s"].tolist(),
        lambda d: d.fillna({"i": 2.0})["i"].tolist(),
        lambda d: d.fillna({"f": 2})["f"].tolist(),
        lambda d: d.fillna({"s": "z"})["s"].tolist(),
        lambda d: d.fillna({"f": 0.0}).shape,
        lambda d: d.set_index("i").fillna({"f": 0.0}).index.name,
        lambda d: d["f"].fillna(0.0).tolist(),
        lambda d: d["s"].fillna("z").tolist(),
        lambda d: d["s"].astype("category").fillna("a").tolist(),
        lambda d: d["f"].fillna(float("nan")).isna().tolist(),
        lambda d: list(d["s"].astype("category").fillna("a").cat.categories),
        lambda d: d["f"].fillna({1: 0.0}).tolist(),
        lambda d: d["f"].fillna(type(d["f"])([7.0, 7.0, 7.0])).tolist(),
        lambda d: d.fillna(type(d)({"f": [7.0, 7.0, 7.0]}))["f"].tolist(),
        lambda d: d.fillna(type(d)({"f": [7.0, 7.0, 7.0]}))["s"].isna().tolist(),
        lambda d: d["s"].astype("category").fillna(d["s"].fillna("a")).tolist(),
        lambda d: type(d)({"f": [float("nan"), 1.0]}).fillna(9.0)["f"].tolist(),
    ]
    for ask in questions:
        assert ask(frame(firepanda)) == ask(frame(pd))


def test_both_libraries_refuse_the_same_two_arguments(firepanda):
    """The limit checks are pandas' sentences, so they are checked against it."""
    for bad in (0, 1.5):
        for module in (firepanda, pd):
            with pytest.raises(ValueError):
                frame(module).fillna({"i": 0}, limit=bad)
