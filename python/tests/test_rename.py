"""Renaming, which changes a name and moves no rows.

Six callables and a property, and the thing they are all checked against is that
nothing in the frame moved. So most of these read the values back as well as the
names, because a rename that quietly reordered the columns would pass a test that
only looked at the schema.

The other half of the file is the doors that refuse. `rename(index=...)` and the
mapping form of `Series.rename` are a pass over the labels rather than an edit to
a schema, and they raise rather than being written as a loop in Python, which is
a decision document 45 argues for and these assert.
"""

from __future__ import annotations

import pandas as pd
import pytest


def frame(module):
    """A frame with a named index and three columns, two of them numbers."""
    return module.DataFrame(
        {"k": [2, 0, 3, 1], "v": [4.0, 5.0, 6.0, 7.0], "s": ["a", "b", "c", "d"]}
    ).set_index("k")


def test_a_dictionary_renames_the_columns_it_names_and_leaves_the_rest(firepanda):
    made = frame(firepanda)
    answered = made.rename(columns={"v": "value"})
    assert list(answered.columns) == ["value", "s"]
    assert answered["value"].tolist() == made["v"].tolist()


def test_a_callable_is_applied_to_every_column(firepanda):
    answered = frame(firepanda).rename(columns=str.upper)
    assert list(answered.columns) == ["V", "S"]


def test_the_positional_form_with_an_axis_is_the_same_door(firepanda):
    made = frame(firepanda)
    assert list(made.rename(str.upper, axis=1).columns) == ["V", "S"]
    assert list(made.rename(str.upper, axis="columns").columns) == ["V", "S"]


def test_two_columns_can_swap_names_in_one_call(firepanda):
    """Each half of a swap is illegal on its own, so this is the case that needs one pass."""
    made = frame(firepanda)
    answered = made.rename(columns={"v": "s", "s": "v"})
    assert list(answered.columns) == ["s", "v"]
    assert answered["s"].tolist() == made["v"].tolist()


def test_the_original_is_not_touched(firepanda):
    made = frame(firepanda)
    made.rename(columns={"v": "value"})
    assert list(made.columns) == ["v", "s"]


def test_a_name_that_is_not_there_is_ignored_by_default(firepanda):
    assert list(frame(firepanda).rename(columns={"zz": "Z"}).columns) == ["v", "s"]


def test_a_name_that_is_not_there_is_a_key_error_when_asked_for(firepanda):
    with pytest.raises(KeyError, match="zz"):
        frame(firepanda).rename(columns={"zz": "Z"}, errors="raise")


def test_every_missing_name_is_reported_and_not_just_the_first(firepanda):
    with pytest.raises(KeyError) as raised:
        frame(firepanda).rename(columns={"zz": "Z", "yy": "Y"}, errors="raise")
    assert "zz" in str(raised.value)
    assert "yy" in str(raised.value)


def test_a_word_that_is_neither_is_refused_rather_than_guessed_at(firepanda):
    with pytest.raises(ValueError, match="ignore"):
        frame(firepanda).rename(columns={"v": "value"}, errors="boom")


def test_a_rename_onto_a_name_the_frame_already_has_is_refused(firepanda):
    """pandas makes the duplicate and a schema here holds one column per name."""
    with pytest.raises(ValueError, match="two columns"):
        frame(firepanda).rename(columns={"v": "s"})


def test_renaming_the_row_labels_says_what_it_would_cost(firepanda):
    with pytest.raises(NotImplementedError, match="pass over the index"):
        frame(firepanda).rename(index={0: 99})


def test_the_bare_positional_form_is_the_row_labels_and_refuses(firepanda):
    with pytest.raises(NotImplementedError, match="pass over the index"):
        frame(firepanda).rename(str)


def test_both_doors_at_once_is_the_pandas_type_error(firepanda):
    with pytest.raises(TypeError, match="Cannot specify both"):
        frame(firepanda).rename(str.upper, columns={"v": "value"})


def test_no_door_at_all_is_the_pandas_type_error(firepanda):
    with pytest.raises(TypeError, match="must pass an index"):
        frame(firepanda).rename()


def test_rename_refuses_inplace(firepanda):
    with pytest.raises(NotImplementedError, match="inplace"):
        frame(firepanda).rename(columns={"v": "value"}, inplace=True)


def test_rename_refuses_a_level(firepanda):
    with pytest.raises(NotImplementedError, match="MultiIndex"):
        frame(firepanda).rename(columns={"v": "value"}, level=0)


def test_rename_axis_names_the_index_and_nothing_else(firepanda):
    made = frame(firepanda)
    answered = made.rename_axis("row")
    assert answered.index.name == "row"
    assert list(answered.columns) == list(made.columns)
    assert answered["v"].tolist() == made["v"].tolist()


def test_rename_axis_clears_the_name_when_given_nothing_to_call_it(firepanda):
    assert frame(firepanda).rename_axis(None).index.name is None


def test_rename_axis_with_no_argument_at_all_leaves_the_name_alone(firepanda):
    """None is a name to clear and a missing argument is not, which is why there is a sentinel."""
    assert frame(firepanda).rename_axis().index.name == "k"


def test_rename_axis_takes_the_sequence_form(firepanda):
    assert frame(firepanda).rename_axis(["row"]).index.name == "row"


def test_rename_axis_refuses_a_sequence_of_any_other_length(firepanda):
    with pytest.raises(ValueError, match="must be 1, got 2"):
        frame(firepanda).rename_axis(["row", "col"])


def test_rename_axis_has_nowhere_to_put_a_name_for_the_columns(firepanda):
    with pytest.raises(NotImplementedError, match="column axis"):
        frame(firepanda).rename_axis("cols", axis=1)


def test_rename_axis_refuses_inplace(firepanda):
    with pytest.raises(NotImplementedError, match="inplace"):
        frame(firepanda).rename_axis("row", inplace=True)


def test_a_column_can_be_renamed(firepanda):
    made = frame(firepanda)["v"]
    answered = made.rename("value")
    assert answered.name == "value"
    assert answered.tolist() == made.tolist()
    assert made.name == "v"


def test_a_column_keeps_its_labels_through_a_rename(firepanda):
    made = frame(firepanda)["v"]
    assert made.rename("value").index.tolist() == made.index.tolist()


def test_the_mapping_form_of_a_column_rename_is_the_label_half(firepanda):
    with pytest.raises(NotImplementedError, match="pass over the index"):
        frame(firepanda)["v"].rename({0: 99})


def test_a_column_rename_to_nothing_clears_the_name(firepanda):
    """Empty is how the core spells no name, and pandas spells the same state None."""
    answered = frame(firepanda)["v"].rename(None)
    assert answered.name == ""
    assert answered.tolist() == frame(firepanda)["v"].tolist()


def test_a_column_can_name_its_row_labels(firepanda):
    made = frame(firepanda)["v"]
    answered = made.rename_axis("row")
    assert answered.index.name == "row"
    assert answered.name == "v"
    assert answered.tolist() == made.tolist()


def test_an_index_reads_its_level_names_as_a_list(firepanda):
    assert frame(firepanda).index.names == ["k"]


def test_set_names_takes_a_name_or_a_sequence_of_one(firepanda):
    made = frame(firepanda).index
    assert made.set_names("row").names == ["row"]
    assert made.set_names(["row"]).names == ["row"]


def test_set_names_refuses_more_names_than_the_index_has_levels(firepanda):
    with pytest.raises(ValueError, match="must be 1, got 2"):
        frame(firepanda).index.set_names(["one", "two"])


def test_set_names_refuses_a_level_the_way_pandas_does(firepanda):
    """Even level zero, which is the only one a flat index has."""
    with pytest.raises(ValueError, match="Level must be None"):
        frame(firepanda).index.set_names("row", level=0)


def test_set_names_writes_in_place_when_asked_the_way_rename_does(firepanda):
    made = frame(firepanda).index
    assert made.set_names("row", inplace=True) is None
    assert made.name == "row"


def test_renaming_an_index_of_instants_keeps_the_calendar(firepanda):
    made = firepanda.DatetimeIndex(["2024-01-02", "2024-01-01"])
    assert type(made.rename("when")) is type(made)


def test_the_answers_match_pandas(firepanda):
    """Both libraries asked the same eight questions, which is what the suite is for."""
    answers = []
    for module in (pd, firepanda):
        made = frame(module)
        answers.append(
            [
                list(made.rename(columns={"v": "value"}).columns),
                list(made.rename(columns=str.upper).columns),
                list(made.rename(str.upper, axis=1).columns),
                list(made.rename(columns={"zz": "Z"}).columns),
                list(made.rename(columns={"v": "s", "s": "v"}).columns),
                made.rename_axis("row").index.name,
                made.rename_axis(None).index.name,
                list(made.index.set_names("row").names),
            ]
        )
    assert answers[0] == answers[1]
