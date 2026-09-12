"""Dropping, which takes a column out of a schema or a row out of an index.

The two halves are checked separately and then together, because pandas spells
them with one word and picks between them from the keyword that arrived, so the
first thing worth asserting is that the right door opened. After that the column
half is checked to have kept the rows and the row half to have kept the columns,
since either one quietly doing the other's work would pass a test that only
counted what came back.

The last part of the file is the three ways of naming the wrong door, the two
parameters that refuse, and the repeated row label this refuses and pandas does
not, which document 46 argues is the cost of the row half being `Index.drop` and
`reindex` composed rather than a kernel of its own.
"""

from __future__ import annotations

import pandas as pd
import pytest


def frame(module):
    """A frame with a named index and three columns, two of them numbers."""
    return module.DataFrame(
        {"k": [2, 0, 3, 1], "v": [4.0, 5.0, 6.0, 7.0], "s": ["a", "b", "c", "d"]}
    ).set_index("k")


def test_a_named_column_goes_and_the_others_stay_in_order(firepanda):
    made = frame(firepanda)
    answered = made.drop(columns=["v"])
    assert list(answered.columns) == ["s"]
    assert answered["s"].tolist() == made["s"].tolist()


def test_dropping_a_column_keeps_every_row_and_the_level_name(firepanda):
    answered = frame(firepanda).drop(columns=["v"])
    assert list(answered.index) == [2, 0, 3, 1]
    assert answered.index.name == "k"


def test_one_column_name_on_its_own_is_a_name_and_not_a_sequence(firepanda):
    """A string is iterable in Python, so `columns="s"` has to mean one name."""
    assert list(frame(firepanda).drop(columns="s").columns) == ["v"]


def test_the_positional_form_with_an_axis_is_the_column_door(firepanda):
    assert list(frame(firepanda).drop("s", axis=1).columns) == ["v"]
    assert list(frame(firepanda).drop("s", axis="columns").columns) == ["v"]


def test_a_set_of_names_is_taken_since_order_does_not_matter_to_a_drop(firepanda):
    assert list(frame(firepanda).drop(columns={"s"}).columns) == ["v"]


def test_an_empty_list_of_names_drops_nothing(firepanda):
    assert list(frame(firepanda).drop(columns=[]).columns) == ["v", "s"]


def test_every_column_can_go_and_what_is_left_still_has_its_rows(firepanda):
    answered = frame(firepanda).drop(columns=["v", "s"])
    assert answered.shape == (4, 0)


def test_the_named_labels_go_and_the_rest_keep_their_order(firepanda):
    made = frame(firepanda)
    answered = made.drop([0, 3])
    assert list(answered.index) == [2, 1]
    assert answered["v"].tolist() == [4.0, 7.0]


def test_a_single_label_is_a_label_and_not_a_sequence(firepanda):
    assert list(frame(firepanda).drop(0).index) == [2, 3, 1]


def test_the_index_keyword_is_the_same_door_as_the_bare_form(firepanda):
    assert list(frame(firepanda).drop(index=[0]).index) == [2, 3, 1]


def test_dropping_a_row_keeps_the_columns_and_the_level_name(firepanda):
    answered = frame(firepanda).drop([0])
    assert list(answered.columns) == ["v", "s"]
    assert answered.index.name == "k"


def test_a_label_that_is_a_string_is_found_the_same_way(firepanda):
    made = firepanda.DataFrame({"i": ["x", "y", "z"], "a": [1, 2, 3]}).set_index("i")
    answered = made.drop(["y"])
    assert list(answered.index) == ["x", "z"]
    assert answered["a"].tolist() == [1, 3]


def test_every_row_can_go_and_the_columns_are_still_there(firepanda):
    assert frame(firepanda).drop([2, 0, 3, 1]).shape == (0, 2)


def test_both_axes_in_one_call_do_both(firepanda):
    answered = frame(firepanda).drop(index=[0], columns=["s"])
    assert list(answered.index) == [2, 3, 1]
    assert list(answered.columns) == ["v"]


def test_a_column_drops_its_own_rows(firepanda):
    made = frame(firepanda)["v"]
    answered = made.drop([0, 3])
    assert answered.tolist() == [4.0, 7.0]
    assert list(answered.index) == [2, 1]
    assert answered.name == "v"


def test_a_column_takes_the_index_keyword_too(firepanda):
    assert frame(firepanda)["v"].drop(index=[0]).tolist() == [4.0, 6.0, 7.0]


def test_a_column_accepts_the_columns_keyword_and_does_nothing_with_it(firepanda):
    """Measured against a running pandas, which takes it and answers unchanged."""
    made = frame(firepanda)["v"]
    assert made.drop(columns=["anything"]).tolist() == made.tolist()


def test_a_missing_column_name_is_a_key_error_listing_all_of_them(firepanda):
    with pytest.raises(KeyError) as raised:
        frame(firepanda).drop(columns=["x", "y"])
    assert "'x', 'y'" in str(raised.value)


def test_a_missing_row_label_is_a_key_error_as_well(firepanda):
    with pytest.raises(KeyError):
        frame(firepanda).drop([99])


def test_ignore_skips_a_column_name_that_is_not_there(firepanda):
    assert list(frame(firepanda).drop(columns=["x"], errors="ignore").columns) == ["v", "s"]


def test_ignore_skips_a_row_label_that_is_not_there(firepanda):
    assert list(frame(firepanda).drop([99], errors="ignore").index) == [2, 0, 3, 1]


def test_ignore_still_drops_the_names_that_are_there(firepanda):
    answered = frame(firepanda).drop(columns=["x", "v"], errors="ignore")
    assert list(answered.columns) == ["s"]


def test_a_misspelled_errors_is_refused_rather_than_read_as_raise(firepanda):
    """pandas reads anything that is not ignore as raise, and document 23 does not."""
    with pytest.raises(ValueError, match="expected 'ignore' or 'raise'"):
        frame(firepanda).drop(columns=["s"], errors="nope")


def test_naming_no_axis_at_all_is_an_error(firepanda):
    with pytest.raises(ValueError, match="at least one of"):
        frame(firepanda).drop()


def test_labels_beside_a_keyword_is_an_error(firepanda):
    with pytest.raises(ValueError, match="both 'labels' and"):
        frame(firepanda).drop([0], index=[1])


def test_a_keyword_beside_the_column_axis_is_an_error(firepanda):
    with pytest.raises(ValueError, match="both 'axis' and"):
        frame(firepanda).drop(columns=["s"], axis=1)


def test_the_row_axis_beside_a_keyword_is_fine_since_it_is_the_default(firepanda):
    assert list(frame(firepanda).drop(index=[0], axis=0).index) == [2, 3, 1]


def test_an_axis_the_frame_does_not_have_says_so(firepanda):
    with pytest.raises(ValueError, match="No axis named 2"):
        frame(firepanda).drop("s", axis=2)


def test_a_column_has_no_second_axis_to_name(firepanda):
    with pytest.raises(ValueError, match="No axis named 1 for object type Series"):
        frame(firepanda)["v"].drop("x", axis=1)


def test_a_level_is_refused_because_there_is_no_multiindex(firepanda):
    with pytest.raises(NotImplementedError, match="level="):
        frame(firepanda).drop([0], level=0)


def test_inplace_settles_on_both_and_answers_nothing(firepanda):
    """`drop` is on the side of the split that answers None, as pandas does."""
    made = frame(firepanda)
    assert made.drop(columns=["s"], inplace=True) is None
    assert "s" not in list(made.columns)
    column = frame(firepanda)["v"]
    assert column.drop([0], inplace=True) is None
    assert column.tolist() == [4.0, 6.0, 7.0]


def test_a_repeated_row_label_is_refused_where_pandas_drops_both(firepanda):
    """The row half is a reindex, and a reindex cannot answer one row per label."""
    made = firepanda.DataFrame({"k": [1, 1, 2], "v": [1, 2, 3]}).set_index("k")
    with pytest.raises(ValueError, match="duplicate labels"):
        made.drop([1])


def test_the_original_is_untouched_by_either_half(firepanda):
    made = frame(firepanda)
    made.drop(columns=["s"])
    made.drop([0])
    assert list(made.columns) == ["v", "s"]
    assert list(made.index) == [2, 0, 3, 1]


def test_both_libraries_answer_the_same_things(firepanda):
    """The questions of this file asked of pandas as well, side by side."""
    questions = [
        lambda d: list(d.drop(columns=["v"]).columns),
        lambda d: list(d.drop("s", axis=1).columns),
        lambda d: list(d.drop([0, 3]).index),
        lambda d: d.drop([0, 3])["v"].tolist(),
        lambda d: list(d.drop(index=[0], columns=["s"]).columns),
        lambda d: list(d.drop(index=[0], columns=["s"]).index),
        lambda d: list(d.drop(columns=["x"], errors="ignore").columns),
        lambda d: list(d.drop([99], errors="ignore").index),
        lambda d: d.drop([0]).index.name,
        lambda d: d.drop(columns=["v", "s"]).shape,
        lambda d: d.drop([2, 0, 3, 1]).shape,
        lambda d: d["v"].drop([0]).tolist(),
        lambda d: d["v"].drop(columns=["anything"]).tolist(),
    ]
    for ask in questions:
        assert ask(frame(firepanda)) == ask(frame(pd))
