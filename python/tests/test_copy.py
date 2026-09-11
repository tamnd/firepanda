"""Copying, in a library where nothing can be written into.

pandas' `copy` exists so that a later assignment into one object is not seen by
the other. There is no later assignment here, so what is worth testing is not
that the data survived the copy but that the copy is a separate object, answers
everything the original answers, and is not affected by anything the original is
put through afterwards.
"""

from __future__ import annotations

import copy as copying

import pandas as pd
import pytest


def frame(module):
    """Three columns of three different types, under labels of their own."""
    return module.DataFrame(
        {
            "k": [2, 0, 3, 1],
            "v": [4.0, 5.0, 6.0, 7.0],
            "s": ["a", "b", "c", "d"],
        }
    ).set_index("k")


def test_a_copied_frame_is_a_different_object_with_the_same_contents(firepanda):
    made = frame(firepanda)
    copied = made.copy()
    assert copied is not made
    assert copied["v"].tolist() == made["v"].tolist()
    assert copied["s"].tolist() == made["s"].tolist()
    assert list(copied.index) == list(made.index)
    assert list(copied.columns) == list(made.columns)


def test_the_original_is_untouched_by_what_the_copy_is_put_through(firepanda):
    made = frame(firepanda)
    copied = made.copy()
    copied.sort_values("v", ascending=False)
    copied.head(1)
    assert made["v"].tolist() == frame(pd)["v"].tolist()


def test_deep_is_accepted_both_ways_and_answers_the_same_thing(firepanda):
    made = frame(firepanda)
    assert made.copy(deep=False)["v"].tolist() == made.copy(deep=True)["v"].tolist()
    assert made.copy(False)["v"].tolist() == made.copy()["v"].tolist()


def test_a_copied_column_keeps_its_name_and_its_labels(firepanda):
    made = frame(firepanda)["v"]
    want = frame(pd)["v"]
    copied = made.copy()
    assert copied is not made
    assert copied.tolist() == want.copy().tolist()
    assert list(copied.index) == list(want.copy().index)
    assert copied.name == want.copy().name == "v"


def test_a_column_takes_deep_the_same_way_the_frame_does(firepanda):
    made = frame(firepanda)["v"]
    assert made.copy(deep=False).tolist() == made.copy(deep=True).tolist()


def test_a_copied_index_is_not_the_same_index_underneath(firepanda):
    made = frame(firepanda).index
    want = frame(pd).index
    assert list(made.copy()) == list(want.copy())
    assert made.copy().is_(made) is want.copy().is_(want) is False
    assert made.is_(made) is want.is_(want) is True


def test_an_index_can_be_copied_under_a_new_name(firepanda):
    made = frame(firepanda).index
    want = frame(pd).index
    assert made.copy(name="row").name == want.copy(name="row").name == "row"
    assert made.name == want.name == "k"


def test_the_copy_module_reaches_all_three(firepanda):
    made = frame(firepanda)
    assert copying.copy(made)["v"].tolist() == made["v"].tolist()
    assert copying.deepcopy(made)["v"].tolist() == made["v"].tolist()
    assert copying.copy(made["v"]).tolist() == made["v"].tolist()
    assert copying.deepcopy(made["v"]).tolist() == made["v"].tolist()
    assert list(copying.copy(made.index)) == list(made.index)
    assert list(copying.deepcopy(made.index)) == list(made.index)


def test_the_copy_module_hands_back_the_types_it_was_given(firepanda):
    made = frame(firepanda)
    assert type(copying.copy(made)) is type(made)
    assert type(copying.deepcopy(made["v"])) is type(made["v"])
    assert type(copying.copy(made.index)) is type(made.index)


def test_an_empty_frame_copies(firepanda):
    made = firepanda.DataFrame({"a": []})
    assert len(made.copy()) == len(pd.DataFrame({"a": []}).copy()) == 0


def test_copy_takes_no_argument_pandas_does_not_take(firepanda):
    with pytest.raises(TypeError):
        frame(firepanda).copy(sideways=True)
    with pytest.raises(TypeError):
        frame(pd).copy(sideways=True)
