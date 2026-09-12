"""Putting an answer back into the object it was asked of.

`inplace=True` was refused everywhere in this library on the grounds that the
Arrow buffers underneath are shared rather than owned. Document 51 is why that
was answering a question pandas stopped asking in 3.0. Under copy on write an
inplace call there is not seen by a column taken out of the frame beforehand,
by a copy, or by the frame when the call was made on one of its columns, so the
only thing that can see it is a second name for the same object.

A wrapper here holds one slot, which is the extension object under it, so
putting an answer back is writing that slot. These tests are mostly about the
two things that are easy to get wrong once that is settled. The first is which
methods answer `None` and which answer the object, which is an undocumented
split in pandas that `Series.rename` sits on the far side of from
`DataFrame.rename`. The second is what pandas will accept as a flag, where a one
is refused, a `numpy.bool_` is taken, and None is taken and means False.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

# The methods that answer nothing once they have settled, as pandas does.
SETTLED = (
    ("drop", lambda m: m.drop(columns=["b"], inplace=True)),
    ("dropna", lambda m: m.dropna(inplace=True)),
    ("drop_duplicates", lambda m: m.drop_duplicates(inplace=True)),
    ("sort_values", lambda m: m.sort_values("a", inplace=True)),
    ("sort_index", lambda m: m.sort_index(inplace=True)),
    ("reset_index", lambda m: m.reset_index(drop=True, inplace=True)),
    ("set_index", lambda m: m.set_index("a", inplace=True)),
    ("rename", lambda m: m.rename(columns={"a": "z"}, inplace=True)),
    ("rename_axis", lambda m: m.rename_axis("r", inplace=True)),
)

# The methods that hand the object back instead, which is pandas' other half.
KEPT = (
    ("fillna", lambda m: m.fillna(0, inplace=True)),
    ("ffill", lambda m: m.ffill(inplace=True)),
    ("bfill", lambda m: m.bfill(inplace=True)),
    ("clip", lambda m: m.clip(0, inplace=True)),
    ("replace", lambda m: m.replace(1.0, 0.0, inplace=True)),
    ("where", lambda m: m.where(m.notna(), 0, inplace=True)),
    ("mask", lambda m: m.mask(m.isna(), 0, inplace=True)),
)

COLUMN_SETTLED = (
    ("dropna", lambda m: m.dropna(inplace=True)),
    ("drop_duplicates", lambda m: m.drop_duplicates(inplace=True)),
    ("sort_values", lambda m: m.sort_values(inplace=True)),
    ("sort_index", lambda m: m.sort_index(inplace=True)),
    ("reset_index", lambda m: m.reset_index(drop=True, inplace=True)),
    ("rename_axis", lambda m: m.rename_axis("r", inplace=True)),
)

COLUMN_KEPT = (*KEPT, ("rename", lambda m: m.rename("z", inplace=True)))


def frame(module):
    """Two columns with a gap in one of them, so every method has work to do."""
    return module.DataFrame({"a": [1.0, None, 3.0], "b": [4.0, 5.0, 6.0]})


def column(module):
    """The same values as a column, built the way each library builds one."""
    return frame(module)["a"]


def values(made):
    """A column as a list with a missing value spelled one way.

    A gap is a real Arrow null here and a NaN in pandas, which document 47 is
    about and this slice is not, so both are read as None before comparing.
    """
    return [None if one is None or one != one else one for one in made.tolist()]


def cells(made):
    """Every column of a frame as a list, for comparing two frames by value."""
    return {str(name): values(made[name]) for name in made.columns}


def test_the_answer_is_put_into_the_frame_that_was_asked(firepanda):
    made = frame(firepanda)
    made.fillna(0, inplace=True)
    assert made["a"].tolist() == [1.0, 0.0, 3.0]


def test_the_answer_is_put_into_the_column_that_was_asked(firepanda):
    made = column(firepanda)
    made.fillna(0, inplace=True)
    assert made.tolist() == [1.0, 0.0, 3.0]


def test_a_column_taken_out_beforehand_does_not_see_it(firepanda):
    """Which is pandas under copy on write, and here is the same answer."""
    made = frame(firepanda)
    taken = made["a"]
    made.fillna(0, inplace=True)
    assert taken.tolist() == [1.0, None, 3.0]
    assert made["a"].tolist() == [1.0, 0.0, 3.0]


def test_the_frame_does_not_see_a_call_made_on_one_of_its_columns(firepanda):
    made = frame(firepanda)
    taken = made["a"]
    taken.fillna(0, inplace=True)
    assert taken.tolist() == [1.0, 0.0, 3.0]
    assert made["a"].tolist() == [1.0, None, 3.0]


def test_a_copy_does_not_see_it(firepanda):
    made = frame(firepanda)
    kept = made.copy()
    made.fillna(0, inplace=True)
    assert kept["a"].tolist() == [1.0, None, 3.0]


def test_a_second_name_for_the_same_frame_sees_it(firepanda):
    """Because it is the same object, which is the only observer either library has."""
    made = frame(firepanda)
    other = made
    made.drop(columns=["b"], inplace=True)
    assert list(other.columns) == ["a"]


def test_nothing_comes_back_from_the_methods_pandas_answers_nothing_from(firepanda):
    for name, run in SETTLED:
        assert run(frame(firepanda)) is None, name
    for name, run in COLUMN_SETTLED:
        assert run(column(firepanda)) is None, name


def test_the_object_comes_back_from_the_methods_pandas_hands_it_back_from(firepanda):
    for name, run in KEPT:
        made = frame(firepanda)
        assert run(made) is made, name
    for name, run in COLUMN_KEPT:
        made = column(firepanda)
        assert run(made) is made, name


def test_a_flag_that_is_a_number_is_refused(firepanda):
    """Even a one, which every other check in the library would have let through."""
    with pytest.raises(firepanda.errors.InvalidArgumentError) as raised:
        frame(firepanda).drop(columns=["b"], inplace=1)
    assert str(raised.value) == 'For argument "inplace" expected type bool, received type int.'


def test_a_flag_that_is_a_word_is_refused(firepanda):
    with pytest.raises(ValueError) as raised:
        frame(firepanda).drop(columns=["b"], inplace="yes")
    assert str(raised.value) == 'For argument "inplace" expected type bool, received type str.'


def test_nothing_is_a_flag_and_it_means_false(firepanda):
    made = frame(firepanda)
    assert made.drop(columns=["b"], inplace=None) is not None
    assert list(made.columns) == ["a", "b"]


def test_a_numpy_flag_is_a_flag(firepanda):
    made = frame(firepanda)
    assert made.drop(columns=["b"], inplace=np.bool_(True)) is None
    assert list(made.columns) == ["a"]


def test_a_column_that_would_answer_a_frame_refuses_to_settle(firepanda):
    with pytest.raises(TypeError) as raised:
        column(firepanda).reset_index(inplace=True)
    assert str(raised.value) == "Cannot reset_index inplace on a Series to create a DataFrame"


def test_the_index_methods_that_already_did_this_still_do(firepanda):
    labels = firepanda.DataFrame({"a": [1, 2]}).set_index("a").index
    assert labels.rename("b", inplace=True) is None
    assert labels.name == "b"
    assert labels.set_names("c", inplace=True) is None
    assert labels.name == "c"


def test_both_libraries_answer_the_same_things(firepanda):
    for name, run in (*SETTLED, *KEPT):
        mine, theirs = frame(firepanda), frame(pd)
        run(mine)
        run(theirs)
        assert cells(mine) == cells(theirs), name
    for name, run in (*COLUMN_SETTLED, *COLUMN_KEPT):
        mine, theirs = column(firepanda), column(pd)
        run(mine)
        run(theirs)
        assert values(mine) == values(theirs), name


def test_both_libraries_hand_back_the_same_kind_of_thing(firepanda):
    for name, run in (*SETTLED, *KEPT):
        mine, theirs = frame(firepanda), frame(pd)
        assert (run(mine) is None) == (run(theirs) is None), name
    for name, run in (*COLUMN_SETTLED, *COLUMN_KEPT):
        mine, theirs = column(firepanda), column(pd)
        assert (run(mine) is None) == (run(theirs) is None), name


def test_both_libraries_refuse_the_same_flags(firepanda):
    questions = [
        lambda m: frame(m).drop(columns=["b"], inplace=1),
        lambda m: frame(m).drop(columns=["b"], inplace="yes"),
        lambda m: frame(m).fillna(0, inplace=0),
        lambda m: column(m).sort_values(inplace=np.int64(1)),
        lambda m: column(m).reset_index(inplace=True),
    ]
    for question in questions:
        with pytest.raises(Exception) as mine:
            question(firepanda)
        with pytest.raises(Exception) as theirs:
            question(pd)
        assert isinstance(mine.value, type(theirs.value)), question
        assert str(mine.value) == str(theirs.value), question
