"""`reindex` on an index, against pandas.

This one is not a smaller version of the frame's method, it is the lookup that
the frame's method is built on, handed back in the open. It answers a pair: the
labels asked for, and where each of them sits in the index that was asked. The
second half of the pair is what a caller would use to move their own rows.

The part worth testing carefully is when the lookup is `None`. pandas leaves it
out when the target is the same index it already had, because then nothing has
to move, and a caller who checks for that can skip the work entirely. An empty
target is not the same thing: there the lookup is a list of no positions, which
says move nothing rather than move everything where it already is.
"""

from __future__ import annotations

import pandas as pd
import pytest

LABELS = [10, 20, 30]


def made(firepanda):
    """The index under test, named key."""
    return firepanda.DataFrame({"key": LABELS, "count": [1, 2, 3]}).set_index("key").index


def theirs():
    """The same index in pandas."""
    return pd.DataFrame({"key": LABELS, "count": [1, 2, 3]}).set_index("key").index


def test_the_labels_asked_for_come_back_with_where_they_sit(firepanda):
    got, where = made(firepanda).reindex([30, 10])
    want, theirs_where = theirs().reindex([30, 10])
    assert list(got) == list(want)
    assert where == list(theirs_where)


def test_a_label_the_index_does_not_have_sits_nowhere(firepanda):
    got, where = made(firepanda).reindex([10, 99])
    assert list(got) == [10, 99]
    assert where == [0, -1]


def test_an_index_asked_for_itself_has_nothing_to_move(firepanda):
    # The one case that answers no lookup at all, and the reason a caller checks
    # for it: there is nothing to do, so there is nothing to hand back.
    got, where = made(firepanda).reindex(made(firepanda))
    assert list(got) == LABELS
    assert where is None
    _, theirs_where = theirs().reindex(theirs())
    assert theirs_where is None


def test_asking_for_no_labels_answers_a_lookup_of_no_positions(firepanda):
    # Not the same as the case above. Nothing moves here either, but it says so
    # by listing no positions rather than by leaving the list out.
    got, where = made(firepanda).reindex([])
    assert len(got) == 0
    assert where == []
    _, theirs_where = theirs().reindex([])
    assert list(theirs_where) == []


def test_a_bare_list_of_labels_takes_the_name_the_index_had(firepanda):
    got, _ = made(firepanda).reindex([30, 10])
    assert got.name == "key"
    want, _ = theirs().reindex([30, 10])
    assert want.name == "key"


def test_an_index_handed_in_keeps_its_own_name(firepanda):
    other = firepanda.DataFrame({"other": [30, 10], "count": [1, 2]}).set_index("other").index
    got, _ = made(firepanda).reindex(other)
    assert got.name == "other"


def test_a_label_asked_for_twice_is_looked_up_twice(firepanda):
    got, where = made(firepanda).reindex([20, 20])
    assert list(got) == [20, 20]
    assert where == [1, 1]


def test_a_repeated_label_in_the_index_is_refused(firepanda):
    twice = firepanda.DataFrame({"key": [10, 10], "count": [1, 2]}).set_index("key").index
    with pytest.raises(ValueError, match="cannot reindex on an axis with duplicate labels"):
        twice.reindex([10])


def test_filling_from_the_label_beside_it_is_not_done_here(firepanda):
    with pytest.raises(NotImplementedError, match="method"):
        made(firepanda).reindex([10, 99], method="ffill")


def test_a_limit_or_a_tolerance_is_refused_because_there_is_no_filling(firepanda):
    with pytest.raises(NotImplementedError, match="limit"):
        made(firepanda).reindex([10], limit=1)
    with pytest.raises(NotImplementedError, match="tolerance"):
        made(firepanda).reindex([10], tolerance=1)


def test_a_label_of_the_wrong_type_is_refused(firepanda):
    # pandas answers minus one for every label here. We refuse, for the reason
    # document 40 section 7 gives: the lookup compares both sets of labels in
    # one column and there is no column that holds a number and a word.
    with pytest.raises(TypeError):
        made(firepanda).reindex(["10"])
