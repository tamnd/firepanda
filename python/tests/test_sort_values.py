"""Sorting by values rather than by labels, against pandas.

The core has had a multi key sort and a one key sort for a long time and neither
was reachable from Python. What is worth testing is the part that is new, which
is the translation: pandas names the direction it sorts in and the core names
the other one, pandas takes one direction or one per key, pandas puts the
missing values where a word says, and pandas can number the rows again
afterwards.
"""

from __future__ import annotations

import pandas as pd
import pytest


def pairs(module):
    """Two columns whose orders disagree, which is what a second key is for."""
    return module.DataFrame(
        {
            "k": ["b", "a", "b", "a", "c"],
            "v": [2, 3, 1, 4, 0],
        }
    )


def gapped(module):
    """A column with a hole in it, so that na_position has something to move."""
    return module.DataFrame({"v": [3.0, None, 1.0, None, 2.0]})


def shuffled(module):
    """A column of four values, not in order, under labels not in order."""
    return module.DataFrame({"k": [2, 0, 3, 1], "v": [30, 10, 40, 20]}).set_index("k")["v"]


def test_one_key_puts_the_rows_in_the_order_of_that_column(firepanda):
    got = pairs(firepanda).sort_values("v")
    want = pairs(pd).sort_values("v")
    assert got["v"].tolist() == want["v"].tolist() == [0, 1, 2, 3, 4]
    assert got["k"].tolist() == want["k"].tolist()
    assert list(got.index) == list(want.index)


def test_a_list_of_one_name_means_the_same_as_the_name(firepanda):
    got = pairs(firepanda).sort_values(["v"])
    want = pairs(pd).sort_values(["v"])
    assert got["v"].tolist() == want["v"].tolist()


def test_the_second_key_settles_what_the_first_left_tied(firepanda):
    got = pairs(firepanda).sort_values(["k", "v"])
    want = pairs(pd).sort_values(["k", "v"])
    assert got["k"].tolist() == want["k"].tolist()
    assert got["v"].tolist() == want["v"].tolist() == [3, 4, 1, 2, 0]


def test_a_direction_per_key_runs_them_different_ways(firepanda):
    got = pairs(firepanda).sort_values(["k", "v"], ascending=[True, False])
    want = pairs(pd).sort_values(["k", "v"], ascending=[True, False])
    assert got["k"].tolist() == want["k"].tolist()
    assert got["v"].tolist() == want["v"].tolist() == [4, 3, 2, 1, 0]


def test_one_direction_covers_every_key(firepanda):
    got = pairs(firepanda).sort_values(["k", "v"], ascending=False)
    want = pairs(pd).sort_values(["k", "v"], ascending=False)
    assert got["k"].tolist() == want["k"].tolist()
    assert got["v"].tolist() == want["v"].tolist()


def test_a_direction_list_of_the_wrong_length_is_refused(firepanda):
    with pytest.raises(ValueError, match="ascending"):
        pairs(firepanda).sort_values(["k", "v"], ascending=[True])
    with pytest.raises(ValueError, match="ascending"):
        pairs(pd).sort_values(["k", "v"], ascending=[True])


def test_the_missing_values_go_where_they_are_told(firepanda):
    got = gapped(firepanda).sort_values("v")
    want = gapped(pd).sort_values("v")
    assert got["v"].tolist()[:3] == want["v"].tolist()[:3] == [1.0, 2.0, 3.0]
    first = gapped(firepanda).sort_values("v", na_position="first")
    wanted = gapped(pd).sort_values("v", na_position="first")
    assert first["v"].tolist()[2:] == wanted["v"].tolist()[2:] == [1.0, 2.0, 3.0]


def test_a_word_that_is_neither_position_is_refused(firepanda):
    with pytest.raises(ValueError, match="na_position"):
        gapped(firepanda).sort_values("v", na_position="middle")


def test_ignore_index_numbers_the_rows_again(firepanda):
    got = pairs(firepanda).sort_values("v", ignore_index=True)
    want = pairs(pd).sort_values("v", ignore_index=True)
    assert list(got.index) == list(want.index) == [0, 1, 2, 3, 4]


def test_a_name_that_is_not_a_column_is_a_key_error(firepanda):
    with pytest.raises(KeyError):
        pairs(firepanda).sort_values("nope")


def test_a_column_sorts_by_its_own_values_and_keeps_its_labels(firepanda):
    got = shuffled(firepanda).sort_values()
    want = shuffled(pd).sort_values()
    assert got.tolist() == want.tolist() == [10, 20, 30, 40]
    assert list(got.index) == list(want.index) == [0, 1, 2, 3]
    assert got.name == want.name == "v"


def test_a_column_can_sort_the_other_way(firepanda):
    got = shuffled(firepanda).sort_values(ascending=False)
    want = shuffled(pd).sort_values(ascending=False)
    assert got.tolist() == want.tolist() == [40, 30, 20, 10]
    assert list(got.index) == list(want.index)


def test_a_column_can_be_numbered_again_after_sorting(firepanda):
    got = shuffled(firepanda).sort_values(ignore_index=True)
    want = shuffled(pd).sort_values(ignore_index=True)
    assert list(got.index) == list(want.index) == [0, 1, 2, 3]


def test_a_column_puts_its_missing_values_where_it_is_told(firepanda):
    made = gapped(firepanda)["v"]
    wanted = gapped(pd)["v"]
    assert made.sort_values().tolist()[:3] == wanted.sort_values().tolist()[:3]
    assert (
        made.sort_values(na_position="first").tolist()[2:]
        == wanted.sort_values(na_position="first").tolist()[2:]
    )


def test_argsort_gives_the_positions_and_keeps_the_labels(firepanda):
    got = shuffled(firepanda).argsort()
    want = shuffled(pd).argsort()
    assert got.tolist() == want.tolist() == [1, 3, 0, 2]
    assert list(got.index) == list(want.index) == [2, 0, 3, 1]
    assert got.name == want.name == "v"


def test_argsort_takes_the_names_numpy_had_and_reads_neither(firepanda):
    plain = shuffled(firepanda).argsort()
    assert shuffled(firepanda).argsort(kind="mergesort").tolist() == plain.tolist()
    assert shuffled(firepanda).argsort(stable=True).tolist() == plain.tolist()


def test_an_index_sorts_its_labels(firepanda):
    made = firepanda.DataFrame({"k": [2, 0, 3, 1], "v": [1, 2, 3, 4]}).set_index("k").index
    want = pd.DataFrame({"k": [2, 0, 3, 1], "v": [1, 2, 3, 4]}).set_index("k").index
    assert list(made.sort_values()) == list(want.sort_values()) == [0, 1, 2, 3]
    assert list(made.sort_values(ascending=False)) == list(want.sort_values(ascending=False))


def test_an_index_can_hand_back_the_order_that_sorted_it(firepanda):
    made = firepanda.DataFrame({"k": [2, 0, 3, 1], "v": [1, 2, 3, 4]}).set_index("k").index
    want = pd.DataFrame({"k": [2, 0, 3, 1], "v": [1, 2, 3, 4]}).set_index("k").index
    ordered, found = made.sort_values(return_indexer=True)
    other, expected = want.sort_values(return_indexer=True)
    assert list(ordered) == list(other)
    assert found == list(expected) == [1, 3, 0, 2]


def test_an_index_of_instants_stays_one_after_sorting(firepanda):
    import datetime

    import pyarrow as pa

    stamps = [datetime.datetime(2024, 1, 3), datetime.datetime(2024, 1, 1)]
    table = pa.table({"t": pa.array(stamps, type=pa.timestamp("us")), "v": [1, 2]})
    made = firepanda.from_arrow(table).set_index("t").index
    assert type(made.sort_values()) is type(made)


def test_an_index_argsorts_to_a_plain_list(firepanda):
    made = firepanda.DataFrame({"k": [2, 0, 3, 1], "v": [1, 2, 3, 4]}).set_index("k").index
    want = pd.DataFrame({"k": [2, 0, 3, 1], "v": [1, 2, 3, 4]}).set_index("k").index
    assert made.argsort() == list(want.argsort()) == [1, 3, 0, 2]
