"""Moving a column into the row labels, and putting the labels back.

`set_index` and `reset_index` are a pair and are tested as one, because almost
every bug either of them can have shows up as a round trip that does not come
back to where it started. The column has to come back under the name it left
under, it has to come back in the first position rather than the last, and the
labels underneath have to go back to counting from zero.

`sort_index` is here because it is the third thing that makes a label and a
position agree, and because it is the operation a caller reaches for after the
other two have made them disagree.

Every answer is measured against a running pandas rather than against a written
down constant, for the reason `test_astype.py` gives. The comparison goes through
Arrow, because a firepanda frame does not convert itself to a pandas one and
should not learn how to.
"""

from __future__ import annotations

import importlib.util
from types import ModuleType
from typing import Any

import pytest

needs_pandas = pytest.mark.skipif(
    importlib.util.find_spec("pandas") is None, reason="pandas is not installed"
)

ROWS: dict[str, list[Any]] = {
    "k": [30, 10, 20, 10, 30],
    "v": [1, 2, 3, 4, 5],
    "w": [1.5, 2.5, 3.5, 4.5, 5.5],
}
"""Five rows, three columns, and a key that repeats and is out of order.

The repeats matter because a label that appears twice is an ordinary thing that
nothing is allowed to refuse, and the disorder matters because it is what makes
`sort_index` do something.
"""


def made(firepanda: ModuleType) -> Any:
    """The frame, in firepanda."""
    return firepanda.DataFrame(ROWS)


def theirs() -> Any:
    """The same frame in pandas."""
    import pandas as pd

    return pd.DataFrame(ROWS)


def same(mine: Any, them: Any) -> None:
    """Asserts that a firepanda frame and a pandas one hold the same thing.

    The labels are compared as a list and the values through Arrow, which is the
    only conversion a firepanda frame offers and is the one the rest of the
    Python tests use.
    """
    import pyarrow as pa

    assert list(mine.index) == list(them.index)
    assert list(mine.columns) == list(them.columns)
    got = pa.table(mine).to_pandas()
    for name in them.columns:
        assert list(got[name]) == list(them[name]), name


@needs_pandas
def test_set_index_moves_a_column_into_the_labels(firepanda: ModuleType) -> None:
    """The column leaves and the labels arrive, which is one move and not two."""
    same(made(firepanda).set_index("k"), theirs().set_index("k"))


@needs_pandas
def test_set_index_can_leave_the_column_where_it_is(firepanda: ModuleType) -> None:
    """And then the values are in the frame twice, which is what was asked for."""
    same(made(firepanda).set_index("k", drop=False), theirs().set_index("k", drop=False))


@needs_pandas
def test_set_index_takes_a_list_of_one(firepanda: ModuleType) -> None:
    """pandas takes a label or a list, and a list of one means the same thing."""
    same(made(firepanda).set_index(["k"]), theirs().set_index(["k"]))


@needs_pandas
def test_set_index_names_the_level_after_the_column(firepanda: ModuleType) -> None:
    """The name is what makes the round trip back through `reset_index` work."""
    assert made(firepanda).set_index("k").index.name == theirs().set_index("k").index.name


@needs_pandas
def test_set_index_keeps_duplicate_labels(firepanda: ModuleType) -> None:
    """An index with repeats is ordinary and nothing here is allowed to refuse it."""
    assert list(made(firepanda).set_index("k").index) == [30, 10, 20, 10, 30]


@needs_pandas
def test_reset_index_puts_the_labels_back_as_the_first_column(firepanda: ModuleType) -> None:
    """The round trip, and the position matters as much as the name does."""
    same(made(firepanda).set_index("k").reset_index(), theirs().set_index("k").reset_index())


@needs_pandas
def test_reset_index_can_throw_the_labels_away(firepanda: ModuleType) -> None:
    same(
        made(firepanda).set_index("k").reset_index(drop=True),
        theirs().set_index("k").reset_index(drop=True),
    )


@needs_pandas
def test_reset_index_names_an_unnamed_level_index(firepanda: ModuleType) -> None:
    """A frame nobody ever indexed still gets a column called `index` out of this."""
    same(made(firepanda).tail(3).reset_index(), theirs().tail(3).reset_index())


@needs_pandas
def test_reset_index_refuses_to_make_two_columns_with_one_name(firepanda: ModuleType) -> None:
    """pandas refuses this too, rather than answering a frame nobody can read."""
    with pytest.raises(ValueError, match="already exists"):
        made(firepanda).set_index("k", drop=False).reset_index()


@needs_pandas
def test_sort_index_carries_the_rows_with_their_labels(firepanda: ModuleType) -> None:
    """The whole point, and what makes a label and a position agree again.

    The round trip is written out rather than stopping at the sort, because a
    sort that moved the labels and left the rows behind would still have sorted
    labels and would be wrong about every value under them.
    """
    same(
        made(firepanda).set_index("k").sort_index().reset_index(),
        theirs().set_index("k").sort_index().reset_index(),
    )


@needs_pandas
def test_sort_index_can_run_the_other_way(firepanda: ModuleType) -> None:
    same(made(firepanda).sort_index(ascending=False), theirs().sort_index(ascending=False))


@needs_pandas
def test_sort_index_orders_labels_that_came_from_a_column(firepanda: ModuleType) -> None:
    """The labels are not a range here, so the sort has to actually look at them."""
    same(
        made(firepanda).set_index("k").sort_index(),
        theirs().set_index("k").sort_index(),
    )


@needs_pandas
def test_sort_index_accepts_the_sort_kinds_and_ignores_them(firepanda: ModuleType) -> None:
    """The one argument in the library that is taken and never looked at.

    The four names are numpy's sort algorithms, the sort underneath is stable
    whichever of them is asked for, and a stable order is a correct answer to all
    four, so there is nothing a caller could observe about the difference.
    """
    for kind in ("quicksort", "mergesort", "heapsort", "stable"):
        same(made(firepanda).sort_index(kind=kind), theirs().sort_index(kind=kind))


@needs_pandas
def test_more_than_one_key_labels_the_rows_by_both(firepanda: ModuleType) -> None:
    """The answer is labelled by a MultiIndex of the two columns."""
    got = made(firepanda).set_index(["k", "v"])
    assert got.index.names == ["k", "v"]


@needs_pandas
def test_the_arguments_that_are_not_implemented_say_so(firepanda: ModuleType) -> None:
    """Each one by name, rather than as an unexpected keyword."""
    with pytest.raises(NotImplementedError, match="level"):
        made(firepanda).reset_index(level=0)


@needs_pandas
def test_sorting_the_columns_orders_the_column_names(firepanda: ModuleType) -> None:
    """`axis=1` sorts the column names rather than the row labels."""
    frame = made(firepanda)
    assert list(frame.sort_index(axis=1).columns) == sorted(frame.columns)
    assert list(frame.sort_index(axis=1, ascending=False).columns) == sorted(
        frame.columns, reverse=True
    )


@needs_pandas
def test_searchsorted_finds_where_a_label_would_have_to_go(firepanda: ModuleType) -> None:
    """One label gives one answer and a list of them gives a list."""
    import pandas as pd

    mine = firepanda.Index([10, 20, 20, 40])
    them = pd.Index([10, 20, 20, 40])
    assert mine.searchsorted(30) == them.searchsorted(30)
    assert list(mine.searchsorted([5, 20, 50])) == list(them.searchsorted([5, 20, 50]))


@needs_pandas
def test_searchsorted_sides_pick_the_two_ends_of_a_run(firepanda: ModuleType) -> None:
    """The whole reason `side` exists, and it only shows on a repeated label."""
    import pandas as pd

    mine = firepanda.Index([10, 20, 20, 40])
    them = pd.Index([10, 20, 20, 40])
    assert mine.searchsorted(20, "left") == them.searchsorted(20, "left")
    assert mine.searchsorted(20, "right") == them.searchsorted(20, "right")


@needs_pandas
def test_searchsorted_refuses_a_side_it_does_not_know(firepanda: ModuleType) -> None:
    with pytest.raises(ValueError, match="search side must be 'left' or 'right'"):
        firepanda.Index([10, 20]).searchsorted(15, "middle")


@needs_pandas
def test_isin_answers_one_bool_per_label(firepanda: ModuleType) -> None:
    """A list of them where pandas gives a numpy array, which document 21 records."""
    import pandas as pd

    mine = firepanda.Index([10, 20, 30])
    them = pd.Index([10, 20, 30])
    assert mine.isin([30, 10]).tolist() == list(them.isin([30, 10]))
    assert mine.isin([]).tolist() == list(them.isin([]))


@needs_pandas
def test_isin_is_not_bothered_by_a_value_it_could_never_hold(firepanda: ModuleType) -> None:
    """Not an error and simply not found, which is what makes a mixed list work."""
    import pandas as pd

    mine = firepanda.Index([10, 20])
    them = pd.Index([10, 20])
    assert mine.isin(["a", 20]).tolist() == list(them.isin(["a", 20]))


def test_verify_integrity_checks_the_new_labels(firepanda: ModuleType) -> None:
    """Labels each seen once are kept, and a repeated one is refused in pandas' words."""
    frame = firepanda.DataFrame({"k": [1, 2, 3], "v": [4, 5, 6]})
    assert frame.set_index("k", verify_integrity=True).index.tolist() == [1, 2, 3]
    repeated = firepanda.DataFrame({"k": [1, 1, 3], "v": [4, 5, 6]})
    with pytest.raises(ValueError, match="Index has duplicate keys"):
        repeated.set_index("k", verify_integrity=True)


def test_a_frame_of_repeated_column_labels_is_refused(firepanda: ModuleType) -> None:
    """Rather than keeping one of the columns and dropping the other."""
    with pytest.raises(NotImplementedError, match="column labels repeat"):
        firepanda.DataFrame([[1, 2]], columns=["x", "x"])
