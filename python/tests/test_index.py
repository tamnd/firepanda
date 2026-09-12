"""The labels, from Python.

`df.index` is the second most written attribute in pandas after `df.columns`,
and until there was a bound index type there was nothing to hand back from it.
So these tests are about the shape of the thing as much as about the methods:
that an index comes out of a frame and out of a series, that it is the same
object both times, that indexing it gives back the three types pandas gives
back, and that the divergences we chose are the divergences that happen.

Every expected answer here was read off a running pandas rather than off its
documentation, which is the practice `docs/specs/19-two-indexes-combined.md`
adopted after the documentation turned out to be wrong about `intersection`.
The handful of places where the answer is deliberately not the pandas one say
so in the test that covers them.
"""

from __future__ import annotations

from pathlib import Path
from types import ModuleType

import pandas as pd
import pytest

CSV = "region,qty\nnorth,10\nsouth,20\neast,30\n"


def _frame(firepanda: ModuleType, tmp_path: Path) -> object:
    """Reads the small fixture the frame tests below work on."""
    path = tmp_path / "t.csv"
    path.write_text(CSV)
    return firepanda.read_csv(str(path))


def test_an_index_is_built_from_a_list(firepanda: ModuleType) -> None:
    """The constructor, which is the only way to get one without a frame."""
    index = firepanda.Index([3, 1, 2], name="k")
    assert isinstance(index, firepanda.Index)
    assert len(index) == 3
    assert index.name == "k"
    assert index.tolist() == [3, 1, 2]


def test_an_index_with_no_name_has_none(firepanda: ModuleType) -> None:
    """An unnamed level and one named with the empty string are different.

    pandas draws that distinction and so does the core, so the boundary has to
    carry both rather than folding the missing name onto `""` the way the series
    name is folded.
    """
    assert firepanda.Index([1]).name is None
    assert firepanda.Index([1], name="").name == ""


def test_an_index_is_built_from_a_series_and_from_another_index(
    firepanda: ModuleType,
) -> None:
    """Neither goes through the reader a list goes through, and both keep their type."""
    series = firepanda.Series([3, 1, 2], name="a")
    from_series = firepanda.Index(series)
    assert from_series.tolist() == [3, 1, 2]
    assert from_series.dtype == series.dtype == "int64"
    made = firepanda.Index([5, 6], name="k")
    assert firepanda.Index(made).tolist() == [5, 6]


def test_a_name_that_was_not_written_comes_off_the_data(firepanda: ModuleType) -> None:
    """pandas' rule, which is that the labels bring their name with them."""
    assert firepanda.Index(firepanda.Series([1, 2], name="a")).name == "a"
    assert firepanda.Index(firepanda.Index([1, 2], name="k")).name == "k"
    theirs = pd.Series([1, 2], name="a")
    assert pd.Index(theirs).name == "a"
    assert pd.Index(pd.Index([1, 2], name="k")).name == "k"


def test_a_name_in_the_call_wins_over_the_one_the_data_had(firepanda: ModuleType) -> None:
    assert firepanda.Index(firepanda.Series([1, 2], name="a"), name="z").name == "z"
    assert pd.Index(pd.Series([1, 2], name="a"), name="z").name == "z"


def test_a_series_with_no_name_and_one_called_nothing_are_two_things(
    firepanda: ModuleType,
) -> None:
    # A column with no name gives an index with no level name, and a column
    # called `""` gives a level called `""`. They used to land on the same
    # answer here, because a series name was a `String` and the empty one was
    # how it spelled an absence, and pandas has told them apart all along.
    assert firepanda.Index(firepanda.Series([1, 2])).name is None
    assert pd.Index(pd.Series([1, 2])).name is None
    assert firepanda.Index(firepanda.Series([1, 2], name="")).name == ""
    assert pd.Index(pd.Series([1, 2], name="")).name == ""


def test_the_labels_of_an_unnamed_list_still_have_no_name(firepanda: ModuleType) -> None:
    assert firepanda.Index([1, 2]).name is None
    assert pd.Index([1, 2]).name is None


def test_an_index_of_instants_takes_its_name_off_the_data_too(firepanda: ModuleType) -> None:
    made = firepanda.DatetimeIndex(firepanda.Series(["2024-01-02", "2024-03-04"], name="s"))
    assert made.name == "s"
    assert firepanda.DatetimeIndex(made).name == "s"
    assert firepanda.DatetimeIndex(made, name="when").name == "when"


def test_a_frame_and_its_columns_agree_about_the_labels(
    firepanda: ModuleType, tmp_path: Path
) -> None:
    """`df.index` and `df["a"].index` are the same labels.

    They are not the same object, which is a divergence: pandas answers True to
    `df.index.is_(df["a"].index)` because a frame and its columns hold one index
    between them, and here each access builds a wrapper around a copy. Making
    that true means the frame holding a shared index rather than an owned one,
    which is a change to the core and not to the binding, so it is recorded in
    document 21 and left for the issue that does it.
    """
    frame = _frame(firepanda, tmp_path)
    assert frame.index.equals(frame["qty"].index)
    assert not frame.index.is_(frame["qty"].index)


def test_a_fresh_frame_has_a_range_index(firepanda: ModuleType, tmp_path: Path) -> None:
    """A frame that was never reindexed carries no labels at all.

    The repr is the pandas `RangeIndex` one, because that is what a user sees
    and it is the cheapest way to notice that the range stayed a range.
    """
    frame = _frame(firepanda, tmp_path)
    assert repr(frame.index) == "RangeIndex(start=0, stop=3, step=1)"
    assert frame.index.tolist() == [0, 1, 2]


def test_a_range_index_reports_no_bytes(firepanda: ModuleType, tmp_path: Path) -> None:
    """A divergence, and one we prefer to the pandas answer.

    pandas says 132 for the `nbytes` of any `RangeIndex`, which is the size of
    the Python object rather than of any labels, and is the same number for
    three rows and for three hundred million. A range stores nothing, so this
    says nothing.
    """
    assert _frame(firepanda, tmp_path).index.nbytes == 0
    assert firepanda.Index([1, 2, 3]).nbytes > 0


def test_the_reported_type_is_both_spellings(firepanda: ModuleType) -> None:
    """`dtype` is what firepanda calls it and `inferred_type` is what pandas does."""
    assert firepanda.Index([1, 2]).dtype == "int64"
    assert firepanda.Index([1, 2]).inferred_type == "integer"
    assert firepanda.Index(["a"]).inferred_type == "string"
    assert firepanda.Index([1.5]).inferred_type == "floating"
    assert firepanda.Index([True]).inferred_type == "boolean"


def test_indexing_gives_back_the_three_things_pandas_gives_back(firepanda: ModuleType) -> None:
    """An integer, a slice and a list all mean something different here."""
    index = firepanda.Index([10, 20, 30, 40])
    assert index[1] == 20
    assert index[-1] == 40
    assert index[1:3].tolist() == [20, 30]
    assert index[::2].tolist() == [10, 30]
    assert index[[0, 3]].tolist() == [10, 40]
    assert index[[True, False, True, False]].tolist() == [10, 30]


def test_a_position_past_the_end_is_an_index_error(firepanda: ModuleType) -> None:
    """`except IndexError` around `index[i]` is an idiom and it has to keep working.

    The class is ours and it is an `IndexError`, so code that predates firepanda
    catches it without knowing anything about firepanda.
    """
    with pytest.raises(IndexError):
        firepanda.Index([1, 2])[5]
    assert issubclass(firepanda.errors.OutOfBoundsError, IndexError)


def test_membership_and_iteration(firepanda: ModuleType) -> None:
    """`in` and `for` both read the labels and neither raises on a bad type."""
    index = firepanda.Index(["a", "b"])
    assert "a" in index
    assert "z" not in index
    assert 7 not in index
    assert list(index) == ["a", "b"]


def test_get_loc_returns_an_int_a_slice_or_a_mask(firepanda: ModuleType) -> None:
    """One method, three return types, chosen by the labels rather than the argument.

    Callers branch on the type, so the rule matters. A run of duplicates is a
    slice only when the index is sorted, which is why the third case here is a
    mask despite its two hits being next to each other.
    """
    assert firepanda.Index(["a", "b", "c"]).get_loc("b") == 1
    assert firepanda.Index(["a", "b", "b", "c"]).get_loc("b") == slice(1, 3, None)
    assert firepanda.Index(["c", "b", "b", "a"]).get_loc("b") == [False, True, True, False]
    with pytest.raises(KeyError):
        firepanda.Index(["a"]).get_loc("z")


def test_get_indexer_marks_the_missing_with_minus_one(firepanda: ModuleType) -> None:
    """The lookup a reindex is made of, with the pandas convention for a miss."""
    assert firepanda.Index(["a", "b", "c"]).get_indexer(["c", "z", "a"]) == [2, -1, 0]


def test_the_set_operations_follow_the_pandas_sorting_rules(firepanda: ModuleType) -> None:
    """Three arbitrary rules, matched on purpose because pandas has them.

    A union sorts by default and keeps duplicates, an intersection and a
    difference keep the left side's order, and a union with an empty index comes
    back unsorted because pandas takes an early return before the sort.
    """
    left = firepanda.Index([3, 1, 2])
    assert left.union(firepanda.Index([2, 5])).tolist() == [1, 2, 3, 5]
    assert left.union(firepanda.Index([])).tolist() == [3, 1, 2]
    assert left.intersection([2, 1]).tolist() == [1, 2]
    assert left.difference([1]).tolist() == [2, 3]
    assert left.symmetric_difference([1, 9]).tolist() == [2, 3, 9]


def test_a_set_operation_takes_a_plain_list(firepanda: ModuleType) -> None:
    """pandas accepts a list wherever it accepts an index and so does this."""
    assert firepanda.Index([1, 2]).union([2, 3]).tolist() == [1, 2, 3]
    with pytest.raises(TypeError):
        firepanda.Index([1, 2]).union(7)


def test_the_editing_operations(firepanda: ModuleType) -> None:
    """The five ways to get a different index out of one, all copying."""
    index = firepanda.Index([3, 1, 2])
    assert index.append(firepanda.Index([7])).tolist() == [3, 1, 2, 7]
    assert index.append([firepanda.Index([7]), firepanda.Index([8])]).tolist() == [3, 1, 2, 7, 8]
    assert index.delete(0).tolist() == [1, 2]
    assert index.delete([0, 2]).tolist() == [1]
    assert index.insert(1, 99).tolist() == [3, 99, 1, 2]
    assert index.drop(1).tolist() == [3, 2]
    assert index.putmask([True, False, False], 8).tolist() == [8, 1, 2]
    assert index.tolist() == [3, 1, 2]


def test_drop_takes_the_pandas_errors_argument(firepanda: ModuleType) -> None:
    """`errors="ignore"` is the difference between a missing label mattering or not."""
    index = firepanda.Index([3, 1, 2])
    assert index.drop([9], errors="ignore").tolist() == [3, 1, 2]
    with pytest.raises(KeyError):
        index.drop([9])


def test_a_slice_of_labels_includes_both_ends(firepanda: ModuleType) -> None:
    """The one range in the library that is not half open.

    `df.loc["b":"d"]` means through d rather than up to it, which is what
    `slice_locs` and `slice_indexer` exist to express.
    """
    index = firepanda.Index(["a", "b", "c", "d"])
    assert index.slice_locs("b", "c") == (1, 3)
    assert index.slice_indexer("b", "c") == slice(1, 3, None)
    assert index.slice_indexer() == slice(0, 4, None)


def test_equals_ignores_the_name_and_identical_does_not(firepanda: ModuleType) -> None:
    """The two questions that look like one, and the third that is about identity."""
    left = firepanda.Index([1, 2], name="k")
    assert left.equals(firepanda.Index([1, 2]))
    assert not left.identical(firepanda.Index([1, 2]))
    assert left.identical(firepanda.Index([1, 2], name="k"))
    assert not left.equals([1, 2])
    assert not left.is_(left.copy())
    assert not left.is_(firepanda.Index([1, 2], name="k"))


def test_rename_and_copy_do_not_touch_the_original(firepanda: ModuleType) -> None:
    """An index is immutable, so everything here gives back a new one."""
    index = firepanda.Index([1, 2], name="k")
    assert index.rename("z").name == "z"
    assert index.copy().name == "k"
    assert index.copy(name="y").name == "y"
    assert index.name == "k"


def test_the_ordering_and_uniqueness_questions(firepanda: ModuleType) -> None:
    """What a sort, a join and a lookup all ask before deciding what to do."""
    assert firepanda.Index([1, 2, 3]).is_monotonic_increasing
    assert not firepanda.Index([1, 3, 2]).is_monotonic_increasing
    assert firepanda.Index([3, 2, 1]).is_monotonic_decreasing
    assert firepanda.Index([1, 2]).is_unique
    assert firepanda.Index([1, 1]).has_duplicates
    assert firepanda.Index([1, 1, 2]).unique().tolist() == [1, 2]
    assert firepanda.Index([]).empty


def test_comparing_an_index_gives_a_list_of_bools(firepanda: ModuleType) -> None:
    """A divergence, recorded rather than hidden.

    pandas gives back a numpy array here and this gives back a list, which is
    the same difference `values` has and is the whole of what numpy would buy
    at this point in the project.
    """
    index = firepanda.Index([1, 2, 3])
    assert (index == 2) == [False, True, False]
    assert (index != 2) == [True, False, True]
    assert (index == [1, 5, 3]) == [True, False, True]
    with pytest.raises(ValueError):
        index.__eq__([1, 2])


def test_an_index_is_not_a_truth_value(firepanda: ModuleType) -> None:
    """pandas refuses and the message is copied exactly, because people search for it."""
    with pytest.raises(ValueError, match="truth value of a Index is ambiguous"):
        bool(firepanda.Index([1, 2]))


def test_an_index_is_unhashable_the_way_pandas_is(firepanda: ModuleType) -> None:
    """A consequence of defining `__eq__`, and the same consequence pandas has."""
    with pytest.raises(TypeError):
        {firepanda.Index([1, 2]): 1}


def test_a_missing_label_keeps_its_dtype(firepanda: ModuleType) -> None:
    """The divergence document 20 already recorded, now visible from Python.

    pandas turns an int64 index with a null in it into float64 with a NaN,
    because a numpy int64 array has nowhere to record absence. An Arrow column
    does, so the dtype survives and the label is None.
    """
    index = firepanda.Index([3, 1]).insert(1, None)
    assert index.dtype == "int64"
    assert index.tolist() == [3, None, 1]
    assert index.hasnans


def test_a_null_label_is_found_rather_than_hidden(firepanda: ModuleType) -> None:
    """Another divergence from the same root.

    pandas answers False for `None in index` even when the index has a null in
    it, because the label it holds is NaN and NaN is not None. Here the label is
    a real absence and the honest answer is that it is there.
    """
    assert None in firepanda.Index([3, None, 1])


def test_an_index_exports_itself_as_arrow(firepanda: ModuleType) -> None:
    """An addition rather than a compatibility, and worth having.

    pandas has no `__arrow_c_array__` on an index at all, so this is the one
    place the surface is wider than the one it copies. A range has to be
    materialised to be exported, which is the only allocation on this path.
    """
    pyarrow = pytest.importorskip("pyarrow")
    assert pyarrow.array(firepanda.Index([3, 1, 2])).to_pylist() == [3, 1, 2]
    assert pyarrow.array(firepanda.Index(["a", "b"])).to_pylist() == ["a", "b"]


def test_the_constructor_refuses_what_it_does_not_do(firepanda: ModuleType) -> None:
    """A declared parameter that is silently ignored is worse than a missing one."""
    with pytest.raises(NotImplementedError, match="dtype="):
        firepanda.Index([1], dtype="int32")
    with pytest.raises(NotImplementedError, match="copy="):
        firepanda.Index([1], copy=True)
    with pytest.raises(NotImplementedError, match="tupleize_cols"):
        firepanda.Index([("a", 1)], tupleize_cols=False)


def test_get_indexer_refuses_the_filling_arguments(firepanda: ModuleType) -> None:
    """`method=` is how pandas fills a missing label from a neighbour, and it is not written."""
    with pytest.raises(NotImplementedError, match="method="):
        firepanda.Index([1, 2]).get_indexer([1], method="pad")


def test_taking_positions_counts_back_from_the_end_by_default(firepanda: ModuleType) -> None:
    """`take` with the default arguments does no filling, so -1 is the last row.

    This is the pandas reading of a pair of arguments that looks like one
    question. Filling only happens when `allow_fill` is on and a `fill_value`
    was actually passed, and the default pair passes no fill value, so a
    negative position counts back from the end the way it does everywhere else
    in Python.
    """
    index = firepanda.Index(["a", "b", "c"])
    assert index.take([0, -1]).tolist() == ["a", "c"]
    assert index.take([0, -1], allow_fill=False).tolist() == ["a", "c"]
    assert index.take([2, 0]).tolist() == ["c", "a"]


def test_taking_positions_with_a_fill_value_makes_a_gap(firepanda: ModuleType) -> None:
    """With filling on, -1 is a row that is not there rather than the last one.

    The fill value is read for whether it is there and not for what it is, which
    is pandas' own behaviour and is the surprising half: the label that lands in
    the gap is a missing label whatever value was named.
    """
    index = firepanda.Index(["a", "b", "c"])
    assert index.take([0, -1], fill_value="z").tolist() == ["a", None]
    with pytest.raises(ValueError, match="all indices must be"):
        index.take([0, -2], fill_value="z")


def test_taking_a_position_that_is_not_there_is_an_index_error(firepanda: ModuleType) -> None:
    """Past the end is past the end either way round."""
    with pytest.raises(IndexError):
        firepanda.Index(["a", "b"]).take([5])
    with pytest.raises(IndexError):
        firepanda.Index(["a", "b"]).take([-5], allow_fill=False)


def test_a_backward_slice_bound_is_a_different_question(firepanda: ModuleType) -> None:
    """A negative step means the caller names the labels in reading order.

    So the pair that comes back is the pair a backward Python slice wants, which
    is not the forward pair reversed. Every number here was read off a running
    pandas.
    """
    index = firepanda.Index(["a", "b", "c", "d", "e"])
    assert index.slice_locs("b", "d") == (1, 4)
    assert index.slice_locs("b", "d", step=1) == (1, 4)
    assert index.slice_locs("b", "d", step=2) == (1, 4)
    assert index.slice_locs("b", "d", step=-1) == (1, 2)
    assert index.slice_locs(step=-1) == (4, -6)


def test_the_one_level_a_flat_index_has(firepanda: ModuleType) -> None:
    """`unique` takes a level because a MultiIndex has levels, and this has one.

    `None`, `0` and `-1` all name it, and so does the index's own name. Anything
    else is an error and not a refusal, because the caller asked for a level
    that does not exist rather than for a feature that is not written.
    """
    index = firepanda.Index(["a", "b", "a"], name="k")
    assert index.unique().tolist() == ["a", "b"]
    assert index.unique(level=0).tolist() == ["a", "b"]
    assert index.unique(level=-1).tolist() == ["a", "b"]
    assert index.unique(level="k").tolist() == ["a", "b"]
    with pytest.raises(IndexError, match="only 1 level, not 2"):
        index.unique(level=1)
    with pytest.raises(IndexError, match="not a valid level number"):
        index.unique(level=-2)
    with pytest.raises(KeyError, match="does not match index name"):
        index.unique(level="nope")


def test_renaming_in_place_changes_the_index_you_are_holding(firepanda: ModuleType) -> None:
    """The one place an index is mutable, and it is mutable in pandas too.

    A level name is not a label, so changing it does not change what the index
    holds, and `inplace=True` answers None the way every pandas `inplace` does.
    """
    index = firepanda.Index([1, 2], name="k")
    assert index.rename("z", inplace=True) is None
    assert index.name == "z"
    assert index.tolist() == [1, 2]
    with pytest.raises(TypeError):
        index.rename("y", True)  # type: ignore[misc]
