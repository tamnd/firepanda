"""Tests for editing an index and for turning labels into slice bounds.

Every expected answer here was read off a running pandas 3.0.5 rather than off
its documentation, the same way the set operation tests were written, because
the documentation has already been wrong once about this type and there is no
reason to trust it more here.

Two families are worth reading as families. The editing operations are all one
gather away from something that already existed, so the cases that matter are
the ones about what a gather does not decide: which name survives, what an out
of bounds position does, and whether a missing label is refused or ignored. The
slice locators are the opposite, since almost every case is about the search
itself, and the ones to read first are the run of duplicates, the descending
index and the non-monotonic fallback.
"""

from std.collections import Optional
from std.testing import TestSuite, assert_equal, assert_false, assert_raises
from std.testing import assert_true

from firepanda.array.any import AnyArray
from firepanda.array.array import Array, from_list
from firepanda.array.strings import StringBuilder
from firepanda.frame.index import Index


def unnamed() -> Optional[String]:
    """A level name of `None`."""
    return Optional[String]()


def named(name: String) -> Optional[String]:
    """A level name."""
    return Optional[String](name)


def ints(values: List[Int64]) raises -> AnyArray:
    """An int64 column, all present."""
    return AnyArray(from_list[DType.int64](values))


def one(value: Int64) raises -> AnyArray:
    """A single int64 label."""
    return ints([value])


def nothing() raises -> AnyArray:
    """A single int64 label that is missing."""
    var out = from_list[DType.int64]([Int64(0)])
    out.set_null(0)
    return AnyArray(out^)


def labels(values: List[Int64]) raises -> Index:
    """An index over the given int64 labels, unnamed."""
    return Index(ints(values), unnamed())


def words(values: List[String]) raises -> AnyArray:
    """A string column, all present."""
    var builder = StringBuilder(capacity=len(values))
    for i in range(len(values)):
        builder.append(values[i].as_bytes())
    return AnyArray(builder^.finish())


def text(values: List[String]) raises -> Index:
    """A string index, all present."""
    return Index(words(values), unnamed())


def mask(values: List[Bool]) raises -> Array[DType.bool]:
    """A boolean column, all present."""
    var out = Array[DType.bool](overwritten=len(values))
    for i in range(len(values)):
        out[i] = values[i]
    return out^


def shown(index: Index) raises -> String:
    """The labels as one comma separated line, with `null` for a missing one.

    One string covers the order, the values, the length and the nulls, which are
    the four things most cases here are about, and a failure prints the whole
    answer rather than the first row that differs.
    """
    var col = index.materialize()
    var out = String()
    for i in range(len(col)):
        if i > 0:
            out += ","
        if not col.is_valid(i):
            out += "null"
        elif col.is_string():
            ref values = col.strings()
            out += String(StringSlice(unsafe_from_utf8=values.unsafe_bytes(i)))
        else:
            out += String(col.as_typed[DType.int64]()[i])
    return out^


def test_append_concatenates_and_keeps_every_duplicate() raises:
    """Not a union, which is the whole difference between the two."""
    assert_equal(
        shown(labels([Int64(3), 1]).append(labels([Int64(1), 5]))),
        "3,1,1,5",
        "both sides in order, nothing removed",
    )


def test_append_keeps_a_name_both_sides_agree_on() raises:
    var left = Index(ints([Int64(1)]), named("k"))
    var right = Index(ints([Int64(2)]), named("k"))
    var out = left.append(right)
    assert_true(Bool(out.name), "the name survives")
    assert_equal(out.name.value(), "k", "and it is the shared one")


def test_append_drops_a_name_the_sides_disagree_on() raises:
    var left = Index(ints([Int64(1)]), named("k"))
    var right = Index(ints([Int64(2)]), named("j"))
    assert_false(
        Bool(left.append(right).name),
        "two differently named levels make an unnamed one",
    )


def test_appending_two_adjacent_ranges_stays_a_range() raises:
    """The case a concatenation of frames read in pieces actually hits."""
    var out = Index(0, 4, unnamed()).append(Index(4, 3, unnamed()))
    assert_true(out.is_range(), "no memory is touched")
    assert_equal(shown(out), "0,1,2,3,4,5,6", "and the labels are right")


def test_appending_two_ranges_that_do_not_meet_materializes() raises:
    var out = Index(0, 3, unnamed()).append(Index(10, 2, unnamed()))
    assert_false(out.is_range(), "there is no range with a hole in it")
    assert_equal(shown(out), "0,1,2,10,11", "so the labels are written down")


def test_appending_an_empty_index_changes_nothing() raises:
    var empty = Index(ints(List[Int64]()), unnamed())
    assert_equal(
        shown(labels([Int64(3), 1]).append(empty)), "3,1", "nothing was added"
    )
    var also = Index(ints(List[Int64]()), unnamed())
    assert_equal(
        shown(also.append(labels([Int64(3), 1]))), "3,1", "nor on the way in"
    )


def test_append_takes_several_indexes_at_once() raises:
    var many: List[Index] = [labels([Int64(5)]), labels([Int64(9), 9])]
    assert_equal(
        shown(labels([Int64(3), 1]).append(many)),
        "3,1,5,9,9",
        "in the order given",
    )


def test_append_of_several_keeps_only_a_name_all_of_them_share() raises:
    var agree: List[Index] = [
        Index(ints([Int64(5)]), named("k")),
        Index(ints([Int64(9)]), named("k")),
    ]
    var out = Index(ints([Int64(1)]), named("k")).append(agree)
    assert_equal(out.name.value(), "k", "three of the same name")

    var disagree: List[Index] = [
        Index(ints([Int64(5)]), named("k")),
        Index(ints([Int64(9)]), named("j")),
    ]
    assert_false(
        Bool(Index(ints([Int64(1)]), named("k")).append(disagree).name),
        "one dissenter is enough",
    )


def test_delete_removes_one_position() raises:
    assert_equal(
        shown(labels([Int64(3), 1, 2]).delete(0)), "1,2", "the first row goes"
    )


def test_delete_counts_a_negative_position_from_the_end() raises:
    assert_equal(
        shown(labels([Int64(3), 1, 2]).delete(-1)), "3,1", "the last row goes"
    )


def test_delete_takes_a_list_of_positions() raises:
    var both: List[Int] = [0, 2]
    assert_equal(
        shown(labels([Int64(3), 1, 2]).delete(both)),
        "1",
        "and the survivors keep their order",
    )


def test_delete_refuses_a_position_off_the_end() raises:
    with assert_raises(contains="out of bounds"):
        _ = labels([Int64(3), 1, 2]).delete(5)
    with assert_raises(contains="out of bounds"):
        _ = labels([Int64(3), 1, 2]).delete(-4)


def test_delete_on_a_range_gives_the_labels_that_are_left() raises:
    var out = Index(5, 4, unnamed()).delete(1)
    assert_equal(shown(out), "5,7,8", "six was the one at position one")


def test_delete_keeps_the_name() raises:
    var out = Index(ints([Int64(3), 1]), named("k")).delete(0)
    assert_equal(out.name.value(), "k", "removing a row is not renaming")


def test_insert_puts_a_label_in_the_middle() raises:
    assert_equal(
        shown(labels([Int64(3), 1, 2]).insert(1, one(9))),
        "3,9,1,2",
        "before the row that was there",
    )


def test_insert_at_the_length_appends() raises:
    assert_equal(
        shown(labels([Int64(3), 1, 2]).insert(3, one(9))),
        "3,1,2,9",
        "which is the one position past the end that is allowed",
    )


def test_insert_counts_a_negative_position_from_the_end() raises:
    assert_equal(
        shown(labels([Int64(3), 1, 2]).insert(-1, one(9))),
        "3,1,9,2",
        "minus one is before the last row and not after it",
    )


def test_insert_refuses_a_position_past_the_length() raises:
    with assert_raises(contains="out of bounds"):
        _ = labels([Int64(3), 1, 2]).insert(9, one(1))


def test_insert_refuses_more_than_one_label() raises:
    with assert_raises(contains="one label"):
        _ = labels([Int64(3), 1]).insert(0, ints([Int64(1), 2]))


def test_inserting_a_null_keeps_the_dtype() raises:
    """Where pandas promotes and we do not, which is better and is a divergence.

    A numpy int64 array has nowhere to put a missing value, so pandas turns the
    whole index into float64 and writes a NaN. An Arrow column has a validity
    bitmap, so the label is simply absent and the other labels are still the
    integers they were.
    """
    assert_equal(
        shown(labels([Int64(3), 1]).insert(1, nothing())),
        "3,null,1",
        "one missing label among two present ones",
    )


def test_insert_works_on_a_string_index() raises:
    assert_equal(
        shown(text(["north", "south"]).insert(1, words(["east"]))),
        "north,east,south",
        "the gather is what knows about strings, not this",
    )


def test_drop_removes_every_row_carrying_a_label() raises:
    assert_equal(
        shown(labels([Int64(1), 1, 2]).drop(one(1))),
        "2",
        "both ones, not the first one",
    )


def test_drop_refuses_a_label_the_index_does_not_have() raises:
    with assert_raises(contains="not found in axis"):
        _ = labels([Int64(3), 1, 2]).drop(one(7))


def test_drop_names_the_missing_label_in_the_message() raises:
    """An error that says which label is worth more than one that says a label.
    """
    with assert_raises(contains="[7]"):
        _ = labels([Int64(3), 1, 2]).drop(one(7))


def test_drop_ignores_a_missing_label_when_asked() raises:
    assert_equal(
        shown(labels([Int64(3), 1, 2]).drop(one(7), errors="ignore")),
        "3,1,2",
        "and the ones that were there would still go",
    )


def test_drop_refuses_a_word_that_is_neither() raises:
    with assert_raises(contains="'raise' or 'ignore'"):
        _ = labels([Int64(3), 1]).drop(one(1), errors="warn")


def test_drop_matches_a_missing_label_with_a_missing_label() raises:
    """The one place in the library where a null equals a null, inherited.

    It comes from the factorize putting every null in one group rather than from
    a rule written here, which is the kind of behaviour that changes without
    anybody deciding to, so it is asserted rather than assumed.
    """
    var holed = from_list[DType.int64]([Int64(1), 0, 2])
    holed.set_null(1)
    var index = Index(AnyArray(holed^), unnamed())
    assert_equal(shown(index.drop(nothing())), "1,2", "the hole is gone")


def test_putmask_replaces_with_one_label() raises:
    assert_equal(
        shown(
            labels([Int64(1), 2, 3]).putmask(mask([True, False, True]), one(0))
        ),
        "0,2,0",
        "the same label everywhere the mask is true",
    )


def test_putmask_replaces_row_for_row_from_a_column() raises:
    assert_equal(
        shown(
            labels([Int64(1), 2, 3]).putmask(
                mask([True, False, True]), ints([Int64(7), 8, 9])
            )
        ),
        "7,2,9",
        "row two of the replacement is never read",
    )


def test_putmask_with_nothing_set_returns_the_index_unchanged() raises:
    var out = labels([Int64(1), 2, 3]).putmask(
        mask([False, False, False]), one(0)
    )
    assert_equal(shown(out), "1,2,3", "no gather, no copy of the labels")


def test_putmask_refuses_a_mask_of_the_wrong_length() raises:
    with assert_raises(contains="same length"):
        _ = labels([Int64(1), 2, 3]).putmask(mask([True, False]), one(0))


def test_putmask_refuses_a_replacement_of_the_wrong_length() raises:
    with assert_raises(contains="one label or 3"):
        _ = labels([Int64(1), 2, 3]).putmask(
            mask([True, False, True]), ints([Int64(7), 8])
        )


def test_a_bound_on_an_ascending_index() raises:
    var index = labels([Int64(1), 3, 5, 7, 9])
    assert_equal(index.get_slice_bound(one(5), "left"), 2, "the row itself")
    assert_equal(index.get_slice_bound(one(5), "right"), 3, "one past it")


def test_a_bound_for_a_label_the_index_does_not_have() raises:
    """Both sides land in the same place, which is what makes the slice empty.
    """
    var index = labels([Int64(1), 3, 5, 7, 9])
    assert_equal(
        index.get_slice_bound(one(4), "left"), 2, "where four would go"
    )
    assert_equal(index.get_slice_bound(one(4), "right"), 2, "and the same")


def test_a_bound_outside_the_index_clamps() raises:
    var index = labels([Int64(1), 3, 5])
    assert_equal(index.get_slice_bound(one(0), "left"), 0, "before everything")
    assert_equal(index.get_slice_bound(one(100), "left"), 3, "after everything")


def test_a_bound_lands_on_the_edge_of_a_run_of_duplicates() raises:
    """The reason this is a search for an edge rather than for a member."""
    var index = labels([Int64(1), 1, 3, 3, 5])
    assert_equal(index.get_slice_bound(one(3), "left"), 2, "the first three")
    assert_equal(index.get_slice_bound(one(3), "right"), 4, "past the last")
    assert_equal(shown(index.slice(2, 4)), "3,3", "which covers both of them")


def test_bounds_on_a_descending_index() raises:
    """Both comparisons turn over and the caller names the larger label first.
    """
    var index = labels([Int64(9), 7, 5, 3, 1])
    var found = index.slice_locs(
        Optional[AnyArray](one(7)), Optional[AnyArray](one(3))
    )
    assert_equal(found[0], 1, "from the seven")
    assert_equal(found[1], 4, "to past the three")


def test_bounds_on_a_descending_index_with_duplicates() raises:
    var index = labels([Int64(5), 3, 3, 1])
    var found = index.slice_locs(
        Optional[AnyArray](one(3)), Optional[AnyArray](one(3))
    )
    assert_equal(found[0], 1, "the first three")
    assert_equal(found[1], 3, "past the last three")


def test_slice_locs_with_no_bounds_is_the_whole_index() raises:
    var index = labels([Int64(1), 3, 5, 7, 9])
    var found = index.slice_locs()
    assert_equal(found[0], 0, "from the start")
    assert_equal(found[1], 5, "to the end")


def test_slice_locs_with_one_bound() raises:
    var index = labels([Int64(1), 3, 5, 7, 9])
    var from_five = index.slice_locs(Optional[AnyArray](one(5)))
    assert_equal(from_five[0], 2, "from the five")
    assert_equal(from_five[1], 5, "to the end")
    var to_five = index.slice_locs(end=Optional[AnyArray](one(5)))
    assert_equal(to_five[0], 0, "from the start")
    assert_equal(to_five[1], 3, "to past the five, since the end is included")


def test_a_bound_on_a_non_monotonic_index_looks_the_label_up() raises:
    var index = labels([Int64(3), 1, 2])
    assert_equal(index.get_slice_bound(one(3), "left"), 0, "where it sits")
    assert_equal(
        index.get_slice_bound(one(2), "right"), 3, "one past where it sits"
    )


def test_a_bound_on_a_non_monotonic_index_refuses_a_missing_label() raises:
    with assert_raises(contains="either sort the index"):
        _ = labels([Int64(3), 1, 2]).get_slice_bound(one(9), "left")


def test_a_bound_on_a_non_monotonic_index_refuses_a_repeated_label() raises:
    with assert_raises(contains="repeated label"):
        _ = labels([Int64(3), 1, 3, 2]).get_slice_bound(one(3), "left")


def test_a_bound_refuses_a_side_that_is_neither() raises:
    with assert_raises(contains="'left' or 'right'"):
        _ = labels([Int64(1), 3]).get_slice_bound(one(1), "middle")


def test_a_bound_refuses_more_than_one_label() raises:
    with assert_raises(contains="one label"):
        _ = labels([Int64(1), 3]).get_slice_bound(ints([Int64(1), 3]), "left")


def test_a_bound_on_a_range_agrees_with_the_general_route() raises:
    """A fast path is a second implementation, so it is checked against the first.

    The range answers by subtracting and the materialized index answers by
    searching, and the two drifting apart is the failure this is here to catch.
    """
    var quick = Index(5, 6, unnamed())
    var slow = Index(quick.materialize(), unnamed())
    for label in range(3, 14):
        var needle = one(Int64(label))
        assert_equal(
            quick.get_slice_bound(needle, "left"),
            slow.get_slice_bound(needle, "left"),
            String("left bound of ", label),
        )
        assert_equal(
            quick.get_slice_bound(needle, "right"),
            slow.get_slice_bound(needle, "right"),
            String("right bound of ", label),
        )


def test_a_missing_label_sorts_past_the_end() raises:
    """Where `argsort_any` puts a null, so where a bound has to put one too."""
    var index = labels([Int64(1), 3, 5])
    assert_equal(index.get_slice_bound(nothing(), "left"), 3, "past everything")


def test_bounds_on_a_string_index() raises:
    var index = text(["ant", "bee", "bee", "cow"])
    assert_equal(
        index.get_slice_bound(words(["bee"]), "left"), 1, "the first bee"
    )
    assert_equal(
        index.get_slice_bound(words(["bee"]), "right"), 3, "past the last"
    )


def test_bounds_on_an_empty_index() raises:
    var index = Index(ints(List[Int64]()), unnamed())
    var found = index.slice_locs(
        Optional[AnyArray](one(1)), Optional[AnyArray](one(2))
    )
    assert_equal(found[0], 0, "nothing before")
    assert_equal(found[1], 0, "and nothing after")


def test_slice_indexer_carries_the_step_through() raises:
    var index = labels([Int64(1), 3, 5, 7, 9])
    var found = index.slice_indexer(
        Optional[AnyArray](one(3)), Optional[AnyArray](one(9)), 2
    )
    assert_equal(found[0], 1, "the same first row")
    assert_equal(found[1], 5, "the same last")
    assert_equal(found[2], 2, "and the step untouched")


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
