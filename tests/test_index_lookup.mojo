"""Tests for looking a label up in an index.

Everything here is one of two shapes. Either it asks a lookup a question and
checks the answer against a position counted by hand, or it asks the same
question of a range and of the array that range materializes to and requires the
two to agree. The second shape is the one that catches real bugs, because the
range fast paths are a separate implementation of the same function and the whole
reason they exist is that they are cheaper, which is exactly the pressure that
makes two implementations drift.

The null cases are written out rather than assumed. A null label matching a null
label is behaviour inherited from the factorize underneath, and inherited
behaviour is the kind that changes without anybody deciding to change it.
"""

from std.collections import Optional
from std.testing import (
    TestSuite,
    assert_equal,
    assert_false,
    assert_raises,
    assert_true,
)

from firepanda.array.any import AnyArray
from firepanda.array.array import Array, from_list
from firepanda.array.strings import StringBuilder
from firepanda.frame.index import NOT_FOUND, Index


def unnamed() -> Optional[String]:
    """A level name of `None`, spelled once because it is written a lot."""
    return Optional[String]()


def named(name: String) -> Optional[String]:
    """A level name, spelled once for the same reason."""
    return Optional[String](name)


def ints(values: List[Int64]) raises -> AnyArray:
    """An int64 column of the given values, all present."""
    return AnyArray(from_list[DType.int64](values))


def labels(values: List[Int64]) raises -> Index:
    """An index over the given int64 labels, unnamed."""
    return Index(ints(values), unnamed())


def with_holes(values: List[Int64], holes: List[Int]) raises -> AnyArray:
    """An int64 column with the named positions marked missing."""
    var out = from_list[DType.int64](values)
    for i in range(len(holes)):
        out.set_null(holes[i])
    return AnyArray(out^)


def text(values: List[String]) raises -> AnyArray:
    """A string column of the given values, all present."""
    var builder = StringBuilder(capacity=len(values))
    for i in range(len(values)):
        builder.append(values[i].as_bytes())
    return AnyArray(builder^.finish())


def positions(found: Array[DType.int64]) -> List[Int]:
    """An indexer as a list, so that a test can compare it in one assertion."""
    var out = List[Int]()
    for i in range(len(found)):
        out.append(Int(found[i]))
    return out^


def assert_positions(
    found: Array[DType.int64], want: List[Int], hint: String
) raises:
    """The whole of an indexer against the whole of a hand counted answer."""
    assert_equal(positions(found), want, hint)


def test_a_range_is_unique_without_being_looked_at() raises:
    """The common case, and the reason the range representation is kept."""
    assert_true(Index(1000).is_unique(), "a range repeats nothing")
    assert_false(Index(1000).has_duplicates(), "the other spelling agrees")
    assert_true(Index(0).is_unique(), "an empty index repeats nothing either")


def test_labels_are_unique_when_they_are_all_different() raises:
    assert_true(labels([Int64(30), 10, 20]).is_unique(), "three distinct")
    assert_false(labels([Int64(30), 10, 30]).is_unique(), "thirty twice")
    assert_true(labels([Int64(30), 10, 30]).has_duplicates(), "and so")


def test_two_missing_labels_are_one_label() raises:
    """A null is a label here, and two of them are the same one.

    This is what makes an index with two holes in it non unique, which is what
    pandas says, and it is the first place the null rule from the factorize shows
    through.
    """
    var one = Index(with_holes([Int64(1), 2, 3], [1]), unnamed())
    assert_true(one.is_unique(), "one hole is one label")
    var two = Index(with_holes([Int64(1), 2, 3], [1, 2]), unnamed())
    assert_false(two.is_unique(), "two holes are the same label twice")


def test_a_range_is_increasing_and_is_not_decreasing() raises:
    """The asymmetry the obvious range fast path gets wrong.

    A range steps by one, so it never decreases, and the only ranges that are
    monotonic decreasing are the ones with nothing to compare.
    """
    assert_true(Index(10).is_monotonic_increasing(), "a range goes up")
    assert_false(Index(10).is_monotonic_decreasing(), "and not down")
    assert_true(Index(1).is_monotonic_decreasing(), "one row is both")
    assert_true(Index(0).is_monotonic_decreasing(), "so is none")


def test_labels_are_scanned_for_monotonicity() raises:
    assert_true(labels([Int64(1), 2, 3]).is_monotonic_increasing(), "up")
    assert_true(
        labels([Int64(1), 1, 3]).is_monotonic_increasing(), "flat is up"
    )
    assert_false(labels([Int64(3), 2, 1]).is_monotonic_increasing(), "down")
    assert_true(labels([Int64(3), 2, 1]).is_monotonic_decreasing(), "down")
    assert_false(labels([Int64(1), 3, 2]).is_monotonic_decreasing(), "neither")


def test_a_missing_label_makes_an_index_neither() raises:
    """What pandas says, and what `Series` already says about a column."""
    var holed = Index(with_holes([Int64(1), 2, 3], [1]), unnamed())
    assert_false(holed.is_monotonic_increasing(), "a hole is not an order")
    assert_false(holed.is_monotonic_decreasing(), "in either direction")


def test_a_range_finds_a_label_by_subtracting() raises:
    """The arithmetic route, checked against positions counted by hand."""
    var found = Index(5, 5, unnamed()).get_indexer(ints([Int64(7), 5, 9]))
    assert_positions(found, [2, 0, 4], "the three positions")


def test_a_range_says_nothing_is_there_when_it_is_outside() raises:
    var found = Index(5, 5, unnamed()).get_indexer(ints([Int64(4), 10, -1]))
    var absent = Int(NOT_FOUND)
    assert_positions(found, [absent, absent, absent], "all three outside")


def test_the_range_route_agrees_with_the_label_route() raises:
    """The property that matters, since the fast path is a second implementation.

    Materializing the range and building an index over the result forces the
    factorize route, so both sides of the `if` in `get_indexer` answer the same
    question about the same index.
    """
    var range_index = Index(5, 40, unnamed())
    var built = Index(range_index.materialize(), unnamed())
    var asked = ints([Int64(5), 44, 20, 4, 45, 21])
    assert_equal(
        positions(range_index.get_indexer(asked)),
        positions(built.get_indexer(asked)),
        "the two routes agree",
    )


def test_labels_are_found_wherever_they_sit() raises:
    var found = labels([Int64(30), 10, 20]).get_indexer(ints([Int64(20), 30]))
    assert_positions(found, [2, 0], "not in index order")


def test_a_label_that_is_not_there_is_not_found() raises:
    var found = labels([Int64(30), 10]).get_indexer(ints([Int64(10), 99]))
    assert_positions(found, [1, Int(NOT_FOUND)], "one of the two")


def test_a_null_finds_a_null() raises:
    """The one place in the library where a missing value equals a missing one.

    pandas matches NaN to NaN here and nowhere else, and the behaviour comes out
    of the factorize putting every null in one group rather than out of anything
    written for it, which is why it is asserted rather than trusted.
    """
    var index = Index(with_holes([Int64(1), 2, 3], [1]), unnamed())
    var asked = with_holes([Int64(0), 3], [0])
    assert_positions(index.get_indexer(asked), [1, 2], "the hole and the three")


def test_text_labels_are_found_too() raises:
    """A string index takes the same route and must not be read as bytes."""
    var index = Index(text(["north", "south", "east"]), unnamed())
    var found = index.get_indexer(text(["east", "west", "north"]))
    assert_positions(found, [2, Int(NOT_FOUND), 0], "by value")


def test_get_indexer_refuses_a_repeated_label() raises:
    """pandas refuses too, because there is no single position to report."""
    with assert_raises(contains="unique"):
        _ = labels([Int64(10), 10]).get_indexer(ints([Int64(10)]))


def test_get_indexer_refuses_a_label_of_another_type() raises:
    """Promoting two dtypes to a common one is not written, so it says so."""
    var index = Index(text(["north"]), unnamed())
    with assert_raises(contains="not written yet"):
        _ = index.get_indexer(ints([Int64(1)]))


def test_the_non_unique_form_gives_every_position() raises:
    var index = labels([Int64(10), 20, 10, 30, 10])
    var found = index.get_indexer_non_unique(ints([Int64(10), 30]))
    assert_positions(found.positions, [0, 2, 4, 3], "four rows for two labels")
    assert_equal(len(found.missing), 0, "both labels were there")


def test_the_non_unique_form_reports_what_it_did_not_find() raises:
    var index = labels([Int64(10), 20, 10])
    var found = index.get_indexer_non_unique(ints([Int64(99), 10, 98]))
    var absent = Int(NOT_FOUND)
    assert_positions(
        found.positions,
        [absent, 0, 2, absent],
        "a placeholder each for the two that are not there",
    )
    assert_positions(found.missing, [0, 2], "by position in the target")


def test_the_non_unique_form_works_on_a_unique_index() raises:
    """It is the general form, so it has to answer the easy case as well."""
    var found = labels([Int64(30), 10, 20]).get_indexer_non_unique(
        ints([Int64(20), 30])
    )
    assert_positions(found.positions, [2, 0], "one position each")


def test_get_loc_gives_every_position_a_label_sits_at() raises:
    var twice: List[Int] = [0, 2]
    assert_equal(
        labels([Int64(10), 20, 10]).get_loc(ints([Int64(10)])), twice, "twice"
    )
    var once: List[Int] = [2]
    assert_equal(Index(5, 5, unnamed()).get_loc(ints([Int64(7)])), once, "once")


def test_get_loc_raises_rather_than_answering_with_a_sentinel() raises:
    """The whole difference between this and `get_indexer`.

    A caller that wants a sentinel has `get_indexer`, and one that writes
    `df.loc["nope"]` wants an error rather than a row of nulls.
    """
    with assert_raises(contains="not in the index"):
        _ = labels([Int64(10), 20]).get_loc(ints([Int64(99)]))


def test_get_loc_takes_one_label() raises:
    with assert_raises(contains="one label"):
        _ = labels([Int64(10), 20]).get_loc(ints([Int64(10), 20]))


def test_a_range_equals_the_labels_it_stands_for() raises:
    """The representation is not part of what an index is."""
    var range_index = Index(5, 4, unnamed())
    var built = Index(range_index.materialize(), unnamed())
    assert_true(range_index.equals(built), "a range equals its labels")
    assert_true(built.equals(range_index), "and the other way round")


def test_two_ranges_compare_without_materializing() raises:
    assert_true(Index(10).equals(Index(10)), "same start and length")
    assert_false(Index(10).equals(Index(1, 10, unnamed())), "different start")
    assert_false(Index(10).equals(Index(9)), "different length")


def test_equality_ignores_the_name_and_identical_does_not() raises:
    """The two disagree, and which one a caller wants decides the answer.

    `align` asks the first, because two frames labelled the same need no
    alignment whatever their levels are called. A frame comparison asks the
    second, because a level that was renamed is a difference somebody made on
    purpose.
    """
    var plain = labels([Int64(10), 20])
    var called = Index(ints([Int64(10), 20]), named("key"))
    assert_true(plain.equals(called), "the labels are the same")
    assert_false(plain.identical(called), "the names are not")
    assert_true(
        called.identical(Index(ints([Int64(10), 20]), named("key"))),
        "same labels and same name",
    )
    assert_false(
        called.identical(Index(ints([Int64(10), 20]), named("other"))),
        "same labels and a different name",
    )


def test_a_missing_label_equals_a_missing_label() raises:
    """An index has to equal itself, holes and all."""
    var one = Index(with_holes([Int64(1), 2, 3], [1]), unnamed())
    var two = Index(with_holes([Int64(1), 2, 3], [1]), unnamed())
    assert_true(one.equals(two), "the same index twice")
    var elsewhere = Index(with_holes([Int64(1), 2, 3], [2]), unnamed())
    assert_false(one.equals(elsewhere), "the hole is in a different row")


def test_text_labels_compare_by_value() raises:
    var one = Index(text(["north", "south"]), unnamed())
    var same = Index(text(["north", "south"]), unnamed())
    var other = Index(text(["north", "east"]), unnamed())
    assert_true(one.equals(same), "the same names")
    assert_false(one.equals(other), "a different one in the second row")


def test_an_empty_index_looks_up_nothing_and_finds_nothing() raises:
    """The boundary that every counting pass here gets wrong if it is wrong."""
    var empty = Index(0)
    assert_equal(len(empty.get_indexer(ints(List[Int64]()))), 0, "nothing out")
    var found = empty.get_indexer(ints([Int64(1)]))
    assert_positions(found, [Int(NOT_FOUND)], "not there")
    var built = Index(ints(List[Int64]()), unnamed())
    assert_equal(len(built.get_indexer(ints([Int64(1), 2]))), 2, "one each")


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
