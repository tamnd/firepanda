"""Tests for combining two indexes.

Every expected answer here was read off a running pandas 3.0.5 rather than off
its documentation, because the documentation is wrong about at least one of them.
`intersection` is still described as keeping the smaller of two duplicate counts
and it has not done that for several versions. Where a case looks arbitrary it is
because pandas is arbitrary there, and the point of the test is to hold us to the
same arbitrary thing.

The three families worth reading as families are the order defaults, which go one
way for `union` and the other way for `intersection` and `difference`; the
duplicate rules, which differ across all four; and the short circuits, which are
observable because they skip the sort and which an implementation that always
sorted would get wrong on the two most common calls there are.
"""

from std.collections import Optional
from std.testing import TestSuite, assert_equal, assert_raises, assert_true

from firepanda.array.any import AnyArray
from firepanda.array.array import from_list
from firepanda.array.strings import StringBuilder
from firepanda.frame.index import NOT_FOUND, Index


def unnamed() -> Optional[String]:
    """A level name of `None`."""
    return Optional[String]()


def named(name: String) -> Optional[String]:
    """A level name."""
    return Optional[String](name)


def ints(values: List[Int64]) raises -> AnyArray:
    """An int64 column, all present."""
    return AnyArray(from_list[DType.int64](values))


def labels(values: List[Int64]) raises -> Index:
    """An index over the given int64 labels, unnamed."""
    return Index(ints(values), unnamed())


def with_holes(values: List[Int64], holes: List[Int]) raises -> Index:
    """An index with the named positions missing.

    pandas can only hold a missing label in a float index, because a numpy int64
    array has nowhere to record absence. An Arrow column has a validity bitmap, so
    the same cases are written here on integers and mean the same thing.
    """
    var out = from_list[DType.int64](values)
    for i in range(len(holes)):
        out.set_null(holes[i])
    return Index(AnyArray(out^), unnamed())


def text(values: List[String]) raises -> Index:
    """A string index, all present."""
    var builder = StringBuilder(capacity=len(values))
    for i in range(len(values)):
        builder.append(values[i].as_bytes())
    return Index(AnyArray(builder^.finish()), unnamed())


def shown(index: Index) raises -> String:
    """The labels as one comma separated line, with `null` for a missing one.

    One string covers the order, the values, the length and the nulls, which are
    the four things every case here is about, and a failure prints the whole
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


def test_a_union_sorts_by_default() raises:
    assert_equal(
        shown(labels([Int64(3), 1, 2]).union(labels([Int64(2), 4, 1]))),
        "1,2,3,4",
        "every label once, in order",
    )


def test_an_unsorted_union_is_this_index_then_the_new_labels() raises:
    assert_equal(
        shown(
            labels([Int64(3), 1, 2]).union(labels([Int64(2), 4, 1]), sort=False)
        ),
        "3,1,2,4",
        "our order, then theirs for the ones only they have",
    )


def test_a_union_keeps_the_larger_of_the_two_counts() raises:
    """The rule that makes a union not a set union.

    One and two appear twice because one side has them twice, three and four once
    because only one side has them at all. Six labels out of eight in, where a set
    would give four and a concatenation would give eight.
    """
    assert_equal(
        shown(labels([Int64(1), 1, 2, 3]).union(labels([Int64(1), 2, 2, 4]))),
        "1,1,2,2,3,4",
        "max of the two counts, each",
    )


def test_a_union_with_itself_does_not_sort() raises:
    """A short circuit that is observable, and it is the most common call there is.

    Two frames with the same labels are added together constantly, and pandas
    returns the labels unchanged rather than sorted because it takes an early
    return before the sort. An implementation that always sorted would reorder
    every such frame.
    """
    assert_equal(
        shown(labels([Int64(3), 1, 2]).union(labels([Int64(3), 1, 2]))),
        "3,1,2",
        "unchanged and unsorted",
    )


def test_a_union_with_an_empty_index_does_not_sort_either() raises:
    var empty = Index(ints(List[Int64]()), unnamed())
    assert_equal(
        shown(labels([Int64(3), 1, 2]).union(empty)), "3,1,2", "left unchanged"
    )
    assert_equal(
        shown(empty.union(labels([Int64(3), 1, 2]))), "3,1,2", "the other way"
    )


def test_a_union_of_two_default_ranges_never_materializes() raises:
    """Two frames nobody set an index on, which is the case that has to be free.

    `equals` compares two ranges by their start and length, so the short circuit
    fires before anything is built, and the answer is the range itself.
    """
    var out = Index(1000).union(Index(1000))
    assert_true(out.is_range(), "still a range")
    assert_equal(len(out), 1000, "and the same length")


def test_an_intersection_keeps_this_index_s_order() raises:
    """The opposite default from `union`, which is pandas' asymmetry and not ours.

    An intersection is a filter of the left side, so there is an order to
    inherit. A union is not a filter of anything, so there is not.
    """
    assert_equal(
        shown(labels([Int64(3), 1, 2]).intersection(labels([Int64(2), 3]))),
        "3,2",
        "our order",
    )
    assert_equal(
        shown(
            labels([Int64(3), 1, 2]).intersection(
                labels([Int64(2), 3]), sort=True
            )
        ),
        "2,3",
        "sorted when asked",
    )


def test_an_intersection_is_unique_even_when_both_sides_are_not() raises:
    """What pandas does, and not what its docstring says it does."""
    assert_equal(
        shown(
            labels([Int64(1), 1, 2, 3]).intersection(
                labels([Int64(1), 2, 2, 4])
            )
        ),
        "1,2",
        "once each, not the smaller count",
    )


def test_an_intersection_with_itself_is_itself() raises:
    assert_equal(
        shown(labels([Int64(3), 1, 2]).intersection(labels([Int64(3), 1, 2]))),
        "3,1,2",
        "unchanged",
    )
    assert_equal(
        shown(labels([Int64(1), 1, 2]).intersection(labels([Int64(1), 1, 2]))),
        "1,2",
        "unchanged except that the repeat goes",
    )


def test_an_intersection_with_nothing_in_common_is_empty() raises:
    var out = labels([Int64(1), 2]).intersection(labels([Int64(3), 4]))
    assert_equal(len(out), 0, "no rows")
    assert_equal(shown(out), "", "and nothing to show")


def test_a_difference_sorts_by_default_and_is_unique() raises:
    assert_equal(
        shown(labels([Int64(5), 1, 3, 1]).difference(labels([Int64(3)]))),
        "1,5",
        "sorted, and the repeated one appears once",
    )
    assert_equal(
        shown(
            labels([Int64(5), 1, 3, 1]).difference(
                labels([Int64(3)]), sort=False
            )
        ),
        "5,1",
        "our order when asked",
    )


def test_a_difference_from_itself_is_empty() raises:
    assert_equal(
        len(labels([Int64(3), 1, 2]).difference(labels([Int64(1), 2, 3]))),
        0,
        "nothing left",
    )


def test_a_symmetric_difference_is_what_only_one_side_has() raises:
    assert_equal(
        shown(
            labels([Int64(3), 1, 2]).symmetric_difference(
                labels([Int64(2), 4, 1])
            )
        ),
        "3,4",
        "sorted by default",
    )


def test_an_unsorted_symmetric_difference_is_ours_then_theirs() raises:
    assert_equal(
        shown(
            labels([Int64(9), 1, 2]).symmetric_difference(
                labels([Int64(2), 4, 1]), sort=False
            )
        ),
        "9,4",
        "our leftovers first",
    )


def test_the_name_survives_when_both_sides_agree() raises:
    var one = Index(ints([Int64(1), 2]), named("k"))
    var same = Index(ints([Int64(2), 3]), named("k"))
    var other = Index(ints([Int64(2), 3]), named("j"))
    assert_true(one.union(same).name, "a name came through")
    assert_equal(one.union(same).name.value(), "k", "and it is the shared one")
    assert_true(not one.union(other).name, "two names is no name")
    assert_true(not one.intersection(other).name, "the same on all four")
    assert_true(not one.difference(other).name, "and here")
    assert_true(not one.symmetric_difference(other).name, "and here")


def test_an_unnamed_index_stays_unnamed() raises:
    assert_true(
        not labels([Int64(1)]).union(labels([Int64(2)])).name, "still None"
    )


def test_a_symmetric_difference_takes_a_name_to_use() raises:
    """The one place in the four where the caller overrides the rule.

    It is here rather than on the other three because a symmetric difference is
    the operation where neither side has a better claim to the name.
    """
    var one = Index(ints([Int64(1), 2]), named("k"))
    var other = Index(ints([Int64(2), 3]), named("j"))
    var out = one.symmetric_difference(other, named("z"))
    assert_true(out.name, "a name")
    assert_equal(out.name.value(), "z", "the one asked for")
    assert_equal(shown(out), "1,3", "and the right labels")


def test_a_missing_label_is_an_ordinary_label() raises:
    """It matches a missing label and it sorts last, both inherited and both asserted.

    Matching comes from the factorize putting every null in one group and the
    sort order comes from `argsort_any` defaulting to nulls last. Neither is
    written here, which is exactly why both are checked here.
    """
    var mine = with_holes([Int64(1), 0, 3], [1])
    var theirs = with_holes([Int64(0), 3, 5], [0])
    assert_equal(shown(mine.union(theirs)), "1,3,5,null", "the null sorts last")
    assert_equal(shown(mine.intersection(theirs)), "null,3", "in our order")
    assert_equal(shown(mine.difference(theirs)), "1", "the null is in both")
    assert_equal(
        shown(mine.symmetric_difference(theirs)), "1,5", "neither null"
    )


def test_two_missing_labels_on_one_side_are_one_label() raises:
    var mine = with_holes([Int64(1), 0, 0], [1, 2])
    var theirs = with_holes([Int64(0), 9], [0])
    assert_equal(
        shown(mine.union(theirs)), "1,9,null,null", "max of two and one"
    )


def test_text_labels_take_the_same_four_routes() raises:
    var mine = text(["north", "south"])
    var theirs = text(["south", "east"])
    assert_equal(shown(mine.union(theirs)), "east,north,south", "sorted")
    assert_equal(
        shown(mine.union(theirs, sort=False)), "north,south,east", "ours"
    )
    assert_equal(shown(mine.intersection(theirs)), "south", "the shared one")
    assert_equal(shown(mine.difference(theirs)), "north", "only ours")
    assert_equal(
        shown(mine.symmetric_difference(theirs)), "east,north", "one side each"
    )


def test_two_dtypes_are_refused_rather_than_guessed_at() raises:
    """The pandas answer is a promotion, and that divergence is one issue.

    Fixing it in `union` alone would leave `equals` and `get_indexer` disagreeing
    with it, so it is refused in the same words everywhere until the promotion is
    written once.
    """
    with assert_raises(contains="not written yet"):
        _ = labels([Int64(1)]).union(text(["north"]))


def test_unique_keeps_the_first_of_each_label() raises:
    assert_equal(
        shown(labels([Int64(3), 1, 3, 2, 1]).unique()),
        "3,1,2",
        "first appearance, not sorted",
    )
    assert_equal(shown(with_holes([Int64(1), 0, 0], [1, 2]).unique()), "1,null")


def test_unique_on_a_range_is_the_range() raises:
    var out = Index(500).unique()
    assert_true(out.is_range(), "nothing to do and nothing done")
    assert_equal(len(out), 500, "same length")


def test_get_indexer_for_answers_whichever_kind_of_index_it_has() raises:
    """The method the rest of pandas calls when it does not know which it holds.
    """
    var unique_index = labels([Int64(10), 20])
    var found = unique_index.get_indexer_for(ints([Int64(20), 99]))
    assert_equal(len(found), 2, "one row per label asked for")
    assert_equal(Int(found[0]), 1, "found")
    assert_equal(Int(found[1]), Int(NOT_FOUND), "and not found")

    var repeated = labels([Int64(10), 20, 10])
    var several = repeated.get_indexer_for(ints([Int64(10), 20]))
    assert_equal(len(several), 3, "three rows for two labels")
    assert_equal(Int(several[0]), 0, "the first ten")
    assert_equal(Int(several[1]), 2, "the second ten")
    assert_equal(Int(several[2]), 1, "then the twenty")


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
