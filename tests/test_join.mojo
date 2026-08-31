"""Tests for joins, from the row pairing up to `DataFrame.join`.

Two layers, and the interesting failures live in different places in each.

`join_indices` decides which rows go with which, and everything that can go wrong
there is about rows that are not one-to-one: a key that appears twice on both
sides has to produce four rows and not two, an unmatched row has to survive or
not depending on the kind, and a null key has to match nothing including another
null. Those are all checked by counting rows and reading the pair lists, because
a join that produces the right values in the wrong multiplicity looks correct
until somebody sums a column.

The frame layer decides what the output columns are called and where a shared
key column's values come from. The second of those is the one worth being
careful about: after a right or outer join an output row can have no left row at
all, and taking the key from the left would put a null in the column the row was
matched on. There is a test for exactly that, on both kinds.

The order of the result is fixed rather than incidental and is pinned here. Left
row order, and within a left row, right row order, with a right join being the
same thing with the sides exchanged. A join whose row order moves between runs
cannot be compared against another engine, which is what the differential harness
at M1 exists to do.
"""

from std.testing import (
    TestSuite,
    assert_equal,
    assert_false,
    assert_raises,
    assert_true,
)

from firepanda.array.any import AnyArray
from firepanda.array.array import Array, from_list
from firepanda.frame.frame import DataFrame
from firepanda.frame.series import Series
from firepanda.join.pairs import JoinKind, join_indices
from firepanda.join.scalar import join_nested


def ints(values: List[Scalar[DType.int64]]) -> Array[DType.int64]:
    """Builds an int64 column."""
    return from_list(values)


def floats(values: List[Scalar[DType.float64]]) -> Array[DType.float64]:
    """Builds a float64 column."""
    return from_list(values)


def nulled(var values: Array[DType.int64], at: Int) -> Array[DType.int64]:
    """Marks one position null."""
    values.set_null(at)
    return values^


def pair_frame(var first: Series, var second: Series) raises -> DataFrame:
    """Builds a two column frame."""
    var columns = List[Series]()
    columns.append(first^)
    columns.append(second^)
    return DataFrame.from_series(columns^)


def left_frame() raises -> DataFrame:
    """The left side of most of these tests.

    Key 1 matches nothing, key 2 appears twice, key 3 matches twice on the right,
    and the last row's key is null.
    """
    return pair_frame(
        Series("k", nulled(ints([1, 2, 2, 3, 0]), 4)),
        Series("a", ints([10, 20, 30, 40, 50])),
    )


def right_frame() raises -> DataFrame:
    """The right side of most of these tests.

    Key 3 appears twice, key 4 matches nothing, and the last row's key is null.
    """
    return pair_frame(
        Series("k", nulled(ints([2, 3, 3, 4, 0]), 4)),
        Series("b", ints([200, 300, 310, 400, 500])),
    )


def keys(at: Int) -> List[Int]:
    """Builds a single element key position list."""
    var out = List[Int]()
    out.append(at)
    return out^


def on(name: String) -> List[String]:
    """Builds a single element key name list."""
    var out = List[String]()
    out.append(name)
    return out^


def test_an_inner_join_keeps_only_the_matches() raises:
    var out = left_frame().join(right_frame(), on("k"))
    assert_equal(len(out), 4)
    assert_equal(out.width(), 3)


def test_an_inner_join_pairs_every_combination_of_a_repeated_key() raises:
    # Key 2 is on two left rows and one right row, key 3 is on one left row and
    # two right rows. Four pairs, not two.
    var out = left_frame().join(right_frame(), on("k"))
    var a = out.column("a").as_typed[DType.int64]()
    var b = out.column("b").as_typed[DType.int64]()
    assert_equal(a[0], 20)
    assert_equal(b[0], 200)
    assert_equal(a[1], 30)
    assert_equal(b[1], 200)
    assert_equal(a[2], 40)
    assert_equal(b[2], 300)
    assert_equal(a[3], 40)
    assert_equal(b[3], 310)


def test_a_left_join_keeps_every_left_row() raises:
    var out = left_frame().join(right_frame(), on("k"), JoinKind.LEFT)
    assert_equal(len(out), 6)
    var a = out.column("a").as_typed[DType.int64]()
    assert_equal(a[0], 10)
    assert_equal(a[5], 50)


def test_a_left_join_nulls_the_right_columns_of_an_unmatched_row() raises:
    var out = left_frame().join(right_frame(), on("k"), JoinKind.LEFT)
    var b = out.column("b").as_typed[DType.int64]()
    assert_false(b.is_valid(0))
    assert_true(b.is_valid(1))
    assert_false(b.is_valid(5))


def test_a_null_key_matches_nothing_including_another_null() raises:
    # Both frames have a null key in their last row. SQL and Polars say those do
    # not join; pandas says they do. This follows SQL.
    var out = left_frame().join(right_frame(), on("k"))
    assert_equal(len(out), 4)
    var a = out.column("a").as_typed[DType.int64]()
    for i in range(len(out)):
        assert_true(a[i] != 50)


def test_a_left_join_still_emits_the_row_whose_key_is_null() raises:
    var out = left_frame().join(right_frame(), on("k"), JoinKind.LEFT)
    var k = out.column("k").as_typed[DType.int64]()
    assert_false(k.is_valid(5))
    var a = out.column("a").as_typed[DType.int64]()
    assert_equal(a[5], 50)


def test_a_right_join_comes_out_in_right_row_order() raises:
    var out = left_frame().join(right_frame(), on("k"), JoinKind.RIGHT)
    assert_equal(len(out), 6)
    var b = out.column("b").as_typed[DType.int64]()
    assert_equal(b[0], 200)
    assert_equal(b[1], 200)
    assert_equal(b[2], 300)
    assert_equal(b[3], 310)
    assert_equal(b[4], 400)
    assert_equal(b[5], 500)


def test_a_right_join_fills_the_shared_key_from_the_right() raises:
    # The last two output rows have no left row at all. The key column is one
    # column in the output, and taking it from the left would make it null on
    # exactly the rows that were matched on it.
    var out = left_frame().join(right_frame(), on("k"), JoinKind.RIGHT)
    var k = out.column("k").as_typed[DType.int64]()
    assert_true(k.is_valid(4))
    assert_equal(k[4], 4)
    var a = out.column("a").as_typed[DType.int64]()
    assert_false(a.is_valid(4))


def test_an_outer_join_keeps_both_sides() raises:
    var out = left_frame().join(right_frame(), on("k"), JoinKind.OUTER)
    assert_equal(len(out), 8)


def test_an_outer_join_puts_the_unmatched_right_rows_last() raises:
    var out = left_frame().join(right_frame(), on("k"), JoinKind.OUTER)
    var a = out.column("a").as_typed[DType.int64]()
    var b = out.column("b").as_typed[DType.int64]()
    assert_true(a.is_valid(5))
    assert_false(a.is_valid(6))
    assert_false(a.is_valid(7))
    assert_equal(b[6], 400)
    assert_equal(b[7], 500)


def test_an_outer_join_fills_the_shared_key_from_whichever_side_had_it() raises:
    var out = left_frame().join(right_frame(), on("k"), JoinKind.OUTER)
    var k = out.column("k").as_typed[DType.int64]()
    assert_equal(k[0], 1)
    assert_equal(k[6], 4)
    assert_false(k.is_valid(5))
    assert_false(k.is_valid(7))


def test_a_semi_join_returns_each_matching_left_row_once() raises:
    # The point of a semi join: key 3 matches twice on the right and still
    # produces one row.
    var out = left_frame().join(right_frame(), on("k"), JoinKind.SEMI)
    assert_equal(len(out), 3)
    var a = out.column("a").as_typed[DType.int64]()
    assert_equal(a[0], 20)
    assert_equal(a[1], 30)
    assert_equal(a[2], 40)


def test_a_semi_join_has_no_right_columns() raises:
    var out = left_frame().join(right_frame(), on("k"), JoinKind.SEMI)
    assert_equal(out.width(), 2)
    assert_false(out.has("b"))


def test_an_anti_join_returns_the_left_rows_that_matched_nothing() raises:
    var out = left_frame().join(right_frame(), on("k"), JoinKind.ANTI)
    assert_equal(len(out), 2)
    var a = out.column("a").as_typed[DType.int64]()
    assert_equal(a[0], 10)
    assert_equal(a[1], 50)


def test_an_anti_join_keeps_a_row_whose_key_is_null() raises:
    # A null key matches nothing, so it is an anti join hit rather than a row to
    # be quietly dropped.
    var out = left_frame().join(right_frame(), on("k"), JoinKind.ANTI)
    var k = out.column("k").as_typed[DType.int64]()
    assert_false(k.is_valid(1))


def test_a_cross_join_is_the_product() raises:
    var out = left_frame().cross_join(right_frame())
    assert_equal(len(out), 25)
    assert_equal(out.width(), 4)
    var a = out.column("a").as_typed[DType.int64]()
    var b = out.column("b").as_typed[DType.int64]()
    assert_equal(a[0], 10)
    assert_equal(b[0], 200)
    assert_equal(a[4], 10)
    assert_equal(b[4], 500)
    assert_equal(a[5], 20)
    assert_equal(b[5], 200)


def test_a_cross_join_refuses_key_columns() raises:
    with assert_raises(contains="takes no key columns"):
        _ = join_indices(
            left_frame().columns,
            keys(0),
            5,
            right_frame().columns,
            keys(0),
            5,
            JoinKind.CROSS,
        )


def test_a_colliding_right_column_gets_the_suffix() raises:
    # Both frames call their second column `a`, and the key is shared so there is
    # only one `k`.
    var right = pair_frame(
        Series("k", ints([2, 3])), Series("a", ints([777, 888]))
    )
    var out = left_frame().join(right, on("k"))
    assert_true(out.has("a"))
    assert_true(out.has("a_right"))
    var mine = out.column("a").as_typed[DType.int64]()
    var theirs = out.column("a_right").as_typed[DType.int64]()
    assert_equal(mine[0], 20)
    assert_equal(theirs[0], 777)


def test_a_suffix_that_still_collides_is_refused() raises:
    var mine = List[Series]()
    mine.append(Series("k", ints([2, 3])))
    mine.append(Series("a", ints([1, 2])))
    mine.append(Series("a_right", ints([5, 6])))
    var left = DataFrame.from_series(mine^)
    var right = pair_frame(Series("k", ints([2, 3])), Series("a", ints([3, 4])))
    # Renaming `a` to `a_right` would produce two columns of that name, and
    # nothing downstream could address the second.
    with assert_raises(contains="collides"):
        _ = left.join(right, on("k"))


def test_a_suffix_can_be_chosen() raises:
    var right = pair_frame(Series("k", ints([2])), Series("a", ints([9])))
    var out = left_frame().join(right, on("k"), JoinKind.INNER, "_theirs")
    assert_true(out.has("a_theirs"))


def test_two_keys_must_both_match() raises:
    var mine = List[Series]()
    mine.append(Series("x", ints([1, 1, 2])))
    mine.append(Series("y", ints([7, 8, 7])))
    mine.append(Series("v", ints([10, 20, 30])))
    var left = DataFrame.from_series(mine^)

    var theirs = List[Series]()
    theirs.append(Series("x", ints([1, 2, 2])))
    theirs.append(Series("y", ints([8, 8, 7])))
    theirs.append(Series("w", ints([100, 200, 300])))
    var right = DataFrame.from_series(theirs^)

    var both = List[String]()
    both.append("x")
    both.append("y")
    var out = left.join(right, both)
    # (1,7) matches nothing, (1,8) matches the first right row, (2,7) matches the
    # third. A join that packed the two keys badly would find more.
    assert_equal(len(out), 2)
    var w = out.column("w").as_typed[DType.int64]()
    assert_equal(w[0], 100)
    assert_equal(w[1], 300)


def test_key_columns_of_different_dtypes_are_refused() raises:
    var right = pair_frame(
        Series("k", floats([2.0, 3.0])), Series("b", ints([1, 2]))
    )
    # Promoting here would mean a join finding fewer matches than either side
    # expected, with nothing on screen to say why.
    with assert_raises(contains="same dtype"):
        _ = left_frame().join(right, on("k"))


def test_keys_named_differently_keep_both_columns() raises:
    var right = pair_frame(
        Series("code", ints([2, 3])), Series("b", ints([200, 300]))
    )
    var out = left_frame().join_on(right, on("k"), on("code"))
    assert_true(out.has("k"))
    assert_true(out.has("code"))
    assert_equal(len(out), 3)


def test_a_missing_key_column_is_refused() raises:
    with assert_raises(contains="nope"):
        _ = left_frame().join(right_frame(), on("nope"))


def test_the_key_lists_must_be_the_same_length() raises:
    var two = List[String]()
    two.append("k")
    two.append("a")
    with assert_raises(contains="same number of keys"):
        _ = left_frame().join_on(right_frame(), two, on("k"))


def test_a_key_given_twice_is_refused() raises:
    var twice = List[String]()
    twice.append("k")
    twice.append("k")
    with assert_raises(contains="given twice"):
        _ = left_frame().join(right_frame(), twice)


def test_an_empty_right_frame_leaves_a_left_join_intact() raises:
    var right = pair_frame(
        Series("k", ints(List[Scalar[DType.int64]]())),
        Series("b", ints(List[Scalar[DType.int64]]())),
    )
    var out = left_frame().join(right, on("k"), JoinKind.LEFT)
    assert_equal(len(out), 5)
    var b = out.column("b").as_typed[DType.int64]()
    assert_equal(b.null_count(), 5)


def test_an_empty_right_frame_gives_an_empty_inner_join() raises:
    var right = pair_frame(
        Series("k", ints(List[Scalar[DType.int64]]())),
        Series("b", ints(List[Scalar[DType.int64]]())),
    )
    var out = left_frame().join(right, on("k"))
    assert_equal(len(out), 0)
    assert_equal(out.width(), 3)


def test_an_empty_left_frame_leaves_a_right_join_intact() raises:
    var left = pair_frame(
        Series("k", ints(List[Scalar[DType.int64]]())),
        Series("a", ints(List[Scalar[DType.int64]]())),
    )
    var out = left.join(right_frame(), on("k"), JoinKind.RIGHT)
    assert_equal(len(out), 5)
    var k = out.column("k").as_typed[DType.int64]()
    assert_equal(k[0], 2)


def test_every_nan_key_is_the_same_key() raises:
    # `key_bits` normalizes NaN, so a NaN joins to a NaN. Comparing with `==`
    # would put each one in a group of its own and this would be zero rows.
    var nan = Float64(0.0) / Float64(0.0)
    var left = pair_frame(
        Series("k", floats([nan, 1.0])), Series("a", ints([10, 20]))
    )
    var right = pair_frame(
        Series("k", floats([nan, 2.0])), Series("b", ints([30, 40]))
    )
    var out = left.join(right, on("k"))
    assert_equal(len(out), 1)
    var a = out.column("a").as_typed[DType.int64]()
    assert_equal(a[0], 10)


def test_negative_zero_is_the_same_key_as_zero() raises:
    var left = pair_frame(Series("k", floats([-0.0])), Series("a", ints([10])))
    var right = pair_frame(Series("k", floats([0.0])), Series("b", ints([20])))
    var out = left.join(right, on("k"))
    assert_equal(len(out), 1)


def test_the_pairing_matches_the_nested_twin() raises:
    # The fuzz harness does this properly. This is here so that a change which
    # breaks the agreement fails in the unit suite as well, where the failure is
    # a great deal easier to read.
    var left = left_frame()
    var right = right_frame()
    var kinds = List[JoinKind]()
    kinds.append(JoinKind.INNER)
    kinds.append(JoinKind.LEFT)
    kinds.append(JoinKind.RIGHT)
    kinds.append(JoinKind.OUTER)
    kinds.append(JoinKind.SEMI)
    kinds.append(JoinKind.ANTI)

    for i in range(len(kinds)):
        var fast = join_indices(
            left.columns, keys(0), 5, right.columns, keys(0), 5, kinds[i]
        )
        var slow = join_nested(
            left.columns, keys(0), 5, right.columns, keys(0), 5, kinds[i]
        )
        assert_equal(len(fast), len(slow), String("row count for ", kinds[i]))
        for r in range(len(fast)):
            assert_equal(
                fast.left_at[r],
                slow.left_at[r],
                String("left row ", r, " for ", kinds[i]),
            )
            assert_equal(
                fast.right_at[r],
                slow.right_at[r],
                String("right row ", r, " for ", kinds[i]),
            )


def test_an_unmatched_left_row_uses_a_negative_index() raises:
    var pairs = join_indices(
        left_frame().columns,
        keys(0),
        5,
        right_frame().columns,
        keys(0),
        5,
        JoinKind.LEFT,
    )
    assert_equal(pairs.left_at[0], 0)
    assert_equal(pairs.right_at[0], -1)


def test_a_join_leaves_both_inputs_alone() raises:
    var left = left_frame()
    var right = right_frame()
    _ = left.join(right, on("k"), JoinKind.OUTER)
    assert_equal(len(left), 5)
    assert_equal(len(right), 5)
    var a = left.column("a").as_typed[DType.int64]()
    assert_equal(a[0], 10)


def wide_left(rows: Int, null_at: Int) raises -> DataFrame:
    """A left frame long enough that the emit splits across workers.

    The key cycles through five values so that most left rows match more than one
    right row, which is what makes a worker's slice emit more rows than it holds
    and puts the write offsets to work.
    """
    var k = Array[DType.int64](rows)
    var a = Array[DType.int64](rows)
    for i in range(rows):
        k[i] = Int64(i % 5)
        a[i] = Int64(i)
    if null_at >= 0:
        k.set_null(null_at)
    return pair_frame(Series("k", k^), Series("a", a^))


def wide_right() raises -> DataFrame:
    """Eight right rows against `wide_left`.

    Keys 0 and 1 appear twice, key 9 matches nothing, and key 4 is on a single
    row, so a left slice can contain rows that emit two, one and zero pairs.
    """
    return pair_frame(
        Series("k", ints([0, 1, 2, 3, 4, 0, 1, 9])),
        Series("b", ints([100, 110, 120, 130, 140, 150, 160, 170])),
    )


def nested_pairs(
    left: DataFrame, right: DataFrame, kind: JoinKind
) raises -> List[Int]:
    """The pairing a single obvious loop produces, flattened to left, right.

    `join_nested` would do this, but it is quadratic in a way that a frame this
    long cannot afford, and it does not need to be: the right side here is eight
    rows, so the obvious loop over both is a few million comparisons.
    """
    var lk = left.column("k").as_typed[DType.int64]()
    var rk = right.column("k").as_typed[DType.int64]()
    var out = List[Int](capacity=4 * len(lk))
    for i in range(len(lk)):
        var here = lk[i]
        var live = lk.is_valid(i)
        var hits = 0
        for r in range(len(rk)):
            if not live or not rk.is_valid(r) or here != rk[r]:
                continue
            hits += 1
            if kind == JoinKind.INNER or kind == JoinKind.LEFT:
                out.append(i)
                out.append(r)
            elif kind == JoinKind.SEMI and hits == 1:
                out.append(i)
                out.append(-1)
        if hits > 0:
            continue
        if kind == JoinKind.LEFT or kind == JoinKind.ANTI:
            out.append(i)
            out.append(-1)
    return out^


def test_a_frame_past_the_split_pairs_what_one_loop_would() raises:
    # The emit runs on one thread below `PARALLEL_LEFT_ROWS` and on every core
    # above it, and the second of those is only correct if each worker writes at
    # exactly the offset the counting pass left for it. An off by one in that
    # arithmetic shows up as a pair in the wrong place or as an uninitialised
    # row, neither of which the short frames above can reach.
    var left = wide_left(140_000, -1)
    var right = wide_right()
    var kinds: List[JoinKind] = [
        JoinKind.INNER,
        JoinKind.LEFT,
        JoinKind.SEMI,
        JoinKind.ANTI,
    ]
    for i in range(len(kinds)):
        var fast = join_indices(
            left.columns, keys(0), 140_000, right.columns, keys(0), 8, kinds[i]
        )
        var want = nested_pairs(left, right, kinds[i])
        assert_equal(
            len(fast) * 2, len(want), String("row count for ", kinds[i])
        )
        for r in range(len(fast)):
            assert_equal(
                fast.left_at[r],
                want[2 * r],
                String("left row ", r, " for ", kinds[i]),
            )
            assert_equal(
                fast.right_at[r],
                want[2 * r + 1],
                String("right row ", r, " for ", kinds[i]),
            )


def test_a_null_key_past_the_split_still_matches_nothing() raises:
    # The null check is skipped outright when no key column has one, so the path
    # that keeps it has to be walked at this length too. The null is put in the
    # back half so that it lands on a worker other than the first.
    var left = wide_left(140_000, 110_001)
    var right = wide_right()
    var inner = join_indices(
        left.columns,
        keys(0),
        140_000,
        right.columns,
        keys(0),
        8,
        JoinKind.INNER,
    )
    assert_equal(
        len(inner), len(nested_pairs(left, right, JoinKind.INNER)) // 2
    )
    for r in range(len(inner)):
        assert_true(inner.left_at[r] != 110_001, "the null key matched")

    var anti = join_indices(
        left.columns, keys(0), 140_000, right.columns, keys(0), 8, JoinKind.ANTI
    )
    var kept = False
    for r in range(len(anti)):
        if anti.left_at[r] == 110_001:
            kept = True
    assert_true(kept, "the null key was dropped by the anti join")


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
