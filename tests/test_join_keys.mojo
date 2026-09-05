"""Tests for the join's key alignment.

`align_keys` has three routes to the same answer and the answer is not a value
anyone can read off. It is a set of ordinals, and what makes them correct is not
what they are but which rows agree: two rows hold the same ordinal exactly when
they hold the same key, and a row that matches nothing on the other side holds
one that no row of the other side holds.

So most of these tests are written as "these two rows agree" and "these two do
not" rather than as expected numbers. The one exception is the miss ordinal,
which is pinned to `groups - 1`, because the caller sizes its bucket tables to
`groups` and a miss landing anywhere else would index a bucket that has rows in
it.

Which route runs is not observable from the outside, which is the point of having
three, so the tests that care drive them by shape instead. A narrow integer key
takes the direct table, a key spread over the whole int64 range takes the hash
table, and two key columns or a string key take the concat and factorize
fallback. Each of those shapes appears here at least once, and the assertions are
the same ones in every case.

The last few tests are about the seam rather than the answer. `build_side` and
`probe_side` are `align_keys`' second route split in half so a streaming join can
keep the table between chunks, and what has to hold is that probing in pieces
writes what probing in one go writes. Each of those tests does it both ways over
the same rows and compares, once on each route.

The wider correctness argument for this file is `tests/fuzz/join.mojo`, which
generates both sides at random sizes with random null densities and compares
every pairing against the nested loop twin. What is here is the cases that fuzzer
is unlikely to reach on purpose and that would be hard to read if it did.
"""

from std.testing import (
    TestSuite,
    assert_equal,
    assert_not_equal,
    assert_raises,
    assert_true,
)

from firepanda.array.any import AnyArray, borrow_columns
from firepanda.array.array import Array, from_list
from firepanda.array.strings import StringArray, StringBuilder
from firepanda.join.keys import KeyAlignment, align_keys, build_side, probe_side


def ints(values: List[Scalar[DType.int64]]) -> Array[DType.int64]:
    """Builds an int64 column."""
    return from_list(values)


def floats(values: List[Scalar[DType.float64]]) -> Array[DType.float64]:
    """Builds a float64 column."""
    return from_list(values)


def text(values: List[String]) raises -> StringArray:
    """Builds a string column."""
    var out = StringBuilder(capacity=len(values))
    for i in range(len(values)):
        out.append(values[i].as_bytes())
    return out^.finish()


def one(at: Int) -> List[Int]:
    """Builds a single element key position list."""
    var out = List[Int]()
    out.append(at)
    return out^


def two(first: Int, second: Int) -> List[Int]:
    """Builds a two element key position list."""
    var out = List[Int]()
    out.append(first)
    out.append(second)
    return out^


def columns(var first: AnyArray) -> List[AnyArray]:
    """Wraps one column as a frame's column list."""
    var out = List[AnyArray]()
    out.append(first^)
    return out^


def paired(var first: AnyArray, var second: AnyArray) -> List[AnyArray]:
    """Wraps two columns as a frame's column list."""
    var out = List[AnyArray]()
    out.append(first^)
    out.append(second^)
    return out^


def align_owned(
    var left_columns: List[AnyArray],
    left_keys: List[Int],
    left_rows: Int,
    var right_columns: List[AnyArray],
    right_keys: List[Int],
    right_rows: Int,
) raises -> KeyAlignment:
    """Calls `align_keys` with columns these tests own.

    `align_keys` borrows its columns, so something has to hold them for the
    length of the call. Taking them here is that something, and it keeps the
    tests reading the way they did.

    Args:
        left_columns: The left frame's columns. Consumed.
        left_keys: Which of them are keys.
        left_rows: The left height.
        right_columns: The right frame's columns. Consumed.
        right_keys: Which of them are keys.
        right_rows: The right height.

    Returns:
        What `align_keys` returns.

    Raises:
        Whatever `align_keys` raises.
    """
    return align_keys(
        borrow_columns(left_columns),
        left_keys,
        left_rows,
        borrow_columns(right_columns),
        right_keys,
        right_rows,
    )


def matched(
    got: KeyAlignment, left_rows: Int, right_rows: Int, row: Int
) raises -> Bool:
    """Whether a left row shares its ordinal with any right row.

    This is what the alignment is for, and it is the assertion to write rather
    than a comparison against a particular ordinal. Only the probe route reserves
    one for a miss; the concat route gets the same answer by handing a row a code
    that the other side simply does not have. Both are correct and only this
    question distinguishes them from a wrong one.

    Args:
        got: The alignment.
        left_rows: How many rows the left side has.
        right_rows: How many rows the right side has.
        row: Which left row to ask about.

    Returns:
        Whether anything on the right pairs with it.
    """
    for r in range(right_rows):
        if got.has_nulls and got.absent[left_rows + r]:
            continue
        if got.codes[left_rows + r] == got.codes[row]:
            return True
    return False


def align_ints(
    var left: Array[DType.int64], var right: Array[DType.int64]
) raises -> KeyAlignment:
    """Aligns two single column int64 frames."""
    var rows = len(left)
    var other = len(right)
    return align_owned(
        columns(AnyArray(left^)),
        one(0),
        rows,
        columns(AnyArray(right^)),
        one(0),
        other,
    )


def test_rows_with_the_same_key_get_the_same_ordinal() raises:
    var got = align_ints(ints([7, 3, 7]), ints([3, 9]))
    assert_equal(got.codes[0], got.codes[2], "both sevens")
    assert_not_equal(got.codes[0], got.codes[1], "seven against three")
    # Left row 1 and right row 0 are both key 3, and the right side starts at 3.
    assert_equal(got.codes[1], got.codes[3], "three across the sides")


def test_a_key_on_only_one_side_gets_the_miss_ordinal() raises:
    var got = align_ints(ints([1, 2]), ints([2, 3]))
    # Key 1 is on the left only, so it matched nothing.
    assert_equal(Int(got.codes[0]), got.groups - 1, "the unmatched left key")
    assert_equal(got.codes[1], got.codes[2], "the key both sides have")


def test_the_miss_ordinal_is_one_past_every_real_one() raises:
    var got = align_ints(ints([5, 6, 7]), ints([5]))
    # One real key on the build side and one ordinal for everything else.
    assert_equal(got.groups, 2, "group count")
    assert_equal(Int(got.codes[0]), 0, "the matched key")
    assert_equal(Int(got.codes[1]), 1, "the first miss")
    assert_equal(Int(got.codes[2]), 1, "the second miss")


def test_every_ordinal_indexes_a_table_of_the_group_count() raises:
    var got = align_ints(ints([1, 4, 9, 16, 25]), ints([4, 16, 36]))
    for i in range(len(got.codes)):
        assert_true(Int(got.codes[i]) < got.groups, String("row ", i))


def test_the_smaller_side_is_built_whichever_side_it_is() raises:
    # The same pairing written both ways round. Which side gets built changes,
    # the ordinals change with it, and which rows agree does not.
    var wide = align_ints(ints([1, 2, 3, 4, 5, 6]), ints([2, 5]))
    assert_equal(wide.codes[1], wide.codes[6], "key 2 with the right built")
    assert_equal(wide.codes[4], wide.codes[7], "key 5 with the right built")
    assert_equal(Int(wide.codes[0]), wide.groups - 1, "key 1 matched nothing")

    var narrow = align_ints(ints([2, 5]), ints([1, 2, 3, 4, 5, 6]))
    assert_equal(narrow.codes[0], narrow.codes[3], "key 2 with the left built")
    assert_equal(narrow.codes[1], narrow.codes[6], "key 5 with the left built")
    assert_equal(
        Int(narrow.codes[2]), narrow.groups - 1, "key 1 matched nothing"
    )


def test_a_key_below_the_build_range_matches_nothing() raises:
    # The direct table is indexed by value minus the build side's minimum, so a
    # probe key under that minimum would index behind the table. The left is the
    # longer side here so that the right is the one built.
    var got = align_ints(ints([-100, 50, 51, 52]), ints([50, 51]))
    assert_true(not matched(got, 4, 2, 0), "under the range")
    assert_equal(got.codes[1], got.codes[4], "inside the range")


def test_a_key_above_the_build_range_matches_nothing() raises:
    var got = align_ints(ints([50, 900, 51, 52]), ints([50, 51]))
    assert_true(not matched(got, 4, 2, 1), "over the range")
    assert_equal(got.codes[0], got.codes[4], "inside the range")


def test_keys_too_far_apart_for_a_table_still_align() raises:
    # A span this wide declines the direct table, so this is the hash route.
    var got = align_ints(ints([1, 1 << 50, 7]), ints([1 << 50, 2 << 50]))
    assert_equal(got.codes[1], got.codes[3], "the shared key")
    assert_equal(Int(got.codes[0]), got.groups - 1, "the left only key")
    assert_equal(Int(got.codes[2]), got.groups - 1, "the other left only key")


def test_a_null_key_on_the_probe_side_matches_nothing() raises:
    var left = ints([1, 2, 3])
    left.set_null(1)
    var got = align_ints(left^, ints([2]))
    assert_equal(Int(got.codes[1]), got.groups - 1, "the null row")
    assert_true(got.has_nulls, "has_nulls")
    assert_true(got.absent[1], "absent")


def test_a_null_key_on_the_build_side_is_flagged() raises:
    var right = ints([2, 3])
    right.set_null(0)
    var got = align_ints(ints([1, 2, 3]), right^)
    assert_true(got.has_nulls, "has_nulls")
    assert_true(got.absent[3], "the null right row")
    assert_true(not got.absent[4], "the row after it")
    assert_equal(got.codes[2], got.codes[4], "key 3 still pairs")


def test_a_column_with_no_nulls_leaves_the_flag_list_empty() raises:
    var got = align_ints(ints([1, 2]), ints([2, 3]))
    assert_true(not got.has_nulls, "has_nulls")
    assert_equal(len(got.absent), 0, "the flag list")


def test_two_key_columns_align_on_the_tuple() raises:
    var left_first = AnyArray(ints([1, 1, 2]))
    var left_second = AnyArray(ints([9, 8, 9]))
    var right_first = AnyArray(ints([1, 2]))
    var right_second = AnyArray(ints([8, 9]))
    var got = align_owned(
        paired(left_first^, left_second^),
        two(0, 1),
        3,
        paired(right_first^, right_second^),
        two(0, 1),
        2,
    )
    # (1, 8) is left row 1 and right row 0, which is position 3.
    assert_equal(got.codes[1], got.codes[3], "the shared tuple")
    # (2, 9) is left row 2 and right row 1.
    assert_equal(got.codes[2], got.codes[4], "the other shared tuple")
    # (1, 9) is on the left only, and it agrees with neither of the above.
    assert_not_equal(got.codes[0], got.codes[1], "first key alone is not it")
    assert_not_equal(got.codes[0], got.codes[2], "second key alone is not it")


def test_a_string_key_aligns() raises:
    var left = AnyArray(text(["red", "green", "red"]))
    var right = AnyArray(text(["green", "blue"]))
    var got = align_owned(columns(left^), one(0), 3, columns(right^), one(0), 2)
    assert_equal(got.codes[0], got.codes[2], "both reds")
    assert_equal(got.codes[1], got.codes[3], "green across the sides")
    assert_not_equal(got.codes[0], got.codes[1], "red against green")


def test_a_float_key_aligns() raises:
    var left = AnyArray(floats([1.5, 2.5]))
    var right = AnyArray(floats([2.5, 3.5]))
    var got = align_owned(columns(left^), one(0), 2, columns(right^), one(0), 2)
    assert_equal(got.codes[1], got.codes[2], "the shared key")
    assert_equal(Int(got.codes[0]), got.groups - 1, "the left only key")


def test_every_nan_is_the_same_key() raises:
    var nan = Float64("nan")
    var left = AnyArray(floats([nan, 1.0]))
    var right = AnyArray(floats([nan]))
    var got = align_owned(columns(left^), one(0), 2, columns(right^), one(0), 1)
    assert_equal(got.codes[0], got.codes[2], "nan against nan")


def test_negative_zero_is_the_same_key_as_zero() raises:
    var left = AnyArray(floats([-0.0]))
    var right = AnyArray(floats([0.0]))
    var got = align_owned(columns(left^), one(0), 1, columns(right^), one(0), 1)
    assert_equal(got.codes[0], got.codes[1], "minus zero against zero")


def test_an_empty_side_leaves_every_row_unmatched() raises:
    # No right rows at all, so the probe route declines and the concat route
    # runs. It reserves nothing, because every code it hands out is a real group
    # and none of them is on a side that does not exist.
    var got = align_ints(ints([1, 2, 3]), ints([]))
    for i in range(3):
        assert_true(not matched(got, 3, 0, i), String("row ", i))
        assert_true(Int(got.codes[i]) < got.groups, String("in range ", i))


def test_a_side_of_one_repeated_key_gives_one_group_and_a_miss() raises:
    var got = align_ints(ints([4, 4, 4, 5]), ints([4, 4]))
    assert_equal(got.groups, 2, "group count")
    assert_equal(got.codes[0], got.codes[4], "left four and right four")
    assert_equal(got.codes[1], got.codes[5], "the repeats")
    assert_equal(Int(got.codes[3]), got.groups - 1, "the five")


def test_a_probe_past_the_split_aligns_what_one_thread_would() raises:
    # Over the parallel threshold, so the probe runs on every core in morsels.
    # Every fourth key is on the right, and the answer is checked row by row
    # against what the key alone says it should be.
    comptime rows = 300_000
    var left = Array[DType.int64](rows)
    for i in range(rows):
        left[i] = Int64(i)
    var right = Array[DType.int64](rows // 4)
    for i in range(rows // 4):
        right[i] = Int64(i * 4)

    var got = align_ints(left^, right^)
    var bad = -1
    for i in range(rows):
        var want_miss = i % 4 != 0
        var is_miss = Int(got.codes[i]) == got.groups - 1
        if want_miss != is_miss:
            bad = i
            break
    assert_equal(bad, -1, String("row ", bad))

    var wrong = -1
    for r in range(rows // 4):
        if got.codes[rows + r] != got.codes[r * 4]:
            wrong = r
            break
    assert_equal(wrong, -1, String("right row ", wrong))


def probed(
    build: Array[DType.int64], probe: Array[DType.int64]
) raises -> Array[DType.uint32]:
    """Builds a table from one column and probes it with another, in one call.

    Args:
        build: The column the table is built from.
        probe: The column asked about it.

    Returns:
        The build side's ordinals followed by the probe side's, which is the
        layout `align_keys` hands back.

    Raises:
        Whatever the build or the probe raises.
    """
    var codes = Array[DType.uint32](overwritten=len(build) + len(probe))
    var built = build_side[DType.int64](build, 0, codes)
    probe_side[DType.int64](built, probe, len(build), codes)
    return codes^


def test_a_table_probed_in_two_pieces_answers_as_it_would_in_one() raises:
    var whole = probed(ints([5, 7, 9, 7]), ints([9, 4, 5, 7, 9, 5, 4, 7]))

    var codes = Array[DType.uint32](overwritten=12)
    var built = build_side[DType.int64](ints([5, 7, 9, 7]), 0, codes)
    probe_side[DType.int64](built, ints([9, 4, 5, 7]), 4, codes)
    probe_side[DType.int64](built, ints([9, 5, 4, 7]), 8, codes)

    assert_equal(built.groups(), 4, "three keys and the miss")
    var bad = -1
    for i in range(12):
        if whole[i] != codes[i]:
            bad = i
            break
    assert_equal(bad, -1, String("row ", bad))


def test_a_wide_key_table_probed_in_two_pieces_answers_the_same_way() raises:
    # Keys this far apart have no table indexed by value small enough to be
    # worth building, so this goes through the hash route and the one above
    # does not.
    var far = Int64(1) << 40
    var whole = probed(
        ints([far, 3, far + 9]), ints([3, far + 9, 8, far, 3, 8])
    )

    var codes = Array[DType.uint32](overwritten=9)
    var built = build_side[DType.int64](ints([far, 3, far + 9]), 0, codes)
    probe_side[DType.int64](built, ints([3, far + 9, 8]), 3, codes)
    probe_side[DType.int64](built, ints([far, 3, 8]), 6, codes)

    assert_equal(built.groups(), 4, "three keys and the miss")
    var bad = -1
    for i in range(9):
        if whole[i] != codes[i]:
            bad = i
            break
    assert_equal(bad, -1, String("row ", bad))


def test_a_probe_of_nothing_leaves_the_table_ready_for_the_next_one() raises:
    var codes = Array[DType.uint32](overwritten=6)
    var built = build_side[DType.int64](ints([2, 4]), 0, codes)
    probe_side[DType.int64](built, ints(List[Scalar[DType.int64]]()), 2, codes)
    probe_side[DType.int64](built, ints([4, 5, 2, 4]), 2, codes)

    assert_equal(codes[2], codes[1], "the four found the four")
    assert_equal(codes[4], codes[0], "the two found the two")
    assert_equal(Int(codes[3]), built.groups() - 1, "the five found nothing")


def test_a_table_built_from_one_dtype_refuses_a_probe_of_another() raises:
    var codes = Array[DType.uint32](overwritten=6)
    var built = build_side[DType.int64](ints([1, 2, 3]), 0, codes)
    with assert_raises(contains="the probe column is"):
        probe_side[DType.float64](built, floats([1.0, 2.0]), 3, codes)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
