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
several, so the tests that care drive them by shape instead. A narrow integer key
takes the direct table, a key spread over the whole int64 range takes the hash
table, a string key takes a table of its own when the side that would be built is
much smaller than the side that would be probed, and two key columns or anything
else takes the concat and factorize fallback. Each of those shapes appears here
at least once, and the assertions are the same ones in every case.

The string table is the one with a trap in it. Every other route settles a row by
comparing hashes, which is exact for a fixed width key. A string is not, so the
bytes have to be compared as well, and the bytes the table kept belong to the side
it was built from while the row being asked belongs to the other one. A string of
twelve bytes or fewer carries its own, so a test that wants to reach that seam has
to use longer keys than that and keys that share a prefix, which the tests below
say when they are doing it.

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


def align_text(left: List[String], right: List[String]) raises -> KeyAlignment:
    """Aligns two single column string frames."""
    var rows = len(left)
    var other = len(right)
    return align_owned(
        columns(AnyArray(text(left))),
        one(0),
        rows,
        columns(AnyArray(text(right))),
        one(0),
        other,
    )


def long_keys(count: Int, of: Int) raises -> List[String]:
    """Builds string keys that are too long to be held inline and share a prefix.

    Both of those matter to the string route and neither is true of the short
    keys the tests above use. A string of twelve bytes or fewer lives inside its
    own view and is compared without anyone reading a payload, and a string whose
    first four bytes differ from another's is settled by the prefix check without
    the rest being read. The keys here are eighteen bytes and differ only in the
    last one, so every comparison between two of them goes all the way to the
    bytes in the column.

    Args:
        count: How many keys to produce.
        of: How many distinct keys to draw from, cycled through in order.

    Returns:
        The keys.

    Raises:
        If a key cannot be built.
    """
    var out = List[String](capacity=count)
    for i in range(count):
        out.append(String("shared-prefix-key-", i % of))
    return out^


def long_key_column(count: Int, of: Int) raises -> StringArray:
    """`long_keys` written straight into a column.

    The same keys, without the list of that many separate strings in between.
    One of these columns is long enough that the difference is most of the test.

    Args:
        count: How many rows to produce.
        of: How many distinct keys to draw from, cycled through in order.

    Returns:
        The column.

    Raises:
        If a key cannot be built.
    """
    var seen = long_keys(of, of)
    var out = StringBuilder(capacity=count)
    for i in range(count):
        out.append(seen[i % of].as_bytes())
    return out^.finish()


def test_a_string_key_against_a_much_smaller_side_aligns() raises:
    # Eight left rows for every right row, which is the shape that takes the
    # string table rather than the concat route. The keys are long and share a
    # prefix, so settling one means comparing a probe row's bytes against a view
    # the table kept of a row of the other column.
    var left = long_keys(24, 6)
    var right = List[String]()
    right.append(String("shared-prefix-key-1"))
    right.append(String("shared-prefix-key-4"))
    right.append(String("shared-prefix-key-1"))
    var got = align_text(left, right)

    assert_equal(got.groups, 3, "two keys and the miss")
    for i in range(24):
        var key = i % 6
        if key == 1:
            assert_equal(got.codes[i], got.codes[24], String("key 1 at ", i))
            assert_equal(got.codes[i], got.codes[26], String("repeat at ", i))
        elif key == 4:
            assert_equal(got.codes[i], got.codes[25], String("key 4 at ", i))
        else:
            assert_equal(
                Int(got.codes[i]), got.groups - 1, String("miss at ", i)
            )


def test_a_string_key_aligns_the_same_way_on_either_route() raises:
    # The same twenty four rows asked about the same three keys twice. The first
    # call has the sides far enough apart to take the string table and the second
    # pads the right side out until it declines and the concat route runs. Codes
    # are not comparable across two alignments, since each numbers its own
    # groups, so what is compared is which rows agree with which.
    var left = long_keys(24, 6)
    var small = List[String]()
    small.append(String("shared-prefix-key-1"))
    small.append(String("shared-prefix-key-4"))
    small.append(String("shared-prefix-key-1"))
    var probed = align_text(left, small)

    var padded = small.copy()
    for i in range(20):
        padded.append(String("padding-that-matches-nothing-", i))
    var concat = align_text(left, padded)

    var bad = -1
    for i in range(24):
        for j in range(3):
            var by_table = probed.codes[i] == probed.codes[24 + j]
            var by_concat = concat.codes[i] == concat.codes[24 + j]
            if by_table != by_concat:
                bad = i * 3 + j
                break
    assert_equal(bad, -1, String("left ", bad // 3, " right ", bad % 3))


def test_a_null_string_key_matches_nothing_on_the_string_table() raises:
    # A null on each side, on the shape that takes the string table. Neither one
    # may pair with anything, including the other one.
    var left = StringBuilder(capacity=24)
    for i in range(24):
        if i == 5:
            left.append_null()
        else:
            left.append(String("shared-prefix-key-", i % 6).as_bytes())
    var right = StringBuilder(capacity=3)
    right.append(String("shared-prefix-key-1").as_bytes())
    right.append_null()
    right.append(String("shared-prefix-key-4").as_bytes())

    var got = align_owned(
        columns(AnyArray(left^.finish())),
        one(0),
        24,
        columns(AnyArray(right^.finish())),
        one(0),
        3,
    )
    assert_true(got.has_nulls, "has_nulls")
    assert_true(got.absent[5], "the null left row")
    assert_true(got.absent[25], "the null right row")
    assert_true(not matched(got, 24, 3, 5), "the null left row pairs with none")
    assert_equal(got.codes[1], got.codes[24], "key 1 still pairs")
    assert_equal(got.codes[4], got.codes[26], "key 4 still pairs")


def test_a_string_probe_past_the_split_aligns_what_one_thread_would() raises:
    # Over `PARALLEL_PROBE_ROWS`, so the string probe hands itself out in
    # morsels, and far enough over the build side that it is the route taken.
    # Every fourth key is on the right and the answer is checked row by row
    # against what the key alone says it should be.
    comptime rows = (1 << 17) + 5
    var right = StringBuilder(capacity=64)
    for i in range(64):
        right.append(String("shared-prefix-key-", i * 4).as_bytes())

    var got = align_owned(
        columns(AnyArray(long_key_column(rows, 256))),
        one(0),
        rows,
        columns(AnyArray(right^.finish())),
        one(0),
        64,
    )
    var bad = -1
    for i in range(rows):
        var want_miss = (i % 256) % 4 != 0
        var is_miss = Int(got.codes[i]) == got.groups - 1
        if want_miss != is_miss:
            bad = i
            break
    assert_equal(bad, -1, String("row ", bad))

    var wrong = -1
    for r in range(64):
        if got.codes[rows + r] != got.codes[r * 4]:
            wrong = r
            break
    assert_equal(wrong, -1, String("right row ", wrong))


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


def test_probing_on_one_core_gives_what_probing_on_all_of_them_gives() raises:
    """The two routes through `spread` write the same ordinals.

    Above `PARALLEL_PROBE_ROWS` the probe hands itself out in morsels, and a
    caller that is already running on a worker passes False to stop it. That is
    a scheduling decision and it must not be a correctness one, so the same
    column goes through both ways and the ordinals are compared row by row.
    """
    var rows = (1 << 17) + 5
    var probe = Array[DType.int64](overwritten=rows)
    for i in range(rows):
        probe[i] = Int64(i % 7)

    var build_codes = Array[DType.uint32](overwritten=4)
    var built = build_side[DType.int64](ints([0, 2, 4, 6]), 0, build_codes)

    var spread = Array[DType.uint32](overwritten=rows)
    probe_side[DType.int64](built, probe, 0, spread)
    var alone = Array[DType.uint32](overwritten=rows)
    probe_side[DType.int64](built, probe, 0, alone, False)

    var bad = -1
    for i in range(rows):
        if spread[i] != alone[i]:
            bad = i
            break
    assert_equal(bad, -1, String("row ", bad))


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
