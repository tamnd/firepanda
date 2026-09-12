"""Tests for choosing between two columns on a condition.

Two things here are easy to get wrong and both have a test aimed straight at
them. The first is the null condition, which takes the false side rather than
answering null, and which the kernel gets for free by reading the values buffer
and never looking at the condition's validity. A kernel that consulted the
validity would still pass every test where the condition has no nulls, so the
one where it does is doing all the work.

The second is the output's validity, which is the only part a SIMD select cannot
compute, and which is built a word at a time out of bits packed from bytes. That
packing runs over sixty four rows whether or not the column has that many, so the
short columns and the ones that end part way through a word are as interesting as
the long ones.
"""

from std.testing import TestSuite, assert_equal, assert_false, assert_raises

from firepanda.array.array import Array
from firepanda.array.strings import (
    StringArray,
    StringBuilder,
    strings_from_list,
)
from firepanda.exec.morsel import MORSEL_ROWS
from firepanda.frame.frame import DataFrame
from firepanda.frame.groupby import AggSpec
from firepanda.frame.series import Series
from firepanda.kernel.group import AggKind
from firepanda.kernel.agg import sum_of
from firepanda.kernel.compare import not_equal
from firepanda.kernel.concat import concat_arrays
from firepanda.kernel.pick import (
    pick,
    pick_const,
    pick_constants,
    pick_one,
    text_pick,
)
from firepanda.kernel.scalar import (
    pick_const_scalar,
    pick_constants_scalar,
    pick_scalar,
    text_pick_scalar,
)


def numbers(values: List[Int]) -> Array[DType.int64]:
    """Builds an int64 column from a list.

    Args:
        values: The values.

    Returns:
        The column.
    """
    var out = Array[DType.int64](len(values))
    for i in range(len(values)):
        out.set_valid(i, Int64(values[i]))
    return out^


def flags(values: List[Int]) -> Array[DType.bool]:
    """Builds a condition column, where a two means null.

    Args:
        values: Zero for false, one for true, two for null.

    Returns:
        The column, with the nulls holding zero the way the invariant says.
    """
    var out = Array[DType.bool](len(values))
    for i in range(len(values)):
        if values[i] == 2:
            out.set_null(i)
        else:
            out.set_valid(i, values[i] == 1)
    return out^


def agrees[dt: DType](got: Array[dt], want: Array[dt], label: String) raises:
    """Asserts that a kernel answer matches the twin's, row by row.

    Args:
        got: The kernel's answer.
        want: The twin's answer.
        label: What to name in the failure.

    Parameters:
        dt: The dtype of both.

    Raises:
        AssertionError: On the first row that differs.
    """
    assert_equal(len(got), len(want), label + ": lengths differ")
    for i in range(len(got)):
        assert_equal(got.is_valid(i), want.is_valid(i), label + ": validity")
        if got.is_valid(i):
            assert_equal(got[i], want[i], label + ": row " + String(i))


def over(
    cond: Array[DType.bool],
    a: Array[DType.int64],
    b: Array[DType.int64],
    label: String,
) raises:
    """Runs all three number entry points against their twins.

    The constant forms are given `b`'s first value, so a caller reading a failure
    can compare the three answers against each other as well as against the
    twins.

    Args:
        cond: The condition.
        a: The true side.
        b: The false side.
        label: What to name in the failure.

    Raises:
        AssertionError: If any of the three disagrees with its twin.
    """
    agrees(pick(cond, a, b), pick_scalar(cond, a, b), label + " columns")
    agrees(
        pick_const(cond, a, Int64(-7)),
        pick_const_scalar(cond, a, Int64(-7)),
        label + " constant on the false side",
    )
    agrees(
        pick_constants(cond, Int64(1), Int64(0)),
        pick_constants_scalar(cond, Int64(1), Int64(0)),
        label + " two constants",
    )


def test_a_condition_with_no_nulls_matches_the_twins() raises:
    var cond = flags([1, 0, 1, 1, 0])
    var a = numbers([10, 20, 30, 40, 50])
    var b = numbers([-1, -2, -3, -4, -5])
    over(cond, a, b, "plain")

    var mask = pick(cond, a, b)
    assert_equal(mask[0], 10)
    assert_equal(mask[1], -2)
    assert_equal(mask[4], -5)


def test_a_null_condition_takes_the_false_side() raises:
    # The rule the whole kernel is built around. It is SQL's answer for
    # `CASE WHEN NULL`, and it is not polars', which gives a null.
    var cond = flags([2, 1, 2, 0])
    var a = numbers([10, 20, 30, 40])
    var b = numbers([-1, -2, -3, -4])
    over(cond, a, b, "null condition")

    var got = pick(cond, a, b)
    assert_equal(got[0], -1)
    assert_equal(got[1], 20)
    assert_equal(got[2], -3)
    assert_equal(got[3], -4)


def test_a_null_on_the_chosen_side_comes_through() raises:
    var cond = flags([1, 1, 0, 0])
    var a = numbers([10, 20, 30, 40])
    a.set_null(1)
    var b = numbers([-1, -2, -3, -4])
    b.set_null(3)
    over(cond, a, b, "nulls on both sides")

    var got = pick(cond, a, b)
    assert_equal(got[0], 10)
    assert_false(got.is_valid(1), "took a null from the true side")
    assert_equal(got[2], -3)
    assert_false(got.is_valid(3), "took a null from the false side")
    # The other side's null is not visible when it was not chosen.
    assert_equal(got.null_count(), 2)


def test_a_null_under_the_output_still_holds_a_zero() raises:
    # `kernel/__init__.mojo` says a null value is zero in the values buffer, and
    # this kernel never repairs the output because the select copies the zero
    # from whichever side was null. That reasoning is what this checks.
    var cond = flags([1, 0])
    var a = numbers([10, 20])
    a.set_null(0)
    var b = numbers([-1, -2])
    b.set_null(1)
    var got = pick(cond, a, b)
    assert_false(got.is_valid(0))
    assert_false(got.is_valid(1))
    assert_equal(got.unsafe_ptr().unsafe_offset(0).unsafe_load(), 0)
    assert_equal(got.unsafe_ptr().unsafe_offset(1).unsafe_load(), 0)


def test_a_constant_on_the_false_side_is_never_null() raises:
    var cond = flags([1, 0, 2])
    var a = numbers([10, 20, 30])
    a.set_null(0)
    a.set_null(1)
    var got = pick_const(cond, a, Int64(99))
    assert_false(got.is_valid(0), "the true side's null came through")
    # Row one took the constant even though `a` is null there, and row two took
    # it because the condition was null.
    assert_equal(got[1], 99)
    assert_equal(got[2], 99)


def test_two_constants_never_produce_a_null() raises:
    var cond = flags([1, 0, 2, 1])
    var got = pick_constants(cond, Int64(1), Int64(0))
    assert_equal(got.null_count(), 0)
    assert_equal(Int(sum_of(got).value), 2)


def test_a_word_hanging_over_the_end_of_the_column() raises:
    # `pick_const` builds the output validity as "valid wherever the condition
    # is false", and every bit past the end of the column is a false. Left
    # unmasked those come out set, and since `count_ones` walks whole words the
    # column then reports fewer nulls than it has. Reading a row would never
    # show it, which is why this asks for the count.
    for length in [1, 3, 33, 65, 100]:
        var cond = Array[DType.bool](length)
        var a = Array[DType.int64](length)
        for i in range(length):
            cond.set_valid(i, True)
            a.set_valid(i, Int64(i))
        a.set_null(0)
        var got = pick_const(cond, a, Int64(99))
        assert_equal(got.null_count(), 1, "length " + String(length))
        assert_equal(
            got.data.validity.count_ones(),
            length - 1,
            "bits set past the end at length " + String(length),
        )


def test_a_column_ending_part_way_through_a_word() raises:
    # The validity is built sixty four rows at a time and these columns are not a
    # multiple of that, so the last word is asked for rows that do not exist.
    for length in [1, 7, 63, 64, 65, 129]:
        var cond = Array[DType.bool](length)
        var a = Array[DType.int64](length)
        var b = Array[DType.int64](length)
        for i in range(length):
            cond.set_valid(i, i % 3 == 0)
            a.set_valid(i, Int64(i))
            b.set_valid(i, Int64(-i))
        a.set_null(length - 1)
        if length > 1:
            b.set_null(0)
        agrees(
            pick(cond, a, b),
            pick_scalar(cond, a, b),
            "length " + String(length),
        )


def test_a_column_either_side_of_the_morsel_split_matches_the_twin() raises:
    var unit = numbers([4, 17, 5, 99])
    var col = unit.copy()
    while len(col) < MORSEL_ROWS:
        var pair = List[Array[DType.int64]]()
        pair.append(col.copy())
        pair.append(col.copy())
        col = concat_arrays(pair)
    var tail = List[Array[DType.int64]]()
    tail.append(col^)
    tail.append(unit.copy())
    col = concat_arrays(tail)
    var n = len(col)
    assert_equal(n, MORSEL_ROWS + 4)

    var cond = Array[DType.bool](n)
    var other = Array[DType.int64](n)
    for i in range(n):
        cond.set_valid(i, i % 2 == 0)
        other.set_valid(i, Int64(-i))
    # A null in the second morsel, which is where a worker repairing or packing
    # the wrong word range comes apart.
    other.set_null(MORSEL_ROWS + 1)

    var want = pick_scalar(cond, col, other)
    assert_equal(len(want), n)
    var got = pick(cond, col, other)
    assert_equal(len(got), n)
    assert_equal(Int(sum_of(not_equal(got, want)).value), 0, "rows disagreeing")
    assert_equal(got.null_count(), want.null_count())


def test_text_matches_the_twin() raises:
    var builder = StringBuilder(capacity=5)
    builder.append("short".as_bytes())
    builder.append_null()
    builder.append("a value far too long to sit inside a view".as_bytes())
    builder.append("".as_bytes())
    builder.append("last".as_bytes())
    var a = builder^.finish()
    var b = strings_from_list(["A", "B", "C", "D", "E"])
    var cond = flags([1, 1, 0, 2, 1])

    var got = text_pick(cond, a, b)
    var want = text_pick_scalar(cond, a, b)
    assert_equal(len(got), len(want))
    for i in range(len(got)):
        assert_equal(got.is_valid(i), want.is_valid(i), "row " + String(i))
        if got.is_valid(i):
            assert_equal(got[i], want[i], "row " + String(i))
    assert_equal(got[0], "short")
    assert_false(got.is_valid(1), "took the true side's null")
    assert_equal(got[2], "C")
    assert_equal(got[3], "D")


def test_text_crosses_the_morsel_split() raises:
    """The two pass build is where a wrong prefix or a shared word comes apart.

    Long elements on the true side so the payload is really used, short ones on
    the false side so both kinds of view go through the same fill, a null in
    each side well past the first morsel boundary, and a height that is a
    multiple of neither the morsel size nor sixty four.
    """
    var n = MORSEL_ROWS + 37
    # Built once and appended by index, because formatting a string per row
    # costs many times what the kernel under test costs.
    var pool = List[String]()
    for k in range(8):
        pool.append(String("a left element far too long to sit in a view ", k))
    var tiny = List[String]()
    for k in range(8):
        tiny.append(String("r", k))

    var left = StringBuilder(capacity=n)
    var right = StringBuilder(capacity=n)
    var cond = Array[DType.bool](n)
    for i in range(n):
        cond.set_valid(i, i % 3 != 0)
        if i % 17 == 0:
            left.append_null()
        else:
            left.append(pool[i % 8].as_bytes())
        if i % 23 == 0:
            right.append_null()
        else:
            right.append(tiny[i % 8].as_bytes())
    var a = left^.finish()
    var b = right^.finish()

    var got = text_pick(cond, a, b)
    var want = text_pick_scalar(cond, a, b)
    assert_equal(len(got), n)
    assert_equal(got.null_count(), want.null_count(), "nulls")

    # Counted rather than asserted a row at a time, because the message a
    # per row assert formats is built whether or not the row fails and a
    # hundred and thirty thousand of them costs more than the kernel does.
    var wrong = -1
    for i in range(n):
        if got.is_valid(i) != want.is_valid(i) or (
            got.is_valid(i) and got.unsafe_bytes(i) != want.unsafe_bytes(i)
        ):
            wrong = i
            break
    assert_equal(wrong, -1, "first row disagreeing with the twin")


def test_text_with_nothing_long_in_it_builds_no_payload() raises:
    """Neither side has a payload, so the counting pass is skipped outright.

    Every element sits inside its own view, so the output's payload is never
    written and the views are carried across whole.
    """
    var n = MORSEL_ROWS + 5
    var yes = List[String]()
    var no = List[String]()
    for k in range(8):
        yes.append(String("y", k))
        no.append(String("n", k))

    var left = StringBuilder(capacity=n)
    var right = StringBuilder(capacity=n)
    var cond = Array[DType.bool](n)
    for i in range(n):
        cond.set_valid(i, i % 2 == 0)
        left.append(yes[i % 8].as_bytes())
        right.append(no[i % 8].as_bytes())
    var a = left^.finish()
    var b = right^.finish()
    assert_equal(len(a.payload), 0, "the true side is all inline")
    assert_equal(len(b.payload), 0, "the false side is all inline")

    var got = text_pick(cond, a, b)
    var want = text_pick_scalar(cond, a, b)
    assert_equal(len(got), n)
    assert_equal(len(got.payload), 0, "the answer is all inline too")
    var wrong = -1
    for i in range(n):
        if got.unsafe_bytes(i) != want.unsafe_bytes(i):
            wrong = i
            break
    assert_equal(wrong, -1, "first row disagreeing with the twin")


def test_a_null_text_condition_takes_the_false_side() raises:
    """The same rule the numeric kernels follow, which is SQL's.

    Worth its own test over text because the text build reads the condition
    twice, once to size and once to fill, and a disagreement between the two
    would put the bytes of one side under the view of the other.
    """
    var a = strings_from_list(
        ["a value far too long to sit inside a view", "x"]
    )
    var b = strings_from_list(
        ["another value far too long to sit in a view", "y"]
    )
    var cond = Array[DType.bool](2)
    cond.set_null(0)
    cond.set_valid(1, True)

    var got = text_pick(cond, a, b)
    assert_equal(got[0], "another value far too long to sit in a view")
    assert_equal(got[1], "x")


def test_a_picked_text_column_groups_by() raises:
    """The only thing q39 does with the column this kernel builds.

    `CASE WHEN (SearchEngineID = 0 AND AdvEngineID = 0) THEN Referer ELSE \'\'
    END AS Src` is a group by key, so the answer has to be a column the hash
    table will take. It is worth a test of its own because the two pass build
    writes the views and the payload separately, and a column whose views are
    right and whose payload offsets are not reads correctly one element at a
    time and groups wrongly.
    """
    var referer = strings_from_list(
        [
            "http://example.com/a/very/long/path/that/does/not/inline",
            "http://example.com/a/very/long/path/that/does/not/inline",
            "short",
            "http://other.example.com/another/long/path/that/does/not/inline",
            "short",
        ]
    )
    var blank = strings_from_list(["", "", "", "", ""])
    var direct = flags([1, 1, 1, 0, 0])

    var src = text_pick(direct, referer, blank)
    var series = List[Series]()
    series.append(Series("src", src^))
    series.append(Series("hits", numbers([1, 2, 4, 8, 16])))
    var frame = DataFrame.from_series(series^)

    var by = List[String]()
    by.append("src")
    var specs = List[AggSpec]()
    specs.append(AggSpec("hits", AggKind.SUM))
    var out = frame.group_by(by, specs)

    # Three groups: the long referer twice, "short" once, and the empty string
    # the two false rows both took.
    assert_equal(len(out), 3)
    var keys = out.column("src").as_strings()
    var sums = out.column("hits_sum").as_typed[DType.int64]()
    for i in range(3):
        if keys[i] == "short":
            assert_equal(Int(sums[i]), 4, "the inline group")
        elif keys[i] == "":
            assert_equal(Int(sums[i]), 24, "the false side's group")
        else:
            assert_equal(Int(sums[i]), 3, "the long referer's group")


def test_a_false_side_of_one_row_is_that_row_everywhere() raises:
    var cond = flags([1, 0, 2, 1, 0])
    var a = numbers([10, 20, 30, 40, 50])
    agrees(
        pick_one(cond, a, numbers([-7])),
        pick_const(cond, a, Int64(-7)),
        "one row against the constant it stands for",
    )
    var got = pick_one(cond, a, numbers([-7]))
    assert_equal(got[0], 10)
    assert_equal(got[1], -7)
    # The null condition takes the false side here too, which is the one row.
    assert_equal(got[2], -7)


def test_a_false_side_of_one_null_row_leaves_those_rows_missing() raises:
    # The case a `Scalar` cannot express, which is why this is a column at all.
    var cond = flags([1, 0, 1, 2])
    var a = numbers([10, 20, 30, 40])
    a.set_null(2)
    var nothing = numbers([0])
    nothing.set_null(0)

    var got = pick_one(cond, a, nothing)
    assert_equal(got[0], 10)
    assert_false(got.is_valid(1), "took the row with nothing in it")
    assert_false(got.is_valid(2), "kept the true side's own null")
    assert_false(got.is_valid(3), "a null condition took the false side")
    assert_equal(got.null_count(), 3)
    # The invariant the rest of the module rests on: a null holds a zero.
    assert_equal(got[1], 0)


def test_a_one_row_false_side_clears_the_bits_past_the_end() raises:
    # Past a word boundary and ending part way through the next one, which is
    # where a tail that was left set would be counted as a row.
    var rows = 130
    var cond = Array[DType.bool](rows)
    var a = Array[DType.int64](rows)
    for i in range(rows):
        cond.set_valid(i, i % 3 == 0)
        a.set_valid(i, Int64(i))
    var nothing = numbers([0])
    nothing.set_null(0)

    var got = pick_one(cond, a, nothing)
    assert_equal(len(got), rows)
    var kept = 0
    for i in range(rows):
        if i % 3 == 0:
            kept += 1
    assert_equal(got.null_count(), rows - kept)


def test_a_one_row_false_side_of_text_carries_its_bytes() raises:
    # The long value is longer than a view, so it lives in the payload and both
    # passes have to agree that every false row needs room for it.
    var cond = flags([1, 0, 2, 0])
    var a = strings_from_list(["first", "second", "third", "fourth"])
    var one = strings_from_list(["a value too long to sit inside a view"])
    var got = text_pick(cond, a, one)
    assert_equal(len(got), 4)
    assert_equal(String(got[0]), "first")
    assert_equal(String(got[1]), "a value too long to sit inside a view")
    assert_equal(String(got[2]), "a value too long to sit inside a view")
    assert_equal(String(got[3]), "a value too long to sit inside a view")


def test_a_series_pick_against_one_row_is_how_a_constant_is_spelled() raises:
    var revenue = Series("revenue", numbers([100, 200, 400]))
    var zero = Series("zero", numbers([0]))
    var kept = revenue.pick(flags([1, 0, 1]), zero)
    assert_equal(len(kept), 3)
    assert_equal(kept.name, "revenue")
    assert_equal(Int(sum_of(kept.as_typed[DType.int64]()).value), 500)


def test_a_frame_pick_leaves_the_other_columns_alone() raises:
    var frame = DataFrame.from_series(
        [
            Series("a", numbers([1, 2, 3])),
            Series("b", numbers([10, 20, 30])),
        ]
    )
    var chosen = frame.pick("a", flags([1, 0, 1]), Series("z", numbers([-1])))
    assert_equal(len(chosen), 3)
    assert_equal(chosen.column("a").as_typed[DType.int64]()[1], -1)
    assert_equal(chosen.column("b").as_typed[DType.int64]()[1], 20)


def test_a_false_side_that_is_neither_one_row_nor_all_is_refused() raises:
    var cond = flags([1, 0, 1])
    with assert_raises(contains="one row or as many as the condition"):
        _ = Series("a", numbers([1, 2, 3])).pick(
            cond, Series("b", numbers([1, 2]))
        )


def test_a_length_mismatch_is_refused() raises:
    var cond = flags([1, 0])
    with assert_raises(contains="pick"):
        _ = pick(cond, numbers([1, 2, 3]), numbers([1, 2]))
    with assert_raises(contains="pick"):
        _ = pick_const(cond, numbers([1, 2, 3]), Int64(0))


def test_a_series_pick_is_the_q14_shape() raises:
    # `sum(case when p_type like 'PROMO%' then revenue else 0 end)`, which is
    # what q14 divides by the total.
    var kinds = Series(
        "p_type",
        strings_from_list(["PROMO BRUSHED", "STANDARD", "PROMO PLATED"]),
    )
    var revenue = Series("revenue", numbers([100, 200, 400]))
    var zero = Series("zero", numbers([0, 0, 0]))
    var promo = revenue.pick(kinds.str_starts_with("PROMO"), zero)
    assert_equal(len(promo), 3)
    assert_equal(promo.name, "revenue")
    assert_equal(Int(sum_of(promo.as_typed[DType.int64]()).value), 500)


def test_a_series_pick_across_types_is_refused() raises:
    var cond = flags([1, 0])
    var a = Series("a", numbers([1, 2]))
    var b = Series("b", strings_from_list(["x", "y"]))
    with assert_raises(contains="cannot choose between"):
        _ = a.pick(cond, b)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
