"""Tests for the per group top-n kernel.

The small tests pin down the answers a person can work out by hand: which rows
come back, in what order, what a tie does, and what a group with fewer than `n`
present values gives.

The large one is different in kind, because writing the expected answer down
would mean writing the kernel a second time and the second copy would have the
same bug as the first. So it checks a property instead, and the property is the
definition of a top-n: no row outside a group's kept set beats the worst row
inside it, and exactly `kept - 1` rows beat that worst row. Both halves are one
pass over the column and neither of them knows how the kernel works. It is sized
past `TOP_PRIVATE_ROWS` so the parallel route and the fold are what get checked.

The last group of tests is the ungrouped frame spelling, `nlargest` and
`nsmallest`, which is the same kernel with one group in it. Those answers are
written down from what pandas gives for the same eight rows, because the
ordering of a tie is the part that is easy to get almost right.
"""

from std.testing import TestSuite, assert_equal, assert_raises

from firepanda.array.any import AnyArray
from firepanda.array.array import Array
from firepanda.array.strings import strings_from_list
from firepanda.frame.frame import DataFrame
from firepanda.frame.series import Series
from firepanda.kernel.topn import group_top_rows, group_top_rows_any


def _codes(values: List[Int]) raises -> Array[DType.uint32]:
    """Builds a group ordinal column out of a list.

    Args:
        values: The ordinals, one per row.

    Returns:
        The column, every row present.
    """
    var out = Array[DType.uint32](len(values))
    for i in range(len(values)):
        out.set_valid(i, UInt32(values[i]))
    return out^


def test_the_two_largest_of_every_group_come_back_best_first() raises:
    var col = Array[DType.float64](7)
    var wanted = [3.0, 1.0, 5.0, 9.0, 2.0, 4.0, 7.0]
    for i in range(7):
        col.set_valid(i, wanted[i])
    var codes = _codes([0, 0, 0, 1, 1, 1, 1])

    var top = group_top_rows(col, codes, 2, 2, True)
    assert_equal(len(top.counts), 2, "one count per group")
    assert_equal(top.counts[0], 2, "the first group kept two")
    assert_equal(top.counts[1], 2, "the second group kept two")
    assert_equal(len(top.rows_at), 4, "four rows in total")
    assert_equal(top.rows_at[0], 2, "the first group's best row")
    assert_equal(top.rows_at[1], 0, "the first group's second best row")
    assert_equal(top.rows_at[2], 3, "the second group's best row")
    assert_equal(top.rows_at[3], 6, "the second group's second best row")


def test_the_smallest_are_the_largest_read_the_other_way() raises:
    var col = Array[DType.int64](6)
    var wanted = [8, 2, 6, 1, 9, 4]
    for i in range(6):
        col.set_valid(i, Int64(wanted[i]))
    var codes = _codes([0, 0, 0, 1, 1, 1])

    var top = group_top_rows(col, codes, 2, 2, False)
    assert_equal(top.rows_at[0], 1, "the first group's smallest row")
    assert_equal(top.rows_at[1], 2, "the first group's next smallest row")
    assert_equal(top.rows_at[2], 3, "the second group's smallest row")
    assert_equal(top.rows_at[3], 5, "the second group's next smallest row")


def test_a_tie_keeps_the_row_that_came_first() raises:
    # Every value in the group is the same, so nothing but the row number can
    # decide which two of the five are kept.
    var col = Array[DType.int64](5)
    for i in range(5):
        col.set_valid(i, Int64(4))
    var codes = _codes([0, 0, 0, 0, 0])

    var top = group_top_rows(col, codes, 1, 2, True)
    assert_equal(top.counts[0], 2, "two rows kept")
    assert_equal(top.rows_at[0], 0, "the first row wins the tie")
    assert_equal(top.rows_at[1], 1, "the second row takes the other slot")

    var bottom = group_top_rows(col, codes, 1, 2, False)
    assert_equal(bottom.rows_at[0], 0, "a tie goes the same way either end")
    assert_equal(bottom.rows_at[1], 1, "and so does the second slot")


def test_a_null_is_never_a_candidate() raises:
    var col = Array[DType.float64](6)
    col.set_valid(0, 1.0)
    col.set_null(1)
    col.set_valid(2, 3.0)
    col.set_null(3)
    col.set_null(4)
    col.set_valid(5, 2.0)
    var codes = _codes([0, 0, 0, 1, 1, 1])

    var top = group_top_rows(col, codes, 2, 3, True)
    assert_equal(top.counts[0], 2, "the first group had two present values")
    assert_equal(top.counts[1], 1, "the second group had one")
    assert_equal(len(top.rows_at), 3, "three rows in total")
    assert_equal(top.rows_at[0], 2, "the first group's best")
    assert_equal(top.rows_at[1], 0, "the first group's other one")
    assert_equal(top.rows_at[2], 5, "the second group's only present row")


def test_a_group_with_nothing_present_keeps_nothing() raises:
    var col = Array[DType.int64](4)
    col.set_valid(0, Int64(5))
    col.set_valid(1, Int64(6))
    col.set_null(2)
    col.set_null(3)
    var codes = _codes([0, 0, 1, 1])

    var top = group_top_rows(col, codes, 2, 2, True)
    assert_equal(top.counts[0], 2, "the first group kept both")
    assert_equal(top.counts[1], 0, "the second kept nothing")
    assert_equal(len(top.rows_at), 2, "and contributed no rows")


def test_a_nan_is_dropped_the_way_a_null_is() raises:
    # A NaN loses every comparison it is in, so one left in a slot would hold a
    # real value out. This is the test that says it never gets there.
    var col = Array[DType.float64](4)
    var nan = Float64(0.0) / Float64(0.0)
    col.set_valid(0, nan)
    col.set_valid(1, 2.0)
    col.set_valid(2, nan)
    col.set_valid(3, 1.0)
    var codes = _codes([0, 0, 0, 0])

    var top = group_top_rows(col, codes, 1, 2, True)
    assert_equal(top.counts[0], 2, "only the two real values were candidates")
    assert_equal(top.rows_at[0], 1, "the larger real value")
    assert_equal(top.rows_at[1], 3, "the smaller one")


def test_asking_for_more_rows_than_a_group_has_gives_what_there_is() raises:
    var col = Array[DType.int64](3)
    for i in range(3):
        col.set_valid(i, Int64(i))
    var codes = _codes([0, 0, 0])

    var top = group_top_rows(col, codes, 1, 10, True)
    assert_equal(top.counts[0], 3, "three rows is all there was")
    assert_equal(top.rows_at[0], 2, "still in ranking order")
    assert_equal(top.rows_at[1], 1, "second")
    assert_equal(top.rows_at[2], 0, "third")


def test_the_erased_spelling_agrees_with_the_typed_one() raises:
    var col = Array[DType.int32](8)
    for i in range(8):
        col.set_valid(i, Int32((i * 5) % 8))
    var codes = _codes([0, 1, 0, 1, 0, 1, 0, 1])

    var typed = group_top_rows(col, codes, 2, 3, True)
    var erased = group_top_rows_any(AnyArray(col.copy()), codes, 2, 3, True)
    assert_equal(len(erased.rows_at), len(typed.rows_at), "same row count")
    for i in range(len(typed.rows_at)):
        assert_equal(erased.rows_at[i], typed.rows_at[i], "row " + String(i))


def test_bad_arguments_are_refused() raises:
    var col = Array[DType.int64](4)
    var codes = _codes([0, 0, 0, 0])
    var raised = 0
    try:
        _ = group_top_rows(col, codes, 1, 0, True)
    except:
        raised += 1
    try:
        _ = group_top_rows(col, _codes([0, 0]), 1, 2, True)
    except:
        raised += 1
    assert_equal(raised, 2, "both arguments were refused")


def test_past_the_split_no_row_outside_a_group_beats_the_worst_inside() raises:
    # Sized past TOP_PRIVATE_ROWS so the private tables and the fold are what is
    # under test, and not a multiple of the worker count so the row split is
    # uneven.
    comptime rows = 150_011
    comptime groups = 499
    comptime n = 3

    var col = Array[DType.float64](rows)
    var codes = Array[DType.uint32](rows)
    var seed = UInt64(0x2545F4914F6CDD1D)
    for i in range(rows):
        seed = seed * 6364136223846793005 + 1442695040888963407
        var draw = Int((seed >> 33) % 100_000)
        codes.set_valid(i, UInt32(i % groups))
        # Every seventeenth row is missing, which leaves a few groups short of
        # three present values and exercises the count as well as the ranking.
        if i % 17 == 0:
            col.set_null(i)
        else:
            col.set_valid(i, Float64(draw) / 1000.0)

    var top = group_top_rows(col, codes, groups, n, True)
    assert_equal(len(top.counts), groups, "one count per group")

    var present = List[Int](length=groups, fill=0)
    for i in range(rows):
        if col.is_valid(i):
            present[i % groups] += 1

    var total = 0
    for g in range(groups):
        var want = n if present[g] > n else present[g]
        assert_equal(top.counts[g], want, "group " + String(g) + " count")
        total += top.counts[g]
    assert_equal(len(top.rows_at), total, "the rows and the counts agree")

    # The worst row each group kept, which is the last one because the slots
    # come back best first.
    var edge = List[Int](length=groups, fill=-1)
    var edge_value = List[Float64](length=groups, fill=0.0)
    var at = 0
    var wrong = 0
    for g in range(groups):
        var have = top.counts[g]
        for k in range(have):
            var row = top.rows_at[at + k]
            if row % groups != g or not col.is_valid(row):
                wrong += 1
        if have > 0:
            edge[g] = top.rows_at[at + have - 1]
            edge_value[g] = col[edge[g]]
        at += have
    assert_equal(wrong, 0, "kept rows that were null or in the wrong group")

    # Nothing knows how the kernel works from here down. A row beats another if
    # its value is larger, or the values are equal and it came first, and the
    # number of rows that beat a group's worst kept row has to be one less than
    # the number it kept.
    var better = List[Int](length=groups, fill=0)
    for i in range(rows):
        if not col.is_valid(i):
            continue
        var g = i % groups
        var against = edge[g]
        if against < 0:
            continue
        var mark = edge_value[g]
        if col[i] > mark or (col[i] == mark and i < against):
            better[g] += 1

    var off = 0
    for g in range(groups):
        if top.counts[g] == 0:
            continue
        if better[g] != top.counts[g] - 1:
            off += 1
    assert_equal(off, 0, "groups whose worst kept row was not the nth best")


def test_the_frame_spelling_keeps_the_rows_the_kernel_picked() raises:
    var key = Array[DType.int64](6)
    var value = Array[DType.float64](6)
    var keys = [1, 2, 1, 2, 1, 2]
    var values = [10.0, 40.0, 30.0, 20.0, 50.0, 60.0]
    for i in range(6):
        key.set_valid(i, Int64(keys[i]))
        value.set_valid(i, values[i])
    var series = List[Series]()
    series.append(Series("k", key^))
    series.append(Series("v", value^))
    var df = DataFrame.from_series(series^)

    var by = List[String]()
    by.append("k")

    var top = df.group_nlargest(by, "v", 2)
    assert_equal(len(top), 4, "two rows per group")
    assert_equal(top.width(), 2, "the frame keeps its columns")
    var kept = top.column("v").as_typed[DType.float64]()
    assert_equal(kept[0], 50.0, "the first group's best")
    assert_equal(kept[1], 30.0, "the first group's second best")
    assert_equal(kept[2], 60.0, "the second group's best")
    assert_equal(kept[3], 40.0, "the second group's second best")

    var bottom = df.group_nsmallest(by, "v", 1)
    assert_equal(len(bottom), 2, "one row per group")
    var least = bottom.column("v").as_typed[DType.float64]()
    assert_equal(least[0], 10.0, "the first group's smallest")
    assert_equal(least[1], 20.0, "the second group's smallest")


def _ranked() raises -> DataFrame:
    """A frame with a tied ranking column and a row number beside it.

    The keys repeat on purpose, so that every answer below turns on the tie
    rule rather than on the ordering of distinct values. The `row` column is
    there because asserting on which rows came back is the whole question and
    reading them out of a column is plainer than reading them off the labels.

    Returns:
        The frame, eight rows of it.
    """
    var row = Array[DType.int64](8)
    var key = Array[DType.int64](8)
    var keys = [3, 1, 3, 2, 1, 3, 2, 1]
    for i in range(8):
        row.set_valid(i, Int64(i))
        key.set_valid(i, Int64(keys[i]))
    var series = List[Series]()
    series.append(Series("row", row^))
    series.append(Series("key", key^))
    return DataFrame.from_series(series^)


def _rows_of(df: DataFrame) raises -> List[Int]:
    """Reads the row numbers a result came back with.

    Args:
        df: A frame that came out of one of the top n methods.

    Returns:
        The `row` column as a list, in the order the frame holds it.
    """
    var column = df.column("row").as_typed[DType.int64]()
    var out = List[Int](capacity=len(column))
    for i in range(len(column)):
        out.append(Int(column[i]))
    return out^


def test_the_largest_rows_of_a_whole_frame_come_back_best_first() raises:
    var got = _rows_of(_ranked().nlargest("key", 4))
    assert_equal(len(got), 4, "four rows kept")
    assert_equal(got[0], 0, "the first three in the tie at the top")
    assert_equal(got[1], 2, "then the second of them")
    assert_equal(got[2], 5, "then the third")
    assert_equal(got[3], 3, "then the only two")


def test_the_smallest_rows_are_the_largest_read_the_other_way() raises:
    var got = _rows_of(_ranked().nsmallest("key", 3))
    assert_equal(len(got), 3, "three rows kept")
    assert_equal(got[0], 1, "the first one")
    assert_equal(got[1], 4, "the second one")
    assert_equal(got[2], 7, "the third one")


def test_keeping_the_last_of_a_tie_hands_them_back_reversed() raises:
    # pandas answers 5, 2, 0, 6 here, and the order matters as much as the
    # membership does. Keeping the last of a tie means reading the column back
    # to front, so the tied rows arrive in reverse order of appearance.
    var got = _rows_of(_ranked().nlargest("key", 4, "last"))
    assert_equal(len(got), 4, "four rows kept")
    assert_equal(got[0], 5, "the last of the tie at the top comes first")
    assert_equal(got[1], 2, "then the one before it")
    assert_equal(got[2], 0, "then the first of them")
    assert_equal(got[3], 6, "then the later of the two")


def test_keeping_the_last_reads_the_small_end_the_same_way() raises:
    var got = _rows_of(_ranked().nsmallest("key", 3, "last"))
    assert_equal(len(got), 3, "three rows kept")
    assert_equal(got[0], 7, "the last of the ones")
    assert_equal(got[1], 4, "then the middle one")
    assert_equal(got[2], 1, "then the first")


def test_asking_for_more_rows_than_there_are_gives_all_of_them() raises:
    # The interesting half of this is that the slot table is `n` wide, so a
    # number written far past the end of the frame has to be cut down before
    # it is spent rather than after.
    var got = _rows_of(_ranked().nlargest("key", 1000))
    assert_equal(len(got), 8, "every row came back")
    assert_equal(got[0], 0, "still sorted best first")
    assert_equal(got[7], 7, "and the last of the smallest is last")


def test_asking_for_no_rows_gives_an_empty_frame() raises:
    var got = _ranked().nlargest("key", 0)
    assert_equal(len(got), 0, "nothing kept")
    assert_equal(got.width(), 2, "the columns are still there")


def _with_nulls() raises -> DataFrame:
    """A frame whose ranking column is missing half its values.

    Returns:
        Four rows, two of them present.
    """
    var row = Array[DType.int64](4)
    var value = Array[DType.float64](4)
    for i in range(4):
        row.set_valid(i, Int64(i))
    value.set_valid(0, 1.0)
    value.set_null(1)
    value.set_valid(2, 9.0)
    value.set_null(3)
    var series = List[Series]()
    series.append(Series("row", row^))
    series.append(Series("value", value^))
    return DataFrame.from_series(series^)


def test_a_missing_value_ranks_last_rather_than_dropping_out() raises:
    # pandas answers rows 2, 0, 1 here. The present values come first in order
    # and then the answer is padded with the missing ones, because a null sorts
    # to the end of a ranking and is still a row. The grouped spelling does not
    # do this, and the difference is on purpose.
    var got = _rows_of(_with_nulls().nlargest("value", 3))
    assert_equal(len(got), 3, "the answer is as tall as it was asked for")
    assert_equal(got[0], 2, "the larger present value")
    assert_equal(got[1], 0, "then the smaller one")
    assert_equal(got[2], 1, "then the first of the missing ones")


def test_the_padding_walks_forwards_under_both_tie_rules() raises:
    # pandas pads in row order whichever way round the ranking was read, so the
    # missing rows do not reverse with everything else.
    var got = _rows_of(_with_nulls().nlargest("value", 4, "last"))
    assert_equal(len(got), 4, "every row came back")
    assert_equal(got[0], 2, "the larger present value still leads")
    assert_equal(got[1], 0, "then the smaller one")
    assert_equal(got[2], 1, "then the earlier missing row")
    assert_equal(got[3], 3, "then the later one")


def test_asking_for_fewer_rows_than_are_present_pads_nothing() raises:
    var got = _rows_of(_with_nulls().nsmallest("value", 2))
    assert_equal(len(got), 2, "two rows kept")
    assert_equal(got[0], 0, "the smaller present value")
    assert_equal(got[1], 2, "then the larger one")


def test_the_kept_rows_carry_the_labels_they_had() raises:
    var got = _ranked().nsmallest("key", 3)
    assert_equal(len(got.index), 3, "one label per kept row")


def test_the_frame_spelling_refuses_a_rule_it_does_not_know() raises:
    with assert_raises(contains="keep must be"):
        _ = _ranked().nlargest("key", 2, "all")


def test_the_frame_spelling_refuses_a_column_that_is_not_there() raises:
    with assert_raises():
        _ = _ranked().nlargest("nope", 2)


def test_a_column_of_words_is_refused_rather_than_read_as_bytes() raises:
    # A string column is laid out as uint8, so a check written against the
    # physical dtype passes it through and then ranks the bytes of the values
    # buffer as if they were the column, which answers rows rather than
    # failing. The check is written against the logical type for that reason.
    var value = Array[DType.int64](3)
    for i in range(3):
        value.set_valid(i, Int64(i))
    var series = List[Series]()
    series.append(Series("row", value^))
    series.append(
        Series("word", AnyArray(strings_from_list(["pear", "apple", "fig"])))
    )
    var df = DataFrame.from_series(series^)

    with assert_raises(contains="numeric"):
        _ = df.nlargest("word", 2)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
