"""Tests for the null-handling kernels and the frame methods over them.

Four operations, and the tests split along the same line the implementation
does. `is_null` and `is_not_null` never read a value, so what can go wrong there
is the word at a time expansion getting a boundary wrong, and the tests put nulls
either side of row sixty four to catch it. `coalesce` and the two fills do read
values, so what can go wrong is the null-is-zero invariant leaking: a filled row
holding the right bits but still flagged missing, or a value copied from a null
and reported as present.

Every kernel here has a twin in `firepanda/kernel/scalar.mojo` and the last test
in this file runs the two against each other over a random column, which is the
same arrangement `tests/test_kernel.mojo` uses. The fuzz harness does it at
volume; this does it once so a broken twin fails a fast test rather than a slow
one.

The order-dependence of the fills is worth stating in a test rather than only in
a docstring, so there is one that fills a column forward and backward and asserts
the two answers differ. That is the property, not a coincidence.
"""

from std.math import isnan, nan
from std.math import isnan, nan
from std.testing import TestSuite, assert_equal, assert_false, assert_true

from firepanda.array.any import AnyArray, borrow_columns
from firepanda.array.array import Array, from_list
from firepanda.dtype.lists import ALL
from firepanda.frame.frame import DataFrame
from firepanda.frame.series import Series
from firepanda.kernel.group import AggKind
from firepanda.kernel.nulls import (
    all_valid_mask,
    coalesce,
    coalesce_any,
    fill_backward,
    fill_backward_any,
    fill_forward,
    fill_forward_any,
    is_not_null,
    is_null,
    is_null_any,
    missing_count_any,
)
from firepanda.kernel.reduce import reduce_any
from firepanda.kernel.scalar import (
    coalesce_scalar,
    fill_scalar,
    is_null_scalar,
)
from firepanda.testing.rng import Rng


def gapped(
    values: List[Int64], missing: List[Int]
) raises -> Array[DType.int64]:
    """Builds an int64 column with nulls at the given positions.

    Args:
        values: The values, one per row.
        missing: Which rows are null. Their value is ignored.

    Returns:
        The column.
    """
    var col = Array[DType.int64](len(values))
    for i in range(len(values)):
        col.set_valid(i, values[i])
    for m in range(len(missing)):
        col.set_null(missing[m])
    return col^


def one(value: Int64) raises -> Array[DType.int64]:
    """Builds a single row int64 column, which is how a scalar is spelled.

    Args:
        value: The value.

    Returns:
        A column of one row.
    """
    var col = Array[DType.int64](1)
    col.set_valid(0, value)
    return col^


def test_is_null_reports_the_missing_rows() raises:
    var col = gapped([Int64(1), 2, 3, 4], [1, 3])
    var mask = is_null(col)
    assert_equal(len(mask), 4, "mask length")
    assert_false(mask[0], "row 0 is present")
    assert_true(mask[1], "row 1 is missing")
    assert_false(mask[2], "row 2 is present")
    assert_true(mask[3], "row 3 is missing")


def test_is_not_null_is_the_opposite() raises:
    var col = gapped([Int64(1), 2, 3, 4], [1, 3])
    var mask = is_not_null(col)
    for i in range(4):
        assert_equal(mask[i], not is_null(col)[i], "row " + String(i))


def test_the_mask_is_never_null_itself() raises:
    # "Is this row missing" has an answer on the missing rows too, so a mask
    # with nulls in it would be reporting that it did not know.
    var col = gapped([Int64(1), 2, 3], [0, 1, 2])
    var mask = is_null(col)
    assert_equal(mask.null_count(), 0, "mask null count")


def test_is_null_crosses_a_word_boundary() raises:
    # A validity word holds sixty four rows, and the expansion takes an all
    # present or all null word in blocks and a mixed one row by row. This
    # column has one of each kind of word and nulls either side of row 64.
    var values = List[Int64]()
    for i in range(200):
        values.append(Int64(i))
    var col = gapped(values, [0, 63, 64, 65, 127, 128, 199])

    var mask = is_null(col)
    assert_equal(len(mask), 200, "mask length")
    var expected = [0, 63, 64, 65, 127, 128, 199]
    var at = 0
    for i in range(200):
        var want = at < len(expected) and expected[at] == i
        assert_equal(mask[i], want, "row " + String(i))
        if want:
            at += 1


def test_is_null_on_a_column_with_no_nulls() raises:
    var col = gapped([Int64(1), 2, 3], List[Int]())
    var mask = is_null(col)
    for i in range(3):
        assert_false(mask[i], "row " + String(i))


def test_is_null_past_the_split_reports_every_row() raises:
    # The expansion runs on every core and each worker derives its word loop
    # from the rows it was handed, so the boundaries are worth a test of their
    # own. The length is a prime past three morsels, which leaves the last one
    # short, and the nulls are placed so that one run is exactly a word wide and
    # sits at a morsel boundary while another straddles one.
    comptime rows = 393_241
    var col = Array[DType.int64](rows)
    for i in range(rows):
        col.set_valid(i, Int64(i))
    for i in range(131_072, 131_136):
        col.set_null(i)
    for i in range(262_100, 262_180):
        col.set_null(i)
    col.set_null(rows - 1)

    var missing = is_null(col)
    var present = is_not_null(col)
    assert_equal(len(missing), rows, "mask length")
    assert_equal(missing.null_count(), 0, "the mask has no nulls of its own")

    # Counted rather than asserted per row, because building a message for four
    # hundred thousand rows costs more than the kernel does.
    var wrong = 0
    for i in range(rows):
        if missing[i] != (not col.is_valid(i)):
            wrong += 1
        if present[i] != col.is_valid(i):
            wrong += 1
    assert_equal(wrong, 0, "rows the mask disagrees with the bitmap on")


def test_is_null_erased_matches_the_typed_one() raises:
    comptime for candidate in ALL:
        var col = Array[candidate](4)
        for i in range(4):
            col.set_valid(i, Scalar[candidate](1))
        col.set_null(2)
        var erased = AnyArray(Array[candidate](copy=col))
        var mask = is_null_any(erased)
        assert_equal(len(mask), 4, "length for " + String(candidate))
        assert_true(mask[2], "null seen for " + String(candidate))
        assert_false(mask[0], "value seen for " + String(candidate))


def test_coalesce_takes_the_second_where_the_first_is_missing() raises:
    var a = gapped([Int64(1), 2, 3, 4], [1, 2])
    var b = gapped([Int64(10), 20, 30, 40], List[Int]())
    var out = coalesce(a, b)
    assert_equal(out[0], 1, "row 0 kept")
    assert_equal(out[1], 20, "row 1 filled")
    assert_equal(out[2], 30, "row 2 filled")
    assert_equal(out[3], 4, "row 3 kept")
    assert_equal(out.null_count(), 0, "nothing left missing")


def test_coalesce_leaves_a_row_null_when_both_are() raises:
    var a = gapped([Int64(1), 2], [0, 1])
    var b = gapped([Int64(10), 20], [1])
    var out = coalesce(a, b)
    assert_equal(out[0], 10, "row 0 filled")
    assert_false(out.is_valid(1), "row 1 still missing")
    assert_equal(out[1], 0, "a null holds a zero")


def test_coalesce_broadcasts_a_single_row() raises:
    var a = gapped([Int64(1), 2, 3], [0, 2])
    var out = coalesce(a, one(99))
    assert_equal(out[0], 99, "row 0 filled")
    assert_equal(out[1], 2, "row 1 kept")
    assert_equal(out[2], 99, "row 2 filled")


def test_coalesce_past_the_split_picks_the_right_side() raises:
    # The pick runs on every core and each worker clears the validity bits of
    # its own rows, so this is as much a test that no worker has trodden on
    # another's word as it is a test of the picking. Every third row of the
    # preferred column is missing and every seventh row of the fallback is,
    # which leaves a row missing from both about once in twenty one and so puts
    # several of them inside every morsel.
    comptime rows = 393_241
    var first = Array[DType.int64](rows)
    var second = Array[DType.int64](rows)
    for i in range(rows):
        first.set_valid(i, Int64(i))
        second.set_valid(i, Int64(-i))
    for i in range(0, rows, 3):
        first.set_null(i)
    for i in range(0, rows, 7):
        second.set_null(i)

    var picked = coalesce(first, second)
    assert_equal(len(picked), rows, "the pick is as tall as the first column")

    var wrong = 0
    for i in range(rows):
        if first.is_valid(i):
            if not picked.is_valid(i) or picked[i] != first[i]:
                wrong += 1
        elif second.is_valid(i):
            if not picked.is_valid(i) or picked[i] != second[i]:
                wrong += 1
        elif picked.is_valid(i):
            wrong += 1
    assert_equal(wrong, 0, "rows the pick got wrong")


def test_coalesce_refuses_a_length_that_is_neither() raises:
    var a = AnyArray(gapped([Int64(1), 2, 3], [0]))
    var b = AnyArray(gapped([Int64(1), 2], List[Int]()))
    var raised = False
    try:
        _ = coalesce_any(a, b)
    except:
        raised = True
    assert_true(raised, "a two row fallback for a three row column")


def test_coalesce_refuses_a_different_dtype() raises:
    var a = AnyArray(gapped([Int64(1), 2], [0]))
    var b = Array[DType.float64](2)
    var raised = False
    try:
        _ = coalesce_any(a, AnyArray(b^))
    except:
        raised = True
    assert_true(raised, "int64 against float64")


def test_fill_forward_carries_the_last_value() raises:
    var col = gapped([Int64(5), 0, 0, 8, 0], [1, 2, 4])
    var out = fill_forward(col)
    assert_equal(out[0], 5, "row 0")
    assert_equal(out[1], 5, "row 1 carried")
    assert_equal(out[2], 5, "row 2 carried")
    assert_equal(out[3], 8, "row 3")
    assert_equal(out[4], 8, "row 4 carried")
    assert_equal(out.null_count(), 0, "nothing left missing")


def test_fill_forward_leaves_the_leading_nulls() raises:
    # There is nothing behind row zero to carry, so it stays missing. pandas
    # does the same and it is the only answer that does not invent a value.
    var col = gapped([Int64(0), 0, 7], [0, 1])
    var out = fill_forward(col)
    assert_false(out.is_valid(0), "row 0 still missing")
    assert_false(out.is_valid(1), "row 1 still missing")
    assert_equal(out[2], 7, "row 2")


def test_fill_backward_carries_the_next_value() raises:
    var col = gapped([Int64(0), 0, 7, 0, 9], [0, 1, 3])
    var out = fill_backward(col)
    assert_equal(out[0], 7, "row 0 carried")
    assert_equal(out[1], 7, "row 1 carried")
    assert_equal(out[2], 7, "row 2")
    assert_equal(out[3], 9, "row 3 carried")
    assert_equal(out[4], 9, "row 4")


def test_fill_backward_leaves_the_trailing_nulls() raises:
    var col = gapped([Int64(7), 0, 0], [1, 2])
    var out = fill_backward(col)
    assert_equal(out[0], 7, "row 0")
    assert_false(out.is_valid(1), "row 1 still missing")
    assert_false(out.is_valid(2), "row 2 still missing")


def test_the_two_directions_disagree() raises:
    # The fills are the only order-dependent operations in the kernel layer, and
    # this is what that means: the same column has two different right answers.
    var col = gapped([Int64(1), 0, 3], [1])
    var forward = fill_forward(col)
    var backward = fill_backward(col)
    assert_equal(forward[1], 1, "forward carried from row 0")
    assert_equal(backward[1], 3, "backward carried from row 2")


def test_a_limit_stops_the_carry() raises:
    var col = gapped([Int64(4), 0, 0, 0], [1, 2, 3])
    var out = fill_forward(col, 2)
    assert_equal(out[1], 4, "one row away")
    assert_equal(out[2], 4, "two rows away")
    assert_false(out.is_valid(3), "three rows away is past the limit")


def test_a_limit_resets_at_each_present_value() raises:
    var col = gapped([Int64(1), 0, 2, 0], [1, 3])
    var out = fill_forward(col, 1)
    assert_equal(out[1], 1, "first gap")
    assert_equal(out[3], 2, "second gap counts from row 2")


def test_a_column_of_nothing_but_nulls_is_unchanged() raises:
    var col = gapped([Int64(0), 0, 0], [0, 1, 2])
    var forward = fill_forward(col)
    var backward = fill_backward(col)
    assert_equal(forward.null_count(), 3, "nothing to carry forward")
    assert_equal(backward.null_count(), 3, "nothing to carry backward")


def test_fill_crosses_a_word_boundary() raises:
    # The all present word shortcut copies a block and takes the carry from
    # whichever end the scan is leaving, which is the part with two ways to be
    # wrong. Rows 63 and 64 sit either side of the seam.
    var values = List[Int64]()
    for i in range(200):
        values.append(Int64(i))
    var col = gapped(values, [64, 65, 128])

    var out = fill_forward(col)
    assert_equal(out[64], 63, "carried across the seam")
    assert_equal(out[65], 63, "carried again")
    assert_equal(out[128], 127, "carried inside the third word")
    assert_equal(out.null_count(), 0, "nothing left missing")

    var back = fill_backward(col)
    assert_equal(back[64], 66, "carried back from row 66")
    assert_equal(back[128], 129, "carried back inside the third word")


def test_fill_erased_matches_the_typed_one() raises:
    comptime for candidate in ALL:
        var col = Array[candidate](4)
        for i in range(4):
            col.set_valid(i, Scalar[candidate](1))
        col.set_null(2)
        var erased = AnyArray(Array[candidate](copy=col))

        var forward = fill_forward_any(erased, 0)
        assert_equal(
            forward.null_count(), 0, "forward filled for " + String(candidate)
        )
        var backward = fill_backward_any(erased, 0)
        assert_equal(
            backward.null_count(), 0, "backward filled for " + String(candidate)
        )


def test_all_valid_mask_is_the_intersection() raises:
    var columns = List[AnyArray]()
    columns.append(AnyArray(gapped([Int64(1), 2, 3, 4], [0])))
    columns.append(AnyArray(gapped([Int64(1), 2, 3, 4], [1])))
    var mask = all_valid_mask(borrow_columns(columns), 4)
    assert_false(mask[0], "row 0 missing in the first")
    assert_false(mask[1], "row 1 missing in the second")
    assert_true(mask[2], "row 2 present in both")
    assert_true(mask[3], "row 3 present in both")


def test_all_valid_mask_of_no_columns_is_all_true() raises:
    var none = List[AnyArray]()
    var mask = all_valid_mask(borrow_columns(none), 3)
    for i in range(3):
        assert_true(mask[i], "row " + String(i))


def test_series_drop_nulls() raises:
    var s = Series("a", gapped([Int64(1), 2, 3, 4], [1, 3]))
    var kept = s.drop_nulls()
    assert_equal(len(kept), 2, "two rows survive")
    assert_equal(kept.as_typed[DType.int64]()[0], 1, "row 0")
    assert_equal(kept.as_typed[DType.int64]()[1], 3, "row 1")
    assert_equal(kept.name, "a", "name kept")


def test_series_fill_null_with_a_scalar() raises:
    var s = Series("a", gapped([Int64(1), 2, 3], [1]))
    var filled = s.fill_null(Series("anything", one(99)))
    assert_equal(filled.name, "a", "the fallback's name is ignored")
    assert_equal(filled.as_typed[DType.int64]()[1], 99, "row 1 filled")
    assert_equal(filled.null_count(), 0, "nothing left missing")


def test_series_fill_directions() raises:
    var s = Series("a", gapped([Int64(1), 0, 3], [1]))
    assert_equal(
        s.fill_forward().as_typed[DType.int64]()[1], 1, "forward on a series"
    )
    assert_equal(
        s.fill_backward().as_typed[DType.int64]()[1], 3, "backward on a series"
    )
    assert_equal(
        s.fill_forward(0).as_typed[DType.int64]()[1],
        1,
        "an explicit zero limit means no limit",
    )


def test_series_is_null_returns_a_usable_mask() raises:
    # The mask comes back as a bare `Array` rather than a `Series` so that it
    # goes straight into `filter`, and this is the round trip that shows why.
    var s = Series("a", gapped([Int64(1), 2, 3], [1]))
    var kept = s.filter(s.is_not_null())
    assert_equal(len(kept), 2, "two rows survive")


def nan_column(
    values: List[Float64], missing: List[Int]
) raises -> Array[DType.float64]:
    """Builds a float64 column, where a NaN in `values` stays a NaN.

    The two ways of being missing are both spellable here on purpose, because
    every test below needs a column that has one of each in it. A position named
    in `missing` gets a cleared bit, and a position holding `nan[float64]()` gets
    a NaN behind a bit that stays set, which is what pandas hands over.

    Args:
        values: The values, one per row.
        missing: Which rows get a cleared validity bit.

    Returns:
        The column.
    """
    var col = Array[DType.float64](len(values))
    for i in range(len(values)):
        col[i] = values[i]
    for i in range(len(missing)):
        col.set_null(missing[i])
    return col^


def test_a_nan_is_missing_and_a_cleared_bit_is_missing() raises:
    # The whole of this change in one assertion. Row 1 is a NaN with its validity
    # bit set, which is how pandas spells missing and how an Arrow file from
    # pandas arrives. Row 3 is a cleared bit, which is how Arrow spells it. Both
    # are missing and rows 0 and 2 are not.
    var col = nan_column(
        [1.0, nan[DType.float64](), 3.0, 0.0, 5.0],
        [3],
    )

    var missing = is_null(col)
    assert_false(missing[0], "a value is not missing")
    assert_true(missing[1], "a NaN is missing")
    assert_false(missing[2], "a value is not missing")
    assert_true(missing[3], "a cleared bit is missing")
    assert_false(missing[4], "a value is not missing")

    var present = is_not_null(col)
    for i in range(5):
        assert_equal(
            present[i], not missing[i], "the two are each other's inverse"
        )


def test_an_int_column_never_reads_its_values() raises:
    # The other half of the rule, and the reason it is keyed on the dtype. An
    # int64 column has no NaN to find, so nothing changed for it, and the value
    # that would be a NaN's bit pattern read as an integer is an ordinary number
    # that is present.
    var col = gapped([Int64(1), 9221120237041090560, 3], [2])

    var missing = is_null(col)
    assert_false(missing[0], "a value is not missing")
    assert_false(missing[1], "a NaN's bits as an integer are a value")
    assert_true(missing[2], "a cleared bit is missing")


def test_a_nan_counts_as_missing_and_a_count_says_so() raises:
    # `Series([1.0, nan]).count()` is 1 in pandas. It was 2 here, because the
    # count was `len` minus a number the bitmap knew and the bitmap does not
    # know about NaN. See #170.
    var col = nan_column([1.0, nan[DType.float64](), 3.0, 0.0], [3])
    assert_equal(missing_count_any(AnyArray(col.copy())), 2, "one of each")

    var counted = reduce_any(AnyArray(col^), AggKind.COUNT)
    assert_equal(counted.as_typed[DType.int64]()[0], 2, "two rows hold a value")


def test_a_count_over_a_float_column_with_no_nans_is_unchanged() raises:
    # The shape that has to stay fast and has to stay right. The scan reads every
    # value, finds nothing, and answers what the bitmap alone would have.
    var col = nan_column([1.0, 2.0, 3.0, 4.0, 5.0], [1, 4])
    assert_equal(missing_count_any(AnyArray(col^)), 2, "the two cleared bits")


def test_a_nan_past_a_word_boundary_is_still_found() raises:
    # The scan asks one word of sixty four rows at a time whether it holds any
    # NaN at all and only takes apart the words that say yes, so a column where
    # the NaN is alone in the second word runs both branches in one call and a
    # boundary error shows up here rather than at ten million rows.
    var values = List[Float64]()
    for _ in range(200):
        values.append(1.0)
    values[63] = nan[DType.float64]()
    values[64] = nan[DType.float64]()
    values[199] = nan[DType.float64]()
    var col = nan_column(values, [0, 128])

    var missing = is_null(col)
    for i in range(200):
        var expected = i == 0 or i == 63 or i == 64 or i == 128 or i == 199
        assert_equal(missing[i], expected, String("row ", i))
    assert_equal(missing_count_any(AnyArray(col^)), 5, "three NaNs, two bits")


def test_a_cleared_row_cannot_contribute_a_nan() raises:
    # The scan reads the values under the cleared bits rather than skipping them,
    # which is safe only because a null holds a zero. `set_null` is what puts the
    # zero there, so this asserts the invariant that makes the shortcut legal.
    var col = nan_column([1.0, nan[DType.float64](), 3.0], [1])
    assert_equal(missing_count_any(AnyArray(col.copy())), 1, "row 1, once")
    assert_false(isnan(col[1]), "a cleared row holds a zero")


def test_series_null_count_is_the_pandas_one() raises:
    # `Series` is the pandas side of the library and counts a NaN. `Array` is the
    # Arrow side and counts bits. Both numbers are right and they are different,
    # which is the thing to be able to point at when somebody asks why.
    var col = nan_column([1.0, nan[DType.float64](), 3.0, 0.0], [3])
    assert_equal(col.null_count(), 1, "the Array counts the cleared bit")

    var s = Series("a", AnyArray(col^))
    assert_equal(s.null_count(), 2, "the Series counts the NaN as well")
    assert_equal(len(s.drop_nulls()), 2, "and drops it")


def test_drop_nulls_drops_a_nan_row() raises:
    # The visible consequence, on a frame as well as on a series. Row 1 is only
    # missing because of a NaN in one column and it goes.
    var columns = List[Series]()
    columns.append(
        Series("a", nan_column([1.0, nan[DType.float64](), 3.0], List[Int]()))
    )
    columns.append(Series("b", gapped([Int64(4), 5, 6], List[Int]())))
    var frame = DataFrame.from_series(columns^)

    var kept = frame.drop_nulls()
    assert_equal(len(kept), 2, "the NaN row is gone")
    assert_equal(kept.column("b").as_typed[DType.int64]()[1], 6, "rows 0 and 2")


def test_frame_drop_nulls_looks_at_every_column() raises:
    var columns = List[Series]()
    columns.append(Series("a", gapped([Int64(1), 2, 3], [0])))
    columns.append(Series("b", gapped([Int64(4), 5, 6], [1])))
    var frame = DataFrame.from_series(columns^)

    var kept = frame.drop_nulls()
    assert_equal(len(kept), 1, "only row 2 is present in both")
    assert_equal(kept.column("a").as_typed[DType.int64]()[0], 3, "row kept")


def test_frame_drop_nulls_narrows_to_a_subset() raises:
    var columns = List[Series]()
    columns.append(Series("a", gapped([Int64(1), 2, 3], [0])))
    columns.append(Series("b", gapped([Int64(4), 5, 6], [1])))
    var frame = DataFrame.from_series(columns^)

    var kept = frame.drop_nulls(["a"])
    assert_equal(len(kept), 2, "b's null does not count")
    assert_false(kept.column("b").is_valid(0), "and it is still there")


def test_frame_drop_nulls_refuses_a_missing_column() raises:
    var columns = List[Series]()
    columns.append(Series("a", gapped([Int64(1), 2], List[Int]())))
    var frame = DataFrame.from_series(columns^)

    var raised = False
    try:
        _ = frame.drop_nulls(["nope"])
    except:
        raised = True
    assert_true(raised, "a subset naming a column that is not there")


def test_frame_fill_null_leaves_the_other_columns_alone() raises:
    var columns = List[Series]()
    columns.append(Series("a", gapped([Int64(1), 2, 3], [1])))
    columns.append(Series("b", gapped([Int64(4), 5, 6], [2])))
    var frame = DataFrame.from_series(columns^)

    var filled = frame.fill_null("a", Series("v", one(0)))
    assert_equal(filled.column("a").null_count(), 0, "a is filled")
    assert_equal(filled.column("b").null_count(), 1, "b is not")
    assert_equal(filled.width(), 2, "the shape is unchanged")
    assert_equal(len(filled), 3, "the height is unchanged")


def test_the_kernels_agree_with_their_twins() raises:
    # The same arrangement as `tests/test_kernel.mojo`: one random column, run
    # through both implementations, compared element by element. The fuzz
    # harness does this at volume and this is here so a broken twin fails fast.
    var rng = Rng(0x5DEECE66D)
    var col = Array[DType.int64](150)
    for i in range(150):
        if rng.next_below(3) == 0:
            col.set_null(i)
        else:
            col.set_valid(i, Int64(rng.next_below(1000)))

    var fallback = Array[DType.int64](150)
    for i in range(150):
        if rng.next_below(4) == 0:
            fallback.set_null(i)
        else:
            fallback.set_valid(i, Int64(rng.next_below(1000)))

    var fast_null = is_null(col)
    var slow_null = is_null_scalar(col)
    for i in range(150):
        assert_equal(fast_null[i], slow_null[i], "is_null row " + String(i))

    var fast_pick = coalesce(col, fallback)
    var slow_pick = coalesce_scalar(col, fallback)
    for i in range(150):
        assert_equal(
            fast_pick.is_valid(i),
            slow_pick.is_valid(i),
            "coalesce validity row " + String(i),
        )
        assert_equal(fast_pick[i], slow_pick[i], "coalesce row " + String(i))

    for limit in [0, 1, 3]:
        var fast_ffill = fill_forward(col, limit)
        var slow_ffill = fill_scalar[forward=True](col, limit)
        var fast_bfill = fill_backward(col, limit)
        var slow_bfill = fill_scalar[forward=False](col, limit)
        for i in range(150):
            assert_equal(
                fast_ffill.is_valid(i),
                slow_ffill.is_valid(i),
                "ffill validity row " + String(i),
            )
            assert_equal(fast_ffill[i], slow_ffill[i], "ffill row " + String(i))
            assert_equal(
                fast_bfill.is_valid(i),
                slow_bfill.is_valid(i),
                "bfill validity row " + String(i),
            )
            assert_equal(fast_bfill[i], slow_bfill[i], "bfill row " + String(i))


def test_is_null_agrees_with_its_twin_on_a_float_column() raises:
    # Three hundred rows on purpose, so the column runs past the fourth bitmap
    # word and both branches of the kernel's word loop are entered many times.
    # A NaN every seventh row and a cleared bit every fifth means most words
    # hold both spellings of missing and neither branch is the only one taken.
    # The twin has no word loop at all, which is what makes the comparison worth
    # anything.
    var rng = Rng(0x2545F4914F6CDD1D)
    var col = Array[DType.float64](300)
    for i in range(300):
        if i % 5 == 0:
            col.set_null(i)
        elif i % 7 == 0:
            col[i] = nan[DType.float64]()
        else:
            col[i] = Float64(Int(rng.next_below(1000)))

    var fast = is_null(col)
    var slow = is_null_scalar(col)
    var missing = 0
    for i in range(300):
        assert_equal(fast[i], slow[i], "is_null row " + String(i))
        if fast[i]:
            missing += 1
    assert_equal(
        missing_count_any(AnyArray(col^)),
        missing,
        "the count and the mask agree on how many there are",
    )


def floats(
    values: List[Float64], missing: List[Int]
) raises -> Array[DType.float64]:
    """Builds a float64 column with nulls at the given positions.

    The float twin of `gapped`, and the reason there are two of them rather than
    one parameterized one is that a float column has two spellings of missing and
    a test that wants a NaN writes it as a value here rather than as a position.

    Args:
        values: The values, one per row. Write a NaN where a NaN is wanted.
        missing: Which rows are null. Their value is ignored.

    Returns:
        The column.
    """
    var col = Array[DType.float64](len(values))
    for i in range(len(values)):
        col.set_valid(i, values[i])
    for m in range(len(missing)):
        col.set_null(missing[m])
    return col^


def test_a_fill_carries_over_a_nan_and_not_from_one() raises:
    # The two halves of the bug in one column. Row 1 is a NaN with its validity
    # bit set, so a fill that only reads the bitmap leaves it standing, and then
    # carries it into row 2, which is a real null. One NaN turning a gap that had
    # a 5 behind it into a run of NaN is the part that does the damage.
    var col = floats([5.0, nan[DType.float64](), 0.0, 8.0], [2])
    var out = fill_forward(col)
    assert_equal(out[0], 5.0, "row 0")
    assert_equal(out[1], 5.0, "the NaN is filled over")
    assert_equal(out[2], 5.0, "the null after it gets the 5 and not the NaN")
    assert_equal(out[3], 8.0, "row 3")
    assert_equal(out.null_count(), 0, "nothing left missing")


def test_a_backward_fill_carries_over_a_nan_and_not_from_one() raises:
    # The same shape read the other way, since the two directions are separate
    # passes over the same corrected word and either one could be the one that
    # was not changed.
    var col = floats([0.0, nan[DType.float64](), 8.0, 3.0], [0])
    var out = fill_backward(col)
    assert_equal(out[0], 8.0, "row 0 carried back past the NaN")
    assert_equal(out[1], 8.0, "the NaN is filled over")
    assert_equal(out[2], 8.0, "row 2")
    assert_equal(out[3], 3.0, "row 3")


def test_a_float_row_that_cannot_be_filled_comes_back_as_a_nan() raises:
    # pandas has no null on a float column, so this is what it gives, and it is
    # the same rule the reductions took in #179: keep the column's dtype and
    # answer with the missing that dtype has.
    var col = floats([0.0, 0.0, 7.0], [0, 1])
    var out = fill_forward(col)
    assert_true(isnan(out[0]), "row 0 has nothing behind it")
    assert_true(isnan(out[1]), "row 1 has nothing behind it either")
    assert_equal(out[2], 7.0, "row 2")
    assert_equal(out.null_count(), 0, "and it is a NaN rather than a null")


def test_an_int_row_that_cannot_be_filled_is_still_a_null() raises:
    # The other half of the dtype rule, stated here rather than left to be
    # inferred from the int tests above, because the point of it is the contrast.
    var col = gapped([Int64(0), 0, 7], [0, 1])
    var out = fill_forward(col)
    assert_equal(out.null_count(), 2, "an int64 column has no NaN to write")


def test_a_float_column_with_nothing_to_carry_fills_to_nothing() raises:
    # Every row missing and no direction to carry from, which is the shape that
    # closed `basics/ffill` on the all null float frame. Both spellings of
    # missing are in it and both come back as the one spelling pandas has.
    var col = floats([nan[DType.float64](), 0.0, nan[DType.float64]()], [1])
    var forward = fill_forward(col)
    var backward = fill_backward(col)
    for i in range(3):
        assert_true(isnan(forward[i]), "forward row " + String(i))
        assert_true(isnan(backward[i]), "backward row " + String(i))
    assert_equal(forward.null_count(), 0, "no nulls forward")
    assert_equal(backward.null_count(), 0, "no nulls backward")


def test_a_nan_counts_against_the_fill_limit() raises:
    # A NaN is missing, so it is a row of the run the limit is counting rather
    # than a present value that resets it. Getting this wrong fills one row too
    # many and only on a float column, which is the kind of difference that
    # survives a long time.
    var col = floats([4.0, nan[DType.float64](), 0.0, 0.0], [2, 3])
    var out = fill_forward(col, 2)
    assert_equal(out[1], 4.0, "one row into the run")
    assert_equal(out[2], 4.0, "two rows into the run")
    assert_true(isnan(out[3]), "three rows is past the limit")


def test_a_float_fill_crosses_a_word_boundary() raises:
    # The float version of the seam test above. The all present shortcut copies a
    # whole block and takes the carry from the end of it, and on a float column
    # that block is only all present after the word has been corrected, so the
    # NaN at row 63 is the one that would be copied through and then carried.
    var values = List[Float64]()
    for i in range(200):
        values.append(Float64(i))
    values[63] = nan[DType.float64]()
    values[64] = nan[DType.float64]()
    var col = floats(values, [128])

    var out = fill_forward(col)
    assert_equal(out[62], 62.0, "the last real value in the first word")
    assert_equal(out[63], 62.0, "the NaN at the end of the first word")
    assert_equal(out[64], 62.0, "carried across the seam and not a NaN")
    assert_equal(out[65], 65.0, "the first real value after it")
    assert_equal(out[128], 127.0, "the null inside the third word")
    assert_equal(out.null_count(), 0, "nothing left missing")


def test_the_fills_agree_with_their_twin_on_a_float_column() raises:
    # Three hundred rows so the column runs past the fourth bitmap word, a NaN
    # every seventh row and a cleared bit every fifth, so most words hold both
    # spellings of missing and the all present shortcut is entered as well as
    # skipped. The twin has no word loop at all, which is what makes this worth
    # running: the kernel corrects a word of sixty four rows before it looks at
    # the block, and the twin walks out from each missing row on its own.
    var col = Array[DType.float64](300)
    for i in range(300):
        if i % 7 == 3:
            col.set_valid(i, nan[DType.float64]())
        elif i % 5 == 0:
            col.set_null(i)
        else:
            col.set_valid(i, Float64(i))

    for limit in [0, 1, 3]:
        var fast_ffill = fill_forward(col, limit)
        var slow_ffill = fill_scalar[forward=True](col, limit)
        var fast_bfill = fill_backward(col, limit)
        var slow_bfill = fill_scalar[forward=False](col, limit)
        for i in range(300):
            assert_equal(
                isnan(fast_ffill[i]),
                isnan(slow_ffill[i]),
                "ffill missing row " + String(i),
            )
            if not isnan(fast_ffill[i]):
                assert_equal(
                    fast_ffill[i], slow_ffill[i], "ffill row " + String(i)
                )
            assert_equal(
                isnan(fast_bfill[i]),
                isnan(slow_bfill[i]),
                "bfill missing row " + String(i),
            )
            if not isnan(fast_bfill[i]):
                assert_equal(
                    fast_bfill[i], slow_bfill[i], "bfill row " + String(i)
                )
        assert_equal(fast_ffill.null_count(), 0, "the kernel writes no null")
        assert_equal(slow_ffill.null_count(), 0, "and neither does the twin")


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
