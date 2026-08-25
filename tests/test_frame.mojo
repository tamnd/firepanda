"""Tests for `Series` and `DataFrame`.

The frame layer has almost no arithmetic in it, so these tests are not about
numbers being right. They are about three other things.

The first is that the invariants hold. A frame's columns are the same length and
its names are distinct, and every constructor that can be reached from outside
this package checks both. Half the tests here are the error paths, because an
invariant that is only enforced on the happy path is not an invariant.

The second is that the operations agree with the kernels they forward to. There
is no scalar twin at this layer and there does not need to be one: the twin lives
under the kernel, and what can go wrong up here is the frame forwarding to the
wrong column, or forgetting to carry the schema across, or getting the row count
out of step with the columns. So the assertions are mostly about which column
came back and what the schema says about it.

The third is the erased dispatch. `take_any`, `filter_any` and `cast_any` resolve
a dtype at runtime and there are twelve of them, so the round trip is checked on
one of each rather than on int64 alone, which is where a `comptime for` that
matched the wrong candidate would show up.
"""

from std.testing import TestSuite, assert_equal, assert_false, assert_true

from firepanda.array.any import AnyArray
from firepanda.array.array import Array, from_list
from firepanda.dtype.logical import LogicalType
from firepanda.dtype.lists import ALL
from firepanda.frame.frame import DataFrame
from firepanda.frame.series import Series
from firepanda.testing.rng import Rng


def sample_frame() raises -> DataFrame:
    """Builds the frame most of these tests work on.

    Six rows, three columns, one null in each of the last two, and `key` chosen
    so that sorting on it is not the identity and is not the reverse either.
    """
    var key = Array[DType.int64](6)
    var score = Array[DType.float64](6)
    var flag = Array[DType.bool](6)

    var keys = [Int64(3), Int64(1), Int64(3), Int64(0), Int64(2), Int64(1)]
    for i in range(6):
        key.set_valid(i, keys[i])
        score.set_valid(i, Float64(i) * 1.5)
        flag.set_valid(i, i % 2 == 0)
    score.set_null(4)
    flag.set_null(1)

    var columns = List[Series]()
    columns.append(Series("key", key^))
    columns.append(Series("score", score^))
    columns.append(Series("flag", flag^))
    return DataFrame.from_series(columns^)


def int_series(name: String, values: List[Int64]) raises -> Series:
    """Builds a fully valid int64 series."""
    var col = Array[DType.int64](len(values))
    for i in range(len(values)):
        col.set_valid(i, values[i])
    return Series(name, col^)


def keys_of(df: DataFrame) raises -> List[Int64]:
    """Reads the `key` column back out as a plain list."""
    var col = df.column("key").as_typed[DType.int64]()
    var out = List[Int64](capacity=len(col))
    for i in range(len(col)):
        out.append(col[i])
    return out^


def test_a_frame_reports_its_shape() raises:
    var df = sample_frame()
    assert_equal(len(df), 6, "rows")
    assert_equal(df.width(), 3, "columns")
    assert_equal(df.shape()[0], 6, "shape rows")
    assert_equal(df.shape()[1], 3, "shape columns")


def test_an_empty_frame_is_zero_by_zero() raises:
    var df = DataFrame()
    assert_equal(len(df), 0, "rows")
    assert_equal(df.width(), 0, "columns")

    var none = List[Series]()
    var built = DataFrame.from_series(none^)
    assert_equal(len(built), 0, "rows from an empty list")
    assert_equal(built.width(), 0, "columns from an empty list")


def test_the_schema_matches_the_columns() raises:
    var df = sample_frame()
    assert_equal(len(df.schema), 3, "schema length")
    assert_equal(df.schema[0].name, "key", "first name")
    assert_true(df.schema[1].dtype == LogicalType.FLOAT64, "second dtype")
    assert_true(df.schema[2].dtype == LogicalType.BOOL, "third dtype")
    assert_equal(df.names()[2], "flag", "names in order")


def test_columns_of_different_lengths_are_rejected() raises:
    var columns = List[Series]()
    columns.append(int_series("a", [Int64(1), Int64(2)]))
    columns.append(int_series("b", [Int64(1)]))

    var raised = False
    try:
        _ = DataFrame.from_series(columns^)
    except:
        raised = True
    assert_true(raised, "unequal column lengths should raise")


def test_duplicate_column_names_are_rejected() raises:
    var columns = List[Series]()
    columns.append(int_series("a", [Int64(1)]))
    columns.append(int_series("a", [Int64(2)]))

    var raised = False
    try:
        _ = DataFrame.from_series(columns^)
    except:
        raised = True
    assert_true(raised, "a repeated name should raise")


def test_a_missing_column_name_raises() raises:
    var df = sample_frame()
    assert_true(df.has("key"), "key exists")
    assert_false(df.has("nope"), "nope does not")

    var raised = False
    try:
        _ = df.index_of("nope")
    except:
        raised = True
    assert_true(raised, "looking up a missing name should raise")


def test_a_column_can_be_borrowed_by_position() raises:
    var df = sample_frame()
    assert_true(df[0].dtype() == DType.int64, "first column dtype")
    assert_equal(len(df[1]), 6, "second column length")
    assert_equal(df[1].null_count(), 1, "second column nulls")


def test_a_column_can_be_copied_by_name() raises:
    var df = sample_frame()
    var score = df.column("score")
    assert_equal(score.name, "score", "name carried across")
    assert_equal(len(score), 6, "length")
    assert_equal(score.null_count(), 1, "nulls")
    assert_false(score.is_valid(4), "the null is where it was put")


def test_select_keeps_the_named_columns_in_order() raises:
    var df = sample_frame()
    var narrowed = df.select(["flag", "key"])
    assert_equal(narrowed.width(), 2, "width")
    assert_equal(len(narrowed), 6, "rows unchanged")
    assert_equal(narrowed.names()[0], "flag", "order follows the argument")
    assert_equal(narrowed.names()[1], "key", "order follows the argument")


def test_select_rejects_a_repeated_name() raises:
    var df = sample_frame()
    var raised = False
    try:
        _ = df.select(["key", "key"])
    except:
        raised = True
    assert_true(raised, "selecting a column twice should raise")


def test_drop_removes_columns_and_keeps_the_rest_in_order() raises:
    var df = sample_frame()
    var narrowed = df.drop(["score"])
    assert_equal(narrowed.width(), 2, "width")
    assert_equal(narrowed.names()[0], "key", "first survivor")
    assert_equal(narrowed.names()[1], "flag", "second survivor")

    var raised = False
    try:
        _ = df.drop(["nope"])
    except:
        raised = True
    assert_true(raised, "dropping a missing column should raise")


def test_rename_changes_one_name_and_nothing_else() raises:
    var df = sample_frame()
    var renamed = df.rename("score", "value")
    assert_equal(renamed.names()[1], "value", "renamed")
    assert_equal(renamed.width(), 3, "width unchanged")
    assert_equal(renamed.column("value").null_count(), 1, "data unchanged")
    assert_equal(df.names()[1], "score", "the original is untouched")

    var raised = False
    try:
        _ = df.rename("key", "flag")
    except:
        raised = True
    assert_true(raised, "renaming onto a taken name should raise")


def test_rename_to_the_same_name_is_allowed() raises:
    var df = sample_frame()
    var renamed = df.rename("key", "key")
    assert_equal(renamed.names()[0], "key", "unchanged")


def test_with_column_replaces_in_place_by_name() raises:
    var df = sample_frame()
    var widened = df.with_column(df.column("key").cast(DType.float64))
    assert_equal(widened.width(), 3, "no column added")
    assert_equal(widened.names()[0], "key", "position kept")
    assert_true(widened[0].dtype() == DType.float64, "column dtype updated")
    assert_true(
        widened.schema[0].dtype == LogicalType.FLOAT64, "schema updated with it"
    )


def test_with_column_appends_a_new_name() raises:
    var df = sample_frame()
    var extra = df.column("key").rename("key_again")
    var wider = df.with_column(extra^)
    assert_equal(wider.width(), 4, "column added")
    assert_equal(wider.names()[3], "key_again", "appended at the end")
    assert_equal(len(wider), 6, "rows unchanged")


def test_with_column_rejects_the_wrong_height() raises:
    var df = sample_frame()
    var raised = False
    try:
        _ = df.with_column(int_series("short", [Int64(1)]))
    except:
        raised = True
    assert_true(raised, "a column of the wrong height should raise")


def test_with_column_on_an_empty_frame_sets_the_height() raises:
    var df = DataFrame()
    var grown = df.with_column(int_series("a", [Int64(1), Int64(2)]))
    assert_equal(len(grown), 2, "height comes from the first column")
    assert_equal(grown.width(), 1, "width")


def test_cast_converts_one_column() raises:
    var df = sample_frame()
    var narrowed = df.cast("key", DType.int8)
    assert_true(narrowed[0].dtype() == DType.int8, "converted")
    assert_true(narrowed[1].dtype() == DType.float64, "the rest untouched")
    assert_equal(len(narrowed), 6, "rows unchanged")

    var values = narrowed.column("key").as_typed[DType.int8]()
    assert_equal(Int(values[0]), 3, "value survived the narrowing")


def test_cast_carries_nulls_across() raises:
    var df = sample_frame()
    var narrowed = df.cast("score", DType.float32)
    assert_equal(narrowed[1].null_count(), 1, "null count")
    assert_false(narrowed[1].is_valid(4), "in the same place")


def test_cast_round_trips_every_dtype() raises:
    # The erased cast dispatches on both ends, so a `comptime for` that matched
    # the wrong candidate would still work for int64 and fail for exactly one
    # other dtype. Walking the list is the only way to see that.
    var source = Array[DType.int32](3)
    for i in range(3):
        source.set_valid(i, Int32(i + 1))
    var df = DataFrame()
    df = df.with_column(Series("a", source^))

    comptime for candidate in ALL:
        var moved = df.cast("a", candidate).cast("a", DType.int32)
        var values = moved.column("a").as_typed[DType.int32]()
        assert_equal(len(values), 3, "length through " + String(candidate))
        if candidate != DType.bool:
            assert_equal(
                Int(values[2]), 3, "value through " + String(candidate)
            )
        else:
            assert_equal(Int(values[2]), 1, "bool flattens to one")


def test_filter_keeps_the_true_rows_of_every_column() raises:
    var df = sample_frame()
    var mask = Array[DType.bool](6)
    for i in range(6):
        mask.set_valid(i, i >= 3)

    var kept = df.filter(mask)
    assert_equal(len(kept), 3, "rows")
    assert_equal(kept.width(), 3, "columns")

    var keys = keys_of(kept)
    assert_equal(Int(keys[0]), 0, "first kept key")
    assert_equal(Int(keys[1]), 2, "second kept key")
    assert_equal(Int(keys[2]), 1, "third kept key")
    assert_equal(kept.column("score").null_count(), 1, "null came across")


def test_filter_drops_the_rows_the_mask_is_null_on() raises:
    var df = sample_frame()
    var mask = Array[DType.bool](6)
    for i in range(6):
        mask.set_valid(i, True)
    mask.set_null(0)
    mask.set_null(5)

    var kept = df.filter(mask)
    assert_equal(len(kept), 4, "a null mask row is dropped, not kept")


def test_filter_rejects_the_wrong_mask_length() raises:
    var df = sample_frame()
    var raised = False
    try:
        _ = df.filter(Array[DType.bool](3))
    except:
        raised = True
    assert_true(raised, "a short mask should raise")


def test_take_gathers_rows_and_a_negative_index_is_a_null() raises:
    var df = sample_frame()
    var taken = df.take([5, -1, 0])
    assert_equal(len(taken), 3, "rows")
    assert_equal(taken.width(), 3, "columns")

    var keys = keys_of(taken)
    assert_equal(Int(keys[0]), 1, "gathered from the end")
    assert_equal(Int(keys[2]), 3, "gathered from the front")
    assert_false(taken[0].is_valid(1), "a negative index produces a null")
    assert_false(taken[1].is_valid(1), "in every column")
    assert_false(taken[2].is_valid(1), "including the bool one")


def test_slice_head_and_tail_agree_with_each_other() raises:
    var df = sample_frame()
    assert_equal(len(df.head(2)), 2, "head length")
    assert_equal(len(df.tail(2)), 2, "tail length")
    assert_equal(Int(keys_of(df.head(2))[1]), 1, "head content")
    assert_equal(Int(keys_of(df.tail(2))[0]), 2, "tail content")
    assert_equal(Int(keys_of(df.slice(2, 4))[0]), 3, "slice content")
    assert_equal(len(df.slice(3, 3)), 0, "an empty slice is legal")


def test_head_and_tail_clamp_rather_than_raise() raises:
    var df = sample_frame()
    assert_equal(len(df.head(100)), 6, "head past the end")
    assert_equal(len(df.tail(100)), 6, "tail past the end")
    assert_equal(len(df.head(-1)), 0, "a negative count gives nothing")
    assert_equal(len(df.tail(-1)), 0, "in both directions")


def test_slice_out_of_range_raises() raises:
    var df = sample_frame()
    var raised = False
    try:
        _ = df.slice(0, 7)
    except:
        raised = True
    assert_true(raised, "slicing past the end should raise")

    raised = False
    try:
        _ = df.slice(4, 2)
    except:
        raised = True
    assert_true(raised, "a reversed slice should raise")


def test_slice_carries_validity_across_a_word_boundary() raises:
    # `AnyArray.slice` moves bytes with no dispatch and hands the bitmap to
    # `Bitmap.slice`, which has a separate path for an unaligned start. A frame
    # is where the two meet.
    var col = Array[DType.int32](200)
    for i in range(200):
        col.set_valid(i, Int32(i))
    col.set_null(70)

    var df = DataFrame()
    df = df.with_column(Series("a", col^))
    var window = df.slice(65, 130)
    assert_equal(len(window), 65, "length")
    assert_equal(window[0].null_count(), 1, "one null in the window")
    assert_false(window[0].is_valid(5), "at the shifted position")

    var values = window.column("a").as_typed[DType.int32]()
    assert_equal(Int(values[0]), 65, "first value")
    assert_equal(Int(values[64]), 129, "last value")


def test_sort_by_one_key_is_stable() raises:
    var df = sample_frame()
    var ordered = df.sort_by("key")
    var keys = keys_of(ordered)
    assert_equal(Int(keys[0]), 0, "smallest first")
    assert_equal(Int(keys[1]), 1, "then the ones")
    assert_equal(Int(keys[2]), 1, "both of them")
    assert_equal(Int(keys[5]), 3, "largest last")

    # Rows 1 and 5 both hold key 1, and row 1 came first, so a stable sort has to
    # put its score first too.
    var scores = ordered.column("score").as_typed[DType.float64]()
    assert_equal(scores[1], 1.5, "the earlier row of the tie stays earlier")
    assert_equal(scores[2], 7.5, "and the later one stays later")


def test_sort_by_descending_reverses_without_breaking_stability() raises:
    var df = sample_frame()
    var ordered = df.sort_by("key", descending=True)
    var keys = keys_of(ordered)
    assert_equal(Int(keys[0]), 3, "largest first")
    assert_equal(Int(keys[5]), 0, "smallest last")

    var scores = ordered.column("score").as_typed[DType.float64]()
    assert_equal(scores[0], 0.0, "row 0 before row 2 in the tie")
    assert_equal(scores[1], 3.0, "row 2 after it")


def test_sort_places_nulls_at_the_chosen_end() raises:
    var df = sample_frame()
    var last = df.sort_by("score")
    assert_false(last[1].is_valid(5), "nulls last by default")

    var first = df.sort_by("score", nulls_first=True)
    assert_false(first[1].is_valid(0), "nulls first when asked")
    assert_true(first[1].is_valid(1), "and only the nulls")


def test_sort_on_two_keys_makes_the_first_dominant() raises:
    var df = sample_frame()
    var ordered = df.sort_values(
        ["key", "score"], [False, True], [False, False]
    )
    var keys = keys_of(ordered)
    var scores = ordered.column("score").as_typed[DType.float64]()

    assert_equal(Int(keys[0]), 0, "key still dominates")
    assert_equal(Int(keys[1]), 1, "the ones are together")
    assert_equal(Int(keys[2]), 1, "both of them")
    assert_equal(scores[1], 7.5, "and descending score orders them")
    assert_equal(scores[2], 1.5, "the other way round from row order")


def test_sort_carries_every_column_by_the_same_permutation() raises:
    var df = sample_frame()
    var ordered = df.sort_by("key")
    assert_equal(ordered.width(), 3, "width")
    assert_equal(ordered.column("score").null_count(), 1, "score null kept")
    assert_equal(ordered.column("flag").null_count(), 1, "flag null kept")

    # Row 4 held key 2, a null score and flag true. Wherever it landed, all three
    # have to have landed together.
    var keys = keys_of(ordered)
    var at = -1
    for i in range(len(keys)):
        if Int(keys[i]) == 2:
            at = i
    assert_true(at >= 0, "key 2 is somewhere")
    assert_false(ordered[1].is_valid(at), "its null score came with it")
    assert_true(ordered[2].is_valid(at), "and its flag did too")


def test_sort_rejects_a_bad_key_list() raises:
    var df = sample_frame()
    var raised = False
    try:
        _ = df.sort_values([], [], [])
    except:
        raised = True
    assert_true(raised, "no keys should raise")

    raised = False
    try:
        _ = df.sort_values(["key"], [False, True], [False])
    except:
        raised = True
    assert_true(raised, "mismatched flag lists should raise")

    raised = False
    try:
        _ = df.sort_values(["nope"], [False], [False])
    except:
        raised = True
    assert_true(raised, "a missing key column should raise")


def test_sort_agrees_with_the_kernel_on_a_random_column() raises:
    # The frame borrows its key columns out of itself rather than copying them
    # into a list, which is the one place the frame sort is not simply calling
    # `argsort_multi`. This checks the two still agree.
    var rng = Rng(0x5EED_1234)
    var n = 900
    var col = Array[DType.int64](n)
    for i in range(n):
        if rng.next_below(8) == 0:
            col.set_null(i)
        else:
            col.set_valid(i, Int64(rng.next_range(-500, 500)))

    var df = DataFrame()
    df = df.with_column(Series("v", col^))
    var order = df.argsort(["v"], [False], [False])
    var direct = df.column("v").argsort()
    assert_equal(len(order), n, "permutation length")
    for i in range(n):
        assert_equal(
            Int(order[i]),
            Int(direct[i]),
            "frame and series agree at " + String(i),
        )


def test_a_series_reports_itself() raises:
    var s = int_series("a", [Int64(4), Int64(2)])
    assert_equal(len(s), 2, "length")
    assert_true(s.dtype() == DType.int64, "dtype")
    assert_true(s.logical() == LogicalType.INT64, "logical type")
    assert_equal(s.null_count(), 0, "nulls")
    assert_equal(s.rename("b").name, "b", "rename")


def test_a_series_slices_takes_and_filters() raises:
    var s = int_series("a", [Int64(0), Int64(1), Int64(2), Int64(3)])
    assert_equal(len(s.head(2)), 2, "head")
    assert_equal(len(s.tail(3)), 3, "tail")
    assert_equal(len(s.slice(1, 3)), 2, "slice")
    assert_equal(Int(s.slice(1, 3).as_typed[DType.int64]()[0]), 1, "content")

    var taken = s.take([3, -1])
    assert_equal(len(taken), 2, "take")
    assert_false(taken.is_valid(1), "negative index is null")

    var mask = Array[DType.bool](4)
    for i in range(4):
        mask.set_valid(i, i != 2)
    assert_equal(len(s.filter(mask)), 3, "filter")

    var raised = False
    try:
        _ = s.filter(Array[DType.bool](2))
    except:
        raised = True
    assert_true(raised, "a short mask should raise")


def test_a_series_sorts_itself() raises:
    var col = Array[DType.int32](5)
    for i in range(5):
        col.set_valid(i, Int32(4 - i))
    col.set_null(0)
    var s = Series("a", col^)

    var ordered = s.sort_values()
    assert_equal(ordered.name, "a", "name kept")
    assert_false(ordered.is_valid(4), "null went last")
    var values = ordered.as_typed[DType.int32]()
    assert_equal(Int(values[0]), 0, "smallest first")
    assert_equal(Int(values[3]), 3, "then up to the largest non null")


def test_a_series_round_trips_through_every_dtype() raises:
    # `take_any` and `filter_any` each walk `ALL`, so this is the same coverage
    # argument as the cast test: one dtype passing proves very little.
    comptime for candidate in ALL:
        var col = Array[candidate](4)
        for i in range(4):
            col.set_valid(i, Scalar[candidate](1))
        col.set_null(2)
        var s = Series("a", col^)

        var taken = s.take([3, 2, 0])
        assert_equal(len(taken), 3, "take length for " + String(candidate))
        assert_false(taken.is_valid(1), "null gathered as " + String(candidate))

        var mask = Array[DType.bool](4)
        for i in range(4):
            mask.set_valid(i, i < 2)
        var kept = s.filter(mask)
        assert_equal(len(kept), 2, "filter length for " + String(candidate))

        var window = s.slice(1, 4)
        assert_equal(len(window), 3, "slice length for " + String(candidate))
        assert_false(window.is_valid(1), "null slid to " + String(candidate))


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
