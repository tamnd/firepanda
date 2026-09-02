"""Tests for a string column carried through `AnyArray`, `Series` and `DataFrame`.

`StringArray` already had its own tests. What is new is that the frame layer can
hold one, and almost everything worth checking here is about the one hazard that
comes with it: `LogicalType.STRING` has physical dtype uint8, so every dispatch
that matches on `dtype()` will select the uint8 arm for a string column unless it
asks first. What it would then read is the first byte of a 16 byte view, which is
a number, and a plausible looking one. A sum over a column of country codes would
have returned a total rather than an error.

So for every operation that reads values there is a test that it either does the
right thing for text or raises, and none of them assert only that the answer is
not a crash. The ones that raise say what is missing, because "unsupported dtype
uint8" for a column of names is not a message anybody can act on.

The other thing checked here is the validity duplication. `AnyArray` keeps the
bitmap in `data` as well as inside the string column, so that `is_null` and the
all-present mask need no string case, and the tests that matter for that are the
ones that go through a row-motion kernel and then ask about nulls on the far side.
"""

from std.testing import (
    TestSuite,
    assert_equal,
    assert_false,
    assert_raises,
    assert_true,
)

from firepanda.array.any import AnyArray
from firepanda.array.array import Array
from firepanda.array.strings import StringBuilder, strings_from_list
from firepanda.dtype.logical import LogicalType
from firepanda.frame.frame import DataFrame
from firepanda.frame.series import Series
from firepanda.kernel.group import AggKind, aggregate_group_any


def long_text(seed: String) -> String:
    """Builds a string too long to live inside its own view.

    Twelve bytes is the boundary, so everything here is well past it and the
    payload is exercised rather than the inline path.

    Args:
        seed: A short string repeated to make the value.

    Returns:
        A string of at least twenty bytes.
    """
    var out = String()
    for _ in range(7):
        out += seed
    return out^


def with_nulls(values: List[String], present: List[Bool]) raises -> Series:
    """Builds a named string column with nulls where asked.

    Args:
        values: The elements. The entry under a false flag is ignored.
        present: One flag per element.

    Returns:
        A series called `s`.
    """
    var builder = StringBuilder(capacity=len(values))
    for i in range(len(values)):
        if present[i]:
            builder.append(values[i].as_bytes())
        else:
            builder.append_null()
    return Series("s", builder^.finish())


def test_a_series_of_text_reports_itself_as_text() raises:
    var s = Series("city", strings_from_list(["oslo", "lima"]))
    assert_true(s.is_string())
    assert_equal(len(s), 2)
    assert_equal(String(s.logical()), "string")
    assert_equal(s.text(0), "oslo")
    assert_equal(s.text(1), "lima")


def test_reading_text_as_bytes_is_refused() raises:
    """The hazard this whole file is about, at its source.

    uint8 is the physical dtype of a string column, so without the check in
    `check_dtype` this hands back a column over the views buffer whose first
    value is 111, the letter o.
    """
    var s = Series("city", strings_from_list(["oslo"]))
    with assert_raises(contains="no fixed width values"):
        _ = s.as_typed[DType.uint8]()


def test_text_prints_its_values_rather_than_its_type() raises:
    var s = Series("city", strings_from_list(["oslo", "lima"]))
    var out = String(s)
    assert_true("oslo" in out, out)
    assert_true("lima" in out, out)
    assert_true("dtype: string" in out, out)


def test_a_null_prints_as_na_and_not_as_empty() raises:
    var s = with_nulls(["a", "b"], [True, False])
    var out = String(s)
    assert_true("<NA>" in out, out)


def test_take_gathers_text_and_a_negative_index_is_a_null() raises:
    var s = Series("s", strings_from_list(["ab", long_text("mno")]))
    var taken = s.take([1, -1, 0, 1])
    assert_equal(len(taken), 4)
    assert_equal(taken.text(0), long_text("mno"))
    assert_false(taken.is_valid(1))
    assert_equal(taken.text(2), "ab")
    assert_equal(taken.text(3), long_text("mno"))
    assert_equal(taken.null_count(), 1)


def test_a_text_take_past_the_split_gathers_only_short_values() raises:
    # Past `PARALLEL_TAKE_ROWS` the gather runs on every core, and a column with
    # nothing in its payload takes the arm that skips the counting pass and
    # copies the sixteen bytes of the view straight across. This is what a group
    # by's key gather does, since a label fits inside its own view. The length is
    # one past a multiple of sixty four so the last worker is left holding a
    # partial validity word it has to store on the way out.
    var builder = StringBuilder(capacity=4096)
    for i in range(4096):
        builder.append(String("id", i).as_bytes())
    var col = builder^.finish()
    var source = Series("s", col^)

    var picks = List[Int](capacity=65_601)
    for i in range(65_601):
        picks.append((i * 4093) % 4096)
    picks[64] = -1
    picks[65] = -1
    picks[65_600] = -1

    var taken = source.take(picks)
    assert_equal(len(taken), len(picks))

    var wrong = -1
    for i in range(len(picks)):
        var at = picks[i]
        if at < 0:
            if taken.is_valid(i):
                wrong = i
                break
        elif taken.text(i) != String("id", at):
            wrong = i
            break
    assert_equal(wrong, -1, String("a gathered value is wrong at row ", wrong))
    assert_equal(taken.null_count(), 3)


def test_a_text_take_past_the_split_carries_the_payload_across() raises:
    # The other arm. Every third value is too long to live in its view, so the
    # counting pass runs, each worker is handed a base in the output payload, and
    # a value's bytes end up somewhere different from where they started. A
    # worker that used the source offset rather than its own cursor would still
    # pass on the short values and fail here.
    var builder = StringBuilder(capacity=3000)
    for i in range(3000):
        if i % 3 == 0:
            builder.append(String("value-that-is-long-", i).as_bytes())
        else:
            builder.append(String("s", i).as_bytes())
    var col = builder^.finish()
    var source = Series("s", col^)

    var picks = List[Int](capacity=70_001)
    for i in range(70_001):
        picks.append((i * 2999) % 3000)
    picks[128] = -1

    var taken = source.take(picks)
    assert_equal(len(taken), len(picks))

    var wrong = -1
    for i in range(len(picks)):
        var at = picks[i]
        if at < 0:
            if taken.is_valid(i):
                wrong = i
                break
            continue
        var want = String("s", at)
        if at % 3 == 0:
            want = String("value-that-is-long-", at)
        if taken.text(i) != want:
            wrong = i
            break
    assert_equal(wrong, -1, String("a gathered value is wrong at row ", wrong))
    assert_equal(taken.null_count(), 1)


def test_take_past_the_end_is_an_error_rather_than_a_null() raises:
    var s = Series("s", strings_from_list(["ab"]))
    with assert_raises(contains="outside a column"):
        _ = s.take([0, 4])


def test_filter_keeps_the_long_elements_intact() raises:
    var s = Series(
        "s",
        strings_from_list(
            [long_text("abc"), "short", long_text("xyz"), "tiny"]
        ),
    )
    var mask = Array[DType.bool](4)
    var bits = mask.unsafe_ptr()
    bits.unsafe_offset(0).unsafe_write(True)
    bits.unsafe_offset(1).unsafe_write(False)
    bits.unsafe_offset(2).unsafe_write(True)
    bits.unsafe_offset(3).unsafe_write(False)
    var kept = s.filter(mask)
    assert_equal(len(kept), 2)
    assert_equal(kept.text(0), long_text("abc"))
    assert_equal(kept.text(1), long_text("xyz"))


def test_a_null_in_the_mask_drops_the_row() raises:
    var s = Series("s", strings_from_list(["a", "b"]))
    var mask = Array[DType.bool](2)
    var bits = mask.unsafe_ptr()
    bits.unsafe_offset(0).unsafe_write(True)
    bits.unsafe_offset(1).unsafe_write(True)
    mask.data.validity.set(1, False)
    var kept = s.filter(mask)
    assert_equal(len(kept), 1)
    assert_equal(kept.text(0), "a")


def test_slice_head_and_tail_cut_text() raises:
    var s = Series("s", strings_from_list(["a", long_text("bb"), "c", "d"]))
    assert_equal(s.slice(1, 3).text(0), long_text("bb"))
    assert_equal(s.head(2).text(1), long_text("bb"))
    assert_equal(s.tail(1).text(0), "d")
    assert_equal(len(s.head(0)), 0)


def test_nulls_survive_a_gather() raises:
    """The validity is in two places and this is where they would disagree."""
    var s = with_nulls(["a", "b", "c"], [True, False, True])
    var taken = s.take([2, 1, 0])
    assert_equal(taken.null_count(), 1)
    assert_false(taken.is_valid(1))
    var mask = taken.is_null()
    assert_false(Bool(mask.unsafe_ptr().unsafe_offset(0).unsafe_load()))
    assert_true(Bool(mask.unsafe_ptr().unsafe_offset(1).unsafe_load()))


def test_drop_nulls_on_text() raises:
    var s = with_nulls(["a", "b", "c"], [True, False, True])
    var kept = s.drop_nulls()
    assert_equal(len(kept), 2)
    assert_equal(kept.text(0), "a")
    assert_equal(kept.text(1), "c")


def test_fill_null_from_a_single_row() raises:
    var s = with_nulls([long_text("q"), "b"], [False, True])
    var filled = s.fill_null(Series("f", strings_from_list(["missing"])))
    assert_equal(filled.null_count(), 0)
    assert_equal(filled.text(0), "missing")
    assert_equal(filled.text(1), "b")


def test_fill_forward_and_backward_carry_text() raises:
    var s = with_nulls(
        ["a", "x", long_text("z"), "y"], [True, False, True, False]
    )
    var forward = s.fill_forward()
    assert_equal(forward.text(1), "a")
    assert_equal(forward.text(3), long_text("z"))
    var backward = s.fill_backward()
    assert_equal(backward.text(1), long_text("z"))
    assert_false(backward.is_valid(3))


def test_fill_forward_honours_the_limit() raises:
    var s = with_nulls(["a", "p", "q"], [True, False, False])
    var filled = s.fill_forward(limit=1)
    assert_equal(filled.text(1), "a")
    assert_false(filled.is_valid(2))


def test_concat_rebases_the_payload_of_the_second_part() raises:
    """A long element in the second part is the one that catches a bad offset.

    Stacking the payload blocks and leaving the views alone would leave every
    long element of the second part reading from the first part's bytes, and the
    result would be the same length and the wrong text.
    """
    from firepanda.kernel.concat import concat_two_any

    var left = AnyArray(strings_from_list([long_text("aa"), "b"]))
    var right = AnyArray(strings_from_list(["c", long_text("dd")]))
    var joined = Series("s", concat_two_any(left, right))
    assert_equal(len(joined), 4)
    assert_equal(joined.text(0), long_text("aa"))
    assert_equal(joined.text(1), "b")
    assert_equal(joined.text(2), "c")
    assert_equal(joined.text(3), long_text("dd"))


def test_concat_of_text_and_bytes_is_refused() raises:
    """Both are uint8 physically, so the dtype check alone lets this through."""
    from firepanda.kernel.concat import concat_two_any

    var text = AnyArray(strings_from_list(["a"]))
    var bytes = AnyArray(Array[DType.uint8](1))
    with assert_raises(contains="same dtype"):
        _ = concat_two_any(text, bytes)


def test_casting_text_reads_it_rather_than_reading_a_view() raises:
    """A string column is physically uint8, so the numeric arm would have found
    it and converted the first byte of every sixteen byte view. It goes to the
    parser instead, and 123 comes back as the number and not as the byte 49."""
    var s = Series("s", strings_from_list(["123"]))
    var out = s.cast(DType.int64)
    assert_equal(
        out.values.unsafe_ptr[DType.int64]().unsafe_offset(0).unsafe_load(), 123
    )


def test_sorting_text_orders_it() raises:
    var s = Series("s", strings_from_list(["b", "a", "c"]))
    var sorted = s.sort_values()
    assert_equal(sorted.text(0), "a")
    assert_equal(sorted.text(1), "b")
    assert_equal(sorted.text(2), "c")


def test_summing_text_says_what_is_missing() raises:
    """Text has six reductions that mean something and seven that do not. The
    seven are covered in `test_text_agg.mojo`; this checks that the erased entry
    point still refuses rather than reading a view as a number."""
    var column = AnyArray(strings_from_list(["a", "b"]))
    var codes = Array[DType.uint32](2)
    with assert_raises(contains="sum is not defined for a string column"):
        _ = aggregate_group_any(column, AggKind.SUM, codes, 1)


def test_a_frame_holds_text_beside_numbers() raises:
    var names = Series("name", strings_from_list(["oslo", "lima", "cairo"]))
    var counts = Array[DType.int64](3)
    var values = counts.unsafe_ptr()
    values.unsafe_offset(0).unsafe_write(1)
    values.unsafe_offset(1).unsafe_write(2)
    values.unsafe_offset(2).unsafe_write(3)
    var frame = DataFrame.from_series([names^, Series("n", counts^)])

    assert_equal(frame.shape()[0], 3)
    assert_equal(frame.shape()[1], 2)
    assert_true("name: string" in frame.describe(), frame.describe())

    var rendered = String(frame)
    assert_true("cairo" in rendered, rendered)

    var taken = frame.take([2, 0])
    assert_equal(taken.column("name").text(0), "cairo")
    assert_equal(
        taken.column("n").as_typed[DType.int64]().unsafe_ptr().unsafe_load(), 3
    )


def test_a_frame_of_text_slices_both_columns_together() raises:
    var a = Series("a", strings_from_list(["p", long_text("q"), "r"]))
    var b = Series("b", strings_from_list(["1", "2", "3"]))
    var frame = DataFrame.from_series([a^, b^])
    var cut = frame.slice(1, 3)
    assert_equal(len(cut), 2)
    assert_equal(cut.column("a").text(0), long_text("q"))
    assert_equal(cut.column("b").text(0), "2")


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
