"""Tests for stacking columns, series and frames.

`concat` has no arithmetic in it, so what these tests are actually checking is
bookkeeping: that the parts land at the right offsets, that the validity comes
across with them, and that the two spellings in the kernel agree.

Two of those need care. The offset a part lands on is the running total of every
part before it, which is almost never a multiple of sixty four, so the validity
bits go across at an offset the bitmap has no fast path for. There is a test with
parts of awkward lengths and nulls in each of them for exactly that. And the
kernel has a two argument spelling next to the list one, added so a join can
concatenate two borrowed columns without deep copying them first, so there is a
test that the two produce the same answer.

The frame level adds the rule that names decide and positions do not, which is
worth a test that stacks two frames whose columns are in different orders and
asserts the values did not get shuffled.
"""

from std.testing import TestSuite, assert_equal, assert_false, assert_true

from firepanda.array.any import AnyArray
from firepanda.array.array import Array, from_list
from firepanda.array.strings import StringArray, StringBuilder
from firepanda.dtype.lists import ALL
from firepanda.frame.concat import concat, concat_series
from firepanda.frame.frame import DataFrame
from firepanda.frame.series import Series
from firepanda.kernel.concat import (
    PARALLEL_ROWS,
    column_ref,
    concat_any,
    concat_refs_any,
    concat_arrays,
    concat_strings,
    concat_two_any,
)
from firepanda.kernel.scalar import concat_scalar
from firepanda.testing.rng import Rng


def gapped(
    values: List[Int64], missing: List[Int]
) raises -> Array[DType.int64]:
    """Builds an int64 column with nulls at the given positions.

    Args:
        values: The values, one per row.
        missing: Which rows are null.

    Returns:
        The column.
    """
    var col = Array[DType.int64](len(values))
    for i in range(len(values)):
        col.set_valid(i, values[i])
    for m in range(len(missing)):
        col.set_null(missing[m])
    return col^


def frame_of(name: String, values: List[Int64]) raises -> DataFrame:
    """Builds a one column frame.

    Args:
        name: The column name.
        values: The values, none of them null.

    Returns:
        The frame.
    """
    var columns = List[Series]()
    columns.append(Series(name, gapped(values, List[Int]())))
    return DataFrame.from_series(columns^)


def test_concat_stacks_in_order() raises:
    var parts = List[Array[DType.int64]]()
    parts.append(gapped([Int64(1), 2], List[Int]()))
    parts.append(gapped([Int64(3)], List[Int]()))
    parts.append(gapped([Int64(4), 5, 6], List[Int]()))

    var out = concat_arrays(parts)
    assert_equal(len(out), 6, "height is the sum")
    for i in range(6):
        assert_equal(out[i], Int64(i + 1), "row " + String(i))


def test_concat_carries_the_nulls() raises:
    var parts = List[Array[DType.int64]]()
    parts.append(gapped([Int64(1), 2], [0]))
    parts.append(gapped([Int64(3), 4], [1]))

    var out = concat_arrays(parts)
    assert_false(out.is_valid(0), "the first part's null")
    assert_true(out.is_valid(1), "the first part's value")
    assert_true(out.is_valid(2), "the second part's value")
    assert_false(out.is_valid(3), "the second part's null")
    assert_equal(out[0], 0, "a null holds a zero")


def test_concat_of_nothing_is_empty() raises:
    var out = concat_arrays(List[Array[DType.int64]]())
    assert_equal(len(out), 0, "an empty list gives an empty column")


def test_concat_of_one_part_is_a_copy() raises:
    var parts = List[Array[DType.int64]]()
    parts.append(gapped([Int64(1), 2, 3], [1]))
    var out = concat_arrays(parts)
    assert_equal(len(out), 3, "height")
    assert_false(out.is_valid(1), "the null came across")


def test_concat_of_empty_parts() raises:
    var parts = List[Array[DType.int64]]()
    parts.append(Array[DType.int64](0))
    parts.append(gapped([Int64(7)], List[Int]()))
    parts.append(Array[DType.int64](0))

    var out = concat_arrays(parts)
    assert_equal(len(out), 1, "only the middle part had rows")
    assert_equal(out[0], 7, "and it landed at row 0")


def test_concat_at_an_unaligned_offset() raises:
    # A part starts on the running total of everything before it, which here is
    # 37 and then 37 plus 90, neither of which is a multiple of sixty four. The
    # validity bits go across at those offsets, which is where an implementation
    # that copied words instead of bits would be off by a shift.
    var first = List[Int64]()
    for i in range(37):
        first.append(Int64(i))
    var second = List[Int64]()
    for i in range(90):
        second.append(Int64(100 + i))
    var third = List[Int64]()
    for i in range(45):
        third.append(Int64(1000 + i))

    var parts = List[Array[DType.int64]]()
    parts.append(gapped(first, [0, 36]))
    parts.append(gapped(second, [0, 27, 89]))
    parts.append(gapped(third, [44]))

    var out = concat_arrays(parts)
    assert_equal(len(out), 172, "height is the sum")

    var expected_missing = [0, 36, 37, 37 + 27, 37 + 89, 171]
    var at = 0
    for i in range(172):
        var want_null = at < len(expected_missing) and expected_missing[at] == i
        assert_equal(out.is_valid(i), not want_null, "row " + String(i))
        if want_null:
            at += 1

    assert_equal(out[1], 1, "a value from the first part")
    assert_equal(out[38], 101, "a value from the second part")
    assert_equal(out[130], 1000 + 130 - 127, "a value from the third part")


def test_concat_any_matches_the_typed_one() raises:
    comptime for candidate in ALL:
        var a = Array[candidate](3)
        var b = Array[candidate](2)
        for i in range(3):
            a.set_valid(i, Scalar[candidate](1))
        for i in range(2):
            b.set_valid(i, Scalar[candidate](1))
        a.set_null(1)

        var parts = List[AnyArray]()
        parts.append(AnyArray(Array[candidate](copy=a)))
        parts.append(AnyArray(Array[candidate](copy=b)))

        var out = concat_any(parts)
        assert_equal(len(out), 5, "height for " + String(candidate))
        assert_false(out.is_valid(1), "null for " + String(candidate))
        assert_equal(out.dtype(), candidate, "dtype for " + String(candidate))


def test_the_two_argument_spelling_agrees_with_the_list_one() raises:
    var a = AnyArray(gapped([Int64(1), 2, 3], [1]))
    var b = AnyArray(gapped([Int64(4), 5], [0]))

    var parts = List[AnyArray]()
    parts.append(AnyArray(copy=a))
    parts.append(AnyArray(copy=b))

    var listed = concat_any(parts)
    var paired = concat_two_any(a, b)
    assert_equal(len(listed), len(paired), "same height")
    for i in range(len(listed)):
        assert_equal(listed.is_valid(i), paired.is_valid(i), "row " + String(i))


def test_concat_any_refuses_a_different_dtype() raises:
    var parts = List[AnyArray]()
    parts.append(AnyArray(gapped([Int64(1)], List[Int]())))
    parts.append(AnyArray(Array[DType.float64](1)))

    var raised = False
    try:
        _ = concat_any(parts)
    except:
        raised = True
    assert_true(raised, "int64 stacked onto float64")


def test_concat_any_refuses_an_empty_list() raises:
    # The typed spelling can return an empty column of a known dtype and this
    # one cannot, because there is nothing to read the dtype off.
    var raised = False
    try:
        _ = concat_any(List[AnyArray]())
    except:
        raised = True
    assert_true(raised, "no columns and no dtype")


def test_concat_series_keeps_the_first_name() raises:
    var parts = List[Series]()
    parts.append(Series("a", gapped([Int64(1), 2], List[Int]())))
    parts.append(Series("b", gapped([Int64(3)], List[Int]())))

    var out = concat_series(parts)
    assert_equal(out.name, "a", "the first name wins")
    assert_equal(len(out), 3, "height is the sum")


def test_concat_frames_stacks_the_rows() raises:
    var frames = List[DataFrame]()
    frames.append(frame_of("a", [Int64(1), 2]))
    frames.append(frame_of("a", [Int64(3), 4, 5]))

    var out = concat(frames)
    assert_equal(len(out), 5, "height is the sum")
    assert_equal(out.width(), 1, "width is unchanged")
    var values = out.column("a").as_typed[DType.int64]()
    for i in range(5):
        assert_equal(values[i], Int64(i + 1), "row " + String(i))


def test_concat_frames_matches_on_names_not_positions() raises:
    var first = List[Series]()
    first.append(Series("a", gapped([Int64(1)], List[Int]())))
    first.append(Series("b", gapped([Int64(10)], List[Int]())))

    var second = List[Series]()
    second.append(Series("b", gapped([Int64(20)], List[Int]())))
    second.append(Series("a", gapped([Int64(2)], List[Int]())))

    var frames = List[DataFrame]()
    frames.append(DataFrame.from_series(first^))
    frames.append(DataFrame.from_series(second^))

    var out = concat(frames)
    assert_equal(out.names()[0], "a", "the first frame's order is kept")
    assert_equal(out.column("a").as_typed[DType.int64]()[1], 2, "a stayed a")
    assert_equal(out.column("b").as_typed[DType.int64]()[1], 20, "b stayed b")


def test_concat_frames_takes_more_than_two() raises:
    # Two frames go through the borrowing path and three or more go through the
    # list one, so both need a test.
    var frames = List[DataFrame]()
    frames.append(frame_of("a", [Int64(1)]))
    frames.append(frame_of("a", [Int64(2)]))
    frames.append(frame_of("a", [Int64(3)]))
    frames.append(frame_of("a", [Int64(4)]))

    var out = concat(frames)
    assert_equal(len(out), 4, "height is the sum")
    var values = out.column("a").as_typed[DType.int64]()
    for i in range(4):
        assert_equal(values[i], Int64(i + 1), "row " + String(i))


def test_concat_frames_refuses_a_different_width() raises:
    var wide = List[Series]()
    wide.append(Series("a", gapped([Int64(1)], List[Int]())))
    wide.append(Series("b", gapped([Int64(2)], List[Int]())))

    var frames = List[DataFrame]()
    frames.append(DataFrame.from_series(wide^))
    frames.append(frame_of("a", [Int64(3)]))

    var raised = False
    try:
        _ = concat(frames)
    except:
        raised = True
    assert_true(raised, "two columns stacked onto one")


def test_concat_frames_refuses_a_missing_column() raises:
    var frames = List[DataFrame]()
    frames.append(frame_of("a", [Int64(1)]))
    frames.append(frame_of("c", [Int64(2)]))

    var raised = False
    try:
        _ = concat(frames)
    except:
        raised = True
    assert_true(raised, "no column named a in the second frame")


def test_concat_frames_refuses_a_different_dtype() raises:
    var floats = List[Series]()
    var col = Array[DType.float64](1)
    col.set_valid(0, 1.0)
    floats.append(Series("a", col^))

    var frames = List[DataFrame]()
    frames.append(frame_of("a", [Int64(1)]))
    frames.append(DataFrame.from_series(floats^))

    var raised = False
    try:
        _ = concat(frames)
    except:
        raised = True
    assert_true(raised, "an int64 column and a float64 one named the same")


def test_concat_of_no_frames_is_empty() raises:
    var out = concat(List[DataFrame]())
    assert_equal(len(out), 0, "no rows")
    assert_equal(out.width(), 0, "no columns")


def test_concat_of_one_frame_is_a_copy() raises:
    var frames = List[DataFrame]()
    frames.append(frame_of("a", [Int64(1), 2]))
    var out = concat(frames)
    assert_equal(len(out), 2, "height")
    assert_equal(out.column("a").as_typed[DType.int64]()[1], 2, "values")


def test_concat_leaves_its_inputs_alone() raises:
    var frames = List[DataFrame]()
    frames.append(frame_of("a", [Int64(1), 2]))
    frames.append(frame_of("a", [Int64(3)]))

    var out = concat(frames)
    assert_equal(len(out), 3, "the result")
    assert_equal(len(frames[0]), 2, "the first input is untouched")
    assert_equal(len(frames[1]), 1, "the second input is untouched")


def test_concat_agrees_with_its_twin() raises:
    var rng = Rng(0xDEADBEEF)
    var parts = List[Array[DType.int64]]()
    for _ in range(5):
        var rows = rng.next_below(70)
        var col = Array[DType.int64](rows)
        for i in range(rows):
            if rng.next_below(4) == 0:
                col.set_null(i)
            else:
                col.set_valid(i, Int64(rng.next_below(1000)))
        parts.append(col^)

    var fast = concat_arrays(parts)
    var slow = concat_scalar(parts)
    assert_equal(len(fast), len(slow), "same height")
    for i in range(len(fast)):
        assert_equal(
            fast.is_valid(i), slow.is_valid(i), "validity row " + String(i)
        )
        assert_equal(fast[i], slow[i], "row " + String(i))


def strings_of(values: List[String], missing: List[Int]) raises -> StringArray:
    """Builds a string column with nulls at the given positions.

    Args:
        values: The values, one per row. The value at a null row is ignored.
        missing: Which rows are null.

    Returns:
        The column.
    """
    var builder = StringBuilder(capacity=len(values))
    for i in range(len(values)):
        var is_null = False
        for m in range(len(missing)):
            if missing[m] == i:
                is_null = True
        if is_null:
            builder.append_null()
        else:
            builder.append(values[i].as_bytes())
    return builder^.finish()


def test_concat_strings_stacks_payloads() raises:
    # Every part has at least one element too long to live inside its view, so
    # every part after the first has offsets that have to be moved along, which
    # is the one thing the block copy has to get right and the element by
    # element version got for free.
    var left = strings_of(
        ["short", "a string well over twelve bytes", "x"], [2]
    )
    var right = strings_of(
        ["another string over twelve bytes", "", "third one over twelve bytes"],
        [],
    )
    var parts = List[StringArray]()
    parts.append(left^)
    parts.append(right^)

    var out = concat_strings(parts)
    assert_equal(len(out), 6, "height")
    assert_equal(out[0], "short", "row 0")
    assert_equal(out[1], "a string well over twelve bytes", "row 1")
    assert_false(out.is_valid(2), "row 2 is null")
    assert_equal(out[3], "another string over twelve bytes", "row 3")
    assert_equal(out[4], "", "row 4")
    assert_equal(out[5], "third one over twelve bytes", "row 5")


def test_concat_strings_agrees_with_element_by_element() raises:
    # The reference is what the kernel used to do: append every element to a
    # builder in order. Lengths straddle the twelve byte inline boundary and the
    # part heights are odd on purpose, so the validity of a part lands at a bit
    # offset that is not a multiple of eight, let alone of sixty four.
    var rng = Rng(0x5EED1234)
    var parts = List[StringArray]()
    for _ in range(6):
        var rows = rng.next_below(40)
        var builder = StringBuilder(capacity=rows)
        for _ in range(rows):
            if rng.next_below(5) == 0:
                builder.append_null()
            else:
                var text = String("")
                for _ in range(rng.next_below(30)):
                    text += String(chr(97 + rng.next_below(26)))
                builder.append(text.as_bytes())
        parts.append(builder^.finish())

    var reference = StringBuilder()
    for p in range(len(parts)):
        for i in range(len(parts[p])):
            if parts[p].is_valid(i):
                reference.append(parts[p].unsafe_bytes(i))
            else:
                reference.append_null()
    var slow = reference^.finish()

    var fast = concat_strings(parts)
    assert_equal(len(fast), len(slow), "same height")
    for i in range(len(fast)):
        assert_equal(
            fast.is_valid(i), slow.is_valid(i), "validity row " + String(i)
        )
        assert_equal(fast[i], slow[i], "row " + String(i))


def test_concat_strings_through_any_and_pairs() raises:
    var first = strings_of(["one over twelve bytes long", "b"], [])
    var second = strings_of(["c", "two over twelve bytes long"], [1])

    var listed = List[AnyArray]()
    listed.append(AnyArray(StringArray(copy=first)))
    listed.append(AnyArray(StringArray(copy=second)))
    var from_list_form = concat_any(listed)
    var from_pair = concat_two_any(AnyArray(first^), AnyArray(second^))

    assert_equal(len(from_list_form), 4, "height")
    ref listed_out = from_list_form.strings()
    ref paired_out = from_pair.strings()
    for i in range(4):
        assert_equal(
            listed_out.is_valid(i),
            paired_out.is_valid(i),
            "validity row " + String(i),
        )
        assert_equal(listed_out[i], paired_out[i], "row " + String(i))
    assert_equal(listed_out[0], "one over twelve bytes long", "row 0")
    assert_equal(listed_out[3], "", "row 3 is the null")


def test_concat_strings_over_the_parallel_threshold() raises:
    # Past `PARALLEL_ROWS` the parts are copied on every core at once rather
    # than one after another, and the two paths have to give the same column.
    # The reference here is the sequential path, reached by splitting the same
    # elements into a single part, and the elements straddle the inline
    # boundary so the payload rebase runs on most parts but not on all of them.
    var rng = Rng(0xC0FFEE01)
    var texts = List[String]()
    var nulls = List[Bool]()
    for _ in range(PARALLEL_ROWS + 517):
        if rng.next_below(9) == 0:
            texts.append(String(""))
            nulls.append(True)
            continue
        var text = String("")
        for _ in range(2 + rng.next_below(24)):
            text += String(chr(97 + rng.next_below(26)))
        texts.append(text)
        nulls.append(False)

    var whole = StringBuilder(capacity=len(texts))
    for i in range(len(texts)):
        if nulls[i]:
            whole.append_null()
        else:
            whole.append(texts[i].as_bytes())
    var one_part = List[StringArray]()
    one_part.append(whole^.finish())
    var slow = concat_strings(one_part)

    var parts = List[StringArray]()
    var at = 0
    var take = 1
    while at < len(texts):
        var stop = at + take
        if stop > len(texts):
            stop = len(texts)
        var builder = StringBuilder(capacity=stop - at)
        for i in range(at, stop):
            if nulls[i]:
                builder.append_null()
            else:
                builder.append(texts[i].as_bytes())
        parts.append(builder^.finish())
        at = stop
        take = take * 2 + 1
    assert_true(len(parts) > 1, "the split has to make several parts")
    var fast = concat_strings(parts)

    assert_equal(len(fast), len(slow), "same height")
    for i in range(len(fast)):
        assert_equal(
            fast.is_valid(i), slow.is_valid(i), "validity row " + String(i)
        )
        assert_equal(fast[i], slow[i], "row " + String(i))


def test_concat_strings_through_any_over_the_threshold() raises:
    # The erased spelling is the one `read_csv` reaches, so it gets its own
    # pass over the parallel path rather than relying on the typed one.
    var parts = List[AnyArray]()
    var expected = List[String]()
    for p in range(4):
        var rows = PARALLEL_ROWS // 3
        var builder = StringBuilder(capacity=rows)
        for i in range(rows):
            var text = String("part ", p, " row ", i, " padded out a bit")
            builder.append(text.as_bytes())
            expected.append(text)
        parts.append(AnyArray(builder^.finish()))

    var out = concat_any(parts)
    ref stacked = out.strings()
    assert_equal(len(stacked), len(expected), "height")
    for i in range(len(expected)):
        assert_equal(stacked[i], expected[i], "row " + String(i))


def test_concat_fixed_over_the_parallel_threshold() raises:
    # Past `PARALLEL_ROWS` the value copies are handed out one part to a worker
    # while the validity still goes in one part at a time, so this is the test
    # that the two halves agree about where a part lands. The part lengths are
    # deliberately not multiples of sixty four, so most boundaries fall inside a
    # byte of the bitmap.
    var parts = List[AnyArray]()
    var expected = List[Int64]()
    var present = List[Bool]()
    var take = 4001
    var value = Int64(0)
    while len(expected) < PARALLEL_ROWS + 1000:
        var part = Array[DType.int64](take)
        for i in range(take):
            if (Int(value) + i) % 7 == 3:
                part.set_null(i)
                expected.append(0)
                present.append(False)
            else:
                part.set_valid(i, value + Int64(i))
                expected.append(value + Int64(i))
                present.append(True)
        value += Int64(take)
        parts.append(AnyArray(part^))
        take += 997
    assert_true(len(parts) > 1, "the split has to make several parts")

    var out = concat_any(parts)
    ref stacked = out.as_typed[DType.int64]()
    assert_equal(len(stacked), len(expected), "height")
    for i in range(len(expected)):
        assert_equal(stacked.is_valid(i), present[i], "validity " + String(i))
        assert_equal(stacked[i], expected[i], "row " + String(i))


def test_concat_refs_any_matches_the_owning_spelling() raises:
    # The borrowing spelling is what the frame layer and the Arrow reader use,
    # so it gets checked against the one that owns its parts rather than being
    # trusted because it shares the body.
    var owned = List[AnyArray]()
    for p in range(5):
        var part = from_list[DType.int32]([Int32(p), Int32(p + 10)])
        owned.append(AnyArray(part^))

    var refs = List[Pointer[AnyArray, ImmUntrackedOrigin]]()
    for p in range(len(owned)):
        refs.append(column_ref(owned[p]))

    var borrowed = concat_refs_any(refs)
    var copied = concat_any(owned)
    ref left = borrowed.as_typed[DType.int32]()
    ref right = copied.as_typed[DType.int32]()
    assert_equal(len(left), 10, "height")
    assert_equal(len(right), 10, "height")
    for i in range(10):
        assert_equal(left[i], right[i], "row " + String(i))

    # The parts are still readable, which is the property that makes handing out
    # references reasonable at all.
    assert_equal(len(owned), 5, "the parts survive")
    assert_equal(owned[4].as_typed[DType.int32]()[1], 14, "part 4 survives")


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
