"""Tests for the display layer.

Rendering is the one part of the frame layer where the output is the whole
contract, so most of these compare a complete string rather than probing at it.
That makes them brittle by design: if a column width or a separator changes, the
test says so, and whether the change was wanted is then a decision someone makes
rather than something that slips through because the assertion was loose enough
to accept both.

The parts that are not compared whole are the two elisions, because a twelve row
frame written out in full in a test file is less readable than the code that
generates it. Those are checked on the properties that matter: the line count, the
positions that survived, and the ones that did not.

`format_float` gets its own tests separate from any frame, because the interesting
inputs are the ones a frame is unlikely to contain by accident. Negative zero,
both infinities, a NaN and the two magnitudes where fixed point rendering is
abandoned are all in here.
"""

from std.testing import TestSuite, assert_equal, assert_false, assert_true

from firepanda.array.any import AnyArray
from firepanda.array.array import Array
from firepanda.dtype.lists import ALL
from firepanda.frame.display import (
    DisplayOptions,
    format_float,
    render_column,
    render_table,
    render_value,
    visible,
)
from firepanda.frame.frame import DataFrame
from firepanda.frame.series import Series


def has(text: String, needle: String) -> Bool:
    """Reports whether `needle` occurs in `text`."""
    return text.find(needle) != -1


def int_column(name: String, values: List[Int64]) raises -> Series:
    """Builds a fully valid int64 series."""
    var col = Array[DType.int64](len(values))
    for i in range(len(values)):
        col.set_valid(i, values[i])
    return Series(name, col^)


def small_frame() raises -> DataFrame:
    """Two rows and two columns, chosen so every cell has a different width."""
    var a = Array[DType.int64](2)
    a.set_valid(0, Int64(1))
    a.set_valid(1, Int64(20))

    var bb = Array[DType.float64](2)
    bb.set_valid(0, Float64(1.5))
    bb.set_null(1)

    var columns = List[Series]()
    columns.append(Series("a", a^))
    columns.append(Series("bb", bb^))
    return DataFrame.from_series(columns^)


def test_a_small_frame_renders_exactly() raises:
    var expected = String(
        "    a    bb\n0   1   1.5\n1  20  <NA>\n\n[2 rows x 2 columns]"
    )
    assert_equal(String(small_frame()), expected, "rendered frame")


def test_a_column_is_as_wide_as_its_own_name() raises:
    # `bb` is wider than both its values, so the width comes from the header and
    # the values are padded up to it rather than the other way round.
    var lines = String(small_frame()).split("\n")
    assert_equal(lines[0].byte_length(), 11, "header width")
    assert_equal(lines[1].byte_length(), 11, "first row width")
    assert_equal(lines[2].byte_length(), 11, "second row width")


def test_an_empty_frame_says_so() raises:
    assert_equal(
        String(DataFrame()),
        String("Empty DataFrame\n\n[0 rows x 0 columns]"),
        "empty frame",
    )


def test_a_tall_frame_elides_its_middle() raises:
    var values = List[Int64]()
    for i in range(12):
        values.append(Int64(100 + i))
    var columns = List[Series]()
    columns.append(int_column("v", values))
    var rendered = String(DataFrame.from_series(columns^))
    var lines = rendered.split("\n")

    # Header, five rows, the elision, five rows, a blank line and the shape.
    assert_equal(len(lines), 14, "line count")
    assert_true(has(String(lines[6]), "..."), "the middle is elided")
    assert_equal(lines[13], String("[12 rows x 1 columns]"), "shape line")

    assert_true(has(rendered, "104"), "the fifth row survives")
    assert_true(has(rendered, "107"), "the eighth row survives")
    assert_false(has(rendered, "105"), "the sixth row is gone")
    assert_false(has(rendered, "106"), "the seventh row is gone")


def test_a_wide_frame_elides_its_columns() raises:
    var columns = List[Series]()
    for c in range(25):
        var values = List[Int64]()
        values.append(Int64(c))
        values.append(Int64(c))
        columns.append(int_column(String("c", c), values))
    var rendered = String(DataFrame.from_series(columns^))
    var header = rendered.split("\n")[0]

    assert_true(has(String(header), "c0"), "the first column survives")
    assert_true(has(String(header), "c9"), "the tenth column survives")
    assert_true(has(String(header), "c15"), "the sixteenth column survives")
    assert_true(has(String(header), "c24"), "the last column survives")
    assert_true(has(String(header), "..."), "the middle is elided")
    assert_false(has(String(header), "c12"), "the thirteenth column is gone")
    assert_true(
        has(rendered, "[2 rows x 25 columns]"),
        "the shape counts every column, not the printed ones",
    )


def test_the_limits_are_options_not_constants() raises:
    var values = List[Int64]()
    for i in range(10):
        values.append(Int64(i))
    var columns = List[Series]()
    columns.append(int_column("v", values))
    var df = DataFrame.from_series(columns^)

    var rendered = render_table(
        df.schema, df.columns, len(df), DisplayOptions(max_rows=4)
    )
    var lines = rendered.split("\n")
    # Header, two rows, the elision, two rows, a blank line and the shape.
    assert_equal(len(lines), 8, "line count at four rows")
    assert_true(has(String(lines[3]), "..."), "the middle is elided")


def test_a_null_and_a_nan_do_not_look_the_same() raises:
    var col = Array[DType.float64](3)
    col.set_valid(0, Float64(1.0))
    col.set_valid(1, Float64(0.0) / Float64(0.0))
    col.set_null(2)
    var rendered = render_column("f", AnyArray(col^), DisplayOptions())

    assert_true(has(rendered, "NaN"), "the NaN prints as a value")
    assert_true(has(rendered, "<NA>"), "the null prints as a null")


def test_bools_print_the_way_python_spells_them() raises:
    var col = Array[DType.bool](2)
    col.set_valid(0, True)
    col.set_valid(1, False)
    var values = AnyArray(col^)
    var options = DisplayOptions()

    assert_equal(render_value(values, 0, options), String("True"), "true")
    assert_equal(render_value(values, 1, options), String("False"), "false")


def test_every_dtype_renders_something() raises:
    var options = DisplayOptions()
    comptime for candidate in ALL:
        var col = Array[candidate](1)
        col.set_valid(0, Scalar[candidate](1))
        var text = render_value(AnyArray(col^), 0, options)
        assert_true(
            text.byte_length() > 0,
            String("dtype ", candidate, " renders"),
        )
        assert_true(
            text != "?",
            String("dtype ", candidate, " resolved in the dispatch"),
        )


def test_a_null_renders_as_the_null_text() raises:
    var col = Array[DType.int64](1)
    col.set_null(0)
    assert_equal(
        render_value(AnyArray(col^), 0, DisplayOptions()),
        String("<NA>"),
        "null text",
    )


def test_a_series_prints_its_values_and_its_dtype() raises:
    var s = int_column("a", [Int64(1), Int64(2), Int64(3)])
    assert_equal(
        String(s),
        String("0    1\n1    2\n2    3\nName: a, dtype: int64"),
        "rendered series",
    )


def test_an_unnamed_series_leaves_the_name_out() raises:
    var s = int_column("", [Int64(1)])
    assert_equal(String(s), String("0    1\ndtype: int64"), "unnamed series")


def test_a_long_series_reports_its_length() raises:
    var values = List[Int64]()
    for i in range(30):
        values.append(Int64(i))
    var rendered = String(int_column("a", values))
    var lines = rendered.split("\n")

    assert_equal(len(lines), 12, "ten rows, the elision and the footer")
    assert_equal(
        lines[11], String("Length: 30, Name: a, dtype: int64"), "footer"
    )


def test_a_short_series_does_not_report_its_length() raises:
    var rendered = String(int_column("a", [Int64(1)]))
    assert_false(has(rendered, "Length:"), "no length line")


def test_an_empty_series_is_one_line() raises:
    var s = int_column("a", List[Int64]())
    assert_equal(
        String(s), String("Series([], Name: a, dtype: int64)"), "empty series"
    )


def test_a_float_series_reports_the_float_dtype() raises:
    var col = Array[DType.float32](1)
    col.set_valid(0, Float32(0.25))
    var rendered = render_column("x", AnyArray(col^), DisplayOptions())
    assert_equal(
        rendered, String("0    0.25\nName: x, dtype: float32"), "float32 series"
    )


def test_floats_are_rounded_and_stripped() raises:
    assert_equal(format_float(Float64(1) / Float64(3), 6), "0.333333", "third")
    assert_equal(format_float(1.5, 6), "1.5", "one trailing digit kept")
    assert_equal(format_float(2.0, 6), "2.0", "an integral float stays a float")
    assert_equal(format_float(123.456, 6), "123.456", "three places")
    assert_equal(format_float(-2.25, 6), "-2.25", "negative")
    assert_equal(format_float(0.0, 6), "0.0", "zero")


def test_the_precision_is_configurable() raises:
    assert_equal(format_float(Float64(1) / Float64(3), 2), "0.33", "two places")
    assert_equal(format_float(Float64(2) / Float64(3), 2), "0.67", "rounds up")
    assert_equal(format_float(1.0 / 3.0, 0), "0", "no places at all")


def test_the_special_values_have_their_own_spellings() raises:
    var zero = Float64(0.0)
    assert_equal(format_float(zero / zero, 6), "NaN", "nan")
    assert_equal(format_float(Float64(1.0) / zero, 6), "inf", "positive inf")
    assert_equal(format_float(Float64(-1.0) / zero, 6), "-inf", "negative inf")
    assert_equal(format_float(-zero, 6), "-0.0", "negative zero keeps its sign")


def test_the_extremes_fall_back_to_mojo_formatting() raises:
    # Above 1e15 the integer part is beyond what six decimals adds anything to,
    # and below 1e-4 every printed place would be a zero. Both switch to an
    # exponent rather than lying about the value.
    assert_true(has(format_float(1.0e20, 6), "e"), "large magnitude")
    assert_true(has(format_float(1.0e-20, 6), "e"), "small magnitude")
    assert_equal(format_float(1.0e-20, 6) != "0.0", True, "not rounded to zero")


def test_visible_keeps_both_ends_and_marks_the_gap() raises:
    var all_of_them = visible(3, 10)
    assert_equal(len(all_of_them), 3, "nothing elided")
    assert_equal(all_of_them[2], 2, "last position")

    var elided = visible(12, 10)
    assert_equal(len(elided), 11, "ten positions plus the gap")
    assert_equal(elided[0], 0, "first")
    assert_equal(elided[4], 4, "last of the head")
    assert_equal(elided[5], -1, "the gap")
    assert_equal(elided[6], 7, "first of the tail")
    assert_equal(elided[10], 11, "last")


def test_a_frame_can_still_report_its_schema_without_its_values() raises:
    var described = small_frame().describe()
    assert_true(has(described, "2 rows x 2 columns"), "shape")
    assert_true(has(described, "a: int64"), "first column")
    assert_true(has(described, "bb: float64"), "second column")
    assert_true(has(described, "1 null"), "the null count")
    assert_false(has(described, "1.5"), "no values")


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
