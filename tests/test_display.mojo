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
positions that survived, the ones that did not, and the elided line itself, which
is compared whole because where the dots sit and how many of them there are is
the thing being asserted.

Every expected string in here that has spacing in it was measured against a
running pandas rather than worked out. That is worth saying because several of
them look wrong: a line that ends in spaces, a column elided by two dots beside
one elided by three, a name held in by a sign that the column will never print.

`format_float` gets its own tests separate from any frame, because the interesting
inputs are the ones a frame is unlikely to contain by accident. Negative zero,
both infinities, a NaN and the two magnitudes where fixed point rendering is
abandoned are all in here.

`float_column` gets its own tests for a different reason. What a float prints as
is not a fact about the float, it is a fact about the column it is in, so the
inputs that matter are pairs: a value that is fine beside a value that is not,
the same value with and without a longer one next to it, a value in the part of
the column that will not be printed. Every expected string in that group came
off a running pandas.
"""

from std.testing import TestSuite, assert_equal, assert_false, assert_true

from std.collections import Optional

from firepanda.array.any import AnyArray
from firepanda.array.array import Array, from_list
from firepanda.array.strings import strings_from_list
from firepanda.dtype.lists import ALL
from firepanda.frame.display import (
    DisplayOptions,
    fixed_text,
    float_column,
    format_float,
    pad_right,
    render_column,
    render_table,
    render_value,
    scientific_text,
    visible,
)
from firepanda.frame.frame import DataFrame
from firepanda.frame.index import Index
from firepanda.frame.series import Series


def has(text: String, needle: String) -> Bool:
    """Reports whether `needle` occurs in `text`."""
    return text.find(needle) != -1


def unnamed() -> Optional[String]:
    """A level name of `None`, spelled once because it is written a lot."""
    return Optional[String]()


def text_index(
    values: List[String], var name: Optional[String]
) raises -> Index:
    """An index over text labels, which is what a `set_index` leaves behind."""
    return Index(AnyArray(strings_from_list(values)), name^)


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
        df.schema, df.column_refs(), len(df), DisplayOptions(max_rows=4)
    )
    var lines = rendered.split("\n")
    # Header, two rows, the elision, two rows, a blank line and the shape.
    assert_equal(len(lines), 8, "line count at four rows")
    # Two dots rather than three, because neither column is wider than three
    # and pandas drops a dot rather than widening a column to fit one.
    assert_true(has(String(lines[3]), ".."), "the middle is elided")


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
        lines[11], String("Name: a, Length: 30, dtype: int64"), "footer"
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
    # At no places at all a third is smaller than the last place being printed,
    # which is the one thing that sends a column to an exponent on its own.
    # pandas prints `3e-01` here and rounding it to `0` would be a lie about a
    # value that is not zero.
    assert_equal(format_float(1.0 / 3.0, 0), "3e-01", "no places at all")
    assert_equal(format_float(123.456, 0), "123", "a value that still fits")


def test_the_special_values_have_their_own_spellings() raises:
    var zero = Float64(0.0)
    assert_equal(format_float(zero / zero, 6), "NaN", "nan")
    assert_equal(format_float(Float64(1.0) / zero, 6), "inf", "positive inf")
    assert_equal(format_float(Float64(-1.0) / zero, 6), "-inf", "negative inf")
    assert_equal(format_float(-zero, 6), "-0.0", "negative zero keeps its sign")


def test_the_extremes_are_written_with_an_exponent() raises:
    # Above 1e15 the integer part is beyond what an `Int` can write out, and
    # below the last printed place every printed place would be a zero. Both
    # switch to an exponent rather than lying about the value.
    assert_equal(format_float(1.0e20, 6), "1.000000e+20", "large magnitude")
    assert_equal(format_float(1.0e-20, 6), "1.000000e-20", "small magnitude")
    assert_equal(format_float(1.0e-20, 6) != "0.0", True, "not rounded to zero")


def floats(
    var values: List[Float64], precision: Int = 6
) raises -> List[String]:
    """A whole column of values, all of them present, rendered together."""
    var present = List[Bool](capacity=len(values))
    for _ in range(len(values)):
        present.append(True)
    return float_column(values, present, precision, String("<NA>"))


def test_the_zeros_come_off_a_column_and_not_off_a_value() raises:
    # `2.0` on its own is `2.0`, and beside a value with three places it is
    # `2.000`, because a place comes off every value in the column or off none.
    var alone = floats([2.0])
    assert_equal(alone[0], String("2.0"), "on its own")
    var beside = floats([1234567.125, 2.0])
    assert_equal(beside[0], String("1234567.125"), "the long one")
    assert_equal(beside[1], String("2.000"), "the short one, padded out")


def test_one_place_always_survives_the_stripping() raises:
    var cells = floats([2.0, 4.0])
    assert_equal(cells[0], String("2.0"), "an integral value is still a float")
    assert_equal(cells[1], String("4.0"), "and so is the one beside it")


def test_a_column_goes_to_an_exponent_when_it_gets_too_long() raises:
    # Twelve characters counting the place kept in front of the value is the
    # most a fixed column may be. `123456789.0` is twelve and stays, and the
    # value ten times larger is thirteen and takes the whole column with it.
    var stays = floats([123456789.0, 2.0])
    assert_equal(stays[0], String("123456789.0"), "still fixed")
    assert_equal(stays[1], String("2.0"), "and so is what is beside it")
    var goes = floats([1234567890.0, 2.0])
    assert_equal(goes[0], String("1.234568e+09"), "too long")
    assert_equal(goes[1], String("2.000000e+00"), "taken along with it")


def test_the_place_in_front_of_a_value_counts_towards_the_length() raises:
    # A negative spends that place on its minus, so `-123456789.0` is twelve
    # like the positive is and stays fixed for the same reason.
    var stays = floats([-123456789.0, 2.0])
    assert_equal(stays[0], String("-123456789.0"), "twelve with the minus")
    var goes = floats([-1234567890.0, 2.0])
    assert_equal(goes[0], String("-1.234568e+09"), "thirteen with the minus")


def test_length_alone_is_not_enough_to_send_a_column_to_an_exponent() raises:
    # Nothing in here is larger than 1e6, so the length does not matter and a
    # long rendering stays fixed.
    var cells = floats([0.123456789, 2.0])
    assert_equal(cells[0], String("0.123457"), "long but small")


def test_a_value_under_the_last_place_sends_the_column_to_an_exponent() raises:
    var goes = floats([1.0e-10, 1.0])
    assert_equal(goes[0], String("1.000000e-10"), "under the last place")
    assert_equal(goes[1], String("1.000000e+00"), "taken along with it")
    # The boundary is the last place itself, which prints fixed.
    assert_equal(floats([1.0e-6])[0], String("0.000001"), "the last place")
    assert_equal(floats([1.0e-5])[0], String("0.00001"), "one place above it")


def test_a_null_and_a_special_value_are_not_numbers_for_any_of_this() raises:
    # An infinity is larger than any threshold there is and none of them apply
    # to it, so the column beside it stays fixed and prints as it would alone.
    var present: List[Bool] = [True, True, False]
    var cells = float_column(
        [Float64(1.0) / Float64(0.0), 1.0, 0.0], present, 6, String("<NA>")
    )
    assert_equal(cells[0], String("inf"), "the infinity")
    assert_equal(cells[1], String("1.0"), "the value beside it")
    assert_equal(cells[2], String("<NA>"), "the null")


def test_the_rounding_goes_to_the_even_digit() raises:
    # Which is what C's own formatting does and therefore what pandas prints.
    # The values that show it are the ones whose half is exact.
    assert_equal(fixed_text(0.0078125, 6), String("0.007812"), "down to even")
    assert_equal(fixed_text(0.0234375, 6), String("0.023438"), "up to even")
    assert_equal(fixed_text(2.5, 0), String("2"), "a half at no places")
    assert_equal(fixed_text(3.5, 0), String("4"), "the next one up")


def test_an_exponent_is_signed_and_at_least_two_digits_wide() raises:
    assert_equal(scientific_text(1.0, 6), String("1.000000e+00"), "no exponent")
    assert_equal(scientific_text(1.0e100, 6), String("1.000000e+100"), "three")
    assert_equal(scientific_text(-1.0e-7, 6), String("-1.000000e-07"), "signed")


def test_the_rounding_of_a_mantissa_can_carry_into_the_exponent() raises:
    # 9.9999999 rounds to ten, which is not a mantissa, so the exponent takes
    # the extra place instead.
    assert_equal(
        scientific_text(9.9999999e20, 6), String("1.000000e+21"), "carried"
    )


def float_series(name: String, values: List[Float64]) raises -> Series:
    """A fully valid float64 series."""
    var col = Array[DType.float64](len(values))
    for i in range(len(values)):
        col.set_valid(i, values[i])
    return Series(name, col^)


def test_a_value_in_the_elided_middle_does_not_decide_the_column() raises:
    # pandas elides first and formats second, so a value nobody will see cannot
    # push the values around it into an exponent.
    var values = List[Float64]()
    for _ in range(6):
        values.append(1.0)
    values.append(1.0e16)
    for _ in range(6):
        values.append(2.0)
    var rendered = String(float_series("v", values))
    assert_true(has(rendered, "1.0"), "the printed values are still fixed")
    assert_false(has(rendered, "e+16"), "and the elided one is not in there")


def test_each_column_of_a_frame_decides_on_its_own() raises:
    var columns = List[Series]()
    columns.append(float_series("a", [1.0e16, 2.0]))
    columns.append(float_series("b", [1.0, 2.0]))
    var rendered = String(DataFrame.from_series(columns^))
    assert_equal(
        rendered,
        String(
            "              a    b\n0  1.000000e+16  1.0\n1  2.000000e+00 "
            " 2.0\n\n[2 rows x 2 columns]"
        ),
        "one column scientific and one not",
    )


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


def test_a_column_prints_its_labels_and_not_its_positions() raises:
    var s = int_column("v", [Int64(1), Int64(2), Int64(3)])
    s.index = text_index(["p", "qq", "r"], "k")
    assert_equal(
        String(s),
        String("k\np     1\nqq    2\nr     3\nName: v, dtype: int64"),
        "labelled column",
    )


def test_the_labels_are_left_aligned_and_the_values_are_not() raises:
    # pandas puts the labels hard against the left edge whatever they are, so a
    # column labelled 10, 200 and 3 does not line its labels up on the last
    # digit the way it lines the values up.
    var s = int_column("v", [Int64(1), Int64(2), Int64(3)])
    s.index = Index(AnyArray(from_list[DType.int64]([10, 200, 3])), unnamed())
    assert_equal(
        String(s),
        String("10     1\n200    2\n3      3\nName: v, dtype: int64"),
        "widths differ",
    )


def test_an_unnamed_index_prints_no_line_above_the_listing() raises:
    var s = int_column("v", [Int64(1), Int64(2)])
    s.index = text_index(["p", "q"], None)
    assert_equal(
        String(s),
        String("p    1\nq    2\nName: v, dtype: int64"),
        "no name line",
    )


def test_a_range_that_was_named_still_prints_its_name() raises:
    var s = int_column("v", [Int64(1), Int64(2)])
    s.index = Index(0, 2, Optional[String]("ix"))
    assert_equal(
        String(s),
        String("ix\n0    1\n1    2\nName: v, dtype: int64"),
        "a named range",
    )


def test_a_range_that_does_not_start_at_zero_prints_its_own_labels() raises:
    var s = int_column("v", [Int64(1), Int64(2)])
    s.index = Index(7, 2, unnamed())
    assert_equal(
        String(s), String("7    1\n8    2\nName: v, dtype: int64"), "offset"
    )


def test_an_empty_column_says_nothing_about_its_labels() raises:
    # pandas leaves the name line out here too, because there is no listing for
    # it to sit above.
    var s = int_column("v", List[Int64]())
    s.index = Index(0, 0, Optional[String]("ix"))
    assert_equal(
        String(s), String("Series([], Name: v, dtype: int64)"), "nothing to say"
    )


def test_a_missing_label_prints_the_way_a_missing_value_does() raises:
    var col = Array[DType.int64](2)
    col.set_valid(0, Int64(10))
    col.set_null(1)
    var s = int_column("v", [Int64(1), Int64(2)])
    s.index = Index(AnyArray(col^), unnamed())
    assert_equal(
        String(s),
        String("10      1\n<NA>    2\nName: v, dtype: int64"),
        "a null label",
    )


def test_the_label_elision_lines_up_with_the_row_elision() raises:
    var values = List[Int64]()
    var names = List[String]()
    for i in range(12):
        values.append(Int64(i))
        names.append(String("r", i))
    var s = int_column("v", values)
    s.index = text_index(names, None)
    var lines = String(s).split("\n")
    assert_equal(len(lines), 12, "ten rows, the gap and the footer")
    assert_true(has(String(lines[0]), "r0"), "the first label")
    # The label on the elided row is blank, which is what pandas leaves there on
    # a column, and the value is the dots.
    assert_equal(String(lines[5]), String("       .."), "the gap")
    assert_true(has(String(lines[6]), "r7"), "the tail resumes where rows do")
    assert_true(has(String(lines[10]), "r11"), "the last label")


def test_a_frame_prints_its_labels_down_the_left() raises:
    var df = small_frame()
    df.index = text_index(["p", "qq"], None)
    assert_equal(
        String(df),
        String(
            "     a    bb\np    1   1.5\nqq  20  <NA>\n\n[2 rows x 2 columns]"
        ),
        "labelled frame",
    )


def test_a_frame_puts_the_index_name_under_the_header() raises:
    # The name row is blank in every column but the first, and pandas leaves the
    # blanks in rather than trimming the line, so this one has trailing spaces.
    var df = small_frame()
    df.index = text_index(["p", "qq"], "k")
    var lines = String(df).split("\n")
    assert_equal(String(lines[0]), String("     a    bb"), "the header")
    assert_equal(String(lines[1]), pad_right("k", 12), "the name")
    assert_equal(String(lines[2]), String("p    1   1.5"), "the first row")


def test_a_renderer_given_no_labels_prints_the_positions() raises:
    # Every caller outside the frame layer passes no labels, and the fallback is
    # what the whole file did before labels reached it.
    var col = Array[DType.int64](2)
    col.set_valid(0, Int64(1))
    col.set_valid(1, Int64(2))
    assert_equal(
        render_column("v", AnyArray(col^), DisplayOptions()),
        String("0    1\n1    2\nName: v, dtype: int64"),
        "positions",
    )


def test_only_the_labels_that_will_be_printed_are_rendered() raises:
    # The cost of rendering an index has to be the cost of the rows on screen
    # and not the cost of the frame, or printing a large frame would be the
    # slowest thing a prompt can do.
    var cells = Index(1000000).display_cells(DisplayOptions())
    assert_equal(len(cells.cells), 11, "ten labels and the gap")
    assert_equal(cells.cells[0], String("0"), "the first label")
    assert_equal(cells.cells[5], String("..."), "the gap")
    assert_equal(cells.cells[10], String("999999"), "the last label")
    assert_false(Bool(cells.name), "and the default range is unnamed")


def named_frame(name: String, var column: AnyArray) raises -> DataFrame:
    """A frame of one column, for the tests about where a name is written."""
    var columns = List[Series]()
    columns.append(Series(name, column^))
    return DataFrame.from_series(columns^)


def test_a_frame_writes_a_minus_into_the_gap() raises:
    # Measured against pandas 3, which pads a column of `1` and `-20` to the
    # width of `20` and lets the minus have the gap. Every expected string in
    # this group came out of a running pandas rather than out of a head.
    var a = Array[DType.int64](2)
    a.set_valid(0, Int64(1))
    a.set_valid(1, Int64(-20))

    var b = Array[DType.float64](2)
    b.set_valid(0, Float64(1.5))
    b.set_valid(1, Float64(-0.5))

    var columns = List[Series]()
    columns.append(Series("a", a^))
    columns.append(Series("b", b^))
    columns.append(Series("c", strings_from_list(["x", "-y"])))
    var df = DataFrame.from_series(columns^)
    df.index = text_index(["p", "qq"], String("k"))

    assert_equal(
        String(df),
        String(
            "     a    b   c\n"
            "k              \n"
            "p    1  1.5   x\n"
            "qq -20 -0.5  -y\n"
            "\n[2 rows x 3 columns]"
        ),
        "a frame with a negative in it",
    )


def test_a_numeric_name_is_held_in_and_a_text_name_is_not() raises:
    # Nothing about the data explains the difference between these two. The
    # integer column's name is indented by the place kept for a sign and the
    # text column's name is not, which is the dtype and nothing else.
    var text = named_frame("aaaaaa", AnyArray(strings_from_list(["x", "y"])))
    assert_equal(
        String(text),
        String("  aaaaaa\n0      x\n1      y\n\n[2 rows x 1 columns]"),
        "a text column",
    )

    var a = Array[DType.int64](2)
    a.set_valid(0, Int64(1))
    a.set_valid(1, Int64(-20))
    assert_equal(
        String(named_frame("aaaaaa", AnyArray(a^))),
        String("   aaaaaa\n0       1\n1     -20\n\n[2 rows x 1 columns]"),
        "an integer column",
    )


def test_a_boolean_name_is_held_in_like_a_number() raises:
    # A sign no boolean will ever print, which is the one place where pandas
    # counting a boolean as numeric can be seen.
    var col = Array[DType.bool](2)
    col.set_valid(0, True)
    col.set_valid(1, False)
    assert_equal(
        String(named_frame("aaaaaa", AnyArray(col^))),
        String("   aaaaaa\n0    True\n1   False\n\n[2 rows x 1 columns]"),
        "a boolean column",
    )


def test_a_narrow_column_is_elided_by_two_dots_and_a_wide_one_by_three() raises:
    var a = List[Int64]()
    var b = Array[DType.float64](30)
    for i in range(30):
        a.append(Int64(i))
        b.set_valid(i, -Float64(i))
    var columns = List[Series]()
    columns.append(int_column("a", a))
    columns.append(Series("b", b^))
    var lines = String(DataFrame.from_series(columns^)).split("\n")

    assert_equal(String(lines[0]), String("     a     b"), "the header")
    assert_equal(String(lines[6]), String("..  ..   ..."), "the elision")


def test_the_dots_under_the_labels_are_left_aligned() raises:
    var values = List[Int64]()
    var names = List[String]()
    for i in range(12):
        values.append(Int64(i))
        names.append(String("label", i))
    var columns = List[Series]()
    columns.append(int_column("a", values))
    var df = DataFrame.from_series(columns^)
    df.index = text_index(names, None)
    var lines = String(df).split("\n")

    assert_equal(String(lines[1]), String("label0    0"), "the first row")
    assert_equal(String(lines[6]), String("...      .."), "the elision")


def test_the_elided_column_is_four_wide_in_every_row() raises:
    var columns = List[Series]()
    for c in range(10):
        var values = List[Int64]()
        for r in range(6):
            values.append(Int64(r + 1))
        columns.append(int_column(String(c), values))
    var df = DataFrame.from_series(columns^)
    df.index = Index(0, 6, String("idx"))
    var rendered = render_table(
        df.schema,
        df.column_refs(),
        len(df),
        DisplayOptions(max_rows=4, max_columns=6),
        df.index.display_cells(DisplayOptions(max_rows=4, max_columns=6)),
    )
    var lines = rendered.split("\n")

    assert_equal(
        String(lines[0]), String("     0  1  2  ...  7  8  9"), "the header"
    )
    # The dots reach the name row too, which is pandas filling the whole column
    # when it inserts it rather than a statement about the index.
    assert_equal(
        String(lines[1]), String("idx           ...         "), "the name row"
    )
    assert_equal(
        String(lines[4]),
        String("..  .. .. ..  ... .. .. .."),
        "the elided row",
    )


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
