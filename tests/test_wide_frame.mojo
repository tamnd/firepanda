"""Tests for the shapes a hundred and five column frame makes.

The widest thing anything else in this repository points at is TPC-H's
`lineitem`, which is sixteen columns. The ClickBench hits table is 105, and
almost every query over it reads three of them, so the operations that are linear
in the width of the frame are the ones that decide what a query costs before a
kernel runs at all.

Nothing here is a timing. These are the shapes the width changes, asserted on a
frame wide enough that a projection which quietly touched every column would be
doing thirty five times the work it should, and none of them would fail on the
three column frames the rest of the tests use.

The binary tests are the other half of the same table. Every text column in the
hits file is a bare `BYTE_ARRAY` with no string logical type on it, so the honest
Arrow reading of all of them is binary, and a suite that measured `LIKE` over
binary would be measuring nothing.

The clock tests are the third of the same. `EventTime` is an int64 count of
seconds with nothing in the file saying it is a time, so it arrives as a number
and every query that reads a minute or truncates to one needs it labelled. That
relabel has to be free, which is what one of these tests measures by address
rather than by argument.
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
from firepanda.array.strings import strings_from_list
from firepanda.dtype.logical import LogicalType
from firepanda.dtype.temporal import TimeUnit
from firepanda.frame.frame import DataFrame
from firepanda.frame.series import Series


def wide_frame(width: Int, rows: Int) raises -> DataFrame:
    """Builds a frame of `width` int64 columns named `c0` through `c<width-1>`.

    Args:
        width: How many columns.
        rows: How many rows each column has.

    Returns:
        The frame, with `c<j>` holding `j * 1000 + i` at row `i`.
    """
    var columns = List[Series]()
    for j in range(width):
        var col = Array[DType.int64](rows)
        for i in range(rows):
            col.set_valid(i, Int64(j * 1000 + i))
        columns.append(Series("c" + String(j), col^))
    return DataFrame.from_series(columns^)


def text_column(name: String, values: List[String]) raises -> Series:
    """Builds a fully valid text series."""
    return Series(name, strings_from_list(values))


def binary_column(name: String, values: List[String]) raises -> Series:
    """Builds a series whose bytes are text and whose type says binary.

    That is what a `BYTE_ARRAY` column with no string logical type on it looks
    like once it is through the Arrow import: the right bytes under a type that
    makes no promise about them.
    """
    var col = AnyArray(strings_from_list(values))
    return Series(name, col^.retyped(LogicalType.BINARY))


def urls() -> List[String]:
    """Eight rows of the kind of thing the `URL` column holds."""
    return [
        String("http://google.com/search?q=mojo"),
        String("http://example.com/"),
        String("http://www.google.co.uk/maps"),
        String(""),
        String("http://news.example.com/google"),
        String("http://example.org/a"),
        String("http://google.com/"),
        String("http://example.net/b"),
    ]


def test_a_projection_of_three_columns_out_of_a_hundred_and_five() raises:
    var frame = wide_frame(105, 4)
    assert_equal(frame.width(), 105)

    var picked = frame.select(["c104", "c7", "c0"])
    assert_equal(picked.width(), 3)
    assert_equal(len(picked), 4)
    assert_equal(picked.schema[0].name, "c104")
    assert_equal(picked.schema[1].name, "c7")
    assert_equal(picked.schema[2].name, "c0")
    assert_equal(
        picked.column("c104")
        .values.unsafe_ptr[DType.int64]()
        .unsafe_offset(0)[],
        104000,
    )
    assert_equal(
        picked.column("c7").values.unsafe_ptr[DType.int64]().unsafe_offset(3)[],
        7003,
    )


def test_dropping_a_hundred_and_two_columns_keeps_the_other_three() raises:
    var frame = wide_frame(105, 2)
    var going = List[String]()
    for j in range(105):
        if j != 0 and j != 50 and j != 104:
            going.append("c" + String(j))
    var kept = frame.drop(going)
    assert_equal(kept.width(), 3)
    assert_equal(kept.schema[0].name, "c0")
    assert_equal(kept.schema[1].name, "c50")
    assert_equal(kept.schema[2].name, "c104")


def test_a_projection_still_refuses_a_name_it_does_not_have() raises:
    # The bulk lookup has its own error path and it has to say what the scan
    # said, because a wide frame is exactly where a typo in a column name is
    # most likely and least visible.
    var frame = wide_frame(105, 2)
    with assert_raises(contains="c105"):
        _ = frame.select(["c0", "c105"])


def test_a_projection_still_refuses_a_name_twice() raises:
    var frame = wide_frame(105, 2)
    with assert_raises(contains="twice"):
        _ = frame.select(["c3", "c9", "c3"])


def test_a_wide_projection_keeps_every_column_it_asked_for() raises:
    # The whole width, reversed. A lookup that fell back to position order, or
    # that hashed the names into the wrong slots, answers this one wrong rather
    # than raising.
    var frame = wide_frame(105, 3)
    var names = List[String]()
    for j in range(105):
        names.append("c" + String(104 - j))
    var reversed = frame.select(names)
    assert_equal(reversed.width(), 105)
    for j in range(105):
        assert_equal(reversed.schema[j].name, "c" + String(104 - j))
        assert_equal(
            reversed.column("c" + String(104 - j))
            .values.unsafe_ptr[DType.int64]()
            .unsafe_offset(1)[],
            Int64((104 - j) * 1000 + 1),
        )


def test_binary_columns_become_text_and_the_rest_are_left_alone() raises:
    var columns = List[Series]()
    columns.append(binary_column("url", urls()))
    columns.append(text_column("title", urls()))
    var counter = Array[DType.int64](8)
    for i in range(8):
        counter.set_valid(i, Int64(i))
    columns.append(Series("counter", counter^))
    var frame = DataFrame.from_series(columns^)

    assert_equal(frame.schema[0].dtype, LogicalType.BINARY)
    var text = frame.text_from_binary()
    assert_equal(text.schema[0].dtype, LogicalType.STRING)
    assert_equal(text.schema[1].dtype, LogicalType.STRING)
    assert_equal(text.schema[2].dtype, LogicalType.INT64)
    assert_equal(text.schema[0].name, "url")
    assert_equal(len(text), 8)


def test_a_substring_match_over_a_relabelled_column_finds_rows() raises:
    # The check the ClickBench port needs. Four of these eight rows hold
    # "google", and a relabel that lost the payload or shifted the views by a
    # byte answers this with zero rows rather than with an error.
    var columns = List[Series]()
    columns.append(binary_column("url", urls()))
    var frame = DataFrame.from_series(columns^).text_from_binary()

    var mask = frame.column("url").str_contains("google")
    assert_equal(len(mask), 8)
    var hits = 0
    for i in range(8):
        if mask[i]:
            hits += 1
    assert_equal(hits, 4)
    assert_true(mask[0])
    assert_false(mask[1])
    assert_true(mask[2])
    assert_false(mask[3])


def test_relabelling_binary_does_not_change_the_values() raises:
    var columns = List[Series]()
    columns.append(binary_column("url", urls()))
    var frame = DataFrame.from_series(columns^).text_from_binary()
    var out = frame.column("url")
    var expected = urls()
    for i in range(8):
        assert_equal(out.values.strings()[i], expected[i])


def test_a_frame_with_no_binary_column_comes_back_unchanged() raises:
    var frame = wide_frame(4, 2)
    var same = frame.text_from_binary()
    assert_equal(same.width(), 4)
    assert_true(same.schema == frame.schema)


def test_an_empty_string_is_still_an_empty_string_after_the_relabel() raises:
    # The hits table has no nulls in it at all. The empty string is what a
    # missing value looks like, and eleven of the queries filter on `<> ''`, so
    # a relabel that turned an empty value into a null would change the answer
    # to those and to nothing else.
    var columns = List[Series]()
    columns.append(binary_column("url", urls()))
    var frame = DataFrame.from_series(columns^).text_from_binary()
    var column = frame.column("url")
    assert_equal(column.values.null_count(), 0)
    assert_equal(column.values.strings().byte_length(3), 0)


def clock_column(name: String, values: List[Int64]) raises -> Series:
    """Builds an int64 column, which is how a clock arrives out of the file."""
    var out = Array[DType.int64](len(values))
    for i in range(len(values)):
        out.set_valid(i, values[i])
    return Series(name, out^)


def seconds() -> List[Int64]:
    """Four instants in July 2013, a leap day, and one before the epoch."""
    return [
        Int64(1372636800),
        Int64(1372636859),
        Int64(1372636860),
        Int64(1375228799),
        Int64(1078012800),
        Int64(-1),
    ]


def test_an_integer_column_is_relabelled_as_a_clock() raises:
    var columns = List[Series]()
    columns.append(clock_column("EventTime", seconds()))
    var counter = Array[DType.int64](6)
    for i in range(6):
        counter.set_valid(i, Int64(i))
    columns.append(Series("counter", counter^))
    var frame = DataFrame.from_series(columns^)

    assert_equal(frame.schema[0].dtype, LogicalType.INT64)
    var clocked = frame.timestamps_from_integers(["EventTime"], TimeUnit.SECOND)
    assert_equal(
        clocked.schema[0].dtype, LogicalType.timestamp(TimeUnit.SECOND)
    )
    assert_equal(clocked.schema[0].name, "EventTime")
    # The column nobody named keeps the type it had, which is the same type the
    # relabelled one is laid out as, so a loop that relabelled by layout rather
    # than by name would turn a row counter into a clock and nothing would say
    # so until the answers came out in 1970.
    assert_equal(clocked.schema[1].dtype, LogicalType.INT64)
    assert_equal(len(clocked), 6)


def test_the_relabel_reads_none_of_the_values() raises:
    """The whole reason this exists rather than `numbers_to_timestamps`.

    `EventTime` at a hundred million rows is eight hundred megabytes, and a
    conversion that copied it would cost more than most of the queries that read
    it. The evidence is the address of the values buffer, since a copy cannot
    give back the one it was handed.
    """
    var columns = List[Series]()
    columns.append(clock_column("EventTime", seconds()))
    var frame = DataFrame.from_series(columns^)
    var before = Int(frame.column("EventTime").values.data.values.unsafe_ptr())

    var clocked = frame.timestamps_from_integers(["EventTime"], TimeUnit.SECOND)
    var after = Int(clocked.column("EventTime").values.data.values.unsafe_ptr())
    assert_equal(after, before)


def test_the_instants_read_back_as_the_counts_they_were() raises:
    var columns = List[Series]()
    columns.append(clock_column("EventTime", seconds()))
    var frame = DataFrame.from_series(columns^).timestamps_from_integers(
        ["EventTime"], TimeUnit.SECOND
    )
    var out = frame.column("EventTime").values.as_typed[DType.int64]()
    var expected = seconds()
    for i in range(len(expected)):
        assert_equal(out[i], expected[i])


def test_a_narrow_integer_is_refused_rather_than_widened() raises:
    """`EventDate` in the same file is a uint16 count of days.

    Widening it is a pass over the data, and a method whose whole claim is that
    it is free has no business doing one quietly. The message says to cast.
    """
    var days = Array[DType.uint16](4)
    for i in range(4):
        days.set_valid(i, UInt16(15887 + i))
    var columns = List[Series]()
    columns.append(Series("EventDate", days^))
    var frame = DataFrame.from_series(columns^)

    with assert_raises(contains="laid out as int64"):
        _ = frame.timestamps_from_integers(["EventDate"], TimeUnit.SECOND)


def test_a_name_the_frame_does_not_have_is_refused() raises:
    var frame = wide_frame(4, 2)
    with assert_raises():
        _ = frame.timestamps_from_integers(["EventTime"], TimeUnit.SECOND)


def test_two_columns_are_relabelled_in_one_call() raises:
    var columns = List[Series]()
    columns.append(clock_column("EventTime", seconds()))
    columns.append(clock_column("ClientEventTime", seconds()))
    columns.append(clock_column("plain", seconds()))
    var frame = DataFrame.from_series(columns^).timestamps_from_integers(
        ["EventTime", "ClientEventTime"], TimeUnit.SECOND
    )
    var clock = LogicalType.timestamp(TimeUnit.SECOND)
    assert_equal(frame.schema[0].dtype, clock)
    assert_equal(frame.schema[1].dtype, clock)
    assert_equal(frame.schema[2].dtype, LogicalType.INT64)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
