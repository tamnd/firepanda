"""Tests for cutting a CSV buffer into blocks and reading it on every core.

The claim the parallel reader makes is not that it is fast, it is that it gives
the answer one thread would have given. So most of what is here compares the two
directly: the same bytes are read once in blocks and once in a single pass, and
the frames have to match value for value, null for null, and type for type.

The unit tests below that ask for an explicit number of blocks rather than
letting the reader choose, because the reader chooses by core count and a test
whose coverage depends on the machine it runs on is not coverage. The end to end
tests take whatever split the machine gives them, and are written so that they
pass on one core and on thirty two.

The hard case is a newline inside a quoted field. It is a byte that looks exactly
like a row boundary and is not one, the whole split turns on telling them apart,
and the file used here has one in the middle of every block.
"""

from std.collections.span import Span
from std.testing import TestSuite, assert_equal, assert_raises, assert_true

from firepanda.dtype import Field, LogicalType, Schema
from firepanda.io import (
    ReadOptions,
    count_bytes,
    default_dialect,
    read_csv_bytes,
    read_csv_bytes_as,
    row_start_at_or_after,
    scan_blocks,
    scan_csv,
    split_buffer,
)


comptime QUOTE = UInt8(34)
"""The double quote, spelled once so the tests read the same as the reader."""


def add(mut dst: List[UInt8], var text: String):
    """Appends a string's bytes to a buffer the reader can span.

    The files here are built a row at a time and read as a span, which is what
    `bytes_of` in `test_csv.mojo` does for the small ones. This is the same thing
    without holding the whole file as a `String` as well.

    Args:
        dst: The buffer to append to.
        text: The text to append.
    """
    var ptr = text.unsafe_ptr()
    for i in range(text.byte_length()):
        dst.append(ptr.unsafe_offset(i).unsafe_load())


def plain_rows(count: Int) -> List[UInt8]:
    """Builds a file of four columns and no quoting.

    Args:
        count: How many value rows to write.

    Returns:
        The file, header included.
    """
    var dst = List[UInt8]()
    add(dst, "id,pair,score,label\n")
    for i in range(count):
        add(dst, String(i, ",", i * 2, ",", i % 10, ".5,row", i, "\n"))
    return dst^


def quoted_rows(count: Int) -> List[UInt8]:
    """Builds a file whose second column holds a delimiter and a newline.

    Every value row carries a quoted field with a line feed inside it, so a
    splitter that looks for newlines without tracking quotes gets a boundary
    wrong roughly every row rather than once in a while.

    Args:
        count: How many value rows to write.

    Returns:
        The file, header included.
    """
    var dst = List[UInt8]()
    add(dst, "id,note,label\n")
    for i in range(count):
        add(dst, String(i, ',"line one,', i, '\nline two",row', i, "\n"))
    return dst^


def test_one_block_is_the_whole_buffer() raises:
    var text = plain_rows(10)
    var split = split_buffer(Span(text), default_dialect(), 1)
    assert_equal(split.blocks(), 1)
    assert_equal(split.bounds[0], 0)
    assert_equal(split.bounds[1], len(text))


def test_every_boundary_follows_a_newline() raises:
    var text = plain_rows(500)
    var bytes = Span(text)
    var split = split_buffer(bytes, default_dialect(), 7)
    assert_equal(split.blocks(), 7)
    assert_equal(split.bounds[0], 0)
    assert_equal(split.bounds[7], len(text))
    for b in range(1, 7):
        var at = split.bounds[b]
        if at == len(text):
            continue
        assert_true(at > 0, String("boundary ", b, " is at ", at))
        assert_equal(Int(bytes[at - 1]), 10)


def test_the_bounds_never_go_backwards() raises:
    var text = plain_rows(3)
    var split = split_buffer(Span(text), default_dialect(), 16)
    for b in range(1, split.blocks() + 1):
        assert_true(split.bounds[b] >= split.bounds[b - 1])
    assert_equal(split.bounds[split.blocks()], len(text))


def test_the_quote_count_is_the_files_own() raises:
    var text = quoted_rows(40)
    var split = split_buffer(Span(text), default_dialect(), 5)
    assert_equal(split.quotes, count_bytes(Span(text), 0, len(text), QUOTE))


def test_counting_a_byte_matches_a_loop() raises:
    """The counter accumulates in a register and flushes on a schedule, so the
    lengths that matter are the ones either side of a flush and of a register.
    """
    for n in [0, 1, 63, 64, 65, 255, 4096, 16321]:
        var text = String()
        var wanted = 0
        for i in range(n):
            if i % 7 == 0:
                text += '"'
                wanted += 1
            else:
                text += "a"
        assert_equal(count_bytes(text.as_bytes(), 0, n, QUOTE), wanted)


def test_a_row_boundary_is_found_from_outside_a_quote() raises:
    var text = String("ab,cd\nef,gh\n")
    assert_equal(row_start_at_or_after(text.as_bytes(), 0, False, QUOTE), 6)


def test_a_newline_inside_a_quote_is_not_a_boundary() raises:
    var text = String('a,"one\ntwo",b\nnext\n')
    # Byte 4 is inside the quoted field, and the feed at byte 6 is data.
    assert_equal(row_start_at_or_after(text.as_bytes(), 4, True, QUOTE), 14)


def test_a_file_with_no_line_feeds_has_no_boundary() raises:
    var text = String("a,b,c")
    assert_equal(
        row_start_at_or_after(text.as_bytes(), 0, False, QUOTE),
        text.byte_length(),
    )


def test_blocks_find_the_same_fields_as_one_pass() raises:
    var text = plain_rows(300)
    var bytes = Span(text)
    var whole = scan_csv(bytes, default_dialect())
    for parts in [2, 3, 5, 9, 31]:
        var blocks = scan_blocks(bytes, default_dialect(), parts)
        var rows = 0
        for b in range(len(blocks)):
            rows += len(blocks[b])
        assert_equal(rows, len(whole), String(parts, " blocks"))

        var at = 0
        for b in range(len(blocks)):
            for r in range(len(blocks[b])):
                for c in range(blocks[b].width(r)):
                    var mine = blocks[b].at(r, c)
                    var theirs = whole.at(at, c)
                    assert_equal(mine.start, theirs.start)
                    assert_equal(mine.end, theirs.end)
                at += 1


def test_blocks_find_the_same_fields_when_quotes_hold_newlines() raises:
    var text = quoted_rows(200)
    var bytes = Span(text)
    var whole = scan_csv(bytes, default_dialect())
    for parts in [2, 4, 13]:
        var blocks = scan_blocks(bytes, default_dialect(), parts)
        var at = 0
        for b in range(len(blocks)):
            for r in range(len(blocks[b])):
                assert_equal(blocks[b].width(r), whole.width(at))
                for c in range(blocks[b].width(r)):
                    assert_equal(
                        blocks[b].at(r, c).start, whole.at(at, c).start
                    )
                    assert_equal(blocks[b].at(r, c).end, whole.at(at, c).end)
                at += 1
        assert_equal(at, len(whole))


def test_a_bare_quote_makes_the_split_untrustworthy() raises:
    """A quote in the middle of an unquoted field is data, and this reader takes
    it as data. It is also the one thing that breaks the parity the split is
    found by, so the blocks have to notice and hand the file back."""
    var text = List[UInt8]()
    add(text, "a,b\n")
    for i in range(200):
        add(text, String("x", i, ",ok\n") if i != 90 else String('sa"id,ok\n'))
    with assert_raises(contains="not trustworthy"):
        _ = scan_blocks(Span(text), default_dialect(), 4)


def test_a_bare_quote_still_reads_the_file() raises:
    """The untrustworthy split falls back rather than failing, so the value the
    caller gets is the one a single pass would have produced."""
    var text = List[UInt8]()
    add(text, "a,b\n")
    for i in range(60000):
        add(
            text,
            String("x", i, ",ok\n") if i != 40000 else String('sa"id,ok\n'),
        )
    var frame = read_csv_bytes(Span(text), ReadOptions())
    assert_equal(len(frame), 60000)
    assert_equal(frame.column("a").text(40000), 'sa"id')
    assert_equal(frame.column("a").text(0), "x0")


def test_an_unterminated_quote_is_reported_by_the_single_pass() raises:
    """The blocks cannot say which row is wrong, because their rows are their
    own. Falling back is what makes the message name the file's row."""
    var text = List[UInt8]()
    add(text, "a,b\n")
    for i in range(60000):
        add(text, String("x", i, ",ok\n"))
    add(text, 'oops,"never closed\n')
    with assert_raises(contains="that is never closed"):
        _ = read_csv_bytes(Span(text), ReadOptions())


def test_a_large_file_reads_the_same_values() raises:
    var rows = 60000
    var text = plain_rows(rows)
    var frame = read_csv_bytes(Span(text), ReadOptions())
    assert_equal(len(frame), rows)
    assert_equal(frame.width(), 4)
    assert_equal(frame.names()[0], "id")
    assert_equal(frame.names()[3], "label")
    assert_equal(String(frame.column("id").logical()), "int64")
    assert_equal(String(frame.column("score").logical()), "float64")
    assert_equal(String(frame.column("label").logical()), "string")

    var ids = frame.column("id").as_typed[DType.int64]().unsafe_ptr()
    for i in [0, 1, 17, 4095, 32768, 59999]:
        assert_equal(ids.unsafe_offset(i).unsafe_load(), Int64(i))
    assert_equal(frame.column("label").text(0), "row0")
    assert_equal(frame.column("label").text(59999), "row59999")


def test_a_large_file_with_quoted_newlines_reads_the_same_values() raises:
    var rows = 40000
    var text = quoted_rows(rows)
    var frame = read_csv_bytes(Span(text), ReadOptions())
    assert_equal(len(frame), rows)
    assert_equal(frame.width(), 3)
    assert_equal(frame.column("note").text(0), "line one,0\nline two")
    assert_equal(
        frame.column("note").text(rows - 1),
        String("line one,", rows - 1, "\nline two"),
    )
    assert_equal(
        frame.column("label").text(rows // 2), String("row", rows // 2)
    )


def test_a_value_in_the_last_block_still_lifts_the_type() raises:
    """Each block climbs the ladder on its own, so a float that only one block
    ever sees has to move the whole column and not just that block's piece."""
    var text = List[UInt8]()
    add(text, "n\n")
    for i in range(80000):
        add(text, String(i, "\n"))
    add(text, "3.5\n")
    var frame = read_csv_bytes(Span(text), ReadOptions())
    assert_equal(String(frame.column("n").logical()), "float64")
    var values = frame.column("n").as_typed[DType.float64]().unsafe_ptr()
    assert_equal(values.unsafe_offset(80000).unsafe_load(), 3.5)
    assert_equal(values.unsafe_offset(7).unsafe_load(), 7.0)


def test_a_block_of_nothing_but_nulls_does_not_force_text() raises:
    """A block that saw no value at all must not vote for string, which is what
    it would do if an empty block reported the same answer an empty column does.
    """
    var text = List[UInt8]()
    add(text, "n,m\n")
    for i in range(80000):
        if i >= 20000 and i < 60000:
            add(text, String(i, ",\n"))
        else:
            add(text, String(i, ",", i * 3, "\n"))
    var frame = read_csv_bytes(Span(text), ReadOptions())
    assert_equal(String(frame.column("m").logical()), "int64")
    assert_equal(frame.column("m").null_count(), 40000)
    var values = frame.column("m").as_typed[DType.int64]().unsafe_ptr()
    assert_equal(values.unsafe_offset(0).unsafe_load(), 0)
    assert_equal(values.unsafe_offset(79999).unsafe_load(), 79999 * 3)


def test_a_bad_value_names_the_row_it_is_actually_on() raises:
    """The row number a block knows is its own, and the number in the message has
    to be the file's."""
    var text = List[UInt8]()
    add(text, "n\n")
    for i in range(80000):
        add(text, String(i, "\n") if i != 70000 else String("banana\n"))
    var fields = List[Field](capacity=1)
    fields.append(Field("n", LogicalType.INT64))
    with assert_raises(contains="row 70002"):
        _ = read_csv_bytes_as(Span(text), Schema(fields^), ReadOptions())


def test_carriage_returns_split_on_the_feed() raises:
    var text = List[UInt8]()
    add(text, "a,b\r\n")
    for i in range(60000):
        add(text, String(i, ",", i + 1, "\r\n"))
    var frame = read_csv_bytes(Span(text), ReadOptions())
    assert_equal(len(frame), 60000)
    var values = frame.column("b").as_typed[DType.int64]().unsafe_ptr()
    assert_equal(values.unsafe_offset(59999).unsafe_load(), 60000)


def test_a_headerless_file_numbers_its_columns_once() raises:
    """Only block zero has a header, and only when there is one. A block that
    thought its first row was a header would drop one row per block."""
    var text = List[UInt8]()
    for i in range(60000):
        add(text, String(i, ",", i + 1, "\n"))
    var options = ReadOptions(default_dialect(), False, 0)
    var frame = read_csv_bytes(Span(text), options)
    assert_equal(len(frame), 60000)
    assert_equal(frame.names()[0], "column_0")
    var values = frame.column("column_0").as_typed[DType.int64]().unsafe_ptr()
    assert_equal(values.unsafe_offset(0).unsafe_load(), 0)
    assert_equal(values.unsafe_offset(59999).unsafe_load(), 59999)


def test_a_declared_schema_reads_a_large_file() raises:
    var text = plain_rows(60000)
    var fields = List[Field](capacity=4)
    fields.append(Field("id", LogicalType.FLOAT64))
    fields.append(Field("pair", LogicalType.INT64))
    fields.append(Field("score", LogicalType.FLOAT64))
    fields.append(Field("label", LogicalType.STRING))
    var frame = read_csv_bytes_as(Span(text), Schema(fields^), ReadOptions())
    assert_equal(len(frame), 60000)
    assert_equal(String(frame.column("id").logical()), "float64")
    var values = frame.column("id").as_typed[DType.float64]().unsafe_ptr()
    assert_equal(values.unsafe_offset(59999).unsafe_load(), 59999.0)


def test_a_bounded_sample_still_stops_where_it_says() raises:
    """The sample is the file's first rows and not each block's first rows, so a
    bound of a hundred has to leave every block after the first looking at
    nothing rather than looking at its own first hundred."""
    var text = List[UInt8]()
    add(text, "n\n")
    for i in range(80000):
        add(text, String(i, "\n"))
    add(text, "3.5\n")
    var options = ReadOptions(default_dialect(), True, 100)
    with assert_raises(contains="row 80002 is not an integer"):
        _ = read_csv_bytes(Span(text), options)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
