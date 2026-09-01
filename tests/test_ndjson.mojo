"""Tests for the newline delimited JSON reader.

Most of these are about the two things a JSON reader has that a CSV reader does
not: a line is allowed to leave a key out, and a value says what kind of thing
it is rather than being guessed at. So the interesting cases are ragged lines,
mixed kinds in one column, and the places where a JSON null and a missing key
have to mean the same thing.

The big file at the bottom is here because the reader splits a buffer into one
block per core and a bug in the merge only shows up once there is more than one
block. A test that only ever reads six lines would not touch that code at all.
"""

from std.os import makedirs
from std.testing import (
    TestSuite,
    assert_equal,
    assert_false,
    assert_raises,
    assert_true,
)

from firepanda.dtype import LogicalType
from firepanda.frame import DataFrame
from firepanda.io.ndjson import (
    NdjsonOptions,
    line_bounds,
    read_ndjson,
    read_ndjson_bytes,
    scan_lines,
)

comptime DIR = "/tmp/firepanda_ndjson"


def _bytes(text: StringSlice) -> List[UInt8]:
    var out = List[UInt8](capacity=text.byte_length())
    for byte in text.as_bytes():
        out.append(byte)
    return out^


def _add(mut out: List[UInt8], text: StringSlice):
    # A line at a time onto the buffer the reader will span, rather than onto a
    # `String` that then has to be copied into one.
    for byte in text.as_bytes():
        out.append(byte)


def _read(text: StringSlice) raises -> DataFrame:
    var raw = _bytes(text)
    return read_ndjson_bytes(Span(raw))


def _put(path: String, text: StringSlice) raises:
    makedirs(DIR, exist_ok=True)
    var handle = open(path, "w")
    handle.write(text)
    handle.close()


def test_one_object_per_line_becomes_one_row_per_line() raises:
    var frame = _read(
        '{"a": 1, "b": "x"}\n{"a": 2, "b": "y"}\n{"a": 3, "b": "z"}\n'
    )
    assert_equal(len(frame), 3)
    assert_equal(frame.width(), 2)
    assert_equal(frame.schema[0].name, "a")
    assert_equal(frame.schema[1].name, "b")


def test_a_column_of_whole_numbers_is_an_integer_column() raises:
    var frame = _read('{"a": 1}\n{"a": -2}\n{"a": 300}\n')
    assert_true(frame.schema[0].dtype == LogicalType.INT64)
    var column = frame[0].as_typed[DType.int64]()
    assert_equal(column[0], 1)
    assert_equal(column[1], -2)
    assert_equal(column[2], 300)


def test_a_number_written_with_a_point_is_a_float_column() raises:
    # The value happens to be whole and the file still said it was a float, so
    # reading it as an integer would be reading something the file did not say.
    var frame = _read('{"a": 1.0}\n{"a": 2}\n')
    assert_true(frame.schema[0].dtype == LogicalType.FLOAT64)
    var column = frame[0].as_typed[DType.float64]()
    assert_equal(column[0], 1.0)
    assert_equal(column[1], 2.0)


def test_an_exponent_is_a_float_too() raises:
    var frame = _read('{"a": 1e3}\n{"a": 2}\n')
    assert_true(frame.schema[0].dtype == LogicalType.FLOAT64)
    var column = frame[0].as_typed[DType.float64]()
    assert_equal(column[0], 1000.0)


def test_true_and_false_make_a_boolean_column() raises:
    var frame = _read('{"a": true}\n{"a": false}\n')
    assert_true(frame.schema[0].dtype == LogicalType.BOOL)
    var column = frame[0].as_typed[DType.bool]()
    assert_true(column[0])
    assert_false(column[1])


def test_a_missing_key_is_a_null() raises:
    # The whole difference from CSV. A line that does not mention a column is an
    # ordinary line, not a ragged row.
    var frame = _read('{"a": 1, "b": 2}\n{"a": 3}\n{"b": 4}\n')
    assert_equal(len(frame), 3)
    var a = frame[0].as_typed[DType.int64]()
    var b = frame[1].as_typed[DType.int64]()
    assert_true(a.is_valid(0))
    assert_true(a.is_valid(1))
    assert_false(a.is_valid(2))
    assert_true(b.is_valid(0))
    assert_false(b.is_valid(1))
    assert_true(b.is_valid(2))
    assert_equal(b[2], 4)


def test_an_explicit_null_is_the_same_as_a_missing_key() raises:
    var frame = _read('{"a": 1}\n{"a": null}\n')
    assert_true(frame.schema[0].dtype == LogicalType.INT64)
    var column = frame[0].as_typed[DType.int64]()
    assert_true(column.is_valid(0))
    assert_false(column.is_valid(1))


def test_a_column_of_nothing_but_nulls_is_a_string_column() raises:
    # A string column of nulls can be cast to anything later. A float column of
    # nulls has already thrown away the text that would have said what it was.
    var frame = _read('{"a": null}\n{"a": null}\n')
    assert_true(frame.schema[0].dtype == LogicalType.STRING)


def test_a_column_appears_where_the_file_first_mentions_it() raises:
    var frame = _read('{"b": 1}\n{"a": 2, "b": 3}\n{"c": 4}\n')
    assert_equal(frame.width(), 3)
    assert_equal(frame.schema[0].name, "b")
    assert_equal(frame.schema[1].name, "a")
    assert_equal(frame.schema[2].name, "c")


def test_mixed_kinds_in_one_column_fall_to_text() raises:
    # Every other answer loses something. Text is the only type that holds a
    # number, a boolean and a word at once.
    var frame = _read('{"a": 1}\n{"a": "two"}\n{"a": true}\n')
    assert_true(frame.schema[0].dtype == LogicalType.STRING)
    ref column = frame[0].strings()
    assert_equal(column[0], "1")
    assert_equal(column[1], "two")
    assert_equal(column[2], "true")


def test_a_nested_value_is_kept_as_the_text_it_was_written_as() raises:
    var frame = _read('{"a": {"b": 1}}\n{"a": [1, 2]}\n')
    assert_true(frame.schema[0].dtype == LogicalType.STRING)
    ref column = frame[0].strings()
    assert_equal(column[0], '{"b": 1}')
    assert_equal(column[1], "[1, 2]")


def test_a_string_with_escapes_comes_out_unescaped() raises:
    var frame = _read('{"a": "one\\ttwo"}\n{"a": "\\u00e9"}\n')
    ref column = frame[0].strings()
    assert_equal(column[0], "one\ttwo")
    assert_equal(column[1], "é")


def test_blank_lines_are_not_rows() raises:
    var frame = _read('{"a": 1}\n\n\n{"a": 2}\n')
    assert_equal(len(frame), 2)


def test_a_file_with_no_trailing_newline_still_reads_its_last_line() raises:
    var frame = _read('{"a": 1}\n{"a": 2}')
    assert_equal(len(frame), 2)


def test_an_empty_buffer_is_an_empty_frame() raises:
    var frame = _read("")
    assert_equal(len(frame), 0)
    assert_equal(frame.width(), 0)


def test_an_object_with_no_members_is_a_row_of_nothing() raises:
    var frame = _read('{}\n{"a": 1}\n')
    assert_equal(len(frame), 2)
    assert_equal(frame.width(), 1)
    var column = frame[0].as_typed[DType.int64]()
    assert_false(column.is_valid(0))
    assert_equal(column[1], 1)


def test_a_repeated_key_takes_the_first_one() raises:
    # Which one wins has to be decided somewhere and first is the one that costs
    # nothing, because the fill stops at the first match.
    var frame = _read('{"a": 1, "a": 2}\n')
    var column = frame[0].as_typed[DType.int64]()
    assert_equal(column[0], 1)


def test_a_key_with_an_escape_in_it_is_the_key_it_spells() raises:
    var frame = _read('{"a\\tb": 1}\n{"a\\tb": 2}\n')
    assert_equal(frame.width(), 1)
    assert_equal(frame.schema[0].name, "a\tb")


def test_a_bound_on_the_sample_is_a_bound_on_what_gets_looked_at() raises:
    # A key that first turns up after the bound is not a column, which is the
    # promise the option makes. Reading it anyway would make the option a lie.
    var options = NdjsonOptions(2)
    var raw = _bytes('{"a": 1}\n{"a": 2}\n{"a": 3, "b": 4}\n')
    var frame = read_ndjson_bytes(Span(raw), options)
    assert_equal(frame.width(), 1)
    assert_equal(len(frame), 3)


def test_a_line_that_is_not_an_object_is_an_error() raises:
    with assert_raises(contains="object"):
        _ = _read('{"a": 1}\n[1, 2]\n')


def test_a_line_that_never_closes_is_an_error() raises:
    with assert_raises(contains="closes"):
        _ = _read('{"a": 1\n')


def test_reading_a_file_from_disk_gives_the_same_frame() raises:
    var path = String(DIR, "/basic.ndjson")
    _put(path, '{"a": 1, "b": "x"}\n{"a": 2, "b": "y"}\n')
    var frame = read_ndjson(path)
    assert_equal(len(frame), 2)
    assert_equal(frame.width(), 2)
    var a = frame[0].as_typed[DType.int64]()
    assert_equal(a[1], 2)
    ref b = frame[1].strings()
    assert_equal(b[0], "x")


def test_a_file_that_is_not_there_is_an_error() raises:
    with assert_raises():
        _ = read_ndjson(String(DIR, "/nothing_here.ndjson"))


def test_an_empty_file_is_an_empty_frame() raises:
    var path = String(DIR, "/empty.ndjson")
    _put(path, "")
    var frame = read_ndjson(path)
    assert_equal(len(frame), 0)


def test_cutting_a_buffer_into_blocks_lands_on_line_starts() raises:
    var raw = _bytes('{"a": 1}\n{"a": 2}\n{"a": 3}\n{"a": 4}\n')
    var bounds = line_bounds(Span(raw), 4)
    assert_equal(len(bounds), 5)
    assert_equal(bounds[0], 0)
    assert_equal(bounds[4], len(raw))
    for i in range(1, 5):
        assert_true(bounds[i] >= bounds[i - 1])
    for i in range(1, 4):
        if bounds[i] > 0 and bounds[i] < len(raw):
            assert_equal(raw[bounds[i] - 1], UInt8(10))


def test_a_newline_inside_a_string_would_be_illegal_so_a_brace_is_safe() raises:
    # A JSON string cannot hold a literal newline, so the split needs no quote
    # state and a brace or a comma inside a string cannot confuse it.
    var frame = _read('{"a": "one, two}"}\n{"a": "{three}"}\n')
    ref column = frame[0].strings()
    assert_equal(column[0], "one, two}")
    assert_equal(column[1], "{three}")


def test_scanning_one_block_reports_its_lines_and_their_members() raises:
    var raw = _bytes('{"a": 1}\n{"a": 2, "b": 3}\n')
    var lines = scan_lines(Span(raw), 0, len(raw))
    assert_equal(len(lines), 2)
    assert_equal(len(lines.members), 3)
    assert_equal(lines.starts[0], 0)
    assert_equal(lines.starts[1], 1)
    assert_equal(lines.starts[2], 3)


def test_a_file_big_enough_to_split_reads_the_same_as_a_small_one() raises:
    # The blocks are what makes the read parallel and every bug in the merge is
    # invisible below the threshold, so this one is deliberately over it.
    var rows = 60000
    var raw = List[UInt8]()
    for i in range(rows):
        _add(
            raw,
            String(
                '{"id": ', i, ', "value": ', i * 2, ', "name": "row', i, '"}\n'
            ),
        )
    var frame = read_ndjson_bytes(Span(raw))
    assert_equal(len(frame), rows)
    assert_equal(frame.width(), 3)
    var ids = frame[0].as_typed[DType.int64]()
    var values = frame[1].as_typed[DType.int64]()
    ref names = frame[2].strings()
    assert_equal(ids[0], 0)
    assert_equal(ids[rows - 1], Int64(rows - 1))
    assert_equal(values[rows - 1], Int64((rows - 1) * 2))
    assert_equal(names[rows - 1], String("row", rows - 1))
    var total = 0
    for i in range(rows):
        total += Int(ids[i])
    assert_equal(total, rows * (rows - 1) // 2)


def test_a_big_file_with_ragged_lines_lines_its_nulls_up() raises:
    # Two blocks that disagree about which columns exist is the case the merge
    # is for, and every third line here leaves a key out.
    var rows = 60000
    var raw = List[UInt8]()
    for i in range(rows):
        if i % 3 == 0:
            _add(raw, String('{"a": ', i, "}\n"))
        else:
            _add(raw, String('{"a": ', i, ', "b": ', i, "}\n"))
    var frame = read_ndjson_bytes(Span(raw))
    assert_equal(len(frame), rows)
    assert_equal(frame.width(), 2)
    var b = frame[1].as_typed[DType.int64]()
    var nulls = 0
    for i in range(rows):
        if not b.is_valid(i):
            nulls += 1
        else:
            assert_equal(b[i], Int64(i))
    assert_equal(nulls, (rows + 2) // 3)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
