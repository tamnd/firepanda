"""Tests for the memory mapped file, and for reading a CSV off disk.

Everything else in the reader is tested against a span of bytes that a test
built in memory, which is the right way to test a parser and the wrong way to
find out whether the file ever arrives. These tests write a real file and read
it back, because mapping is the one part of the reader that a byte span cannot
exercise at all.

Two of the three failure paths are tested here rather than in a comment: a file
that does not exist, and a file with nothing in it. Both have to come back as
`None` from `map_file` rather than as an exception, because `read_csv` treats
that as its signal to fall back, and a file with nothing in it is the ordinary
case that `mmap` rejects.
"""

from std.testing import (
    TestSuite,
    assert_equal,
    assert_false,
    assert_raises,
    assert_true,
)

from firepanda.dtype import Field, LogicalType, Schema
from firepanda.io import (
    ReadOptions,
    map_file,
    read_csv,
    read_csv_as,
    read_csv_bytes,
)


def write_file(path: String, text: String) raises:
    """Writes a file, replacing whatever was there.

    Args:
        path: Where to write.
        text: What to write.
    """
    var handle = open(path, "w")
    handle.write(text)
    handle.close()


def test_a_mapping_holds_the_bytes_that_are_in_the_file() raises:
    """The whole point, and the only test that compares byte for byte."""
    var path = String("/tmp/firepanda-mapped-bytes.txt")
    var text = String("the quick brown fox,1,2,3\n")
    write_file(path, text)
    var mapped = map_file(path)
    assert_equal(Bool(mapped), True)
    ref file = mapped.value()
    assert_equal(len(file), text.byte_length())
    var bytes = file.bytes()
    var source = text.as_bytes()
    for i in range(text.byte_length()):
        assert_equal(bytes[i], source[i])


def test_a_mapping_survives_a_file_larger_than_one_page() raises:
    """A one page file would not notice an offset mistake past the first page.
    """
    var path = String("/tmp/firepanda-mapped-pages.txt")
    var line = String("0123456789abcdef0123456789abcdef0123456789abcdef\n")
    var text = String()
    for _ in range(400):
        text += line
    write_file(path, text)
    var mapped = map_file(path)
    assert_equal(Bool(mapped), True)
    ref file = mapped.value()
    assert_equal(len(file), text.byte_length())
    var bytes = file.bytes()
    var source = text.as_bytes()
    # The last byte of the last page is the one an off by a page would miss.
    assert_equal(bytes[text.byte_length() - 1], source[text.byte_length() - 1])
    assert_equal(bytes[text.byte_length() - 2], source[text.byte_length() - 2])


def test_a_file_that_is_not_there_maps_to_nothing() raises:
    """The failure has to be a value, because `read_csv` branches on it."""
    assert_false(Bool(map_file(String("/tmp/firepanda-no-such-file-here"))))


def test_an_empty_file_maps_to_nothing() raises:
    """`mmap` rejects a length of zero, and an empty file is not exotic."""
    var path = String("/tmp/firepanda-mapped-empty.txt")
    write_file(path, String(""))
    assert_false(Bool(map_file(path)))


def test_reading_a_file_agrees_with_reading_its_bytes() raises:
    """The mapped path and the in memory path are the same read."""
    var text = String("a,b,c\n1,2.5,x\n3,4.5,yy\n5,6.5,zzz\n")
    var path = String("/tmp/firepanda-mapped-read.csv")
    write_file(path, text)
    var from_disk = read_csv(path)
    var from_bytes = read_csv_bytes(text.as_bytes(), ReadOptions())
    assert_equal(len(from_disk), len(from_bytes))
    assert_equal(from_disk.width(), from_bytes.width())
    for c in range(from_disk.width()):
        assert_equal(from_disk.names()[c], from_bytes.names()[c])
    assert_true(from_disk.schema[0].dtype == LogicalType.INT64)
    assert_equal(from_disk[0].as_typed[DType.int64]()[2], Int64(5))
    assert_equal(from_disk[1].as_typed[DType.float64]()[2], Float64(6.5))
    assert_equal(from_disk[2].strings()[2], "zzz")


def test_reading_a_file_as_a_declared_schema() raises:
    """A declared schema skips inference and can widen a column on the way in.

    The first column holds integers and is asked for as a float, which is the
    thing a caller cannot get out of inference at all, and the last is asked
    for as text so that a column of digits stays digits.
    """
    var path = String("/tmp/firepanda-mapped-typed.csv")
    write_file(path, String("a,b\n1,10\n2,20\n3,30\n"))
    var fields = List[Field]()
    fields.append(Field("a", LogicalType.FLOAT64))
    fields.append(Field("b", LogicalType.STRING))
    var frame = read_csv_as(path, Schema(fields^))
    assert_equal(len(frame), 3)
    assert_true(frame.schema[0].dtype == LogicalType.FLOAT64)
    assert_true(frame.schema[1].dtype == LogicalType.STRING)
    assert_equal(frame[0].as_typed[DType.float64]()[2], Float64(3.0))
    assert_equal(frame[1].strings()[2], "30")


def test_a_declared_schema_that_does_not_fit_the_file_is_refused() raises:
    """The width check has to happen on the mapped path too."""
    var path = String("/tmp/firepanda-mapped-typed-wrong.csv")
    write_file(path, String("a,b\n1,10\n"))
    var fields = List[Field]()
    fields.append(Field("a", LogicalType.INT64))
    with assert_raises():
        _ = read_csv_as(path, Schema(fields^))


def test_reading_an_empty_file_goes_through_the_fallback() raises:
    """An empty file is the one ordinary file that cannot be mapped.

    It has to read as the empty frame, which is what reading empty bytes gives,
    rather than failing on an errno from a mapping that was never going to
    work.
    """
    var path = String("/tmp/firepanda-mapped-empty-read.csv")
    write_file(path, String(""))
    var frame = read_csv(path)
    assert_equal(frame.width(), 0)
    assert_equal(len(frame), 0)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
