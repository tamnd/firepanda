"""Tests for the Parquet reader.

The fixture is a Parquet file pyarrow 25 wrote, checked in as the bytes it
produced and written back out to a temporary file by the first test that needs
it. A reader is only worth anything if it reads what somebody else wrote, and a
file this code also produced would agree with itself about any misunderstanding
it happens to have.

Five columns and six rows, which is enough to cover an int64, an int32, a
double, a string and a bool, a null in the middle of four of them, an empty
string next to a long one, and a column with no nulls at all so the reader has
to notice the difference. The file is snappy compressed, which is what pyarrow
writes by default and therefore what most Parquet in the world is.

A Hive partitioned directory is built by the tests that read one, two copies
of the same file under `year=/month=` directories. Identical files are the point
rather than a shortcut: everything that tells the two halves of that result apart
is in the paths and nowhere in the bytes. DuckDB finds a `key=value` directory by
itself, so the interesting default is the one that turns that off, and the
partition columns come back after the file's own columns sorted by name rather
than in the order the path visits them.

There is no test here for a file of several row groups, because DuckDB hands
back chunks of two thousand rows and a fixture that large cannot be checked in
as hex. The glob test covers the same code path from the other side: two files
read as one frame is two batches through the same assembler.

These tests need libduckdb, which pixi.toml puts in the development environment.
Without it every one of them fails at the first call with a message saying so,
which is the intended behaviour and not a broken test.
"""

from std.os import makedirs
from std.testing import (
    TestSuite,
    assert_equal,
    assert_false,
    assert_raises,
    assert_true,
)

from firepanda.dtype.logical import LogicalType
from firepanda.io.parquet import ParquetOptions, quote, read_parquet

comptime SINGLE = "/tmp/firepanda_parquet_one.parquet"
"""Where the fixture is written for the tests that read one file."""

comptime FIRST = "/tmp/firepanda_parquet_glob_a.parquet"
"""One of the two files the glob test reads."""

comptime SECOND = "/tmp/firepanda_parquet_glob_b.parquet"
"""The other."""

comptime PATTERN = "/tmp/firepanda_parquet_glob_*.parquet"
"""What matches both of them and nothing else."""

comptime HIVE = "/tmp/firepanda_parquet_hive"
"""The root of the partitioned directory the dataset tests read."""

comptime HIVE_GLOB = "/tmp/firepanda_parquet_hive/*/*/*.parquet"
"""What matches every file under it, which is what a scan is pointed at."""


def _nibble(byte: UInt8) raises -> Int:
    var value = Int(byte)
    if value >= ord("0") and value <= ord("9"):
        return value - ord("0")
    if value >= ord("a") and value <= ord("f"):
        return value - ord("a") + 10
    raise Error(String("not a hex digit: ", value))


def _from_hex(text: StringSlice) raises -> List[UInt8]:
    """Decodes a hex string into bytes, so a wire capture can be checked in."""
    var digits = text.as_bytes()
    var out = List[UInt8]()
    for i in range(0, len(digits), 2):
        out.append(UInt8(_nibble(digits[i]) * 16 + _nibble(digits[i + 1])))
    return out^


def _put(path: StringSlice) raises:
    """Writes the fixture to a path, replacing whatever was there.

    Args:
        path: Where to write it.

    Raises:
        Error: If the file cannot be written.
    """
    var bytes = _from_hex(_FIXTURE)
    var handle = open(String(path), "w")
    handle.write_bytes(Span(bytes))
    handle.close()


def test_a_parquet_file_reads_with_its_schema() raises:
    _put(SINGLE)
    var frame = read_parquet(SINGLE)
    assert_equal(len(frame), 6)
    assert_equal(frame.width(), 5)
    assert_equal(frame.names()[0], "id")
    assert_equal(frame.names()[1], "small")
    assert_equal(frame.names()[2], "value")
    assert_equal(frame.names()[3], "label")
    assert_equal(frame.names()[4], "flag")
    assert_true(frame.schema[0].dtype == LogicalType.INT64)
    assert_true(frame.schema[1].dtype == LogicalType.INT32)
    assert_true(frame.schema[2].dtype == LogicalType.FLOAT64)
    assert_true(frame.schema[3].dtype == LogicalType.STRING)
    assert_true(frame.schema[4].dtype == LogicalType.BOOL)


def test_the_numbers_come_back_the_way_they_went_in() raises:
    _put(SINGLE)
    var frame = read_parquet(SINGLE)
    var ids = frame[0].as_typed[DType.int64]()
    for i in range(6):
        assert_equal(ids[i], Int64(i + 1))
    assert_equal(frame[0].null_count(), 0)

    var small = frame[1].as_typed[DType.int32]()
    assert_equal(small[0], Int32(10))
    assert_equal(small[1], Int32(20))
    assert_false(frame[1].is_valid(2))
    assert_equal(small[3], Int32(40))
    assert_equal(frame[1].null_count(), 1)


def test_a_double_column_keeps_its_null_and_its_extremes() raises:
    _put(SINGLE)
    var frame = read_parquet(SINGLE)
    var values = frame[2].as_typed[DType.float64]()
    assert_equal(values[0], Float64(1.5))
    assert_equal(values[1], Float64(-2.25))
    assert_false(frame[2].is_valid(2))
    assert_equal(values[3], Float64(0.0))
    assert_equal(values[4], Float64(1e10))
    assert_equal(values[5], Float64(-0.5))
    assert_equal(frame[2].null_count(), 1)


def test_a_string_column_reads_empty_null_and_long() raises:
    # An empty string and a null are the same length and are not the same thing,
    # and a string too long to inline takes a different path through the
    # importer than a short one does.
    _put(SINGLE)
    var frame = read_parquet(SINGLE)
    assert_equal(frame[3].strings()[0], "a")
    assert_equal(frame[3].strings()[1], "")
    assert_false(frame[3].is_valid(2))
    assert_equal(frame[3].strings()[3], "the quick brown fox jumps over")
    assert_equal(frame[3].strings()[4], "z")
    assert_equal(frame[3].null_count(), 1)


def test_a_bool_column_unpacks_from_bits() raises:
    _put(SINGLE)
    var frame = read_parquet(SINGLE)
    var flags = frame[4].as_typed[DType.bool]()
    assert_equal(flags[0], True)
    assert_equal(flags[1], False)
    assert_false(frame[4].is_valid(2))
    assert_equal(flags[3], True)
    assert_equal(frame[4].null_count(), 1)


def test_naming_columns_reads_only_those_and_in_that_order() raises:
    # A projection rather than a read and a drop. The order asked for is the
    # order that comes back, which is why label is first here.
    _put(SINGLE)
    var wanted: List[String] = ["label", "id"]
    var frame = read_parquet(SINGLE, wanted)
    assert_equal(frame.width(), 2)
    assert_equal(len(frame), 6)
    assert_equal(frame.names()[0], "label")
    assert_equal(frame.names()[1], "id")
    assert_equal(frame[0].strings()[0], "a")
    assert_equal(frame[1].as_typed[DType.int64]()[5], Int64(6))


def _put_hive() raises:
    """Writes the fixture twice into a `year=/month=` tree.

    The two files are identical, which is the point: everything that tells the
    two halves of the result apart is in the paths and nowhere in the bytes, so a
    test that reads the partition columns back cannot be passing by accident.

    Raises:
        Error: If the directories or the files cannot be written.
    """
    makedirs(String(HIVE, "/year=2024/month=3"), exist_ok=True)
    makedirs(String(HIVE, "/year=2025/month=7"), exist_ok=True)
    _put(String(HIVE, "/year=2024/month=3/part.parquet"))
    _put(String(HIVE, "/year=2025/month=7/part.parquet"))


def test_a_partitioned_directory_reads_its_directories_as_columns() raises:
    # The two files are the same six rows, so year and month are the only thing
    # that distinguishes the first half of this frame from the second, and
    # neither of them is in either file.
    _put_hive()
    var options = ParquetOptions()
    var frame = read_parquet(HIVE_GLOB, options)
    assert_equal(len(frame), 12)
    assert_equal(frame.width(), 7)
    # After the file's own columns and sorted by name, not in path order.
    assert_equal(frame.names()[5], "month")
    assert_equal(frame.names()[6], "year")
    var months = frame[5].as_typed[DType.int64]()
    var years = frame[6].as_typed[DType.int64]()
    assert_equal(years[0], Int64(2024))
    assert_equal(months[0], Int64(3))
    assert_equal(years[11], Int64(2025))
    assert_equal(months[11], Int64(7))
    assert_equal(frame[6].null_count(), 0)


def test_turning_the_partitioning_off_leaves_just_the_files() raises:
    # Off has to be sayable, because a directory with an equals sign in its name
    # that does not mean anything by it is a real thing and DuckDB reads one as
    # partitions unless it is told not to.
    _put_hive()
    var options = ParquetOptions()
    options.hive_partitioning = False
    var frame = read_parquet(HIVE_GLOB, options)
    assert_equal(len(frame), 12)
    assert_equal(frame.width(), 5)


def test_a_partition_column_can_be_projected_like_any_other() raises:
    # Naming year alongside a real column is the case where the projection and
    # the partitioning have to agree with each other, because one of the two
    # names is in the file and the other one is in the path.
    _put_hive()
    var options = ParquetOptions()
    options.columns = ["year", "id"]
    var frame = read_parquet(HIVE_GLOB, options)
    assert_equal(frame.width(), 2)
    assert_equal(len(frame), 12)
    assert_equal(frame.names()[0], "year")
    assert_equal(frame.names()[1], "id")
    assert_equal(frame[0].as_typed[DType.int64]()[0], Int64(2024))
    assert_equal(frame[1].as_typed[DType.int64]()[0], Int64(1))


def test_asking_where_a_row_came_from_adds_a_filename_column() raises:
    _put_hive()
    var options = ParquetOptions()
    options.hive_partitioning = False
    options.filename = True
    var frame = read_parquet(HIVE_GLOB, options)
    assert_equal(frame.width(), 6)
    assert_equal(frame.names()[5], "filename")
    assert_true("year=2024" in frame[5].strings()[0])
    assert_true("year=2025" in frame[5].strings()[11])


def test_every_option_at_once_is_still_one_query() raises:
    # Each setting is a clause in the same table function call, so the one thing
    # that can go wrong with all three on is the punctuation between them.
    # Unioning by name over files that already agree is a no op by design, and
    # here it is checking that asking for it does not break anything.
    _put_hive()
    var options = ParquetOptions()
    options.union_by_name = True
    options.filename = True
    var frame = read_parquet(HIVE_GLOB, options)
    assert_equal(len(frame), 12)
    assert_equal(frame.width(), 8)
    assert_equal(frame.names()[5], "filename")
    assert_equal(frame.names()[6], "month")
    assert_equal(frame.names()[7], "year")


def test_a_glob_of_two_files_reads_as_one_frame() raises:
    # Two files is two batches to the assembler, which is the same path a file
    # of two row groups takes and is the one that can be checked in.
    _put(FIRST)
    _put(SECOND)
    var frame = read_parquet(PATTERN)
    assert_equal(len(frame), 12)
    assert_equal(frame.width(), 5)
    assert_equal(frame[0].null_count(), 0)
    assert_equal(frame[3].null_count(), 2)
    assert_equal(frame[3].strings()[0], "a")
    assert_equal(frame[3].strings()[6], "a")


def test_asking_for_no_columns_is_refused() raises:
    var none = List[String]()
    with assert_raises(contains="no columns"):
        _ = read_parquet(SINGLE, none)


def test_a_file_that_is_not_there_says_so() raises:
    with assert_raises(contains="duckdb"):
        _ = read_parquet("/tmp/firepanda_parquet_no_such_file.parquet")


def test_a_column_that_is_not_there_says_so() raises:
    _put(SINGLE)
    var wanted: List[String] = ["nope"]
    with assert_raises(contains="duckdb"):
        _ = read_parquet(SINGLE, wanted)


def test_a_path_with_an_apostrophe_is_a_path_and_not_syntax() raises:
    # The quoting is what stands between a file name and a SQL injection, so it
    # is checked directly rather than only through a read.
    assert_equal(quote("plain.parquet"), "'plain.parquet'")
    assert_equal(quote("it's.parquet"), "'it''s.parquet'")


comptime _FIXTURE = String(
    "504152311504156015484c150c1500120000300401000901000209070400030d"
    "0800040d083c0500000000000000060000000000000015001516151a2c150c15"
    "10150615061c1808060000000000000018080100000000000000160028080600"
    "0000000000001808010000000000000011110000000b28020000000c01030388"
    "c60215041528152c4c150a1500120000144c0a00000014000000280000003200"
    "00003c00000015001516151a2c150c1510150615061c18043c00000018040a00"
    "0000160228043c00000018040a00000011110000000b2802000000033b030388"
    "46001504155015404c150a1500120000280000050104f83f0507080002c00908"
    "050130205fa00242000000000000e0bf15001516151a2c150c1510150615061c"
    "1808000000205fa00242180800000000000002c016022808000000205fa00242"
    "180800000000000002c011110000000b2802000000033b030388460015041576"
    "157a4c150a15001200003be80100000061000000001e00000074686520717569"
    "636b2062726f776e20666f78206a756d7073206f766572010000007a07000000"
    "756e69636f646515001516151a2c150c1510150615061c360228017a18001111"
    "0000000b2802000000033b03038846001500150e15122c150c1500150615061c"
    "18010118010016022801011801001111000000071802000000033b151504196c"
    "35001806736368656d61150a00150425021802696400150225021805736d616c"
    "6c00150a2502180576616c756500150c250218056c6162656c25004c1c000000"
    "150025021804666c616700160c191c195c26001c150419350006101918026964"
    "1502160c16900216fc01266c26081c1808060000000000000018080100000000"
    "00000016002808060000000000000018080100000000000000111100192c1504"
    "1500150200150015101502003c29061926000c00000026001c15021935000610"
    "191805736d616c6c1502160c16b80116c00126cc022684021c18043c00000018"
    "040a000000160228043c00000018040a000000111100192c1504150015020015"
    "0015101502003c29061926020a00000026001c150a193500061019180576616c"
    "75651502160c16800216f40126a00426c4031c1808000000205fa00242180800"
    "000000000002c016022808000000205fa00242180800000000000002c0111100"
    "192c15041500150200150015101502003c29061926020a00000026001c150c19"
    "350006101918056c6162656c1502160c16e00116e80126ce0626b8051c360228"
    "017a1800111100192c15041500150200150015101502003c164e19061926020a"
    "00000026001c150019250600191804666c61671502160c1654165826a0073c18"
    "01011801001602280101180100111100191c150015001502003c29061926020a"
    "00000016fc07160c260816f00700191c180c4152524f573a736368656d6118b8"
    "032f2f2f2f2f30414241414151414141414141414b4141774142674146414167"
    "4143674141414141424241414d41414141434141494141414142414149414141"
    "41424141414141554141414459414141416c4141414147414141414177414141"
    "414241414141457a2f2f2f384141414547454141414142674141414145414141"
    "4141414141414151414141426d6247466e41414141414e6a2f2f2f39302f2f2f"
    "2f41414142425241414141416341414141424141414141414141414146414141"
    "41624746695a577741414141454141514142414141414b442f2f2f3841414145"
    "4445414141414277414141414541414141414141414141554141414232595778"
    "315a51414741416741426741474141414141414143414e442f2f2f3841414145"
    "434541414141426741414141454141414141414141414155414141427a625746"
    "7362414141414d542f2f2f384141414142494141414142414146414149414159"
    "414277414d414141414541415141414141414141424168414141414163414141"
    "4142414141414141414141414341414141615751414141674144414149414163"
    "41434141414141414141414641414141414141414141413d3d00182070617271"
    "7565742d6370702d6172726f772076657273696f6e2032342e302e30195c1c00"
    "001c00001c00001c00001c0000001204000050415231"
)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
