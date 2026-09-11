"""Tests for a read that comes back in morsels rather than in one chunk.

A frame in one chunk is a query on one core. The pipeline runs its elementwise
operators a chunk per worker, so a frame the reader handed back as a single
allocation gives it nothing to divide, and cutting one up afterwards copies every
byte a second time. The reader already fills its output in parallel pieces, so
the chunk boundaries cost nothing to put in while the copy is happening.

What these check is that the boundaries are the only thing that changed. The same
query read in morsels has to hold the same values in the same order, with the
nulls on the same rows and the strings still next to their payload, and a null or
a long string that lands across a boundary is the case that would say otherwise.
Five thousand rows is deliberate: DuckDB hands its answer over in chunks of two
thousand and forty eight, so a morsel of a thousand rows cuts inside a batch and
a batch crosses three morsels, which is the arrangement a reader gets from a real
file and cannot get from a fixture small enough to check in.

These tests need libduckdb, which pixi.toml puts in the development environment.
Without it every one of them fails at the first call with a message saying so.
"""

from std.testing import (
    TestSuite,
    assert_equal,
    assert_false,
    assert_raises,
    assert_true,
)

from firepanda.frame.frame import DataFrame
from firepanda.io.parquet import Session

comptime ROWS = 5000
"""How many rows the generated result holds. Larger than one DuckDB chunk, and
not a multiple of the morsel height below, so the last morsel is a remainder."""

comptime HEIGHT = 1000
"""The morsel height the tests ask for. Not a divisor of DuckDB's chunk size, so
the two sets of boundaries interleave rather than lining up."""


def _numbers(morsel_rows: Int) raises -> DataFrame:
    """Reads a column of the numbers zero to `ROWS`, at a morsel height.

    Args:
        morsel_rows: The height to ask for, or zero for one chunk.

    Returns:
        One int64 column named `i`.
    """
    var session = Session()
    return session.run(
        String("SELECT i FROM range(", ROWS, ") t(i)"), morsel_rows
    )


def _chunk_lengths(frame: DataFrame, column: Int) raises -> List[Int]:
    """Returns the row count of every chunk of one column.

    Args:
        frame: The frame.
        column: Which column.

    Returns:
        One length per chunk, in order.
    """
    var out = List[Int]()
    ref held = frame.columns[column]
    for c in range(held.num_chunks()):
        out.append(len(held.chunks[c]))
    return out^


def test_a_read_with_no_height_comes_back_in_one_chunk() raises:
    # The default, and what every eager caller depends on: `DataFrame` borrows a
    # column by position only when there is a single chunk to borrow.
    var frame = _numbers(0)
    assert_equal(len(frame), ROWS)
    assert_equal(frame.columns[0].num_chunks(), 1)
    assert_equal(frame[0].as_typed[DType.int64]()[4999], Int64(4999))


def test_asking_for_a_height_cuts_the_frame_into_chunks_of_it() raises:
    var frame = _numbers(HEIGHT)
    assert_equal(len(frame), ROWS)
    var lengths = _chunk_lengths(frame, 0)
    assert_equal(len(lengths), 5)
    for c in range(5):
        assert_equal(lengths[c], HEIGHT)


def test_the_last_chunk_holds_the_remainder() raises:
    var session = Session()
    var frame = session.run(String("SELECT i FROM range(2500) t(i)"), HEIGHT)
    var lengths = _chunk_lengths(frame, 0)
    assert_equal(len(lengths), 3)
    assert_equal(lengths[0], 1000)
    assert_equal(lengths[1], 1000)
    assert_equal(lengths[2], 500)


def test_the_values_arrive_in_the_same_order_they_did_before() raises:
    # The one that would catch a piece written into the wrong morsel, or into
    # the right morsel at the offset it would have had in a single allocation.
    var frame = _numbers(HEIGHT)
    ref held = frame.columns[0]
    var row = 0
    for c in range(held.num_chunks()):
        var values = held.chunks[c].as_typed[DType.int64]()
        for i in range(len(held.chunks[c])):
            assert_equal(values[i], Int64(row))
            row += 1
    assert_equal(row, ROWS)


def test_a_null_keeps_its_row_across_a_boundary() raises:
    # Validity is pasted after the fill rather than during it, and it is pasted
    # into a morsel's own bitmap at an offset within that morsel. Every third
    # row is null, so no morsel boundary is also a run boundary.
    var session = Session()
    var frame = session.run(
        String(
            "SELECT CASE WHEN i % 3 = 0 THEN NULL ELSE i END AS v FROM range(",
            ROWS,
            ") t(i)",
        ),
        HEIGHT,
    )
    ref held = frame.columns[0]
    assert_true(held.num_chunks() > 1)
    var row = 0
    var nulls = 0
    for c in range(held.num_chunks()):
        var values = held.chunks[c].as_typed[DType.int64]()
        for i in range(len(held.chunks[c])):
            if row % 3 == 0:
                assert_false(held.chunks[c].is_valid(i))
                nulls += 1
            else:
                assert_true(held.chunks[c].is_valid(i))
                assert_equal(values[i], Int64(row))
            row += 1
    assert_equal(row, ROWS)
    assert_equal(nulls, 1667)


def test_a_string_column_keeps_its_payload_in_its_own_chunk() raises:
    # A string column plans its payload before it allocates, and the running
    # total it plans is now per morsel. A string too long to inline is the one
    # that goes to the payload at all, so every row here is one.
    var session = Session()
    var frame = session.run(
        String(
            (
                "SELECT 'a string long enough not to inline ' || i AS s"
                " FROM range("
            ),
            ROWS,
            ") t(i)",
        ),
        HEIGHT,
    )
    ref held = frame.columns[0]
    assert_equal(held.num_chunks(), 5)
    var row = 0
    for c in range(held.num_chunks()):
        for i in range(len(held.chunks[c])):
            assert_equal(
                held.chunks[c].strings()[i],
                String("a string long enough not to inline ", row),
            )
            row += 1
    assert_equal(row, ROWS)


def test_a_height_taller_than_the_answer_is_one_chunk() raises:
    var session = Session()
    var frame = session.run(String("SELECT i FROM range(10) t(i)"), HEIGHT)
    assert_equal(frame.columns[0].num_chunks(), 1)
    assert_equal(len(frame), 10)
    assert_equal(frame[0].as_typed[DType.int64]()[9], Int64(9))


def test_an_answer_of_no_rows_is_still_one_chunk() raises:
    # Zero morsels is not a frame, so an empty answer keeps the single empty
    # chunk it always had.
    var session = Session()
    var frame = session.run(
        String("SELECT i FROM range(10) t(i) WHERE i > 100"), HEIGHT
    )
    assert_equal(len(frame), 0)
    assert_equal(frame.columns[0].num_chunks(), 1)


def test_several_columns_are_cut_on_the_same_boundaries() raises:
    # A frame whose columns disagree about their chunk boundaries is not a
    # frame, and the row count of every chunk is what says they agree.
    var session = Session()
    var frame = session.run(
        String(
            "SELECT i, i * 2 AS twice, 'row ' || i AS name FROM range(",
            ROWS,
            ") t(i)",
        ),
        HEIGHT,
    )
    var first = _chunk_lengths(frame, 0)
    assert_equal(len(first), 5)
    for c in range(1, 3):
        var lengths = _chunk_lengths(frame, c)
        assert_equal(len(lengths), len(first))
        for k in range(len(first)):
            assert_equal(lengths[k], first[k])


def test_a_morselled_column_refuses_to_be_borrowed_whole() raises:
    # Not a limitation being asserted for its own sake. It is why the height is
    # not on `read_parquet` yet: a frame in morsels is for the pipeline, and an
    # eager caller that was handed one would find `frame[0]` raising.
    var frame = _numbers(HEIGHT)
    with assert_raises():
        _ = frame[0].null_count()


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
