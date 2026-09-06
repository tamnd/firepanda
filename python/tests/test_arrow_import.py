"""Reading a frame from somebody else's Arrow, checked against their libraries.

The export direction is in `test_arrow.py` and this is the other one. It matters
more than the symmetry suggests, because until it exists the only way to make a
firepanda frame is to read a CSV, and a library nobody can hand data to is a
library nobody can put in the middle of anything.

The producers here are pyarrow, Polars and pandas, and they are the point. A test
that built its own conforming stream would be testing this code against my
reading of the specification, which is the reading that produced the code. These
three implemented the protocol without knowing firepanda existed, so what they
hand over is what the protocol actually is.

One thing the writing of this found, which is worth stating because it is the
shape of the whole file. `__arrow_c_array__` is almost never what a table offers.
Of the four container types tested here only `pyarrow.RecordBatch` has it. A
`pyarrow.Table`, a `polars.DataFrame` and a `pandas.DataFrame` all offer only
`__arrow_c_stream__`. So the stream is not the follow up to the array direction,
it is the main road, and the tests are arranged that way.
"""

from __future__ import annotations

import gc
import importlib.util
from types import ModuleType

import pytest

needs = {
    name: pytest.mark.skipif(
        importlib.util.find_spec(name) is None, reason=f"{name} is not installed"
    )
    for name in ("pyarrow", "polars", "pandas")
}

NAMES = ["rivet", "bolt", "a much longer part name than twelve bytes"]
QTY = [4, None, 25]


def columns() -> dict[str, list[object]]:
    """The data every producer in this file hands over, so one set of asserts fits."""
    return {"name": list(NAMES), "qty": list(QTY)}


def check(frame: object) -> None:
    """Asserts a frame holds what `columns` describes.

    Written once because the interesting variable is which library produced the
    data, not which assertion is being made about it. Every case below ends here.
    """
    import pyarrow as pa

    table = pa.table(frame)
    assert table.num_rows == 3
    assert table.column_names == ["name", "qty"]
    assert table.column("name").to_pylist() == NAMES
    assert table.column("qty").to_pylist() == QTY


@needs["pyarrow"]
def test_a_record_batch_is_read(firepanda: ModuleType) -> None:
    """The one container that offers the array half of the protocol."""
    import pyarrow as pa

    check(firepanda.from_arrow(pa.record_batch(columns())))


@needs["pyarrow"]
def test_a_table_is_read(firepanda: ModuleType) -> None:
    """The common case, and it arrives as a stream rather than as an array."""
    import pyarrow as pa

    check(firepanda.from_arrow(pa.table(columns())))


@needs["pyarrow"]
def test_a_table_of_several_chunks_becomes_one_frame(firepanda: ModuleType) -> None:
    """What the single allocation in the assembler is for.

    A chunked table hands out a batch at a time and firepanda has no chunking, so
    the rows have to land in one column each. The chunk boundary is put in the
    middle of the long string on purpose, since a long string is the one value
    whose bytes live somewhere other than the row.
    """
    import pyarrow as pa

    table = pa.concat_tables(
        [
            pa.table({"name": NAMES[:2], "qty": QTY[:2]}),
            pa.table({"name": NAMES[2:], "qty": QTY[2:]}),
        ]
    )
    assert table.column("name").num_chunks == 2
    check(firepanda.from_arrow(table))


@needs["pyarrow"]
def test_a_slice_is_read_as_the_slice_and_not_the_whole(firepanda: ModuleType) -> None:
    """Arrow slices by recording an offset, and a consumer that ignores it is wrong.

    Which structure the offset is recorded on is up to the producer. pyarrow puts
    it on the children of a record batch and leaves the batch itself at zero, so
    a consumer that only reads the parent's offset passes this by accident and
    then fails on a producer that chose the other way.
    """
    import pyarrow as pa

    frame = firepanda.from_arrow(pa.record_batch(columns()).slice(1))
    table = pa.table(frame)
    assert table.num_rows == 2
    assert table.column("name").to_pylist() == NAMES[1:]
    assert table.column("qty").to_pylist() == QTY[1:]


@needs["pyarrow"]
def test_a_table_with_no_rows_is_read_as_its_schema(firepanda: ModuleType) -> None:
    """A stream may hand out no batches at all, and it still described itself."""
    import pyarrow as pa

    schema = pa.schema([("name", pa.string()), ("qty", pa.int64())])
    frame = firepanda.from_arrow(pa.table({"name": [], "qty": []}, schema=schema))
    table = pa.table(frame)
    assert table.num_rows == 0
    assert table.column_names == ["name", "qty"]


@needs["polars"]
def test_a_polars_frame_is_read(firepanda: ModuleType) -> None:
    """Polars is the producer that found a real bug in the import.

    Its string columns are Arrow views with no data buffer at all when every
    string is short enough to sit inside its view, which is three buffers rather
    than the four a view column is usually described as having. The importer
    required four and refused every such column until this test existed.
    """
    import polars as pl

    check(firepanda.from_arrow(pl.DataFrame(columns())))


@needs["polars"]
def test_a_polars_frame_of_only_short_strings_is_read(firepanda: ModuleType) -> None:
    """The three buffer case on its own, so a regression names itself."""
    import polars as pl
    import pyarrow as pa

    frame = firepanda.from_arrow(pl.DataFrame({"name": ["a", "bb", "ccc"]}))
    assert pa.table(frame).column("name").to_pylist() == ["a", "bb", "ccc"]


@needs["pandas"]
def test_a_pandas_frame_is_read(firepanda: ModuleType) -> None:
    """The library the compatibility work exists for, read rather than written."""
    import pandas as pd

    check(firepanda.from_arrow(pd.DataFrame(columns())))


@needs["pyarrow"]
def test_the_frame_owns_its_rows_after_the_producer_is_gone(
    firepanda: ModuleType,
) -> None:
    """The import copies, and this is the assertion that says so.

    The export direction borrows and keeps the frame alive to make that safe. The
    import cannot do the same thing in reverse, because the producer's memory is
    released when the capsule's release callback is called, which happens before
    this line. So the values read here are freed memory unless they were copied.
    """
    import pyarrow as pa

    source = pa.table(columns())
    frame = firepanda.from_arrow(source)
    del source
    gc.collect()
    check(frame)


@needs["pyarrow"]
def test_a_frame_survives_a_round_trip_through_firepanda(
    firepanda: ModuleType,
) -> None:
    """Out and back, which is the two directions checked against each other."""
    import pyarrow as pa

    once = firepanda.from_arrow(pa.table(columns()))
    twice = firepanda.from_arrow(pa.table(once))
    check(twice)


@needs["pyarrow"]
def test_reading_a_thousand_times_does_not_grow_without_bound(
    firepanda: ModuleType,
) -> None:
    """A release that is missed or made twice shows up here and almost nowhere else.

    A leak of one frame is invisible and a double release is a crash that a single
    import is unlikely to reach. Repetition turns both into something a test can
    see: the loop either finishes or it does not.
    """
    import pyarrow as pa

    source = pa.table(columns())
    for _ in range(1000):
        firepanda.from_arrow(source)


def test_an_object_that_speaks_neither_half_is_refused(firepanda: ModuleType) -> None:
    """The error a caller gets for passing the wrong thing, which says what to pass."""
    with pytest.raises(firepanda.errors.UnsupportedError) as caught:
        firepanda.from_arrow([1, 2, 3])
    assert "cannot read a list" in str(caught.value)
    assert "__arrow_c_stream__" in str(caught.value)


@needs["pyarrow"]
def test_a_single_column_is_not_a_frame(firepanda: ModuleType) -> None:
    """A column offers the protocol too, and it is not a table.

    Worth its own case because the failure without it is confusing rather than
    loud: an array of int64 read as a struct is a schema with no children, which
    is a frame with no columns rather than an error.
    """
    import pyarrow as pa

    with pytest.raises(firepanda.errors.InvalidArgumentError) as caught:
        firepanda.from_arrow(pa.array([1, 2, 3]))
    assert "a frame is a struct" in str(caught.value)


@needs["pyarrow"]
def test_a_type_firepanda_does_not_have_is_refused_by_name(
    firepanda: ModuleType,
) -> None:
    """The producer is allowed to send anything Arrow has, and we have less of it."""
    import pyarrow as pa

    # A duration used to be the example here and firepanda reads one now, so the
    # example is a time of day, which is the next Arrow type with no firepanda
    # spelling and is not going to acquire one by accident.
    with pytest.raises(firepanda.errors.UnsupportedError) as caught:
        firepanda.from_arrow(pa.table({"when": pa.array([1, 2], type=pa.time64("us"))}))
    assert "unsupported format string" in str(caught.value)


@needs["pyarrow"]
def test_the_two_ways_of_refusing_arrive_as_two_exceptions(
    firepanda: ModuleType,
) -> None:
    """A gap in firepanda and a bad array are different problems for the caller.

    One is answered by waiting for the feature and the other by fixing whatever
    produced the data, so they arrive as `NotImplementedError` and `ValueError`
    rather than as one exception with two meanings.
    """
    import pyarrow as pa

    assert issubclass(firepanda.errors.UnsupportedError, NotImplementedError)
    assert issubclass(firepanda.errors.InvalidArgumentError, ValueError)
    with pytest.raises(NotImplementedError):
        firepanda.from_arrow(pa.table({"of": pa.array([[1], [2]])}))
    with pytest.raises(ValueError):
        firepanda.from_arrow(pa.array([1, 2, 3]))
