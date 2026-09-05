"""Tests for reading a whole table from an Arrow producer.

The producer here is firepanda's own exporter, wrapped in a hand built
`ArrowArrayStream`. That is not a compromise: it is the only way to get a
conforming stream without linking somebody else's library into the Mojo suite,
and a mistake made symmetrically in both halves would still be caught, because
every assertion is against the values that went in rather than against what the
exporter said about them. The Python suite is where the other half of this is
checked, against pyarrow, Polars and pandas, which are three producers that owe
firepanda nothing.

What the stream adds over a round trip of a single array is the two things a real
producer does and ours cannot be made to do by accident. It hands out more than
one batch, which is what a table with several chunks is and what the single
allocation in `assemble.mojo` exists for. And it is allowed to fail part way
through, after some batches have already been taken, which is a path no round trip
reaches.
"""

from std.ffi import c_char, external_call
from std.sys import size_of
from std.memory import ArcPointer, Pointer
from std.testing import TestSuite, assert_equal, assert_raises, assert_true

from firepanda.array.any import AnyArray
from firepanda.array.array import Array
from firepanda.array.strings import strings_from_list
from firepanda.dtype.logical import LogicalType
from firepanda.frame import DataFrame, Series
from firepanda.io.arrow_c import (
    ArrayPtr,
    ArrowArray,
    ArrowArrayStream,
    ArrowSchema,
    NullableCString,
    SchemaPtr,
    StreamPtr,
    VoidPtr,
    release_array,
    release_schema,
    release_stream,
    stream_next,
    stream_schema,
)
from firepanda.io.arrow_export import export_frame_array, export_frame_schema
from firepanda.io.arrow_stream import (
    export_frame_stream,
    frame_layout,
    import_frame,
    import_stream,
    struct_columns,
)


def _frame(var qty: List[Int64], var names: List[String]) raises -> DataFrame:
    """Builds a two column frame of int64 and string, with a null for -1."""
    var column = Array[DType.int64](len(qty))
    for i in range(len(qty)):
        if qty[i] == -1:
            column.set_null(i)
        else:
            column.set_valid(i, qty[i])
    var series = List[Series](capacity=2)
    series.append(Series("qty", AnyArray(column^)))
    series.append(Series("name", AnyArray(strings_from_list(names))))
    return DataFrame.from_series(series^)


def _one() raises -> DataFrame:
    """The frame most of these tests use."""
    return _frame(
        [Int64(4), Int64(-1), Int64(25)],
        ["rivet", "bolt", "a name longer than twelve bytes"],
    )


def _held() raises -> ArcPointer[DataFrame]:
    """The frame most of these tests use, behind the share the export needs.

    The export borrows a frame's buffers rather than copying them, so something
    has to keep the frame alive for as long as the exported array is readable.
    That is what the keep alive argument is for, and a share is what the binding
    passes, so it is what these tests pass too.

    Returns:
        A share of the frame.
    """
    return ArcPointer(_one())


def _schema_of(frame: ArcPointer[DataFrame]) raises -> ArrowSchema:
    """Exports a frame's schema, the way the binding does.

    Args:
        frame: The frame.

    Returns:
        The schema, owned by the caller.
    """
    var types = List[LogicalType](capacity=frame[].width())
    for i in range(frame[].width()):
        types.append(frame[].columns[i].type)
    return export_frame_schema(types, frame[].names())


def _array_of(frame: ArcPointer[DataFrame]) raises -> ArrowArray:
    """Exports a frame's columns without copying them, the way the binding does.

    Args:
        frame: The frame. The array takes its own share of it, so the caller may
            drop this one and the array stays readable.

    Returns:
        The array, owned by the caller.
    """
    var columns = List[Pointer[AnyArray, MutAnyOrigin]](
        capacity=frame[].width()
    )
    for i in range(frame[].width()):
        columns.append(
            Pointer(to=frame[].columns[i].only()).unsafe_origin_cast[
                MutAnyOrigin
            ]()
        )
    return export_frame_array(columns, len(frame[]), frame)


def test_a_frame_round_trips_through_one_struct_array() raises:
    var source = _held()
    var schema = _schema_of(source)
    var array = _array_of(source)
    var frame = import_frame(schema, array)
    assert_equal(len(frame), 3)
    assert_equal(frame.width(), 2)
    assert_equal(frame.names()[0], "qty")
    assert_equal(frame.names()[1], "name")
    assert_equal(frame.column("qty").as_typed[DType.int64]()[0], 4)
    assert_equal(frame.column("qty").null_count(), 1)
    assert_equal(
        frame.column("name").text(2), "a name longer than twelve bytes"
    )


def test_the_imported_frame_owns_its_memory() raises:
    # The source is dropped before anything is read, so a frame that borrowed
    # rather than copied is reading memory that has been freed.
    var frame: DataFrame

    var source = _held()
    var schema = _schema_of(source)
    var array = _array_of(source)
    frame = import_frame(schema, array)
    _ = source^

    assert_equal(frame.column("qty").as_typed[DType.int64]()[2], 25)
    assert_equal(frame.column("name").text(0), "rivet")


def test_both_structures_are_released_by_the_import() raises:
    var source = _held()
    var schema = _schema_of(source)
    var array = _array_of(source)
    _ = import_frame(schema, array)
    assert_true(schema.is_released())
    assert_true(array.is_released())


def test_a_refused_import_still_releases_what_it_was_handed() raises:
    # A producer that has handed a structure over has no way to reclaim it, so a
    # consumer that refuses the contents still owes the release call.
    var source = _held()
    var schema = _schema_of(source)
    var array = _array_of(source)
    array.n_children = 1
    with assert_raises(contains="fields and a batch has"):
        _ = import_frame(schema, array)
    assert_true(schema.is_released())
    assert_true(array.is_released())


def test_a_single_column_is_not_a_frame() raises:
    # What a consumer gets when it is handed one column and asked for a table.
    var schema = export_frame_schema([LogicalType.INT64], ["qty"])
    var child = schema.children.value().unsafe_offset(0)[]
    with assert_raises(contains="a frame is a struct"):
        _ = frame_layout(child[])
    release_schema(schema)


def test_a_struct_with_null_rows_is_refused() raises:
    # A frame has rows that are present and columns that are null, and Arrow
    # allows the other way round, so the shape has to be refused by name.
    var source = _held()
    var schema = _schema_of(source)
    var array = _array_of(source)
    array.null_count = 1
    var slot = _one_pointer()
    array.buffers = Pointer(to=slot).unsafe_origin_cast[MutUntrackedOrigin]()
    with assert_raises(contains="null rows cannot become a frame"):
        _ = struct_columns(array, 2)
    array.buffers = None
    array.null_count = 0
    _free_pointer(slot^)
    release_array(array)
    release_schema(schema)


def test_a_child_too_short_for_the_parent_is_refused() raises:
    var source = _held()
    var schema = _schema_of(source)
    var array = _array_of(source)
    array.length = 4
    with assert_raises(contains="which is not enough for"):
        _ = struct_columns(array, 2)
    array.length = 3
    release_array(array)
    release_schema(schema)


def test_the_windows_a_struct_hands_out_own_nothing() raises:
    # A window is a view of the producer's array, and a copy that kept its
    # release callback would release a structure the producer is still holding.
    var source = _held()
    var schema = _schema_of(source)
    var array = _array_of(source)
    var windows = struct_columns(array, 2)
    assert_equal(len(windows), 2)
    for window in windows:
        assert_true(window.is_released())
    release_array(array)
    release_schema(schema)


def test_a_parents_offset_is_added_to_every_child() raises:
    # Arrow lets a producer record a slice on the struct or on its children, and
    # both have to be honoured, because different producers use different ones.
    var source = _held()
    var schema = _schema_of(source)
    var array = _array_of(source)
    array.offset = 1
    array.length = 2
    var windows = struct_columns(array, 2)
    assert_equal(windows[0].offset, 1)
    assert_equal(windows[0].length, 2)
    array.offset = 0
    array.length = 3
    release_array(array)
    release_schema(schema)


struct _Batches(Movable):
    """What a test stream hands out, and the state of how far it has got."""

    var frames: List[DataFrame]
    """The batches, in order. The first one also supplies the schema."""

    var at: Int
    """How many have been handed out."""

    var fail_at: Int
    """Which call fails, or -1 for a stream that does not fail."""

    var message: Pointer[c_char, MutUntrackedOrigin]
    """The failure message, allocated so that it can outlive any Mojo scope."""

    def __init__(out self, var frames: List[DataFrame], fail_at: Int = -1):
        """Builds the state behind a test stream.

        Args:
            frames: The batches to hand out.
            fail_at: Which `get_next` call reports an error, or -1 for none.
        """
        self.frames = frames^
        self.at = 0
        self.fail_at = fail_at
        var text = "the producer gave up"
        self.message = external_call[
            "malloc", Pointer[c_char, MutUntrackedOrigin]
        ](text.byte_length() + 1)
        for i in range(text.byte_length()):
            self.message.unsafe_offset(i).unsafe_write(
                c_char(text.as_bytes()[i])
            )
        self.message.unsafe_offset(text.byte_length()).unsafe_write(c_char(0))


def _get_schema(stream: StreamPtr, out_schema: SchemaPtr) abi("C") -> Int32:
    """Describes the stream, from the first batch. The C entry point."""
    try:
        var box = stream[].private_data.value().unsafe_bitcast[_Batches]()
        ref frame = box[].frames[0]
        var types = List[LogicalType](capacity=frame.width())
        for i in range(frame.width()):
            types.append(frame.columns[i].type)
        out_schema.unsafe_write(export_frame_schema(types, frame.names()))
        return 0
    except:
        return 5


def _get_next(stream: StreamPtr, out_array: ArrayPtr) abi("C") -> Int32:
    """Hands out the next batch, or the released state at the end."""
    try:
        var box = stream[].private_data.value().unsafe_bitcast[_Batches]()
        if box[].at == box[].fail_at:
            return 5
        if box[].at >= len(box[].frames):
            out_array.unsafe_write(ArrowArray())
            return 0
        ref frame = box[].frames[box[].at]
        box[].at += 1
        var columns = List[Pointer[AnyArray, MutAnyOrigin]](
            capacity=frame.width()
        )
        for i in range(frame.width()):
            columns.append(
                Pointer(to=frame.columns[i].only()).unsafe_origin_cast[
                    MutAnyOrigin
                ]()
            )
        # No keep alive, because the batches live in the box and the box lives
        # until the stream is released, which a consumer does after it has read
        # everything. That is what a real producer's ownership looks like too.
        out_array.unsafe_write(export_frame_array(columns, len(frame), 0))
        return 0
    except:
        return 5


def _get_last_error(stream: StreamPtr) -> NullableCString:
    """Reports what went wrong, which is always the same thing here."""
    if not stream[].private_data:
        return None
    var box = stream[].private_data.value().unsafe_bitcast[_Batches]()
    return box[].message


def _release(stream: StreamPtr) abi("C") -> None:
    """Frees the state behind a test stream, and everything it handed out."""
    if not stream[].release:
        return
    if stream[].private_data:
        var box = stream[].private_data.value().unsafe_bitcast[_Batches]()
        external_call["free", NoneType](box[].message)
        box.unsafe_deinit_pointee()
        external_call["free", NoneType](box)
        stream[].private_data = None
    stream[].get_schema = None
    stream[].get_next = None
    stream[].get_last_error = None
    stream[].release = None


def _stream(var frames: List[DataFrame], fail_at: Int = -1) -> ArrowArrayStream:
    """Builds a conforming stream over a list of frames.

    Args:
        frames: The batches, which must all have the same schema.
        fail_at: Which `get_next` call reports an error, or -1 for none.

    Returns:
        The stream, which the caller must release.
    """
    var box = external_call["malloc", Pointer[_Batches, MutUntrackedOrigin]](
        size_of[_Batches]()
    )
    box.unsafe_write(_Batches(frames^, fail_at))
    var out = ArrowArrayStream()
    out.private_data = box.unsafe_bitcast[NoneType]()
    out.get_schema = _as_void(_get_schema)
    out.get_next = _as_void(_get_next)
    out.get_last_error = _as_void(_get_last_error)
    out.release = _as_void(_release)
    return out^


def test_a_stream_of_one_batch_is_a_frame() raises:
    var frames = List[DataFrame]()
    frames.append(_one())
    var stream = _stream(frames^)
    var frame = import_stream(stream)
    assert_equal(len(frame), 3)
    assert_equal(frame.width(), 2)
    assert_true(stream.is_released())


def test_a_stream_of_several_batches_is_one_frame() raises:
    # The case a round trip cannot produce, and the reason `assemble` is used
    # rather than a frame per batch and a concatenate.
    var frames = List[DataFrame]()
    frames.append(_frame([Int64(1)], ["one"]))
    frames.append(
        _frame([Int64(2), Int64(-1)], ["a value longer than twelve bytes", "x"])
    )
    frames.append(_frame([Int64(3)], ["three"]))
    var stream = _stream(frames^)
    var frame = import_stream(stream)
    assert_equal(len(frame), 4)
    assert_equal(frame.column("qty").as_typed[DType.int64]()[0], 1)
    assert_equal(frame.column("qty").as_typed[DType.int64]()[3], 3)
    assert_equal(frame.column("qty").null_count(), 1)
    assert_equal(frame.column("name").text(0), "one")
    assert_equal(
        frame.column("name").text(1), "a value longer than twelve bytes"
    )
    assert_equal(frame.column("name").text(3), "three")


def test_a_stream_of_no_batches_is_a_frame_with_no_rows() raises:
    # It still has a schema, so it still has its columns.
    var frames = List[DataFrame]()
    var stream = _stream(frames^)

    # The schema has to come from somewhere, so the box keeps one frame that is
    # never handed out as a batch.
    var box = stream.private_data.value().unsafe_bitcast[_Batches]()
    box[].frames.append(_one())
    box[].at = 1

    var frame = import_stream(stream)
    assert_equal(len(frame), 0)
    assert_equal(frame.width(), 2)
    assert_equal(frame.names()[0], "qty")
    assert_equal(frame.names()[1], "name")


def test_a_stream_that_fails_part_way_says_what_the_producer_said() raises:
    var frames = List[DataFrame]()
    frames.append(_one())
    frames.append(_one())
    var stream = _stream(frames^, fail_at=1)
    with assert_raises(contains="the producer gave up"):
        _ = import_stream(stream)
    assert_true(stream.is_released())


def test_a_released_stream_is_refused() raises:
    var stream = ArrowArrayStream()
    with assert_raises(contains="already been released"):
        _ = import_stream(stream)


def _types_of(frame: ArcPointer[DataFrame]) -> List[LogicalType]:
    """The column types, which the stream export takes rather than the frame."""
    var out = List[LogicalType](capacity=frame[].width())
    for i in range(frame[].width()):
        out.append(frame[].columns[i].type)
    return out^


def _pointers_of(
    frame: ArcPointer[DataFrame],
) -> List[Pointer[AnyArray, MutAnyOrigin]]:
    """One pointer per column, the way the binding builds them."""
    var out = List[Pointer[AnyArray, MutAnyOrigin]](capacity=frame[].width())
    for i in range(frame[].width()):
        try:
            out.append(
                Pointer(to=frame[].columns[i].only()).unsafe_origin_cast[
                    MutAnyOrigin
                ]()
            )
        except:
            pass
    return out^


def _exported(frame: ArcPointer[DataFrame]) raises -> ArrowArrayStream:
    """Exports a frame as a stream, the way the binding does."""
    return export_frame_stream(
        _pointers_of(frame),
        _types_of(frame),
        frame[].names(),
        len(frame[]),
        frame,
    )


def test_an_exported_stream_describes_the_frame() raises:
    var source = _held()
    var stream = _exported(source)
    var schema = ArrowSchema()
    assert_equal(stream_schema(stream, schema), 0)
    var layout = frame_layout(schema)
    assert_equal(len(layout), 2)
    assert_equal(layout.names[0], "qty")
    assert_equal(layout.names[1], "name")
    assert_equal(layout.formats[0], "l")
    assert_equal(layout.formats[1], "vu")
    release_schema(schema)
    release_stream(stream)


def test_an_exported_stream_describes_itself_as_often_as_it_is_asked() raises:
    # The consumer owns what it is given and releases it, so a producer that
    # handed out the same structure twice would be asking to have it released
    # twice.
    var source = _held()
    var stream = _exported(source)
    var first = ArrowSchema()
    var second = ArrowSchema()
    assert_equal(stream_schema(stream, first), 0)
    assert_equal(stream_schema(stream, second), 0)
    assert_true(first.format.value() != second.format.value())
    release_schema(first)
    release_schema(second)
    release_stream(stream)


def test_an_exported_stream_hands_out_one_batch_and_then_ends() raises:
    # A frame has no chunking, so its stream is one batch. The end is the
    # released state written into the caller's structure rather than an error.
    var source = _held()
    var stream = _exported(source)
    var first = ArrowArray()
    assert_equal(stream_next(stream, first), 0)
    assert_true(not first.is_released())
    assert_equal(first.length, 3)
    assert_equal(first.n_children, 2)
    release_array(first)

    var second = ArrowArray()
    assert_equal(stream_next(stream, second), 0)
    assert_true(second.is_released())
    release_stream(stream)


def test_an_exported_stream_borrows_rather_than_copies() raises:
    # The claim the whole export exists for. The batch's buffers are the frame's
    # own, which a producer that copied could not manage while the frame is still
    # alive and holding them.
    var source = _held()
    var stream = _exported(source)
    var batch = ArrowArray()
    assert_equal(stream_next(stream, batch), 0)
    var child = batch.children.value().unsafe_offset(0)[]
    assert_equal(
        Int(child[].buffers.value().unsafe_offset(1)[].value()),
        Int(source[].columns[0].only().data.values.unsafe_ptr()),
    )
    release_array(batch)
    release_stream(stream)


def test_an_exported_stream_keeps_the_frame_alive() raises:
    # The ownership question the borrow creates. Every other holder lets go
    # inside the block and the consumer keeps reading afterwards, so the values
    # read are freed memory unless the stream is holding a share.
    var stream: ArrowArrayStream

    def build() raises -> ArrowArrayStream:
        return _exported(_held())

    stream = build()
    var frame = import_stream(stream)
    assert_equal(len(frame), 3)
    assert_equal(frame.column("qty").as_typed[DType.int64]()[2], 25)
    assert_equal(frame.column("name").text(0), "rivet")


def test_a_frame_round_trips_through_its_own_stream() raises:
    # The two directions checked against each other rather than against a
    # reading of the specification, which is the only check that catches a
    # mistake made in one of them and not the other.
    var source = _held()
    var stream = _exported(source)
    var frame = import_stream(stream)
    assert_equal(len(frame), 3)
    assert_equal(frame.width(), 2)
    assert_equal(frame.names()[0], "qty")
    assert_equal(frame.column("qty").as_typed[DType.int64]()[0], 4)
    assert_equal(frame.column("qty").null_count(), 1)
    assert_equal(
        frame.column("name").text(2), "a name longer than twelve bytes"
    )


def test_an_exported_stream_released_without_being_read_is_not_a_leak() raises:
    # A consumer is allowed to change its mind, which is why the batch is built
    # when it is asked for rather than when the stream is made.
    var source = _held()
    var stream = _exported(source)
    release_stream(stream)
    assert_true(stream.is_released())
    release_stream(stream)
    assert_true(stream.is_released())


def _as_void[T: AnyType](f: T) -> VoidPtr:
    """Reinterprets a C function as the void pointer the stream fields hold.

    The reason the fields are void pointers rather than function pointers is in
    `arrow_c.mojo`: an `Optional` around a function pointer is two words in Mojo
    and would move every field after it.

    Args:
        f: The function.

    Returns:
        Its address.
    """
    return Pointer(to=f).unsafe_bitcast[VoidPtr]()[]


def _one_pointer() -> Optional[VoidPtr]:
    """A buffer slot that is not null, for the struct validity check.

    Nothing reads through it, so a byte from `malloc` is as good as a real
    validity bitmap. The check refuses a struct that has a validity buffer and a
    non-zero null count, and it refuses it before anything is read. The caller
    frees it with `_free_pointer`.

    Returns:
        A slot holding a pointer that is not null.
    """
    return external_call["malloc", VoidPtr](1)


def _free_pointer(var slot: Optional[VoidPtr]) -> None:
    """Frees what `_one_pointer` allocated.

    Args:
        slot: The slot it returned.
    """
    external_call["free", NoneType](slot.value())


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
