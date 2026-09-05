"""The `DataFrame` binding, which is the narrow half of the front door.

What is here is deliberately not the pandas API. It is a private calling
convention that the Python layer in `python/firepanda/` is written against, and
the reason for the split is measured in
`docs/specs/13-the-bound-type-is-not-a-dataframe.md`. `PythonTypeBuilder` can
attach methods to a type and nothing else, so `df["a"]`, `len(df)`, `df.shape`
and `for row in df` are not expressible here by any route, and 28 percent of the
pandas surface is properties and operators. Those live in Python and call flat
named methods on this type.

So the naming below reads oddly on purpose. `length` rather than `__len__`,
`shape` returning a tuple that Python turns into a property, and no `__getitem__`
at all. Nothing in this file is public API and nothing in it should be shaped to
look like pandas.
"""

from std.os import abort
from std.memory import ArcPointer, Pointer
from std.python import Python, PythonObject
from std.python.bindings import check_arguments_arity

from firepanda.array.any import AnyArray
from firepanda.dtype.logical import LogicalType
from firepanda.frame import DataFrame
from firepanda.io.arrow_c import (
    ArrowArray,
    ArrowArrayStream,
    ArrowSchema,
    release_schema,
)
from firepanda.io.arrow_export import export_frame_array, export_frame_schema
from firepanda.io.arrow_stream import import_frame, import_stream
from firepanda.io.read import read_csv
from firepanda.py.build import frame_from
from firepanda.py.convert import (
    array_capsule,
    schema_capsule,
    take_array,
    take_schema,
    take_stream,
)
from firepanda.py.errors import (
    CANCELLED,
    COLUMN,
    DTYPE,
    IO,
    UNSUPPORTED,
    VALUE,
    retagged,
    tagged,
)
from firepanda.py.series import PySeries


def _int(value: PythonObject, name: String) raises -> Int:
    """Reads a Python integer, and says which argument was wrong if it is not one.

    `Int(py=value)` raises `invalid literal for int() with base 10: 'x'`, which
    is the right complaint and names neither the argument nor the function. A
    user with three integer arguments cannot tell from it which one they got
    wrong.

    Args:
        value: What Python passed.
        name: The parameter name, for the message.

    Returns:
        The integer.
    """
    try:
        return Int(py=value)
    except:
        raise tagged(
            DTYPE,
            String(
                name,
                " must be an integer, got ",
                Python.type(value).__name__,
                " ",
                value.__repr__(),
            ),
        )


@fieldwise_init
struct PyDataFrame(Movable, Writable):
    """A firepanda `DataFrame` with a CPython object wrapped around it."""

    var frame: ArcPointer[DataFrame]
    """The frame itself, shared rather than owned.

    The Python object holding this value is one holder of the frame and an
    exported Arrow array is another, which is what makes `__arrow_c_array__` zero
    copy: the consumer gets pointers into this frame rather than into a copy of
    it, and the frame stays alive until both have let go. See document 15.

    Nothing else about the binding cares. Every method here reads through the
    share exactly as it would have read through the value.
    """

    @staticmethod
    def py_init(
        out self: Self, args: PythonObject, kwargs: PythonObject
    ) raises:
        """Builds a frame out of a mapping of column name to values.

        One positional argument and nothing else. The pandas constructor takes
        five and the other four are refused by name on the Python side, which is
        where the pandas signature lives, because a caller who passes `columns=`
        should get a message about `columns` rather than a complaint about an
        unexpected keyword. Document 18 section 4 is why the split is that way
        round.

        Args:
            args: The data, or nothing for an empty frame.
            kwargs: Keyword arguments, of which none are accepted, because the
                Python layer has already turned them into the positional one.
        """
        check_arguments_arity(1, args, "DataFrame")
        self = Self(ArcPointer(frame_from(args[0])))

    @staticmethod
    def _frame(py_self: PythonObject) -> Pointer[Self, MutAnyOrigin]:
        """Recovers the Mojo value out of the Python object holding it.

        A failure here means the object is not a `PyDataFrame`, which the binding
        layer has already checked by the time a method body runs, so it is a bug
        in this file rather than a thing a caller can cause.

        Args:
            py_self: The Python object.

        Returns:
            A pointer to the wrapped value.
        """
        try:
            return py_self.downcast_value_ptr[Self]()
        except e:
            abort(String("not a firepanda DataFrame: ", e))

    @staticmethod
    def length(py_self: PythonObject) raises -> PythonObject:
        """Reports the row count.

        Args:
            py_self: The frame.

        Returns:
            The number of rows.
        """
        return PythonObject(len(Self._frame(py_self)[].frame[]))

    @staticmethod
    def width(py_self: PythonObject) raises -> PythonObject:
        """Reports the column count.

        Args:
            py_self: The frame.

        Returns:
            The number of columns.
        """
        return PythonObject(Self._frame(py_self)[].frame[].width())

    @staticmethod
    def names(py_self: PythonObject) raises -> PythonObject:
        """Reports the column names, in order.

        Args:
            py_self: The frame.

        Returns:
            A list of strings.
        """
        var out = Python.list()
        for name in Self._frame(py_self)[].frame[].names():
            out.append(PythonObject(name))
        return out

    @staticmethod
    def head(py_self: PythonObject, n: PythonObject) raises -> PythonObject:
        """Takes the first `n` rows.

        Args:
            py_self: The frame.
            n: How many rows to take.

        Returns:
            A new frame.
        """
        return PythonObject(
            alloc=Self(
                ArcPointer(Self._frame(py_self)[].frame[].head(_int(n, "n")))
            )
        )

    @staticmethod
    def tail(py_self: PythonObject, n: PythonObject) raises -> PythonObject:
        """Takes the last `n` rows.

        Args:
            py_self: The frame.
            n: How many rows to take.

        Returns:
            A new frame.
        """
        return PythonObject(
            alloc=Self(
                ArcPointer(Self._frame(py_self)[].frame[].tail(_int(n, "n")))
            )
        )

    @staticmethod
    def column(
        py_self: PythonObject, name: PythonObject
    ) raises -> PythonObject:
        """Takes one column out, as a series.

        This is what `df["a"]` reaches, which is the most written expression in
        pandas and had no answer at all before there was a bound series type.

        It copies, and it flattens a column stored in more than one piece,
        because a `Series` in the core is one contiguous array. Document 13 has
        the argument for why a borrowing version is not a small change: the
        Python object would have to keep the frame alive without owning it, which
        is the same lifetime problem the Arrow export solves and would want
        solving the same way.

        Args:
            py_self: The frame.
            name: The column name.

        Returns:
            A new series carrying the column's name.
        """
        var wanted = String(name)
        try:
            return PythonObject(
                alloc=PySeries(
                    ArcPointer(Self._frame(py_self)[].frame[].column(wanted))
                )
            )
        except:
            raise tagged(COLUMN, String("no such column ", name.__repr__()))

    @staticmethod
    def select(
        py_self: PythonObject, names: PythonObject
    ) raises -> PythonObject:
        """Takes several columns out, as a frame.

        This is the other half of `df[...]`, the one where the key is a list.
        Naming a column twice is an error rather than a duplication, which is the
        core's rule and is right: the result would have two columns under one
        name and nothing downstream could address the second.

        Args:
            py_self: The frame.
            names: The column names, in the order they should come out.

        Returns:
            A new frame with only those columns.
        """
        var wanted = List[String](capacity=Int(len(names)))
        for name in names:
            wanted.append(String(name))
        try:
            return PythonObject(
                alloc=Self(
                    ArcPointer(Self._frame(py_self)[].frame[].select(wanted))
                )
            )
        except cause:
            raise retagged(COLUMN, cause)

    @staticmethod
    def _borrowed(
        py_self: PythonObject,
    ) raises -> List[Pointer[AnyArray, MutAnyOrigin]]:
        """Points at every column of the frame, without copying any of them.

        A column with more than one chunk has no single Arrow array to be, which
        is what `__arrow_c_stream__` is for and this is not. The refusal happens
        here rather than half way through an export, so nothing has been
        allocated by the time it is raised.

        Args:
            py_self: The frame.

        Returns:
            One pointer per column, in schema order, each pointing into the
            frame rather than at a copy of it.
        """
        ref frame = Self._frame(py_self)[].frame[]
        var columns = List[Pointer[AnyArray, MutAnyOrigin]](
            capacity=frame.width()
        )
        for i in range(frame.width()):
            try:
                columns.append(
                    Pointer(to=frame.columns[i].only()).unsafe_origin_cast[
                        MutAnyOrigin
                    ]()
                )
            except:
                raise tagged(
                    UNSUPPORTED,
                    String(
                        "column '",
                        frame.names()[i],
                        "'",
                        (
                            " is stored in more than one chunk, which the Arrow"
                            " array protocol cannot express; use the stream"
                            " protocol instead"
                        ),
                    ),
                )
        return columns^

    @staticmethod
    def arrow_c_schema(py_self: PythonObject) raises -> PythonObject:
        """Describes the frame as an Arrow schema capsule.

        This is the Mojo half of `__arrow_c_schema__`. A frame is a struct in
        Arrow's type system, with one child per column, so what comes back is one
        capsule and not one per column.

        Args:
            py_self: The frame.

        Returns:
            A `PyCapsule` named `arrow_schema`.
        """
        ref frame = Self._frame(py_self)[].frame[]
        var types = List[LogicalType](capacity=frame.width())
        for i in range(frame.width()):
            types.append(frame.columns[i].type)
        try:
            return schema_capsule(export_frame_schema(types, frame.names()))
        except cause:
            raise retagged(UNSUPPORTED, cause)

    @staticmethod
    def arrow_c_array(
        py_self: PythonObject, requested_schema: PythonObject
    ) raises -> PythonObject:
        """Hands the frame's columns out as an Arrow array capsule, without copying.

        This is the Mojo half of `__arrow_c_array__`. The buffers in the exported
        array are the frame's own, and the export holds a share of the frame, so
        the consumer can outlive the Python object it came from and still be
        reading live memory rather than freed memory.

        Args:
            py_self: The frame.
            requested_schema: A schema capsule the consumer would rather have, or
                `None`. Anything other than `None` is refused, because converting
                on the way out is not written and a consumer is entitled to
                assume that what it asked for is what it got.

        Returns:
            A list of two capsules, the schema and the array, which the Python
            layer hands back as the tuple the protocol asks for.
        """
        if requested_schema is not Python.none():
            raise tagged(
                UNSUPPORTED,
                (
                    "requested_schema is not supported yet; pass None and cast"
                    " the result instead"
                ),
            )
        var columns = Self._borrowed(py_self)
        var keep = Self._frame(py_self)[].frame
        var rows = len(Self._frame(py_self)[].frame[])
        var pair = Python.list()
        pair.append(Self.arrow_c_schema(py_self))
        try:
            pair.append(array_capsule(export_frame_array(columns, rows, keep^)))
        except cause:
            raise retagged(UNSUPPORTED, cause)
        return pair

    def write_to(self, mut writer: Some[Writer]):
        """Writes the frame the way `describe` does.

        Both `__str__` and `__repr__` on the Python side come from
        `write_repr_to`, and `write_to` is never reached, which is recorded in
        document 13 section 2. It is written anyway so that the Mojo value
        behaves like every other `Writable` in the tree.

        Args:
            writer: Where to write.
        """
        writer.write(self.frame[].describe())

    def write_repr_to(self, mut writer: Some[Writer]):
        """Writes the frame. This is what Python sees for both `str` and `repr`.

        Args:
            writer: Where to write.
        """
        writer.write(self.frame[].describe())


def open_csv(path: PythonObject) raises -> PythonObject:
    """Reads a CSV file into a frame.

    The reader's own message is kept, because it says which file and what the
    operating system said about it, which is more than this function knows. All
    that is added is the classification, which this function knows and the
    reader does not: everything that goes wrong reading a file is an `OSError`
    to a Python caller.

    Args:
        path: The path to read.

    Returns:
        A new frame.
    """
    try:
        return PythonObject(
            alloc=PyDataFrame(ArcPointer(read_csv(String(path))))
        )
    except cause:
        raise retagged(IO, cause)


def _import_kind(cause: Error) -> String:
    """Decides which Python exception an import failure should arrive as.

    Two different things go wrong on the way in and a caller does different work
    about each. Either firepanda has no column for what the producer sent, which
    is a gap in this library and nothing the caller can fix by passing better
    data, or the data itself does not hold together, which usually means the
    producer has a bug. The first should reach Python as `NotImplementedError`
    and the second as `ValueError`, and reporting a malformed buffer as a missing
    feature sends the reader looking in the wrong place.

    The two are told apart by the message, which is not lovely and is the honest
    option available. Tagging them at the point they are raised would put the
    binding's error vocabulary inside `firepanda/io/`, which is code that has to
    work with no Python anywhere near it. So the rule is written down here where
    it can be read: every refusal in the import path that means a gap says so
    with the word supported, and none of the ones about malformed data use it.

    Args:
        cause: The error the import raised.

    Returns:
        The prefix to tag it with.
    """
    var message = String(cause)
    if "supported" in message or "firepanda has no" in message:
        return UNSUPPORTED
    return VALUE


def _from_stream(source: PythonObject) raises -> PythonObject:
    """Reads a frame through the stream half of the protocol.

    The path nearly everything takes. `pyarrow.Table`, Polars and pandas all
    offer `__arrow_c_stream__` and none of them offers `__arrow_c_array__`, so an
    importer that read only arrays would read almost nothing.

    Args:
        source: The object, already known to have `__arrow_c_stream__`.

    Returns:
        A new frame.
    """
    var capsule: PythonObject
    try:
        capsule = source.__arrow_c_stream__(Python.none())
    except cause:
        raise retagged(UNSUPPORTED, cause)

    var stream: ArrowArrayStream
    try:
        stream = take_stream(capsule)
    except cause:
        raise retagged(VALUE, cause)

    try:
        return PythonObject(
            alloc=PyDataFrame(ArcPointer(import_stream(stream)))
        )
    except cause:
        raise retagged(_import_kind(cause), cause)


def open_arrow(source: PythonObject) raises -> PythonObject:
    """Builds a frame from anything that speaks the Arrow PyCapsule protocol.

    This is the way in for every library on the other side of the boundary. A
    pyarrow table, a Polars frame, a pandas frame, or anything else that answers
    either half of the protocol arrives the same way and with no code here that
    knows which one it was. The stream is tried first, because it is what nearly
    everything at the table level actually offers.

    Unlike the export, this copies. `firepanda/io/arrow_import.mojo` says why at
    length, and the short version is that a firepanda buffer is 64-byte aligned
    and over allocated so that kernels can read past the end of a column, which
    is a promise no foreign buffer makes.

    Args:
        source: The object to read. It must offer `__arrow_c_stream__` or
            `__arrow_c_array__`, and it must describe a struct, which is what a
            table is in Arrow's type system.

    Returns:
        A new frame, owning all of its memory.
    """
    var builtins = Python.import_module("builtins")
    if builtins.hasattr(source, "__arrow_c_stream__"):
        return _from_stream(source)
    if not builtins.hasattr(source, "__arrow_c_array__"):
        raise tagged(
            UNSUPPORTED,
            String(
                "cannot read a ",
                Python.type(source).__name__,
                (
                    "; firepanda.from_arrow takes an object with"
                    " __arrow_c_stream__ or __arrow_c_array__, such as a"
                    " pyarrow table, a polars frame or a pandas frame"
                ),
            ),
        )

    var pair: PythonObject
    try:
        pair = source.__arrow_c_array__(Python.none())
    except cause:
        raise retagged(UNSUPPORTED, cause)
    if len(pair) != 2:
        raise tagged(
            VALUE,
            String(
                "__arrow_c_array__ returned ",
                len(pair),
                " values and the protocol says two",
            ),
        )

    var schema: ArrowSchema
    try:
        schema = take_schema(pair[0])
    except cause:
        raise retagged(VALUE, cause)

    var array: ArrowArray
    try:
        array = take_array(pair[1])
    except cause:
        # The schema is ours now and the capsule it came from will not release
        # it, so an array that never arrives still leaves this side owing one.
        release_schema(schema)
        raise retagged(VALUE, cause)

    try:
        return PythonObject(
            alloc=PyDataFrame(ArcPointer(import_frame(schema, array)))
        )
    except cause:
        raise retagged(_import_kind(cause), cause)


def raise_for_test(kind: PythonObject) raises -> PythonObject:
    """Raises one classified error of each kind, so the table can be tested.

    Every row of the mapping in `python/firepanda/errors.py` has to be exercised
    from Python, and the bound surface is five methods, none of which can reach
    most of the rows. This is the way in. It is registered as `_raise_for_test`
    and it is the one entry point in the extension that exists for the tests
    rather than for a user.

    The alternative was to test the mapping against messages written in the test
    file, which would have tested the Python half against itself and left the
    thing that actually matters, that the two halves agree on the wire format,
    unchecked.

    Args:
        kind: The bare kind, such as `column`.

    Returns:
        Never. It always raises.
    """
    var which = String(kind)
    if which == "column":
        raise tagged(COLUMN, "no such column 'regoin'")
    if which == "dtype":
        raise tagged(DTYPE, "cannot add int64 and float64")
    if which == "value":
        raise tagged(VALUE, "n must not be negative")
    if which == "io":
        raise tagged(IO, "no such file '/nowhere'")
    if which == "unsupported":
        raise tagged(UNSUPPORTED, "object dtype is not supported")
    if which == "cancelled":
        raise tagged(CANCELLED, "interrupted")
    if which == "untagged":
        raise Error("something went wrong a long way down")
    raise tagged(VALUE, String("no such kind ", kind.__repr__()))
