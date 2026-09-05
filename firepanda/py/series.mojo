"""The `Series` binding, which is the other half of the narrow front door.

Same rules as `frame.mojo`. Nothing here is the pandas API, everything here is
flat named methods that the generated Python class in `python/firepanda/` calls,
and the reason for the split is document 13.

A `Series` matters more than its size suggests. `df["a"]` is the most written
expression in pandas, and until this type exists the answer to it is that there
is no answer: a frame that can only ever hand back another frame is not a thing
anybody can port code to. So this is the type that turns the bound `DataFrame`
from a demonstration into something a caller can take a column out of.
"""

from std.os import abort
from std.memory import ArcPointer, Pointer
from std.python import Python, PythonObject
from std.python.bindings import check_arguments_arity

from firepanda.array.any import AnyArray
from firepanda.array.array import Array
from firepanda.dtype.dispatch import dispatch_typed
from firepanda.dtype.lists import ALL
from firepanda.dtype.logical import TypeKind
from firepanda.frame.series import Series
from firepanda.io.arrow_export import export_array_borrowed, export_schema
from firepanda.py.convert import array_capsule, schema_capsule
from firepanda.py.errors import DTYPE, UNSUPPORTED, retagged, tagged


def _int(value: PythonObject, name: String) raises -> Int:
    """Reads a Python integer, and says which argument was wrong if it is not one.

    The same helper `frame.mojo` has, for the same reason. It is written twice
    rather than shared because the shared version would have to live somewhere
    both files import, and a two line helper is not worth a module.

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


def _numbers[dt: DType](values: Array[dt]) raises -> PythonObject:
    """Turns one typed column into a Python list.

    A null becomes `None` rather than a NaN. pandas would have widened an integer
    column with a missing value to float64 and put a NaN in the hole, because a
    numpy int64 array has nowhere to record absence. A firepanda column is Arrow
    and has a validity bitmap, so the value is missing rather than approximated,
    and `None` is what says that. It is also what `pyarrow.Array.to_pylist` and
    `polars.Series.to_list` both return, so the surprising answer here would be
    the pandas one.

    Args:
        values: The column, borrowed rather than copied.

    Parameters:
        dt: The column's dtype, bound by the dispatch.

    Returns:
        A list with one element per row.
    """
    var out = Python.list()
    for i in range(len(values)):
        if not values.is_valid(i):
            out.append(Python.none())
        elif dt == DType.bool:
            out.append(PythonObject(Bool(values[i])))
        elif dt.is_floating_point():
            out.append(PythonObject(Float64(values[i].cast[DType.float64]())))
        else:
            out.append(PythonObject(Int(values[i].cast[DType.int64]())))
    return out


@fieldwise_init
struct PySeries(Movable, Writable):
    """A firepanda `Series` with a CPython object wrapped around it."""

    var series: ArcPointer[Series]
    """The column itself, shared rather than owned.

    Held the same way `PyDataFrame` holds its frame and for the same reason,
    which is document 15: an Arrow export hands out pointers into this memory and
    takes its own share, so a consumer can outlive the Python object the column
    came from and still be reading live memory. Nothing else about the binding
    cares.
    """

    @staticmethod
    def py_init(
        out self: Self, args: PythonObject, kwargs: PythonObject
    ) raises:
        """Refuses to build a series from Python.

        `pd.Series([1, 2, 3])` needs the Python to Arrow conversion that is not
        written, and the same argument `PyDataFrame.py_init` makes applies here:
        accepting the call and handing back an empty column would look like it
        worked.

        Args:
            args: Positional arguments, of which none are accepted.
            kwargs: Keyword arguments, of which none are accepted.
        """
        check_arguments_arity(0, args, "Series")
        raise tagged(
            UNSUPPORTED,
            (
                "take a series out of a frame with df['name']; building one"
                " from values is not wired up yet"
            ),
        )

    @staticmethod
    def _held(py_self: PythonObject) -> Pointer[Self, MutAnyOrigin]:
        """Recovers the Mojo value out of the Python object holding it.

        A failure here means the object is not a `PySeries`, which the binding
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
            abort(String("not a firepanda Series: ", e))

    @staticmethod
    def length(py_self: PythonObject) raises -> PythonObject:
        """Reports the row count.

        Args:
            py_self: The series.

        Returns:
            The number of rows.
        """
        return PythonObject(len(Self._held(py_self)[].series[]))

    @staticmethod
    def label(py_self: PythonObject) raises -> PythonObject:
        """Reports the column name.

        Called `label` rather than `name` because `name` on the extension side
        would collide with the attribute Python itself puts on a bound method,
        and the Python class exposes it as `name` anyway.

        Args:
            py_self: The series.

        Returns:
            The name, as a string.
        """
        return PythonObject(Self._held(py_self)[].series[].name)

    @staticmethod
    def dtype(py_self: PythonObject) raises -> PythonObject:
        """Reports the type, as firepanda spells it.

        This is a string and pandas returns a numpy dtype object, which is a
        difference worth being straight about rather than papering over. The
        names agree for every type both libraries have, so `str(s.dtype)` reads
        the same on both sides, and a caller comparing against `numpy.int64`
        will notice immediately rather than subtly.

        Args:
            py_self: The series.

        Returns:
            The type name, such as `int64` or `string`.
        """
        return PythonObject(String(Self._held(py_self)[].series[].logical()))

    @staticmethod
    def null_count(py_self: PythonObject) raises -> PythonObject:
        """Reports how many rows pandas would call missing.

        On a float column that is the cleared validity bits plus the NaNs, which
        is what `Series.null_count` in the core is careful about and why this
        does not simply count bits.

        Args:
            py_self: The series.

        Returns:
            The count.
        """
        return PythonObject(Self._held(py_self)[].series[].null_count())

    @staticmethod
    def head(py_self: PythonObject, n: PythonObject) raises -> PythonObject:
        """Takes the first `n` rows.

        Args:
            py_self: The series.
            n: How many rows to take.

        Returns:
            A new series.
        """
        return PythonObject(
            alloc=Self(
                ArcPointer(Self._held(py_self)[].series[].head(_int(n, "n")))
            )
        )

    @staticmethod
    def tail(py_self: PythonObject, n: PythonObject) raises -> PythonObject:
        """Takes the last `n` rows.

        Args:
            py_self: The series.
            n: How many rows to take.

        Returns:
            A new series.
        """
        return PythonObject(
            alloc=Self(
                ArcPointer(Self._held(py_self)[].series[].tail(_int(n, "n")))
            )
        )

    @staticmethod
    def to_list(py_self: PythonObject) raises -> PythonObject:
        """Copies every value out into a Python list.

        This is the slow way out of a column and it is here anyway, because until
        `__arrow_c_array__` exists on a series it is the only way out, and
        because a test that cannot read a value can only assert about shapes.
        Nothing on a hot path should call it.

        Three kinds of column are handled separately. Text is not dispatched over
        `ALL` because a string column is physically uint8 and would come back as
        a list of bytes. A column of the null type has no buffer at all, so there
        is nothing to read and every row is missing. Everything else is one
        typed pass.

        Args:
            py_self: The series.

        Returns:
            A list with one element per row, with `None` for the missing ones.
        """
        ref series = Self._held(py_self)[].series[]
        if series.logical().kind == TypeKind.NULL:
            var out = Python.list()
            for _ in range(len(series)):
                out.append(Python.none())
            return out
        if series.is_string():
            var out = Python.list()
            for i in range(len(series)):
                if series.is_valid(i):
                    out.append(PythonObject(series.text(i)))
                else:
                    out.append(Python.none())
            return out
        return dispatch_typed[ALL](series.values, _numbers)

    @staticmethod
    def arrow_c_schema(py_self: PythonObject) raises -> PythonObject:
        """Describes the column as an Arrow schema capsule.

        This is the Mojo half of `__arrow_c_schema__`. A series is one Arrow
        array rather than a struct of them, so what comes back describes the
        column's own type and not a wrapper around it, which is the whole
        difference between this and the frame's version.

        The field carries the series name. Arrow allows a top level array to have
        no name and pyarrow exports its own arrays that way, but a name that is
        already known is worth handing over: it is what `polars.Series` picks up
        as its name, and losing it here would mean the caller has to put it back.

        Args:
            py_self: The series.

        Returns:
            A `PyCapsule` named `arrow_schema`.
        """
        ref series = Self._held(py_self)[].series[]
        try:
            return schema_capsule(export_schema(series.logical(), series.name))
        except cause:
            raise retagged(UNSUPPORTED, cause)

    @staticmethod
    def arrow_c_array(
        py_self: PythonObject, requested_schema: PythonObject
    ) raises -> PythonObject:
        """Hands the column out as an Arrow array capsule, without copying.

        This is the Mojo half of `__arrow_c_array__`, and it is the fast way out
        of a column that `to_list` is the slow way out of. The buffers in the
        exported array are the column's own, and the export takes a share of the
        series, so a consumer can outlive the Python object and still be reading
        live memory.

        The frame's version has to refuse a column stored in more than one chunk,
        because a struct array has no way to express one. A series does not, since
        taking a column out of a frame flattens it, so there is exactly one array
        here by construction and nothing to refuse.

        Args:
            py_self: The series.
            requested_schema: A schema capsule the consumer would rather have, or
                `None`. Anything other than `None` is refused, for the reason the
                frame gives: converting on the way out is not written, and a
                consumer is entitled to assume it got what it asked for.

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
        var keep = Self._held(py_self)[].series
        var column = Pointer(to=keep[].values).unsafe_origin_cast[
            MutAnyOrigin
        ]()
        var pair = Python.list()
        pair.append(Self.arrow_c_schema(py_self))
        try:
            pair.append(array_capsule(export_array_borrowed(column, keep^)))
        except cause:
            raise retagged(UNSUPPORTED, cause)
        return pair

    def write_to(self, mut writer: Some[Writer]):
        """Writes the series the way the core writes it.

        Args:
            writer: Where to write.
        """
        writer.write(self.series[])

    def write_repr_to(self, mut writer: Some[Writer]):
        """Writes the series. This is what Python sees for both `str` and `repr`.

        Args:
            writer: Where to write.
        """
        writer.write(self.series[])
