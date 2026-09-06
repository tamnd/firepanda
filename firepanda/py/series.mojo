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

from firepanda.frame.index import Index
from firepanda.frame.series import Series
from firepanda.py.args import flag, whole, words
from firepanda.py.build import column_from, empty_column
from firepanda.io.arrow_export import export_array_borrowed, export_schema
from firepanda.py.convert import array_capsule, schema_capsule
from firepanda.py.index import PyIndex
from firepanda.py.errors import DTYPE, UNSUPPORTED, VALUE, retagged, tagged
from firepanda.py.ops import binary_op, constant, fill, unary_op
from firepanda.py.values import python_list


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
        """Builds a series out of a Python sequence.

        Two positional arguments, the values and the name, and nothing else. The
        pandas constructor takes five and the other three are refused by name on
        the Python side, for the reason `PyDataFrame.py_init` gives.

        Args:
            args: The values and the name.
            kwargs: Keyword arguments, of which none are accepted, because the
                Python layer has already turned them into positional ones.
        """
        check_arguments_arity(2, args, "Series")
        var name = String(args[1])
        if args[0] is Python.none():
            self = Self(ArcPointer(Series(name, empty_column(0))))
        else:
            self = Self(ArcPointer(column_from(name, args[0])))

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
                ArcPointer(Self._held(py_self)[].series[].head(whole(n, "n")))
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
                ArcPointer(Self._held(py_self)[].series[].tail(whole(n, "n")))
            )
        )

    @staticmethod
    def labels(py_self: PythonObject) raises -> PythonObject:
        """Hands out the row labels, as an index.

        This is what `s.index` reaches. A series taken out of a frame carries the
        frame's labels, so this is how a caller finds out which rows a column's
        values belong to.

        Args:
            py_self: The series.

        Returns:
            A new index carrying the series' labels.
        """
        return PythonObject(
            alloc=PyIndex(
                ArcPointer(Index(copy=Self._held(py_self)[].series[].index))
            )
        )

    @staticmethod
    def to_list(py_self: PythonObject) raises -> PythonObject:
        """Copies every value out into a Python list.

        This is the slow way out of a column and it is here anyway, because a
        test that cannot read a value can only assert about shapes. Nothing on a
        hot path should call it, and `__arrow_c_array__` is the way that does
        not copy. What it does with a null and with a string column is
        `firepanda/py/values.mojo`, which the index reads through as well.

        Args:
            py_self: The series.

        Returns:
            A list with one element per row, with `None` for the missing ones.
        """
        return python_list(Self._held(py_self)[].series[].values)

    @staticmethod
    def _other(value: PythonObject, name: String) raises -> ArcPointer[Series]:
        """Recovers the series out of an argument that should be one.

        Args:
            value: What Python passed.
            name: The parameter name, for the message.

        Returns:
            A share of the other series, so the caller can read it without
            copying.

        Raises:
            Error: Tagged `dtype`, if the argument is not a series.
        """
        try:
            return value.downcast_value_ptr[Self]()[].series
        except:
            raise tagged(
                DTYPE,
                String(
                    name,
                    " must be a Series, got ",
                    Python.type(value).__name__,
                    " ",
                    value.__repr__(),
                ),
            )

    @staticmethod
    def binary_series(
        py_self: PythonObject,
        other: PythonObject,
        op: PythonObject,
        flip: PythonObject,
        fill_value: PythonObject,
    ) raises -> PythonObject:
        """Applies an operation to two series, matching rows by label.

        One entry point for twenty six operators and named forms, with the
        operation crossing as a word. `firepanda/py/ops.mojo` says why the
        boundary is this shape rather than one bound method per operation.

        Args:
            py_self: The left operand.
            other: The right operand, which has to be a series.
            op: The operation, such as `add` or `lt`.
            flip: True for `other op self`, which is what a reflected form needs
                and which swapping the two arguments cannot express, because the
                result keeps this series' name.
            fill_value: What to put where exactly one of the two sides is
                missing, or `None`.

        Returns:
            A new series, on the union of the two indexes.

        Raises:
            Error: Tagged `dtype`, if an argument is the wrong type or the
                operation is not defined on the two dtypes.
        """
        var right = Self._other(other, "other")
        var which = binary_op(words(op, "op"))
        var filled = fill(fill_value)
        var flipped = flag(flip, "flip")
        try:
            return PythonObject(
                alloc=Self(
                    ArcPointer(
                        Self._held(py_self)[]
                        .series[]
                        .binary(right[], which, filled, flipped)
                    )
                )
            )
        except cause:
            raise retagged(DTYPE, cause)

    @staticmethod
    def binary_value(
        py_self: PythonObject,
        other: PythonObject,
        op: PythonObject,
        flip: PythonObject,
    ) raises -> PythonObject:
        """Applies an operation to every row of a series and one constant.

        There is no `fill_value` here and that is not an omission. pandas accepts
        one and ignores it, because a constant is never the side a row is missing
        from, so the Python layer drops it before the call rather than carrying an
        argument across the boundary that has nothing to do.

        Args:
            py_self: The series.
            other: The constant.
            op: The operation, such as `add` or `lt`.
            flip: True for `constant op series`, which is what `5 - s` needs.

        Returns:
            A new series of the same height, on the same labels.

        Raises:
            Error: Tagged `dtype`, if an argument is the wrong type or the
                operation is not defined on the two types.
        """
        var right = constant(other, "other")
        var which = binary_op(words(op, "op"))
        var flipped = flag(flip, "flip")
        try:
            return PythonObject(
                alloc=Self(
                    ArcPointer(
                        Self._held(py_self)[]
                        .series[]
                        .binary(right, which, flipped)
                    )
                )
            )
        except cause:
            raise retagged(DTYPE, cause)

    @staticmethod
    def compare_series(
        py_self: PythonObject, other: PythonObject, op: PythonObject
    ) raises -> PythonObject:
        """Compares two series row by row, refusing to align them.

        This is what the six comparison operators do and it is deliberately not
        what `binary_series` does. The core method says why at length: pandas
        aligns arithmetic and refuses to align a comparison, because a row only
        one side has has no true or false answer, and the flexible `eq` and its
        five relatives are the ones that align.

        This is the one call here that can fail two ways, and pandas raises a
        different class for each: a `ValueError` when the labels disagree and a
        `TypeError` when the dtypes do. The core raises one untagged error for
        both, so the labels are compared again on the failure path to tell them
        apart. Doing it there rather than up front costs nothing on the call that
        succeeds, which is every call but one.

        Args:
            py_self: The left operand.
            other: The right operand, which has to be a series.
            op: The comparison, such as `eq` or `lt`.

        Returns:
            A new boolean series.

        Raises:
            Error: Tagged `value` if the two are not labelled identically, and
                `dtype` if the comparison is not defined on the two dtypes.
        """
        var right = Self._other(other, "other")
        var which = binary_op(words(op, "op"))
        var held = Self._held(py_self)
        try:
            return PythonObject(
                alloc=Self(ArcPointer(held[].series[].compare(right[], which)))
            )
        except cause:
            if not held[].series[].index.equals(right[].index):
                raise retagged(VALUE, cause)
            raise retagged(DTYPE, cause)

    @staticmethod
    def unary(py_self: PythonObject, op: PythonObject) raises -> PythonObject:
        """Applies one of the four unary operations to a column.

        Args:
            py_self: The series.
            op: The operation, one of `neg`, `pos`, `abs` or `invert`.

        Returns:
            A new series of the same height, on the same labels.

        Raises:
            Error: Tagged `dtype`, if the operation is not defined on the
                column's dtype.
        """
        var which = unary_op(words(op, "op"))
        try:
            return PythonObject(
                alloc=Self(
                    ArcPointer(Self._held(py_self)[].series[].unary(which))
                )
            )
        except cause:
            raise retagged(DTYPE, cause)

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
