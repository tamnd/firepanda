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

from firepanda.dtype.logical import LogicalType, named_type
from firepanda.frame.index import Index
from firepanda.frame.series import Series
from firepanda.kernel.reduce import reduce_any
from firepanda.py.args import flag, number, whole, words
from firepanda.py.build import column_from, empty_column
from firepanda.io.arrow_export import export_array_borrowed, export_schema
from firepanda.py.convert import array_capsule, schema_capsule
from firepanda.py.index import PyIndex
from firepanda.py.errors import DTYPE, UNSUPPORTED, VALUE, retagged, tagged
from firepanda.py.ops import (
    binary_op,
    constant,
    binary_tag,
    constant_tag,
    fill,
    unary_op,
)
from firepanda.py.reduce import reduction
from firepanda.py.temporal import column_part
from firepanda.py.temporal import part as temporal_part
from firepanda.py.temporal import word as temporal_word
from firepanda.py.temporal import word_part
from firepanda.py.transform import transformation, transformed
from firepanda.py.values import python_list, python_value


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
        # A firepanda series arriving here is copied rather than iterated.
        # `pd.Series(a_series)` is ordinary pandas, and going out through a
        # Python list and back would infer the type again off the values, which
        # loses a column of instants entirely and is slow for the columns it
        # does not lose. An empty name means the caller passed none, so the
        # source keeps the name it had, which is what pandas does too.
        var held = Optional[ArcPointer[Series]]()
        try:
            held = args[0].downcast_value_ptr[Self]()[].series
        except:
            held = Optional[ArcPointer[Series]]()

        if held:
            var copied = Series(copy=held.value()[])
            if name.byte_length() != 0:
                copied.name = name
            self = Self(ArcPointer(copied^))
        elif args[0] is Python.none():
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
    def relabel(
        py_self: PythonObject, name: PythonObject
    ) raises -> PythonObject:
        """Returns the column under a different name.

        A copy rather than a change in place, which is what pandas does and is
        also the only thing available here: the series behind a Python wrapper is
        shared with whoever else is holding it, so renaming in place would rename
        somebody else's column.

        Empty means no name. A pandas series with no name has `None` there and
        this side has a `String` with nothing in it, which is the same
        arrangement `label` reports through and is turned back into `None` in
        Python.

        Args:
            py_self: The series.
            name: The new name, and empty for none.

        Returns:
            A copy carrying the new name.
        """
        var out = Series(copy=Self._held(py_self)[].series[])
        out.name = words(name, "name")
        return PythonObject(alloc=Self(ArcPointer(out^)))

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
    def reduce(
        py_self: PythonObject, kind: PythonObject, param: PythonObject
    ) raises -> PythonObject:
        """Reduces the whole column to one Python value.

        Twelve of the reductions come through here rather than through twelve
        bound methods, for the reason `ops.mojo` gives about the operators: they
        differ by a word and the word can cross.

        The answer comes back out as an ordinary Python number and not as a one
        row series, because that is what pandas hands back and because a caller
        who wrote `s.sum() > 10` is holding it in a Python expression a moment
        later. `python_value` is what decides what a missing answer looks like,
        so `s.mean()` on a column with nothing in it is `None` here and the
        Python layer is where that becomes the NaN pandas gives.

        Args:
            py_self: The series.
            kind: The reduction, as pandas spells the method.
            param: The delta degrees of freedom or the quantile, and zero for
                the reductions that take neither.

        Returns:
            The value, or `None` if the reduction has no answer.

        Raises:
            Error: Tagged `dtype`, if the column has a type the reduction
                cannot read, and tagged `value` if the name is not a reduction.
        """
        var wanted = reduction(words(kind, "kind"), number(param, "param"))
        try:
            return python_value(
                reduce_any(Self._held(py_self)[].series[].values, wanted), 0
            )
        except e:
            raise retagged(DTYPE, e)

    @staticmethod
    def transform(
        py_self: PythonObject, kind: PythonObject, periods: PythonObject
    ) raises -> PythonObject:
        """Applies one named transformation and hands back a column.

        Twelve of them come through here for the reason `reduce` gives above,
        and the answer is a series rather than a Python value because a
        transformation answers a column. `firepanda/py/transform.mojo` is the
        list and the argument for the list being one door.

        Args:
            py_self: The series.
            kind: The transformation, as pandas spells the method.
            periods: The `periods` or the `limit`, and zero for the ones that
                take neither.

        Returns:
            A new series.

        Raises:
            Error: Tagged `dtype`, if the column has a type the transformation
                cannot read, and tagged `value` if the name is not one of the
                twelve.
        """
        var wanted = transformation(words(kind, "kind"))
        try:
            return PythonObject(
                alloc=Self(
                    ArcPointer(
                        transformed(
                            Self._held(py_self)[].series[],
                            wanted,
                            whole(periods, "periods"),
                        )
                    )
                )
            )
        except e:
            raise retagged(DTYPE, e)

    @staticmethod
    def cast(
        py_self: PythonObject, dtype: PythonObject, strict: PythonObject
    ) raises -> PythonObject:
        """Converts the column to another type and hands back a new one.

        The name that arrives here is already canonical, because the Python
        layer resolves what pandas accepts, which is aliases and python types
        and numpy dtypes, down to one of the spellings `dtype` prints. So this
        reads a name and nothing else, and a name it does not know is a bug on
        the other side rather than a caller's mistake.

        Args:
            py_self: The series.
            dtype: The target type, spelled the way `dtype` prints it.
            strict: Whether a text value that is not a number raises rather
                than becoming a null.

        Returns:
            A new series of that type.

        Raises:
            Error: Tagged `value` if the name is not one this layer prints, and
                tagged `dtype` if the conversion is not one firepanda has or a
                text value is not a number.
        """
        var wanted: LogicalType
        try:
            wanted = named_type(words(dtype, "dtype"))
        except cause:
            raise retagged(VALUE, cause)
        try:
            return PythonObject(
                alloc=Self(
                    ArcPointer(
                        Self._held(py_self)[]
                        .series[]
                        .cast(wanted, flag(strict, "strict"))
                    )
                )
            )
        except cause:
            raise retagged(DTYPE, cause)

    @staticmethod
    def monotonic(
        py_self: PythonObject, increasing: PythonObject
    ) raises -> PythonObject:
        """Answers whether the column is sorted, one way or the other.

        The two of these are a separate door from `transform` because they
        answer a bool rather than a column, which is the thing that decides
        which door something goes through. They are properties on the pandas
        side and so take no argument there, and the direction has to cross as a
        flag rather than as part of the name only because the name is what
        `transform` already uses for something else.

        A column with a missing row in it is not monotonic in either direction,
        which is the core's rule and is also pandas', since a value that is not
        there cannot be said to be in order with respect to anything.

        Args:
            py_self: The series.
            increasing: True for increasing, False for decreasing.

        Returns:
            A Python bool.

        Raises:
            Error: Tagged `dtype`, if the column has a type with no order on it.
        """
        try:
            ref column = Self._held(py_self)[].series[]
            if flag(increasing, "increasing"):
                return PythonObject(column.is_monotonic_increasing())
            return PythonObject(column.is_monotonic_decreasing())
        except e:
            raise retagged(DTYPE, e)

    @staticmethod
    def temporal_part(
        py_self: PythonObject, kind: PythonObject, arg: PythonObject
    ) raises -> PythonObject:
        """Reads one part of a temporal column and hands back a column.

        The `dt` accessor's main door. Twenty five of its names take nothing and
        nine take one string, so the string crosses beside the name and the
        twenty five are handed an empty one, which is the argument
        `firepanda/py/temporal.mojo` makes at length.

        Args:
            py_self: The series.
            kind: The part, as pandas spells the attribute.
            arg: The frequency, unit, format, zone or locale, and the empty
                string for the ones that take none.

        Returns:
            A new series.

        Raises:
            Error: Tagged `dtype`, if the column has a type with no such part,
                and tagged `value` if the name is not one the accessor has.
        """
        var wanted = column_part(words(kind, "kind"))
        try:
            return PythonObject(
                alloc=Self(
                    ArcPointer(
                        temporal_part(
                            Self._held(py_self)[].series[],
                            wanted,
                            words(arg, "arg"),
                        )
                    )
                )
            )
        except e:
            raise retagged(DTYPE, e)

    @staticmethod
    def temporal_word(
        py_self: PythonObject, kind: PythonObject
    ) raises -> PythonObject:
        """Reads the clock or the resolution of a temporal column, as a string.

        A separate door from `temporal_part` because `tz` and `unit` answer a
        word rather than a column, which is the rule for how many doors there
        should be. `tz` comes back as the empty string when the column carries
        no zone, and the Python layer turns that into the `None` pandas answers,
        since a zone called nothing and no zone at all are the same thing and
        only one of them is spellable in a Mojo `String`.

        Args:
            py_self: The series.
            kind: Either `tz` or `unit`.

        Returns:
            A Python string.

        Raises:
            Error: Tagged `dtype`, if the column is not temporal, and tagged
                `value` if the name is not one of the two.
        """
        var wanted = word_part(words(kind, "kind"))
        try:
            return PythonObject(
                temporal_word(Self._held(py_self)[].series[], wanted)
            )
        except e:
            raise retagged(DTYPE, e)

    @staticmethod
    def to_datetime(
        py_self: PythonObject,
        fmt: PythonObject,
        unit: PythonObject,
        coerce: PythonObject,
        utc: PythonObject,
    ) raises -> PythonObject:
        """Reads a column of text or of whole numbers as a column of instants.

        `pandas.to_datetime` is a free function and this is a method, for the
        reason `Series.to_timedelta` gives: the column is the thing being read.
        The Python layer is where it is a free function again, and where a list
        or a tuple becomes a series before it arrives here, so everything that
        reaches this point is already a column.

        Every failure arrives as a `ValueError`, which is what pandas raises
        for a row that will not parse and for a column carrying offsets it
        cannot reconcile. pandas raises a `TypeError` for a column whose type
        has no reading at all, such as a column of booleans, and firepanda
        raises a `ValueError` there instead. That is a difference worth knowing
        about and it is recorded rather than hidden. See #353.

        Args:
            py_self: The series.
            fmt: The format the text is written in, or the empty string to work
                it out from the first row that is not missing.
            unit: What whole numbers are counts of.
            coerce: Whether a row that will not read becomes missing.
            utc: Whether to read every row against UTC.

        Returns:
            A new series of instants.

        Raises:
            Error: Tagged `value`, for a row that does not match the format, a
                column carrying more than one offset with no `utc`, and a
                column whose type has no reading.
        """
        try:
            return PythonObject(
                alloc=Self(
                    ArcPointer(
                        Self._held(py_self)[]
                        .series[]
                        .to_datetime(
                            words(fmt, "format"),
                            words(unit, "unit"),
                            flag(coerce, "coerce"),
                            flag(utc, "utc"),
                        )
                    )
                )
            )
        except e:
            raise retagged(VALUE, e)

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
            var mine = List[LogicalType](capacity=1)
            mine.append(Self._held(py_self)[].series[].logical())
            var theirs = List[LogicalType](capacity=1)
            theirs.append(right[].logical())
            raise retagged(binary_tag(which, mine, theirs), cause)

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
                operation is not defined on the two types. Tagged `overflow`, if
                the constant is a number too large for this series' dtype.
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
            var dtypes = List[LogicalType](capacity=1)
            dtypes.append(Self._held(py_self)[].series[].logical())
            raise retagged(constant_tag(dtypes, right, which), cause)

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
