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

from std.collections import Optional
from std.os import abort
from std.memory import ArcPointer, Pointer
from std.python import Python, PythonObject
from std.python.bindings import check_arguments_arity

from firepanda.array.any import AnyArray
from firepanda.array.strings import strings_from_list
from firepanda.dtype.logical import LogicalType, TypeKind, named_type
from firepanda.frame import DataFrame
from firepanda.frame.concat import concat_series
from firepanda.frame.index import Index
from firepanda.frame.series import Series
from firepanda.kernel.reduce import reduce_any
from firepanda.io.arrow_c import (
    ArrowArray,
    ArrowArrayStream,
    ArrowSchema,
    release_schema,
)
from firepanda.io.arrow_export import export_frame_array, export_frame_schema
from firepanda.io.arrow_stream import (
    export_frame_stream,
    import_frame,
    import_stream,
)
from firepanda.io.read import read_csv
from firepanda.py.args import flag, maybe_whole, number, whole, words
from firepanda.py.build import array_from, empty_column, frame_from
from firepanda.py.cast import (
    NOT_FINITE,
    checks_finite,
    refuse_if_not_finite,
)
from firepanda.py.convert import (
    array_capsule,
    schema_capsule,
    stream_capsule,
    take_array,
    take_schema,
    take_stream,
)
from firepanda.py.errors import (
    CANCELLED,
    COLUMN,
    DTYPE,
    IO,
    NONFINITE,
    OVERFLOW,
    POSITION,
    UNSUPPORTED,
    VALUE,
    reindex_refusal,
    retagged,
    tagged,
)
from firepanda.py.index import PyIndex
from firepanda.py.ops import (
    binary_failure,
    binary_op,
    constant,
    constant_tag,
    fill,
    unary_op,
)
from firepanda.kernel.group import AggKind
from firepanda.py.ewm import ewm_frame
from firepanda.py.reduce import grouped_reduction, reduction
from firepanda.py.series import PySeries
from firepanda.py.temporal import iso_calendar
from firepanda.py.transform import transformation, transformed
from firepanda.py.values import python_value
from firepanda.py.window import window_frame, window_settings


def _within(at: Int, extent: Int, what: String) raises -> Int:
    """Turns a position that may count from the end into one that does not.

    Args:
        at: The position, counting from the end when negative.
        extent: How many there are.
        what: The word for one of them, for the message.

    Returns:
        A position between zero and `extent`, exclusive.

    Raises:
        Error: Tagged `position`, if it lands outside.
    """
    var found = at + extent if at < 0 else at
    if found < 0 or found >= extent:
        raise tagged(
            POSITION,
            String(
                "index ",
                at,
                " is out of bounds for a frame with ",
                extent,
                " ",
                what,
                "s",
            ),
        )
    return found


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
    def dtypes(py_self: PythonObject) raises -> PythonObject:
        """Reports the column types, in order, as `dtype` spells them.

        This reads the schema and not the columns. Asking a frame what its
        types are through `column` would copy every column in order to look at
        the name of each one's type, which on a frame of five hundred columns
        is the whole frame copied to answer a question the schema already
        holds. The schema is the frame's shape and knowing the shape should
        not cost the contents.

        Args:
            py_self: The frame.

        Returns:
            A list of strings, one per column, in column order.
        """
        ref schema = Self._frame(py_self)[].frame[].schema
        var out = Python.list()
        for i in range(len(schema)):
            out.append(PythonObject(String(schema[i].dtype)))
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
                ArcPointer(Self._frame(py_self)[].frame[].head(whole(n, "n")))
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
                ArcPointer(Self._frame(py_self)[].frame[].tail(whole(n, "n")))
            )
        )

    @staticmethod
    def take(
        py_self: PythonObject, positions: PythonObject
    ) raises -> PythonObject:
        """Gathers rows by position, in the order asked for.

        A negative position counts from the end, which is pandas' rule and is
        not the core's: `DataFrame.take` answers a null row for a negative
        index, because that is what an outer join needs from it. So the
        counting back happens here, and every position is checked against the
        height afterwards, which is what stops a null row leaking out of a
        gather that a caller wrote as a selection.

        Args:
            py_self: The frame.
            positions: The rows to gather.

        Returns:
            A new frame with one row per position.

        Raises:
            Error: Tagged `position`, if a position is off either end.
        """
        ref frame = Self._frame(py_self)[].frame[]
        var height = len(frame)
        var picks = List[Int](capacity=Int(len(positions)))
        for i in range(Int(len(positions))):
            var at = whole(positions[i], "indices")
            if at < 0:
                at += height
            if at < 0 or at >= height:
                raise tagged(
                    POSITION,
                    String(
                        "positional indexers are out-of-bounds; index ",
                        positions[i],
                        " is not in a frame of ",
                        height,
                        " rows",
                    ),
                )
            picks.append(at)
        return PythonObject(alloc=Self(ArcPointer(frame.take(picks))))

    @staticmethod
    def slice_rows(
        py_self: PythonObject, start: PythonObject, end: PythonObject
    ) raises -> PythonObject:
        """Takes a half open range of rows.

        The bounds arrive already counted from the front and already clamped,
        because the Python layer got them from a Python slice and a Python
        slice has done both by the time it is read.

        Args:
            py_self: The frame.
            start: The first row, inclusive.
            end: The last row, exclusive.

        Returns:
            A new frame of `end - start` rows.

        Raises:
            Error: Tagged `position`, if the range runs off either end.
        """
        try:
            return PythonObject(
                alloc=Self(
                    ArcPointer(
                        Self._frame(py_self)[]
                        .frame[]
                        .slice(whole(start, "start"), whole(end, "end"))
                    )
                )
            )
        except cause:
            raise retagged(POSITION, cause)

    @staticmethod
    def filter_rows(
        py_self: PythonObject, mask: PythonObject
    ) raises -> PythonObject:
        """Keeps the rows a boolean column is true at.

        The mask crosses as a column rather than as a list of Python bools,
        which is the whole reason this is a method of its own instead of the
        Python layer turning a mask into positions: a mask that came out of
        `df["v"] > 0` is already a column on this side, and sending it back out
        as objects and in again as positions would cost two conversions to ask
        a question the kernel can answer from the bits it already has.

        Nothing is aligned. pandas matches a boolean series against the frame's
        labels and refuses an unalignable one, and this checks the length
        instead, which is the same check for the masks that come out of a
        comparison against the frame itself and a weaker one for the rest.

        Args:
            py_self: The frame.
            mask: A boolean column as tall as the frame.

        Returns:
            A new frame of the rows the mask kept.

        Raises:
            Error: Tagged `dtype`, if the column is not boolean, or `position`,
                if it is not as tall as the frame.
        """
        var right = PySeries._other(mask, "key")
        if right[].values.dtype() != DType.bool:
            raise tagged(
                DTYPE,
                String(
                    "cannot mask with a column of ",
                    right[].values.type_name(),
                    "; a boolean key has to be boolean",
                ),
            )
        try:
            return PythonObject(
                alloc=Self(
                    ArcPointer(
                        Self._frame(py_self)[]
                        .frame[]
                        .filter(right[].values.as_typed[DType.bool]())
                    )
                )
            )
        except cause:
            raise retagged(POSITION, cause)

    @staticmethod
    def cell(
        py_self: PythonObject, row: PythonObject, at: PythonObject
    ) raises -> PythonObject:
        """Reads one value out, by row and by column position.

        This is what `at` and `iat` reach, and it exists rather than being
        `column(name)` followed by a read because `column` copies: asking for
        one cell of a million row frame through a column would copy the million
        values to answer with one of them.

        Args:
            py_self: The frame.
            row: The row, counting from the end when negative.
            at: The column position, counting from the end when negative.

        Returns:
            The value, or `None` if it is missing.

        Raises:
            Error: Tagged `position`, if either coordinate is off its end.
        """
        ref frame = Self._frame(py_self)[].frame[]
        var down = _within(whole(row, "row"), len(frame), "row")
        var across = _within(whole(at, "at"), frame.width(), "column")
        ref chunked = frame.columns[across]
        var found = chunked.locate(down)
        return python_value(chunked.chunks[found[0]], found[1])

    @staticmethod
    def set_index(
        py_self: PythonObject, name: PythonObject, drop: PythonObject
    ) raises -> PythonObject:
        """Moves one column into the row labels.

        Args:
            py_self: The frame.
            name: The column to move.
            drop: Whether to take the column out of the frame.

        Returns:
            A new frame of the same height.
        """
        try:
            return PythonObject(
                alloc=Self(
                    ArcPointer(
                        Self._frame(py_self)[]
                        .frame[]
                        .set_index(words(name, "keys"), flag(drop, "drop"))
                    )
                )
            )
        except cause:
            raise retagged(COLUMN, cause)

    @staticmethod
    def reset_index(
        py_self: PythonObject, drop: PythonObject
    ) raises -> PythonObject:
        """Puts the row labels back to a count from zero.

        Args:
            py_self: The frame.
            drop: Whether to throw the old labels away rather than keeping them
                as the first column.

        Returns:
            A new frame of the same height.
        """
        try:
            return PythonObject(
                alloc=Self(
                    ArcPointer(
                        Self._frame(py_self)[]
                        .frame[]
                        .reset_index(flag(drop, "drop"))
                    )
                )
            )
        except cause:
            raise retagged(VALUE, cause)

    @staticmethod
    def sort_index(
        py_self: PythonObject, ascending: PythonObject
    ) raises -> PythonObject:
        """Puts the rows in the order of their labels.

        Args:
            py_self: The frame.
            ascending: Whether the labels increase.

        Returns:
            A new frame of the same height.
        """
        return PythonObject(
            alloc=Self(
                ArcPointer(
                    Self._frame(py_self)[]
                    .frame[]
                    .sort_index(flag(ascending, "ascending"))
                )
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
    def labels(py_self: PythonObject) raises -> PythonObject:
        """Hands out the row labels, as an index.

        This is what `df.index` reaches, and it is the reason the `Index` type is
        bound at all: every question the core index can answer had no way of
        being asked from a program before this existed.

        It copies, the way `column` copies, and for a default index that copy is
        two integers and no memory. A frame that has been gathered, filtered or
        sorted carries real labels and the copy is a column, which is the same
        trade `df["a"]` makes and the same argument in document 13 about why a
        borrowing version is not a small change.

        Args:
            py_self: The frame.

        Returns:
            A new index carrying the frame's labels.
        """
        return PythonObject(
            alloc=PyIndex(
                ArcPointer(Index(copy=Self._frame(py_self)[].frame[].index))
            )
        )

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
    def duplicated(
        py_self: PythonObject, subset: PythonObject, keep: PythonObject
    ) raises -> PythonObject:
        """Which rows repeat a key another row already carries.

        The rule arrives as a word and not as the value pandas spells it with.
        pandas writes the third rule as `False`, so two of its three settings are
        strings and one is a bool, and a boundary that carried that would be
        carrying a Python type to say which of three branches to take. The
        mixin turns it into `"none"` before the crossing, so one kind of thing
        comes across and the core reads one kind of thing.

        Args:
            py_self: The frame.
            subset: The column names that decide whether two rows are the same.
                Resolved from `None` to every column by the caller, because
                which columns a frame has is a question the mixin can already
                ask and sending an absence across to be filled in on the other
                side would put the default in two places.
            keep: `"first"`, `"last"` or `"none"`.

        Returns:
            A bool series as tall as the frame, carrying the frame's labels.
        """
        var wanted = List[String](capacity=Int(len(subset)))
        for name in subset:
            wanted.append(String(name))
        try:
            var out = (
                Self._frame(py_self)[].frame[].duplicated(wanted, String(keep))
            )
            return PythonObject(alloc=PySeries(ArcPointer(out^)))
        except cause:
            raise retagged(COLUMN, cause)

    @staticmethod
    def drop_duplicates(
        py_self: PythonObject, subset: PythonObject, keep: PythonObject
    ) raises -> PythonObject:
        """The frame with the repeated rows removed, by a chosen rule.

        Args:
            py_self: The frame.
            subset: The column names that decide whether two rows are the same.
            keep: `"first"`, `"last"` or `"none"`.

        Returns:
            A new frame holding the rows the rule spares, in input order.
        """
        var wanted = List[String](capacity=Int(len(subset)))
        for name in subset:
            wanted.append(String(name))
        try:
            return PythonObject(
                alloc=Self(
                    ArcPointer(
                        Self._frame(py_self)[]
                        .frame[]
                        .drop_duplicates(wanted, String(keep))
                    )
                )
            )
        except cause:
            raise retagged(COLUMN, cause)

    @staticmethod
    def top_rows(
        py_self: PythonObject,
        column: PythonObject,
        n: PythonObject,
        largest: PythonObject,
        keep: PythonObject,
    ) raises -> PythonObject:
        """The `n` best rows of a frame, by one column.

        One entry point for `nlargest` and `nsmallest` rather than two, because
        the two differ by a flag the whole way down and a second binding would
        be the same six lines with one word changed.

        Two things can go wrong under here and they are different kinds of
        wrong. A name that is not a column is pandas' `KeyError` and a column
        that cannot be ranked is its `TypeError`, and the core says both with a
        plain error. They are told apart by the message, the same way the
        import path does it above, so the rule is written down here: the only
        refusal in this path that is about a type uses the word numeric.

        Args:
            py_self: The frame.
            column: The name of the column to rank by.
            n: How many rows to keep.
            largest: True for the top of the column, False for the bottom.
            keep: `"first"` or `"last"`, checked by the caller.

        Returns:
            A new frame of the kept rows, best first.
        """
        var wanted = whole(n, "n")
        var high = flag(largest, "largest")
        try:
            return PythonObject(
                alloc=Self(
                    ArcPointer(
                        Self._frame(py_self)[]
                        .frame[]
                        ._top_rows(String(column), wanted, high, String(keep))
                    )
                )
            )
        except cause:
            if "numeric" in String(cause):
                raise retagged(DTYPE, cause)
            raise retagged(COLUMN, cause)

    @staticmethod
    def reindex(
        py_self: PythonObject,
        labels: PythonObject,
        fill_value: PythonObject,
    ) raises -> PythonObject:
        """Puts the frame on a set of row labels, whether it has them or not.

        The labels arrive as a Python sequence and go through the same builder
        that makes a column out of one, so the type of the labels is inferred
        the same way everywhere and an index of words is asked for with words.

        One thing can go wrong under here that pandas has a class for, which is
        a frame whose own labels repeat, and `reindex_refusal` is where that is
        told apart from the type errors and given pandas' own sentence.

        Args:
            py_self: The frame.
            labels: The row labels the result should have, in order.
            fill_value: What to put in a row whose label was not found, or
                `None` to leave it missing.

        Returns:
            A new frame of one row per label.
        """
        var wanted = array_from("labels", labels)
        var value = fill(fill_value)
        try:
            return PythonObject(
                alloc=Self(
                    ArcPointer(
                        Self._frame(py_self)[].frame[].reindex(wanted, value)
                    )
                )
            )
        except cause:
            raise reindex_refusal(cause)

    @staticmethod
    def reindex_columns(
        py_self: PythonObject,
        names: PythonObject,
        fill_value: PythonObject,
    ) raises -> PythonObject:
        """Puts the frame under a set of column names, in that order.

        The only refusal is a name asked for twice, which pandas answers rather
        than refusing: it hands back two columns under one name. A schema here
        cannot hold that, so it is reported as something the library does not do
        rather than as something the caller got wrong.

        Args:
            py_self: The frame.
            names: The column names the result should have, in order.
            fill_value: What to put in a column that is not there, or `None` to
                leave it missing.

        Returns:
            A new frame under those names.
        """
        var wanted = List[String](capacity=Int(len(names)))
        for name in names:
            wanted.append(String(name))
        var value = fill(fill_value)
        try:
            return PythonObject(
                alloc=Self(
                    ArcPointer(
                        Self._frame(py_self)[]
                        .frame[]
                        .reindex_columns(wanted, value)
                    )
                )
            )
        except cause:
            raise retagged(UNSUPPORTED, cause)

    @staticmethod
    def reduce(
        py_self: PythonObject, kind: PythonObject, param: PythonObject
    ) raises -> PythonObject:
        """Reduces every column to one value, and hands back a series of them.

        `df.sum()` in pandas is a series with one entry per column, labelled by
        the column names, and that is a different shape from the one row frame
        `DataFrame.agg_all` produces. The turn from one into the other is here
        rather than in the core because it is a pandas shape: a one row frame is
        the answer that keeps every column's own type, and a series is the answer
        that has to pick one type for all of them.

        Picking it is the whole of the work. Every result of the same type stays
        that type, since nothing has to be given up. A mix of numbers becomes
        float64, which is what pandas gives for the same frame and is the only
        type that holds an integer total and a float total at once. A mix that is
        not all numbers is refused rather than widened to text, because a series
        of strings that used to be a sum is not an answer anybody asked for.

        Args:
            py_self: The frame.
            kind: The reduction, as pandas spells the method.
            param: The delta degrees of freedom or the quantile, and zero for
                the reductions that take neither.

        Returns:
            A series with one row per column, labelled by the column names.

        Raises:
            Error: Tagged `dtype`, if a column has a type the reduction cannot
                read or the results have no type in common.
        """
        var wanted = reduction(words(kind, "kind"), number(param, "param"))
        ref frame = Self._frame(py_self)[].frame[]
        var names = List[String](capacity=frame.width())
        var parts = List[Series](capacity=frame.width())
        var same = True
        var numeric = True
        for i in range(frame.width()):
            names.append(frame.schema[i].name)
            var one: AnyArray
            try:
                one = reduce_any(frame.columns[i].only(), wanted)
            except cause:
                raise retagged(DTYPE, cause)
            if not one.type.is_numeric():
                numeric = False
            if len(parts) > 0 and one.type != parts[0].values.type:
                same = False
            parts.append(Series(names[i], one^))

        var labels = Index(
            AnyArray(strings_from_list(names)), Optional[String](None)
        )
        if len(parts) == 0:
            var out = Series(String(""), empty_column(0))
            out.index = labels^
            return PythonObject(alloc=PySeries(ArcPointer(out^)))

        if not same:
            if not numeric:
                raise tagged(
                    DTYPE,
                    String(
                        "the columns reduce to types with nothing in common, so"
                        " there is no one type the answers can share"
                    ),
                )
            for i in range(len(parts)):
                parts[i] = parts[i].cast(DType.float64)

        var out = concat_series(parts)
        out.name = String("")
        out.index = labels^
        return PythonObject(alloc=PySeries(ArcPointer(out^)))

    @staticmethod
    def transform(
        py_self: PythonObject, kind: PythonObject, periods: PythonObject
    ) raises -> PythonObject:
        """Applies one named transformation to every column.

        Eleven of the twelve, because a frame `dropna` is not this operation.
        The other eleven are per column and a frame is the columns run one at a
        time and put back together, which is what pandas does and is why
        `df.cumsum()` totals down each column rather than across the row.

        A column type can change on the way through and that is not a slip.
        `shift` on a complete integer column makes a gap and pandas has no
        integer that means absent, so the column widens to float64, which means
        a frame of integers can come back holding floats. The rule is the core's
        and it is the same one the read path applies to a file.

        Args:
            py_self: The frame.
            kind: The transformation, as pandas spells the method.
            periods: The `periods` or the `limit`, and zero for the ones that
                take neither.

        Returns:
            A new frame with the same column names in the same order.

        Raises:
            Error: Tagged `dtype`, if a column has a type the transformation
                cannot read, and tagged `value` if the name is not one of the
                eleven.
        """
        var wanted = transformation(words(kind, "kind"))
        if wanted == "dropna":
            raise tagged(
                VALUE,
                String(
                    "dropna on a frame removes rows rather than transforming"
                    " columns, so it does not come through here"
                ),
            )
        ref frame = Self._frame(py_self)[].frame[]
        var moved = whole(periods, "periods")
        var parts = List[Series](capacity=frame.width())
        for i in range(frame.width()):
            try:
                parts.append(
                    transformed(
                        frame.column(frame.schema[i].name), wanted, moved
                    )
                )
            except cause:
                raise retagged(DTYPE, cause)
        if len(parts) == 0:
            return PythonObject(alloc=Self(ArcPointer(DataFrame(copy=frame))))
        var out = DataFrame.from_series(parts^)
        out.index = Index(copy=frame.index)
        return PythonObject(alloc=Self(ArcPointer(out^)))

    @staticmethod
    def window_agg(
        py_self: PythonObject,
        kind: PythonObject,
        width: PythonObject,
        min_periods: PythonObject,
        center: PythonObject,
        closed: PythonObject,
        step: PythonObject,
        settings: PythonObject,
    ) raises -> PythonObject:
        """Runs one reduction over every window of every column.

        The frame half of the one door behind `Rolling` and `Expanding`. It
        takes the same seven arguments the column one takes and means the same
        thing by all seven of them, because a window is a pair of row numbers
        and a frame's columns all have the same rows.

        Args:
            py_self: The frame.
            kind: The reduction, as pandas spells the method.
            width: How many rows wide, or `None` for an expanding window.
            min_periods: How many values a window needs, or `None` for the
                default of whichever window type this is.
            center: Whether the window sits around its row.
            closed: Which of the two ends the window keeps.
            step: How many rows apart the answered rows are, or `None`.
            settings: The parameters the reduction reads and the window does
                not, as a tuple in the order pandas declares them, and empty for
                the eight reductions that read none.

        Returns:
            A new frame of the same column names in the same order, every one
            of them float64.

        Raises:
            Error: Tagged `dtype` if any column holds nothing a window can
                reduce, and tagged `value` if the parameters do not describe a
                window.
        """
        return PythonObject(
            alloc=Self(
                ArcPointer(
                    window_frame(
                        Self._frame(py_self)[].frame[],
                        words(kind, "kind"),
                        maybe_whole(width, "window"),
                        maybe_whole(min_periods, "min_periods"),
                        flag(center, "center"),
                        words(closed, "closed"),
                        maybe_whole(step, "step"),
                        window_settings(words(kind, "kind"), settings),
                    )
                )
            )
        )

    @staticmethod
    def ewm_agg(
        py_self: PythonObject,
        kind: PythonObject,
        alpha: PythonObject,
        min_periods: PythonObject,
        adjust: PythonObject,
        ignore_na: PythonObject,
        settings: PythonObject,
    ) raises -> PythonObject:
        """Runs one exponentially weighted reduction down every column.

        The frame half of the door behind `ewm`. It takes the same six arguments
        the column one takes and means the same thing by all six, because the
        decay runs down a column and every column of a frame has the same rows.
        That is pandas' `method="single"`, which is its default and the only
        reading this library answers.

        Args:
            py_self: The frame.
            kind: The reduction, as pandas spells the method.
            alpha: The smoothing factor, above nought and at most one.
            min_periods: How many values a row needs before it is answered.
            adjust: Whether every row weighs one rather than the factor.
            ignore_na: Whether a missing row is skipped rather than taking up a
                slot in the decay.
            settings: The parameters the reduction reads and the decay does not,
                as a tuple, which is `bias` for the two spreads and empty for
                the mean and the total.

        Returns:
            A new frame of the same column names in the same order, every one of
            them float64.

        Raises:
            Error: Tagged `dtype` if any column holds nothing a reduction can
                read, and tagged `value` if a parameter is out of range or the
                combination has no pandas answer.
        """
        return PythonObject(
            alloc=Self(
                ArcPointer(
                    ewm_frame(
                        Self._frame(py_self)[].frame[],
                        words(kind, "kind"),
                        number(alpha, "alpha"),
                        whole(min_periods, "min_periods"),
                        flag(adjust, "adjust"),
                        flag(ignore_na, "ignore_na"),
                        settings,
                    )
                )
            )
        )

    @staticmethod
    def cast(
        py_self: PythonObject,
        names: PythonObject,
        dtypes: PythonObject,
        strict: PythonObject,
    ) raises -> PythonObject:
        """Converts some columns to other types and hands back a new frame.

        Two lists rather than a mapping, because a mapping would have to be read
        out of a Python dict in an order the caller did not choose and the order
        matters here: a column named twice would be converted twice and the
        second answer would win silently. Paired lists keep the order the caller
        wrote and let the Python side be the one that decides what a repeat
        means.

        A name not in the frame is the caller's mistake and is reported as one,
        because pandas reports it too and a frame that quietly ignored it would
        hand back the column unconverted with nothing said.

        Args:
            py_self: The frame.
            names: The columns to convert.
            dtypes: The type for each of them, in the same order.
            strict: Whether a text value that is not a number raises rather
                than becoming a null.

        Returns:
            A new frame with those columns converted and the rest untouched.

        Raises:
            Error: Tagged `value` if the two lists are different lengths, a name
                is not one this layer prints or a text value is not a number,
                tagged `column` if a column is not in the frame, tagged
                `nonfinite` if an integer column was asked for and there is a
                missing value, a NaN or an infinity in the way, and tagged
                `dtype` if the conversion is not one firepanda has.
        """
        if len(names) != len(dtypes):
            raise tagged(
                VALUE,
                String(
                    "cast was given ",
                    len(names),
                    " columns and ",
                    len(dtypes),
                    " types, which have to come in pairs",
                ),
            )
        var strictly = flag(strict, "strict")
        var out = DataFrame(copy=Self._frame(py_self)[].frame[])
        for i in range(len(names)):
            var name = String(names[i])
            var wanted: LogicalType
            try:
                wanted = named_type(String(dtypes[i]))
            except cause:
                raise retagged(VALUE, cause)
            if not out.schema.has(name):
                raise tagged(COLUMN, String("no column named ", name))
            var text = out.schema[
                out.schema.index_of(name)
            ].dtype.is_variable_width()
            # `column` copies the column and flattens it, so it is only asked
            # for when the answer can matter, which the predicate decides.
            if checks_finite(wanted):
                refuse_if_not_finite(out.column(name).values, wanted)
            try:
                out = out.cast(name, wanted, strictly)
            except cause:
                # The source decides the tag, for the reason `PySeries.cast`
                # gives: out of text the only failure left is a value that will
                # not read, which is a value error in pandas.
                if text:
                    raise retagged(VALUE, cause)
                # And the same gap `PySeries.cast` names: a category out of a
                # column that is not text is a thing pandas does and firepanda
                # has not written, rather than a type error.
                if wanted.kind == TypeKind.DICTIONARY:
                    raise retagged(UNSUPPORTED, cause)
                raise retagged(DTYPE, cause)
        return PythonObject(alloc=Self(ArcPointer(out^)))

    @staticmethod
    def group_agg(
        py_self: PythonObject,
        by: PythonObject,
        kind: PythonObject,
        param: PythonObject,
        dropna: PythonObject,
        sort: PythonObject,
        as_index: PythonObject,
    ) raises -> PythonObject:
        """Groups the rows by some columns and reduces every other column.

        This is `df.groupby(keys).sum()` and its fourteen siblings, and it is one
        method rather than fifteen for the same reason `reduce` is one method
        rather than twelve. The reduction crosses as the word pandas spells it,
        the four that take a number carry it beside the word, and the Python
        layer holds the vocabulary. A generator that wrote fifteen bindings here
        would be writing fifteen copies of one call.

        `size` is the one word that does not reduce a column. It counts the rows
        in each group, so it answers one column called `size` and does not touch
        the others, which is why it goes to a different method in the core. It is
        still spelled here rather than given its own binding, because from
        Python it is `df.groupby(keys).size()` and looks exactly like the other
        fourteen.

        The three flags are pandas' own and all three mean what they mean there.
        `as_index` is the one worth knowing about: it puts the key into the row
        labels rather than leaving it as a column, it is the pandas default, and
        it needs exactly one key here because two keys are a MultiIndex in pandas
        and firepanda has none yet. The core raises for that case with the reason
        in the message rather than quietly handing back one level.

        Args:
            py_self: The frame.
            by: The key columns, at least one and no repeats.
            kind: The reduction, as pandas spells the method on a group.
            param: The delta degrees of freedom or the quantile, and zero for
                the reductions that take neither.
            dropna: Drop the groups whose key is missing, as pandas does.
            sort: Order the result by the key, as pandas does.
            as_index: Put the key in the row labels rather than in a column.

        Returns:
            A new frame with one row per group.

        Raises:
            Error: Tagged `column`, if a key is missing or named twice, tagged
                `value` if the name is not one of the fifteen or `as_index` was
                asked for with more than one key, and tagged `dtype` if a column
                has a type the reduction cannot read.
        """
        var keys = List[String](capacity=Int(len(by)))
        for name in by:
            keys.append(String(name))
        var wanted = grouped_reduction(
            words(kind, "kind"), number(param, "param")
        )
        var drop = flag(dropna, "dropna")
        var ordered = flag(sort, "sort")
        var indexed = flag(as_index, "as_index")
        ref frame = Self._frame(py_self)[].frame[]

        var out: DataFrame
        try:
            if wanted == AggKind.SIZE:
                out = frame.group_count(keys, drop, ordered, indexed)
            else:
                out = frame.group_agg(keys, wanted, drop, ordered, indexed)
        except cause:
            # The core raises one error type for three different mistakes here
            # and the tag decides which of the Python exceptions a user sees, so
            # the text is read rather than guessed at. Getting this wrong is not
            # a small thing: a missing column arriving as a TypeError is an
            # exception a pandas program's error handling does not catch.
            var text = String(cause)
            if "as_index" in text:
                raise retagged(VALUE, cause)
            raise retagged(COLUMN, cause)
        return PythonObject(alloc=Self(ArcPointer(out^)))

    @staticmethod
    def dropna(
        py_self: PythonObject, subset: PythonObject
    ) raises -> PythonObject:
        """Removes the rows that have a missing value in them.

        Its own door, because it is the one name on the transformation list that
        does something different to a frame than it does to a column. On a
        column it removes the missing values. On a frame it removes whole rows,
        which means the answer to `df.dropna()` depends on every column at once
        and no per column loop produces it.

        `subset` narrows which columns are allowed to disqualify a row. An empty
        list means all of them, which is the pandas default.

        Args:
            py_self: The frame.
            subset: The column names to look at, or an empty list for all of
                them.

        Returns:
            A new frame with the same columns and fewer rows.

        Raises:
            Error: Tagged `column`, if a named column is not in the frame.
        """
        var wanted = List[String](capacity=Int(len(subset)))
        for name in subset:
            wanted.append(String(name))
        try:
            return PythonObject(
                alloc=Self(
                    ArcPointer(
                        Self._frame(py_self)[].frame[].drop_nulls(wanted)
                    )
                )
            )
        except cause:
            raise retagged(COLUMN, cause)

    @staticmethod
    def _other(
        value: PythonObject, name: String
    ) raises -> ArcPointer[DataFrame]:
        """Recovers the frame out of an argument that should be one.

        Args:
            value: What Python passed.
            name: The parameter name, for the message.

        Returns:
            A share of the other frame, so the caller can read it without
            copying.

        Raises:
            Error: Tagged `dtype`, if the argument is not a frame.
        """
        try:
            return value.downcast_value_ptr[Self]()[].frame
        except:
            raise tagged(
                DTYPE,
                String(
                    name,
                    " must be a DataFrame, got ",
                    Python.type(value).__name__,
                    " ",
                    value.__repr__(),
                ),
            )

    @staticmethod
    def binary_frame(
        py_self: PythonObject,
        other: PythonObject,
        op: PythonObject,
        flip: PythonObject,
        fill_value: PythonObject,
    ) raises -> PythonObject:
        """Applies an operation to two frames, aligning on both axes.

        One entry point for twenty of the operators and named forms, with the
        operation crossing as a word. `firepanda/py/ops.mojo` says why the
        boundary is this shape rather than one bound method per operation.

        There is no `axis` here. Two frames align on their rows and on their
        columns whatever `axis` says, so pandas takes the argument and ignores
        it, and the Python layer drops it rather than carrying it across.

        Args:
            py_self: The left operand.
            other: The right operand, which has to be a frame.
            op: The operation, such as `add` or `lt`.
            flip: True for `other op self`, which is what a reflected form needs.
            fill_value: What to put where exactly one of the two sides has a
                row or a column, or `None`.

        Returns:
            A new frame, on the union of both axes.

        Raises:
            Error: Tagged `dtype`, if an argument is the wrong type or the
                operation is not defined on a pair of columns it reached.
        """
        var right = Self._other(other, "other")
        var spelling = words(op, "op")
        var which = binary_op(spelling)
        var filled = fill(fill_value)
        var flipped = flag(flip, "flip")
        try:
            return PythonObject(
                alloc=Self(
                    ArcPointer(
                        Self._frame(py_self)[]
                        .frame[]
                        .binary(right[], which, filled, flipped)
                    )
                )
            )
        except cause:
            raise binary_failure(
                spelling,
                which,
                Self._dtypes_of(Self._frame(py_self)[].frame[]),
                Self._dtypes_of(right[]),
                cause,
            )

    @staticmethod
    def binary_series(
        py_self: PythonObject,
        other: PythonObject,
        op: PythonObject,
        axis: PythonObject,
        flip: PythonObject,
    ) raises -> PythonObject:
        """Broadcasts a series across a frame, along one axis or the other.

        `axis` is the argument that matters here and it is the only place the
        operators cannot reach, because an operator has to pick an axis and
        pandas picks the columns. `df.add(s, axis=0)` is the only spelling of
        adding a series down the rows there is.

        There is no `fill_value`. pandas raises `NotImplementedError` for one
        here and so does the Python layer, before the call, because a series is
        broadcast rather than aligned cell by cell so there is no second side a
        fill could stand in for.

        Args:
            py_self: The frame.
            other: The series to broadcast.
            op: The operation, such as `add` or `lt`.
            axis: 1 to match the series' labels against the column names, 0 to
                match them against the row labels.
            flip: True for `series op frame`.

        Returns:
            A new frame.

        Raises:
            Error: Tagged `dtype`, if an argument is the wrong type or the
                operation is not defined on a pair it reached.
        """
        var right = PySeries._other(other, "other")
        var spelling = words(op, "op")
        var which = binary_op(spelling)
        var along = whole(axis, "axis")
        var flipped = flag(flip, "flip")
        try:
            return PythonObject(
                alloc=Self(
                    ArcPointer(
                        Self._frame(py_self)[]
                        .frame[]
                        .binary(right[], which, along, flipped)
                    )
                )
            )
        except cause:
            var theirs = List[LogicalType](capacity=1)
            theirs.append(right[].logical())
            raise binary_failure(
                spelling,
                which,
                Self._dtypes_of(Self._frame(py_self)[].frame[]),
                theirs,
                cause,
            )

    @staticmethod
    def binary_value(
        py_self: PythonObject,
        other: PythonObject,
        op: PythonObject,
        flip: PythonObject,
    ) raises -> PythonObject:
        """Applies an operation to every cell of a frame and one constant.

        Args:
            py_self: The frame.
            other: The constant.
            op: The operation, such as `add` or `lt`.
            flip: True for `constant op frame`, which is what `5 - df` needs.

        Returns:
            A new frame of the same shape, on the same labels.

        Raises:
            Error: Tagged `dtype`, if an argument is the wrong type or the
                operation is not defined on a column it reached. Tagged
                `overflow`, if the constant is a number too large for a column
                it reached.
        """
        var right = constant(other, "other")
        var which = binary_op(words(op, "op"))
        var flipped = flag(flip, "flip")
        try:
            return PythonObject(
                alloc=Self(
                    ArcPointer(
                        Self._frame(py_self)[]
                        .frame[]
                        .binary(right, which, flipped)
                    )
                )
            )
        except cause:
            # Every column, because one narrow column among wide ones is
            # exactly the case that fails, and the frame stopped at the first
            # one that had no answer without saying which it was.
            var dtypes = Self._dtypes_of(Self._frame(py_self)[].frame[])
            raise retagged(constant_tag(dtypes, right, which), cause)

    @staticmethod
    def _dtypes_of(frame: DataFrame) -> List[LogicalType]:
        """The dtype of every column, for a tag that has to look at all of them.

        A frame's arithmetic stops at the first pair with no answer and does not
        say which pair that was, so the binding cannot ask about one column and
        has to ask about the set.

        Args:
            frame: The frame.

        Returns:
            One dtype per column, left to right.
        """
        var dtypes = List[LogicalType](capacity=frame.width())
        for i in range(frame.width()):
            dtypes.append(frame.columns[i].type)
        return dtypes^

    @staticmethod
    def compare_frame(
        py_self: PythonObject, other: PythonObject, op: PythonObject
    ) raises -> PythonObject:
        """Compares two frames cell by cell, refusing to align them.

        This is what the six comparison operators do between two frames, and the
        refusal is pandas' rather than an implementation limit. The flexible
        `eq` and its five relatives are the ones that align, and the error names
        them.

        Two failures and two classes, as in `PySeries.compare_series`, with the
        difference that either axis can be the one that disagrees, so both are
        compared again on the failure path.

        Args:
            py_self: The left operand.
            other: The right operand, which has to be a frame.
            op: The comparison, such as `eq` or `lt`.

        Returns:
            A new frame of booleans.

        Raises:
            Error: Tagged `value` if the two are not labelled identically on both
                axes, and `dtype` if the comparison is not defined on a pair of
                columns it reached.
        """
        var right = Self._other(other, "other")
        var which = binary_op(words(op, "op"))
        var held = Self._frame(py_self)
        try:
            return PythonObject(
                alloc=Self(ArcPointer(held[].frame[].compare(right[], which)))
            )
        except cause:
            var mine = held[].frame
            if (
                not mine[].index.equals(right[].index)
                or mine[].names() != right[].names()
            ):
                raise retagged(VALUE, cause)
            raise retagged(DTYPE, cause)

    @staticmethod
    def unary(py_self: PythonObject, op: PythonObject) raises -> PythonObject:
        """Applies one of the four unary operations to every column.

        Args:
            py_self: The frame.
            op: The operation, one of `neg`, `pos`, `abs` or `invert`.

        Returns:
            A new frame of the same shape, on the same labels.

        Raises:
            Error: Tagged `dtype`, if the operation is not defined on a column's
                dtype.
        """
        var which = unary_op(words(op, "op"))
        try:
            return PythonObject(
                alloc=Self(
                    ArcPointer(Self._frame(py_self)[].frame[].unary(which))
                )
            )
        except cause:
            raise retagged(DTYPE, cause)

    @staticmethod
    def _borrowed(
        py_self: PythonObject,
    ) raises -> List[Pointer[AnyArray, MutAnyOrigin]]:
        """Points at every column of the frame, without copying any of them.

        A column with more than one chunk has no single Arrow array to be. The
        stream protocol is where that is expressible in principle, and the export
        hands out one batch per frame, so it is not expressible here either yet.
        The refusal happens before anything is allocated rather than half way
        through an export, and it is shared by both directions of the export for
        that reason.

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
                            " is stored in more than one chunk, and an export"
                            " hands out one batch, so there is nowhere for the"
                            " second chunk to go yet"
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

    @staticmethod
    def arrow_c_stream(
        py_self: PythonObject, requested_schema: PythonObject
    ) raises -> PythonObject:
        """Hands the frame out as an Arrow stream capsule, without copying.

        This is the Mojo half of `__arrow_c_stream__`, and it is the half of the
        protocol nearly every consumer reaches for first. DuckDB accepts nothing
        else, and `pyarrow.table`, `polars.DataFrame` and `pandas.DataFrame` all
        look for it before they look for an array.

        A firepanda frame has no chunking, so the stream is one batch and then
        the end. That is a stream a consumer cannot tell from any other, which is
        the point: what it costs to be read is the same either way.

        Args:
            py_self: The frame.
            requested_schema: A schema capsule the consumer would rather have, or
                `None`. Anything other than `None` is refused, for the reason
                `arrow_c_array` gives.

        Returns:
            A `PyCapsule` named `arrow_array_stream`.
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
        ref frame = Self._frame(py_self)[].frame[]
        var types = List[LogicalType](capacity=frame.width())
        for i in range(frame.width()):
            types.append(frame.columns[i].type)
        var names = frame.names()
        var rows = len(frame)
        var keep = Self._frame(py_self)[].frame
        try:
            return stream_capsule(
                export_frame_stream(columns^, types^, names^, rows, keep^)
            )
        except cause:
            raise retagged(UNSUPPORTED, cause)

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

    The frame is widened on the way out. `read_csv` is a pandas name with a
    pandas meaning, and what pandas means by it is that a column of numbers with
    a gap in it comes back as float64 with a NaN in the gap, because a numpy
    integer array has nowhere to record absence. Arrow does have somewhere, and
    this library uses it everywhere else, but a caller who typed `read_csv` has
    asked for the pandas reading of the file and should be handed the frame
    pandas would have handed them. `firepanda.from_arrow` is the door that keeps
    Arrow's answer, and `open_arrow` below says why. See #171 and document 20.

    Args:
        path: The path to read.

    Returns:
        A new frame.
    """
    try:
        return PythonObject(
            alloc=PyDataFrame(
                ArcPointer(read_csv(String(path)).widen_for_missing())
            )
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


def isocalendar(column: PythonObject) raises -> PythonObject:
    """Reads the three ISO 8601 week date fields of a column, as a frame.

    A free function rather than a method, and the reason is the import graph
    rather than taste. It reads a series and answers a frame, `series.mojo`
    cannot import `frame.mojo` because `frame.mojo` already imports it, and the
    only other place it could live is a method on the frame that takes a column,
    which reads backwards. So it is what it is: a function of a series that
    gives a frame, belonging to neither type.

    It is the third door of the `dt` accessor and the only member of it that
    answers a frame, which is what makes it a door rather than a name in the
    table. `firepanda/py/temporal.mojo` is that argument.

    Args:
        column: The series to read.

    Returns:
        A new frame of three columns, `year`, `week` and `day`.

    Raises:
        Error: Tagged `dtype`, if the column is not a date or a naive timestamp.
    """
    try:
        return PythonObject(
            alloc=PyDataFrame(
                ArcPointer(iso_calendar(PySeries._held(column)[].series[]))
            )
        )
    except cause:
        raise retagged(DTYPE, cause)


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
    if which == "nonfinite":
        raise tagged(NONFINITE, NOT_FINITE)
    if which == "overflow":
        raise tagged(OVERFLOW, "Python integer 128 out of bounds for int8")
    if which == "position":
        raise tagged(POSITION, "index 5 is out of bounds for an index of 2")
    if which == "io":
        raise tagged(IO, "no such file '/nowhere'")
    if which == "unsupported":
        raise tagged(UNSUPPORTED, "object dtype is not supported")
    if which == "cancelled":
        raise tagged(CANCELLED, "interrupted")
    if which == "untagged":
        raise Error("something went wrong a long way down")
    raise tagged(VALUE, String("no such kind ", kind.__repr__()))
