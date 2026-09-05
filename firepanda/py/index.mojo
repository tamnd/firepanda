"""The `Index` binding, which makes the row labels something a program can hold.

Same rules as `frame.mojo` and `series.mojo`. Nothing here is the pandas API,
everything here is flat named methods that the generated Python class calls, and
the reason for the split is document 13.

`Index` is the third bound type and the first one whose methods take another
bound object as an argument. `union`, `intersection`, `difference`,
`symmetric_difference`, `append`, `equals` and `identical` all do, and what that
costs is one helper, `_other`, which downcasts and complains by name when the
argument is not an index. The frame's own `_frame` recovery aborts on a failed
downcast because it can only fail through a bug in this file; this one raises,
because a user can reach it by writing `index.union([1, 2])`.

The other new thing is that several methods take labels rather than positions,
and a label is a one row Arrow column. `_labels` builds one through `array_from`
in `build.mojo`, the same reader the frame constructor uses, so an index infers
its dtype by exactly the rules document 18 wrote and there is no second
inference to keep in agreement.

### Why this is a type and not a list

The cheap version of `df.index` returns a Python list of labels. A caller who
gets one back cannot ask it whether it is unique, cannot union it with another,
and cannot hand it to anything, so every method the index grew stays unreachable
and the surface has to be built again later on top of the real type. The list is
`tolist`, which is one member on this.
"""

from std.os import abort
from std.memory import ArcPointer, Pointer
from std.python import Python, PythonObject
from std.python.bindings import check_arguments_arity

from firepanda.array.any import AnyArray
from firepanda.array.array import Array
from firepanda.array.strings import StringBuilder
from firepanda.dtype.lists import ALL
from firepanda.dtype.logical import LogicalType, TypeKind
from firepanda.frame.index import Index
from firepanda.io.arrow_export import export_array_borrowed, export_schema
from firepanda.py.args import flag, whole, words
from firepanda.py.build import array_from
from firepanda.py.convert import array_capsule, schema_capsule
from firepanda.py.errors import (
    COLUMN,
    DTYPE,
    POSITION,
    UNSUPPORTED,
    VALUE,
    retagged,
    tagged,
)
from firepanda.py.values import python_list, python_value


def _positions(value: PythonObject, name: String) raises -> List[Int]:
    """Reads a sequence of row numbers.

    Args:
        value: What Python passed, which the Python layer has already made a
            list of, so a bare integer does not arrive here.
        name: The parameter name, for the message.

    Returns:
        The positions, in the order they were given.

    Raises:
        Error: Tagged `dtype`, if the value is not a sequence of integers.
    """
    var builtins = Python.import_module("builtins")
    var items: PythonObject
    try:
        items = builtins.list(value)
    except:
        raise tagged(
            DTYPE,
            String(
                name,
                " must be a sequence of positions, got ",
                Python.type(value).__name__,
                " ",
                value.__repr__(),
            ),
        )
    var out = List[Int](capacity=Int(len(items)))
    for i in range(Int(len(items))):
        out.append(whole(items[i], name))
    return out^


def _mask(value: PythonObject, name: String) raises -> Array[DType.bool]:
    """Reads a sequence of booleans into a column.

    A `None` is a missing mask entry rather than a false, which matters because
    `putmask` treats them the same way and the difference would only show up if
    somebody later gave a null mask a different meaning.

    Args:
        value: What Python passed.
        name: The parameter name, for the message.

    Returns:
        The mask.

    Raises:
        Error: Tagged `dtype`, if the value is not a sequence of booleans.
    """
    var builtins = Python.import_module("builtins")
    var items: PythonObject
    try:
        items = builtins.list(value)
    except:
        raise tagged(
            DTYPE,
            String(
                name,
                " must be a sequence of booleans, got ",
                Python.type(value).__name__,
                " ",
                value.__repr__(),
            ),
        )
    var count = Int(len(items))
    var out = Array[DType.bool](count)
    for i in range(count):
        if items[i] is Python.none():
            out.set_null(i)
        else:
            out[i] = flag(items[i], name)
    return out^


def _labels(value: PythonObject, name: String) raises -> AnyArray:
    """Reads a sequence of labels into a column, inferring the dtype.

    Args:
        value: What Python passed.
        name: What to call it in a message.

    Returns:
        The labels.

    Raises:
        Error: Whatever `array_from` raises about the values.
    """
    return array_from(name, value)


def _one_null(type: LogicalType) raises -> AnyArray:
    """Builds a column of one missing value in a type the caller chose.

    `array_from` cannot do this, because a list holding nothing but `None` has
    no type in it and `empty_column` answers float64 for want of anything
    better. That answer is right when the column is being built out of the
    values and wrong when it is a label going into an index that already has a
    type: inserting a missing label into an int64 index has to produce an int64
    null, or the concatenation underneath refuses two columns of different
    types and the user is told about a float64 they never mentioned.

    Args:
        type: The type to build the null in.

    Returns:
        A column of one row, missing.

    Raises:
        Error: If the type has no physical layout this can build.
    """
    if type.kind == TypeKind.STRING or type.kind == TypeKind.BINARY:
        var builder = StringBuilder(1)
        builder.append_null()
        return AnyArray(builder^.finish())
    comptime for target in ALL:
        if type.physical == target:
            var out = Array[target](1)
            out.set_null(0)
            return AnyArray(out^)
    raise tagged(DTYPE, String("cannot build a missing ", type, " label"))


def _one_label(
    value: PythonObject, name: String, type: LogicalType
) raises -> AnyArray:
    """Reads a single label into a column of exactly one row.

    A bare string is a scalar here, which is the opposite of what `array_from`
    assumes, so the value is wrapped in a list before it gets there rather than
    being iterated into a column of characters.

    Args:
        value: What Python passed.
        name: What to call it in a message.
        type: The type of the index the label is going into, used only when the
            label is missing and there is nothing to infer a type from.

    Returns:
        A column of one row.

    Raises:
        Error: Whatever `array_from` raises about the value.
    """
    if value is Python.none():
        return _one_null(type)
    var one = Python.list()
    one.append(value)
    return array_from(name, one)


def _type(py_self: PythonObject) raises -> LogicalType:
    """Reads the type of an index's labels off the wrapper.

    Every method that takes a label needs this, so that a missing label is built
    in the index's own type rather than in the float64 `array_from` falls back
    to. It is a free function rather than a method because it is called from
    inside argument lists, where a second `Self.` would be noise.

    Args:
        py_self: The index.

    Returns:
        The label type.
    """
    return PyIndex._held(py_self)[].index[].logical()


@fieldwise_init
struct PyIndex(Movable, Writable):
    """A firepanda `Index` with a CPython object wrapped around it."""

    var index: ArcPointer[Index]
    """The labels themselves, shared rather than owned.

    Held the same way `PySeries` holds its column and for the same reason, which
    is document 15: an Arrow export hands out pointers into this memory and takes
    its own share, so a consumer can outlive the Python object and still be
    reading live memory.

    A range index shares two integers and no memory at all, which is the whole
    point of the type and survives being wrapped: `df.index` on a frame nobody
    reindexed allocates nothing except the Python object.
    """

    @staticmethod
    def py_init(
        out self: Self, args: PythonObject, kwargs: PythonObject
    ) raises:
        """Builds an index out of a Python sequence.

        Two positional arguments, the labels and the name, and nothing else. The
        pandas constructor takes six and the other four are refused by name on
        the Python side, for the reason `PyDataFrame.py_init` gives.

        The name is passed as a string or as `None`, and the two are different:
        pandas distinguishes an unnamed level from one named with the empty
        string, and so does the core, so the boundary has to carry both.

        Args:
            args: The labels and the name.
            kwargs: Keyword arguments, of which none are accepted, because the
                Python layer has already turned them into positional ones.
        """
        check_arguments_arity(2, args, "Index")
        var name = Optional[String]()
        if args[1] is not Python.none():
            name = Optional[String](words(args[1], "name"))
        if args[0] is Python.none():
            self = Self(ArcPointer(Index(0, 0, name^)))
        else:
            self = Self(ArcPointer(Index(_labels(args[0], "labels"), name^)))

    @staticmethod
    def _held(py_self: PythonObject) -> Pointer[Self, MutAnyOrigin]:
        """Recovers the Mojo value out of the Python object holding it.

        A failure here means the object is not a `PyIndex`, which the binding
        layer has already checked by the time a method body runs, so it is a bug
        in this file rather than a thing a caller can cause. `_other` is the one
        a caller can cause and it raises instead.

        Args:
            py_self: The Python object.

        Returns:
            A pointer to the wrapped value.
        """
        try:
            return py_self.downcast_value_ptr[Self]()
        except e:
            abort(String("not a firepanda Index: ", e))

    @staticmethod
    def _other(value: PythonObject, name: String) raises -> ArcPointer[Index]:
        """Recovers the index out of an argument that should be one.

        Args:
            value: What Python passed.
            name: The parameter name, for the message.

        Returns:
            A share of the other index, so the caller can read it without
            copying and without worrying about who lets go first.

        Raises:
            Error: Tagged `dtype`, if the argument is not an index.
        """
        try:
            return value.downcast_value_ptr[Self]()[].index
        except:
            raise tagged(
                DTYPE,
                String(
                    name,
                    " must be an Index, got ",
                    Python.type(value).__name__,
                    " ",
                    value.__repr__(),
                ),
            )

    @staticmethod
    def _wrap(var made: Index) raises -> PythonObject:
        """Puts a Python object around an index this file just built.

        Args:
            made: The index. Consumed.

        Returns:
            The Python object.
        """
        return PythonObject(alloc=Self(ArcPointer(made^)))

    @staticmethod
    def length(py_self: PythonObject) raises -> PythonObject:
        """Reports the label count.

        Args:
            py_self: The index.

        Returns:
            The number of labels.
        """
        return PythonObject(len(Self._held(py_self)[].index[]))

    @staticmethod
    def label(py_self: PythonObject) raises -> PythonObject:
        """Reports the level name, or `None` when there is not one.

        Called `label` rather than `name` for the reason `PySeries.label` gives,
        which is that `name` on the extension side collides with the attribute
        Python puts on a bound method.

        Args:
            py_self: The index.

        Returns:
            The name as a string, or `None`.
        """
        ref index = Self._held(py_self)[].index[]
        if index.name:
            return PythonObject(index.name.value())
        return Python.none()

    @staticmethod
    def dtype(py_self: PythonObject) raises -> PythonObject:
        """Reports the type of the labels, as firepanda spells it.

        A range answers `int64` without building itself out, which is the type it
        would have had if it were asked to.

        Args:
            py_self: The index.

        Returns:
            The type name, such as `int64` or `string`.
        """
        return PythonObject(String(Self._held(py_self)[].index[].logical()))

    @staticmethod
    def inferred_type(py_self: PythonObject) raises -> PythonObject:
        """Reports what pandas would call the kind of the labels.

        pandas has its own vocabulary here that is neither its dtype names nor
        numpy's: an int64 index is `integer`, a float64 index is `floating`, a
        string index is `string`, and a bool index is `boolean`. It is the one
        place in the type where the pandas spelling and the firepanda spelling
        differ on purpose, because the whole value of the member is that code
        written against pandas branches on those exact words.

        Args:
            py_self: The index.

        Returns:
            One of pandas' kind names.
        """
        var type = Self._held(py_self)[].index[].logical()
        if type.kind == TypeKind.BOOL:
            return PythonObject(String("boolean"))
        if type.is_float():
            return PythonObject(String("floating"))
        if type.is_integer():
            return PythonObject(String("integer"))
        return PythonObject(String(type))

    @staticmethod
    def is_range(py_self: PythonObject) raises -> PythonObject:
        """Reports whether the labels are still an arithmetic range.

        This is not a pandas member and the Python layer does not expose it. It
        is here because the Python `repr` has to say `RangeIndex` or `Index`, and
        the alternative was to make the Python side parse the rendered text to
        find out which one it got.

        Args:
            py_self: The index.

        Returns:
            True when no labels have been materialized.
        """
        return PythonObject(Self._held(py_self)[].index[].is_range())

    @staticmethod
    def start(py_self: PythonObject) raises -> PythonObject:
        """Reports the first label of a range, which is meaningless otherwise.

        Args:
            py_self: The index.

        Returns:
            The start.
        """
        return PythonObject(Self._held(py_self)[].index[].start)

    @staticmethod
    def nbytes(py_self: PythonObject) raises -> PythonObject:
        """Reports how much memory the labels occupy.

        A range occupies none, which is the honest answer and is not the one
        pandas gives: `pd.RangeIndex(5).nbytes` is 132, the size of the Python
        object rather than of any labels, and reporting that here would be
        copying a number that means nothing about the data.

        Args:
            py_self: The index.

        Returns:
            The bytes in the label buffers, zero for a range.
        """
        ref index = Self._held(py_self)[].index[]
        if index.is_range():
            return PythonObject(0)
        return PythonObject(index.labels.value().nbytes())

    @staticmethod
    def null_count(py_self: PythonObject) raises -> PythonObject:
        """Reports how many labels are missing.

        Args:
            py_self: The index.

        Returns:
            The count, zero for a range.
        """
        ref index = Self._held(py_self)[].index[]
        if index.is_range():
            return PythonObject(0)
        return PythonObject(index.labels.value().null_count())

    @staticmethod
    def at(py_self: PythonObject, i: PythonObject) raises -> PythonObject:
        """Reads one label out as a Python value.

        Args:
            py_self: The index.
            i: The position, counting from the end when negative.

        Returns:
            The label, or `None` if it is missing.
        """
        ref index = Self._held(py_self)[].index[]
        var at = whole(i, "i")
        if at < 0:
            at += len(index)
        if at < 0 or at >= len(index):
            raise tagged(
                POSITION,
                String(
                    "index ",
                    i,
                    " is out of bounds for an index of length ",
                    len(index),
                ),
            )
        if index.is_range():
            return PythonObject(index.start + at)
        return python_value(index.labels.value(), at)

    @staticmethod
    def to_list(py_self: PythonObject) raises -> PythonObject:
        """Copies every label out into a Python list.

        A range is built out to do it, which is the one place in the type where
        asking a cheap question is expensive. There is no way round it: the
        caller asked for the labels as objects and a range's labels do not exist
        until somebody asks.

        Args:
            py_self: The index.

        Returns:
            A list with one element per label, with `None` for the missing ones.
        """
        return python_list(Self._held(py_self)[].index[].materialize())

    @staticmethod
    def slice_rows(
        py_self: PythonObject, start: PythonObject, end: PythonObject
    ) raises -> PythonObject:
        """Takes a half open range of rows.

        A range stays a range, so this is what makes a slice of a default index
        cost nothing.

        Args:
            py_self: The index.
            start: The first row.
            end: One past the last row.

        Returns:
            A new index.
        """
        try:
            return Self._wrap(
                Self._held(py_self)[]
                .index[]
                .slice(whole(start, "start"), whole(end, "end"))
            )
        except cause:
            raise retagged(POSITION, cause)

    @staticmethod
    def take(
        py_self: PythonObject, positions: PythonObject
    ) raises -> PythonObject:
        """Gathers labels by position.

        Args:
            py_self: The index.
            positions: The rows to take, in the order they should come out.

        Returns:
            A new index.
        """
        try:
            return Self._wrap(
                Self._held(py_self)[]
                .index[]
                .take(_positions(positions, "positions"))
            )
        except cause:
            raise retagged(POSITION, cause)

    @staticmethod
    def is_unique(py_self: PythonObject) raises -> PythonObject:
        """Reports whether every label appears once.

        Args:
            py_self: The index.

        Returns:
            True when no label repeats.
        """
        try:
            return PythonObject(Self._held(py_self)[].index[].is_unique())
        except cause:
            raise retagged(DTYPE, cause)

    @staticmethod
    def is_monotonic_increasing(py_self: PythonObject) raises -> PythonObject:
        """Reports whether the labels never decrease.

        Args:
            py_self: The index.

        Returns:
            True when the labels ascend.
        """
        try:
            return PythonObject(
                Self._held(py_self)[].index[].is_monotonic_increasing()
            )
        except cause:
            raise retagged(DTYPE, cause)

    @staticmethod
    def is_monotonic_decreasing(py_self: PythonObject) raises -> PythonObject:
        """Reports whether the labels never increase.

        Args:
            py_self: The index.

        Returns:
            True when the labels descend.
        """
        try:
            return PythonObject(
                Self._held(py_self)[].index[].is_monotonic_decreasing()
            )
        except cause:
            raise retagged(DTYPE, cause)

    @staticmethod
    def get_loc(
        py_self: PythonObject, label: PythonObject
    ) raises -> PythonObject:
        """Reports every position one label sits at.

        The Python layer turns this into the three shapes pandas returns, which
        are an integer, a slice and a mask. Positions are what all three are made
        of, so the choice is made once over there rather than three times here.

        Args:
            py_self: The index.
            label: The label to find.

        Returns:
            The positions, in order.
        """
        try:
            var found = (
                Self._held(py_self)[]
                .index[]
                .get_loc(_one_label(label, "label", _type(py_self)))
            )
            var out = Python.list()
            for i in range(len(found)):
                out.append(PythonObject(found[i]))
            return out
        except cause:
            raise retagged(COLUMN, cause)

    @staticmethod
    def get_indexer(
        py_self: PythonObject, target: PythonObject
    ) raises -> PythonObject:
        """Reports where each of a set of labels sits, with -1 for the missing.

        Args:
            py_self: The index.
            target: The labels to find.

        Returns:
            One position per label asked for.
        """
        try:
            var found = (
                Self._held(py_self)[]
                .index[]
                .get_indexer(_labels(target, "target"))
            )
            var out = Python.list()
            for i in range(len(found)):
                out.append(PythonObject(Int(found[i])))
            return out
        except cause:
            raise retagged(VALUE, cause)

    @staticmethod
    def contains(
        py_self: PythonObject, label: PythonObject
    ) raises -> PythonObject:
        """Reports whether a label is in the index.

        `get_indexer` cannot answer this, because it refuses a non unique index
        and `1 in Index([1, 1])` is a fair question. `get_indexer_for` is the
        form that does not refuse, and this is the one place the Python surface
        reaches it.

        Args:
            py_self: The index.
            label: The label to look for.

        Returns:
            True when the label is there.
        """
        try:
            var found = (
                Self._held(py_self)[]
                .index[]
                .get_indexer_for(_one_label(label, "label", _type(py_self)))
            )
            return PythonObject(len(found) > 0 and found[0] >= 0)
        except:
            # A label of the wrong dtype is not in the index, which is what
            # `in` should say. pandas agrees: `"a" in Index([1, 2])` is False
            # rather than a TypeError, and a containment test that raises is
            # one nobody can write a condition around.
            return PythonObject(False)

    @staticmethod
    def equals(
        py_self: PythonObject, other: PythonObject
    ) raises -> PythonObject:
        """Reports whether two indexes hold the same labels, ignoring names.

        Args:
            py_self: The index.
            other: The other index.

        Returns:
            True when the labels match.
        """
        var against = Self._other(other, "other")
        try:
            return PythonObject(Self._held(py_self)[].index[].equals(against[]))
        except cause:
            raise retagged(DTYPE, cause)

    @staticmethod
    def identical(
        py_self: PythonObject, other: PythonObject
    ) raises -> PythonObject:
        """Reports whether two indexes hold the same labels under the same name.

        Args:
            py_self: The index.
            other: The other index.

        Returns:
            True when the labels and the name both match.
        """
        var against = Self._other(other, "other")
        try:
            return PythonObject(
                Self._held(py_self)[].index[].identical(against[])
            )
        except cause:
            raise retagged(DTYPE, cause)

    @staticmethod
    def same_as(
        py_self: PythonObject, other: PythonObject
    ) raises -> PythonObject:
        """Reports whether two indexes are the same object underneath.

        This is pandas' `is_`, which asks about identity rather than about
        labels, and it is a real question here rather than a curiosity: two
        Python wrappers around one shared index are two objects and one index,
        and `is` in Python would say no while pandas says yes.

        Args:
            py_self: The index.
            other: The other index.

        Returns:
            True when both wrappers share one set of labels.
        """
        var against = Self._other(other, "other")
        return PythonObject(
            Int(Pointer(to=Self._held(py_self)[].index[]))
            == Int(Pointer(to=against[]))
        )

    @staticmethod
    def unique(py_self: PythonObject) raises -> PythonObject:
        """Keeps the first of each label.

        Args:
            py_self: The index.

        Returns:
            A new index with no repeats, in first seen order.
        """
        try:
            return Self._wrap(Self._held(py_self)[].index[].unique())
        except cause:
            raise retagged(DTYPE, cause)

    @staticmethod
    def renamed(
        py_self: PythonObject, name: PythonObject
    ) raises -> PythonObject:
        """Returns the index under a different level name.

        Args:
            py_self: The index.
            name: The new name, or `None` for unnamed.

        Returns:
            A copy carrying the new name.
        """
        var wanted = Optional[String]()
        if name is not Python.none():
            wanted = Optional[String](words(name, "name"))
        return Self._wrap(Self._held(py_self)[].index[].renamed(wanted^))

    @staticmethod
    def union(
        py_self: PythonObject, other: PythonObject, sort: PythonObject
    ) raises -> PythonObject:
        """Every label either side has, keeping the larger of the two counts.

        Args:
            py_self: The index.
            other: The other index.
            sort: Whether to sort the result.

        Returns:
            A new index.
        """
        var against = Self._other(other, "other")
        try:
            return Self._wrap(
                Self._held(py_self)[]
                .index[]
                .union(against[], flag(sort, "sort"))
            )
        except cause:
            raise retagged(DTYPE, cause)

    @staticmethod
    def intersection(
        py_self: PythonObject, other: PythonObject, sort: PythonObject
    ) raises -> PythonObject:
        """Every label both sides have, once each.

        Args:
            py_self: The index.
            other: The other index.
            sort: Whether to sort the result.

        Returns:
            A new index.
        """
        var against = Self._other(other, "other")
        try:
            return Self._wrap(
                Self._held(py_self)[]
                .index[]
                .intersection(against[], flag(sort, "sort"))
            )
        except cause:
            raise retagged(DTYPE, cause)

    @staticmethod
    def difference(
        py_self: PythonObject, other: PythonObject, sort: PythonObject
    ) raises -> PythonObject:
        """Every label this index has and the other does not, once each.

        Args:
            py_self: The index.
            other: The other index.
            sort: Whether to sort the result.

        Returns:
            A new index.
        """
        var against = Self._other(other, "other")
        try:
            return Self._wrap(
                Self._held(py_self)[]
                .index[]
                .difference(against[], flag(sort, "sort"))
            )
        except cause:
            raise retagged(DTYPE, cause)

    @staticmethod
    def symmetric_difference(
        py_self: PythonObject,
        other: PythonObject,
        sort: PythonObject,
        result_name: PythonObject,
    ) raises -> PythonObject:
        """Every label exactly one of the two sides has, once each.

        Args:
            py_self: The index.
            other: The other index.
            sort: Whether to sort the result.
            result_name: A name to put on the result, or `None` to apply the
                usual rule, which is to keep the name only when both sides
                agree.

        Returns:
            A new index.
        """
        var against = Self._other(other, "other")
        var named = Optional[String]()
        if result_name is not Python.none():
            named = Optional[String](words(result_name, "result_name"))
        try:
            return Self._wrap(
                Self._held(py_self)[]
                .index[]
                .symmetric_difference(against[], named^, flag(sort, "sort"))
            )
        except cause:
            raise retagged(DTYPE, cause)

    @staticmethod
    def append(
        py_self: PythonObject, others: PythonObject
    ) raises -> PythonObject:
        """Puts one or several indexes on the end of this one.

        The Python layer always passes a list, even for the one index form, so
        there is one shape to read here rather than two.

        Args:
            py_self: The index.
            others: The indexes to append, in order.

        Returns:
            A new index as long as all the sides together.
        """
        var builtins = Python.import_module("builtins")
        var items = builtins.list(others)
        var rest = List[Index]()
        for i in range(Int(len(items))):
            var one = Self._other(items[i], "other")
            rest.append(Index(copy=one[]))
        try:
            return Self._wrap(Self._held(py_self)[].index[].append(rest))
        except cause:
            raise retagged(DTYPE, cause)

    @staticmethod
    def delete(
        py_self: PythonObject, positions: PythonObject
    ) raises -> PythonObject:
        """Removes the labels at a set of positions.

        Args:
            py_self: The index.
            positions: The rows to remove, counting from the end when negative.

        Returns:
            A shorter index.
        """
        try:
            return Self._wrap(
                Self._held(py_self)[]
                .index[]
                .delete(_positions(positions, "positions"))
            )
        except cause:
            raise retagged(POSITION, cause)

    @staticmethod
    def insert(
        py_self: PythonObject, position: PythonObject, label: PythonObject
    ) raises -> PythonObject:
        """Puts one label in at a position.

        Args:
            py_self: The index.
            position: Where it lands, from zero to the length inclusive.
            label: The label.

        Returns:
            An index one row longer.
        """
        var built = _one_label(label, "item", _type(py_self))
        try:
            return Self._wrap(
                Self._held(py_self)[]
                .index[]
                .insert(whole(position, "loc"), built)
            )
        except cause:
            raise retagged(POSITION, cause)

    @staticmethod
    def drop(
        py_self: PythonObject, labels: PythonObject, errors: PythonObject
    ) raises -> PythonObject:
        """Removes every row carrying one of a set of labels.

        Args:
            py_self: The index.
            labels: The labels to remove.
            errors: `"raise"` or `"ignore"`.

        Returns:
            A shorter index.
        """
        var built = _labels(labels, "labels")
        try:
            return Self._wrap(
                Self._held(py_self)[]
                .index[]
                .drop(built, words(errors, "errors"))
            )
        except cause:
            raise retagged(COLUMN, cause)

    @staticmethod
    def putmask(
        py_self: PythonObject, mask: PythonObject, value: PythonObject
    ) raises -> PythonObject:
        """Replaces the labels a mask picks out.

        Args:
            py_self: The index.
            mask: Which rows to replace.
            value: The replacement, one label or as many as the index is long.
                The Python layer always passes a list, so a scalar arrives here
                as a list of one.

        Returns:
            An index of the same length.
        """
        var built = _labels(value, "value")
        try:
            return Self._wrap(
                Self._held(py_self)[]
                .index[]
                .putmask(_mask(mask, "mask"), built)
            )
        except cause:
            raise retagged(VALUE, cause)

    @staticmethod
    def get_slice_bound(
        py_self: PythonObject, label: PythonObject, side: PythonObject
    ) raises -> PythonObject:
        """Reports where a label sits when the index is read as an ordered thing.

        Args:
            py_self: The index.
            label: The label.
            side: `"left"` or `"right"`.

        Returns:
            A position between zero and the length of the index.
        """
        var built = _one_label(label, "label", _type(py_self))
        try:
            return PythonObject(
                Self._held(py_self)[]
                .index[]
                .get_slice_bound(built, words(side, "side"))
            )
        except cause:
            raise retagged(COLUMN, cause)

    @staticmethod
    def slice_locs(
        py_self: PythonObject, start: PythonObject, end: PythonObject
    ) raises -> PythonObject:
        """Reports the half open row range a pair of labels describes.

        Args:
            py_self: The index.
            start: The first label, or `None` for the start of the index.
            end: The last label, or `None` for the end of the index.

        Returns:
            A list of two positions, which the Python layer hands back as a
            tuple.
        """
        var first = Optional[AnyArray]()
        if start is not Python.none():
            first = Optional[AnyArray](
                _one_label(start, "start", _type(py_self))
            )
        var last = Optional[AnyArray]()
        if end is not Python.none():
            last = Optional[AnyArray](_one_label(end, "end", _type(py_self)))
        try:
            var found = Self._held(py_self)[].index[].slice_locs(first^, last^)
            var out = Python.list()
            out.append(PythonObject(found[0]))
            out.append(PythonObject(found[1]))
            return out
        except cause:
            raise retagged(COLUMN, cause)

    @staticmethod
    def slice_indexer(
        py_self: PythonObject,
        start: PythonObject,
        end: PythonObject,
        step: PythonObject,
    ) raises -> PythonObject:
        """Reports the same range as `slice_locs` with the step carried through.

        Args:
            py_self: The index.
            start: The first label, or `None` for the start of the index.
            end: The last label, or `None` for the end of the index.
            step: The stride, returned unchanged.

        Returns:
            A list of three numbers, which the Python layer turns into a slice.
        """
        var first = Optional[AnyArray]()
        if start is not Python.none():
            first = Optional[AnyArray](
                _one_label(start, "start", _type(py_self))
            )
        var last = Optional[AnyArray]()
        if end is not Python.none():
            last = Optional[AnyArray](_one_label(end, "end", _type(py_self)))
        try:
            var found = (
                Self._held(py_self)[]
                .index[]
                .slice_indexer(first^, last^, whole(step, "step"))
            )
            var out = Python.list()
            out.append(PythonObject(found[0]))
            out.append(PythonObject(found[1]))
            out.append(PythonObject(found[2]))
            return out
        except cause:
            raise retagged(COLUMN, cause)

    @staticmethod
    def arrow_c_schema(py_self: PythonObject) raises -> PythonObject:
        """Describes the labels as an Arrow schema capsule.

        pandas has no `__arrow_c_schema__` on an `Index` at all, so this is an
        addition rather than a divergence. It is worth having because an index is
        one Arrow array and every consumer that reads a series can read it.

        The field is named after the level, or left empty when the index is
        unnamed, which is the same choice `PySeries.arrow_c_schema` makes about a
        column name.

        Args:
            py_self: The index.

        Returns:
            A `PyCapsule` named `arrow_schema`.
        """
        ref index = Self._held(py_self)[].index[]
        var named = index.name.value() if index.name else String("")
        try:
            return schema_capsule(export_schema(index.logical(), named))
        except cause:
            raise retagged(UNSUPPORTED, cause)

    @staticmethod
    def arrow_c_array(
        py_self: PythonObject, requested_schema: PythonObject
    ) raises -> PythonObject:
        """Hands the labels out as an Arrow array capsule.

        A materialized index exports its own buffers and copies nothing, the same
        way a series does. A range has no buffers to export, so it is built out
        first and what leaves is a copy, which is the honest cost of asking a
        thing that stores no labels for its labels.

        Args:
            py_self: The index.
            requested_schema: A schema capsule the consumer would rather have, or
                `None`. Anything else is refused, for the reason the frame gives.

        Returns:
            A list of two capsules, the schema and the array.
        """
        if requested_schema is not Python.none():
            raise tagged(
                UNSUPPORTED,
                (
                    "requested_schema is not supported yet; pass None and cast"
                    " the result instead"
                ),
            )
        var keep = Self._held(py_self)[].index
        if keep[].is_range():
            keep = ArcPointer(
                Index(keep[].materialize(), Optional[String](copy=keep[].name))
            )
        var column = Pointer(to=keep[].labels.value()).unsafe_origin_cast[
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
        """Writes the index the way the core writes it.

        Args:
            writer: Where to write.
        """
        writer.write(self.index[])

    def write_repr_to(self, mut writer: Some[Writer]):
        """Writes the index. This is what Python sees for both `str` and `repr`.

        Args:
            writer: Where to write.
        """
        writer.write(self.index[])
