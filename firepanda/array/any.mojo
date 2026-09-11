"""The type-erased column.

A frame holds columns of different types in one list, so at the frame boundary
the dtype has to stop being a parameter and start being a value. `AnyArray` is
that boundary. It holds the same storage `Array[dt]` holds plus the dtype as a
field, and it hands out a typed column through `as_typed` and `as_typed_view`.
The first copies and the second borrows, and anything that only reads should be
using the second.

`as_typed[dt]()` is a checked reinterpretation, not a conversion. It raises if the
requested dtype does not match the one in the field. That check is the only thing
standing between a wrong dispatch and reading an int64 column as float64, so it is
never elided, not even in release; the cost is one comparison against a value that
is in cache and predicted.

A string column is the one thing that does not fit in `ColumnData`, because its
elements are not all the same width. It is carried in `text` alongside, and `data`
holds an empty values buffer, the length, and a copy of the validity.

The copy of the validity is the one piece of duplication in this struct and it is
deliberate. Every kernel that only looks at whether a row is present, which is
`is_null`, `is_not_null` and the all-present mask a join and a group by both build,
reads `data.validity` and does not care what the values are. Keeping the bitmap
where those already look means they need no string case at all, and a column is
immutable once constructed, so the two copies are written from the same source in
the same call and cannot drift. The cost is one bit per row per string column, and
it goes away when columns are refcounted rather than deep copied.

What does need a string case is anything that reads values, because
`LogicalType.STRING` has physical dtype uint8 and would otherwise match the `uint8`
arm of a dispatch and read the first byte of a 16 byte view as the value. Every one
of those call sites asks `is_string()` first.

A list or a struct column does not fit either, and it goes further than a string
does: its values are not in this struct at all. They are in the child columns in
`nested`, and what `data` holds is one offset per row for a list and nothing but
the validity for a struct. `check_dtype` refuses both, so the call sites that ask
`is_string()` need no nested case, and the ones that want the children ask for
them by name through `child` and `field`.
"""

from std.memory import unsafe_memcpy

from firepanda.bitmap.bitmap import Bitmap
from firepanda.buffer.buffer import Buffer
from firepanda.dtype.lists import dtype_size
from firepanda.dtype.logical import LogicalType, TypeKind, logical_for

from .array import Array
from .data import ColumnData
from .nested import (
    ROOT,
    NestedNode,
    field_named,
    nested_type_name,
    subtree_of,
)
from .strings import StringArray


comptime ColumnRefs[o: ImmOrigin] = List[Pointer[AnyArray, o]]
"""A borrowed set of columns.

Anything that reads several of a frame's columns wants them borrowed. Taking a
`List[AnyArray]` means the caller either gives up ownership or copies, and for a
group by on six key columns of ten million rows the copy is more work than the
group by itself. This costs a pointer a column and the caller keeps what it has.

The origin is carried rather than erased, and that is not a formality. A borrow
with an untracked origin lets the compiler destroy the frame after the argument
is evaluated and before the callee runs, and what the callee then reads is freed
memory that still looks like a column: `group_ordinals` on a frame built inline
returned one group for eight rows with three distinct keys rather than crashing.
With the origin in the type that program does not compile.
"""


struct AnyArray(Copyable, Movable, Sized):
    """A column whose dtype is a runtime value."""

    var data: ColumnData
    """The values buffer, validity bitmap and length."""

    var type: LogicalType
    """The column type, carried as data rather than as a parameter."""

    var text: Optional[StringArray]
    """The variable width elements, present only for a string column."""

    var dict_values: Optional[StringArray]
    """The categories a dictionary column's codes refer to, present only for a
    dictionary column.

    Kept apart from `text` rather than reusing it, and the separation is the
    point. These are one entry per category and `text` is one entry per row, so
    a dictionary column sharing the field would answer `is_string` yes and hand
    four categories back to a caller that asked for a million rows of values.
    Two fields cost an eight byte discriminant on a struct that is already three
    buffers wide, and they make that mistake impossible to write.
    """

    var nested: List[NestedNode]
    """The child columns of a list or a struct, flattened, and empty for every
    other type.

    A plain list rather than an `Optional` of one, because a list already has an
    empty state and it means exactly what absent would mean here. `is_nested`
    asks the type rather than this field, so that a struct with no fields at all,
    which Arrow allows, is still a struct.

    The tree is read back through the helpers in `nested.mojo`. The column itself
    is the root and is not in here: a list column's offsets are in `data.values`
    and its row count in `data.length`, the same fields any other column uses,
    and these are the nodes below it.
    """

    var _slack: UInt64
    """Eight bytes nothing reads, here to keep the fields above from stopping
    short of the struct's own alignment.

    Mojo miscompiles an `Optional` of a struct that has trailing padding. The
    value goes in and the `Optional` still answers that it is empty, so a column
    put in one comes back as though it had never been there. A frame's index
    holds its labels in exactly that shape, which is how this was found: every
    index in the library silently became the range 0, 1, 2 the moment this struct
    grew a field that left it eight bytes short of sixteen.

    The fields above come to 472 bytes and the alignment is 16, so this is the 8
    that make it 480 and land on the boundary. `test_a_column_has_no_trailing
    _padding` fails if a later field breaks that again, which is the only warning
    there is: nothing about the miscompile happens at compile time.

    Issue #286 carries the bisection and takes this field back out when a
    toolchain lands that does not need it.
    """

    def __init__(out self, var data: ColumnData, type: LogicalType):
        """Constructs a type-erased column over storage the caller built.

        Args:
            data: The storage.
            type: The column type.
        """
        self.data = data^
        self.type = type
        self.text = None
        self.dict_values = None
        self.nested = List[NestedNode]()
        self._slack = 0

    def __init__[dt: DType](out self, var typed: Array[dt]):
        """Erases the dtype of a typed array, taking ownership of its buffers.

        Args:
            typed: The array to erase. Consumed, with no copy of the buffers.

        Parameters:
            dt: The dtype being erased.
        """
        self.data = typed^.into_data()
        self.type = logical_for(dt)
        self.text = None
        self.dict_values = None
        self.nested = List[NestedNode]()
        self._slack = 0

    def __init__(out self, var strings: StringArray):
        """Erases a string column, taking ownership of its buffers.

        Args:
            strings: The column to erase. Consumed.
        """
        var length = len(strings)
        self.data = ColumnData(Buffer(0), Bitmap(copy=strings.validity), length)
        self.type = LogicalType.STRING
        self.text = strings^
        self.dict_values = None
        self.nested = List[NestedNode]()
        self._slack = 0

    @staticmethod
    def dictionary[
        dt: DType
    ](
        var codes: Array[dt], var categories: StringArray, ordered: Bool = False
    ) -> Self:
        """Builds a dictionary column from its codes and its categories.

        Nothing here checks that every code is in range. The check costs a pass
        over the column, and a caller that built the codes itself already knows
        they are good. The one caller that does not is the Arrow reader, which
        is handed them by somebody else, and it does the pass itself in
        `attach_dictionary`.

        Args:
            codes: One position per row, pointing into the categories.
            categories: The distinct values, held once each.
            ordered: Whether the categories have a meaningful order.

        Parameters:
            dt: The dtype of the codes.

        Returns:
            The column.
        """
        var out = Self(codes^.into_data(), LogicalType.dictionary(dt, ordered))
        out.dict_values = categories^
        return out^

    @staticmethod
    def nested_from(var nodes: List[NestedNode]) raises -> Self:
        """Turns a flattened tree whose root is node zero into a column.

        This is how a reader builds a list or a struct. It walks the Arrow field
        nodes in order, which is the order this list is in, and hands the whole
        thing over at the end rather than building a column and adding children
        to it, because a list node cannot be finished until its element column is
        known and that node comes later in the walk.

        Dropping the root shifts every position down by one, and a node whose
        parent was the root lands on `ROOT` by the same subtraction, since `ROOT`
        is minus one. So the renumbering is one pass and one operation.

        Args:
            nodes: The tree, root first, each node naming its parent by position.

        Returns:
            The root as a column, carrying the rest as its children.

        Raises:
            If the list is empty, which is a caller that built nothing.
        """
        if len(nodes) == 0:
            raise Error("a nested column needs at least the node it is")
        var root = nodes.pop(0)
        for i in range(len(nodes)):
            nodes[i].parent -= 1
        var type = root.type
        var strings = Bool(root.text)
        var out: Self
        if strings:
            out = Self(root^.take_text())
            out.type = type
        else:
            out = Self(root^.take_data(), type)
        out.nested = nodes^
        return out^

    def into_node(
        deinit self, var name: String, parent: Int
    ) raises -> NestedNode:
        """Turns a finished column into a node of some other column's tree.

        The buffers are moved rather than copied, so a leaf that was just built
        by the ordinary import path costs nothing to put in its place.

        Args:
            name: The field name the node takes.
            parent: The node it hangs off, or `ROOT`.

        Returns:
            The node.

        Raises:
            If the column is nested or dictionary encoded, neither of which fits
            in one node and both of which the caller has to take apart itself.
        """
        if self.is_nested() or self.is_dictionary():
            raise Error(
                "column is "
                + String(self.type)
                + " and does not fit in one node of a nested column"
            )
        var type = self.type
        if self.text:
            var held = self.text^
            return NestedNode(name^, parent, held.take(), type)
        return NestedNode(name^, type, parent, self.data^)

    def __init__(out self, *, copy: Self):
        """Copies a column, sharing its bytes until one side writes.

        Args:
            copy: The column to copy.
        """
        self.data = ColumnData(copy=copy.data)
        self.type = copy.type
        self.text = Optional[StringArray](copy=copy.text)
        self.dict_values = Optional[StringArray](copy=copy.dict_values)
        self.nested = List[NestedNode](copy=copy.nested)
        self._slack = 0

    def __len__(self) -> Int:
        """Returns the number of values.

        Returns:
            The length.
        """
        return self.data.length

    def retyped(var self, type: LogicalType) raises -> Self:
        """Returns the column carrying another logical type over the same bytes.

        This exists for the kernels that move rows around without changing what
        a value means. A filter over a date column builds its output through
        `Array[DType.int32]`, because that is the layout a date is stored in, and
        erasing that array gives back a column that says it is an int32. The
        bytes are right and the type is wrong, and the wrongness is quiet: the
        frame's schema still says date, and the first thing that compares the two
        raises somewhere unrelated. So the kernel puts the input's type back on
        the output, and this is how.

        The physical layout has to be the one already there. Relabelling an
        int32 buffer as a float32 one would not convert anything, it would read
        the same bits as a different number, and that is `cast_any`'s job.

        Args:
            type: The type to carry. Its physical dtype must be the one this
                column already has.

        Returns:
            The same column, relabelled.

        Raises:
            If the physical layouts differ.
        """
        if type.physical != self.type.physical:
            raise Error(
                "retyped: "
                + String(type)
                + " is laid out as "
                + String(type.physical)
                + " and this column is laid out as "
                + String(self.type.physical)
            )
        self.type = type
        return self^

    def dtype(self) -> DType:
        """Returns the physical dtype.

        Returns:
            The dtype the values buffer is laid out as.
        """
        return self.type.physical

    def is_valid(self, i: Int) -> Bool:
        """Reports whether the value at a position is present.

        Args:
            i: The position. Must be less than the length.

        Returns:
            True if the value is not null.
        """
        return self.data.validity.get(i)

    def null_count(self) -> Int:
        """Returns the number of null values.

        Returns:
            The count of clear validity bits.
        """
        return self.data.validity.null_count()

    def nbytes(self) -> Int:
        """Returns the bytes the column's buffers occupy.

        Every buffer the column actually has, which is not the same set for
        every column: a fixed width column has values and maybe a validity
        bitmap, and a string column has views and a payload instead of values.
        The bitmap is counted at one byte per eight rows rather than at its
        allocated capacity, so the answer is about the data and not about the
        allocator's rounding.

        pandas reports this as `nbytes` and reports it for a `RangeIndex` too,
        where the number it gives is the size of the Python object rather than of
        any labels. `firepanda/py/index.mojo` answers zero there instead.

        Returns:
            The size in bytes.
        """
        var out = self.data.validity.byte_length()
        if self.text:
            ref held = self.text.value()
            return out + len(held.views) + len(held.payload)
        if self.dict_values:
            # Codes and categories both, which is what pandas counts for a
            # categorical. It is also the number that makes the type worth
            # having: a million rows over four categories is four megabytes of
            # codes and a few dozen bytes of text, and reporting only one half
            # would make the saving look either imaginary or free.
            ref held = self.dict_values.value()
            return (
                out
                + len(self.data.values)
                + len(held.views)
                + len(held.payload)
            )
        out += len(self.data.values)
        # A nested column's own buffers are its validity and, for a list, its
        # offsets. Everything a user would call the data is one level down, so a
        # count that stopped here would report a column of a million lists as a
        # few megabytes of offsets and nothing else.
        for i in range(len(self.nested)):
            ref node = self.nested[i]
            out += node.data.validity.byte_length()
            if node.text:
                ref held = node.text.value()
                out += len(held.views) + len(held.payload)
            else:
                out += len(node.data.values)
        return out

    def is_string(self) -> Bool:
        """Reports whether the column holds variable width elements.

        Every dispatch that reads values has to ask this before it matches on
        `dtype()`, because a string column's physical dtype is uint8 and would
        otherwise select the uint8 arm and read a view byte as a value.

        Returns:
            True for a string or binary column.
        """
        return Bool(self.text)

    def strings(ref self) raises -> ref[self.text.value()] StringArray:
        """Returns the variable width elements without copying them.

        Returns:
            A reference to the string column, valid as long as this one is.

        Raises:
            If the column is not a string column.
        """
        if not self.text:
            raise Error(
                "column is " + String(self.type) + ", not a string column"
            )
        return self.text.value()

    def into_strings(deinit self) raises -> StringArray:
        """Converts to a string column without copying, consuming this one.

        Returns:
            The string column.

        Raises:
            If the column is not a string column.
        """
        if not self.text:
            raise Error(
                "column is " + String(self.type) + ", not a string column"
            )
        var held = self.text^
        return held.take()

    def is_dictionary(self) -> Bool:
        """Reports whether the column stores positions into a category list.

        Returns:
            True for a dictionary column.
        """
        return Bool(self.dict_values)

    def categories(
        ref self,
    ) raises -> ref[self.dict_values.value()] StringArray:
        """Returns the values a dictionary column's codes refer to.

        The length of what comes back is the number of categories and not the
        number of rows, which is the whole difference between this and
        `strings`.

        Returns:
            A reference to the categories, valid as long as this column is.

        Raises:
            If the column is not a dictionary column.
        """
        if not self.dict_values:
            raise Error(
                "column is " + String(self.type) + ", not a dictionary column"
            )
        return self.dict_values.value()

    def codes[dt: DType](self) raises -> Array[dt]:
        """Returns a dictionary column's codes as a typed array.

        This is the deliberate way past the refusal in `check_dtype`, and every
        caller of it is saying it knows the numbers are positions rather than
        values. Reading a categorical's codes without meaning to is the mistake
        `check_dtype` exists to stop, so the way to do it on purpose is spelled
        differently rather than being the same call with a comment.

        Parameters:
            dt: The dtype of the codes, which must match the index type.

        Returns:
            A typed copy of the codes, the same deep copy `as_typed` makes.

        Raises:
            If the column is not a dictionary column, or if the index dtype
            differs from the requested one.
        """
        if not self.dict_values:
            raise Error(
                "column is " + String(self.type) + ", not a dictionary column"
            )
        self._check_physical[dt]()
        return Array[dt](ColumnData(copy=self.data))

    def offsets[dt: DType](self) raises -> Array[dt]:
        """Returns a list column's offsets as a typed array.

        The same deliberate way past `check_dtype` that `codes` is, and for the
        same reason: the numbers are positions in the element column rather than
        anything a row of this column is. There is one more of them than there
        are rows, because a row is the gap between two.

        Parameters:
            dt: The width of the offsets, int32 for a list and int64 for a large
                one.

        Returns:
            A typed copy of the offsets.

        Raises:
            If the column is not a list column, or if the offsets are not the
            requested width.
        """
        if not self.type.is_list():
            raise Error(
                "column is " + String(self.type) + ", not a list column"
            )
        self._check_physical[dt]()
        return Array[dt](ColumnData(copy=self.data))

    def is_nested(self) -> Bool:
        """Reports whether the column's values live in child columns.

        Asked of the type and not of the child list, so that a struct with no
        fields, which Arrow allows and which has an empty list here, is still a
        struct rather than an ordinary column with a strange dtype.

        Returns:
            True for a list or a struct column.
        """
        return self.type.is_nested()

    def type_name(self) raises -> String:
        """Returns the column's dtype as a name, children and all.

        `String(column.type)` gets as far as `list` or `struct` and stops, since
        what is inside a nested type is in the column rather than in the type.
        This is the whole of it, and for an ordinary column it is the same string
        the type prints on its own.

        Returns:
            The dtype name.

        Raises:
            If the column is a list without exactly one child, which is a column
            built wrong.
        """
        if not self.is_nested():
            return String(self.type)
        return nested_type_name(self.type, self.nested)

    def child_count(self) -> Int:
        """Returns how many child columns hang off this one directly.

        One for a list, one per field for a struct, and none for anything else.
        The children of those children are not counted.

        Returns:
            The number of top level children.
        """
        var out = 0
        for i in range(len(self.nested)):
            if self.nested[i].parent == ROOT:
                out += 1
        return out

    def child(self, at: Int) raises -> Self:
        """Returns one child column, with everything under it.

        A list's element column is child zero, and a struct's fields are its
        children in schema order. What comes back is a column in its own right,
        which for a nested child means it carries the part of the tree that was
        below it with the positions renumbered against its own list.

        The row count of a struct's field is the struct's own. The row count of a
        list's element column is not: it holds every element of every row laid
        end to end, and which of them belong to a row is what the offsets say.

        Args:
            at: The position of the child, counting the direct children only.

        Returns:
            The child as a column.

        Raises:
            If the column is not nested, or has no child at that position.
        """
        if not self.is_nested():
            raise Error(
                "column is " + String(self.type) + ", not a nested column"
            )
        var direct = List[Int]()
        for i in range(len(self.nested)):
            if self.nested[i].parent == ROOT:
                direct.append(i)
        if at < 0 or at >= len(direct):
            raise Error(
                String(
                    "column has ",
                    len(direct),
                    " children and child ",
                    at,
                    " was asked for",
                )
            )
        return self._subtree(direct[at])

    def field(self, name: StringSlice) raises -> Self:
        """Returns one field of a struct column by name.

        Args:
            name: The field name.

        Returns:
            The field as a column.

        Raises:
            If the column is not a struct, or has no field of that name.
        """
        if not self.type.is_struct():
            raise Error(
                "column is " + String(self.type) + ", not a struct column"
            )
        var at = field_named(self.nested, name)
        if at < 0:
            raise Error(String("struct column has no field named '", name, "'"))
        return self._subtree(at)

    def _subtree(self, root: Int) raises -> Self:
        """Lifts one node and everything under it into a column of its own.

        The nodes come back in the order they are stored, which puts a parent
        before its children, so renumbering is a lookup in the list built so far
        and never a forward reference.

        Args:
            root: The node to lift.

        Returns:
            The subtree as a column.

        Raises:
            If a node's storage cannot be copied.
        """
        var picked = subtree_of(self.nested, root)
        ref top = self.nested[root]
        var out: Self
        if top.text:
            out = Self(StringArray(copy=top.text.value()))
            out.type = top.type
        else:
            out = Self(ColumnData(copy=top.data), top.type)
        for k in range(1, len(picked)):
            var moved = NestedNode(copy=self.nested[picked[k]])
            var parent = ROOT
            for j in range(1, len(picked)):
                if picked[j] == moved.parent:
                    parent = j - 1
                    break
            moved.parent = parent
            out.nested.append(moved^)
        return out^

    def _check_physical[dt: DType](self) raises:
        """Raises unless the values buffer is laid out as a given dtype.

        Parameters:
            dt: The expected dtype.

        Raises:
            If the column's physical dtype differs.
        """
        if self.type.physical != dt:
            raise Error(
                "dtype mismatch: column is "
                + String(self.type.physical)
                + ", requested "
                + String(dt)
            )

    def check_dtype[dt: DType](self) raises:
        """Raises unless the column has a given dtype.

        A string column fails this for every `dt`. Its physical dtype is uint8,
        so without the first check `as_typed[DType.uint8]()` would hand back a
        column over the views buffer, whose values are not the column's values.

        A dictionary column fails it for the same reason and a worse one. Its
        physical dtype is the index type, which is a perfectly ordinary int32,
        so the check below would pass and hand back a column of category
        positions that a mean or a sum would then happily compute over and
        return a number for. There is no wrong dtype to catch it: the answer is
        wrong and nothing about it looks wrong. Every kernel that reads values
        goes through here, which is why one refusal in this function covers all
        of them rather than nineteen files each remembering to ask.

        Parameters:
            dt: The expected dtype.

        Raises:
            If the column's dtype differs, or if it is a string, dictionary,
            list or struct column.
        """
        if self.is_string():
            raise Error(
                "column is "
                + String(self.type)
                + " and has no fixed width values; use strings() instead"
            )
        if self.is_dictionary():
            raise Error(
                "column is "
                + String(self.type)
                + " and stores positions into its categories rather than"
                " values; use categories() and codes() instead"
            )
        if self.is_nested():
            # A list is the dictionary's trap again with different numbers. Its
            # physical dtype is int32, the check below would pass, and what came
            # back would be the offsets, which are a sorted run of small integers
            # that a sum or a mean will answer for without complaint. A struct is
            # worse in the other direction: it has no values buffer at all, so
            # what a kernel would read is an empty one.
            raise Error(
                "column is "
                + String(self.type)
                + " and keeps its values in child columns rather than in a"
                " buffer of its own; use child() or field() instead"
            )
        self._check_physical[dt]()

    def as_typed[dt: DType](self) raises -> Array[dt]:
        """Returns a typed copy of the column.

        Use this only when the caller needs to own the result. Anything that
        reads and hands the column back should take `as_typed_view`, which
        borrows the same storage and copies nothing.

        Parameters:
            dt: The dtype to read the column as.

        Returns:
            A typed array holding the same values.

        Raises:
            If `dt` is not the column's dtype.
        """
        self.check_dtype[dt]()
        return Array[dt](ColumnData(copy=self.data))

    def as_typed_view[dt: DType](ref self) raises -> ref[self.data] Array[dt]:
        """Returns a typed reference to the column, borrowing rather than copying.

        `as_typed` is a deep copy of every byte the column holds, which for a
        forty megabyte key column is a real cost, and a group by pays it once
        per key column on every query because `factorize` takes an `Array[dt]`
        and there was no other way to produce one from a borrowed `AnyArray`.
        This is that other way. It hands back a reference into the column's own
        storage, so it copies nothing and the caller may only read.

        `Array[dt]` holds one field, a `ColumnData`, and a struct of one field
        has that field's layout, so a pointer to the storage is a pointer to the
        array. That is an assumption about layout and it is asserted in
        `tests/test_array.mojo` rather than left to be noticed later.

        Parameters:
            dt: The dtype to read the column as.

        Returns:
            A reference to the column read as a typed array.

        Raises:
            If `dt` is not the column's dtype.
        """
        self.check_dtype[dt]()
        return Pointer(to=self.data).unsafe_bitcast[Array[dt]]()[]

    def into_typed[dt: DType](deinit self) raises -> Array[dt]:
        """Converts to a typed column without copying, consuming this one.

        Parameters:
            dt: The dtype to read the column as.

        Returns:
            A typed array over the same buffers.

        Raises:
            If `dt` is not the column's dtype.
        """
        self.check_dtype[dt]()
        return Array[dt](self.data^)

    def slice(self, start: Int, end: Int) raises -> Self:
        """Returns a copy of a half-open range of the column.

        For a fixed width column this needs no dispatch at all. A slice moves
        bytes and does not look at them, so the element width coming from
        `dtype_size` as a runtime value is all it needs, and the whole thing
        compiles to a single copy of the loop instead of one per dtype.

        A string column cannot be cut that way, because a run of views refers to
        payload bytes scattered through the block, so it is rebuilt element by
        element. That is the same choice `StringArray.slice` documents.

        Args:
            start: The first position, inclusive.
            end: The last position, exclusive.

        Returns:
            A new column of length `end - start` and the same dtype.

        Raises:
            If the range is outside a string column.
        """
        if self.is_string():
            return Self(self.strings().slice(start, end))
        if self.is_nested():
            # A struct slices by slicing every field, and a list does not slice
            # that way at all: the rows it keeps name a run of the element column
            # that the offsets have to be rebased against. Both are worth having
            # and neither is here yet, and the refusal is by name because the
            # loop below would otherwise cut a list column's offsets and hand
            # back something that looks like a column and reads the wrong
            # elements.
            raise Error(
                "column is "
                + String(self.type)
                + " and slicing a nested column is not implemented yet"
            )
        var width = dtype_size(self.type.physical)
        var n = end - start
        var values = Buffer(n * width)
        if n > 0:
            unsafe_memcpy(
                dest=values.unsafe_mut_ptr(),
                src=self.data.values.unsafe_ptr().unsafe_offset(start * width),
                count=n * width,
            )
        var out = Self(
            ColumnData(values^, self.data.validity.slice(start, end), n),
            self.type,
        )
        # A slice moves rows and a row moving cannot change what a code means,
        # so the categories come across whole, unused ones included. Without
        # this the result says it is a category and has nothing behind it, which
        # is the state document 29 is about.
        if self.dict_values:
            out.dict_values = StringArray(copy=self.dict_values.value())
        return out^

    def unsafe_ptr[dt: DType](self) -> Pointer[Scalar[dt], origin_of(self)]:
        """Returns a typed pointer to the values, for reading, dtype unchecked.

        Callers must have checked the dtype already, normally by going through
        `dispatch`. This exists so that dispatch does not pay for a second check
        and a buffer copy on the hot path.

        Borrowed rather than `ref`, so the buffer underneath stays shared. See
        `Buffer` for why reading and writing are separate names.

        Parameters:
            dt: The dtype to view the values as.

        Returns:
            A pointer to the first value.
        """
        return self.data.values.bitcast[dt]().unsafe_origin_cast[
            origin_of(self)
        ]()

    def unsafe_mut_ptr[
        dt: DType
    ](mut self) -> Pointer[Scalar[dt], origin_of(self)]:
        """Returns a typed pointer to the values, for writing, dtype unchecked.

        Takes a private copy of the buffer first if anything else is holding it.

        Parameters:
            dt: The dtype to view the values as.

        Returns:
            A pointer to the first value.
        """
        return self.data.values.mut_bitcast[dt]().unsafe_origin_cast[
            origin_of(self)
        ]()


def empty_any(type: LogicalType) raises -> AnyArray:
    """Builds a column of that type with no rows in it.

    A column no rows reached is not the same thing as a column that does not
    exist, and the difference matters wherever a kernel asks a column what it
    holds rather than how many rows it has. A join builds its key table from the
    build side's key column, and a build side a filter emptied still has to
    answer what dtype that column is.

    A text column takes the text route rather than the plain one, because
    `is_string` reads whether the string half is there and not what the logical
    type says. An empty column built the plain way would answer no to that and
    then be read as a column of bytes.

    A dictionary, a list and a struct get the plain form, which is the form they
    had before this was a function. None of them is a join key and nothing yet
    asks one of them for an empty column.

    Args:
        type: What the column would hold.

    Returns:
        The column, of length zero.

    Raises:
        If the type is text and is not laid out the way text is, which nothing
        can arrange and which is checked because relabelling is what checks it.
    """
    if type.kind == TypeKind.STRING or type.kind == TypeKind.BINARY:
        var text = AnyArray(StringArray(Buffer(0), Buffer(0), Bitmap(0), 0))
        return text^.retyped(type)
    return AnyArray(ColumnData(Buffer(0), Bitmap(0), 0), type)


def borrow_columns[o: ImmOrigin](ref[o] cols: List[AnyArray]) -> ColumnRefs[o]:
    """Borrows every column in a list.

    Parameters:
        o: The origin of the list, which the references inherit.

    Args:
        cols: The columns.

    Returns:
        One reference per column, in order.
    """
    var out = ColumnRefs[o](capacity=len(cols))
    for i in range(len(cols)):
        out.append(Pointer(to=cols[i]).unsafe_origin_cast[o]())
    return out^
