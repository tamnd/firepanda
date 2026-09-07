"""The children a list or a struct column keeps its values in.

A nested column is the first type in firepanda whose values are not in it. A list
row is a run of values in a child column, marked out by one offset per row, and a
struct row is one row of every one of its fields. Either way the bytes a user
asked for are one level down, and a column that holds them needs a shape a flat
`ColumnData` does not have.

The obvious shape is a tree of columns, and Mojo will not have it: a struct
cannot hold a `List` of itself, because working out the list's destructor needs
the layout of the type being defined. The way around it is to hold the tree
flattened. A column carries a list of `NestedNode`, each naming its parent, and
the tree is read back by looking at those names. Lookups are a scan over a list
that is two entries long in the common case and nine in the deepest thing in the
conformance corpus, so the scan is cheaper than the index arithmetic it replaces
and much cheaper than a map.

Flattening is not only a workaround. It is the shape Arrow already describes a
batch in: the IPC metadata is a pre-order vector of field nodes and a vector of
buffers laid end to end, with no pointers anywhere. Reading one is a walk down
this list in step with those vectors, and writing one is the same walk. A tree of
heap nodes would have to be built on the way in and taken apart on the way out.

Where the buffers go, and it is the same rule at every level:

- A list node keeps its offsets in `data.values` and its type's physical dtype is
  the width of those offsets, int32 for Arrow's list and int64 for its large one.
  It has one child, which is the run of elements with no row boundaries in it.
- A struct node has no values buffer at all. Its `data.length` is the row count
  and its `data.validity` says which rows are there, and everything else is in
  its fields.
- A leaf keeps its values exactly where any other column does, which is
  `data.values` for a fixed width type and `text` for a string.

The column itself is the root and is not in the list. Its own offsets, validity
and length sit on the `AnyArray` in the ordinary fields, so a list column looks
like an int32 column that happens to have a child, and the nodes are its
children. Not repeating the root is what keeps one row count in one place.
"""

from firepanda.array.strings import StringArray
from firepanda.bitmap.bitmap import Bitmap
from firepanda.buffer.buffer import Buffer
from firepanda.dtype.logical import LogicalType

from .data import ColumnData

comptime ITEM = "item"
"""What Arrow names the child of a list, and what pyarrow prints in a list type.
A list's child is one column and has no name of its own, so this is a convention
rather than data, and it is here so that the reader and the writer agree on it."""

comptime ROOT = -1
"""The parent of a node that hangs off the column itself rather than off another
node."""


struct NestedNode(Copyable, Movable):
    """One child column of a nested column, and which node it belongs to."""

    var name: String
    """The field name for a struct's field, and `item` for a list's child."""

    var type: LogicalType
    """What this node holds. `LIST` and `STRUCT` mean the node has children of
    its own and the list below carries them."""

    var parent: Int
    """The position of the node this one hangs off, or `ROOT` for a child of the
    column itself."""

    var data: ColumnData
    """The offsets for a list node, the values for a fixed width leaf, and for a
    struct node nothing but the validity and the length.

    The length is this node's own row count and not its parent's. Those are the
    same number for a struct's field and are not for a list's child, where the
    child holds every element of every row laid end to end and is as long as the
    last offset says.
    """

    var text: Optional[StringArray]
    """The elements of a string or binary leaf, absent for every other node."""

    def __init__(
        out self, var name: String, type: LogicalType, parent: Int, length: Int
    ):
        """Constructs a node with no buffers, for a caller that fills them in.

        Args:
            name: The field name.
            type: What the node holds.
            parent: The node this one hangs off, or `ROOT`.
            length: The node's own row count.
        """
        self.name = name^
        self.type = type
        self.parent = parent
        self.data = ColumnData(byte_size=0, length=length)
        self.text = None

    def __init__(
        out self,
        var name: String,
        type: LogicalType,
        parent: Int,
        var data: ColumnData,
    ):
        """Constructs a node over storage the caller built.

        Args:
            name: The field name.
            type: What the node holds.
            parent: The node this one hangs off, or `ROOT`.
            data: The offsets, the values, or for a struct just the validity.
        """
        self.name = name^
        self.type = type
        self.parent = parent
        self.data = data^
        self.text = None

    def __init__(
        out self,
        var name: String,
        parent: Int,
        var strings: StringArray,
        type: LogicalType,
    ):
        """Constructs a string or binary leaf.

        The validity is copied into `data` as well as being carried in the
        string column, which is the same duplication `AnyArray` documents and is
        there for the same reason: everything that only asks whether a row is
        present reads one place whatever the type is.

        Args:
            name: The field name.
            parent: The node this one hangs off, or `ROOT`.
            strings: The elements.
            type: `LogicalType.STRING` or `LogicalType.BINARY`.
        """
        var length = len(strings)
        self.name = name^
        self.type = type
        self.parent = parent
        self.data = ColumnData(Buffer(0), Bitmap(copy=strings.validity), length)
        self.text = strings^

    def __init__(out self, *, copy: Self):
        """Deep-copies a node.

        Args:
            copy: The node to copy.
        """
        self.name = String(copy.name)
        self.type = copy.type
        self.parent = copy.parent
        self.data = ColumnData(copy=copy.data)
        self.text = Optional[StringArray](copy=copy.text)

    def take_data(deinit self) -> ColumnData:
        """Hands over the node's storage without copying it.

        Returns:
            The offsets, the values, or a struct node's validity and length.
        """
        return self.data^

    def take_text(deinit self) raises -> StringArray:
        """Hands over a string leaf's elements without copying them.

        Returns:
            The elements.

        Raises:
            Error: If the node is not a string or binary leaf.
        """
        var held = self.text^
        if not held:
            raise Error(
                "a node of type " + String(self.type) + " holds no strings"
            )
        return held.take()

    def __len__(self) -> Int:
        """Returns the node's own row count.

        Returns:
            The length.
        """
        return self.data.length


def children_of(nodes: List[NestedNode], parent: Int) -> List[Int]:
    """Finds the nodes hanging off one node, in the order they were built.

    Args:
        nodes: The flattened tree.
        parent: The node to look under, or `ROOT` for the column itself.

    Returns:
        The positions of the children, left to right.
    """
    var out = List[Int]()
    for i in range(len(nodes)):
        if nodes[i].parent == parent:
            out.append(i)
    return out^


def subtree_of(nodes: List[NestedNode], root: Int) -> List[Int]:
    """Finds a node and everything under it, in the order they are stored.

    One pass is enough because a node is always stored after the node it hangs
    off. That is the pre-order Arrow writes its field nodes in, it is the order
    the reader builds them in, and keeping it means the answer here comes back in
    the same order too, so a caller rebuilding a column out of a subtree can take
    the nodes as they come.

    Args:
        nodes: The flattened tree.
        root: The node at the top of the subtree.

    Returns:
        The positions of `root` and of every node below it, `root` first.
    """
    var taken = List[Bool](length=len(nodes), fill=False)
    var out = List[Int]()
    for i in range(len(nodes)):
        if i == root:
            taken[i] = True
        elif nodes[i].parent >= 0 and taken[nodes[i].parent]:
            taken[i] = True
        if taken[i]:
            out.append(i)
    return out^


def field_named(nodes: List[NestedNode], name: StringSlice) -> Int:
    """Finds a top level field by name.

    Args:
        nodes: The flattened tree.
        name: The field name.

    Returns:
        The position of the field, or -1 if the column has no such field.
    """
    for i in range(len(nodes)):
        if nodes[i].parent == ROOT and nodes[i].name == name:
            return i
    return -1


def nested_type_name(
    type: LogicalType, nodes: List[NestedNode]
) raises -> String:
    """Spells out a nested type in full, the way pyarrow prints one.

    Args:
        type: The column's own type.
        nodes: Its flattened tree.

    Returns:
        Something like `large_list<item: int64>` or `struct<a: int64>`, and for
        a type that is not nested whatever that type prints as on its own.

    Raises:
        Error: If a list node does not have exactly one child.
    """
    var out = String()
    write_nested_type(out, type, nodes, ROOT)
    return out^


def write_nested_type(
    mut writer: Some[Writer],
    type: LogicalType,
    nodes: List[NestedNode],
    parent: Int,
) raises:
    """Writes a nested type the way Arrow spells it, children and all.

    `LogicalType.write_to` can only write the part of a nested type that is in
    the type, which is `list` or `struct` and nothing about what is inside. The
    rest is in the column, so the full spelling is written from here, where both
    halves are in hand. A user comparing a dtype string against pyarrow's is
    comparing against `large_list<item: int64>` or `struct<a: int64, b: string>`,
    and those are what this produces.

    Args:
        writer: The destination.
        type: The type of the node being written, which for the outermost call is
            the column's own.
        nodes: The flattened tree.
        parent: The node whose children are to be written, or `ROOT` for the
            column itself.

    Raises:
        Error: If a list node does not have exactly one child, which is a column
            built wrong rather than a file read wrong.
    """
    if not type.is_nested():
        writer.write(type)
        return

    var kids = children_of(nodes, parent)
    if type.is_list():
        if len(kids) != 1:
            raise Error(
                String(
                    "a list column has one child and this one has ",
                    len(kids),
                )
            )
        writer.write(type, "<", nodes[kids[0]].name, ": ")
        write_nested_type(writer, nodes[kids[0]].type, nodes, kids[0])
        writer.write(">")
        return

    writer.write("struct<")
    for i in range(len(kids)):
        if i > 0:
            writer.write(", ")
        writer.write(nodes[kids[i]].name, ": ")
        write_nested_type(writer, nodes[kids[i]].type, nodes, kids[i])
    writer.write(">")
