"""Tests for the children a list or a struct column keeps its values in.

The columns here are built by hand rather than read out of a file, which is the
point of the file: `tests/test_arrow_ipc.mojo` covers the same tree from the
reader's side and would go on passing if the helpers agreed with the reader and
with nothing else. A hand built column also reaches the shapes a real file cannot
produce, such as a list node with two children, and those are where the helpers
have to refuse rather than print something plausible.

The column both halves of the file use is `struct<a: list<item: int64>, b:
string>`, which is the smallest thing with a node at every depth: a struct at the
root, a list under it whose child is not the same length as its parent, a leaf
that holds strings rather than a values buffer, and a second field so that order
between siblings is visible.
"""

from std.testing import (
    TestSuite,
    assert_equal,
    assert_false,
    assert_raises,
    assert_true,
)

from firepanda.array.any import AnyArray
from firepanda.array.array import Array, from_list
from firepanda.array.data import ColumnData
from firepanda.array.nested import (
    ITEM,
    ROOT,
    NestedNode,
    children_of,
    field_named,
    nested_type_name,
    subtree_of,
)
from firepanda.array.strings import StringArray, StringBuilder
from firepanda.bitmap.bitmap import Bitmap
from firepanda.dtype.logical import LogicalType


def _offsets(var values: List[Int32], rows: Int) raises -> ColumnData:
    """Builds a list node's offsets buffer.

    The length of the storage is the row count and not the number of offsets,
    which is one more than that. Getting this backwards is the mistake the row
    count of a list is there to catch, so the fixture makes it the way the
    reader does rather than letting `from_list` decide.

    Args:
        values: The offsets, one more than the row count.
        rows: The row count.

    Returns:
        The storage, with no nulls in it.
    """
    var column = AnyArray(from_list[DType.int32](values))
    var data = column^.into_node(String(ITEM), ROOT).take_data()
    data.validity = Bitmap(rows)
    data.length = rows
    return data^


def _strings(var values: List[String]) raises -> StringArray:
    """Builds a string leaf's elements.

    Args:
        values: The strings, none of them null.

    Returns:
        The elements.
    """
    var builder = StringBuilder()
    for i in range(len(values)):
        builder.append(values[i].as_bytes())
    return builder^.finish()


def _tree() raises -> List[NestedNode]:
    """Builds the nodes of `struct<a: list<item: int64>, b: string>`.

    Two rows. The first list row holds two elements and the second holds one,
    which is what makes the element column three long against a struct that is
    two long.

    Returns:
        The nodes in pre-order, the column's own node first.
    """
    var nodes = List[NestedNode]()
    nodes.append(NestedNode(String("col"), LogicalType.STRUCT, ROOT, Int(2)))
    var offsets = NestedNode(
        String("a"),
        LogicalType.list_of(DType.int32),
        0,
        _offsets([Int32(0), Int32(2), Int32(3)], 2),
    )
    nodes.append(offsets^)
    var items = AnyArray(from_list[DType.int64]([Int64(7), Int64(8), Int64(9)]))
    nodes.append(items^.into_node(String(ITEM), 1))
    nodes.append(
        NestedNode(
            String("b"),
            0,
            _strings([String("x"), String("y")]),
            LogicalType.STRING,
        )
    )
    return nodes^


def _column() raises -> AnyArray:
    """Builds the column the tree describes.

    Returns:
        The struct column.
    """
    return AnyArray.nested_from(_tree())


def test_the_children_of_a_node_are_the_ones_that_name_it() raises:
    # Read off the parent field rather than off position, so a node moving in
    # the list does not silently reparent anything.
    var nodes = _tree()
    var top = children_of(nodes, ROOT)
    assert_equal(len(top), 1)
    assert_equal(top[0], 0)

    var fields = children_of(nodes, 0)
    assert_equal(len(fields), 2)
    assert_equal(nodes[fields[0]].name, "a")
    assert_equal(nodes[fields[1]].name, "b")

    var elements = children_of(nodes, 1)
    assert_equal(len(elements), 1)
    assert_equal(nodes[elements[0]].name, ITEM)

    assert_equal(len(children_of(nodes, 3)), 0)


def test_a_subtree_is_the_node_and_everything_under_it() raises:
    # The list and its element column, and not the string field beside them.
    var nodes = _tree()
    var picked = subtree_of(nodes, 1)
    assert_equal(len(picked), 2)
    assert_equal(picked[0], 1)
    assert_equal(picked[1], 2)

    var leaf = subtree_of(nodes, 3)
    assert_equal(len(leaf), 1)
    assert_equal(leaf[0], 3)

    assert_equal(len(subtree_of(nodes, 0)), 4)


def test_a_subtree_comes_back_with_a_parent_before_its_children() raises:
    # A caller renumbering the nodes against a new list looks the parent up in
    # what it has built so far, so a child arriving first would be a forward
    # reference into a list that does not have it yet.
    var nodes = _tree()
    var picked = subtree_of(nodes, 0)
    for k in range(len(picked)):
        var parent = nodes[picked[k]].parent
        if parent == ROOT:
            continue
        var seen = False
        for j in range(k):
            if picked[j] == parent:
                seen = True
        assert_true(seen)


def test_a_field_is_found_by_name_only_at_the_top_level() raises:
    # `item` is in the tree and is not a field of the column, so a lookup that
    # scanned every node would answer with the element column of a list when a
    # user asked a struct for a field it does not have.
    var nodes = _tree()
    var fields = _tree()
    _ = fields.pop(0)
    for i in range(len(fields)):
        fields[i].parent -= 1

    assert_equal(field_named(fields, "a"), 0)
    assert_equal(field_named(fields, "b"), 2)
    assert_equal(field_named(fields, ITEM), -1)
    assert_equal(field_named(fields, "nothing"), -1)
    assert_equal(field_named(nodes, "a"), -1)


def test_a_type_with_nothing_under_it_prints_as_itself() raises:
    # The nested spelling and the plain one come out of the same call, so a
    # caller printing a dtype does not have to ask which kind it has first.
    var nodes = List[NestedNode]()
    assert_equal(nested_type_name(LogicalType.INT64, nodes), "int64")
    assert_equal(nested_type_name(LogicalType.STRING, nodes), "string")


def test_a_list_with_two_children_is_refused_rather_than_printed() raises:
    # A list has exactly one child. Two is a column built wrong, and printing
    # the first one would hide it behind a type name that reads fine.
    var nodes = List[NestedNode]()
    nodes.append(NestedNode(String(ITEM), LogicalType.INT64, ROOT, Int(1)))
    nodes.append(NestedNode(String("extra"), LogicalType.INT64, ROOT, Int(1)))
    with assert_raises(contains="one child and this one has 2"):
        _ = nested_type_name(LogicalType.list_of(DType.int32), nodes)


def test_an_empty_struct_is_still_a_struct() raises:
    # Arrow allows a struct with no fields, and the node list of one is empty,
    # so nothing downstream may take an empty list to mean not nested.
    var nodes = List[NestedNode]()
    assert_equal(nested_type_name(LogicalType.STRUCT, nodes), "struct<>")


def test_a_nested_column_needs_at_least_the_node_it_is() raises:
    # The column's own node is the first of the list and is not kept in it, so
    # an empty list is a caller that has not built anything.
    with assert_raises(contains="at least the node it is"):
        _ = AnyArray.nested_from(List[NestedNode]())


def test_a_node_that_holds_no_strings_says_so() raises:
    # `take_text` is how a string leaf's elements get out. Asking it of a node
    # that keeps a values buffer would otherwise unwrap an empty optional.
    var node = NestedNode(String(ITEM), LogicalType.INT64, ROOT, Int(1))
    with assert_raises(contains="holds no strings"):
        _ = node^.take_text()


def test_a_nested_column_does_not_fit_in_one_node() raises:
    # One node holds one set of buffers. A nested column is a tree of them, and
    # putting one under another column would drop everything below the root.
    var column = _column()
    with assert_raises(contains="does not fit in one node"):
        _ = column^.into_node(String("inner"), ROOT)


def test_the_type_name_of_a_column_is_the_whole_type() raises:
    # What pyarrow prints, which is what a user comparing dtypes is comparing
    # against.
    var column = _column()
    assert_true(column.is_nested())
    assert_equal(column.type_name(), "struct<a: list<item: int64>, b: string>")
    assert_equal(String(column.type), "struct")
    assert_equal(column.child_count(), 2)
    assert_equal(len(column), 2)


def test_a_field_lifted_out_carries_the_part_of_the_tree_below_it() raises:
    # The lifted node becomes a column of its own, so the nodes under it are
    # renumbered against its own list and its element column hangs off the root
    # rather than off the position the field used to be at.
    var column = _column()
    var listed = column.field("a")
    assert_true(listed.type.is_list())
    assert_equal(listed.type_name(), "list<item: int64>")
    assert_equal(listed.child_count(), 1)
    assert_equal(listed.nested[0].parent, ROOT)
    assert_equal(listed.offsets[DType.int32]()[2], Int32(3))

    var items = listed.child(0)
    assert_false(items.is_nested())
    assert_equal(len(items), 3)
    assert_equal(items.as_typed[DType.int64]()[2], Int64(9))

    var text = column.field("b")
    assert_true(text.is_string())
    assert_equal(text.strings()[1], "y")


def test_the_element_column_of_a_list_is_not_as_long_as_the_list() raises:
    # Two rows of lists holding three elements between them. A row count taken
    # from the child would be the number a reader has to check the offsets
    # against, not the number of rows in the column.
    var column = _column()
    assert_equal(len(column), 2)
    assert_equal(len(column.field("a")), 2)
    assert_equal(len(column.field("a").child(0)), 3)


def test_a_field_that_is_not_there_is_refused_by_name() raises:
    var column = _column()
    with assert_raises(contains="no field named 'c'"):
        _ = column.field("c")
    with assert_raises(contains="children and child 2 was asked for"):
        _ = column.child(2)


def test_the_calls_that_want_a_tree_refuse_a_column_that_is_not_one() raises:
    # An int64 column has no children and is not a struct, and both refusals
    # name the type so that the message says what was passed rather than what
    # was wanted.
    var plain = AnyArray(from_list[DType.int64]([Int64(1)]))
    assert_false(plain.is_nested())
    assert_equal(plain.child_count(), 0)
    assert_equal(plain.type_name(), "int64")
    with assert_raises(contains="not a nested column"):
        _ = plain.child(0)
    with assert_raises(contains="not a struct column"):
        _ = plain.field("a")
    with assert_raises(contains="not a list column"):
        _ = plain.offsets[DType.int32]()


def test_a_struct_is_not_a_list_and_has_no_offsets() raises:
    # A struct's rows line up with its fields' rows one for one, so there is
    # nothing to mark out and no buffer to hand back.
    var column = _column()
    with assert_raises(contains="not a list column"):
        _ = column.offsets[DType.int32]()


def test_the_offsets_of_a_list_are_not_reachable_as_values() raises:
    # The trap this refusal exists for: the offsets are a sorted run of small
    # integers, so a sum or a mean over them answers a number and nothing about
    # the number looks wrong.
    var column = _column()
    with assert_raises(contains="keeps its values in child columns"):
        _ = column.as_typed[DType.int32]()
    with assert_raises(contains="keeps its values in child columns"):
        _ = column.field("a").as_typed[DType.int32]()


def test_slicing_a_nested_column_is_refused_rather_than_guessed() raises:
    # Cutting a list's offsets without rebasing them against the elements they
    # point at gives back something that looks like a column and reads the
    # wrong rows.
    var column = _column()
    with assert_raises(contains="slicing a nested column is not implemented"):
        _ = column.slice(0, 1)


def test_a_nested_column_counts_the_bytes_of_every_level() raises:
    # A count that stopped at the column's own buffers would report a million
    # lists as a few megabytes of offsets and no data at all.
    var column = _column()
    var own = column.data.validity.byte_length() + len(column.data.values)
    assert_equal(
        column.nbytes(),
        own + column.field("a").nbytes() + column.field("b").nbytes(),
    )
    assert_true(column.nbytes() > own)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
