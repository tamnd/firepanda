"""Every row local operator answers the same whether the chunk carries a
selection or not.

This is the invariant the whole selection vector design rests on. A chunk is
either dense, meaning element i of a column is row i of the chunk, or it carries
a list of positions and some of its columns are read through them. An operator
is free to notice which it was handed and take a cheaper route, and several of
them do, but it is not free to answer differently.

The test is the same shape for every operator. The same rows are built twice,
once under a selection and once flattened, both are pushed through the same
node, and the two answers are rendered value by value and compared as text. The
rendering is the frame's own, so a null is `<NA>` and a float is written the way
the frame writes one, which means a difference in what a column holds shows up
as a difference in a string rather than needing a comparison per type.

Rendering both sides also catches the failure that a length check misses, which
is an operator that keeps the right number of rows and keeps the wrong ones.
That is the mistake a selection invites: the positions are indices into the
array underneath, the rows are positions into the selection, and an operator
that confuses the two comes back with the right count of the wrong values.

The join is not here. Its output is still materialized per side and the box for
making it a selection is open on #521, so there is nothing yet to disagree.
"""

from std.testing import TestSuite, assert_equal, assert_false, assert_true

from firepanda.array.any import AnyArray
from firepanda.array.array import Array
from firepanda.array.strings import strings_from_list
from firepanda.array.value import Value
from firepanda.dtype.logical import LogicalType
from firepanda.exec import (
    Apply,
    Case,
    Cast,
    Choose,
    Chunk,
    Compute,
    Connective,
    Constant,
    Cut,
    Expand,
    Fill,
    Filter,
    Length,
    Locate,
    Match,
    Node,
    Part,
    Presence,
    Project,
    Trim,
    Truncate,
    node_apply,
)
from firepanda.frame.display import DisplayOptions, render_value
from firepanda.kernel.binary import BinaryOp
from firepanda.kernel.logic import LogicOp
from firepanda.kernel.pattern import MatchKind, Pattern
from firepanda.kernel.temporal import TRUNC_MONTH, TemporalField
from firepanda.kernel.unary import UnaryOp


def picks() raises -> List[UInt32]:
    """Which rows of the eight underneath the selection keeps.

    Not a prefix, not every other row, and it ends one short of the last row, so
    an operator that walked the array rather than the selection comes back with
    values nothing in the answer should hold.
    """
    return [UInt32(1), UInt32(2), UInt32(4), UInt32(5), UInt32(7)]


def counting() raises -> AnyArray:
    """Eight int64 values with a null in the middle of them.

    The values are the row numbers times ten, so a value says which row of the
    array underneath it came from and a wrong row is legible in the failure
    rather than being another number.
    """
    var col = Array[DType.int64](8)
    for i in range(8):
        if i == 4:
            col.set_null(i)
        else:
            col.set_valid(i, Int64((i + 1) * 10))
    return AnyArray(col^)


def seconds() raises -> AnyArray:
    """Eight int64 values, small and positive, for the operator that wants a
    count rather than a measurement."""
    var col = Array[DType.int64](8)
    for i in range(8):
        col.set_valid(i, Int64(i % 3))
    return AnyArray(col^)


def fractions() raises -> AnyArray:
    """Eight float64 values, one of them null and one of them negative."""
    var col = Array[DType.float64](8)
    for i in range(8):
        if i == 6:
            col.set_null(i)
        else:
            col.set_valid(i, Float64(i) * 1.5 - 2.0)
    return AnyArray(col^)


def truths(values: List[Bool]) raises -> AnyArray:
    """Eight bools with nothing missing."""
    var col = Array[DType.bool](len(values))
    for i in range(len(values)):
        col.set_valid(i, values[i])
    return AnyArray(col^)


def days() raises -> AnyArray:
    """Eight dates spread over four months, one of them missing.

    The 30th of June 2013 and on, a month boundary either side of a month, which
    is where a truncation and a month field both have something to get wrong.
    """
    var col = Array[DType.int32](8)
    for i in range(8):
        if i == 3:
            col.set_null(i)
        else:
            col.set_valid(i, Int32(15886 + i * 12))
    return AnyArray(col^.into_data(), LogicalType.DATE32)


def columns() raises -> List[AnyArray]:
    """The eight rows every test in this file starts from.

    Nine columns, because between them the operators want a number, a second
    number to compute against, a float, two masks, text with something on the
    ends of it, a date, and a small count.
    """
    var out = List[AnyArray]()
    out.append(counting())
    out.append(truths([True, False, True, True, False, True, False, True]))
    out.append(truths([True, True, False, True, False, False, True, True]))
    out.append(fractions())
    out.append(
        AnyArray(
            strings_from_list(
                [
                    "  ok  ",
                    "fail",
                    "ok",
                    "",
                    "no",
                    "  long word  ",
                    "ok",
                    "done",
                ]
            )
        )
    )
    out.append(days())
    out.append(seconds())
    out.append(counting())
    out.append(fractions())
    return out^


def selected() raises -> Chunk:
    """The rows under a selection, with two of the columns dense.

    Columns 1 and 2 are dense, which is what a mask written after an earlier
    filter looks like: one value per row of the chunk rather than one per value
    in the array underneath. Everything else is read through the positions. A
    chunk with both kinds in it is the one worth testing, because an operator
    that reads the wrong one still has a column of the right length to hand
    back.
    """
    var cols = columns()
    var at = picks()
    var narrow = List[AnyArray]()
    for i in range(len(cols)):
        if i == 1 or i == 2:
            ref view = cols[i].as_typed_view[DType.bool]()
            var kept = List[Bool](capacity=len(at))
            for j in range(len(at)):
                kept.append(Bool(view[Int(at[j])]))
            narrow.append(truths(kept))
        else:
            narrow.append(AnyArray(copy=cols[i]))
    var dense = List[Bool](capacity=len(narrow))
    for i in range(len(narrow)):
        dense.append(i == 1 or i == 2)
    return Chunk(narrow^, at^, dense^)


def flattened() raises -> Chunk:
    """The same rows with the selection gathered away."""
    var chunk = selected()
    chunk.flatten()
    return chunk^


def digest(var chunk: Chunk) raises -> String:
    """Writes out everything a chunk holds, in the frame's own spelling.

    The chunk is flattened first, so a chunk that came back under a selection
    and one that came back dense are compared on what they hold rather than on
    how they hold it, which is the whole question.

    Args:
        chunk: The chunk. Consumed.

    Returns:
        Every value, column by column, with the separators that stop two
        different shapes rendering as the same text.

    Raises:
        If the chunk cannot be flattened.
    """
    chunk.flatten()
    var options = DisplayOptions()
    var out = String(chunk.width(), " by ", chunk.rows, ":")
    for i in range(chunk.width()):
        for r in range(chunk.rows):
            out += render_value(chunk.columns[i], r, options)
            out += "|"
        out += ";"
    return out^


def both_ways(node: Node, what: StringSlice) raises:
    """Pushes the same rows through one node twice and compares the answers.

    Args:
        node: The operator, read only, which is what `node_apply` wants.
        what: The operator's name, for the failure message.

    Raises:
        If the node cannot process either chunk, or if the two disagree.
    """
    var under = node_apply(node, selected())
    var copied = node_apply(node, flattened())
    assert_true(
        under.__bool__() == copied.__bool__(),
        String(what, ": both emit a chunk or neither does"),
    )
    if not under:
        return
    assert_equal(
        digest(under.take()),
        digest(copied.take()),
        String(what, ": the same rows either way"),
    )


def test_the_selection_is_the_shape_the_rest_of_the_file_assumes() raises:
    """The fixture, checked once so that a failure below is about the operator
    rather than about the rows it was handed."""
    var chunk = selected()
    assert_true(chunk.selected(), "the chunk carries a selection")
    assert_equal(chunk.rows, 5, "five of the eight rows")
    assert_equal(chunk.width(), 9, "nine columns")
    assert_true(chunk.dense[1], "the first mask is at the rows")
    assert_true(chunk.dense[2], "and so is the second")
    assert_false(chunk.dense[0], "the values are read through the positions")
    var flat = flattened()
    assert_false(flat.selected(), "the other one has no selection")
    assert_equal(flat.rows, 5, "and the same five rows")
    var written = digest(selected())
    assert_equal(
        written,
        digest(flattened()),
        "the two fixtures hold the same rows before any operator runs",
    )
    # A digest that rendered nothing would compare equal to another one that
    # rendered nothing, so the one assertion this file cannot do without is that
    # the rendering says what is there. These are the five values the selection
    # keeps out of the counting column, the middle one being the null.
    assert_true(
        written.find("20|30|<NA>|60|80|;") >= 0,
        "the digest writes out the values the selection keeps",
    )


def test_a_filter_agrees_whether_the_chunk_was_selected() raises:
    """Four filters: a mask, a mask that narrows to two columns, a comparison
    the filter does itself, and the same comparison with the constant on the
    left. The last two are the fused form, which reads its operand through the
    positions rather than gathering it, so it is the one with the most to get
    wrong about a selected chunk."""
    both_ways(Node(Filter(1)), "filter on a mask")
    both_ways(Node(Filter(1, [4, 0])), "filter that narrows")
    both_ways(
        Node(Filter(0, Value(Int64(30)), BinaryOp.GT)),
        "filter that compares",
    )
    both_ways(
        Node(Filter(0, Value(Int64(30)), BinaryOp.GT, value_on_left=True)),
        "filter that compares the other way round",
    )
    both_ways(
        Node(Filter(0, Value(Int64(30)), BinaryOp.GT, [3, 1])),
        "filter that compares and narrows",
    )


def test_a_filter_that_keeps_nothing_agrees_either_way() raises:
    """The empty answer is its own case, because one route returns None and the
    other has to return None as well rather than an empty chunk."""
    both_ways(
        Node(Filter(0, Value(Int64(9000)), BinaryOp.GT)),
        "filter that keeps nothing",
    )


def test_the_operators_that_move_columns_about_agree() raises:
    """A projection passes a selection through, an expansion and a constant do
    not, and a cast reads through one. Between them they are every way an
    operator can change the shape of a chunk without looking at a row."""
    both_ways(Node(Project([4, 0, 0])), "projection")
    both_ways(Node(Expand(6, [0, 4])), "expansion")
    both_ways(
        Node(Constant(Value(Int64(7)), LogicalType.INT64, "seven")), "constant"
    )
    both_ways(Node(Cast(0, LogicalType.FLOAT64)), "cast")


def test_the_operators_that_compute_over_numbers_agree() raises:
    """A compute gathers the operands it names out of a selected chunk, which
    is the one place a wrong index shows up as a plausible answer rather than as
    a crash."""
    both_ways(Node(Compute(0, 7, BinaryOp.ADD, "sum")), "compute over two")
    both_ways(Node(Compute(0, 3, BinaryOp.MUL, "scaled")), "compute over types")
    both_ways(
        Node(Compute(0, Value(Int64(5)), BinaryOp.SUB, "less")),
        "compute against a constant",
    )
    both_ways(Node(Apply(0, UnaryOp.NEG, "down")), "unary")
    both_ways(Node(Connective(1, 2, LogicOp.AND, "both")), "connective")
    both_ways(Node(Choose(1, 0, 7, "picked")), "choose")


def test_the_operators_that_read_text_agree() raises:
    """Text is held as offsets into a payload, so a row read at the wrong index
    is a different length as well as a different value, and both show up in the
    rendering."""
    both_ways(
        Node(Match(4, Pattern(MatchKind.CONTAINS, "o", ""), "hit")), "match"
    )
    both_ways(Node(Cut(4, 1, 2, "front")), "cut")
    both_ways(Node(Length(4, False, "wide")), "length")
    both_ways(Node(Case(4, True, "loud")), "case")
    both_ways(Node(Trim(4, "", False, True, True, "trimmed")), "trim")
    both_ways(Node(Locate(4, "o", "where")), "locate")


def test_the_operators_that_read_a_date_agree() raises:
    """A date is an int32 count and a truncation writes a new one, so both
    directions of the temporal pair are here."""
    both_ways(Node(Part(5, TemporalField.MONTH, "m")), "part")
    both_ways(Node(Truncate(5, TRUNC_MONTH, "start")), "truncate")


def test_the_operators_that_answer_about_missing_rows_agree() raises:
    """The selection has a null under it and a null beside it, so a presence
    that read the wrong row would answer that the wrong row was missing."""
    both_ways(Node(Presence(0, True, "gone")), "presence")
    both_ways(Node(Presence(0, False, "there")), "presence the other way")
    both_ways(Node(Fill(0, 7, "filled")), "fill")


def test_two_operators_in_a_row_agree_with_the_pair_flattened_between() raises:
    """One operator at a time is not the whole claim. A filter hands the next
    operator a chunk under a selection it composed, and the answer has to be
    what the same two operators give when the chunk is flattened in the middle,
    which is the case a selection that composed wrongly survives."""
    var kept = node_apply(Node(Filter(1)), selected())
    assert_true(kept.__bool__(), "the filter kept something")
    var node = Node(Compute(0, 7, BinaryOp.ADD, "sum"))
    var carried = node_apply(node, kept.take())
    assert_true(carried.__bool__(), "and the compute answered")

    var again = node_apply(Node(Filter(1)), flattened())
    assert_true(again.__bool__(), "the same rows the other way")
    var flat = again.take()
    flat.flatten()
    var copied = node_apply(node, flat^)
    assert_true(copied.__bool__(), "and the same answer")

    assert_equal(
        digest(carried.take()),
        digest(copied.take()),
        "a composed selection and a flattened chunk agree",
    )


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
