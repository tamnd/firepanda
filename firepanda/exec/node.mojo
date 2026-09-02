"""The operators a pipeline pushes chunks through.

A node is one step of a query. It is handed a chunk, it does one thing to it,
and it hands back a chunk or nothing. Three methods, the shape Polars uses for
`ComputeNode`:

    update_state()  what the node wants next, and whether it is done
    process(chunk)  transform one chunk, called once per chunk
    finish()        produce what the node has been holding, one chunk at a time

`process` is the whole of an operator that carries no state across rows. A
filter looks at one chunk, decides which of its rows survive, and emits them.
Nothing it learns from one chunk affects the next, so it never needs `finish`
and it returns None from it.

`finish` is what makes a breaker a breaker. A group by cannot emit anything
until it has seen the last row, because the last row might belong to the first
group. So it swallows every chunk in `process`, returns None each time, and
`finish` is where its answer comes out. `finish` is called repeatedly and
returns None when there is nothing left, so a node holding ten million rows
gives them back in chunks rather than as one array.

`update_state` is the one that looks like paperwork and is not. A `Limit` that
has emitted the rows it was asked for says FINISHED, and because a pipeline is
a line, one finished node makes everything upstream of it useless. The driver
stops reading the source. That is limit pushdown, and it comes out of the
interface rather than out of an optimizer pass.

There is no trait here. Mojo 1.0 has trait objects but a `List` of them is not
yet expressible, so the node set is a closed union and dispatch is a chain of
type tests in the free functions at the bottom of this file. The cost is one
branch per chunk, which at a hundred and twenty eight thousand rows a chunk is
not a cost. A closed set is also the truth: an engine has the operators it has,
and every one of them is in this file or in the plan.

The elementwise family is `Filter`, `Project`, `Compute` and `Cast`, and what
they have in common is that an output row depends on its own input row and
nothing else. None of them is a breaker and none of them holds anything between
chunks. `Compute` is the one that is more general than it looks: an expression is
a tree and a tree is a line of these, so `(a + b) < c` is one node that appends
`a + b` and a second that compares the appended column, with a `Project` at the
end to drop the intermediate. That is why it appends rather than replaces, and
why `Filter` can name its mask by position: the plan counted the appends.

`Group` is the first breaker that is not the fallback, and it is the one that
shows what the interface buys. A materialised group by needs memory the size of
its input because it holds every row until the last one arrives. This one holds
one row per group and merges each chunk into that as the chunk goes past, which
is the same answer at a fraction of the memory, for the reductions that can be
folded. The rest still go through the fallback, and which ones those are is a
list rather than a judgement: a sum of sums is a sum, a median of medians is
not.

`Materialize` is the escape hatch and it is not temporary. It collects every
chunk into one frame, calls a whole frame function, and gives the answer back as
chunks. Anything with no chunked implementation goes through it, which is what
lets the engine be built one pull request at a time instead of in one commit:
every operator starts as a `Materialize`, the pipeline is exactly as fast as the
tree is today, and each later change removes one. Polars shipped `InMemoryMap`
and `InMemoryJoin` for the same reason and still has them.
"""

from std.utils import Variant

from firepanda.array.any import AnyArray, borrow_columns
from firepanda.array.array import Array
from firepanda.array.chunked import ChunkedArray
from firepanda.array.value import Value
from firepanda.dtype.logical import LogicalType
from firepanda.dtype.schema import Field, Schema
from firepanda.frame.frame import DataFrame
from firepanda.hash.grouping import group_ordinals
from firepanda.kernel.accum import accumulator
from firepanda.kernel.binary import (
    BinaryOp,
    binary_any,
    binary_type,
    binary_value_any,
)
from firepanda.kernel.cast import cast_any
from firepanda.kernel.concat import concat_two_any
from firepanda.kernel.group import AggKind, aggregate_group_any
from firepanda.kernel.select import filter_any, take_any

from .chunk import Chunk
from .morsel import MORSEL_ROWS


struct NodeStatus(Equatable, ImplicitlyCopyable, Movable, Writable):
    """What a node wants the driver to do next."""

    var code: UInt8
    """The state, as a small integer."""

    comptime NEED_MORE_INPUT = Self(0)
    """Keep pushing chunks in. The ordinary answer."""

    comptime HAVE_OUTPUT = Self(1)
    """The node is holding output that `finish` will hand over."""

    comptime FINISHED = Self(2)
    """The node will not emit another row whatever it is given.

    Everything upstream of a finished node in a pipeline is wasted work, so the
    driver stops reading the source when it sees this.
    """

    def __init__(out self, code: UInt8):
        """Constructs a status from its code.

        Args:
            code: The state.
        """
        self.code = code

    def __eq__(self, other: Self) -> Bool:
        """Compares two statuses.

        Args:
            other: The status to compare against.

        Returns:
            True if they are the same state.
        """
        return self.code == other.code

    def __ne__(self, other: Self) -> Bool:
        """Compares two statuses.

        Args:
            other: The status to compare against.

        Returns:
            True if they are different states.
        """
        return self.code != other.code

    def write_to(self, mut writer: Some[Writer]):
        """Writes the state name.

        Args:
            writer: The destination.
        """
        if self == Self.NEED_MORE_INPUT:
            writer.write("need more input")
        elif self == Self.HAVE_OUTPUT:
            writer.write("have output")
        else:
            writer.write("finished")


struct Filter(Movable):
    """Keeps the rows a boolean column of the chunk is true on.

    The mask is a column of the chunk rather than something handed to the node,
    which is what makes this an operator rather than a call. Whatever produced
    the mask, a comparison or an and of several, is another node earlier in the
    pipeline that wrote its answer into a column, and this one reads it by
    position. The mask column is filtered along with the rest and comes out all
    true, so a `Project` after this is what drops it.
    """

    var on: Int
    """The position of the boolean column to filter by."""

    def __init__(out self, on: Int):
        """Constructs a filter over one column of its input.

        Args:
            on: The position of the boolean column.
        """
        self.on = on

    def process(mut self, var chunk: Chunk) raises -> Optional[Chunk]:
        """Keeps the rows the mask is true on.

        Args:
            chunk: The chunk. Consumed.

        Returns:
            The surviving rows, or None if none survived, since a chunk of no
            rows is work for everything downstream and no information.

        Raises:
            If the position is out of range or the column is not boolean.
        """
        if self.on < 0 or self.on >= chunk.width():
            raise Error(
                "filter: column "
                + String(self.on)
                + " is outside a chunk of "
                + String(chunk.width())
                + " columns"
            )
        ref mask = chunk.columns[self.on].as_typed_view[DType.bool]()
        var kept = List[AnyArray](capacity=chunk.width())
        for i in range(chunk.width()):
            kept.append(filter_any(chunk.columns[i], mask))
        var rows = 0 if len(kept) == 0 else len(kept[0])
        if rows == 0:
            return None
        return Chunk(kept^, rows)


struct Project(Movable):
    """Keeps some columns of the chunk, in an order the plan chose.

    A column named twice is copied once and moved once rather than copied twice,
    because the last use of a position can give the array up. Every other
    position is moved out, so the ordinary projection, which is a subset in some
    order, copies nothing at all. That matters more than it sounds: a projection
    that copied would put a full copy of every kept column into the cost of
    every chunk, and dropping columns early is one of the main things a plan
    does.
    """

    var keep: List[Int]
    """The input positions to keep, in output order."""

    def __init__(out self, var keep: List[Int]):
        """Constructs a projection.

        Args:
            keep: The input positions, in output order. May repeat.
        """
        self.keep = keep^

    def process(mut self, var chunk: Chunk) raises -> Optional[Chunk]:
        """Rearranges the chunk's columns.

        Args:
            chunk: The chunk. Consumed.

        Returns:
            A chunk of the kept columns, with the same number of rows.

        Raises:
            If a position is outside the chunk.
        """
        var rows = len(chunk)
        var width = chunk.width()
        for i in range(len(self.keep)):
            if self.keep[i] < 0 or self.keep[i] >= width:
                raise Error(
                    "project: column "
                    + String(self.keep[i])
                    + " is outside a chunk of "
                    + String(width)
                    + " columns"
                )
        var held = List[Optional[AnyArray]](capacity=width)
        var backwards = chunk^.into_columns()
        var flipped = List[AnyArray](capacity=width)
        while len(backwards) > 0:
            flipped.append(backwards.pop())
        while len(flipped) > 0:
            held.append(Optional[AnyArray](flipped.pop()))
        var last = List[Bool](length=len(self.keep), fill=True)
        var seen = List[Bool](length=width, fill=False)
        for i in range(len(self.keep) - 1, -1, -1):
            last[i] = not seen[self.keep[i]]
            seen[self.keep[i]] = True
        var out = List[AnyArray](capacity=len(self.keep))
        for i in range(len(self.keep)):
            if last[i]:
                out.append(held[self.keep[i]].take())
            else:
                out.append(AnyArray(copy=held[self.keep[i]].value()))
        return Chunk(out^, rows)


struct Limit(Movable):
    """Passes the first n rows through and then stops the pipeline.

    Stopping is the interesting half. Once this has emitted the rows it was
    asked for it reports FINISHED, and a pipeline is a line, so nothing upstream
    can produce a row that reaches the sink. The driver stops reading the source
    the moment it sees that. A `head` over a file of ten million rows therefore
    reads one chunk, not the file.
    """

    var n: Int
    """The number of rows to let through."""

    var emitted: Int
    """How many have gone through so far."""

    def __init__(out self, n: Int):
        """Constructs a limit.

        Args:
            n: The number of rows to keep. Zero finishes immediately.
        """
        self.n = n if n > 0 else 0
        self.emitted = 0

    def update_state(self) -> NodeStatus:
        """Reports whether the limit has been reached.

        Returns:
            FINISHED once n rows have gone through, NEED_MORE_INPUT before that.
        """
        if self.emitted >= self.n:
            return NodeStatus.FINISHED
        return NodeStatus.NEED_MORE_INPUT

    def process(mut self, var chunk: Chunk) raises -> Optional[Chunk]:
        """Passes rows through until the limit is reached.

        Args:
            chunk: The chunk. Consumed.

        Returns:
            The whole chunk while there is room for it, a prefix of it on the
            chunk that reaches the limit, and None afterwards.

        Raises:
            If a column cannot be sliced.
        """
        if self.emitted >= self.n:
            return None
        var room = self.n - self.emitted
        if len(chunk) <= room:
            self.emitted += len(chunk)
            return chunk^
        var cut = List[AnyArray](capacity=chunk.width())
        for i in range(chunk.width()):
            cut.append(chunk.columns[i].slice(0, room))
        self.emitted = self.n
        return Chunk(cut^, room)


struct Compute(Movable):
    """Appends a column computed from two columns of the chunk.

    This is `with_column` and it is also every arithmetic and comparison
    expression, because an expression is a tree and a tree is a line of these.
    `(a + b) < c` is a `Compute` that appends `a + b` and a second one that
    compares the appended column against `c`, and the intermediate is dropped by
    a `Project` at the end rather than by anything here. Writing it that way
    means no node has to hold an expression tree, and it means the plan can see
    every intermediate and decide when to stop keeping it.

    Appending rather than replacing is the reason `Filter` can read its mask by
    position: the node that computed the mask put it at the end and the plan
    counted. A node that replaced a column would make the position of everything
    after it depend on what the expression was.

    An operand is either another column or a constant, and both forms are this
    one node rather than two, because a plan that had to pick between two node
    types every time it walked an expression would be picking on something that
    makes no difference to anything downstream. `x > 5` and `x > y` produce the
    same shape of output and break a pipeline in the same way, which is not at
    all.
    """

    var left: Int
    """The position of the left operand, or of the only one against a constant.
    """

    var right: Int
    """The position of the right operand. Ignored when there is a constant."""

    var op: BinaryOp
    """The operation."""

    var name: String
    """The name the appended column gets in the output schema."""

    var constant: Optional[Value]
    """The constant operand, if the other side is not a column."""

    var value_on_left: Bool
    """True for `5 - x` rather than `x - 5`. Only read with a constant."""

    def __init__(out self, left: Int, right: Int, op: BinaryOp, name: String):
        """Constructs a computed column from two columns.

        Args:
            left: The position of the left operand.
            right: The position of the right operand.
            op: The operation.
            name: The name of the appended column.
        """
        self.left = left
        self.right = right
        self.op = op
        self.name = name
        self.constant = None
        self.value_on_left = False

    def __init__(
        out self,
        column: Int,
        var constant: Value,
        op: BinaryOp,
        name: String,
        value_on_left: Bool = False,
    ):
        """Constructs a computed column from one column and a constant.

        Args:
            column: The position of the column operand.
            constant: The constant operand. Consumed.
            op: The operation.
            name: The name of the appended column.
            value_on_left: True if the constant is the left operand, which
                changes the answer for subtraction, division and the four
                ordered comparisons.
        """
        self.left = column
        self.right = column
        self.op = op
        self.name = name
        self.constant = constant^
        self.value_on_left = value_on_left

    def bind(mut self, var input: Schema) raises -> Schema:
        """Reports the input schema with the computed column appended.

        The result type comes from the two operand types and the operation, so
        it is known here, before a row moves, and a wrong operand position or an
        operation with no answer on those types is an error at plan time rather
        than on the first chunk.

        Args:
            input: The schema of the chunks that will arrive. Consumed.

        Returns:
            The input schema with one field on the end.

        Raises:
            If a position is outside the schema, or the operation is not defined
            on those two types.
        """
        var out = input^
        if self.left < 0 or self.left >= len(out):
            raise Error(
                "compute: column "
                + String(self.left)
                + " is outside a schema of "
                + String(len(out))
                + " columns"
            )
        if self.right < 0 or self.right >= len(out):
            raise Error(
                "compute: column "
                + String(self.right)
                + " is outside a schema of "
                + String(len(out))
                + " columns"
            )
        # Two interior references into one list cannot be alive at once, so the
        # two operand types are read one at a time.
        var a = out[self.left].dtype
        if self.constant:
            var k = self.constant.value().type
            var left = a if not self.value_on_left else k
            var right = k if not self.value_on_left else a
            out.append(Field(self.name, binary_type(self.op, left, right)))
            return out^
        var b = out[self.right].dtype
        out.append(Field(self.name, binary_type(self.op, a, b)))
        return out^

    def process(mut self, var chunk: Chunk) raises -> Optional[Chunk]:
        """Computes the column and puts it on the end of the chunk.

        Args:
            chunk: The chunk. Consumed.

        Returns:
            The chunk with one more column and the same number of rows.

        Raises:
            If a position is outside the chunk, or the operation is not defined
            on the two columns.
        """
        var width = chunk.width()
        if self.left < 0 or self.left >= width:
            raise Error(
                "compute: column "
                + String(self.left)
                + " is outside a chunk of "
                + String(width)
                + " columns"
            )
        if self.right < 0 or self.right >= width:
            raise Error(
                "compute: column "
                + String(self.right)
                + " is outside a chunk of "
                + String(width)
                + " columns"
            )
        var made: AnyArray
        if self.constant:
            made = binary_value_any(
                chunk.columns[self.left],
                self.constant.value(),
                self.op,
                self.value_on_left,
            )
        else:
            made = binary_any(
                chunk.columns[self.left], chunk.columns[self.right], self.op
            )
        var rows = len(chunk)
        var columns = chunk^.into_columns()
        columns.append(made^)
        return Chunk(columns^, rows)


struct Cast(Movable):
    """Converts one column of the chunk to another type, in place.

    In place is right here and it is wrong for `Compute`, and the difference is
    what the operation means. A cast says this column is that type now, so the
    position keeps its meaning and everything downstream that referred to it
    still refers to the same thing. An expression makes a new column, and giving
    it a position of its own is what lets the old one still be read.
    """

    var on: Int
    """The position of the column to convert."""

    var to: LogicalType
    """The type to convert it to."""

    var strict: Bool
    """Whether text that is not a number raises rather than becoming a null."""

    def __init__(out self, on: Int, to: LogicalType, strict: Bool = True):
        """Constructs a cast.

        Args:
            on: The position of the column.
            to: The target type.
            strict: Whether a text value that is not a number raises rather than
                becoming a null. Ignored for a column that is not text.
        """
        self.on = on
        self.to = to
        self.strict = strict

    def bind(mut self, var input: Schema) raises -> Schema:
        """Reports the input schema with one field's type changed.

        Args:
            input: The schema of the chunks that will arrive. Consumed.

        Returns:
            The same schema with the cast column's type replaced.

        Raises:
            If the position is outside the schema.
        """
        var out = input^
        if self.on < 0 or self.on >= len(out):
            raise Error(
                "cast: column "
                + String(self.on)
                + " is outside a schema of "
                + String(len(out))
                + " columns"
            )
        var fields = List[Field](capacity=len(out))
        for i in range(len(out)):
            if i == self.on:
                var name = out[i].name
                var nullable = out[i].nullable
                fields.append(Field(name, self.to, nullable))
            else:
                fields.append(out[i].copy())
        return Schema(fields^)

    def process(mut self, var chunk: Chunk) raises -> Optional[Chunk]:
        """Converts the column and hands the chunk back.

        Args:
            chunk: The chunk. Consumed.

        Returns:
            The chunk with one column converted.

        Raises:
            If the position is outside the chunk, or the conversion fails.
        """
        var width = chunk.width()
        if self.on < 0 or self.on >= width:
            raise Error(
                "cast: column "
                + String(self.on)
                + " is outside a chunk of "
                + String(width)
                + " columns"
            )
        var rows = len(chunk)
        var columns = chunk^.into_columns()
        columns[self.on] = cast_any(columns[self.on], self.to, self.strict)
        return Chunk(columns^, rows)


struct GroupAgg(Copyable, Movable, Writable):
    """One output column of a grouped aggregation."""

    var column: Int
    """The position of the column to reduce, in the node's input."""

    var kind: AggKind
    """Which reduction."""

    var name: String
    """The name the output column gets."""

    def __init__(out self, column: Int, kind: AggKind, name: String):
        """Constructs one output column.

        Args:
            column: The position of the column to reduce.
            kind: The reduction.
            name: The output column's name.
        """
        self.column = column
        self.kind = kind
        self.name = name

    def write_to(self, mut writer: Some[Writer]):
        """Writes the aggregate as it would be read back.

        Args:
            writer: The sink.
        """
        writer.write(self.kind, "(", self.column, ") as ", self.name)


def _folds(kind: AggKind) -> Bool:
    """Reports whether a reduction can be computed a chunk at a time.

    A reduction folds when the answer over two pieces can be recovered from the
    answers over each piece. A sum of sums is a sum and a maximum of maxima is a
    maximum, so those two need nothing but their own running value. A median of
    medians is not a median and no amount of state short of the values
    themselves makes it one, so the order statistics do not fold, and neither
    does a distinct count, whose partial answer has thrown away exactly the
    thing the merge would need.

    A mean folds, but not as a mean: the running state is a sum and a count and
    the division happens once at the end. That is why the node keeps state
    columns rather than output columns, and it is the only kind where the two
    are not the same thing.

    Args:
        kind: The reduction.

    Returns:
        True if `Group` can run it, False if it belongs on `Materialize`.
    """
    return (
        kind == AggKind.SUM
        or kind == AggKind.MEAN
        or kind == AggKind.MIN
        or kind == AggKind.MAX
        or kind == AggKind.COUNT
        or kind == AggKind.SIZE
        or kind == AggKind.FIRST
        or kind == AggKind.LAST
    )


def _merge_kind(kind: AggKind) -> AggKind:
    """Returns the reduction that combines two partial answers of a kind.

    Args:
        kind: The reduction that produced the partials.

    Returns:
        The reduction to run over the partials. A sum for the two counts,
        because merging counts means adding them rather than counting them
        again, and the kind itself for everything else.
    """
    if kind == AggKind.COUNT or kind == AggKind.SIZE:
        return AggKind.SUM
    return kind


def _agg_type(kind: AggKind, input: LogicalType) -> LogicalType:
    """Returns the logical type a reduction produces over a column.

    `AggKind.result_dtype` answers this in physical dtypes, which is the right
    answer for a kernel and not enough for a schema: a minimum over a column of
    timestamps is a timestamp and not an int64, and the only way to keep that is
    to hand the input type straight back for the reductions that report a value
    the column held.

    Args:
        kind: The reduction.
        input: The logical type of the column being reduced.

    Returns:
        The logical type of the output column.
    """
    if kind == AggKind.COUNT or kind == AggKind.SIZE:
        return LogicalType.INT64
    if kind == AggKind.MEAN:
        return LogicalType.FLOAT64
    if kind == AggKind.SUM:
        var acc = accumulator(input.physical)
        if acc == DType.float64:
            return LogicalType.FLOAT64
        if acc == DType.int64:
            return LogicalType.INT64
        return LogicalType.UINT64
    return input


def _mean_of(sums: AnyArray, counts: AnyArray) raises -> AnyArray:
    """Divides a running sum by a running count, one group at a time.

    A group whose count is zero saw no value that was not null, and pandas calls
    that null rather than a division by zero, so the row is left null rather
    than filled with a NaN that would compare unequal to itself downstream.

    Args:
        sums: One sum per group.
        counts: One count of non-null values per group.

    Returns:
        One mean per group, float64, null where the count is zero.

    Raises:
        If the sums cannot be converted to float64.
    """
    var totals = cast_any(sums, DType.float64)
    var t = totals.unsafe_ptr[DType.float64]()
    var n = counts.unsafe_ptr[DType.int64]()
    var out = Array[DType.float64](len(counts))
    var dst = out.unsafe_ptr()
    for g in range(len(out)):
        var count = n.unsafe_offset(g).unsafe_load()
        if count == 0:
            out.set_null(g)
        else:
            dst.unsafe_offset(g).unsafe_store(
                t.unsafe_offset(g).unsafe_load() / Float64(count)
            )
    return AnyArray(out^)


struct Group(Movable):
    """Groups rows by one or more key columns and reduces each group.

    This is the first breaker that is not `Materialize`, and the difference
    between the two is the only thing about it worth understanding. A
    materialised group by holds every row until the last one has arrived and
    then groups ten million rows at once, so the memory it needs is the size of
    the input. This one holds one row per group: it groups each chunk on its
    own, merges that chunk's answers into a running table of groups, and throws
    the chunk away. A billion rows in a thousand groups is a table of a thousand
    rows the whole way through, which is the difference between a query that
    runs and one that does not.

    The merge is the same operation as the aggregation. Two partial answers for
    a group are two rows, and reducing two rows to one is what a group by does,
    so merging the running table with a chunk's table is a group by over their
    concatenation. That is why there is no accumulator kernel here and no
    second implementation of anything: the running table and the chunk's table
    are stacked, grouped by the same keys, and reduced by the kind that combines
    partials, which is the kind itself except for the two counts, where merging
    means adding rather than counting again.

    Not every reduction survives that. `_folds` is the list that does, and the
    rest keep the `Materialize` fallback, which is the right answer rather than a
    gap: a median needs the values and there is no state short of the values that
    would give it one. Asking for one here is an error at plan time.

    Two things this node deliberately does not do. It does not sort, so the
    groups come out in the order they were first seen, which is what pandas
    calls `sort=False`; a sort over one row per group is a separate operator and
    putting it here would make every query pay for it. It does not drop groups
    whose key is null, which is `dropna=False`; that is a filter over the result
    and the plan can add one. Both are decisions about the output rather than
    about the grouping, and neither of them needs to see a row of input.

    Floating point is the one place the answer can differ from the materialised
    path. Adding a column in chunks and then adding the chunk sums is a different
    order of additions from adding it in one pass, and floating point addition is
    not associative, so a sum of floats can differ in the last bits. Every other
    kind here is exact.
    """

    var keys: List[Int]
    """The positions of the key columns, in the order the output carries them."""

    var aggs: List[GroupAgg]
    """What to compute for each group."""

    var input: Schema
    """The schema of the chunks coming in, filled in by `bind`."""

    var output: Schema
    """The schema of the rows going out, worked out by `bind`."""

    var state: List[AnyArray]
    """The running table: the key columns, then one column per state slot, one
    row per group seen so far."""

    var _source: List[Int]
    """Per state slot, the input column it reduces."""

    var _produce: List[AggKind]
    """Per state slot, the reduction run over a chunk."""

    var _merge: List[AggKind]
    """Per state slot, the reduction that combines two partial answers."""

    var _at: List[Int]
    """Per aggregate, the state slot it starts at. A mean owns two."""

    var started: Bool
    """Whether a chunk with rows in it has arrived."""

    var ran: Bool
    """Whether `finish` has turned the running table into output chunks."""

    var emit: List[AnyArray]
    """The result, in chunks, in reverse order so `finish` can pop."""

    var width: Int
    """The number of output columns."""

    def __init__(out self, var keys: List[Int], var aggs: List[GroupAgg]):
        """Constructs a group by.

        Args:
            keys: The positions of the key columns. Consumed.
            aggs: What to compute for each group. Consumed.
        """
        self.keys = keys^
        self.aggs = aggs^
        self.input = Schema()
        self.output = Schema()
        self.state = List[AnyArray]()
        self._source = List[Int]()
        self._produce = List[AggKind]()
        self._merge = List[AggKind]()
        self._at = List[Int]()
        self.started = False
        self.ran = False
        self.emit = List[AnyArray]()
        self.width = 0

    def bind(mut self, var input: Schema) raises -> Schema:
        """Checks the keys and the aggregates, and reports the output schema.

        Everything that can be wrong with a group by that does not depend on the
        data is wrong here: a key that is not a column, a key given twice, a
        reduction that does not fold, a sum of a column of names, two output
        columns with the same name. None of those needs a row to detect and all
        of them are cheaper to report before the first one moves.

        Args:
            input: The schema of the chunks that will arrive. Consumed.

        Returns:
            The key columns in the order they were given, then one column per
            aggregate.

        Raises:
            If a position is outside the schema, if a key is repeated, if a
            reduction does not fold, if a reduction has no meaning on its
            column's type, or if two output columns would have the same name.
        """
        self.input = input^
        var fields = List[Field]()

        if len(self.keys) == 0:
            raise Error("group: at least one key column is required")
        for k in range(len(self.keys)):
            var at = self.keys[k]
            if at < 0 or at >= len(self.input):
                raise Error(
                    "group: key column "
                    + String(at)
                    + " is outside a schema of "
                    + String(len(self.input))
                    + " columns"
                )
            for j in range(k):
                if self.keys[j] == at:
                    raise Error(
                        "group: key column " + String(at) + " was given twice"
                    )
            fields.append(self.input[at].copy())

        for a in range(len(self.aggs)):
            var at = self.aggs[a].column
            var kind = self.aggs[a].kind
            if at < 0 or at >= len(self.input):
                raise Error(
                    "group: column "
                    + String(at)
                    + " is outside a schema of "
                    + String(len(self.input))
                    + " columns"
                )
            if not _folds(kind):
                raise Error(
                    "group: "
                    + String(kind)
                    + " cannot be computed a chunk at a time"
                )
            var source = self.input[at].dtype
            if source.is_variable_width() and (
                kind == AggKind.SUM or kind == AggKind.MEAN
            ):
                raise Error(
                    "group: " + String(kind) + " is not defined on text"
                )
            var name = self.aggs[a].name
            for f in range(len(fields)):
                if fields[f].name == name:
                    raise Error(
                        "group: two output columns would both be called " + name
                    )
            fields.append(Field(name, _agg_type(kind, source)))

            self._at.append(len(self._source))
            if kind == AggKind.MEAN:
                # A mean is a sum and a count until the last moment. Keeping the
                # two apart is what lets the merge be an addition, and dividing
                # earlier would make the running value a mean of means, which is
                # only the mean when every group is the same size.
                self._source.append(at)
                self._produce.append(AggKind.SUM)
                self._merge.append(AggKind.SUM)
                self._source.append(at)
                self._produce.append(AggKind.COUNT)
                self._merge.append(AggKind.SUM)
            else:
                self._source.append(at)
                self._produce.append(kind)
                self._merge.append(_merge_kind(kind))

        self.output = Schema(fields^)
        return Schema(copy=self.output)

    def update_state(self) -> NodeStatus:
        """Reports whether the running table has been turned into output.

        Returns:
            NEED_MORE_INPUT until `finish` has run, then HAVE_OUTPUT while
            chunks remain and FINISHED after that.
        """
        if not self.ran:
            return NodeStatus.NEED_MORE_INPUT
        if len(self.emit) > 0:
            return NodeStatus.HAVE_OUTPUT
        return NodeStatus.FINISHED

    def process(mut self, var chunk: Chunk) raises -> Optional[Chunk]:
        """Groups the chunk and merges its answers into the running table.

        Args:
            chunk: The chunk. Consumed.

        Returns:
            None, always. A breaker has nothing to say until it has seen
            everything.

        Raises:
            If the chunk is not as wide as the input schema, or if a reduction
            fails on the chunk's data.
        """
        if chunk.width() != len(self.input):
            raise Error(
                "group: chunk has "
                + String(chunk.width())
                + " columns and the input schema has "
                + String(len(self.input))
            )
        var rows = len(chunk)
        if rows == 0:
            return None
        var columns = chunk^.into_columns()
        var refs = borrow_columns(columns)
        var local = group_ordinals(refs, self.keys, rows)
        var made = List[AnyArray](capacity=len(self.keys) + len(self._source))
        for k in range(len(self.keys)):
            made.append(take_any(columns[self.keys[k]], local.rows_at))
        for s in range(len(self._source)):
            made.append(
                aggregate_group_any(
                    columns[self._source[s]],
                    self._produce[s],
                    local.codes,
                    local.groups,
                )
            )
        self._absorb(made^)
        return None

    def _absorb(mut self, var made: List[AnyArray]) raises:
        """Merges one chunk's per group answers into the running table.

        Args:
            made: The chunk's table: key columns then state columns. Consumed.

        Raises:
            If the merge fails.
        """
        if not self.started:
            self.started = True
            self.state = made^
            return

        var stacked = List[AnyArray](capacity=len(made))
        for i in range(len(self.state)):
            stacked.append(concat_two_any(self.state[i], made[i]))
        var rows = len(stacked[0])
        var refs = borrow_columns(stacked)
        var at = List[Int](capacity=len(self.keys))
        for k in range(len(self.keys)):
            at.append(k)
        var merged = group_ordinals(refs, at, rows)

        var next = List[AnyArray](capacity=len(stacked))
        for k in range(len(self.keys)):
            next.append(take_any(stacked[k], merged.rows_at))
        for s in range(len(self._source)):
            next.append(
                aggregate_group_any(
                    stacked[len(self.keys) + s],
                    self._merge[s],
                    merged.codes,
                    merged.groups,
                )
            )
        self.state = next^

    def finish(mut self) raises -> Optional[Chunk]:
        """Hands the running table back, one chunk at a time.

        Returns:
            One chunk of the result per call, in first seen group order, and
            None when there are none left.

        Raises:
            If turning the running state into output fails.
        """
        if not self.ran:
            self.ran = True
            self._settle()
        if len(self.emit) == 0:
            return None
        var row = List[AnyArray](capacity=self.width)
        for _ in range(self.width):
            row.append(self.emit.pop())
        return Chunk(row^)

    def _settle(mut self) raises:
        """Turns the running state into output columns and cuts them into chunks.

        Raises:
            If a mean cannot be computed from its sum and its count.
        """
        self.width = len(self.keys) + len(self.aggs)
        if not self.started:
            return

        var out = List[AnyArray](capacity=self.width)
        for k in range(len(self.keys)):
            out.append(AnyArray(copy=self.state[k]))
        var base = len(self.keys)
        for a in range(len(self.aggs)):
            var at = base + self._at[a]
            if self.aggs[a].kind == AggKind.MEAN:
                out.append(_mean_of(self.state[at], self.state[at + 1]))
            else:
                out.append(AnyArray(copy=self.state[at]))
        self.state = List[AnyArray]()

        # Same reversed, chunk major layout `_stripe` produces, for the same
        # reason: `finish` pops `width` arrays off the back and has one chunk in
        # column order without copying anything.
        var total = len(out[0])
        var pieces = (total + MORSEL_ROWS - 1) // MORSEL_ROWS
        for c in range(pieces - 1, -1, -1):
            var begin = c * MORSEL_ROWS
            var stop = min(begin + MORSEL_ROWS, total)
            for i in range(self.width - 1, -1, -1):
                self.emit.append(out[i].slice(begin, stop))


struct Materialize(Movable):
    """Collects every chunk, calls a whole frame function, emits chunks again.

    This is the fallback, and the reason the engine can be built one operator at
    a time. An operation with no chunked implementation is wrapped in one of
    these and works exactly as it does today, at exactly today's cost, while
    sitting in a pipeline beside operators that have been ported. Removing a
    fallback is then one self contained change with a benchmark attached.

    It is a breaker: nothing comes out until everything has gone in, because the
    function it wraps wants a whole frame. The chunk boundaries of the input are
    kept on the way in, so the frame the function sees is chunked and the
    kernels that walk chunks still walk them, and the boundaries of whatever the
    function returns are what comes out.

    The function is a plain pointer and captures nothing. That is a real limit
    and it is the right one: an operation that needs an argument, a mask or a
    key list or a join side, is an operation whose state has to live somewhere,
    and the place for it is a node of its own with fields. Wrapping it in a
    closure would hide that the state exists.
    """

    var op: def(var DataFrame) raises thin -> DataFrame
    """The whole frame operation to run once the input is complete."""

    var declared: Schema
    """The schema the caller says `op` produces. Checked when it runs."""

    var input: Schema
    """The schema of the chunks coming in, filled in by `Pipeline.add`."""

    var held: List[ChunkedArray]
    """The chunks seen so far, one column per position."""

    var output: List[AnyArray]
    """The result, in chunks, in reverse order so `finish` can pop."""

    var ran: Bool
    """Whether `op` has been called."""

    var width: Int
    """The number of columns in the output, known once `op` has run."""

    def __init__(out self, op: def(var DataFrame) raises thin -> DataFrame):
        """Constructs a fallback whose function leaves the schema alone.

        Args:
            op: The whole frame operation.
        """
        self.op = op
        self.declared = Schema()
        self.input = Schema()
        self.held = List[ChunkedArray]()
        self.output = List[AnyArray]()
        self.ran = False
        self.width = 0

    def __init__(
        out self,
        op: def(var DataFrame) raises thin -> DataFrame,
        var declared: Schema,
    ):
        """Constructs a fallback whose function changes the schema.

        The caller has to say what comes out, because there is no way to ask a
        function pointer. What the function actually returns is compared against
        this when it runs, so a wrong declaration is an error rather than a
        frame whose column names do not describe its columns.

        Args:
            op: The whole frame operation.
            declared: The schema `op` produces. Consumed.
        """
        self.op = op
        self.declared = declared^
        self.input = Schema()
        self.held = List[ChunkedArray]()
        self.output = List[AnyArray]()
        self.ran = False
        self.width = 0

    def bind(mut self, var input: Schema) raises -> Schema:
        """Records the input schema and reports the output one.

        Args:
            input: The schema of the chunks that will arrive. Consumed.

        Returns:
            The declared output schema, which is the input schema unless the
            caller said otherwise.

        Raises:
            Never. Declared for the dispatch signature.
        """
        if len(self.declared) == 0:
            self.declared = Schema(copy=input)
        self.input = input^
        for i in range(len(self.input)):
            self.held.append(ChunkedArray(self.input[i].dtype))
        return Schema(copy=self.declared)

    def update_state(self) -> NodeStatus:
        """Reports whether the collected input has been turned into output.

        Returns:
            NEED_MORE_INPUT until `finish` has run the function, then
            HAVE_OUTPUT while chunks remain and FINISHED after that.
        """
        if not self.ran:
            return NodeStatus.NEED_MORE_INPUT
        if len(self.output) > 0:
            return NodeStatus.HAVE_OUTPUT
        return NodeStatus.FINISHED

    def process(mut self, var chunk: Chunk) raises -> Optional[Chunk]:
        """Keeps the chunk and emits nothing.

        Args:
            chunk: The chunk. Consumed.

        Returns:
            None, always. A breaker has nothing to say until it has seen
            everything.

        Raises:
            If the chunk's width or its dtypes do not match the input schema.
        """
        if chunk.width() != len(self.held):
            raise Error(
                "materialize: chunk has "
                + String(chunk.width())
                + " columns and the input schema has "
                + String(len(self.held))
            )
        var backwards = chunk^.into_columns()
        var forwards = List[AnyArray](capacity=len(backwards))
        while len(backwards) > 0:
            forwards.append(backwards.pop())
        for i in range(len(self.held)):
            self.held[i].append(forwards.pop())
        return None

    def finish(mut self) raises -> Optional[Chunk]:
        """Runs the function the first time, then hands the result back.

        Returns:
            One chunk of the result per call, in order, and None when there are
            none left.

        Raises:
            If the function raises, or returns a frame that does not match the
            declared schema.
        """
        if not self.ran:
            self.ran = True
            var flipped = List[ChunkedArray](capacity=len(self.held))
            while len(self.held) > 0:
                flipped.append(self.held.pop())
            var columns = List[ChunkedArray](capacity=len(flipped))
            while len(flipped) > 0:
                columns.append(flipped.pop())
            var frame = DataFrame(Schema(copy=self.input), columns^)
            var result = self.op(frame^)
            if result.schema != self.declared:
                raise Error(
                    "materialize: the operation returned "
                    + String(result.schema)
                    + " and was declared to return "
                    + String(self.declared)
                )
            self.width = result.width()
            self.output = _stripe(result^)
        if len(self.output) == 0:
            return None
        var row = List[AnyArray](capacity=self.width)
        for _ in range(self.width):
            row.append(self.output.pop())
        return Chunk(row^)


def _stripe(var frame: DataFrame) raises -> List[AnyArray]:
    """Lays a frame out as chunks, in reverse order, ready to be popped.

    The result is column major within a chunk and chunk major overall, reversed,
    so popping `width` arrays off the back gives one chunk in column order. That
    is a strange shape to look at and it is the one that lets `finish` hand
    chunks over without copying an array or shuffling a list.

    Args:
        frame: The frame. Consumed.

    Returns:
        Every array of the frame, chunk by chunk, reversed.

    Raises:
        If the frame's columns are not chunked the same way.
    """
    var width = frame.width()
    var columns = frame^.into_columns()
    var flipped = List[List[AnyArray]](capacity=width)
    while len(columns) > 0:
        flipped.append(columns.pop().into_chunks())
    var pieces = List[List[AnyArray]](capacity=width)
    while len(flipped) > 0:
        pieces.append(flipped.pop())
    var count = -1
    for i in range(width):
        if count < 0:
            count = len(pieces[i])
        elif len(pieces[i]) != count:
            raise Error(
                "materialize: column "
                + String(i)
                + " has "
                + String(len(pieces[i]))
                + " chunks and column 0 has "
                + String(count)
            )
    var out = List[AnyArray]()
    if count <= 0:
        return out^
    for _ in range(count):
        for i in range(width - 1, -1, -1):
            out.append(pieces[i].pop())
    return out^


comptime Node = Variant[
    Filter, Project, Compute, Cast, Limit, Group, Materialize
]
"""One operator, as a value the pipeline can hold in a list.

Mojo 1.0 can express a trait object but not a list of them, so this is a closed
union and the five functions below are the dispatch. A closed set is also what
an engine has: the operators are the ones in this file.
"""


def node_bind(mut node: Node, var input: Schema) raises -> Schema:
    """Tells a node what its input looks like and asks what its output does.

    Called once per node when the pipeline is built, before any row moves, so a
    node that needs to know its schema, which today is only `Materialize`, has
    it before the first chunk arrives.

    Args:
        node: The node.
        input: The schema of the chunks it will be given. Consumed.

    Returns:
        The schema of the chunks it will emit.

    Raises:
        If the node cannot accept that input.
    """
    if node.isa[Materialize]():
        return node[Materialize].bind(input^)
    if node.isa[Group]():
        return node[Group].bind(input^)
    if node.isa[Compute]():
        return node[Compute].bind(input^)
    if node.isa[Cast]():
        return node[Cast].bind(input^)
    if node.isa[Project]():
        ref keep = node[Project].keep
        var fields = List[Field](capacity=len(keep))
        for i in range(len(keep)):
            if keep[i] < 0 or keep[i] >= len(input):
                raise Error(
                    "project: column "
                    + String(keep[i])
                    + " is outside a schema of "
                    + String(len(input))
                    + " columns"
                )
            fields.append(input[keep[i]].copy())
        return Schema(fields^)
    return input^


def node_status(node: Node) -> NodeStatus:
    """Asks a node what it wants next.

    Args:
        node: The node.

    Returns:
        The node's state.
    """
    if node.isa[Limit]():
        return node[Limit].update_state()
    if node.isa[Materialize]():
        return node[Materialize].update_state()
    if node.isa[Group]():
        return node[Group].update_state()
    return NodeStatus.NEED_MORE_INPUT


def node_is_breaker(node: Node) -> Bool:
    """Reports whether a node has to see all its input before it emits.

    Args:
        node: The node.

    Returns:
        True for a breaker, which is where a pipeline is cut.
    """
    return node.isa[Materialize]() or node.isa[Group]()


def node_process(mut node: Node, var chunk: Chunk) raises -> Optional[Chunk]:
    """Pushes one chunk through a node.

    Args:
        node: The node.
        chunk: The chunk. Consumed.

    Returns:
        What the node emits, or None if it emits nothing for this chunk.

    Raises:
        If the node cannot process the chunk.
    """
    if node.isa[Filter]():
        return node[Filter].process(chunk^)
    if node.isa[Project]():
        return node[Project].process(chunk^)
    if node.isa[Compute]():
        return node[Compute].process(chunk^)
    if node.isa[Cast]():
        return node[Cast].process(chunk^)
    if node.isa[Limit]():
        return node[Limit].process(chunk^)
    if node.isa[Group]():
        return node[Group].process(chunk^)
    return node[Materialize].process(chunk^)


def node_finish(mut node: Node) raises -> Optional[Chunk]:
    """Asks a node for what it has been holding, one chunk at a time.

    Called repeatedly after the input is exhausted until it returns None, so a
    breaker holding ten million rows gives them back in chunks rather than as
    one array.

    Args:
        node: The node.

    Returns:
        The next chunk of the node's result, or None when there is none.

    Raises:
        If producing the result raises.
    """
    if node.isa[Materialize]():
        return node[Materialize].finish()
    if node.isa[Group]():
        return node[Group].finish()
    return None
