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

`Reduce` is the same idea with nothing to group by, which makes it the cheapest
breaker there is: it holds one row whatever it is given. It is a separate node
rather than a `Group` with an empty key list because a group by hashes every row
to find out where it belongs and there is nothing here to find out, and it earns
its place in a query that ends in a reduction, where the last operator's output
is folded away while it is still in cache instead of being written to memory for
something else to read back.

`Join` is the one operator here that is not a breaker and still holds something
between chunks, and the something is not state: it is a table it built once
before the first chunk arrived and only ever reads afterwards. That is why it
counts as row local, and it is the whole reason a join belongs in a pipeline at
all. A join done as a whole frame operation writes its output to memory and then
whatever comes next reads it back, which on a five column join of a million rows
is a hundred and sixty megabytes each way for an answer that might be three
numbers. Done a chunk at a time, ahead of a `Reduce`, the chunk that came out of
the probe is folded away while it is still in L2.

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
from firepanda.bitmap.bitmap import Bitmap
from firepanda.dtype.lists import ALL
from firepanda.dtype.logical import LogicalType
from firepanda.dtype.schema import Field, Schema
from firepanda.frame.frame import DataFrame
from firepanda.hash.grouping import group_ordinals
from firepanda.hash.lasting import LastingKeys
from firepanda.join.keys import BuildSide, build_side, probe_side
from firepanda.join.pairs import (
    JoinKind,
    ProbeTable,
    bucket_side,
    pair_probe,
)
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
from firepanda.kernel.reduce import reduce_any
from firepanda.kernel.running import (
    accumulate_any,
    settle_any,
    state_capacity,
    widen_any,
)
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

    def process(self, var chunk: Chunk) raises -> Optional[Chunk]:
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

    def process(self, var chunk: Chunk) raises -> Optional[Chunk]:
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

    def process(self, var chunk: Chunk) raises -> Optional[Chunk]:
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

    def process(self, var chunk: Chunk) raises -> Optional[Chunk]:
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


struct Join(Movable):
    """Pairs every chunk against a frame it built a table from once.

    The build side is the whole right frame and it is held here, because a join
    in a pipeline is a join whose right side is already in memory and whose left
    side is arriving. `bind` hashes the right frame's key into a `BuildSide` and
    buckets it into a `ProbeTable`, and after that both are read only, so
    `process` is a pure function of its chunk and the node can be handed to every
    core at once.

    What that buys is not a faster probe. Pairing a million rows against a
    thousand is half a millisecond and the whole join is three, so five sixths of
    a join is the gathers that build its output. Those gathers do not get cheaper
    here. What gets cheaper is what happens to their output: a whole frame join
    writes a hundred and sixty megabytes and hands them to the next operator,
    which reads them back, and a join in a pipeline hands the next operator a
    chunk that is still in cache. On a query that ends in a reduction the
    intermediate is never written at all.

    ## What it will not do

    Right and outer joins are refused, and so is a key that is not a single
    fixed width column on each side. Both refusals are the same refusal: this
    node emits a chunk per chunk and nothing else. An outer join has to emit the
    right rows that nothing matched, which it cannot know until the last chunk
    has gone past, so it is a breaker wearing this node's clothes. A right join
    is an outer join's left half by the same argument. A composite or text key
    needs the ordinal space that `align_keys` builds by concatenating both sides,
    and concatenating both sides is having them both, which a stream does not.

    Neither refusal loses anything. A planner that meets one of them uses
    `Materialize` and the whole frame `join_on`, which is what it did before this
    node existed.
    """

    var right: DataFrame
    """The build side, held whole."""

    var left_on: String
    """The key column's name on the probe side."""

    var right_on: String
    """The key column's name on the build side."""

    var kind: JoinKind
    """Which rows to keep."""

    var suffix: String
    """Appended to a right column whose name collides."""

    var columns: List[String]
    """The projection, or empty for every output column.

    Here rather than left to a `Project` afterwards for the reason `join_on` has
    it: gathering is most of what a join costs, so a column that is going to be
    dropped is not a small waste at the end, it is most of the work.
    """

    var _left_at: Int
    """Where the key sits in the chunk, settled by `bind`."""

    var _side: BuildSide
    """The right frame's key table, filled by `bind`."""

    var _table: ProbeTable
    """The right frame's code to row lists, filled by `bind`."""

    var _absent: List[Bool]
    """Which right rows have a null key, or empty when none do."""

    var _from_right: List[Bool]
    """Per wanted column, whether it comes from the right frame."""

    var _source: List[Int]
    """Per wanted column, its position in the frame it comes from."""

    def __init__(
        out self,
        var right: DataFrame,
        var left_on: String,
        var right_on: String,
        kind: JoinKind = JoinKind.INNER,
        var suffix: String = "_right",
        var columns: List[String] = List[String](),
    ):
        """Constructs a join against a frame.

        Nothing is built here. The table is built in `bind`, because building it
        needs the key dtype and the key dtype is a question about the chunks that
        will arrive as much as about the frame held here.

        Args:
            right: The build side. Consumed.
            left_on: The key column's name on the probe side. Consumed.
            right_on: The key column's name on the build side. Consumed.
            kind: Which rows to keep. Inner, left, semi and anti only.
            suffix: Appended to a right column whose name collides. Consumed.
            columns: Which output columns to build, in the order wanted, or
                empty for all of them in their natural order. Consumed.
        """
        self.right = right^
        self.left_on = left_on^
        self.right_on = right_on^
        self.kind = kind
        self.suffix = suffix^
        self.columns = columns^
        self._left_at = -1
        self._side = BuildSide()
        self._table = ProbeTable()
        self._absent = List[Bool]()
        self._from_right = List[Bool]()
        self._source = List[Int]()

    def bind(mut self, var input: Schema) raises -> Schema:
        """Builds the table from the right frame and plans the output.

        Args:
            input: The schema of the chunks that will arrive. Consumed.

        Returns:
            The schema of the chunks this emits.

        Raises:
            If the kind is one this node does not do, if either key name is
            missing, if the two keys have different dtypes, if the key is text,
            or if a projected name is not one the result has.
        """
        if self.kind == JoinKind.RIGHT or self.kind == JoinKind.OUTER:
            raise Error(
                "join: a "
                + String(self.kind)
                + " join has to emit right rows that nothing matched, which is"
                " not known until the last chunk; use the whole frame join"
            )
        if self.kind == JoinKind.CROSS:
            raise Error("join: a cross join has no key to build a table from")

        var here = input.index_of(self.left_on)
        var there = self.right.schema.index_of(self.right_on)
        var dt = self.right.schema[there].dtype.physical
        if (
            input[here].dtype.physical != dt
            or self.right.columns[there].only().is_string()
        ):
            raise Error(
                "join: this node needs one fixed width key of the same dtype on"
                " each side; got "
                + String(input[here].dtype.physical)
                + " and "
                + String(dt)
            )

        var rows = self.right.rows
        var codes = Array[DType.uint32](overwritten=rows)
        var side = _build_key(self.right.columns[there].only(), codes)
        var absent = _key_nulls(self.right.columns[there].only(), rows)
        self._side = side^
        self._absent = absent^
        self._left_at = here
        self._table = bucket_side(
            codes,
            0,
            rows,
            self._absent,
            0,
            len(self._absent) > 0,
            self._side.groups(),
        )

        # The same plan `join_on` makes, without the coalescing branch: an
        # output row of these four kinds always has a probe side row behind it,
        # so a shared key column is gathered from the probe side and never
        # filled from both.
        var fields = List[Field]()
        var from_right = List[Bool]()
        var source = List[Int]()
        for i in range(len(input)):
            var kind = input[i].dtype
            fields.append(Field(input[i].name, kind))
            from_right.append(False)
            source.append(i)
        if self.kind.keeps_right_columns():
            for j in range(len(self.right.columns)):
                if j == there and self.left_on == self.right_on:
                    continue
                var name = self.right.schema[j].name
                if _names_include(fields, name):
                    name = name + self.suffix
                    if _names_include(fields, name):
                        raise Error(
                            "join: the right frame's column '"
                            + self.right.schema[j].name
                            + "' collides and so does '"
                            + name
                            + "'; pass a different suffix"
                        )
                fields.append(Field(name, self.right.schema[j].dtype))
                from_right.append(True)
                source.append(j)

        var kept = List[Field]()
        self._from_right = List[Bool]()
        self._source = List[Int]()
        if len(self.columns) == 0:
            for i in range(len(fields)):
                kept.append(fields[i].copy())
                self._from_right.append(from_right[i])
                self._source.append(source[i])
        else:
            for c in range(len(self.columns)):
                var found = -1
                for i in range(len(fields)):
                    if fields[i].name == self.columns[c]:
                        found = i
                        break
                if found < 0:
                    raise Error(
                        "join: the result has no column '"
                        + self.columns[c]
                        + "' to keep"
                    )
                for w in range(len(kept)):
                    if kept[w].name == self.columns[c]:
                        raise Error(
                            "join: column '"
                            + self.columns[c]
                            + "' was asked for twice"
                        )
                kept.append(fields[found].copy())
                self._from_right.append(from_right[found])
                self._source.append(source[found])
        return Schema(kept^)

    def process(
        self, var chunk: Chunk, spread: Bool = True
    ) raises -> Optional[Chunk]:
        """Probes one chunk against the built table and gathers what paired.

        Args:
            chunk: The chunk. Consumed.
            spread: Whether this chunk may be worked on by more than one core.
                False when a worker is running this, which is the ordinary case
                in a parallel pipeline and is what `node_apply` passes. The
                probe, the pairing and both gathers each have a row count above
                which they hand themselves out in morsels, and a pipeline chunk
                is above two of those, so a worker calling this without the flag
                starts a second layer of tasks inside the one it is already in.
                Measured at ten million rows joined against ten thousand, that
                nesting cost nothing, because a task that finds no free worker
                runs on the one that made it. It is still wrong to ask for, and
                it stops being free the moment the queue is not saturated.

        Returns:
            The paired rows, or None when nothing paired, which is a chunk of no
            rows and is skipped rather than pushed.

        Raises:
            If the node was not bound, or if the probe or a gather raises.
        """
        if self._left_at < 0:
            raise Error("join: this node has not been bound to a schema")
        var rows = len(chunk)
        if rows == 0:
            return None

        ref key = chunk.columns[self._left_at]
        var codes = Array[DType.uint32](overwritten=rows)
        _probe_key(self._side, key, codes, spread)
        var absent = _key_nulls(key, rows)
        var matched = Bitmap(0, all_valid=False)
        var pairs = pair_probe(
            self._table,
            codes,
            0,
            rows,
            absent,
            0,
            len(absent) > 0,
            self.kind,
            matched,
            spread,
        )
        if len(pairs) == 0:
            return None

        var out = List[AnyArray](capacity=len(self._source))
        for w in range(len(self._source)):
            if self._from_right[w]:
                out.append(
                    take_any(
                        self.right.columns[self._source[w]].only(),
                        pairs.right_at,
                        spread,
                    )
                )
            else:
                out.append(
                    take_any(
                        chunk.columns[self._source[w]], pairs.left_at, spread
                    )
                )
        var height = len(pairs)
        _ = chunk^
        return Chunk(out^, height)


def _names_include(fields: List[Field], name: String) -> Bool:
    """Reports whether a field of that name has already been planned.

    Args:
        fields: The output fields so far.
        name: The name to look for.

    Returns:
        True if one of them is called that.
    """
    for i in range(len(fields)):
        if fields[i].name == name:
            return True
    return False


def _key_nulls(key: AnyArray, rows: Int) raises -> List[Bool]:
    """Flags every row of a key column whose value is missing.

    The null count is asked first, because a key column with no nulls is the
    ordinary case and the loop below is a pass over the column that answers
    False every time.

    Args:
        key: The key column.
        rows: How many rows of it to look at.

    Returns:
        One flag per row, or an empty list when the column has no nulls.

    Raises:
        If reading the validity bitmap raises.
    """
    if key.null_count() == 0:
        return List[Bool]()
    var out = List[Bool](capacity=rows)
    for i in range(rows):
        out.append(not key.is_valid(i))
    return out^


def _build_key(
    key: AnyArray, mut codes: Array[DType.uint32]
) raises -> BuildSide:
    """Builds the key table for a column whose dtype is a runtime value.

    Args:
        key: The build side's key column.
        codes: Filled with one ordinal per row of it.

    Returns:
        The table, ready to probe.

    Raises:
        If the dtype is text or has no fixed width layout.
    """
    # Before the dispatch, because uint8 is in ALL and a string column would
    # match it and build a table over the first byte of each view.
    if not key.is_string():
        comptime for candidate in ALL:
            if key.dtype() == candidate:
                ref view = key.as_typed_view[candidate]()
                return build_side[candidate](view, 0, codes)
    raise Error("join: no key table for dtype " + String(key.dtype()))


def _probe_key(
    built: BuildSide,
    key: AnyArray,
    mut codes: Array[DType.uint32],
    spread: Bool = True,
) raises:
    """Probes the built table with a column whose dtype is a runtime value.

    Args:
        built: The table the build side filled.
        key: The probe side's key column.
        codes: Filled with one ordinal per row of it.
        spread: Whether the probe may use more than one core.

    Raises:
        If the dtype is text, has no fixed width layout, or is not the one the
        table was built from.
    """
    if not key.is_string():
        comptime for candidate in ALL:
            if key.dtype() == candidate:
                ref view = key.as_typed_view[candidate]()
                return probe_side[candidate](built, view, 0, codes, spread)
    raise Error("join: no key table for dtype " + String(key.dtype()))


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
    concatenation, reduced by the kind that combines partials, which is the kind
    itself except for the two counts, where merging means adding rather than
    counting again. That is `_absorb`, it needs no kernel of its own, and it is
    what this node was built on.

    ## Why that is not the whole story

    Stacking the running table with a chunk's table and grouping the result
    costs the height of the running table on every chunk. The running table is
    as tall as the number of groups seen so far, so the work that is not
    proportional to the input grows as the number of chunks times the number of
    groups, and the number of chunks grows with the input. On a thousand groups
    that term is invisible. On a hundred thousand it is the whole cost:
    `group/pipeline_stream_wide` against `group/pipeline_materialize_wide`
    measured 4.3x slower at a million rows and 12.7x at four million, and the
    per row cost sat flat at sixty five nanoseconds while the materialised
    fallback's fell, which is what a term of that shape looks like from the
    outside.

    The fix is to stop rediscovering which group each row belongs to. A chunk's
    keys are looked up in a map that outlives the chunk, so a group keeps the
    same ordinal from the moment it is first seen until the end of the query,
    and a chunk is absorbed by adding its rows to the slots those ordinals name.
    `_push` is that route and `LastingKeys` is that map. Nothing in it is
    proportional to the number of groups: a chunk costs a lookup and a fold per
    row, which is what the materialised fallback pays over the same rows, and
    the running table's height is not read at all. The kernels that do the
    folding are `firepanda/kernel/running.mojo`, and they are the accumulator
    this node was written without.

    Two things keep `_absorb` alive rather than deleting it. The map is either
    an array indexed by the key or a table of 64 bit hashes, and the hash is a
    bijection on the key bits, so both are exact for a fixed width key and
    neither is for text, where two names longer than eight bytes can land on one
    hash. A key tuple of several columns has no single hash that is exact
    either, for the same reason. A running slot is a number in an array as well,
    so a minimum over a column of names has nowhere to live and falls back too.
    So `_push` takes one fixed width key with fixed width values and `_absorb`
    takes everything else. A null key falls back as well, because the map
    reserves no ordinal for one and the group order the result promises is the
    order the groups were first seen, which a reserved ordinal would not give.
    `_demote` is the handover, and it can happen in the middle of a query,
    because whether a key column has a null is not known until the chunk holding
    it arrives.

    What that is worth, on an i9-13900K with the same query over the same rows
    through the same driver, in nanoseconds a row: at a hundred thousand groups
    and sixteen million rows the operator went from 64.054 to 3.199, and the
    materialised fallback it is now measured against takes 5.157. At a thousand
    groups it went from 6.223 to 2.593 against the fallback's 4.315. So the
    breaker is no longer a trade of speed for memory. It holds one row per group
    instead of every row and it is also 1.6x faster than holding every row, at
    both ends of the group count.

    One shape is slower than it was and is meant to be. A single chunk of more
    than four million rows used to go through a factorize that splits across
    workers at that size, and the lasting map is one thread. That is
    `group/pipeline_stream_one_chunk`, 2.409 before and 2.925 after, and the
    engine does not make chunks of four million rows: `MORSEL_ROWS` is a hundred
    and twenty eight thousand, which is where `group/pipeline_stream` sits and
    where the operator is 2.4x faster than it was.

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

    var _map: LastingKeys
    """The key to ordinal map that outlives the chunk, and the keys it has been
    given. Empty unless `_fast`."""

    var _fast: Bool
    """Whether chunks are still going through `_push` rather than `_absorb`."""

    var _values: List[AnyArray]
    """The running table's state columns on the `_push` route, without the key
    beside them. `_gather` puts the two back together into `state`."""

    var _room: Int
    """How many slots each of `_values` holds, which is at least `_groups` and
    grows by doubling, so that a table which gains a few groups on every chunk is
    copied a logarithmic number of times rather than every chunk."""

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
        self._map = LastingKeys()
        self._fast = False
        self._values = List[AnyArray]()
        self._room = 0

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

        # One fixed width key is the shape the persistent table is exact for,
        # and a fixed width value is the shape a running slot can accumulate.
        # Everything else keeps the stacking merge. This is the schema half of
        # the question; the data half is the null check in `process`, which
        # cannot be asked until a chunk arrives.
        self._fast = (
            len(self.keys) == 1
            and not self.input[self.keys[0]].dtype.is_variable_width()
        )
        for s in range(len(self._source)):
            if self.input[self._source[s]].dtype.is_variable_width():
                self._fast = False

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

        if self._fast:
            if columns[self.keys[0]].null_count() == 0:
                self._push(columns, rows)
                return None
            # A null arrived. Hand what the table has built to the running
            # table and let this chunk and every one after it take the route
            # that puts a null group where its first null was.
            self._demote()

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
                    trusted=True,
                )
            )
        self._absorb(made^)
        return None

    def _push(mut self, columns: List[AnyArray], rows: Int) raises:
        """Adds one chunk to the running table through the persistent map.

        Every row's key goes into a map that survives the chunk, so the
        ordinal a group is given the first time it is seen is the ordinal it
        keeps. That makes merging the chunk a reduction over ordinals both sides
        already agree on rather than a group by that has to work out which rows
        of the running table the chunk's rows belong to, and it is the whole
        difference between this and `_absorb`.

        The chunk's keys are looked up rather than grouped, so there is no per
        chunk grouping pass, and its rows are folded into slots rather than
        reduced into a second table, so there is no per chunk table either. What
        is left is a hash, a probe and a fold per row, which is what the
        materialised fallback pays over the same rows once, so this route does
        the fallback's work and holds one row per group while doing it.

        Args:
            columns: The chunk's columns, borrowed.
            rows: The chunk's height.

        Raises:
            If the key dtype has no physical layout, or if a reduction fails.
        """
        ref key = columns[self.keys[0]]
        var before = self._map.groups
        var codes = Array[DType.uint32](rows)
        self._map.ordinals(key, rows, codes)
        var after = self._map.groups

        if before == 0:
            # The first chunk gets its state from the ordinary kernel, which is
            # what settles the dtype each slot accumulates in without this
            # having to work it out. `widen_any` then turns that answer into an
            # accumulator, which is not quite the same thing: a group whose rows
            # were all null comes back holding a zero, and a running minimum has
            # to hold the identity instead or the next chunk compares against a
            # zero that is not a value the column ever had.
            self._room = state_capacity(after, 0)
            for s in range(len(self._source)):
                var made = aggregate_group_any(
                    columns[self._source[s]],
                    self._produce[s],
                    codes,
                    after,
                    trusted=True,
                )
                widen_any(made, self._room, self._produce[s])
                self._values.append(made^)
            self.started = True
            return

        if after > self._room:
            var room = state_capacity(after, self._room)
            for s in range(len(self._values)):
                widen_any(self._values[s], room, self._produce[s])
            self._room = room

        # The rows go straight into the slots their groups already own. Nothing
        # here is proportional to the number of groups, which is the whole point
        # of the route: what a chunk costs is a pass over the chunk, the same as
        # the materialised fallback pays over the same rows, and the running
        # table's height is never read at all.
        for s in range(len(self._source)):
            accumulate_any(
                self._values[s],
                columns[self._source[s]],
                self._produce[s],
                codes,
                rows,
            )

    def _gather(mut self) raises:
        """Puts the key column back beside the state columns.

        The `_push` route keeps the keys in one piece per chunk that introduced
        a group, and the state in accumulator columns of its own that are longer
        than the group count and hold an identity where a group was never
        reached. Everything downstream wants the `state` layout, which is the
        keys and then the state slots at exactly the group count, so this is
        where the pieces are stacked, the accumulators are cut down and the
        identities become the nulls they stand for.

        Raises:
            If the key pieces cannot be stacked.
        """
        if self._map.groups == 0:
            return
        var out = List[AnyArray](capacity=1 + len(self._values))
        out.append(self._map.take_keys())
        for s in range(len(self._values)):
            out.append(
                settle_any(
                    AnyArray(copy=self._values[s]),
                    self._produce[s],
                    self._map.groups,
                )
            )
        self._values = List[AnyArray]()
        self._room = 0
        self.state = out^

    def _demote(mut self) raises:
        """Gives up the persistent map and goes back to the stacking merge.

        Called when a chunk turns up with a null key, which is the one thing
        about the data that decides the route and that no amount of looking at
        the schema will tell you in advance. Whatever the map has built is a
        valid running table, so handing it over and carrying on costs the
        query nothing beyond the route it loses.

        Raises:
            If the key pieces cannot be stacked.
        """
        self._fast = False
        self._gather()
        self._map = LastingKeys()

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
                    trusted=True,
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
        if self._fast:
            self._fast = False
            self._gather()
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


struct Reduce(Movable):
    """Reduces every row that goes past to one row, a chunk at a time.

    This is `DataFrame.agg` as an operator: a sum, a mean, a minimum, a maximum
    or a count over the whole input, with no key to group by. It is a breaker,
    because the answer is not known until the last row has been seen, but it is
    the cheapest breaker there is: what it holds between chunks is one row per
    state slot, whatever the input was.

    It is a separate node rather than a `Group` with an empty key list for the
    reason `agg` is a separate method from `group_by`. A group by hashes every
    row to find out which group it belongs to, and there is nothing to find out
    here, so the reductions read the column straight through and the hashing
    never happens. On ten million rows that is the difference between eighty
    five milliseconds and five, and the whole point of putting this in a
    pipeline is what happens to a query that ends in one.

    ## What that is for

    A join followed by a reduction is the shape every one of the db-benchmark
    join queries has, and run as two whole frame calls it writes the join's
    output to memory and then reads it back. On ten million rows and two float
    columns that is a hundred and sixty megabytes written and a hundred and
    sixty read, for an answer that is three numbers. Run as a pipeline the join
    hands the reduction a chunk, the reduction folds it into three running
    values, and the chunk is dropped while it is still in cache. The bytes never
    reach memory. That is the difference this node exists to make, and it is why
    it is worth having before the streaming join rather than after.

    ## How the merge works

    The same way `Group`'s does, and for the same reason it needs no kernel of
    its own. A chunk's answer and the running answer are both one row, so
    combining them is a reduction over a column of two rows, run with the kind
    that combines partials: the kind itself for a sum, a minimum or a maximum,
    and a sum for the two counts, since merging counts means adding them rather
    than counting them again. A mean is a sum and a count in two slots and the
    division happens once, at the end, because a mean of means is only the mean
    when every chunk is the same size.

    `_folds` is the list of reductions that can be done this way, and the ones
    that cannot are not a gap. A median of medians is not a median and no state
    short of the values themselves would make it one, so asking for one here is
    an error at plan time and the answer is `Materialize`.

    Floating point is the one place the answer can differ from calling `agg` on
    the whole frame. Adding a column in chunks and then adding the chunk sums is
    a different order of additions from adding it in one pass, and floating
    point addition is not associative, so a sum of floats can differ in the last
    bits. Every other kind here is exact.

    An input that hands over no chunks at all produces no rows rather than a row
    of nothing, which is what `Group` does with the same input and is the thing
    a pipeline can say. `agg` on a frame of no rows answers one row, and a
    pipeline that has to match it can put a `Materialize` here instead.
    """

    var aggs: List[GroupAgg]
    """What to compute, one output column each. The `column` field is a position
    in the input and the key list a `Group` would carry is not here."""

    var input: Schema
    """The schema of the chunks coming in, filled in by `bind`."""

    var output: Schema
    """The schema of the one row going out, worked out by `bind`."""

    var state: List[AnyArray]
    """The running answer: one array of exactly one row per state slot."""

    var _source: List[Int]
    """Per state slot, the input column it reduces."""

    var _produce: List[AggKind]
    """Per state slot, the reduction run over a chunk."""

    var _merge: List[AggKind]
    """Per state slot, the reduction that combines two partial answers."""

    var _at: List[Int]
    """Per aggregate, the state slot it starts at. A mean owns two."""

    var started: Bool
    """Whether a chunk has arrived."""

    var ran: Bool
    """Whether `finish` has handed the answer back."""

    def __init__(out self, var aggs: List[GroupAgg]):
        """Constructs a whole input reduction.

        Args:
            aggs: What to compute. Consumed.
        """
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

    def bind(mut self, var input: Schema) raises -> Schema:
        """Checks the aggregates and reports the output schema.

        Everything that can be wrong with a reduction that does not depend on
        the data is wrong here, and none of it needs a row to detect.

        Args:
            input: The schema of the chunks that will arrive. Consumed.

        Returns:
            One column per aggregate, in the order they were given.

        Raises:
            If no aggregates were given, if a position is outside the schema, if
            a reduction does not fold, if a reduction has no meaning on its
            column's type, or if two output columns would have the same name.
        """
        self.input = input^
        if len(self.aggs) == 0:
            raise Error("reduce: at least one aggregate is required")

        var fields = List[Field](capacity=len(self.aggs))
        for a in range(len(self.aggs)):
            var at = self.aggs[a].column
            var kind = self.aggs[a].kind
            if at < 0 or at >= len(self.input):
                raise Error(
                    "reduce: column "
                    + String(at)
                    + " is outside a schema of "
                    + String(len(self.input))
                    + " columns"
                )
            if not _folds(kind):
                raise Error(
                    "reduce: "
                    + String(kind)
                    + " cannot be computed a chunk at a time"
                )
            var source = self.input[at].dtype
            if source.is_variable_width() and (
                kind == AggKind.SUM or kind == AggKind.MEAN
            ):
                raise Error(
                    "reduce: " + String(kind) + " is not defined on text"
                )
            var name = self.aggs[a].name
            for f in range(len(fields)):
                if fields[f].name == name:
                    raise Error(
                        "reduce: two output columns would both be called "
                        + name
                    )
            fields.append(Field(name, _agg_type(kind, source)))

            self._at.append(len(self._source))
            if kind == AggKind.MEAN:
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
        """Reports whether the answer has been handed back.

        Returns:
            NEED_MORE_INPUT until `finish` has run, and FINISHED after it.
        """
        if not self.ran:
            return NodeStatus.NEED_MORE_INPUT
        return NodeStatus.FINISHED

    def partial(self, var chunk: Chunk) raises -> Optional[Chunk]:
        """Reduces one chunk on its own, without touching the running answer.

        The first half of `process`, split off because it is the expensive half
        and it is the half that does not need the node. It reads the node and
        writes nothing to it, so a batch of chunks can be reduced to a batch of
        one row partials on every core at once, and the merging that follows is
        a pass over one row per chunk. That matters because a chunk is a hundred
        and twenty eight thousand rows and a morsel is the same, so a chunk
        reduced on its own takes the serial route inside the kernel and there is
        no other parallelism in a fold to have.

        The row is in `_source` order and not the output's, and a mean is still
        two columns at this point. `absorb` is what reads it and it is written
        for that.

        Args:
            chunk: The chunk. Consumed.

        Returns:
            One row of partial answers, or None for a chunk with no rows.

        Raises:
            If the chunk is not as wide as the input schema, or if a reduction
            fails on the chunk's data.
        """
        if chunk.width() != len(self.input):
            raise Error(
                "reduce: chunk has "
                + String(chunk.width())
                + " columns and the input schema has "
                + String(len(self.input))
            )
        if len(chunk) == 0:
            return None
        var columns = chunk^.into_columns()
        var made = List[AnyArray](capacity=len(self._source))
        for s in range(len(self._source)):
            made.append(reduce_any(columns[self._source[s]], self._produce[s]))
        return Chunk(made^)

    def absorb(mut self, var partial: Chunk) raises:
        """Merges one chunk's partial answers into the running row.

        Args:
            partial: A row from `partial`, in `_source` order. Consumed.

        Raises:
            If the row is not the width `partial` produces, or if merging
            raises.
        """
        if partial.width() != len(self._source):
            raise Error(
                "reduce: a partial row has "
                + String(partial.width())
                + " columns and this reduction produces "
                + String(len(self._source))
            )
        var made = partial^.into_columns()
        if not self.started:
            self.started = True
            self.state = made^
            return
        for s in range(len(self._source)):
            var pair = concat_two_any(self.state[s], made[s])
            self.state[s] = reduce_any(pair, self._merge[s])

    def process(mut self, var chunk: Chunk) raises -> Optional[Chunk]:
        """Folds the chunk into the running answer.

        Args:
            chunk: The chunk. Consumed.

        Returns:
            None, always. A breaker has nothing to say until it has seen
            everything.

        Raises:
            If the chunk is not as wide as the input schema, or if a reduction
            fails on the chunk's data.
        """
        var made = self.partial(chunk^)
        if made:
            self.absorb(made.take())
        return None

    def finish(mut self) raises -> Optional[Chunk]:
        """Hands the one row answer back.

        Returns:
            One chunk of exactly one row the first time, and None after that or
            if no chunk with rows in it ever arrived.

        Raises:
            If a mean cannot be computed from its sum and its count.
        """
        if self.ran:
            return None
        self.ran = True
        if not self.started:
            return None

        var out = List[AnyArray](capacity=len(self.aggs))
        for a in range(len(self.aggs)):
            var at = self._at[a]
            if self.aggs[a].kind == AggKind.MEAN:
                out.append(_mean_of(self.state[at], self.state[at + 1]))
            else:
                out.append(AnyArray(copy=self.state[at]))
        self.state = List[AnyArray]()
        return Chunk(out^)


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
    Filter, Project, Compute, Cast, Join, Limit, Group, Reduce, Materialize
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
    if node.isa[Reduce]():
        return node[Reduce].bind(input^)
    if node.isa[Compute]():
        return node[Compute].bind(input^)
    if node.isa[Cast]():
        return node[Cast].bind(input^)
    if node.isa[Join]():
        return node[Join].bind(input^)
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
    if node.isa[Reduce]():
        return node[Reduce].update_state()
    return NodeStatus.NEED_MORE_INPUT


def node_is_row_local(node: Node) -> Bool:
    """Reports whether a node's output row depends only on its own input row.

    The four elementwise operators say yes, and so does `Join`, whose output row
    depends on its own input row and on a table that was finished before the
    first chunk arrived. What that buys is that the node reads itself and never
    writes itself, so one of them can be handed to every core at once without a
    copy per worker and without a lock. `Limit` counts rows, `Group` holds a
    table it is still filling, `Reduce` holds a running answer and `Materialize`
    holds the input, so all four say no.

    Args:
        node: The node.

    Returns:
        True for `Filter`, `Project`, `Compute`, `Cast` and `Join`.
    """
    return (
        node.isa[Filter]()
        or node.isa[Project]()
        or node.isa[Compute]()
        or node.isa[Cast]()
        or node.isa[Join]()
    )


def node_ends_early(node: Node) -> Bool:
    """Reports whether a node can say FINISHED before its input runs out.

    Only `Limit` can. `Group` and `Materialize` say FINISHED too, but not until
    `finish` has handed back everything they held, which is after the source is
    empty. The distinction matters to the driver: a pipeline that can stop early
    must be fed one chunk at a time, because reading ahead on behalf of thirty
    two cores is reading rows that a limit was about to make unnecessary.

    Args:
        node: The node.

    Returns:
        True for `Limit`.
    """
    return node.isa[Limit]()


def node_computes_per_row(node: Node) -> Bool:
    """Reports whether a node works out a value for every row it is given.

    `Filter` evaluates a predicate and `Compute` evaluates an expression, so
    both do arithmetic once per row and both get faster on more cores. `Join`
    hashes a key and gathers a row per output row, which is more work per row
    than either, and its table is read only once `bind` has run. `Project`
    only rebuilds a chunk out of columns it already has and `Cast` walks a
    column through the allocator, so neither has much for a second core to do
    and both are held up by memory rather than by arithmetic. Measured on the
    i9-13900K, a line with a compute and a filter in it ran three times faster
    spread over the cores, while a project on its own ran no faster at all and
    paid for the tasks on top.

    Args:
        node: The node.

    Returns:
        True for `Filter`, `Compute` and `Join`.
    """
    return node.isa[Filter]() or node.isa[Compute]() or node.isa[Join]()


def node_is_breaker(node: Node) -> Bool:
    """Reports whether a node has to see all its input before it emits.

    Args:
        node: The node.

    Returns:
        True for a breaker, which is where a pipeline is cut.
    """
    return node.isa[Materialize]() or node.isa[Group]() or node.isa[Reduce]()


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
    if node.isa[Join]():
        return node[Join].process(chunk^)
    if node.isa[Limit]():
        return node[Limit].process(chunk^)
    if node.isa[Group]():
        return node[Group].process(chunk^)
    if node.isa[Reduce]():
        return node[Reduce].process(chunk^)
    return node[Materialize].process(chunk^)


def node_apply(node: Node, var chunk: Chunk) raises -> Optional[Chunk]:
    """Pushes one chunk through a row local node without mutating it.

    The same call as `node_process` for the four elementwise operators, except
    that the node is read rather than borrowed mutably, which is what lets the
    same node be used by several workers at once. Anything else raises rather
    than being run, because a node that carries state between chunks run this
    way would be racing itself.

    Args:
        node: The node. Read only.
        chunk: The chunk. Consumed.

    Returns:
        What the node emits, or None if it emits nothing for this chunk.

    Raises:
        If the node is not row local, or if it cannot process the chunk.
    """
    if node.isa[Filter]():
        return node[Filter].process(chunk^)
    if node.isa[Project]():
        return node[Project].process(chunk^)
    if node.isa[Compute]():
        return node[Compute].process(chunk^)
    if node.isa[Cast]():
        return node[Cast].process(chunk^)
    if node.isa[Join]():
        # False, because this is the entry point several workers share and a
        # join's own kernels would each hand themselves out to workers again.
        return node[Join].process(chunk^, False)
    raise Error("apply: this node carries state between chunks")


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
    if node.isa[Reduce]():
        return node[Reduce].finish()
    return None
