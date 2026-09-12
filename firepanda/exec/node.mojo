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

`Expand` is the odd one beside them. It reads one column and writes each row as
many times as that column says, which is a filter whose mask is a number rather
than a yes or a no, and it is the only operator here that can turn a number into
rows. `EXCEPT ALL` and `INTERSECT ALL` are what needed one.

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

from firepanda.array.any import AnyArray, borrow_columns, empty_any
from firepanda.array.array import Array
from firepanda.array.chunked import ChunkedArray
from firepanda.array.value import Value
from firepanda.bitmap.bitmap import Bitmap
from firepanda.dtype.lists import ALL
from firepanda.dtype.logical import LogicalType, TypeKind
from firepanda.dtype.schema import Field, Schema
from firepanda.frame.frame import DataFrame
from firepanda.hash.grouping import group_ordinals
from firepanda.hash.lasting import LastingKeys
from firepanda.join.keys import (
    BuildSide,
    build_side,
    build_side_strings,
    probe_side,
    probe_side_strings,
)
from firepanda.join.pairs import (
    JoinKind,
    ProbeTable,
    bucket_side,
    mark_probe,
    pair_probe,
)
from firepanda.kernel.accum import accumulator
from firepanda.kernel.binary import (
    BinaryOp,
    binary_any,
    binary_type,
    binary_value_any,
    filled_block,
    resolve_constant,
)
from firepanda.kernel.cast import cast_any
from firepanda.kernel.concat import concat_two_any
from firepanda.kernel.group import AggKind, agg_type, aggregate_group_any
from firepanda.kernel.logic import LogicOp, logic_any, logic_type
from firepanda.kernel.pick import pick_any
from firepanda.kernel.reduce import reduce_any
from firepanda.kernel.running import (
    accumulate_any,
    settle_any,
    state_capacity,
    widen_any,
)
from firepanda.kernel.select import filter_any, take_any
from firepanda.kernel.sort import argsort_any_into, identity_permutation

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
    position.

    ## Filtering only what is wanted

    Filtering a column means writing a new one, so a column nobody downstream
    reads is a whole array written for nothing. The mask itself is always such a
    column, since filtering by it produces a column that is all true, and so is
    every intermediate the expression that built it left behind.

    So the node can be told which positions to write, in the order to write
    them, which makes it a filter and a projection in one pass. `Filter(on)`
    keeps everything, which is what a caller assembling a pipeline by hand
    wants. `Filter(on, keep)` keeps those positions and writes nothing else,
    which is what lowering asks for, and it is why a predicate of four
    conditions over a wide chunk no longer leaves four dead masks behind it.
    """

    var on: Int
    """The position of the boolean column to filter by."""

    var keep: List[Int]
    """The input positions to write, in output order. Empty when narrows is
    False."""

    var narrows: Bool
    """Whether keep is what comes out, rather than the whole chunk.

    A flag rather than an empty keep meaning everything, because a filter that
    is asked for no columns at all is a row count and is a thing a caller may
    reasonably want.
    """

    def __init__(out self, on: Int):
        """Constructs a filter that keeps every column of its input.

        Args:
            on: The position of the boolean column.
        """
        self.on = on
        self.keep = List[Int]()
        self.narrows = False

    def __init__(out self, on: Int, var keep: List[Int]):
        """Constructs a filter that writes only some columns.

        Args:
            on: The position of the boolean column.
            keep: The input positions to write, in output order. May repeat,
                and need not include the mask.
        """
        self.on = on
        self.keep = keep^
        self.narrows = True

    def process(self, var chunk: Chunk) raises -> Optional[Chunk]:
        """Keeps the rows the mask is true on.

        Args:
            chunk: The chunk. Consumed.

        Returns:
            The surviving rows, or None if none survived, since a chunk of no
            rows is work for everything downstream and no information.

        Raises:
            If a position is out of range or the mask column is not boolean.
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
        if not self.narrows:
            var all = List[AnyArray](capacity=chunk.width())
            for i in range(chunk.width()):
                all.append(filter_any(chunk.columns[i], mask))
            var every = 0 if len(all) == 0 else len(all[0])
            if every == 0:
                return None
            return Chunk(all^, every)
        var kept = List[AnyArray](capacity=len(self.keep))
        for i in range(len(self.keep)):
            if self.keep[i] < 0 or self.keep[i] >= chunk.width():
                raise Error(
                    "filter: column "
                    + String(self.keep[i])
                    + " is outside a chunk of "
                    + String(chunk.width())
                    + " columns"
                )
            kept.append(filter_any(chunk.columns[self.keep[i]], mask))
        # A filter asked for no columns at all still knows how many rows
        # survived, and the only thing that knows what a null in the mask means
        # is the kernel, so the count comes from filtering the mask by itself
        # rather than from reading it here.
        var rows: Int
        if len(kept) > 0:
            rows = len(kept[0])
        else:
            rows = len(filter_any(chunk.columns[self.on], mask))
        if rows == 0:
            return None
        return Chunk(kept^, rows)


struct Expand(Movable):
    """Writes each row of a chunk as many times as a column of the chunk says.

    The opposite of a filter, and the same shape as one. A filter reads a
    boolean column and writes each row once or not at all, and this reads a
    whole number column and writes each row that many times. A filter is the
    case where the number is only ever zero or one, which is why the two take
    the same arguments and why this one also says which positions to write.

    It exists for `EXCEPT ALL` and `INTERSECT ALL`. Both of those count the
    copies of a row on each side and answer some number of copies that the two
    counts work out, and until there was an operator that could turn a number
    into that many rows, a group by could work the number out and had no way to
    say it. Nothing else here repeats a row: a join does, but only as many times
    as its table happens to hold, which is not a number a plan can name.

    A count of zero or less writes nothing, which is the rule rather than an
    edge case. `EXCEPT ALL` asks for the copies on the left minus the copies on
    the right, and that difference is negative whenever the right side has more,
    so clamping here is what saves the lowering from building a maximum against
    a constant for every set difference. A null count writes nothing too, for
    the same reason a filter drops a row its mask is null on.

    The gather is one call per column with one index list shared between them,
    so a row written five times is five reads of the same cache line rather than
    five passes over the chunk.
    """

    var counts: Int
    """The position of the whole number column saying how many copies."""

    var keep: List[Int]
    """The input positions to write, in output order."""

    def __init__(out self, counts: Int, var keep: List[Int]):
        """Constructs an expansion.

        Args:
            counts: The position of the count column, which must be `Int64`.
            keep: The input positions to write, in output order. May repeat,
                and need not include the count column.
        """
        self.counts = counts
        self.keep = keep^

    def bind(mut self, var input: Schema) raises -> Schema:
        """Reports the schema of the kept positions, in the order given.

        Args:
            input: The schema of the chunks coming in. Consumed.

        Returns:
            The schema that comes out.

        Raises:
            Error: If a position is outside the input.
        """
        if self.counts < 0 or self.counts >= len(input):
            raise Error(
                "expand: column "
                + String(self.counts)
                + " is outside a schema of "
                + String(len(input))
                + " columns"
            )
        return _narrow(self.keep, input, "expand")

    def process(self, var chunk: Chunk) raises -> Optional[Chunk]:
        """Writes each row as many times as the count column says.

        Args:
            chunk: The chunk. Consumed.

        Returns:
            The rows written out, or None if the counts asked for none, since a
            chunk of no rows is work for everything downstream and no
            information.

        Raises:
            Error: If a position is out of range or the count column is not a
                whole number of sixty four bits.
        """
        if self.counts < 0 or self.counts >= chunk.width():
            raise Error(
                "expand: column "
                + String(self.counts)
                + " is outside a chunk of "
                + String(chunk.width())
                + " columns"
            )
        ref how_many = chunk.columns[self.counts].as_typed_view[DType.int64]()
        var indices = List[Int]()
        for i in range(len(how_many)):
            if not how_many.is_valid(i):
                continue
            var copies = Int(how_many[i])
            for _ in range(copies):
                indices.append(i)
        if len(indices) == 0:
            return None
        var written = List[AnyArray](capacity=len(self.keep))
        for i in range(len(self.keep)):
            if self.keep[i] < 0 or self.keep[i] >= chunk.width():
                raise Error(
                    "expand: column "
                    + String(self.keep[i])
                    + " is outside a chunk of "
                    + String(chunk.width())
                    + " columns"
                )
            written.append(take_any(chunk.columns[self.keep[i]], indices))
        return Chunk(written^, len(indices))


struct Project(Movable):
    """Keeps some columns of the chunk, in an order the plan chose.

    A column named twice is copied once and moved once rather than copied twice,
    because the last use of a position can give the array up. Every other
    position is moved out, so the ordinary projection, which is a subset in some
    order, copies nothing at all. That matters more than it sounds: a projection
    that copied would put a full copy of every kept column into the cost of
    every chunk, and dropping columns early is one of the main things a plan
    does.

    It may also rename what it keeps, which costs nothing at all, because a
    chunk is arrays and the names live on the pipeline's schema. That is what
    `SELECT qty AS howmany` is, and what every aggregate with an alias is, since
    the aggregate writes its answer to a column of its own naming and the
    projection above it is the only thing that knows what the query called it.
    """

    var keep: List[Int]
    """The input positions to keep, in output order."""

    var names: List[String]
    """What to call them, or empty to keep the names they came with."""

    def __init__(out self, var keep: List[Int]):
        """Constructs a projection that keeps the names it is handed.

        Args:
            keep: The input positions, in output order. May repeat.
        """
        self.keep = keep^
        self.names = List[String]()

    def __init__(out self, var keep: List[Int], var names: List[String]):
        """Constructs a projection that renames what it keeps.

        Args:
            keep: The input positions, in output order. May repeat.
            names: One name per kept position, or empty for no renaming.
        """
        self.keep = keep^
        self.names = names^

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
    """Drops the first rows, passes the next n through, stops the pipeline.

    Stopping is the interesting half. Once this has emitted the rows it was
    asked for it reports FINISHED, and a pipeline is a line, so nothing upstream
    can produce a row that reaches the sink. The driver stops reading the source
    the moment it sees that. A `head` over a file of ten million rows therefore
    reads one chunk, not the file.

    The skip is counted in rows and not in chunks, so an offset that lands in
    the middle of a chunk cuts that chunk and lets the rest of it through. There
    is nothing cheaper available here: the rows before the offset still have to
    be produced to be counted, and only a source that knows its own row count
    can do better than reading them.

    A limit may also have no bound, which is what `LIMIT NULL OFFSET 20` and a
    bare `OFFSET` mean. Then it never says FINISHED and the only thing it does
    is the skip. That is why the bound is a flag beside the count rather than a
    count of minus one: `LIMIT 0` keeps nothing and has to finish at once, and
    the two would otherwise be the same number.
    """

    var n: Int
    """The number of rows to let through, when there is a bound."""

    var bounded: Bool
    """Whether `n` means anything. False is a limit that only skips."""

    var skip: Int
    """How many rows to drop before the first one that counts."""

    var emitted: Int
    """How many have gone through so far."""

    var dropped: Int
    """How many of the skip have been taken off the front so far."""

    def __init__(out self, n: Int, skip: Int = 0):
        """Constructs a limit.

        Args:
            n: The number of rows to keep. Zero finishes immediately, and a
                negative number is a limit with no bound.
            skip: How many rows to drop first.
        """
        self.n = n if n > 0 else 0
        self.bounded = n >= 0
        self.skip = skip if skip > 0 else 0
        self.emitted = 0
        self.dropped = 0

    def update_state(self) -> NodeStatus:
        """Reports whether the limit has been reached.

        Returns:
            FINISHED once n rows have gone through, NEED_MORE_INPUT before that
            and always when there is no bound.
        """
        if self.bounded and self.emitted >= self.n:
            return NodeStatus.FINISHED
        return NodeStatus.NEED_MORE_INPUT

    def process(mut self, var chunk: Chunk) raises -> Optional[Chunk]:
        """Drops rows until the offset is passed, then passes them through.

        Args:
            chunk: The chunk. Consumed.

        Returns:
            The whole chunk when none of it is skipped and there is room for it,
            a piece of it when either end is cut, and None when all of it falls
            before the offset or after the limit.

        Raises:
            If a column cannot be sliced.
        """
        if self.bounded and self.emitted >= self.n:
            return None
        var rows = len(chunk)
        var start = 0
        if self.dropped < self.skip:
            start = min(self.skip - self.dropped, rows)
            self.dropped += start
            if start == rows:
                return None
        var take = rows - start
        if self.bounded and take > self.n - self.emitted:
            take = self.n - self.emitted
        self.emitted += take
        if start == 0 and take == rows:
            return chunk^
        var cut = List[AnyArray](capacity=chunk.width())
        for i in range(chunk.width()):
            cut.append(chunk.columns[i].slice(start, start + take))
        return Chunk(cut^, take)


struct Sort(Movable):
    """Holds every row, orders them on a set of keys, emits chunks again.

    A breaker, and the one kind of breaker that cannot be anything else. A group
    by holds one row per group because two rows with the same key are one answer.
    A sort has no such reduction: the last row to arrive can belong at the front,
    so the first row of the answer is not known until the last row of the input
    has been seen, and there is nothing shorter than the input to hold in the
    meantime.

    What it does instead of getting smaller is get cheaper per row. The sort is
    the least significant digit pass in `firepanda/kernel/sort.mojo`, run once
    per key from the last key to the first, each pass refining the permutation
    the one after it produced rather than starting again. Every pass is stable,
    so the earlier key stays dominant, and none of them compares a tuple. Then
    the permutation is applied to every column once, which is the only place the
    rows move.

    The keys are column positions rather than names, because a chunk has no
    schema and the pipeline hands one in at `bind` time. The nulls flag is
    `nulls_first` rather than the plan's `nulls_last` for the same reason the
    kernel spells it that way, and the lowering flips it.

    Chunk boundaries survive. The input's row counts are recorded on the way in
    and the output is cut at the same places, so a sort in the middle of a
    pipeline does not turn ten million rows into one chunk for whatever is above
    it. The rows in each are of course not the rows that arrived in it.

    This does not honour a limit above it. A sort that only has to get the first
    n rows right is a different operator with a heap in it rather than a
    permutation, and the plan writes the bound down on the node for one to read
    later. Ignoring it is slow rather than wrong, because the limit is still
    sitting above the sort and still doing the cutting.
    """

    var keys: List[Int]
    """The key columns, as input positions, most significant first."""

    var descending: List[Bool]
    """One flag per key."""

    var nulls_first: List[Bool]
    """One flag per key."""

    var input: Schema
    """The schema of the chunks coming in, filled in by `Pipeline.add`."""

    var held: List[ChunkedArray]
    """The chunks seen so far, one column per position."""

    var sizes: List[Int]
    """How many rows each chunk that arrived had, in order."""

    var output: List[AnyArray]
    """The result, in chunks, in reverse order so `finish` can pop."""

    var ran: Bool
    """Whether the rows have been ordered."""

    var width: Int
    """The number of columns, known once `bind` has run."""

    def __init__(
        out self,
        var keys: List[Int],
        var descending: List[Bool],
        var nulls_first: List[Bool],
    ) raises:
        """Constructs a sort.

        Args:
            keys: The key columns, as input positions, most significant first.
                Consumed.
            descending: Whether each key orders downwards. Consumed.
            nulls_first: Whether each key's missing values go at the front.
                Consumed.

        Raises:
            If there are no keys, or if the flag lists are a different length
            from the keys.
        """
        if len(keys) == 0:
            raise Error("sort: a sort with no key does not order anything")
        if len(descending) != len(keys) or len(nulls_first) != len(keys):
            raise Error(
                String(
                    "sort: ",
                    len(keys),
                    " keys, ",
                    len(descending),
                    " directions and ",
                    len(nulls_first),
                    " null placements",
                )
            )
        self.keys = keys^
        self.descending = descending^
        self.nulls_first = nulls_first^
        self.input = Schema()
        self.held = List[ChunkedArray]()
        self.sizes = List[Int]()
        self.output = List[AnyArray]()
        self.ran = False
        self.width = 0

    def bind(mut self, var input: Schema) raises -> Schema:
        """Records the input schema, which is also the output one.

        Args:
            input: The schema of the chunks that will arrive. Consumed.

        Returns:
            The same schema. A sort moves rows and leaves columns alone.

        Raises:
            If a key is outside the input.
        """
        for i in range(len(self.keys)):
            if self.keys[i] < 0 or self.keys[i] >= len(input):
                raise Error(
                    String(
                        "sort: key column ",
                        self.keys[i],
                        " is outside a schema of ",
                        len(input),
                        " columns",
                    )
                )
        self.input = input^
        self.width = len(self.input)
        for i in range(self.width):
            self.held.append(ChunkedArray(self.input[i].dtype))
        return Schema(copy=self.input)

    def update_state(self) -> NodeStatus:
        """Reports whether the rows have been ordered and handed back.

        Returns:
            NEED_MORE_INPUT until `finish` has sorted, then HAVE_OUTPUT while
            chunks remain and FINISHED after that.
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
            None, always. The first row of the answer is not known until the
            last row of the input has arrived.

        Raises:
            If the chunk's width does not match the input schema.
        """
        if chunk.width() != self.width:
            raise Error(
                String(
                    "sort: chunk has ",
                    chunk.width(),
                    " columns and the input schema has ",
                    self.width,
                )
            )
        self.sizes.append(len(chunk))
        var backwards = chunk^.into_columns()
        var forwards = List[AnyArray](capacity=len(backwards))
        while len(backwards) > 0:
            forwards.append(backwards.pop())
        for i in range(self.width):
            self.held[i].append(forwards.pop())
        return None

    def finish(mut self) raises -> Optional[Chunk]:
        """Orders the rows the first time, then hands them back in chunks.

        Returns:
            One chunk of the ordered result per call, in order, and None when
            there are none left.

        Raises:
            If a key column's dtype is not one firepanda can sort.
        """
        if not self.ran:
            self.ran = True
            self._order()
        if len(self.output) == 0:
            return None
        var row = List[AnyArray](capacity=self.width)
        for _ in range(self.width):
            row.append(self.output.pop())
        return Chunk(row^)

    def _order(mut self) raises:
        """Sorts what was held and lays it out as chunks ready to be popped.

        The columns are flattened once each, because a permutation reaches any
        row from any chunk and a gather that had to find which chunk a row was
        in per row would pay for the chunking on every row rather than once per
        column. Then the permutation is cut at the boundaries the input had and
        each piece gathers its own chunk, so the flattening is the one copy.

        Raises:
            If a key column's dtype is not one firepanda can sort.
        """
        var backwards = List[AnyArray](capacity=self.width)
        while len(self.held) > 0:
            backwards.append(self.held.pop().combine())
        var flat = List[AnyArray](capacity=self.width)
        while len(backwards) > 0:
            flat.append(backwards.pop())
        if self.width == 0:
            return

        var order = identity_permutation(len(flat[0]))
        for i in range(len(self.keys) - 1, -1, -1):
            argsort_any_into(
                flat[self.keys[i]],
                order,
                self.descending[i],
                self.nulls_first[i],
            )
        var rows = List[Int](capacity=len(order))
        for i in range(len(order)):
            rows.append(Int(order[i]))

        # Backwards, and columns backwards inside a chunk, because `finish`
        # takes what it hands over off the end of this list.
        var at = len(rows)
        for c in range(len(self.sizes) - 1, -1, -1):
            var start = at - self.sizes[c]
            var piece = List[Int](capacity=self.sizes[c])
            for i in range(start, at):
                piece.append(rows[i])
            for i in range(self.width - 1, -1, -1):
                self.output.append(take_any(flat[i], piece))
            at = start


struct Window(Movable):
    """Holds every row, reduces each partition, writes the answer on every row.

    A breaker for the same reason a sort is one. `sum(x) OVER (PARTITION BY k)`
    on the first row is a sum over rows that have not arrived yet, so there is
    nothing to emit until the input has run out, and unlike a group by there is
    nothing shorter than the input to hold in the meantime, because every row
    that went in comes back out.

    What comes out is wider than what went in. The input's columns are handed
    through at the positions they had and the windows are appended after them,
    which is what `SELECT x, sum(x) OVER ()` asks for and what the logical node
    above this says it produces. It is the only operator here that adds a column
    without computing it from the row it is on.

    One partitioning per operator, and the partition is the whole frame when
    there are no keys. Two windows over different keys are two of these stacked,
    which is also how DuckDB runs them, and the plan is what decides the order
    they go in.

    The frame is the partition and nothing else. There is no `ROWS BETWEEN`
    here and no ordering inside the window, so every row of a partition gets the
    same value, which is the whole of `OVER (PARTITION BY ...)` with no frame
    clause and none of `OVER (ORDER BY ...)`. An ordered window is a running
    fold rather than one value broadcast, and it is a different loop rather than
    an argument to this one, so lowering refuses it rather than answering it
    wrongly.

    Chunk boundaries survive, as they do through a sort. The input's row counts
    are recorded on the way in and the output is cut at the same places, so a
    window in the middle of a pipeline does not hand ten million rows up as one
    chunk. Unlike a sort, each output chunk holds the rows that arrived in it.
    """

    var keys: List[Int]
    """The partition columns, as input positions. Empty is the whole frame."""

    var sources: List[Int]
    """The column each window reduces, as input positions."""

    var kinds: List[AggKind]
    """Which reduction each window is."""

    var names: List[String]
    """The name each appended column gets."""

    var input: Schema
    """The schema of the chunks coming in, filled in by `Pipeline.add`."""

    var held: List[ChunkedArray]
    """The chunks seen so far, one column per position."""

    var sizes: List[Int]
    """How many rows each chunk that arrived had, in order."""

    var output: List[AnyArray]
    """The result, in chunks, in reverse order so `finish` can pop."""

    var ran: Bool
    """Whether the windows have been computed."""

    var width: Int
    """The number of input columns, known once `bind` has run."""

    def __init__(
        out self,
        var keys: List[Int],
        var sources: List[Int],
        var kinds: List[AggKind],
        var names: List[String],
    ) raises:
        """Constructs a window.

        Args:
            keys: The partition columns, as input positions, or none for the
                whole frame. Consumed.
            sources: The column each window reduces. Consumed.
            kinds: The reduction each window runs. Consumed.
            names: The name each appended column gets. Consumed.

        Raises:
            If there is no window to compute, or if the three lists that
            describe them are a different length from each other.
        """
        if len(sources) == 0:
            raise Error(
                "window: a window operator that computes no window is the"
                " operator below it"
            )
        if len(kinds) != len(sources) or len(names) != len(sources):
            raise Error(
                String(
                    "window: ",
                    len(sources),
                    " columns, ",
                    len(kinds),
                    " reductions and ",
                    len(names),
                    " names",
                )
            )
        self.keys = keys^
        self.sources = sources^
        self.kinds = kinds^
        self.names = names^
        self.input = Schema()
        self.held = List[ChunkedArray]()
        self.sizes = List[Int]()
        self.output = List[AnyArray]()
        self.ran = False
        self.width = 0

    def bind(mut self, var input: Schema) raises -> Schema:
        """Records the input schema and adds the columns the windows produce.

        Args:
            input: The schema of the chunks that will arrive. Consumed.

        Returns:
            The input's columns as they were, then one column per window.

        Raises:
            If a position is outside the input, if a key is repeated, or if a
            reduction has no meaning on the column it was given.
        """
        for k in range(len(self.keys)):
            var at = self.keys[k]
            if at < 0 or at >= len(input):
                raise Error(
                    String(
                        "window: partition column ",
                        at,
                        " is outside a schema of ",
                        len(input),
                        " columns",
                    )
                )
            for j in range(k):
                if self.keys[j] == at:
                    raise Error(
                        String(
                            "window: partition column ", at, " was given twice"
                        )
                    )

        var fields = List[Field](capacity=len(input) + len(self.sources))
        for i in range(len(input)):
            fields.append(input[i].copy())
        for a in range(len(self.sources)):
            var at = self.sources[a]
            if at < 0 or at >= len(input):
                raise Error(
                    String(
                        "window: column ",
                        at,
                        " is outside a schema of ",
                        len(input),
                        " columns",
                    )
                )
            var kind = self.kinds[a]
            if kind.reads_two_columns():
                raise Error(
                    String(
                        "window: ",
                        kind,
                        " reads two columns and a window here reads one",
                    )
                )
            var source = input[at].dtype
            if source.is_variable_width() and (
                kind == AggKind.SUM or kind == AggKind.MEAN
            ):
                raise Error(String("window: ", kind, " is not defined on text"))
            fields.append(Field(self.names[a], agg_type(kind, source)))

        self.input = input^
        self.width = len(self.input)
        for i in range(self.width):
            self.held.append(ChunkedArray(self.input[i].dtype))
        return Schema(fields^)

    def update_state(self) -> NodeStatus:
        """Reports whether the windows have been computed and handed back.

        Returns:
            NEED_MORE_INPUT until `finish` has run them, then HAVE_OUTPUT while
            chunks remain and FINISHED after that.
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
            None, always. The first row's answer reads rows that have not
            arrived.

        Raises:
            If the chunk's width does not match the input schema.
        """
        if chunk.width() != self.width:
            raise Error(
                String(
                    "window: chunk has ",
                    chunk.width(),
                    " columns and the input schema has ",
                    self.width,
                )
            )
        self.sizes.append(len(chunk))
        var backwards = chunk^.into_columns()
        var forwards = List[AnyArray](capacity=len(backwards))
        while len(backwards) > 0:
            forwards.append(backwards.pop())
        for i in range(self.width):
            self.held[i].append(forwards.pop())
        return None

    def finish(mut self) raises -> Optional[Chunk]:
        """Runs the windows the first time, then hands the rows back in chunks.

        Returns:
            One chunk of the result per call, in the order the chunks arrived,
            and None when there are none left.

        Raises:
            If a partition key's dtype has no physical layout, or if a reduction
            fails on the column it was given.
        """
        if not self.ran:
            self.ran = True
            self._run()
        if len(self.output) == 0:
            return None
        var wide = self.width + len(self.names)
        var row = List[AnyArray](capacity=wide)
        for _ in range(wide):
            row.append(self.output.pop())
        return Chunk(row^)

    def _run(mut self) raises:
        """Reduces each partition and lays the rows out as chunks to be popped.

        The columns are flattened once each, because a partition reaches any row
        from any chunk and the grouping is one pass over the whole column. Then
        each window is reduced to one value per partition and read back out
        through the same ordinals that produced it, which is a gather per window
        rather than a scan per row.

        The whole frame case gets a column of zero ordinals rather than a branch
        of its own. It is one pass over four bytes a row to avoid a second way
        of writing the same three lines, and a whole frame window on a frame big
        enough for that to matter is rare enough to leave until it is measured.

        Raises:
            If a partition key's dtype has no physical layout, or if a reduction
            fails on the column it was given.
        """
        var backwards = List[AnyArray](capacity=self.width)
        while len(self.held) > 0:
            backwards.append(self.held.pop().combine())
        var flat = List[AnyArray](capacity=self.width)
        while len(backwards) > 0:
            flat.append(backwards.pop())
        if self.width == 0:
            return
        var rows = len(flat[0])
        if rows == 0:
            return

        var codes = Array[DType.uint32](rows)
        var groups = 1
        if len(self.keys) > 0:
            var refs = borrow_columns(flat)
            var local = group_ordinals(refs, self.keys, rows)
            groups = local.groups
            codes = local^.into_codes()

        var made = List[AnyArray](capacity=len(self.sources))
        for a in range(len(self.sources)):
            made.append(
                aggregate_group_any(
                    flat[self.sources[a]],
                    self.kinds[a],
                    codes,
                    groups,
                    trusted=True,
                )
            )

        # Backwards, and columns backwards inside a chunk, because `finish`
        # takes what it hands over off the end of this list.
        var at = rows
        for c in range(len(self.sizes) - 1, -1, -1):
            var start = at - self.sizes[c]
            var pick = List[Int](capacity=self.sizes[c])
            for i in range(start, at):
                pick.append(Int(codes[i]))
            for i in range(len(made) - 1, -1, -1):
                self.output.append(take_any(made[i], pick))
            for i in range(self.width - 1, -1, -1):
                self.output.append(flat[i].slice(start, at))
            at = start


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
            # A Python scalar has no width of its own and takes one from the
            # column, so it has to be resolved here for the same reason
            # `binary_value_any` resolves it: this declares what that call will
            # produce, and a rule applied in one and not the other is a plan
            # whose stated dtype is not the dtype of the data it describes. It
            # is also where a constant too large for the column is caught, at
            # plan time, before a row moves.
            var k = resolve_constant(a, self.constant.value(), self.op).type
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


struct Connective(Movable):
    """Appends a column that is two boolean columns joined by and, or or not.

    The same shape as `Compute` and a separate node for the same reason
    `logic.mojo` is a separate kernel: a connective decides a row where one of
    its operands is null and the other one settles it, and every operation
    `Compute` can do answers null there instead. Folding the three into
    `BinaryOp` would mean one of the two rules being applied by a flag on a node,
    which is the kind of thing that is right in the test that thought of it and
    wrong six months later.

    Negation reads one column and is here rather than in a node of its own, for
    the reason `LogicOp` holds all three: an expression that says `not` is the
    same shape of thing as one that says `or`, and splitting them would put two
    node types on one branch of the lowering with nothing downstream telling them
    apart.

    `a AND b` in a WHERE clause does not normally reach here. The optimizer
    splits a top level conjunction into a line of filters, which is both cheaper
    and better for pushdown, so what is left for this node is the conjunction
    nested inside something else, the disjunction, the negation, and every
    boolean expression in a select list.
    """

    var left: Int
    """The position of the left operand, or of the only one under a negation."""

    var right: Int
    """The position of the right operand. Ignored under a negation."""

    var op: LogicOp
    """The connective."""

    var name: String
    """The name the appended column gets in the output schema."""

    def __init__(out self, left: Int, right: Int, op: LogicOp, name: String):
        """Constructs a connective over two columns.

        Args:
            left: The position of the left operand.
            right: The position of the right operand.
            op: The connective, which must read two columns.
            name: The name of the appended column.
        """
        self.left = left
        self.right = right
        self.op = op
        self.name = name

    def __init__(out self, column: Int, name: String):
        """Constructs a negation, which reads one column.

        Args:
            column: The position of the operand.
            name: The name of the appended column.
        """
        self.left = column
        self.right = column
        self.op = LogicOp.NOT
        self.name = name

    def bind(mut self, var input: Schema) raises -> Schema:
        """Reports the input schema with the computed column appended.

        The answer is a bool whatever the operands are, so what this is really
        for is the other half: an operand that is not boolean is a query that
        means nothing, and it is caught here rather than on the first chunk.

        Args:
            input: The schema of the chunks that will arrive. Consumed.

        Returns:
            The input schema with one field on the end.

        Raises:
            If a position is outside the schema, or an operand is not boolean.
        """
        var out = input^
        if self.left < 0 or self.left >= len(out):
            raise Error(
                "connective: column "
                + String(self.left)
                + " is outside a schema of "
                + String(len(out))
                + " columns"
            )
        if self.right < 0 or self.right >= len(out):
            raise Error(
                "connective: column "
                + String(self.right)
                + " is outside a schema of "
                + String(len(out))
                + " columns"
            )
        # One interior reference at a time, as `Compute.bind` does, since two
        # into the same list cannot both be alive.
        var made = logic_type(self.op, out[self.left].dtype)
        if self.op.reads_two_columns():
            made = logic_type(self.op, out[self.right].dtype)
        out.append(Field(self.name, made))
        return out^

    def process(self, var chunk: Chunk) raises -> Optional[Chunk]:
        """Applies the connective and puts the column on the end of the chunk.

        Args:
            chunk: The chunk. Consumed.

        Returns:
            The chunk with one more column and the same number of rows.

        Raises:
            If a position is outside the chunk, or an operand is not boolean.
        """
        var width = chunk.width()
        if self.left < 0 or self.left >= width:
            raise Error(
                "connective: column "
                + String(self.left)
                + " is outside a chunk of "
                + String(width)
                + " columns"
            )
        if self.right < 0 or self.right >= width:
            raise Error(
                "connective: column "
                + String(self.right)
                + " is outside a chunk of "
                + String(width)
                + " columns"
            )
        var made: AnyArray
        if self.op.reads_two_columns():
            made = logic_any(
                chunk.columns[self.left], chunk.columns[self.right], self.op
            )
        else:
            made = logic_any(chunk.columns[self.left], self.op)
        var rows = len(chunk)
        var columns = chunk^.into_columns()
        columns.append(made^)
        return Chunk(columns^, rows)


struct Choose(Movable):
    """Appends a column taking each row from one of two columns, on a condition.

    This is SQL's `CASE WHEN c THEN a ELSE b END` and the frame API's `where`.
    The shape is `Compute`'s with one more position, and it is a node of its own
    for the reason `Connective` is: the rule for a null is not the one an
    operation follows. A null condition is not a null answer here, it takes the
    else side, because a row the question could not be asked about is a row the
    question did not hold for. That is what the SQL standard says and it is what
    `pick` already does, so nothing in the kernel changes.

    A chain of `WHEN`s is a chain of these, each one's else side being the next,
    which is how the parser already builds it and why there is no list here. An
    expression with four branches is three nodes and two intermediates, and the
    intermediates are dropped by the projection at the end the way every other
    expression's are.

    Both sides have to be the same type, because choosing between an int and a
    string is not a column. Lowering casts them to the type binding worked out
    before it builds this, so a mismatch here is a plan that was not bound.
    """

    var on: Int
    """The position of the condition, which must be boolean."""

    var left: Int
    """The position of the column the rows where the condition holds come from.
    """

    var right: Int
    """The position of the column the other rows come from."""

    var name: String
    """The name the appended column gets in the output schema."""

    def __init__(out self, on: Int, left: Int, right: Int, name: String):
        """Constructs a choice between two columns.

        Args:
            on: The position of the condition.
            left: The position of the true side.
            right: The position of the false side.
            name: The name of the appended column.
        """
        self.on = on
        self.left = left
        self.right = right
        self.name = name

    def _check(self, width: Int) raises:
        """Raises if any of the three positions is outside a chunk that wide.

        Args:
            width: How many columns there are.

        Raises:
            Error: If a position is outside the range.
        """
        var each = [self.on, self.left, self.right]
        for i in range(len(each)):
            if each[i] < 0 or each[i] >= width:
                raise Error(
                    "choose: column "
                    + String(each[i])
                    + " is outside a schema of "
                    + String(width)
                    + " columns"
                )

    def bind(mut self, var input: Schema) raises -> Schema:
        """Reports the input schema with the chosen column appended.

        Args:
            input: The schema of the chunks that will arrive. Consumed.

        Returns:
            The input schema with one field on the end.

        Raises:
            Error: If a position is outside the schema, the condition is not
                boolean, or the two sides are different types.
        """
        var out = input^
        self._check(len(out))
        if out[self.on].dtype.kind != TypeKind.BOOL:
            raise Error(
                "choose: the condition is "
                + String(out[self.on].dtype)
                + " and a condition is a yes or no question"
            )
        # One interior reference at a time, as `Compute.bind` does, since two
        # into the same list cannot both be alive.
        var made = out[self.left].dtype
        var other = out[self.right].dtype
        if made != other:
            raise Error(
                "choose: cannot choose between "
                + String(made)
                + " and "
                + String(other)
                + ", so the plan was not bound before it was lowered"
            )
        out.append(Field(self.name, made))
        return out^

    def process(self, var chunk: Chunk) raises -> Optional[Chunk]:
        """Applies the choice and puts the column on the end of the chunk.

        Args:
            chunk: The chunk. Consumed.

        Returns:
            The chunk with one more column and the same number of rows.

        Raises:
            Error: If a position is outside the chunk, the condition is not
                boolean, or the two sides are different types.
        """
        self._check(chunk.width())
        ref cond = chunk.columns[self.on].as_typed_view[DType.bool]()
        var made = pick_any(
            cond, chunk.columns[self.left], chunk.columns[self.right]
        )
        var rows = len(chunk)
        var columns = chunk^.into_columns()
        columns.append(made^)
        return Chunk(columns^, rows)


struct Cast(Movable):
    """Converts one column of the chunk to another type.

    It does that one of two ways, and which one is right depends on who else can
    see the column. Converting in place says this column is that type now, so
    the position keeps its meaning and everything downstream that referred to it
    still refers to the same thing. That is what `astype` on a frame means and
    it is why a cast is not a `Compute`, which always makes a column of its own.

    Appending is for the other case, and the query `SELECT a, CAST(a AS BIGINT)`
    is the whole of it. The input column is still wanted at its own type, so
    converting it where it lies would change what position zero means for every
    expression already bound against it. A cast written inside an expression
    lands in a column of its own for the same reason every other expression
    does, and the name it is given is the name that expression was given.
    """

    var on: Int
    """The position of the column to convert."""

    var to: LogicalType
    """The type to convert it to."""

    var strict: Bool
    """Whether text that is not a number raises rather than becoming a null."""

    var appends: Bool
    """Whether the converted column lands at the end rather than in place."""

    var name: String
    """The name the appended column takes. Empty when the cast is in place."""

    def __init__(out self, on: Int, to: LogicalType, strict: Bool = True):
        """Constructs a cast that converts the column where it lies.

        Args:
            on: The position of the column.
            to: The target type.
            strict: Whether a text value that is not a number raises rather than
                becoming a null. Ignored for a column that is not text.
        """
        self.on = on
        self.to = to
        self.strict = strict
        self.appends = False
        self.name = String()

    def __init__(
        out self,
        on: Int,
        to: LogicalType,
        var name: String,
        strict: Bool = True,
    ):
        """Constructs a cast that appends the converted column.

        Args:
            on: The position of the column to read.
            to: The target type.
            name: The name the new column gets.
            strict: Whether a text value that is not a number raises rather than
                becoming a null. Ignored for a column that is not text.
        """
        self.on = on
        self.to = to
        self.strict = strict
        self.appends = True
        self.name = name^

    def bind(mut self, var input: Schema) raises -> Schema:
        """Reports the input schema with the converted column in it.

        Args:
            input: The schema of the chunks that will arrive. Consumed.

        Returns:
            The same schema with the cast column's type replaced, or with the
            converted column added at the end.

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
        if self.appends:
            var nullable = out[self.on].nullable
            out.append(Field(self.name, self.to, nullable))
            return out^
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
            The chunk with one column converted, or with the converted column
            added at the end.

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
        var made = cast_any(columns[self.on], self.to, self.strict)
        if self.appends:
            columns.append(made^)
        else:
            columns[self.on] = made^
        return Chunk(columns^, rows)


struct Constant(Movable):
    """Appends a column holding the same value in every row.

    `Compute` covers an expression over columns and it covers an expression
    against a constant, but both of those need a column to start from, so
    neither can answer `SELECT 1`. A query that selects a bare constant asks for
    a column that nothing in the input decides, and that is what this is.

    Naming it a node rather than a special case of a projection is what keeps
    the constant where the rest of the plan can see it. Every other expression
    appends at the end and is read back by position, and a constant that arrived
    some other way would be a column the plan did not count.

    The value is stored with the type it should take rather than the type it
    happens to have, because a constant has no width of its own. `1` in a query
    is an integer of no particular size until the binder decides, and a filled
    timestamp is a timestamp rather than an integer of the same width, so what
    the plan worked out is what the column gets.
    """

    var value: Value
    """The value every row holds. A null value fills the column with nulls."""

    var type: LogicalType
    """The type the column takes."""

    var name: String
    """The name the appended column gets in the output schema."""

    def __init__(
        out self, var value: Value, type: LogicalType, var name: String
    ):
        """Constructs a constant column.

        Args:
            value: The value every row holds. Consumed.
            type: The type the column takes.
            name: The name of the appended column. Consumed.
        """
        self.value = value^
        self.type = type
        self.name = name^

    def bind(mut self, var input: Schema) raises -> Schema:
        """Reports the input schema with the constant column appended.

        Args:
            input: The schema of the chunks that will arrive. Consumed.

        Returns:
            The input schema with one field on the end.

        Raises:
            If the type has no physical layout to fill.
        """
        var out = input^
        if self.type.is_nested() or self.type.is_dictionary():
            raise Error(
                "constant: a column of "
                + String(self.type)
                + " is not filled from one value"
            )
        out.append(Field(self.name, self.type, self.value.is_null()))
        return out^

    def process(self, var chunk: Chunk) raises -> Optional[Chunk]:
        """Appends the column and hands the chunk back.

        Args:
            chunk: The chunk. Consumed.

        Returns:
            The chunk with one column on the end.

        Raises:
            If the value cannot be read as that type.
        """
        var rows = len(chunk)
        var made = filled_block(self.type, rows, self.value)
        var columns = chunk^.into_columns()
        columns.append(made^)
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

    Right and outer joins are refused, and so is a key that is more than one
    column. Both refusals are the same refusal: this node emits a chunk per chunk
    and nothing else. An outer join has to emit the right rows that nothing
    matched, which it cannot know until the last chunk has gone past, so it is a
    breaker wearing this node's clothes. A right join is an outer join's left
    half by the same argument. A composite key needs the ordinal space that
    `align_keys` builds by concatenating both sides, and concatenating both sides
    is having them both, which a stream does not.

    A text key used to be refused for that same reason and is not any more. It
    only ever needed the concatenation because the table it had stored a hash and
    a hash is not an exact answer for a string. A table that compares the bytes
    on a hash match does not need it, and once one side can be built and read
    without the other, text is an ordinary key here: build over the right frame,
    probe with each chunk as it arrives, hand the probe the right frame's key
    column back so it has the bytes to compare against.

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

    var mark: String
    """What a mark join's boolean column is called, or empty for every other
    kind. A mark join keeps no right column and adds this one instead, so it is
    the only kind with an output name of its own to settle."""

    var columns: List[String]
    """The projection, or empty for every output column.

    Here rather than left to a `Project` afterwards for the reason `join_on` has
    it: gathering is most of what a join costs, so a column that is going to be
    dropped is not a small waste at the end, it is most of the work.
    """

    var wanted: List[Int]
    """The projection by position, or empty for the one `columns` names.

    A position below the probe side's width is a probe column and the rest are
    the build side's, counted on from that width, which is how a plan numbers
    the two schemas end to end. A name cannot say which side it meant, so a
    caller whose two sides share one says so here instead, and then nothing is
    renamed and nothing is dropped: the output is this list and in this order.
    """

    var left_at: Int
    """The key's position on the probe side, or -1 to find it by name."""

    var right_at: Int
    """The key's position on the build side, or -1 to find it by name.

    Both are here for the same reason `wanted` is. A name finds the first column
    that has it, which is the wrong column as soon as two of them do, and a
    caller that numbered its columns already knows which one it meant.
    """

    var _left_at: Int
    """Where the key sits in the chunk, settled by `bind`."""

    var _right_at: Int
    """Where the key sits in the right frame, settled by `bind`.

    Kept because a text probe has to be handed the column the table was built
    from, since the views the table kept point into it. A fixed width probe never
    reads it.
    """

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

    var _build: List[AnyArray]
    """One chunk per right column, settled by `bind`.

    The right frame's columns are chunked and everything here reads them as one
    array, so this is where the two meet. A column of one chunk lends it, which
    shares the bytes and copies nothing. A column of no chunks, which is what a
    build side a filter emptied has, gets an empty array of its own type made
    here, so that a join against nothing joins against nothing rather than
    raising on a column shape.
    """

    def __init__(
        out self,
        var right: DataFrame,
        var left_on: String,
        var right_on: String,
        kind: JoinKind = JoinKind.INNER,
        var suffix: String = "_right",
        var columns: List[String] = List[String](),
        var wanted: List[Int] = List[Int](),
        left_at: Int = -1,
        right_at: Int = -1,
        var mark: String = String(),
    ):
        """Constructs a join against a frame.

        Nothing is built here. The table is built in `bind`, because building it
        needs the key dtype and the key dtype is a question about the chunks that
        will arrive as much as about the frame held here.

        Args:
            right: The build side. Consumed.
            left_on: The key column's name on the probe side. Consumed.
            right_on: The key column's name on the build side. Consumed.
            kind: Which rows to keep. Inner, left, semi, anti and mark only.
            suffix: Appended to a right column whose name collides. Consumed.
            columns: Which output columns to build, in the order wanted, or
                empty for all of them in their natural order. Consumed.
            wanted: The same thing by position over the two schemas end to end,
                or empty to go by name. Consumed.
            left_at: The key's position on the probe side, or -1 for by name.
            right_at: The key's position on the build side, or -1 for by name.
            mark: What a mark join's boolean column is called. Consumed.
                Required for a mark join and ignored by every other kind.
        """
        self.right = right^
        self.left_on = left_on^
        self.right_on = right_on^
        self.kind = kind
        self.suffix = suffix^
        self.mark = mark^
        self.columns = columns^
        self.wanted = wanted^
        self.left_at = left_at
        self.right_at = right_at
        self._left_at = -1
        self._right_at = -1
        self._side = BuildSide()
        self._table = ProbeTable()
        self._absent = List[Bool]()
        self._from_right = List[Bool]()
        self._source = List[Int]()
        self._build = List[AnyArray]()

    def bind(mut self, var input: Schema) raises -> Schema:
        """Builds the table from the right frame and plans the output.

        Args:
            input: The schema of the chunks that will arrive. Consumed.

        Returns:
            The schema of the chunks this emits.

        Raises:
            If the kind is one this node does not do, if either key name is
            missing, if the two keys have different dtypes, if one is text and
            the other is not, or if a projected name is not one the result has.
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
        if self.kind == JoinKind.MARK and self.mark.byte_length() == 0:
            raise Error(
                "join: a mark join hands out a boolean column and the column"
                " has to be called something, so the name is not optional"
            )

        var here = self.left_at
        var there = self.right_at
        if here < 0:
            here = input.index_of(self.left_on)
        if there < 0:
            there = self.right.schema.index_of(self.right_on)
        var dt = self.right.schema[there].dtype.physical
        # The string test is separate from the dtype test rather than folded into
        # it, because a string column's physical dtype is its byte type and a
        # column of bytes has the same one. Without this a text key on one side
        # and a uint8 key on the other would agree here and then build a table
        # over the first byte of each view.
        var mine = self.right.schema[there].dtype.kind == TypeKind.STRING
        var theirs = input[here].dtype.kind == TypeKind.STRING
        if input[here].dtype.physical != dt or theirs != mine:
            raise Error(
                "join: this node needs one key of the same dtype on each side;"
                " got "
                + ("text" if theirs else String(input[here].dtype.physical))
                + " and "
                + ("text" if mine else String(dt))
            )

        var rows = self.right.rows
        self._build = List[AnyArray](capacity=len(self.right.columns))
        for j in range(len(self.right.columns)):
            if self.right.columns[j].num_chunks() == 0:
                self._build.append(empty_any(self.right.schema[j].dtype))
            else:
                self._build.append(AnyArray(copy=self.right.columns[j].only()))

        var codes = Array[DType.uint32](overwritten=rows)
        var side = _build_key(self._build[there], codes)
        var absent = _key_nulls(self._build[there], rows)
        self._side = side^
        self._absent = absent^
        self._left_at = here
        self._right_at = there
        self._table = bucket_side(
            codes,
            0,
            rows,
            self._absent,
            0,
            len(self._absent) > 0,
            self._side.groups(),
        )

        if len(self.wanted) != 0:
            return self._marked(self._picked(input))

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
        return self._marked(Schema(kept^))

    def _marked(self, var planned: Schema) raises -> Schema:
        """Puts the mark join's own column on the end of what it gathers.

        Every other kind's output is columns of one side or the other, so this
        does nothing for them. A mark join's is the probe side and then one
        column that neither side holds, which is why it is added here rather
        than through `_source`: there is no position to gather it from.

        Args:
            planned: The gathered columns. Consumed.

        Returns:
            Those columns, with the mark on the end for a mark join.

        Raises:
            If the mark's name is one the gathered columns already use.
        """
        if self.kind != JoinKind.MARK:
            return planned^
        if planned.has(self.mark):
            raise Error(
                "join: a mark join was told to call its column '"
                + self.mark
                + "', and the probe side already has a column of that name"
            )
        var fields = List[Field]()
        for i in range(len(planned)):
            fields.append(planned[i].copy())
        fields.append(Field(String(self.mark), LogicalType.BOOL))
        return Schema(fields^)

    def _picked(mut self, input: Schema) raises -> Schema:
        """Plans the output from the positions `wanted` asked for.

        This is the other half of `bind`, and it is the half that does nothing
        clever. A caller that numbered the two schemas end to end has already
        decided what the output is, so there is no collision to rename around
        and no shared key to drop: every column asked for is gathered from the
        side its number falls on and keeps the name it had there.

        Args:
            input: The schema of the chunks that will arrive.

        Returns:
            The schema of the chunks this emits.

        Raises:
            If a position is not a column of either side, or if it is a build
            side column on a kind that keeps none of them.
        """
        var kept = List[Field]()
        self._from_right = List[Bool]()
        self._source = List[Int]()
        for i in range(len(self.wanted)):
            var p = self.wanted[i]
            if p < 0 or p >= len(input) + len(self.right.schema):
                raise Error(
                    String(
                        "join: column ",
                        p,
                        " was asked for and the two sides have ",
                        len(input) + len(self.right.schema),
                        " between them",
                    )
                )
            if p < len(input):
                kept.append(Field(input[p].name, input[p].dtype))
                self._from_right.append(False)
                self._source.append(p)
                continue
            if not self.kind.keeps_right_columns():
                raise Error(
                    String(
                        "join: a ",
                        self.kind,
                        " join keeps no column of the right side, and column ",
                        p,
                        " is one",
                    )
                )
            var j = p - len(input)
            var field = self.right.schema[j].copy()
            kept.append(Field(field.name, field.dtype))
            self._from_right.append(True)
            self._source.append(j)
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
        _probe_key(
            self._side,
            self._build[self._right_at],
            key,
            codes,
            spread,
        )
        var absent = _key_nulls(key, rows)
        if self.kind == JoinKind.MARK:
            # Nothing is gathered. Every row that arrived leaves, in the order
            # it arrived, so the chunk's own columns are the output columns and
            # the only new one is the answer.
            var marks = mark_probe(
                self._table,
                codes,
                0,
                rows,
                absent,
                0,
                len(absent) > 0,
                len(self._absent) > 0,
                spread,
            )
            var out = List[AnyArray](capacity=len(self._source) + 1)
            for w in range(len(self._source)):
                out.append(chunk.columns[self._source[w]].copy())
            out.append(AnyArray(marks^))
            _ = chunk^
            return Chunk(out^, rows)
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
                        self._build[self._source[w]],
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
        The table, ready to probe. A text one has to be probed with the same
        column handed back, because the views it kept point into it.

    Raises:
        If the dtype has no fixed width layout and is not text.
    """
    # Before the dispatch, because uint8 is in ALL and a string column would
    # match it and build a table over the first byte of each view.
    if key.is_string():
        return build_side_strings(key.strings(), 0, codes)
    comptime for candidate in ALL:
        if key.dtype() == candidate:
            ref view = key.as_typed_view[candidate]()
            return build_side[candidate](view, 0, codes)
    raise Error("join: no key table for dtype " + String(key.dtype()))


def _probe_key(
    built: BuildSide,
    source: AnyArray,
    key: AnyArray,
    mut codes: Array[DType.uint32],
    spread: Bool = True,
) raises:
    """Probes the built table with a column whose dtype is a runtime value.

    Args:
        built: The table the build side filled.
        source: The column the table was built from. Read only on the text
            route, where the views the table kept point into it and the
            comparison needs its bytes. Ignored otherwise.
        key: The probe side's key column.
        codes: Filled with one ordinal per row of it.
        spread: Whether the probe may use more than one core.

    Raises:
        If the dtype has no fixed width layout and is not text, or is not the
        one the table was built from.
    """
    if key.is_string():
        return probe_side_strings(
            built, source.strings(), key.strings(), 0, codes, spread
        )
    comptime for candidate in ALL:
        if key.dtype() == candidate:
            ref view = key.as_typed_view[candidate]()
            return probe_side[candidate](built, view, 0, codes, spread)
    raise Error("join: no key table for dtype " + String(key.dtype()))


struct GroupAgg(Copyable, Movable, Writable):
    """One output column of a grouped aggregation.

    An aggregate may carry an operation against a constant, which means it
    reduces `column op constant` rather than `column`. That is the same thing a
    `Compute` below the reduction would have produced, and the reason it can be
    said here instead is `Reduce`'s docstring: ninety sums over ninety
    expressions built by ninety `Compute` nodes is ninety columns of the chunk
    resident at once, and said here it is one at a time.

    Only `Reduce` reads it. A `Group` is given aggregates with no operation on
    them, because the lowering only folds the expression in when there are no
    keys, and `Group.bind` says so rather than ignoring it.
    """

    var column: Int
    """The position of the column to reduce, in the node's input."""

    var kind: AggKind
    """Which reduction."""

    var name: String
    """The name the output column gets."""

    var op: Optional[BinaryOp]
    """The operation applied to the column before reducing it, if any."""

    var constant: Value
    """The other operand of that operation. Read only when `op` is set."""

    var value_on_left: Bool
    """True for `5 - x` rather than `x - 5`. Read only when `op` is set."""

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
        self.op = None
        # Nothing reads this while `op` is empty, so it is the value that says
        # so rather than a second optional wrapped around the first.
        self.constant = Value(null=LogicalType.NULL)
        self.value_on_left = False

    def __init__(
        out self,
        column: Int,
        kind: AggKind,
        name: String,
        op: BinaryOp,
        var constant: Value,
        value_on_left: Bool = False,
    ):
        """Constructs one output column over an expression rather than a column.

        Args:
            column: The position of the column the operation reads.
            kind: The reduction.
            name: The output column's name.
            op: The operation applied before the reduction.
            constant: The operation's other operand. Consumed.
            value_on_left: Whether the constant is the left operand.
        """
        self.column = column
        self.kind = kind
        self.name = name
        self.op = op
        self.constant = constant^
        self.value_on_left = value_on_left

    def write_to(self, mut writer: Some[Writer]):
        """Writes the aggregate as it would be read back.

        Args:
            writer: The sink.
        """
        writer.write(self.kind, "(", self.column)
        if self.op:
            if self.value_on_left:
                writer.write(" on the right of ", self.op.value())
            else:
                writer.write(" ", self.op.value(), " a constant")
        writer.write(") as ", self.name)


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

    A kind that does not fold is not a kind nothing can run. It means the state
    is the values, so the operator holds the column and calls the whole frame
    kernel once at the end. `Reduce` does that and `Group` does it keyed, with
    the key columns held alongside. What this answers is which of the two a
    reduction gets, not whether it runs.

    Args:
        kind: The reduction.

    Returns:
        True if a running state is enough, False if the values themselves are
        the state.
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
    var dst = out.unsafe_mut_ptr()
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

    ## The ones that do not fold

    Not every reduction survives that. `_folds` is the list that does, and a
    median is not on it, because a median needs the values and there is no state
    short of the values that would give it one.

    So the values are what this holds for them. A reduction that does not fold
    names its source column, that column's chunks go by and are kept, the key
    columns are kept beside them, and at `finish` the whole thing is grouped once
    and the whole frame kernel is called. That is the same thing `Window` does
    with its partition and the same thing `Reduce` does without a key.

    The keys are grouped a second time here rather than reusing the ordinals the
    folding route already has, and that is deliberate. There are two folding
    routes with two different ideas of an ordinal, `_push` keeps one that lasts
    and `_absorb` makes a new one per chunk, and `_demote` can switch between
    them in the middle of a query. What both of them agree on is the group order
    the output promises, which is the order the groups were first seen, and that
    is exactly what a group by over the held keys gives. So the second grouping
    is one pass over the kept rows and it is correct whichever route the folds
    took. `_settle` checks the two group counts against each other.

    The cost is the whole of every held column resident, which is the thing this
    node exists to avoid, and it is paid only for the key columns and the columns
    a non folding reduction reads. `sum(x), median(y) GROUP BY k` holds `k` and
    `y` and still folds `x` into one row per group. The alternative to paying it
    is refusing the query, which is what this node used to do.

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
    """Per aggregate, the slot it starts at. A state slot when it folds and a
    held slot when it does not. A mean folds and owns two state slots."""

    var _holds: List[Bool]
    """Per aggregate, whether `_at` points into `held` rather than `state`."""

    var held: List[ChunkedArray]
    """Per held slot, every chunk of the column that slot reduces."""

    var _kept: List[Int]
    """Per held slot, the input column it holds."""

    var _late: List[AggKind]
    """Per held slot, the reduction to run over each group at the end."""

    var _key_held: List[ChunkedArray]
    """Per key column, every chunk of it, kept beside `held` so the held rows
    can be grouped at the end. Empty when nothing is held."""

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
        self._holds = List[Bool]()
        self.held = List[ChunkedArray]()
        self._kept = List[Int]()
        self._late = List[AggKind]()
        self._key_held = List[ChunkedArray]()
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
        reduction that reads a column this node has no second name for, a sum of
        a column of names, two output columns with the same name. None of those
        needs a row to detect and all of them are cheaper to report before the
        first one moves.

        Whether a reduction folds is settled here too, and it is not an error
        either way. A reduction that does not gets a held slot instead of a state
        slot, and `_at` points into whichever of the two `_holds` says.

        Args:
            input: The schema of the chunks that will arrive. Consumed.

        Returns:
            The key columns in the order they were given, then one column per
            aggregate.

        Raises:
            If a position is outside the schema, if a key is repeated, if a
            reduction reads two columns, if a reduction has no meaning on its
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
            if kind.reads_two_columns():
                raise Error(
                    "group: "
                    + String(kind)
                    + " reads two columns and a reduction here names one"
                )
            if self.aggs[a].op:
                # Only `Reduce` folds an operation into the reduction, because
                # only `Reduce` reads every row of the column it reduces. Here
                # the rows are scattered across groups and the operation would
                # have to move with them, which is what a `Compute` in front of
                # this node already does.
                raise Error(
                    "group: a reduction here cannot carry an operation, the"
                    " column has to be computed first"
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
            fields.append(Field(name, agg_type(kind, source)))

            if not _folds(kind):
                # The values themselves are the state, so the column goes by and
                # the kernel runs once over each group's share of it at the end.
                self._holds.append(True)
                self._at.append(len(self._kept))
                self._kept.append(at)
                self._late.append(kind)
                self.held.append(ChunkedArray(source))
                continue

            self._holds.append(False)
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

        # The keys are only kept when something is held, because they are only
        # kept so that the held rows can be grouped, and a query with nothing
        # held should carry nothing extra.
        if len(self._kept) > 0:
            for k in range(len(self.keys)):
                self._key_held.append(
                    ChunkedArray(self.input[self.keys[k]].dtype)
                )

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

        if len(self._kept) > 0:
            # Before either folding route, and the same on both, because a
            # reduction whose state is the values has no opinion about which one
            # the folds took and `_demote` can change it halfway through.
            for k in range(len(self.keys)):
                self._key_held[k].append(AnyArray(copy=columns[self.keys[k]]))
            for h in range(len(self._kept)):
                self.held[h].append(AnyArray(copy=columns[self._kept[h]]))

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

    def _reduce_held(mut self, groups: Int) raises -> List[AnyArray]:
        """Groups the held rows once and reduces each group's share of them.

        The keys are grouped again rather than reused, for the reason in the
        struct's docstring: the two folding routes keep two different kinds of
        ordinal and `_demote` can swap one for the other mid query, while the
        group order the output promises is the same on both and is what a group
        by over the held keys gives back.

        Args:
            groups: How many groups the running table ended up with, which is
                what this has to agree with.

        Returns:
            One column per held slot, one row per group, in the running table's
            group order.

        Raises:
            If the held chunks cannot be stacked, if grouping them raises, or if
            the two group counts disagree.
        """
        var keys = List[AnyArray](capacity=len(self.keys))
        for k in range(len(self.keys)):
            keys.append(ChunkedArray(copy=self._key_held[k]).combine())
        var rows = len(keys[0])
        var refs = borrow_columns(keys)
        var at = List[Int](capacity=len(self.keys))
        for k in range(len(self.keys)):
            at.append(k)
        var found = group_ordinals(refs, at, rows)
        if found.groups != groups:
            raise Error(
                "group: the held rows fall into "
                + String(found.groups)
                + " groups and the running table has "
                + String(groups)
            )

        var out = List[AnyArray](capacity=len(self._kept))
        for h in range(len(self._kept)):
            out.append(
                aggregate_group_any(
                    ChunkedArray(copy=self.held[h]).combine(),
                    self._late[h],
                    found.codes,
                    found.groups,
                    trusted=True,
                )
            )
        self.held = List[ChunkedArray]()
        self._key_held = List[ChunkedArray]()
        return out^

    def _settle(mut self) raises:
        """Turns the running state into output columns and cuts them into chunks.

        Raises:
            If a mean cannot be computed from its sum and its count, or if the
            held columns cannot be grouped and reduced.
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
        var late = List[AnyArray]()
        if len(self._kept) > 0:
            late = self._reduce_held(len(self.state[0]))
        var base = len(self.keys)
        for a in range(len(self.aggs)):
            var at = self._at[a]
            if self._holds[a]:
                out.append(AnyArray(copy=late[at]))
            elif self.aggs[a].kind == AggKind.MEAN:
                out.append(
                    _mean_of(self.state[base + at], self.state[base + at + 1])
                )
            else:
                out.append(AnyArray(copy=self.state[base + at]))
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

    `_folds` is the list of reductions that can be done this way. A median of
    medians is not a median and no state short of the values themselves would
    make it one, and a distinct count's partial answer has thrown away exactly
    what the merge would need, so those reductions are not folded here.

    ## The ones that do not fold

    They are held instead. A reduction that does not fold names its source
    column, the node keeps that column's chunks as they go past, and at `finish`
    it flattens them and calls the whole frame kernel once. That is what
    `Window` already does with its partition, which is why `count(DISTINCT x)
    OVER ()` has worked all along, and it is the only shape there is: the values
    themselves are the state.

    The cost is honest and it is the whole column resident, which is the thing a
    chunked engine exists to avoid. It is paid only for the columns the non
    folding reductions read. `sum(x), median(y)` folds `x` a chunk at a time and
    holds `y`, so a query that asks for one median does not stop being streaming
    everywhere else. The alternative to paying it is refusing the query, which
    is what this node used to do.

    A held column can be made smaller, and for a distinct count it can be made
    into nothing at all, since a set of values seen is not the values. That is a
    kernel change rather than an operator one and it goes under this without
    changing it.

    Floating point is the one place the answer can differ from calling `agg` on
    the whole frame. Adding a column in chunks and then adding the chunk sums is
    a different order of additions from adding it in one pass, and floating
    point addition is not associative, so a sum of floats can differ in the last
    bits. Every other kind here is exact.

    An input that hands over no rows still produces one row, because a fold with
    no key is one group whether or not anything was read. That is not what
    `Group` does with the same input, and the difference is the point: a group
    by with keys finds no groups in nothing and so has no rows to hand out,
    while the whole input is a group that is always there. What is in that row
    is whatever the same reduction answers over a column of no rows, which is a
    zero for a count and a null for a minimum and a maximum, and it is read off
    the kernel rather than written out here so that the two cannot drift.

    An aggregate here may carry an operation against a constant and reduce what
    that produces rather than the column itself. The one that made it worth
    doing is ClickBench q29, ninety sums over one column under ninety different
    constants, which as ninety `Compute` nodes is ninety columns of the chunk
    alive at once and costs ninety times the column being read. Folded in, each
    one is computed, reduced and dropped before the next is built, so at any
    moment the chunk holds what arrived and one more column. At ten million
    rows in one chunk that is four and a half gigabytes of difference, and the
    fused route is also the faster of the two by a wide margin once the
    materializing one starts paging.
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

    var _shift: List[Int]
    """Per state slot, the entry of `_shifts` that transforms its column before
    the reduction reads it, or minus one for a slot that reads the column as it
    arrived."""

    var _shifts: List[GroupAgg]
    """The aggregates that carry an operation, kept for their operand. An
    aggregate rather than a tuple of its own because that is already what holds
    the three fields together and what `bind` was handed."""

    var _produce: List[AggKind]
    """Per state slot, the reduction run over a chunk."""

    var _merge: List[AggKind]
    """Per state slot, the reduction that combines two partial answers."""

    var _at: List[Int]
    """Per aggregate, the slot it starts at. A state slot when it folds and a
    held slot when it does not. A mean folds and owns two state slots."""

    var _holds: List[Bool]
    """Per aggregate, whether `_at` points into `held` rather than `state`."""

    var held: List[ChunkedArray]
    """Per held slot, every chunk of the column that slot reduces."""

    var _kept: List[Int]
    """Per held slot, the input column it holds."""

    var _kept_shift: List[Int]
    """Per held slot, its entry in `_shifts`, or minus one."""

    var _late: List[AggKind]
    """Per held slot, the reduction to run over the whole column at the end."""

    var started: Bool
    """Whether a chunk with rows in it has arrived."""

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
        self._shift = List[Int]()
        self._shifts = List[GroupAgg]()
        self._produce = List[AggKind]()
        self._merge = List[AggKind]()
        self._at = List[Int]()
        self._holds = List[Bool]()
        self.held = List[ChunkedArray]()
        self._kept = List[Int]()
        self._kept_shift = List[Int]()
        self._late = List[AggKind]()
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
            a reduction reads two columns, if a reduction has no meaning on its
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
            if kind.reads_two_columns():
                raise Error(
                    "reduce: "
                    + String(kind)
                    + " reads two columns and a reduction here names one"
                )
            # An aggregate over an expression reduces what the operation
            # produces, not what the column holds, and the promotion is worked
            # out here for the reason `Compute.bind` works it out there: this
            # declares the dtype of a call that has not run yet, and a rule
            # applied in one place and not the other is a schema that does not
            # describe the data. Both go through `resolve_constant` and
            # `binary_type`, so the two cannot drift.
            var shift = -1
            var source = self.input[at].dtype
            if self.aggs[a].op:
                var op = self.aggs[a].op.value()
                var k = resolve_constant(source, self.aggs[a].constant, op).type
                var on_left = self.aggs[a].value_on_left
                var left = source if not on_left else k
                var right = k if not on_left else source
                source = binary_type(op, left, right)
                shift = len(self._shifts)
                self._shifts.append(self.aggs[a].copy())
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
            fields.append(Field(name, agg_type(kind, source)))

            if not _folds(kind):
                # The values themselves are the state, so the column goes by
                # and the kernel runs once over all of it at the end.
                self._holds.append(True)
                self._at.append(len(self._kept))
                self._kept.append(at)
                self._kept_shift.append(shift)
                self._late.append(kind)
                self.held.append(ChunkedArray(source))
                continue

            self._holds.append(False)
            self._at.append(len(self._source))
            if kind == AggKind.MEAN:
                self._source.append(at)
                self._shift.append(shift)
                self._produce.append(AggKind.SUM)
                self._merge.append(AggKind.SUM)
                self._source.append(at)
                self._shift.append(shift)
                self._produce.append(AggKind.COUNT)
                self._merge.append(AggKind.SUM)
            else:
                self._source.append(at)
                self._shift.append(shift)
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

        A reduction that does not fold has nothing to put in that row, so its
        column rides along behind it at full height instead. That keeps the two
        halves of this node the same shape whichever route a chunk took, which
        matters because the parallel path calls this from a worker and `absorb`
        from the thread that owns the node. What comes back is one row per
        partial answer and then one whole column per held slot, so it is a chunk
        of two different heights and nothing but `absorb` may read it.

        Args:
            chunk: The chunk. Consumed.

        Returns:
            The partial answers and the held columns, or None for a chunk with
            no rows.

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
        var made = List[AnyArray](capacity=len(self._source) + len(self._kept))
        for s in range(len(self._source)):
            if self._shift[s] < 0:
                made.append(
                    reduce_any(columns[self._source[s]], self._produce[s])
                )
                continue
            # Computed, reduced and dropped inside the one iteration, which is
            # the whole point of folding the operation in here. The `Compute`
            # nodes this replaces each left their column on the chunk until the
            # reduction above them had read all of them, so ninety of them meant
            # ninety columns of the chunk resident at once. Here it is one.
            ref it = self._shifts[self._shift[s]]
            var made_here = binary_value_any(
                columns[self._source[s]],
                it.constant,
                it.op.value(),
                it.value_on_left,
            )
            made.append(reduce_any(made_here, self._produce[s]))
        for k in range(len(self._kept)):
            if self._kept_shift[k] < 0:
                made.append(AnyArray(copy=columns[self._kept[k]]))
                continue
            # A held slot keeps the whole column whatever happens, so there is
            # no footprint to save here and the operation is applied for the
            # one reason that matters, which is that the answer has to be the
            # same as the `Compute` route's.
            ref it = self._shifts[self._kept_shift[k]]
            made.append(
                binary_value_any(
                    columns[self._kept[k]],
                    it.constant,
                    it.op.value(),
                    it.value_on_left,
                )
            )
        # Unchecked, because the held columns are the chunk's height and the
        # partial answers are one row, and nothing but `absorb` reads this.
        return Chunk(made^, 1)

    def absorb(mut self, var partial: Chunk) raises:
        """Merges one chunk's partial answers into the running row.

        Args:
            partial: The output of `partial`, which is the folded answers in
                `_source` order and then the held columns. Consumed.

        Raises:
            If it is not the width `partial` produces, or if merging raises.
        """
        var want = len(self._source) + len(self._kept)
        if partial.width() != want:
            raise Error(
                "reduce: a partial row has "
                + String(partial.width())
                + " columns and this reduction produces "
                + String(want)
            )
        var made = partial^.into_raw_columns()
        for k in range(len(self._kept)):
            self.held[k].append(AnyArray(copy=made[len(self._source) + k]))
        if not self.started:
            self.started = True
            self.state = List[AnyArray](capacity=len(self._source))
            for s in range(len(self._source)):
                self.state.append(AnyArray(copy=made[s]))
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
            One chunk of exactly one row the first time and None after that,
            including when no chunk with rows in it ever arrived.

        Raises:
            If a mean cannot be computed from its sum and its count, or if a
            state slot's column has no empty form, or if a held column cannot
            be flattened or reduced.
        """
        if self.ran:
            return None
        self.ran = True
        if not self.started:
            # Nothing was read, and the answer is still one row. Each slot gets
            # what its reduction answers over a column of no rows, which is the
            # kernel's answer rather than a table of identities written here: a
            # count finds nothing and is zero, a minimum and a maximum find
            # nothing and are null, and a sum is zero because that is what the
            # same kernel answers over a column that is entirely null and the
            # two cases have to agree with each other.
            self.state = List[AnyArray](capacity=len(self._source))
            for s in range(len(self._source)):
                var none = empty_any(self.input[self._source[s]].dtype)
                self.state.append(reduce_any(none, self._produce[s]))

        var out = List[AnyArray](capacity=len(self.aggs))
        for a in range(len(self.aggs)):
            var at = self._at[a]
            if self._holds[a]:
                # One kernel call over the whole column, which is the same call
                # `agg` on a frame would have made and the only one there is.
                out.append(
                    reduce_any(
                        ChunkedArray(copy=self.held[at]).combine(),
                        self._late[at],
                    )
                )
            elif self.aggs[a].kind == AggKind.MEAN:
                out.append(_mean_of(self.state[at], self.state[at + 1]))
            else:
                out.append(AnyArray(copy=self.state[at]))
        self.state = List[AnyArray]()
        self.held = List[ChunkedArray]()
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
    Filter,
    Expand,
    Project,
    Compute,
    Connective,
    Choose,
    Constant,
    Cast,
    Join,
    Limit,
    Sort,
    Window,
    Group,
    Reduce,
    Materialize,
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
    if node.isa[Connective]():
        return node[Connective].bind(input^)
    if node.isa[Choose]():
        return node[Choose].bind(input^)
    if node.isa[Constant]():
        return node[Constant].bind(input^)
    if node.isa[Cast]():
        return node[Cast].bind(input^)
    if node.isa[Join]():
        return node[Join].bind(input^)
    if node.isa[Sort]():
        return node[Sort].bind(input^)
    if node.isa[Window]():
        return node[Window].bind(input^)
    if node.isa[Project]():
        return _rename(
            node[Project].names,
            _narrow(node[Project].keep, input, "project"),
        )
    if node.isa[Expand]():
        return node[Expand].bind(input^)
    if node.isa[Filter]() and node[Filter].narrows:
        return _narrow(node[Filter].keep, input, "filter")
    return input^


def _narrow(keep: List[Int], input: Schema, who: String) raises -> Schema:
    """Returns the schema of the kept positions, in the order given.

    Args:
        keep: The input positions.
        input: The schema they are positions into.
        who: The operator, for the message when one is out of range.

    Returns:
        The schema that comes out.

    Raises:
        If a position is outside the input.
    """
    var fields = List[Field](capacity=len(keep))
    for i in range(len(keep)):
        if keep[i] < 0 or keep[i] >= len(input):
            raise Error(
                who
                + ": column "
                + String(keep[i])
                + " is outside a schema of "
                + String(len(input))
                + " columns"
            )
        fields.append(input[keep[i]].copy())
    return Schema(fields^)


def _rename(names: List[String], var schema: Schema) raises -> Schema:
    """Puts a projection's output names onto the columns it kept.

    Args:
        names: One name per column, or empty to leave them alone.
        schema: What the projection produces. Consumed.

    Returns:
        The same columns under the names given.

    Raises:
        If there is a name for a column that is not there, or a column with no
        name, since either one means the two lists came from different places.
    """
    if len(names) == 0:
        return schema^
    if len(names) != len(schema):
        raise Error(
            "project: "
            + String(len(names))
            + " names for "
            + String(len(schema))
            + " columns"
        )
    var fields = List[Field](capacity=len(names))
    for i in range(len(names)):
        var field = schema[i].copy()
        field.name = names[i]
        fields.append(field^)
    return Schema(fields^)


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
    if node.isa[Sort]():
        return node[Sort].update_state()
    if node.isa[Window]():
        return node[Window].update_state()
    if node.isa[Group]():
        return node[Group].update_state()
    if node.isa[Reduce]():
        return node[Reduce].update_state()
    return NodeStatus.NEED_MORE_INPUT


def node_is_row_local(node: Node) -> Bool:
    """Reports whether a node's output row depends only on its own input row.

    The elementwise operators say yes, and so do `Expand`, whose output rows are
    copies of the input row it is on, and `Join`, whose output row depends on
    its own input row and on a table that was finished before the first chunk
    arrived. What that buys is that the node reads itself and never
    writes itself, so one of them can be handed to every core at once without a
    copy per worker and without a lock. `Limit` counts rows, `Sort` holds every
    row until it knows where the first one goes, `Window` holds every row until
    it knows what the partition sums to, `Group` holds a table it is still
    filling, `Reduce` holds a running answer and `Materialize` holds the input,
    so all six say no.

    Args:
        node: The node.

    Returns:
        True for `Filter`, `Expand`, `Project`, `Compute`, `Connective`,
        `Choose`, `Constant`, `Cast` and `Join`.
    """
    return (
        node.isa[Filter]()
        or node.isa[Expand]()
        or node.isa[Project]()
        or node.isa[Compute]()
        or node.isa[Connective]()
        or node.isa[Choose]()
        or node.isa[Constant]()
        or node.isa[Cast]()
        or node.isa[Join]()
    )


def node_ends_early(node: Node) -> Bool:
    """Reports whether a node can say FINISHED before its input runs out.

    Only `Limit` can. `Sort`, `Window`, `Group` and `Materialize` say FINISHED
    too, but
    not until `finish` has handed back everything they held, which is after the
    source is empty. The distinction matters to the driver: a pipeline that can stop early
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

    `Filter` evaluates a predicate, and `Compute` and `Connective` evaluate an
    expression, so all three do work once per row and all three get faster on
    more cores. `Join`
    hashes a key and gathers a row per output row, which is more work per row
    than either, and its table is read only once `bind` has run. `Project`
    only rebuilds a chunk out of columns it already has, `Cast` walks a
    column through the allocator and `Constant` writes the same value down a
    buffer, so none of the three has much for a second core to do and all three
    are held up by memory rather than by arithmetic. Measured on the
    i9-13900K, a line with a compute and a filter in it ran three times faster
    spread over the cores, while a project on its own ran no faster at all and
    paid for the tasks on top.

    Args:
        node: The node.

    Returns:
        True for `Filter`, `Compute`, `Connective`, `Choose` and `Join`.
    """
    return (
        node.isa[Filter]()
        or node.isa[Compute]()
        or node.isa[Connective]()
        or node.isa[Choose]()
        or node.isa[Join]()
    )


def node_is_breaker(node: Node) -> Bool:
    """Reports whether a node has to see all its input before it emits.

    Args:
        node: The node.

    Returns:
        True for a breaker, which is where a pipeline is cut.
    """
    return (
        node.isa[Materialize]()
        or node.isa[Sort]()
        or node.isa[Window]()
        or node.isa[Group]()
        or node.isa[Reduce]()
    )


def node_reads_selection(node: Node) -> Bool:
    """Reports whether a node can take a chunk that carries a selection.

    A chunk under a selection is one whose columns are not all at its rows, and
    a kernel handed one without knowing would read the wrong values rather than
    fail. So this is the list of what has been taught, and everything else has
    `flatten` called on its input by the two dispatchers below before it sees
    it. That is what lets the selection be turned on one operator at a time
    without any answer changing in between.

    Nothing reads one yet. Issue #521 turns them on.

    Args:
        node: The node.

    Returns:
        True if the node handles a selected chunk itself.
    """
    return False


def node_process(mut node: Node, var chunk: Chunk) raises -> Optional[Chunk]:
    """Pushes one chunk through a node.

    Flattens the chunk first unless the node says it reads a selection, so a
    node that has not been taught about selections cannot be given one.

    Args:
        node: The node.
        chunk: The chunk. Consumed.

    Returns:
        What the node emits, or None if it emits nothing for this chunk.

    Raises:
        If the node cannot process the chunk.
    """
    if chunk.selected() and not node_reads_selection(node):
        chunk.flatten()
    if node.isa[Filter]():
        return node[Filter].process(chunk^)
    if node.isa[Expand]():
        return node[Expand].process(chunk^)
    if node.isa[Project]():
        return node[Project].process(chunk^)
    if node.isa[Compute]():
        return node[Compute].process(chunk^)
    if node.isa[Connective]():
        return node[Connective].process(chunk^)
    if node.isa[Choose]():
        return node[Choose].process(chunk^)
    if node.isa[Constant]():
        return node[Constant].process(chunk^)
    if node.isa[Cast]():
        return node[Cast].process(chunk^)
    if node.isa[Join]():
        return node[Join].process(chunk^)
    if node.isa[Limit]():
        return node[Limit].process(chunk^)
    if node.isa[Sort]():
        return node[Sort].process(chunk^)
    if node.isa[Window]():
        return node[Window].process(chunk^)
    if node.isa[Group]():
        return node[Group].process(chunk^)
    if node.isa[Reduce]():
        return node[Reduce].process(chunk^)
    return node[Materialize].process(chunk^)


def node_apply(node: Node, var chunk: Chunk) raises -> Optional[Chunk]:
    """Pushes one chunk through a row local node without mutating it.

    The same call as `node_process` for the five elementwise operators, except
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
    if chunk.selected() and not node_reads_selection(node):
        # False, for the reason the join below gives: this already runs on a
        # worker, so a gather here must not hand itself out to workers again.
        chunk.flatten(False)
    if node.isa[Filter]():
        return node[Filter].process(chunk^)
    if node.isa[Expand]():
        return node[Expand].process(chunk^)
    if node.isa[Project]():
        return node[Project].process(chunk^)
    if node.isa[Compute]():
        return node[Compute].process(chunk^)
    if node.isa[Connective]():
        return node[Connective].process(chunk^)
    if node.isa[Choose]():
        return node[Choose].process(chunk^)
    if node.isa[Constant]():
        return node[Constant].process(chunk^)
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
    if node.isa[Sort]():
        return node[Sort].finish()
    if node.isa[Window]():
        return node[Window].finish()
    if node.isa[Group]():
        return node[Group].finish()
    if node.isa[Reduce]():
        return node[Reduce].finish()
    return None
