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

`Materialize` is the escape hatch and it is not temporary. It collects every
chunk into one frame, calls a whole frame function, and gives the answer back as
chunks. Anything with no chunked implementation goes through it, which is what
lets the engine be built one pull request at a time instead of in one commit:
every operator starts as a `Materialize`, the pipeline is exactly as fast as the
tree is today, and each later change removes one. Polars shipped `InMemoryMap`
and `InMemoryJoin` for the same reason and still has them.
"""

from std.utils import Variant

from firepanda.array.any import AnyArray
from firepanda.array.chunked import ChunkedArray
from firepanda.dtype.schema import Field, Schema
from firepanda.frame.frame import DataFrame
from firepanda.kernel.select import filter_any

from .chunk import Chunk


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


comptime Node = Variant[Filter, Project, Limit, Materialize]
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
    return NodeStatus.NEED_MORE_INPUT


def node_is_breaker(node: Node) -> Bool:
    """Reports whether a node has to see all its input before it emits.

    Args:
        node: The node.

    Returns:
        True for a breaker, which is where a pipeline is cut.
    """
    return node.isa[Materialize]()


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
    if node.isa[Limit]():
        return node[Limit].process(chunk^)
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
    return None
