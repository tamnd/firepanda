"""A source, a line of operators, and a sink, executed by pushing chunks.

The driver is fifteen lines and the shape of it is the whole design. Take a
chunk from the source, hand it to the first operator, hand what comes out to the
second, and put whatever reaches the end into the sink. A chunk that an operator
swallows, because a filter kept none of its rows or because a breaker is still
collecting, stops there and the next chunk starts over. When the source is
empty, walk the operators in order asking each one to finish, and push whatever
it gives up through the operators after it.

That second loop is what pipeline cutting is. A breaker splits the line into two
stages that cannot overlap in time, and running the stages one after another is
exactly what draining in order does, with the breaker itself holding the
material between them. There is no separate plan structure and no second
traversal; `cut_points` reports where the cuts are because a user of the engine
wants to see them, not because the driver needs them.

Chunks are pushed rather than pulled. A pull engine asks the sink for a row and
the request walks up the operators to the source, which is the natural shape for
a nested loop join and an awkward one for everything else, because every
operator has to be able to suspend in the middle of its work. Pushing means an
operator is a function from a chunk to a chunk, it runs to completion, and it
keeps no resumption state. DuckDB and the Polars streaming engine both push, and
the reason both give is this one.

The leading run of elementwise operators runs on every core. A filter, a
projection, an expression and a cast all produce an output row from their own
input row and nothing else, and none of them writes anything to itself, so the
same node can be handed to thirty two workers at once with no copy and no lock.
The driver finds how many operators at the front of the line are like that, and
runs the source's chunks through that prefix in parallel. Whatever comes out is
pushed through the rest of the line in chunk order on the calling thread, so the
result is the same frame in the same order as the sequential driver produces.

It runs a batch at a time rather than the whole source at once, and the batch is
one chunk per worker. That is what keeps the engine streaming. Running every
chunk through the prefix first and then draining would hold the whole
intermediate result in memory, which is what a chunked engine exists to avoid;
a batch holds one chunk per core and no more.

A pipeline containing a `Limit` is fed one chunk at a time as before. Reading
ahead on behalf of thirty two cores is reading rows that the limit was about to
make unnecessary, and a `head` over a large file is the case where that is the
whole cost of the query.

What is still not here is a parallel breaker. A group by merges every chunk into
one table on the calling thread, so a query whose work is in the grouping gets
the prefix in parallel and the grouping serial. Thread local tables merged
partition wise is the next step and it is a change to `Group`, not to this file.
"""

from firepanda.array.any import AnyArray
from firepanda.array.chunked import ChunkedArray
from firepanda.dtype.schema import Schema
from firepanda.frame.frame import DataFrame

from .chunk import Chunk
from .node import Node, NodeStatus, node_apply, node_bind, node_computes_per_row
from .node import node_ends_early, node_finish, node_is_breaker
from .node import node_is_row_local, node_process, node_status
from .parallel import parallel_for, worker_count


struct Scan(Movable):
    """The source, which takes a frame apart into the chunks it is made of.

    A frame's columns are already chunked, so this copies nothing. It takes the
    chunk lists out of the columns, and each call hands one array per column
    downstream and forgets it. The frame is consumed, which is what lets the
    memory of a chunk be released as soon as the operators are done with it
    rather than at the end of the query.

    Every column has to be chunked the same way, which is true of any frame the
    tree produces, because a chunk is a horizontal slice and there is no such
    slice if column A breaks at row 100 and column B breaks at row 150. A frame
    that does not satisfy that is rejected here rather than half way through.
    """

    var columns: List[List[AnyArray]]
    """Per column, the chunks still to be handed out, in reverse order."""

    var remaining: Int
    """The number of chunks left."""

    def __init__(out self, var frame: DataFrame) raises:
        """Constructs a scan over a frame, consuming it.

        Args:
            frame: The frame. Consumed.

        Raises:
            If the columns are not chunked the same way.
        """
        var owned = frame^.into_columns()
        var flipped = List[List[AnyArray]](capacity=len(owned))
        while len(owned) > 0:
            var chunks = owned.pop().into_chunks()
            var backwards = List[AnyArray](capacity=len(chunks))
            while len(chunks) > 0:
                backwards.append(chunks.pop())
            flipped.append(backwards^)
        var columns = List[List[AnyArray]](capacity=len(flipped))
        while len(flipped) > 0:
            columns.append(flipped.pop())
        var count = 0 if len(columns) == 0 else len(columns[0])
        for i in range(len(columns)):
            if len(columns[i]) != count:
                raise Error(
                    "scan: column "
                    + String(i)
                    + " has "
                    + String(len(columns[i]))
                    + " chunks and column 0 has "
                    + String(count)
                )
        self.columns = columns^
        self.remaining = count

    def num_chunks(self) -> Int:
        """Returns how many chunks are left to hand out.

        Returns:
            The count.
        """
        return self.remaining

    def next(mut self) raises -> Optional[Chunk]:
        """Hands out the next chunk, or None when there are none left.

        Returns:
            One array per column, all of the same length.

        Raises:
            If the arrays that come out are not all the same length, which
            would mean a column's prefix sums disagreed with its chunks.
        """
        if self.remaining == 0:
            return None
        var row = List[AnyArray](capacity=len(self.columns))
        for i in range(len(self.columns)):
            row.append(self.columns[i].pop())
        self.remaining -= 1
        return Chunk(row^)


struct Collect(Movable):
    """The sink, which puts the chunks that reach it back into a frame.

    The chunk boundaries are kept rather than stacked, so a query that read
    sixteen row groups and filtered them gives back a frame of sixteen columns
    worth of pieces and never builds a contiguous array of the result. Anything
    that wants one calls `combine` on the column it wants.
    """

    var columns: List[ChunkedArray]
    """One column per position, holding the chunks seen so far."""

    var started: Bool
    """Whether the first chunk has arrived and set the column types."""

    def __init__(out self):
        """Constructs an empty sink."""
        self.columns = List[ChunkedArray]()
        self.started = False

    def push(mut self, var chunk: Chunk) raises:
        """Adds one chunk to the result.

        Args:
            chunk: The chunk. Consumed.

        Raises:
            If the chunk's width or dtypes disagree with the chunks before it.
        """
        if not self.started:
            self.started = True
            for i in range(chunk.width()):
                self.columns.append(ChunkedArray(chunk.columns[i].type))
        if chunk.width() != len(self.columns):
            raise Error(
                "collect: chunk has "
                + String(chunk.width())
                + " columns and the ones before it had "
                + String(len(self.columns))
            )
        var backwards = chunk^.into_columns()
        var forwards = List[AnyArray](capacity=len(backwards))
        while len(backwards) > 0:
            forwards.append(backwards.pop())
        for i in range(len(self.columns)):
            self.columns[i].append(forwards.pop())

    def into_frame(deinit self, var schema: Schema) raises -> DataFrame:
        """Turns what was collected into a frame, consuming the sink.

        A query whose sink saw nothing still has a shape, so the schema is what
        decides the width and an empty column of the right type is made for each
        field. That is the difference between a frame of no rows and no frame.

        Args:
            schema: The schema of the result. Consumed.

        Returns:
            The result.

        Raises:
            If the collected width does not match the schema.
        """
        if not self.started:
            var empty = List[ChunkedArray](capacity=len(schema))
            for i in range(len(schema)):
                empty.append(ChunkedArray(schema[i].dtype))
            return DataFrame(schema^, empty^)
        if len(self.columns) != len(schema):
            raise Error(
                "collect: the result has "
                + String(len(self.columns))
                + " columns and the schema has "
                + String(len(schema))
            )
        return DataFrame(schema^, self.columns^)


def _run_head(
    ops: List[Node], lead: Int, mut taken: List[Optional[Chunk]]
) raises -> List[Optional[Chunk]]:
    """Runs one batch of chunks through the first `lead` operators, in parallel.

    A free function rather than a method so that the body handed to
    `parallel_for` captures the two lists and the operators and nothing else.
    A closure written inside `Pipeline` would capture the whole pipeline, and
    the source and the sink have no business being reachable from a worker.

    `taken` and `made` are read and written at one index per worker, and both
    lists are their final length before the first body runs, so no two workers
    touch the same element and nothing moves under anyone. The operators are
    shared and never written; that is what `node_apply` is for.

    Args:
        ops: The line of operators. Only the first `lead` are used.
        lead: How many leading operators to run. At least one.
        taken: The batch. Every element is moved out.

    Returns:
        One slot per input chunk, empty where the prefix kept no rows.

    Raises:
        If any operator raises.
    """
    var count = len(taken)
    var made = List[Optional[Chunk]](capacity=count)
    for _ in range(count):
        made.append(Optional[Chunk]())

    def head(at: Int) raises {mut taken, mut made, imm}:
        var current = taken[at].take()
        for i in range(lead):
            var out = node_apply(ops[i], current^)
            if not out:
                return
            current = out.take()
        made[at] = Optional[Chunk](current^)

    parallel_for(head, count)
    return made^


struct Pipeline(Movable):
    """A source, a line of operators and a sink, run by pushing chunks."""

    var source: Scan
    """Where the chunks come from."""

    var operators: List[Node]
    """The operators, in the order a chunk goes through them."""

    var schema: Schema
    """The schema of what comes out, tracked as operators are added."""

    def __init__(out self, var frame: DataFrame) raises:
        """Constructs a pipeline that reads a frame and does nothing to it.

        Args:
            frame: The input. Consumed.

        Raises:
            If the frame's columns are not chunked the same way.
        """
        self.schema = Schema(copy=frame.schema)
        self.source = Scan(frame^)
        self.operators = List[Node]()

    def add(mut self, var node: Node) raises:
        """Puts an operator at the end of the line.

        The node is told what its input looks like as it goes in, and says what
        its output looks like, so the schema is known before the first row moves
        and a node that needs it has it.

        Args:
            node: The operator. Consumed.

        Raises:
            If the node cannot accept the schema it is being given.
        """
        self.schema = node_bind(node, Schema(copy=self.schema))
        self.operators.append(node^)

    def cut_points(self) -> List[Int]:
        """Returns the operator positions where the pipeline is cut.

        A breaker ends a stage, because nothing after it can start until
        everything before it has finished. This is one traversal of the list and
        it is for looking at a plan; the driver does not need it, since draining
        the operators in order is the same execution.

        Returns:
            The position of each breaker, ascending. Empty for a pipeline that
            runs in one stage.
        """
        var cuts = List[Int]()
        for i in range(len(self.operators)):
            if node_is_breaker(self.operators[i]):
                cuts.append(i)
        return cuts^

    def stages(self) -> Int:
        """Returns how many stages the pipeline runs in.

        Returns:
            One more than the number of breakers.
        """
        return len(self.cut_points()) + 1

    def run(deinit self) raises -> DataFrame:
        """Runs the pipeline and returns the result.

        Returns:
            The frame the sink collected, with the schema the operators
            produced.

        Raises:
            If any operator raises.
        """
        var sink = Collect()
        var lead = self._parallel_lead()
        if lead > 0:
            self._run_batched(lead, sink)
        else:
            while True:
                if self._finished():
                    break
                var chunk = self.source.next()
                if not chunk:
                    break
                self._push(0, chunk.take(), sink)
        for i in range(len(self.operators)):
            while True:
                var out = node_finish(self.operators[i])
                if not out:
                    break
                self._push(i + 1, out.take(), sink)
        return sink^.into_frame(self.schema^)

    def _parallel_lead(self) raises -> Int:
        """Returns how many operators at the front of the line run in parallel.

        Zero means the whole pipeline runs on the calling thread, which is the
        answer when the first operator carries state, when there is a limit
        anywhere in the line, when there is only one chunk to run, when the
        runtime has one worker, or when the prefix is not worth a task.

        Returns:
            The number of leading operators to run on every core, or zero.
        """
        if worker_count() < 2 or self.source.num_chunks() < 2:
            return 0
        for i in range(len(self.operators)):
            if node_ends_early(self.operators[i]):
                return 0
        var lead = 0
        while lead < len(self.operators) and node_is_row_local(
            self.operators[lead]
        ):
            lead += 1

        # A task costs about ten microseconds to create and the creating
        # happens on this thread, one after another, so the prefix has to have
        # something for the other cores to do or the tasks are all that is
        # added. A project or a cast on its own does not: measured on the
        # i9-13900K a project over sixty four chunks took twice as long spread
        # out as it did in a line, and over eight chunks it was still slower.
        for i in range(lead):
            if node_computes_per_row(self.operators[i]):
                return lead
        return 0

    def _run_batched(mut self, lead: Int, mut sink: Collect) raises:
        """Runs the source through the parallel prefix and then the rest.

        One batch is one chunk per worker. The batch is read off the source,
        every chunk of it goes through the first `lead` operators on a core of
        its own, and then the survivors are pushed through the operators after
        the prefix in the order the source handed them out. Chunk order is what
        makes this the same execution as the sequential driver rather than an
        approximation of it.

        Args:
            lead: How many leading operators to run in parallel. At least one.
            sink: Where the rows that reach the end go.

        Raises:
            If any operator raises.
        """
        # No `_finished` check anywhere in here. This route is only taken when
        # no operator can end early, which is what `_parallel_lead` checked, so
        # the answer would be False every time it was asked.
        var batch = worker_count()
        while True:
            var taken = List[Optional[Chunk]](capacity=batch)
            for _ in range(batch):
                var chunk = self.source.next()
                if not chunk:
                    break
                taken.append(Optional[Chunk](chunk.take()))
            var count = len(taken)
            if count == 0:
                return

            var made = _run_head(self.operators, lead, taken)

            for at in range(count):
                if not made[at]:
                    continue
                self._push(lead, made[at].take(), sink)

    def _finished(self) -> Bool:
        """Reports whether reading more of the source would be wasted work.

        A pipeline is a line, so one finished operator makes everything before
        it useless: whatever those produce dies at the finished one. That is
        limit pushdown, and it falls out of asking rather than out of a rule.

        Returns:
            True if any operator will not emit another row.
        """
        for i in range(len(self.operators)):
            if node_status(self.operators[i]) == NodeStatus.FINISHED:
                return True
        return False

    def _push(mut self, start: Int, var chunk: Chunk, mut sink: Collect) raises:
        """Pushes one chunk through the operators from a position onwards.

        Args:
            start: The first operator to hand it to.
            chunk: The chunk. Consumed.
            sink: Where it goes if it survives to the end.

        Raises:
            If any operator raises.
        """
        var current = chunk^
        for i in range(start, len(self.operators)):
            var out = node_process(self.operators[i], current^)
            if not out:
                return
            current = out.take()
        sink.push(current^)
