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
a few chunks per worker. That is what keeps the engine streaming. Running every
chunk through the prefix first and then draining would hold the whole
intermediate result in memory, which is what a chunked engine exists to avoid;
a batch holds a fixed number of chunks per core and no more. Inside a batch the
chunks are handed out by the morsel queue, one at a time to whichever worker
asks next, so the batch can be wider than the machine without costing a task per
chunk and a slow chunk costs the batch its own length rather than a worker's
whole share.

A pipeline containing a `Limit` is fed one chunk at a time as before. Reading
ahead on behalf of thirty two cores is reading rows that the limit was about to
make unnecessary, and a `head` over a large file is the case where that is the
whole cost of the query.

What is still not here is a parallel breaker. A group by merges every chunk into
one table on the calling thread, so a query whose work is in the grouping gets
the prefix in parallel and the grouping serial. Thread local tables merged
partition wise is the next step and it is a change to `Group`, not to this file.
"""

from firepanda.array.any import AnyArray, empty_any
from firepanda.array.chunked import ChunkedArray
from firepanda.dtype.schema import Schema
from firepanda.frame.frame import DataFrame

from .chunk import Chunk
from .morsel import MORSEL_ROWS, parallel_morsels
from .node import Node, NodeStatus, Reduce, node_apply, node_bind
from .node import node_computes_per_row, node_ends_early, node_finish
from .node import mark_chained_filters, node_is_breaker
from .node import node_is_row_local, node_process
from .node import node_status
from .parallel import worker_count

comptime BATCH_CHUNKS_PER_WORKER = 4
"""How many chunks of a batch one worker is expected to get through.

The batch has to be at least as wide as the machine or cores sit idle, and it
cannot be the whole source or the engine stops streaming. Wider than one chunk
per worker is worth having because the tasks are started once per batch and a
batch of four chunks each pays for them a quarter as often. Four rather than
forty because a batch is held in memory all at once, and at a hundred and twenty
eight thousand rows a chunk, four per worker on a thirty two thread machine is
already sixteen million rows in flight.
"""


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

    A chunk taller than a morsel can be cut into morsels, but only when the
    pipeline above asks for it by calling `cut`. Every reader hands back a frame
    in one chunk, because that is what an eager caller wants, and a line over
    such a frame used to run about 1.65 times slower than the same rows in
    chunks: with one chunk there is nothing for `_parallel_lead` to hand out, so
    the whole line runs on the calling thread and every operator forks and joins
    its own workers instead of the prefix forking once. Measured on the
    i9-13900K at four million rows, one chunk was 3.46 milliseconds and two
    chunks was 2.42, and the whole of the difference is that one step. See #800.

    The asking is the point. A line whose front is a group by or a reduction has
    no prefix to hand out, so the morsels are pushed through it one after
    another on the calling thread, and the kernel that used to be given a
    million rows and spread them itself is given a hundred and twenty eight
    thousand eight times over, each under whatever width it splits at. That cost
    two to four times on every ClickBench query with a group by in it, which is
    #918.
    """

    var columns: List[List[AnyArray]]
    """Per column, the chunks still to be handed out, in reverse order."""

    var remaining: Int
    """The number of chunks left."""

    var cuttable: Bool
    """Whether `cut` would do anything, which is false for a nested column."""

    def __init__(out self, var frame: DataFrame) raises:
        """Constructs a scan over a frame, consuming it.

        Args:
            frame: The frame. Consumed.

        Raises:
            If the columns are not chunked the same way.
        """
        var owned = frame^.into_columns()
        # A nested column is the one shape a window cannot be taken of, and the
        # decision has to be made for the frame rather than per column, since
        # cutting the others would leave the frame chunked differently from
        # column to column and the check below would refuse it.
        var cuttable = True
        for i in range(len(owned)):
            if owned[i].type.is_nested():
                cuttable = False
                break
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
        self.cuttable = cuttable

    def cut(mut self) raises:
        """Cuts every chunk taller than a morsel into morsels.

        The pieces are windows rather than copies, so this costs nothing. They
        share the chunk's buffers and go private only if something writes to
        them, and cutting on whole morsels is what keeps every kernel's right to
        read a register past the end of a column, which `AnyArray.window` sets
        out. Cutting with `slice` instead was measured too and it is far worse
        than leaving the frame alone, 11.9 milliseconds against 3.46, because a
        copy of the source costs more than the query.

        A frame holding a list or a struct column is left exactly as it arrived,
        because a nested column is the one shape a window cannot be taken of and
        cutting the rest would leave the columns chunked differently from each
        other.

        Raises:
            If a window cannot be taken of a chunk.
        """
        if not self.cuttable:
            return
        var flipped = List[List[AnyArray]](capacity=len(self.columns))
        while len(self.columns) > 0:
            var backwards = self.columns.pop()
            var forwards = List[AnyArray](capacity=len(backwards))
            while len(backwards) > 0:
                var chunk = backwards.pop()
                var rows = len(chunk)
                if rows <= MORSEL_ROWS:
                    forwards.append(chunk^)
                    continue
                var pieces = (rows + MORSEL_ROWS - 1) // MORSEL_ROWS
                for p in range(pieces):
                    var at = p * MORSEL_ROWS
                    var take = rows - at
                    if take > MORSEL_ROWS:
                        take = MORSEL_ROWS
                    forwards.append(chunk.window(at, take))
            # Back to front, because `next` takes from the back of the list.
            var cut = List[AnyArray](capacity=len(forwards))
            while len(forwards) > 0:
                cut.append(forwards.pop())
            flipped.append(cut^)
        var columns = List[List[AnyArray]](capacity=len(flipped))
        while len(flipped) > 0:
            columns.append(flipped.pop())
        self.remaining = 0 if len(columns) == 0 else len(columns[0])
        self.columns = columns^

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
            var piece = self.columns[i].pop()
            # A column held encoded is decoded here, one morsel at a time,
            # because none of the operators reads an encoding yet and every one
            # of them would refuse it. Decoding the frame instead would put the
            # flat copy back in memory for the whole query, which is the thing
            # the encoding is there to avoid; a morsel's worth is a few
            # megabytes and is gone when the morsel is. An operator taught to
            # read codes is what removes this, one operator at a time.
            if not piece.is_flat():
                piece = piece.decoded()
            row.append(piece^)
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

        Each of those columns holds one chunk of no rows rather than no chunks.
        A column of no chunks is the same zero rows and reads as a column that
        was never built: `DataFrame.__getitem__` is `ChunkedArray.only`, which
        wants exactly one, so a frame made the other way is one nothing can read
        a column out of and a predicate that happened to keep nothing would
        raise where it should answer.

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
                # Built from the chunk rather than appended to, because
                # `append` drops a chunk of no rows to keep two chunks from
                # starting at the same row, and here there is no second chunk
                # and no row to start.
                empty.append(ChunkedArray(empty_any(schema[i].dtype)))
            return DataFrame(schema^, empty^)
        if len(self.columns) != len(schema):
            raise Error(
                "collect: the result has "
                + String(len(self.columns))
                + " columns and the schema has "
                + String(len(schema))
            )
        return DataFrame(schema^, self.columns^)


def _apply_prefix(
    ops: List[Node], lead: Int, var chunk: Chunk
) raises -> Optional[Chunk]:
    """Runs one chunk through the first `lead` operators.

    A selection stops here. A filter writes one rather than copying every
    column, and whatever reads the column next gathers the rows out of it, so if
    the last operator of the prefix was the filter the gather would otherwise
    fall to the sink, which runs on this thread after the batch is done. That
    turns work that was spread over every core into work done one chunk at a
    time in the serial part of the run, which is the opposite of what writing
    the selection was for.

    Args:
        ops: The line of operators. Only the first `lead` are used.
        lead: How many leading operators to run. At least one.
        chunk: The chunk, consumed.

    Returns:
        What came out, or nothing if an operator kept no rows.

    Raises:
        If any operator raises.
    """
    for i in range(lead):
        var out = node_apply(ops[i], chunk^)
        if not out:
            return None
        chunk = out.take()
    if chunk.selected():
        # False because this is a worker and the batch above it is already
        # using the machine.
        chunk.flatten(False)
    return Optional[Chunk](chunk^)


def _run_head(
    ops: List[Node],
    lead: Int,
    fold_at: Int,
    mut taken: List[Optional[Chunk]],
) raises -> List[Optional[Chunk]]:
    """Runs one batch of chunks through the first `lead` operators, in parallel.

    A free function rather than a method so that the body handed to the
    scheduler captures the two lists and the operators and nothing else. A
    closure written inside `Pipeline` would capture the whole pipeline, and the
    source and the sink have no business being reachable from a worker.

    `taken` and `made` are read and written at one index per chunk, and both
    lists are their final length before the first body runs, so no two workers
    touch the same element and nothing moves under anyone. The operators are
    shared and never written; that is what `node_apply` is for.

    The chunks are handed out a morsel of one at a time rather than given out
    one task each. A task costs about ten microseconds to create and the
    creating is serial on this thread, so a batch of a hundred and twenty eight
    chunks would spend more than a millisecond starting tasks before any of them
    ran. Through the queue it starts one task per worker whatever the batch
    holds, and a chunk that turns out to be expensive costs the batch one chunk
    of tail rather than one worker's whole share.

    A reduction sitting immediately behind the prefix is folded here too, which
    is what `fold_at` is. A fold of one chunk is a reduction of a hundred and
    twenty eight thousand rows and a morsel is the same size, so the kernel
    takes its serial route and there is no parallelism inside one fold to have.
    Across a batch there is: each chunk becomes a one row partial on the core
    that produced the chunk, while it is still in that core's cache, and the
    caller merges the partials afterwards, which is a pass over one row per
    chunk. Merging is left to the caller because the running row is the node's
    and a worker has no business writing it.

    Args:
        ops: The line of operators. Only the first `lead` are used, plus the
            one at `fold_at` if there is one.
        lead: How many leading operators to run. At least one.
        fold_at: The position of a `Reduce` to fold each chunk into a partial
            row with, or -1 for none.
        taken: The batch. Every element is moved out.

    Returns:
        One slot per input chunk, empty where the prefix kept no rows, holding
        a partial row rather than the chunk when `fold_at` is set.

    Raises:
        If any operator raises.
    """
    var count = len(taken)
    var made = List[Optional[Chunk]](capacity=count)
    for _ in range(count):
        made.append(Optional[Chunk]())

    def head(start: Int, stop: Int) raises {mut taken, mut made, imm}:
        for at in range(start, stop):
            var out = _apply_prefix(ops, lead, taken[at].take())
            if not out:
                continue
            if fold_at < 0:
                made[at] = Optional[Chunk](out.take())
                continue
            var row = ops[fold_at][Reduce].partial(out.take())
            if row:
                made[at] = Optional[Chunk](row.take())

    parallel_morsels(head, count, 1)
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
        # A filter with another filter above it writes a selection whatever the
        # share it keeps, because the copy it would make is one the next filter
        # makes again. Here rather than at `add`, because a node is built before
        # the one above it exists.
        mark_chained_filters(self.operators)
        # The source is cut into morsels here rather than when it was built,
        # because a morsel is only worth having when there is a prefix to hand
        # out over it. This is the first point where both halves of that are
        # known: the scan was built before `add` was called, and the line is
        # complete now.
        if worker_count() > 1 and self._prefix_lead() > 0:
            self.source.cut()
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
        answer when the first operator carries state, when there is a limit that
        can still stop the source, when there is only one chunk to run, when the
        runtime has one worker, or when the prefix is not worth a task.

        A limit only stops the source while rows are still reaching it. Put a
        breaker under one and they are not: a sort or a group by holds every row
        it is given and emits nothing until the source has run out, so the limit
        above it counts its first row after the last chunk has been read. The
        search for a limit therefore stops at the first breaker, and the rows
        below that breaker are read on every core whatever sits above it.

        `SELECT * FROM hits WHERE URL LIKE '%google%' ORDER BY EventTime LIMIT
        10` is the query that made this worth writing down. Ninety five rows out
        of a million survive the filter and ten of those are the answer, so
        almost the whole query is the search down the million, and it was running
        on one core because of a limit that could not have stopped it. At 1M the
        same statement without the `LIMIT 10` was 11.5 milliseconds and with it
        was 23.8, on the same rows through the same operators. With this it is
        13.6. See #682 and document 98.

        Returns:
            The number of leading operators to run on every core, or zero.
        """
        if worker_count() < 2 or self.source.num_chunks() < 2:
            return 0
        return self._prefix_lead()

    def _prefix_lead(self) raises -> Int:
        """Returns the same as `_parallel_lead` for the shape of the line alone.

        What is left out is how many chunks there are and how many workers the
        machine has. `run` asks this before the source has been cut into
        morsels, when the chunk count is still whatever the reader handed back
        and is therefore not the count the decision is about.

        Returns:
            The number of leading operators worth handing out, or zero.
        """
        for i in range(len(self.operators)):
            if node_is_breaker(self.operators[i]):
                break
            if node_ends_early(self.operators[i]):
                return 0
        var lead = 0
        while lead < len(self.operators) and node_is_row_local(
            self.operators[lead]
        ):
            lead += 1

        # Starting the workers costs about ten microseconds a task and the
        # starting is serial on this thread, so a prefix has to have something
        # for the other cores to do before it is worth handing out at all. A
        # project or a cast does not: measured on the i9-13900K, a project over
        # sixty four chunks was forty per cent slower spread out even with the
        # tasks down to one per worker, because a project only rebuilds a chunk
        # out of columns it already has and a cast walks a column through the
        # allocator, and neither is waiting on arithmetic.
        for i in range(lead):
            if node_computes_per_row(self.operators[i]):
                return lead
        return 0

    def _run_batched(mut self, lead: Int, mut sink: Collect) raises:
        """Runs the source through the parallel prefix and then the rest.

        One batch is `BATCH_CHUNKS_PER_WORKER` chunks per worker. The batch is
        read off the source, every chunk of it goes through the first `lead`
        operators on whichever core asks for it, and then the survivors are
        pushed through the operators after the prefix in the order the source
        handed them out. Chunk order is what makes this the same execution as
        the sequential driver rather than an approximation of it.

        Args:
            lead: How many leading operators to run in parallel. At least one.
            sink: Where the rows that reach the end go.

        Raises:
            If any operator raises.
        """
        # No `_finished` check anywhere in here. This route is only taken when
        # nothing can end early while the source is being read, which is what
        # `_parallel_lead` checked, so the answer would be False every time it
        # was asked. A limit beyond a breaker is the one that gets past that
        # check, and it has emitted no row and cannot report FINISHED until the
        # breaker under it has seen the last chunk, which is after this returns.
        var batch = worker_count() * BATCH_CHUNKS_PER_WORKER
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

            var fold_at = self._fold_at(lead)
            var made = _run_head(self.operators, lead, fold_at, taken)

            for at in range(count):
                if not made[at]:
                    continue
                if fold_at < 0:
                    self._push(lead, made[at].take(), sink)
                else:
                    self.operators[fold_at][Reduce].absorb(made[at].take())

    def _fold_at(self, lead: Int) -> Int:
        """Returns where a reduction the prefix can fold into sits, or -1.

        It has to be the operator immediately behind the prefix, because that is
        the only position where the chunk a worker is holding is the chunk the
        reduction would be given.

        Args:
            lead: How many leading operators run in parallel.

        Returns:
            The position of that `Reduce`, or -1 if there is not one.
        """
        if lead >= len(self.operators):
            return -1
        if not self.operators[lead].isa[Reduce]():
            return -1
        return lead

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
