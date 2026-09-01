"""Handing work out in small pieces, to whichever core asks next.

`parallel_for` next door cuts a job into one piece per index and starts a task
for each. That is the right shape when the pieces cost the same, and the wrong
one whenever they do not. A file of ten blocks where one block holds the long
strings finishes when that block finishes, and the other nine cores wait. A
group by where one partition holds half the keys is the same picture. The
scheduler cannot fix it because the split was decided before any work was done.

A morsel queue decides nothing in advance. The work is one range of rows, the
range is cut into fixed size morsels, and a worker that finishes one takes the
next by bumping a shared counter. A worker that draws a cheap morsel comes back
sooner and takes more of them. The tail is one morsel long instead of one block
long, which is the whole point: the slowest worker can only ever be one morsel
behind the others rather than one nth of the job.

The counter is the only thing shared, it is bumped with an atomic add, and the
add happens once per morsel rather than once per row. At the default morsel of a
hundred and twenty eight thousand rows a scan does one atomic per eight hundred
kilobytes of data, which is not a number worth thinking about.

What this is not is a work stealing pool. There are no per worker deques and
nothing is ever taken from another worker, because a single counter is enough
when the work is a range of rows and every morsel is drawn from the same place.
Stealing is for trees of tasks, and a dataframe engine has ranges.

Sizing the morsel is a trade with two ends and a wide floor between them. Too
large and the tail comes back: a morsel is the granularity of imbalance. Too
small and the atomic starts to matter, and worse, the morsel stops fitting in L2
and the per morsel setup that every kernel does stops being amortised. DuckDB
uses a hundred thousand or so and Velox uses ten thousand; a hundred and twenty
eight thousand is DuckDB's number rounded to a power of two and matches the chunk
size in `docs/specs/engine/02-execution-model.md`, so a morsel is one chunk and a
kernel that walks a morsel is walking a chunk.
"""

from std.atomic import Atomic
from std.runtime.asyncrt import TaskGroup

from .parallel import worker_count

comptime MORSEL_ROWS = 128 * 1024
"""Rows in a morsel, unless a caller says otherwise.

One chunk's worth, so a morsel and a chunk are the same thing and a kernel that
was written to walk one walks the other.
"""


@fieldwise_init
struct Morsel(Boolable, ImplicitlyCopyable, Movable, Sized):
    """A half open range of rows for one worker to do."""

    var start: Int
    """The first row."""

    var end: Int
    """One past the last row."""

    def __bool__(self) -> Bool:
        """Whether there is anything to do.

        Returns:
            False for the empty morsel a drained queue hands back.
        """
        return self.start < self.end

    def __len__(self) -> Int:
        """Returns how many rows the morsel covers.

        Returns:
            The row count, zero for the empty morsel.
        """
        return self.end - self.start if self.end > self.start else 0


struct MorselQueue:
    """A range of rows that hands itself out a morsel at a time.

    Not `Movable`, deliberately. Every worker reaches the same queue through a
    pointer and moving it under them would be moving the counter they are adding
    to. It is made once, on the frame that runs the job, and lives until that
    job is done.
    """

    var cursor: Atomic[DType.int64]
    """How many rows have been handed out. Only ever increases."""

    var total: Int
    """How many rows there are."""

    var rows: Int
    """How many rows a morsel holds, except the last."""

    def __init__(out self, total: Int, rows: Int):
        """Constructs a queue over a range of rows.

        Args:
            total: How many rows there are.
            rows: How many rows a morsel holds. Clamped to at least one, because
                a morsel of zero rows is a queue that never drains.
        """
        self.cursor = Atomic[DType.int64](0)
        self.total = total if total > 0 else 0
        self.rows = rows if rows > 0 else 1

    def take(mut self) -> Morsel:
        """Takes the next morsel, or an empty one when there are none left.

        Safe to call from every worker at once. That is the only reason this
        type exists.

        Returns:
            The morsel. Test it with `if morsel:` rather than comparing bounds.
        """
        var at = Int(self.cursor.fetch_add(Int64(self.rows)))
        if at >= self.total:
            # The counter runs past the end and is left there. Clamping it would
            # mean a compare and swap on a value nobody reads.
            return Morsel(self.total, self.total)
        var end = at + self.rows
        return Morsel(at, end if end < self.total else self.total)

    def morsels(self) -> Int:
        """Returns how many morsels the queue will hand out in total.

        This is a division rather than anything the queue remembers, and it is
        here for callers sizing a per morsel result list up front.

        Returns:
            The morsel count.
        """
        return (self.total + self.rows - 1) // self.rows


async def _drain[
    F: def(Int, Int) raises -> None
](
    body: F,
    queue: MutPointer[MorselQueue, MutUntrackedOrigin],
    worker: Int,
    failures: MutPointer[List[String], MutUntrackedOrigin],
):
    """Takes morsels until there are none, running the body on each.

    Takes the body as an argument rather than closing over it, for the reason
    `parallel.mojo` gives at length: a coroutine that captures a closure crashes
    this toolchain with no diagnostic.

    Args:
        body: What to run on each morsel.
        queue: The shared queue.
        worker: Which worker this is, and which failure slot is ours.
        failures: One slot per worker, written only at this worker's index.

    Parameters:
        F: The body's type.
    """
    while True:
        var morsel = queue[].take()
        if not morsel:
            return
        try:
            body(morsel.start, morsel.end)
        except e:
            # Stop at the first failure in this worker but let the others drain,
            # because the alternative is returning while another worker is still
            # reading state this one's caller is about to drop.
            failures[][worker] = String(e)
            return


def parallel_morsels[
    F: def(Int, Int) raises -> None
](body: F, total: Int, rows: Int = MORSEL_ROWS) raises:
    """Runs `body(start, end)` over every row, in morsels, on every core.

    Starts one task per worker rather than one per morsel. A thousand morsels is
    a thousand ranges to hand out and should not be a thousand tasks to create,
    and the whole point of the queue is that a worker asks for more work rather
    than being given it up front.

    A job that fits in one morsel runs inline, for the same reason a
    `parallel_for` of one does: it keeps a small job out of the runtime, so the
    error it raises is raised from the caller's own stack.

    Args:
        body: What to run on each morsel. It may raise.
        total: How many rows there are.
        rows: How many rows a morsel holds.

    Raises:
        Error: The first failure, by worker, if any morsel failed. Workers that
            had already started still drain first.

    Parameters:
        F: The body's type.
    """
    if total <= 0:
        return
    if total <= rows:
        body(0, total)
        return

    var workers = worker_count()
    var queue = MorselQueue(total, rows)
    var wanted = queue.morsels()
    if wanted < workers:
        # More workers than morsels is workers that wake up, find the queue
        # empty and go back to sleep.
        workers = wanted

    var failures = List[String](length=workers, fill=String())
    var slots = Pointer(to=failures).unsafe_origin_cast[MutUntrackedOrigin]()
    var shared = Pointer(to=queue).unsafe_origin_cast[MutUntrackedOrigin]()
    var group = TaskGroup()
    for worker in range(workers):
        group.create_task(_drain[F](body, shared, worker, slots))
    group.wait()

    for worker in range(workers):
        if failures[worker]:
            raise Error(failures[worker])
    _ = queue^
