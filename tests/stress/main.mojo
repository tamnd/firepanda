"""Concurrency and allocation stress.

At M0 there are no parallel operators to stress, so this harness stresses the
thing that has to be right before there can be any: building and tearing down
columns from several threads at once. Every worker owns its own buffers and
touches nothing another worker touches, which is the discipline the whole engine
is going to run on, and the point of the harness is to notice the day something
in the allocation path stops honouring it.

Each round picks a random number of chunks, a random length per chunk, and a
random worker count, builds every chunk in parallel, and compares the per-chunk
checksums against the same chunks built one at a time on this thread. The chunk
contents come from a generator seeded by the chunk index, so the answer does not
depend on the order the workers ran in, and a mismatch is a real disagreement
rather than a scheduling artifact.

The second half of each round churns the buffer pool: take and give buffers at
random sizes and confirm that every buffer handed out is aligned and zeroed. The
pool is deliberately not shared between threads and not guarded by a lock, so it
is exercised on one thread only; making it global and atomic is the design this
project is trying not to have.

Usage:
    mojo run -I . tests/stress/main.mojo [--rounds=N] [--seed=N] [--max-total-time=SECONDS]
"""

from max.algorithm import parallelize
from std.sys import argv
from std.time import perf_counter_ns

from firepanda.array.array import Array
from firepanda.buffer.buffer import Buffer
from firepanda.buffer.pool import BufferPool
from firepanda.testing.rng import Rng

comptime DEFAULT_ROUNDS = 200
"""Enough rounds to see a few hundred distinct shapes without taking a minute."""

comptime MAX_CHUNKS = 64
comptime MAX_CHUNK_LENGTH = 20_000


def checksum_of_chunk(index: Int, length: Int) -> Int64:
    """Builds one chunk and returns a checksum over its live values.

    The chunk is derived from its index alone, so this is a pure function of
    `(index, length)` and gives the same answer on a worker thread and on the
    main thread.

    Args:
        index: The chunk index, which seeds the contents.
        length: The number of rows.

    Returns:
        The sum of the values at positions that are not null.
    """
    var rng = Rng(UInt64(0x9E3779B97F4A7C15) * UInt64(index + 1))
    var column = Array[DType.int64](length)
    for i in range(length):
        column[i] = Int64(rng.next_below(1_000_000))
        if rng.next_below(8) == 0:
            column.set_null(i)

    var total = Int64(0)
    for i in range(length):
        if column.is_valid(i):
            total += column[i]
    return total


struct Options(Copyable, Movable):
    """What the harness was asked to do."""

    var rounds: Int
    """The number of rounds to run."""

    var seed: UInt64
    """The generator seed for the shapes, printed so a failure can be replayed."""

    var max_seconds: Float64
    """A wall clock budget. Zero means no budget."""

    def __init__(out self):
        """Constructs the defaults."""
        self.rounds = DEFAULT_ROUNDS
        self.seed = 0xB5026F5AA96619E9
        self.max_seconds = 0.0


def parse_options() raises -> Options:
    """Reads the command line.

    Returns:
        The options, with anything unspecified left at its default.

    Raises:
        If a flag is not recognized or its value is not a number.
    """
    var options = Options()
    var args = argv()
    for i in range(1, len(args)):
        var arg = args[i]
        if arg.startswith("--rounds="):
            options.rounds = Int(arg[byte=9:])
        elif arg.startswith("--seed="):
            options.seed = UInt64(Int(arg[byte=7:]))
        elif arg.startswith("--max-total-time="):
            options.max_seconds = Float64(Int(arg[byte=17:]))
        else:
            raise Error(String("unrecognized argument: ", arg))
    return options^


def churn_the_pool(mut rng: Rng, mut pool: BufferPool, operations: Int) raises:
    """Takes and gives buffers at random sizes, checking the invariants each time.

    Args:
        rng: The generator.
        pool: The pool under test.
        operations: The number of take or give operations to perform.

    Raises:
        If a buffer comes back misaligned, short, or holding old bytes.
    """
    var held = List[Buffer]()
    for _ in range(operations):
        if len(held) > 0 and rng.next_below(3) == 0:
            # Give back a buffer other than the most recent one, so the pool sees
            # a use order that is not a stack.
            var index = rng.next_below(len(held))
            held.swap_elements(index, len(held) - 1)
            pool.give(held.pop())
            continue

        var size = rng.next_range(1, 200_000)
        var buffer = pool.take(size)
        if not buffer.is_aligned():
            raise Error(
                String("pool returned a misaligned buffer of size ", size)
            )
        if len(buffer) != size:
            raise Error(
                String(
                    "pool returned ",
                    len(buffer),
                    " bytes for a request of ",
                    size,
                )
            )
        for i in range(0, len(buffer), 997):
            if buffer.unsafe_ptr().unsafe_offset(i).unsafe_load() != UInt8(0):
                raise Error(
                    String(
                        "pool returned a dirty buffer of size ",
                        size,
                        " at byte ",
                        i,
                    )
                )
        # Dirty it so that the next taker would see the bytes if the pool ever
        # stopped zeroing.
        for i in range(0, len(buffer), 97):
            buffer.unsafe_ptr().unsafe_offset(i).unsafe_write(UInt8(0xCD))
        held.append(buffer^)

    while len(held) > 0:
        pool.give(held.pop())


def main() raises:
    var options = parse_options()
    print(
        "stressing parallel column construction:",
        options.rounds,
        "rounds, seed",
        options.seed,
        "max_seconds",
        options.max_seconds,
    )

    var rng = Rng(options.seed)
    var pool = BufferPool()
    var started = perf_counter_ns()
    var completed = 0
    var rows = 0

    for round in range(options.rounds):
        if options.max_seconds > 0.0:
            var elapsed = Float64(perf_counter_ns() - started) / 1.0e9
            if elapsed >= options.max_seconds:
                print("stopped on the time budget after", completed, "rounds")
                break

        var chunks = rng.next_range(2, MAX_CHUNKS + 1)
        var workers = rng.next_range(1, 17)
        var lengths = List[Int]()
        for _ in range(chunks):
            lengths.append(rng.next_range(1, MAX_CHUNK_LENGTH))

        var parallel_results = List[Int64](length=chunks, fill=0)
        var results_ptr = parallel_results.unsafe_ptr()
        var lengths_ptr = lengths.unsafe_ptr()

        def build(i: Int) {imm results_ptr, imm lengths_ptr}:
            var length = lengths_ptr.unsafe_offset(i).unsafe_load()
            results_ptr.unsafe_offset(i).unsafe_write(
                checksum_of_chunk(i, length)
            )

        parallelize(build, chunks, workers)

        for i in range(chunks):
            var expected = checksum_of_chunk(i, lengths[i])
            if parallel_results[i] != expected:
                raise Error(
                    String(
                        "round ",
                        round,
                        " seed ",
                        options.seed,
                        ": chunk ",
                        i,
                        " of length ",
                        lengths[i],
                        " came back ",
                        parallel_results[i],
                        " but single threaded gives ",
                        expected,
                        " (",
                        chunks,
                        " chunks, ",
                        workers,
                        " workers)",
                    )
                )
            rows += lengths[i]

        churn_the_pool(rng, pool, 200)
        completed += 1

    var seconds = Float64(perf_counter_ns() - started) / 1.0e9
    print(
        "ok:",
        completed,
        "rounds,",
        rows,
        "rows built twice each, in",
        seconds,
        "s",
    )
    print("pool hits", pool.hits(), "misses", pool.misses())
