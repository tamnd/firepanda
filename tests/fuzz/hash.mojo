"""Differential fuzzing of factorize against its linear twin.

Same arrangement as `kernel.mojo`: the fast thing is compared against the slow
thing anybody would have written first, which is `firepanda/hash/scalar.mojo`.

What the generator is built to hit here is different from the kernel fuzzer,
because what can go wrong is different. A kernel is wrong at the edge of a
register. A hash table is wrong at the edge of a growth, or on a key the table
confuses with an empty slot, or on a value distribution that pushes the choice
between the direct route and the hashed route across its boundary.

So the value range is drawn per case rather than fixed. A narrow range takes the
direct route, a wide one takes the hashed route, and a range near `DIRECT_LIMIT`
lands on the boundary itself. Every case also checks that the two routes agree
where both apply, which is the property that makes the route choice an
optimization rather than a behavior.

Group counts are drawn to straddle the table's growth points, because a table
that loses a key during a rehash produces a duplicate group rather than a crash
and nothing else in the system will notice.

The float cases are separate and deliberate. NaN and negative zero are the two
values where the right answer disagrees with `==`, and a generator that only
draws ordinary floats will never produce either.

The parallel case is separate for the opposite reason: it needs a column long
enough to be cut into slices, which is thirty times longer than anything else
here draws, and it compares the two hashed routes against each other rather than
against the twin because what it is looking for is a merge that renumbers
correctly-partitioned rows in the wrong order.

Usage:
    mojo run -I . tests/fuzz/hash.mojo [--cases=N] [--seed=N] [--max-total-time=SECONDS]
"""

from std.sys import argv
from std.time import perf_counter_ns

from firepanda.array.array import Array
from firepanda.buffer.buffer import Buffer
from firepanda.hash import (
    DIRECT_LIMIT,
    HashTable,
    factorize,
    factorize_linear,
    hash_of,
    key_bits,
    mix,
    radix_partition,
)
from firepanda.hash.factorize import (
    _factorize_hashed_parallel,
    _factorize_hashed_serial,
)
from firepanda.testing.rng import Rng

comptime DEFAULT_CASES = 2_000_000
"""A case is a column of up to three hundred rows factorized twice, once
quadratically, or a table hammered directly, or a partitioning checked for being
a permutation. The default runs in about ten seconds."""

comptime MAX_LENGTH = 307
"""A prime, so the lengths land on every offset modulo the register width."""

comptime SPREAD_TIGHT = 0
"""Values in a range far smaller than the column. Many rows per group, direct
route, and the case where the table is small and hot."""

comptime SPREAD_WIDE = 1
"""Values spread wider than the column. Close to one group per row, which is the
case that grows the table the most times."""

comptime SPREAD_BOUNDARY = 2
"""Values in a range straddling `DIRECT_LIMIT`, so that neighbouring cases take
different routes for reasons the generator does not control."""

comptime PARALLEL_LENGTH = 9_000
"""Rows in the longest column the parallel check draws.

The slices a parallel build cuts land on chunk boundaries, so a column shorter
than a chunk cannot be cut into more than one piece and a check on one would be
checking nothing. Nine thousand rows is eight chunks, which is enough slices to
have a merge worth being wrong about, and still small enough that a case runs in
about a millisecond.
"""

comptime SPREAD_HUGE = 3
"""Values large enough that the direct route declines on the range check rather
than on the span."""


struct Options(Copyable, Movable):
    """What the harness was asked to do."""

    var cases: Int
    """The number of columns to factorize."""

    var seed: UInt64
    """The generator seed. Printed on every run so a failure can be replayed."""

    var max_seconds: Float64
    """A wall clock budget. Zero means no budget."""

    def __init__(out self):
        """Constructs the defaults."""
        self.cases = DEFAULT_CASES
        self.seed = 0x9E3779B97F4A7C15
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
        if arg.startswith("--cases="):
            options.cases = Int(arg[byte=8:])
        elif arg.startswith("--seed="):
            options.seed = UInt64(Int(arg[byte=7:]))
        elif arg.startswith("--max-total-time="):
            options.max_seconds = Float64(Int(arg[byte=17:]))
        else:
            raise Error(String("unrecognized argument: ", arg))
    return options^


def fail(step: Int, seed: UInt64, what: String, detail: String) raises:
    """Raises the standard failure message.

    Args:
        step: The case number.
        seed: The seed, so the run can be replayed.
        what: What disagreed.
        detail: What the disagreement was.

    Raises:
        Always.
    """
    raise Error(
        String("step ", step, " seed ", seed, " in ", what, ": ", detail)
    )


def check[
    dt: DType
](col: Array[dt], seed_for_run: UInt64, step: Int, seed: UInt64) raises:
    """Factorizes a column and checks everything that should hold of the result.

    The twin comparison is the main event. The three checks after it are
    properties the twin cannot catch, because the twin does not build a keys
    array and so cannot disagree about one.

    Args:
        col: The column.
        seed_for_run: The hash seed to factorize with.
        step: The case number.
        seed: The generator seed, for the message.

    Parameters:
        dt: The dtype.

    Raises:
        If anything disagrees.
    """
    var got = factorize(col, seed_for_run)
    var twin = factorize_linear(col)
    var n = len(col)

    if len(got.codes) != n:
        fail(step, seed, "factorize", String("code count ", len(got.codes)))

    for i in range(n):
        if got.codes[i] != twin[i]:
            fail(
                step,
                seed,
                "factorize",
                String("row ", i, " code ", got.codes[i], " twin ", twin[i]),
            )

    var keys = got.keys(col)
    for i in range(n):
        var at = Int(got.codes[i])
        if at >= got.count():
            fail(step, seed, "keys", String("row ", i, " code ", at))
        if col.is_valid(i):
            if not keys.is_valid(at):
                fail(step, seed, "keys", String("row ", i, " key is null"))
            # Compared as key bits rather than as values, because the group a
            # NaN belongs to is defined by its canonical bits and `!=` on two
            # NaNs is true however well the grouping worked.
            if key_bits(keys[at]) != key_bits(col[i]):
                fail(
                    step,
                    seed,
                    "keys",
                    String("row ", i, " key ", keys[at], " value ", col[i]),
                )
        elif at != got.null_group:
            fail(step, seed, "nulls", String("row ", i, " code ", at))

    # Every ordinal below the count has to be reachable, or the codes are not
    # dense and an aggregation indexed by them allocates slots it never fills.
    var used = List[Bool]()
    for _ in range(got.count()):
        used.append(False)
    for i in range(n):
        used[Int(got.codes[i])] = True
    for g in range(got.count()):
        if not used[g]:
            fail(step, seed, "density", String("group ", g, " unreachable"))


def random_column[
    dt: DType
](mut rng: Rng, length: Int, spread: Int, nulls: Bool) -> Array[dt]:
    """Builds a column whose value range is chosen to steer the route.

    Args:
        rng: The generator.
        length: The number of rows.
        spread: One of the `SPREAD_` constants.
        nulls: Whether to punch nulls into it.

    Parameters:
        dt: The dtype.

    Returns:
        The column.
    """
    var groups = rng.next_range(1, 8)
    var stride = 1
    if spread == SPREAD_WIDE:
        groups = rng.next_range(1, MAX_LENGTH)
    elif spread == SPREAD_BOUNDARY:
        # A span within a few of DIRECT_LIMIT, so that the route flips between
        # neighbouring cases and both sides of the branch get exercised without
        # the generator knowing which is which.
        groups = rng.next_range(1, 16)
        stride = (DIRECT_LIMIT + rng.next_range(0, 4) - 2) // groups
    elif spread == SPREAD_HUGE:
        groups = rng.next_range(1, 32)
        stride = 1 << 22

    var out = Array[dt](length)
    for i in range(length):
        out[i] = Scalar[dt](Int(rng.next_below(groups)) * stride)

    if nulls:
        for i in range(length):
            if rng.next_below(5) == 0:
                out.set_null(i)
    return out^


def random_floats(
    mut rng: Rng, length: Int, nulls: Bool
) -> Array[DType.float64]:
    """Builds a float column salted with NaN, negative zero and infinities.

    Args:
        rng: The generator.
        length: The number of rows.
        nulls: Whether to punch nulls into it.

    Returns:
        The column.
    """
    var nan = Float64(0.0) / Float64(0.0)
    var out = Array[DType.float64](length)
    for i in range(length):
        var which = rng.next_below(8)
        if which == 0:
            # A different NaN payload each time, which is the case that has to
            # collapse to one group.
            out[i] = nan * Float64(Int(rng.next_range(1, 100)))
        elif which == 1:
            out[i] = Float64(-0.0)
        elif which == 2:
            out[i] = Float64(0.0)
        elif which == 3:
            out[i] = Float64(1.0) / Float64(0.0)
        elif which == 4:
            out[i] = Float64(-1.0) / Float64(0.0)
        else:
            out[i] = Float64(Int(rng.next_range(0, 20))) * Float64(0.5)

    if nulls:
        for i in range(length):
            if rng.next_below(5) == 0:
                out.set_null(i)
    return out^


def parallel_column(
    mut rng: Rng, length: Int, nulls: Bool
) -> Array[DType.int64]:
    """Builds a column for the parallel check.

    Mostly the ordinary generator, and one case in four a column where every row
    is its own group. That is the shape where the merge does the most work and
    where a worker's table grows the most times, and the ordinary generator
    never draws it because its group count is capped well below the lengths this
    check uses.

    Args:
        rng: The generator.
        length: The number of rows.
        nulls: Whether to punch nulls into it.

    Returns:
        The column.
    """
    if rng.next_below(4) != 0:
        return random_column[DType.int64](rng, length, rng.next_below(4), nulls)

    var out = Array[DType.int64](length)
    for i in range(length):
        out[i] = Int64(i) * Int64(DIRECT_LIMIT + 17)
    if nulls:
        for i in range(length):
            if rng.next_below(5) == 0:
                out.set_null(i)
    return out^


def check_parallel(mut rng: Rng, step: Int, seed: UInt64) raises:
    """Checks that the parallel hashed route lands on the serial one's answer.

    Everything else in this file works on columns of a few hundred rows, which
    is the right size for finding a table that loses a key and the wrong size
    for finding a merge that renumbers wrongly, because a column that short is
    never cut into more than one slice. So this case draws a longer one and
    compares the two routes against each other rather than against the twin.

    The failure it is looking for does not look like a crash. A merge that
    walked the workers out of order, or that numbered a worker's groups by
    anything other than when it first saw them, still partitions the rows
    correctly and still hands back dense ordinals. What it loses is
    first-appearance order, and only the serial answer says what that is.

    Args:
        rng: The generator.
        step: The case number.
        seed: The generator seed, for the message.

    Raises:
        If the two routes disagree.
    """
    var length = rng.next_below(PARALLEL_LENGTH)
    var workers = rng.next_range(1, 17)
    var run = rng.next_u64()
    var nulls = rng.next_bool()

    var col = parallel_column(rng, length, nulls)

    var one = _factorize_hashed_serial(col, run)
    var many = _factorize_hashed_parallel(col, run, workers)

    if many.count() != one.count():
        fail(
            step,
            seed,
            "parallel",
            String(
                workers,
                " workers found ",
                many.count(),
                " groups, serial found ",
                one.count(),
            ),
        )
    if many.null_group != one.null_group:
        fail(
            step,
            seed,
            "parallel",
            String("null group ", many.null_group, " serial ", one.null_group),
        )

    for i in range(length):
        if many.codes[i] != one.codes[i]:
            fail(
                step,
                seed,
                "parallel",
                String(
                    workers,
                    " workers, row ",
                    i,
                    " code ",
                    many.codes[i],
                    " serial ",
                    one.codes[i],
                ),
            )

    var many_keys = many.keys(col)
    var one_keys = one.keys(col)
    for g in range(one.count()):
        if many_keys.is_valid(g) != one_keys.is_valid(g):
            fail(step, seed, "parallel", String("key ", g, " validity"))
        if not one_keys.is_valid(g):
            continue
        if key_bits(many_keys[g]) != key_bits(one_keys[g]):
            fail(
                step,
                seed,
                "parallel",
                String(
                    "key ",
                    g,
                    " is ",
                    many_keys[g],
                    " serial ",
                    one_keys[g],
                ),
            )


def check_partition(mut rng: Rng, step: Int, seed: UInt64) raises:
    """Checks that a partitioning is a permutation and respects its own offsets.

    Args:
        rng: The generator.
        step: The case number.
        seed: The generator seed, for the message.

    Raises:
        If a row is misplaced, duplicated or lost.
    """
    var n = rng.next_below(MAX_LENGTH)
    if n == 0:
        return
    var bits = rng.next_range(1, 6)
    var hashes = Buffer(n * 8)
    var out = hashes.bitcast[DType.uint64]()
    for i in range(n):
        out.unsafe_offset(i).unsafe_write(hash_of(Int64(rng.next_below(1000))))

    var parts = radix_partition(hashes, n, bits)
    if parts.offsets[parts.count()] != n:
        fail(step, seed, "partition", "offsets do not total the row count")

    var shift = UInt64(64 - bits)
    var seen = List[Bool]()
    for _ in range(n):
        seen.append(False)
    for p in range(parts.count()):
        var last = -1
        for at in range(parts.offsets[p], parts.offsets[p + 1]):
            var row = parts.order[at]
            if seen[row]:
                fail(step, seed, "partition", String("row ", row, " twice"))
            seen[row] = True
            if row <= last:
                fail(step, seed, "partition", "not stable")
            last = row
            if Int(out.unsafe_offset(row).unsafe_load() >> shift) != p:
                fail(step, seed, "partition", String("row ", row, " misplaced"))
    for i in range(n):
        if not seen[i]:
            fail(step, seed, "partition", String("row ", i, " lost"))


def check_table(mut rng: Rng, step: Int, seed: UInt64) raises:
    """Hammers the table directly, across a growth, against a `List` of keys.

    `factorize` never inserts the same key bits from two different values, so it
    cannot catch a table that confuses a key with an empty slot. This can, by
    drawing keys that include zero and by asking for keys that are not there.

    Args:
        rng: The generator.
        step: The case number.
        seed: The generator seed, for the message.

    Raises:
        If an ordinal or a lookup disagrees with the reference.
    """
    var table = HashTable(Int(rng.next_below(64)), rng.next_u64())
    var keys = List[UInt64]()

    for _ in range(rng.next_range(1, 400)):
        var bits = UInt64(rng.next_below(200))
        var expected = -1
        for j in range(len(keys)):
            if keys[j] == bits:
                expected = j
                break

        var found = table.find(mix(bits, table.seed()))
        if found != expected:
            fail(
                step,
                seed,
                "find",
                String("key ", bits, " gave ", found, " want ", expected),
            )

        var ordinal = table.insert(mix(bits, table.seed()))
        if expected < 0:
            if ordinal != len(keys):
                fail(
                    step,
                    seed,
                    "insert",
                    String("new key ", bits, " got ordinal ", ordinal),
                )
            keys.append(bits)
        elif ordinal != expected:
            fail(
                step,
                seed,
                "insert",
                String("key ", bits, " moved to ", ordinal),
            )

    if len(table) != len(keys):
        fail(
            step, seed, "table", String("holds ", len(table), " of ", len(keys))
        )


def run_one[dt: DType](mut rng: Rng, step: Int, seed: UInt64) raises:
    """Draws a column and checks it.

    Args:
        rng: The generator.
        step: The case number.
        seed: The generator seed.

    Parameters:
        dt: The dtype to test at.

    Raises:
        If anything disagrees.
    """
    var length = rng.next_below(MAX_LENGTH)
    var spread = Int(rng.next_below(4))
    var nulls = rng.next_bool()
    var col = random_column[dt](rng, length, spread, nulls)

    # A seed drawn per case, because the seed is per query in production and a
    # table that only ever sees one seed is a table whose collision handling has
    # only ever been tested against one set of collisions.
    check(col, rng.next_u64(), step, seed)


def main() raises:
    var options = parse_options()
    print(
        "fuzzing hash:",
        options.cases,
        "cases, seed",
        options.seed,
        "max_seconds",
        options.max_seconds,
    )

    var rng = Rng(options.seed)
    var started = perf_counter_ns()
    var applied = 0
    var stopped_early = False

    for step in range(options.cases):
        if options.max_seconds > 0.0 and step % 256 == 0 and step > 0:
            var elapsed = Float64(perf_counter_ns() - started) / 1.0e9
            if elapsed >= options.max_seconds:
                stopped_early = True
                break

        # Rotating rather than drawing keeps the mix even over a short run, the
        # same reason the kernel fuzzer rotates.
        var which = step % 9
        if which == 0:
            run_one[DType.int8](rng, step, options.seed)
        elif which == 1:
            run_one[DType.int32](rng, step, options.seed)
        elif which == 2:
            run_one[DType.int64](rng, step, options.seed)
        elif which == 3:
            run_one[DType.uint16](rng, step, options.seed)
        elif which == 4:
            run_one[DType.uint64](rng, step, options.seed)
        elif which == 5:
            var length = rng.next_below(MAX_LENGTH)
            var col = random_floats(rng, length, rng.next_bool())
            check(col, rng.next_u64(), step, options.seed)
        elif which == 6:
            check_table(rng, step, options.seed)
        elif which == 7:
            check_partition(rng, step, options.seed)
        else:
            check_parallel(rng, step, options.seed)

        applied += 1

    var seconds = Float64(perf_counter_ns() - started) / 1.0e9
    print(
        "ok:",
        applied,
        "cases in",
        seconds,
        "s",
        "(" + String(Int(Float64(applied) / seconds)) + " cases/s)",
    )
    if stopped_early:
        print(
            "stopped on the time budget before reaching", options.cases, "cases"
        )
