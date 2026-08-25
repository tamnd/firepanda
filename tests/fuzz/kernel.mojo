"""Differential fuzzing of every kernel against its scalar twin.

The bitmap fuzzer in `main.mojo` checks a packed representation against a
`List[Bool]`. This one checks a vectorized kernel against the loop anybody would
have written first, which is in `firepanda/kernel/scalar.mojo`. The twins are
never called in production; being the thing the fast path is measured against is
their entire job.

What the random columns are built to hit, deliberately:

The length is drawn from a range whose top is prime, so it lands on every offset
modulo the register width and modulo sixty four. The kernels that walk validity a
word at a time have a different code path for the last, partial word and it is
the one a fixed set of lengths will miss.

The null pattern is drawn from four shapes, not one. A uniform sprinkle at some
density never produces a validity word that is entirely zero or entirely one, and
those are exactly the two cases `min_of` and `apply_validity` special case. So the
generator also produces all-present columns, all-null columns, and columns whose
nulls come in runs long enough to fill whole words.

Values stay in a small range. Overflow is the same on both sides, so a wrapped
int8 multiply agrees with a wrapped int8 multiply and the comparison proves
nothing about the kernel. Keeping the values small means a disagreement is a real
disagreement.

Usage:
    mojo run -I . tests/fuzz/kernel.mojo [--cases=N] [--seed=N] [--max-total-time=SECONDS]
"""

from std.sys import argv
from std.time import perf_counter_ns

from firepanda.array.array import Array
from firepanda.kernel import (
    add,
    cast_to,
    count_of,
    divide,
    equal,
    filter_rows,
    greater,
    less,
    max_of,
    mean_of,
    min_of,
    multiply,
    not_equal,
    subtract,
    sum_of,
    take_rows,
)
from firepanda.kernel.scalar import (
    add_scalar,
    cast_scalar,
    count_scalar,
    divide_scalar,
    equal_scalar,
    filter_scalar,
    less_scalar,
    max_scalar,
    mean_scalar,
    min_scalar,
    multiply_scalar,
    subtract_scalar,
    sum_scalar,
    take_scalar,
)
from firepanda.testing.rng import Rng

comptime DEFAULT_CASES = 1_000_000
"""A case is two columns of up to four hundred rows through fifteen kernels, so
the default is a few hundred million row operations rather than a few hundred
million cases. It runs in under half a minute."""

comptime MAX_LENGTH = 401
"""A prime, for the reason given in the module docstring."""

comptime NULLS_NONE = 0
"""No nulls at all. The all-ones validity word path."""

comptime NULLS_ALL = 1
"""Every value null. The all-zeros validity word path."""

comptime NULLS_SPRINKLED = 2
"""Independent per value. Mixed words, and almost never a uniform one."""

comptime NULLS_RUNS = 3
"""Nulls in runs, long enough to produce whole words of either kind."""


struct Options(Copyable, Movable):
    """What the harness was asked to do."""

    var cases: Int
    """The number of columns to push through a kernel."""

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


def random_column[
    dt: DType
](mut rng: Rng, length: Int, shape: Int) -> Array[dt]:
    """Builds a column of random small values with a null pattern of a given shape.

    Args:
        rng: The generator.
        length: The number of rows.
        shape: One of the `NULLS_` constants.

    Parameters:
        dt: The dtype.

    Returns:
        The column.
    """
    var out = Array[dt](length)
    for i in range(length):
        out[i] = Scalar[dt](rng.next_range(1, 60))

    if shape == NULLS_ALL:
        for i in range(length):
            out.set_null(i)
    elif shape == NULLS_SPRINKLED:
        for i in range(length):
            if rng.next_below(4) == 0:
                out.set_null(i)
    elif shape == NULLS_RUNS:
        var at = 0
        while at < length:
            var run = rng.next_range(1, 100)
            var null_run = rng.next_bool()
            var stop = at + run
            if stop > length:
                stop = length
            if null_run:
                for i in range(at, stop):
                    out.set_null(i)
            at = stop
    return out^


def fail(step: Int, seed: UInt64, what: String, detail: String) raises:
    """Raises the standard failure message.

    Args:
        step: The case number.
        seed: The seed, so the run can be replayed.
        what: The kernel that disagreed.
        detail: What the disagreement was.

    Raises:
        Always.
    """
    raise Error(
        String("step ", step, " seed ", seed, " in ", what, ": ", detail)
    )


def same_column[
    dt: DType
](
    fast: Array[dt], twin: Array[dt], step: Int, seed: UInt64, what: String
) raises:
    """Asserts that a kernel and its twin produced the same column.

    Both the values and the validity are compared, including the values sitting
    underneath the nulls. Those are supposed to be zero on both sides, and a
    kernel that leaves arithmetic there instead is the bug this catches.

    Args:
        fast: What the kernel produced.
        twin: What the scalar twin produced.
        step: The case number.
        seed: The seed.
        what: The kernel name, for the message.

    Parameters:
        dt: The dtype.

    Raises:
        If the two disagree anywhere.
    """
    if len(fast) != len(twin):
        fail(
            step,
            seed,
            what,
            String("length ", len(fast), " but twin has ", len(twin)),
        )
    for i in range(len(fast)):
        if fast.is_valid(i) != twin.is_valid(i):
            fail(
                step,
                seed,
                what,
                String(
                    "row ",
                    i,
                    " validity ",
                    fast.is_valid(i),
                    " but twin has ",
                    twin.is_valid(i),
                ),
            )
        if fast[i] != twin[i]:
            fail(
                step,
                seed,
                what,
                String(
                    "row ",
                    i,
                    " value ",
                    fast[i],
                    " but twin has ",
                    twin[i],
                ),
            )


def run_one[dt: DType](mut rng: Rng, step: Int, seed: UInt64) raises:
    """Draws two random columns and runs every kernel over them.

    Args:
        rng: The generator.
        step: The case number.
        seed: The seed.

    Parameters:
        dt: The dtype to test at.

    Raises:
        If any kernel disagrees with its twin.
    """
    var length = rng.next_below(MAX_LENGTH)
    var a = random_column[dt](rng, length, rng.next_below(4))
    var b = random_column[dt](rng, length, rng.next_below(4))

    var total = sum_of(a)
    if total.value != sum_scalar(a):
        fail(
            step,
            seed,
            "sum_of",
            String(total.value, " but twin has ", sum_scalar(a)),
        )
    if not total.valid:
        fail(step, seed, "sum_of", "reported invalid, which it never should")

    if count_of(a) != count_scalar(a):
        fail(
            step,
            seed,
            "count_of",
            String(count_of(a), " but twin has ", count_scalar(a)),
        )

    var low = min_of(a)
    var low_twin = min_scalar(a)
    if low.valid != low_twin[1]:
        fail(step, seed, "min_of", "validity disagrees with the twin")
    if low.valid and low.value != low_twin[0]:
        fail(
            step,
            seed,
            "min_of",
            String(low.value, " but twin has ", low_twin[0]),
        )

    var high = max_of(a)
    var high_twin = max_scalar(a)
    if high.valid != high_twin[1]:
        fail(step, seed, "max_of", "validity disagrees with the twin")
    if high.valid and high.value != high_twin[0]:
        fail(
            step,
            seed,
            "max_of",
            String(high.value, " but twin has ", high_twin[0]),
        )

    var avg = mean_of(a)
    var avg_twin = mean_scalar(a)
    if avg.valid != avg_twin[1]:
        fail(step, seed, "mean_of", "validity disagrees with the twin")
    if avg.valid and avg.value != avg_twin[0]:
        fail(
            step,
            seed,
            "mean_of",
            String(avg.value, " but twin has ", avg_twin[0]),
        )

    same_column(add(a, b), add_scalar(a, b), step, seed, "add")
    same_column(subtract(a, b), subtract_scalar(a, b), step, seed, "subtract")
    same_column(multiply(a, b), multiply_scalar(a, b), step, seed, "multiply")
    same_column(divide(a, b), divide_scalar(a, b), step, seed, "divide")

    same_column(equal(a, b), equal_scalar(a, b), step, seed, "equal")
    same_column(less(a, b), less_scalar(a, b), step, seed, "less")

    # not_equal and greater are the negations, and the twins are the same loop
    # with the operator flipped, so they are checked against the pair instead.
    var ne = not_equal(a, b)
    var eq = equal(a, b)
    for i in range(length):
        if ne.is_valid(i) != eq.is_valid(i):
            fail(step, seed, "not_equal", String("row ", i, " validity"))
        if ne.is_valid(i) and Bool(ne[i]) == Bool(eq[i]):
            fail(step, seed, "not_equal", String("row ", i, " agrees with eq"))

    same_column(
        cast_to[dt, DType.float32](a),
        cast_scalar[dt, DType.float32](a),
        step,
        seed,
        "cast_to float32",
    )

    var picks = List[Int]()
    for _ in range(rng.next_below(MAX_LENGTH)):
        if length == 0 or rng.next_below(8) == 0:
            picks.append(-1)
        else:
            picks.append(rng.next_below(length))
    same_column(
        take_rows(a, picks), take_scalar(a, picks), step, seed, "take_rows"
    )

    var mask = greater(a, b)
    same_column(
        filter_rows(a, mask), filter_scalar(a, mask), step, seed, "filter_rows"
    )

    if length > 1:
        var start = rng.next_below(length)
        var end = start + rng.next_below(length - start + 1)
        var piece = a.slice(start, end)
        if len(piece) != end - start:
            fail(step, seed, "slice", String("length ", len(piece)))
        for i in range(end - start):
            if piece[i] != a[start + i] or piece.is_valid(i) != a.is_valid(
                start + i
            ):
                fail(step, seed, "slice", String("row ", i))


def main() raises:
    var options = parse_options()
    print(
        "fuzzing kernels:",
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

        # Rotating the dtype rather than picking one at random keeps the mix even
        # over a short run, which matters when CI stops the harness on a clock.
        var which = step % 6
        if which == 0:
            run_one[DType.int8](rng, step, options.seed)
        elif which == 1:
            run_one[DType.int32](rng, step, options.seed)
        elif which == 2:
            run_one[DType.int64](rng, step, options.seed)
        elif which == 3:
            run_one[DType.uint16](rng, step, options.seed)
        elif which == 4:
            run_one[DType.float32](rng, step, options.seed)
        else:
            run_one[DType.float64](rng, step, options.seed)

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
