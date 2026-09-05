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

from std.math import isnan, nan
from std.sys import argv
from std.time import perf_counter_ns

from firepanda.array.array import Array
from firepanda.kernel import (
    AggKind,
    add,
    aggregate_group,
    argsort,
    arith_const,
    cast_any,
    cast_to,
    compare_const,
    count_of,
    divide,
    divide_const,
    equal,
    filter_range,
    filter_rows,
    greater,
    group_top_rows,
    less,
    max_of,
    mean_of,
    min_of,
    multiply,
    not_equal,
    subtract,
    sum_of,
    take_range,
    take_rows,
)
from firepanda.kernel.arith import OP_ADD, OP_MUL, OP_SUB
from firepanda.kernel.compare import CMP_GE, CMP_LT
from firepanda.kernel.nulls import fill_backward, fill_forward
from firepanda.kernel.scalar import (
    add_scalar,
    argsort_scalar,
    arith_const_scalar,
    cast_scalar,
    compare_const_scalar,
    count_scalar,
    divide_const_scalar,
    divide_scalar,
    equal_scalar,
    fill_scalar,
    filter_scalar,
    group_scalar,
    group_top_scalar,
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
"""A case is two columns of up to four hundred rows through every kernel, so the
default is a few hundred million row operations rather than a few hundred million
cases. It runs in about a minute."""

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


def nan_column[dt: DType](mut rng: Rng, length: Int, shape: Int) -> Array[dt]:
    """Builds a float column holding both spellings of missing.

    `random_column` never draws a NaN and should not start. Most of the kernels
    it feeds treat a NaN as an ordinary value on purpose, so putting one in would
    only check that a kernel and its twin agree about a value neither of them
    looks at. The reductions are the ones that step over a NaN, so they get a
    column of their own.

    Args:
        rng: The generator.
        length: The number of rows.
        shape: 0 for NaN only, 1 for null only, 2 for both, 3 for nothing but
            NaN.

    Parameters:
        dt: The dtype, which is always a floating point one here.

    Returns:
        The column.
    """
    var out = Array[dt](length)
    for i in range(length):
        if shape == 3:
            out[i] = nan[dt]()
            continue
        out[i] = Scalar[dt](rng.next_range(1, 60))
        var draw = rng.next_below(8)
        if draw == 0 and shape != 1:
            out[i] = nan[dt]()
        elif draw == 1 and shape != 0:
            out.set_null(i)
    return out^


def nan_reductions[dt: DType](mut rng: Rng, step: Int, seed: UInt64) raises:
    """Checks the whole column reductions over a column with NaNs in it.

    The values are small integers, so the sums are exact and can be compared for
    equality rather than within a tolerance, the same as everywhere else in this
    harness. What is under test is which rows were read and not the arithmetic.

    Args:
        rng: The generator.
        step: The case number.
        seed: The seed.

    Parameters:
        dt: The dtype to test at.

    Raises:
        If any reduction disagrees with its twin.
    """
    var a = nan_column[dt](rng, rng.next_below(MAX_LENGTH), step % 4)

    var total = sum_of(a)
    if total.value != sum_scalar(a):
        fail(
            step,
            seed,
            "sum_of over NaN",
            String(total.value, " but twin has ", sum_scalar(a)),
        )

    var low = min_of(a)
    var low_twin = min_scalar(a)
    if low.valid != low_twin[1]:
        fail(step, seed, "min_of over NaN", "validity disagrees with the twin")
    if low.valid and low.value != low_twin[0]:
        fail(
            step,
            seed,
            "min_of over NaN",
            String(low.value, " but twin has ", low_twin[0]),
        )

    var high = max_of(a)
    var high_twin = max_scalar(a)
    if high.valid != high_twin[1]:
        fail(step, seed, "max_of over NaN", "validity disagrees with the twin")
    if high.valid and high.value != high_twin[0]:
        fail(
            step,
            seed,
            "max_of over NaN",
            String(high.value, " but twin has ", high_twin[0]),
        )

    var avg = mean_of(a)
    var avg_twin = mean_scalar(a)
    if avg.valid != avg_twin[1]:
        fail(step, seed, "mean_of over NaN", "validity disagrees with the twin")
    if avg.valid and avg.value != avg_twin[0]:
        fail(
            step,
            seed,
            "mean_of over NaN",
            String(avg.value, " but twin has ", avg_twin[0]),
        )

    nan_fills(a, step, seed)


def nan_fills[dt: DType](a: Array[dt], step: Int, seed: UInt64) raises:
    """Checks both fills over a column with NaNs in it against their twin.

    This is the check the fills did not have at all before, and it is the one
    they most needed. The kernel decides for a whole block of sixty four rows at
    once whether it can copy them and take the carry from the end, and the twin
    walks out from each missing row on its own, so the two disagree first at a
    word boundary and the length here is drawn from a range whose top is prime
    precisely so that the boundary lands in a different place every case.

    The limit is drawn as well, because a NaN is a row of the run the limit is
    counting rather than a value that resets it, and a fill that got that wrong
    would fill one row too many and only on a float column.

    Args:
        a: The column, which has NaNs in it.
        step: The case number.
        seed: The seed.

    Parameters:
        dt: The dtype to test at.

    Raises:
        If either fill disagrees with its twin.
    """
    var limit = step % 4
    var forward = fill_forward(a, limit)
    var forward_twin = fill_scalar[forward=True](a, limit)
    var backward = fill_backward(a, limit)
    var backward_twin = fill_scalar[forward=False](a, limit)

    for i in range(len(a)):
        # A NaN is what missing looks like coming out of a fill on a float
        # column, so the comparison is on which rows are a NaN and then on the
        # values of the rest. Comparing the values first would call every
        # missing row a difference, since a NaN is not equal to itself.
        if isnan(forward[i]) != isnan(forward_twin[i]):
            fail(
                step,
                seed,
                "fill_forward over NaN",
                String("row ", i, " disagrees about being missing"),
            )
        if not isnan(forward[i]) and forward[i] != forward_twin[i]:
            fail(
                step,
                seed,
                "fill_forward over NaN",
                String(
                    "row ",
                    i,
                    ": ",
                    forward[i],
                    " but twin has ",
                    forward_twin[i],
                ),
            )
        if isnan(backward[i]) != isnan(backward_twin[i]):
            fail(
                step,
                seed,
                "fill_backward over NaN",
                String("row ", i, " disagrees about being missing"),
            )
        if not isnan(backward[i]) and backward[i] != backward_twin[i]:
            fail(
                step,
                seed,
                "fill_backward over NaN",
                String(
                    "row ",
                    i,
                    ": ",
                    backward[i],
                    " but twin has ",
                    backward_twin[i],
                ),
            )

    if forward.null_count() != 0 or backward.null_count() != 0:
        fail(
            step,
            seed,
            "fill over NaN",
            "a float column came out of a fill carrying a null",
        )


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


def same_grouped[
    dt: DType
](
    a: Array[dt],
    kind: AggKind,
    codes: Array[DType.uint32],
    groups: Int,
    step: Int,
    seed: UInt64,
) raises:
    """Runs one grouped reduction against its twin and reports a disagreement.

    Pulled out of `run_one` so it can be run twice over the same codes, once on
    the column the generator drew and once on a column with NaNs in it. See #170.

    Args:
        a: The column being aggregated.
        kind: Which reduction.
        codes: One group ordinal per row.
        groups: The number of distinct ordinals.
        step: The case number.
        seed: The seed.

    Raises:
        Whatever the reduction raises, and nothing of its own.
    """
    var reduced = cast_any(
        aggregate_group(a, kind, codes, groups), DType.float64
    ).as_typed[DType.float64]()
    var twin = group_scalar(a, kind, codes, groups)
    for g in range(groups):
        if reduced.is_valid(g) != twin[1][g]:
            fail(
                step,
                seed,
                "aggregate_group",
                String("group ", g, " validity under ", kind),
            )
        if reduced.is_valid(g):
            # NaN is an answer here and not an error, because a float valued
            # reduction with nothing to reduce reports one rather than a null.
            # It has to be checked before the subtraction rather than after,
            # since NaN minus anything is NaN and NaN fails every comparison, so
            # a difference of one NaN against a real number would slide through
            # the tolerance below without ever firing. See #170.
            if isnan(reduced[g]) or isnan(twin[0][g]):
                if isnan(reduced[g]) != isnan(twin[0][g]):
                    fail(
                        step,
                        seed,
                        "aggregate_group",
                        String(
                            "group ",
                            g,
                            " is ",
                            reduced[g],
                            " but twin has ",
                            twin[0][g],
                            " under ",
                            kind,
                        ),
                    )
                continue
            var delta = reduced[g] - twin[0][g]
            if delta < 0.0:
                delta = -delta
            # Relative once the numbers get big. The two sides add the same terms
            # in different orders, which is exact for a sum of integers and is
            # not for a variance, where the result is a square and a column of
            # values near a million lands near 1e12. An absolute tolerance there
            # is asking floating point addition to be associative.
            var scale = reduced[g] if reduced[g] >= 0.0 else -reduced[g]
            var tolerance = 1.0e-9
            if scale > 1.0:
                tolerance = 1.0e-9 * scale
            if delta > tolerance:
                fail(
                    step,
                    seed,
                    "aggregate_group",
                    String(
                        "group ",
                        g,
                        " is ",
                        reduced[g],
                        " but twin has ",
                        twin[0][g],
                        " under ",
                        kind,
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

    # Sorting is checked against an insertion sort over `<`, so the permutation
    # and its stability are both under test rather than only the sorted values.
    # The direction and the null placement rotate with the case number instead
    # of being drawn, so even a short run covers all four combinations.
    # Short columns, and one case in four. The twin is quadratic and the sort
    # allocates four buffers per call, so running it on every case costs more
    # time than the other fourteen kernels put together, which buys coverage of
    # one kernel by taking it away from all of them. Everything the sort has a
    # separate code path for, the partial validity word and the digit skip,
    # happens under a hundred rows too. The four combinations of direction and
    # null placement rotate on the cases that do run, not on all of them.
    if length <= 96 and step % 4 == 0:
        var mode = (step // 4) % 4
        var descending = mode % 2 == 1
        var nulls_first = mode >= 2
        var order = argsort(a, descending, nulls_first)
        var order_twin = argsort_scalar(a, descending, nulls_first)
        if len(order) != len(order_twin):
            fail(step, seed, "argsort", "length disagrees with the twin")
        for i in range(len(order_twin)):
            if Int(order[i]) != order_twin[i]:
                fail(
                    step,
                    seed,
                    "argsort",
                    String(
                        "row ",
                        i,
                        " is ",
                        Int(order[i]),
                        " but twin has ",
                        order_twin[i],
                    ),
                )

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

    comptime if dt.is_floating_point():
        nan_reductions[dt](rng, step, seed)

    same_column(add(a, b), add_scalar(a, b), step, seed, "add")
    same_column(subtract(a, b), subtract_scalar(a, b), step, seed, "subtract")
    same_column(multiply(a, b), multiply_scalar(a, b), step, seed, "multiply")
    same_column(divide(a, b), divide_scalar(a, b), step, seed, "divide")

    # The constant forms take the same path through `apply_validity` and a
    # different path to the operand, so they are checked separately rather than
    # assumed to follow from the two column forms agreeing. The constant is drawn
    # from the same distribution as the columns, which keeps it away from zero
    # and so keeps division out of the one case where a value cannot be compared
    # against itself.
    var k = Scalar[dt](rng.next_range(1, 60))
    same_column(
        arith_const[dt, OP_ADD](a, k),
        arith_const_scalar[dt, OP_ADD](a, k),
        step,
        seed,
        "arith_const add",
    )
    same_column(
        arith_const[dt, OP_SUB](a, k),
        arith_const_scalar[dt, OP_SUB](a, k),
        step,
        seed,
        "arith_const subtract",
    )
    same_column(
        arith_const[dt, OP_SUB](a, k, True),
        arith_const_scalar[dt, OP_SUB](a, k, True),
        step,
        seed,
        "arith_const subtract flipped",
    )
    same_column(
        arith_const[dt, OP_MUL](a, k),
        arith_const_scalar[dt, OP_MUL](a, k),
        step,
        seed,
        "arith_const multiply",
    )
    same_column(
        divide_const(a, k),
        divide_const_scalar(a, k),
        step,
        seed,
        "divide_const",
    )
    same_column(
        divide_const(a, k, True),
        divide_const_scalar(a, k, True),
        step,
        seed,
        "divide_const flipped",
    )
    same_column(
        compare_const[dt, CMP_LT](a, k),
        compare_const_scalar[dt, CMP_LT](a, k),
        step,
        seed,
        "compare_const less",
    )
    same_column(
        compare_const[dt, CMP_GE](a, k),
        compare_const_scalar[dt, CMP_GE](a, k),
        step,
        seed,
        "compare_const greater or equal",
    )

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

    # The range forms against the general ones. `take_range` and `filter_range`
    # exist to avoid building the range they read from, so the only way to know
    # the shortcut agrees with the long way round is to build the range and put
    # it through the kernel that is already fuzzed. The start is drawn rather
    # than left at zero, because zero is the one value where using it wrongly
    # cannot be seen.
    var start = Int(rng.next_below(1000))
    var line = Array[DType.int64](overwritten=length)
    for i in range(length):
        line.store[1](i, Int64(start + i))
    same_column(
        take_range(start, picks),
        take_rows(line, picks),
        step,
        seed,
        "take_range",
    )
    same_column(
        filter_range(start, mask),
        filter_rows(line, mask),
        step,
        seed,
        "filter_range",
    )

    # Grouped reductions, one kind per case so that a run covers all eight. The
    # codes are drawn rather than factorized: what is under test here is the
    # scatter and the null policy, and running the real grouping first would
    # produce a code distribution shaped by the column rather than by chance.
    # Few groups on purpose, because a group with one row in it exercises none of
    # the accumulate path and the interesting cases are the crowded ones.
    var groups = 1 + rng.next_below(6)
    var codes = Array[DType.uint32](length)
    for i in range(length):
        codes[i] = UInt32(rng.next_below(groups))
    # One reduction per case, so every one of them sees every column shape the
    # generator makes rather than a fifteenth of them. The list is written out
    # rather than counted from zero, because the codes of the single column
    # kinds are no longer contiguous: `CORR` and `COV` sit between `NUNIQUE` and
    # `SEM` and they read a second column, so a rotation over the raw codes would
    # hand this one a pair kind. `QUANTILE` gets a different position each time
    # it comes round, because a quantile that only ever runs at the median is a
    # median with extra arithmetic and the interpolation between two values is
    # the part worth checking.
    var single = [
        AggKind.SUM,
        AggKind.MEAN,
        AggKind.MIN,
        AggKind.MAX,
        AggKind.COUNT,
        AggKind.FIRST,
        AggKind.LAST,
        AggKind.SIZE,
        AggKind.VAR,
        AggKind.STD,
        AggKind.MEDIAN,
        AggKind.QUANTILE,
        AggKind.NUNIQUE,
        AggKind.SEM,
        AggKind.SKEW,
    ]
    var kind = single[(step // 4) % len(single)]
    if kind == AggKind.QUANTILE:
        kind = AggKind.quantile_at(Float64(rng.next_below(101)) / 100.0)
    elif kind == AggKind.VAR or kind == AggKind.STD or kind == AggKind.SEM:
        kind = AggKind(kind.code, Float64(rng.next_below(3)))
    same_grouped(a, kind, codes, groups, step, seed)

    # The same reduction again over a column that has NaNs in it as well as
    # nulls. `random_column` never draws a NaN, so without this the grouped path
    # would only ever be checked on one of the two spellings of missing, and the
    # grouped kernels have to step over both. Only on a float dtype, because
    # there is no NaN to draw on any other one. See #170.
    comptime if dt.is_floating_point():
        var poisoned = nan_column[dt](rng, length, step % 4)
        same_grouped(poisoned, kind, codes, groups, step, seed)

    # Top-n per group, against the twin that scans the column once per slot.
    # The same drawn codes, because the interesting shape here is a crowded group
    # where several rows are competing for the last slot, and `n` alternates so a
    # run covers the single slot case as well as the crowded one.
    var slots = 1 + (step % 3)
    var wants_largest = (step // 3) % 2 == 0
    var top = group_top_rows(a, codes, groups, slots, wants_largest)
    var top_twin = group_top_scalar(a, codes, groups, slots, wants_largest)
    var cursor = 0
    for g in range(groups):
        var expected = 0
        for k in range(slots):
            if top_twin[g * slots + k] >= 0:
                expected += 1
        if top.counts[g] != expected:
            fail(
                step,
                seed,
                "group_top_rows",
                String("group ", g, " kept ", top.counts[g], " not ", expected),
            )
        for k in range(top.counts[g]):
            if top.rows_at[cursor + k] != top_twin[g * slots + k]:
                fail(
                    step,
                    seed,
                    "group_top_rows",
                    String(
                        "group ",
                        g,
                        " slot ",
                        k,
                        " is row ",
                        top.rows_at[cursor + k],
                        " but twin has ",
                        top_twin[g * slots + k],
                    ),
                )
        cursor += top.counts[g]

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
