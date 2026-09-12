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

from firepanda.array.any import AnyArray
from firepanda.array.array import Array
from firepanda.kernel import (
    AggKind,
    absolute,
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
    floor_divide,
    floor_divide_const,
    greater,
    group_top_rows,
    invert,
    less,
    max_of,
    mean_of,
    min_of,
    modulo,
    modulo_const,
    multiply,
    negate,
    not_equal,
    power,
    power_const,
    prod_of,
    subtract,
    sum_of,
    take_range,
    take_rows,
    truth_over,
)
from firepanda.dtype.logical import LogicalType
from firepanda.dtype.temporal import TimeUnit
from firepanda.kernel.arith import OP_ADD, OP_MUL, OP_SUB
from firepanda.kernel.compare import CMP_GE, CMP_LT
from firepanda.kernel.cumulative import (
    OP_CUMMAX,
    OP_CUMMIN,
    OP_CUMPROD,
    OP_CUMSUM,
    cumulative,
)
from firepanda.kernel.nulls import fill_backward, fill_forward
from firepanda.kernel.reduce import reduce_any
from firepanda.kernel.scalar import (
    absolute_scalar,
    add_scalar,
    argsort_scalar,
    arith_const_scalar,
    cast_scalar,
    compare_const_scalar,
    count_scalar,
    cumulative_scalar,
    divide_const_scalar,
    divide_scalar,
    equal_scalar,
    fill_scalar,
    filter_scalar,
    duration_days_scalar,
    floor_divide_const_scalar,
    floor_divide_scalar,
    group_scalar,
    group_top_scalar,
    invert_scalar,
    less_scalar,
    max_scalar,
    mean_scalar,
    min_scalar,
    modulo_const_scalar,
    modulo_scalar,
    multiply_scalar,
    negate_scalar,
    power_const_scalar,
    power_scalar,
    prod_scalar,
    round_to_period_scalar,
    subtract_scalar,
    sum_scalar,
    take_scalar,
    temporal_field_scalar,
    total_seconds_scalar,
    truth_scalar,
)
from firepanda.kernel.temporal import (
    FIELD_CODES,
    ROUND_DOWN,
    ROUND_HALF_EVEN,
    ROUND_UP,
    TemporalField,
    extract_field,
    field_dtype,
    round_to_period,
    temporal_day_name,
    temporal_duration_days,
    temporal_month_name,
    temporal_strftime,
    temporal_total_seconds,
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


def bounded_column[
    dt: DType
](mut rng: Rng, length: Int, shape: Int, low: Int, high: Int) -> Array[dt]:
    """Builds a column like `random_column` over a different range of values.

    `random_column` draws from one upwards and stops short of sixty, which suits
    every kernel that was here before floor division arrived. Two of the new ones
    want something else. The divisor of a floor division or a remainder has to be
    zero sometimes, because the rule about a zero divisor is the only rule in the
    kernel that depends on a value rather than on a type and it never fires on a
    column drawn from one upwards. And both operands of a power have to be
    smaller than sixty, because an int8 raised to anything in that range wraps,
    and two wrapped answers agreeing proves nothing about either of them.

    The null shape is drawn first and the values are written over the top of it,
    so a null row can end up with a zero underneath it. That is on purpose: it is
    the case where the kernel has to tell a divisor that is not there from a
    divisor that is zero.

    Args:
        rng: The generator.
        length: The number of rows.
        shape: One of the `NULLS_` constants.
        low: The smallest value to draw.
        high: One past the largest value to draw.

    Parameters:
        dt: The dtype.

    Returns:
        The column.
    """
    var out = random_column[dt](rng, length, shape)
    for i in range(length):
        out[i] = Scalar[dt](rng.next_range(low, high))
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

    truths(a, step, seed)

    nan_fills(a, step, seed)


def truths[dt: DType](a: Array[dt], step: Int, seed: UInt64) raises:
    """Checks both truth values against their twin.

    Both spellings are asked on every column rather than one of them per case,
    because the two take different exits out of the same body and a column that
    catches one of them is usually the column that would catch the other. The
    answer is a single bit either way, so there is no arithmetic to be off by
    and the comparison is exact on every dtype.

    Args:
        a: The column.
        step: The case number.
        seed: The seed.

    Parameters:
        dt: The column's dtype.

    Raises:
        If either answer disagrees with the twin.
    """
    var ptr = a.unsafe_ptr()
    var n = len(a)

    var some = truth_over[want_all=False](ptr, a.data.validity, n)
    if some != truth_scalar(a, False):
        fail(
            step,
            seed,
            "truth_over for any",
            String(some, " but twin has ", truth_scalar(a, False)),
        )

    var every = truth_over[want_all=True](ptr, a.data.validity, n)
    if every != truth_scalar(a, True):
        fail(
            step,
            seed,
            "truth_over for all",
            String(every, " but twin has ", truth_scalar(a, True)),
        )


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


def running_folds[dt: DType](col: Array[dt], step: Int, seed: UInt64) raises:
    """Runs all four scans over one column and checks each against its twin.

    All four rather than one, because the identity and the fold change together
    and a run that only checked the total would not notice a running minimum
    that had been seeded from the wrong end of the dtype.

    The column goes in at its own width, which is what `cumulative` is for. The
    entry point that answers pandas widens an int8 column to int64 before it
    scans, so fuzzing through that one would leave the narrow dtypes untested
    and would test int64 six times.

    Args:
        col: The column.
        step: The case number.
        seed: The seed.

    Parameters:
        dt: The dtype.

    Raises:
        If any of the four disagrees with its twin.
    """
    same_column(
        cumulative[code=OP_CUMSUM](col),
        cumulative_scalar[code=OP_CUMSUM](col),
        step,
        seed,
        "cumsum",
    )
    same_column(
        cumulative[code=OP_CUMPROD](col),
        cumulative_scalar[code=OP_CUMPROD](col),
        step,
        seed,
        "cumprod",
    )
    same_column(
        cumulative[code=OP_CUMMAX](col),
        cumulative_scalar[code=OP_CUMMAX](col),
        step,
        seed,
        "cummax",
    )
    same_column(
        cumulative[code=OP_CUMMIN](col),
        cumulative_scalar[code=OP_CUMMIN](col),
        step,
        seed,
        "cummin",
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
        # A float remainder by zero is a NaN on both sides, and a NaN is not
        # equal to itself, so the plain comparison below would read two kernels
        # agreeing as a disagreement. Nothing else in here can produce one.
        comptime if dt.is_floating_point():
            if isnan(fast[i]) and isnan(twin[i]):
                continue
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

    truths(a, step, seed)

    # The product is only checked on an integer dtype. Integer multiplication
    # wraps, and wrapping multiplication is associative, so the kernel folding
    # per lane and per morsel lands on the same bits as the twin folding left to
    # right. Floating point multiplication is not associative, so the two would
    # be free to disagree in the last place for reasons that are nobody's bug.
    comptime if dt.is_integral():
        var product = prod_of(a)
        if product != prod_scalar(a):
            fail(
                step,
                seed,
                "prod_of",
                String(product, " but twin has ", prod_scalar(a)),
            )

    comptime if dt.is_floating_point():
        nan_reductions[dt](rng, step, seed)

    # The running folds. On a float dtype they go round twice, because the
    # column the generator draws has nulls in it and never a NaN, and the two
    # spellings of missing take different routes through the scan: a null is a
    # cleared bit the block reads out of the validity word, and a NaN is a
    # compare on the value. Both have to end up skipped. See #170.
    running_folds(a, step, seed)
    comptime if dt.is_floating_point():
        running_folds(nan_column[dt](rng, length, step % 4), step, seed)

    same_column(add(a, b), add_scalar(a, b), step, seed, "add")
    same_column(subtract(a, b), subtract_scalar(a, b), step, seed, "subtract")
    same_column(multiply(a, b), multiply_scalar(a, b), step, seed, "multiply")
    same_column(divide(a, b), divide_scalar(a, b), step, seed, "divide")

    # Floor division and the remainder get a divisor column of their own rather
    # than reusing `b`. Their one interesting rule is what happens on a zero
    # divisor and `b` never holds a zero, so run against it they would only ever
    # check the arithmetic, which is the part of them least likely to be wrong.
    var divisors = bounded_column[dt](rng, length, rng.next_below(4), 0, 12)
    same_column(
        floor_divide(a, divisors),
        floor_divide_scalar(a, divisors),
        step,
        seed,
        "floor_divide",
    )
    same_column(
        modulo(a, divisors),
        modulo_scalar(a, divisors),
        step,
        seed,
        "modulo",
    )

    # The power gets two columns of its own for the reason in `bounded_column`,
    # which is that five cubed is the largest thing that fits in an int8 and the
    # other columns are nowhere near that small. Neither column can hold a
    # negative exponent, so the refusal is left to `tests/test_kernel.mojo`,
    # where it can be asserted rather than merely not happening.
    var bases = bounded_column[dt](rng, length, rng.next_below(4), 0, 6)
    var exponents = bounded_column[dt](rng, length, rng.next_below(4), 0, 4)
    same_column(
        power(bases, exponents),
        power_scalar(bases, exponents),
        step,
        seed,
        "power",
    )

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

    # A constant divisor gets its own draw, from a range that includes zero, so
    # that one call in twelve lands on the branch the kernel answers without
    # looking at the column at all. The flipped calls run against the column
    # that holds zeros, because there the divisor comes back out of the column
    # and the kernel is walking a register at a time again.
    var kd = Scalar[dt](rng.next_range(0, 12))
    same_column(
        floor_divide_const(a, kd),
        floor_divide_const_scalar(a, kd),
        step,
        seed,
        "floor_divide_const",
    )
    same_column(
        floor_divide_const(divisors, kd, True),
        floor_divide_const_scalar(divisors, kd, True),
        step,
        seed,
        "floor_divide_const flipped",
    )
    same_column(
        modulo_const(a, kd),
        modulo_const_scalar(a, kd),
        step,
        seed,
        "modulo_const",
    )
    same_column(
        modulo_const(divisors, kd, True),
        modulo_const_scalar(divisors, kd, True),
        step,
        seed,
        "modulo_const flipped",
    )

    # The power constant is the exponent one way round and the base the other,
    # so it is drawn twice from the two ranges the columns came from rather than
    # once. Both calls have to be handed the same constant the twin gets, which
    # is why these are named and not drawn inline.
    var ke = Scalar[dt](rng.next_range(0, 4))
    var kb = Scalar[dt](rng.next_range(0, 6))
    same_column(
        power_const(bases, ke),
        power_const_scalar(bases, ke),
        step,
        seed,
        "power_const",
    )
    same_column(
        power_const(exponents, kb, True),
        power_const_scalar(exponents, kb, True),
        step,
        seed,
        "power_const flipped",
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

    # The unary loops want a column with negatives in it, which `random_column`
    # never draws, so the sign is flipped on every third row of a column of its
    # own. An unsigned column is left alone, since flipping a sign there wraps
    # the value back into the same set of values and tests nothing the plain
    # column does not already test.
    var signed = random_column[dt](rng, length, rng.next_below(4))
    comptime if dt.is_signed():
        for i in range(length):
            if i % 3 == 0 and signed.is_valid(i):
                signed.set_valid(i, -signed[i])

    same_column(negate(signed), negate_scalar(signed), step, seed, "negate")
    same_column(
        absolute(signed), absolute_scalar(signed), step, seed, "absolute"
    )
    # `~` on a float column raises on both sides rather than answering, so the
    # comparison would have nothing to compare. `tests/test_unary.mojo` asserts
    # the refusal instead.
    comptime if not dt.is_floating_point():
        same_column(invert(signed), invert_scalar(signed), step, seed, "invert")

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


def _per_second_of(which: Int) -> Int64:
    """How many of the stored unit make a second, for the four resolutions.

    Written as four branches rather than as a list because a `comptime` list
    does not become a runtime value, and four branches on a value in a register
    is what the list would have compiled to anyway.

    Args:
        which: 0 for seconds, 1 for milliseconds, 2 for microseconds and
            anything else for nanoseconds.

    Returns:
        The divisor.
    """
    if which == 0:
        return 1
    if which == 1:
        return 1_000
    if which == 2:
        return 1_000_000
    return 1_000_000_000


def _temporal_column(
    mut rng: Rng, per_second: Int64
) raises -> Array[DType.int64]:
    """Draws a random column of instants at one of the four resolutions.

    The values are drawn as a count of seconds and then scaled, so that a
    nanosecond column gets instants spread over centuries rather than over the
    four hours that sixty four random bits of nanoseconds would cover. The range
    is the whole of what a second resolution timestamp holds, because the small
    range the arithmetic kernels use is entirely inside 1970 and a year that is
    not 1970 is the thing most likely to be wrong.

    Args:
        rng: The generator.
        per_second: How many of the stored unit make a second.

    Returns:
        The column, with one of the four null shapes over it.

    Raises:
        If the generator does.
    """
    var length = rng.next_below(MAX_LENGTH)
    var col = Array[DType.int64](length)
    var shape = rng.next_below(4)
    for i in range(length):
        var seconds = Int64(rng.next_u64() % 17_179_869_184) - 8_589_934_592
        col[i] = seconds * per_second
    if shape == NULLS_ALL:
        for i in range(length):
            col.set_null(i)
    elif shape == NULLS_SPRINKLED:
        for i in range(length):
            if rng.next_below(4) == 0:
                col.set_null(i)
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
                    col.set_null(i)
            at = stop
    return col^


def run_rounding(mut rng: Rng, step: Int, seed: UInt64) raises:
    """Rounds a random temporal column to a random period, twice.

    The period is drawn rather than parsed, so that the run covers periods the
    frequency grammar has no spelling for as well as the ones it has, and it is
    allowed to be negative because pandas allows a negative frequency. Zero is
    the one value excluded, because the callers above turn that into a copy
    before either of these is reached.

    Args:
        rng: The generator.
        step: The case number.
        seed: The seed.

    Raises:
        If a kernel disagrees with its twin.
    """
    var per_second = _per_second_of(step % 4)
    var col = _temporal_column(rng, per_second)

    var period = Int64(rng.next_u64() % 100_000) + 1
    if rng.next_bool():
        period = period * per_second
    if rng.next_bool():
        period = -period

    same_column(
        round_to_period[ROUND_DOWN](col, period),
        round_to_period_scalar[ROUND_DOWN](col, period),
        step,
        seed,
        String("floor by ", period),
    )
    same_column(
        round_to_period[ROUND_UP](col, period),
        round_to_period_scalar[ROUND_UP](col, period),
        step,
        seed,
        String("ceil by ", period),
    )
    same_column(
        round_to_period[ROUND_HALF_EVEN](col, period),
        round_to_period_scalar[ROUND_HALF_EVEN](col, period),
        step,
        seed,
        String("round by ", period),
    )


def run_temporal(mut rng: Rng, step: Int, seed: UInt64) raises:
    """Draws a random temporal column and reads every field off it twice.

    This is not folded into `run_one` because the calendar fields are not a
    per dtype operation. The column is always int64 and what rotates instead is
    the resolution, so the four divisors below all get exercised over a run.

    The values are drawn over the whole range a second resolution timestamp can
    hold rather than over the small range the arithmetic kernels use, because
    the small range is entirely inside 1970 and the thing most likely to be
    wrong is a year that is not.

    Args:
        rng: The generator.
        step: The case number.
        seed: The seed.

    Raises:
        If the kernel disagrees with its twin on any field.
    """
    var per_second = _per_second_of(step % 4)
    var per_day = per_second * 86400
    var col = _temporal_column(rng, per_second)

    comptime for code in FIELD_CODES:
        comptime result = field_dtype(code)
        var fast = extract_field[DType.int64, code, result](
            col, per_day, per_second
        )
        var twin = temporal_field_scalar[code, result](col, per_day, per_second)
        same_column(
            fast, twin, step, seed, String("temporal ", TemporalField(code))
        )


def _unit_of(which: Int) -> TimeUnit:
    """The resolution `_per_second_of` counts in, as a type rather than a
    divisor.

    Args:
        which: 0 for seconds, 1 for milliseconds, 2 for microseconds and
            anything else for nanoseconds.

    Returns:
        The unit.
    """
    if which == 0:
        return TimeUnit.SECOND
    if which == 1:
        return TimeUnit.MILLI
    if which == 2:
        return TimeUnit.MICRO
    return TimeUnit.NANO


def _duration_column(mut rng: Rng) raises -> Array[DType.int64]:
    """Draws a random column of elapsed times.

    Not `_temporal_column`, and the difference is the point. That one draws a
    whole number of seconds and then scales it, which is right for an instant
    and wrong here, because every value it produced would be a whole second and
    the two readers below both round. What is drawn instead is a magnitude with
    a random number of bits in it, so a run sees spans of a few units beside
    spans of a few centuries, and the sign is drawn separately so that the
    negative side of the rounding gets the same coverage as the positive one.

    The width is capped at fifty six bits rather than sixty three so that the
    sum over a column of them does not wrap. That is not a case being avoided,
    since the wrap is what the hardware does and both sides do it, it is a case
    that would tell nobody anything.

    Args:
        rng: The generator.

    Returns:
        The column, with one of the four null shapes over it.

    Raises:
        If the generator does.
    """
    var length = rng.next_below(MAX_LENGTH)
    var col = Array[DType.int64](length)
    var shape = rng.next_below(4)
    for i in range(length):
        var bits = rng.next_below(57)
        var magnitude = Int64(rng.next_u64() >> UInt64(63 - bits))
        col[i] = -magnitude if rng.next_bool() else magnitude
    if shape == NULLS_ALL:
        for i in range(length):
            col.set_null(i)
    elif shape == NULLS_SPRINKLED:
        for i in range(length):
            if rng.next_below(4) == 0:
                col.set_null(i)
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
                    col.set_null(i)
            at = stop
    return col^


def run_durations(mut rng: Rng, step: Int, seed: UInt64) raises:
    """Reads a random column of elapsed times two ways and reduces it three.

    The two readers are the new loops and are what this case is for. `dt.days`
    rounds downward, which is the property the twin is written not to share, and
    `dt.total_seconds` divides into float64, which is where a column of whole
    milliseconds either keeps its fraction or quietly loses it.

    The three reductions are here for the type and not for the arithmetic. The
    adding and the comparing are the same loops `run_one` already fuzzes over
    every dtype, so what is being checked is that the temporal route through
    `reduce_any` reaches them at all and hands back an answer that is still a
    length of time rather than the int64 it is stored in.

    The arithmetic between two duration columns is deliberately not here. It is
    `temporal_as_unit` followed by the same add and subtract loops, and both
    halves are fuzzed already, so a case for the pair would be drawing operands
    small enough that the rescale cannot overflow and then testing nothing that
    is not tested twice over.

    Args:
        rng: The generator.
        step: The case number.
        seed: The seed.

    Raises:
        If a kernel disagrees with its twin.
    """
    var which = step % 4
    var per_second = _per_second_of(which)
    var per_day = per_second * 86400
    var col = _duration_column(rng)
    var any = AnyArray(
        Array[DType.int64](copy=col).into_data(),
        LogicalType.duration(_unit_of(which)),
    )

    var days = temporal_duration_days(any)
    same_column(
        days.as_typed_view[DType.int64](),
        duration_days_scalar(col, per_day),
        step,
        seed,
        "duration days",
    )
    var seconds = temporal_total_seconds(any)
    same_column(
        seconds.as_typed_view[DType.float64](),
        total_seconds_scalar(col, per_second),
        step,
        seed,
        "total seconds",
    )

    var total = reduce_any(any, AggKind.SUM)
    if String(total.type) != String(any.type):
        fail(step, seed, "duration sum", String("answered ", total.type))
    if total.as_typed_view[DType.int64]()[0] != Int64(sum_scalar(col)):
        fail(step, seed, "duration sum", "disagrees with the twin")

    var least = reduce_any(any, AggKind.MIN)
    var low = min_scalar(col)
    if String(least.type) != String(any.type):
        fail(step, seed, "duration min", String("answered ", least.type))
    if least.data.validity.get(0) != low[1]:
        fail(step, seed, "duration min", "disagrees about finding anything")
    if low[1] and least.as_typed_view[DType.int64]()[0] != low[0]:
        fail(step, seed, "duration min", "disagrees with the twin")

    var most = reduce_any(any, AggKind.MAX)
    var high = max_scalar(col)
    if high[1] and most.as_typed_view[DType.int64]()[0] != high[0]:
        fail(step, seed, "duration max", "disagrees with the twin")


def _padded(value: Int64, width: Int) -> String:
    """Writes a number with leading zeroes to a minimum width.

    Args:
        value: The number, never negative here.
        width: The minimum number of digits.

    Returns:
        The digits.
    """
    var digits = String(value)
    var out = String()
    for _ in range(width - digits.byte_length()):
        out += "0"
    return out + digits


def _weekday_names() -> List[String]:
    """The seven names, in the order the ISO day numbers them.

    Returns:
        Monday first.
    """
    return [
        String("Monday"),
        "Tuesday",
        "Wednesday",
        "Thursday",
        "Friday",
        "Saturday",
        "Sunday",
    ]


def _month_names() -> List[String]:
    """The twelve names, in the order the calendar numbers them.

    Returns:
        January first.
    """
    return [
        String("January"),
        "February",
        "March",
        "April",
        "May",
        "June",
        "July",
        "August",
        "September",
        "October",
        "November",
        "December",
    ]


def run_names(mut rng: Rng, step: Int, seed: UInt64) raises:
    """Names and formats a random temporal column, and checks the text by hand.

    The expected text is assembled out of `temporal_field_scalar`, which is the
    one row at a time twin the calendar fields are already checked against, so
    this is the formatter against a different implementation of the calendar
    rather than against itself. The names are looked up in a list written out
    here, so a table that is rotated by one day shows up as a disagreement.

    Args:
        rng: The generator.
        step: The case number.
        seed: The seed.

    Raises:
        If the text disagrees with the fields it is supposed to be made of.
    """
    var which = step % 4
    var per_second = _per_second_of(which)
    var per_day = per_second * 86400
    var col = _temporal_column(rng, per_second)

    var years = temporal_field_scalar[0, DType.int32](col, per_day, per_second)
    var months = temporal_field_scalar[1, DType.int32](col, per_day, per_second)
    var days = temporal_field_scalar[2, DType.int32](col, per_day, per_second)
    var hours = temporal_field_scalar[3, DType.int32](col, per_day, per_second)
    var minutes = temporal_field_scalar[4, DType.int32](
        col, per_day, per_second
    )
    var seconds = temporal_field_scalar[5, DType.int32](
        col, per_day, per_second
    )
    var doys = temporal_field_scalar[9, DType.int32](col, per_day, per_second)
    var iso_years = temporal_field_scalar[19, DType.uint32](
        col, per_day, per_second
    )
    var iso_weeks = temporal_field_scalar[20, DType.uint32](
        col, per_day, per_second
    )
    var iso_days = temporal_field_scalar[21, DType.uint32](
        col, per_day, per_second
    )

    var height = len(col)
    var any = AnyArray(col^.into_data(), LogicalType.timestamp(_unit_of(which)))
    var text = temporal_strftime(any, "%Y-%m-%d %H:%M:%S %j %G %V %u")
    var day_names = temporal_day_name(any, "")
    var month_names = temporal_month_name(any, "")
    var weekdays = _weekday_names()
    var calendar = _month_names()

    for i in range(height):
        if not years.is_valid(i):
            if text.is_valid(i) or day_names.is_valid(i):
                raise Error(
                    String(
                        "names: row ",
                        i,
                        " of case ",
                        step,
                        " at seed ",
                        seed,
                        " has no calendar and has text",
                    )
                )
            continue

        var want = _padded(Int64(years[i]), 4)
        want += "-" + _padded(Int64(months[i]), 2)
        want += "-" + _padded(Int64(days[i]), 2)
        want += " " + _padded(Int64(hours[i]), 2)
        want += ":" + _padded(Int64(minutes[i]), 2)
        want += ":" + _padded(Int64(seconds[i]), 2)
        want += " " + _padded(Int64(doys[i]), 3)
        want += " " + _padded(Int64(iso_years[i]), 4)
        want += " " + _padded(Int64(iso_weeks[i]), 2)
        want += " " + String(Int64(iso_days[i]))
        if text[i] != want:
            raise Error(
                String(
                    "names: row ",
                    i,
                    " of case ",
                    step,
                    " at seed ",
                    seed,
                    " formatted as '",
                    text[i],
                    "' and the fields say '",
                    want,
                    "'",
                )
            )

        var want_day = weekdays[Int(iso_days[i]) - 1]
        if day_names[i] != want_day:
            raise Error(
                String(
                    "names: row ",
                    i,
                    " of case ",
                    step,
                    " at seed ",
                    seed,
                    " is a ",
                    day_names[i],
                    " and the ISO day says ",
                    want_day,
                )
            )

        var want_month = calendar[Int(months[i]) - 1]
        if month_names[i] != want_month:
            raise Error(
                String(
                    "names: row ",
                    i,
                    " of case ",
                    step,
                    " at seed ",
                    seed,
                    " is in ",
                    month_names[i],
                    " and the month says ",
                    want_month,
                )
            )


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

        # One case in eight, because nineteen fields over a column is nineteen
        # passes and running it every time would take the run away from the
        # other kernels for coverage of one of them.
        if step % 8 == 0:
            run_temporal(rng, step, options.seed)

        # The other half of the rotation, so that the temporal families get one
        # case in eight each rather than sharing one.
        if step % 8 == 4:
            run_rounding(rng, step, options.seed)

        if step % 8 == 2:
            run_names(rng, step, options.seed)

        if step % 8 == 6:
            run_durations(rng, step, options.seed)

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
