"""Differential fuzzing of `join_indices` against the nested loop twin.

Same arrangement as `kernel.mojo` and `hash.mojo`. The fast implementation is
compared against the slow one anybody would have written first, which here is
`firepanda/join/scalar.mojo` comparing every row against every row.

What the generator is built to hit is multiplicity. A join is not a mapping, it
is a relation, and almost every way of getting one wrong shows up as the right
values in the wrong number of rows. So the key range is drawn per case and drawn
small: with keys in `[0, 3)` and forty rows a side, a single key is on a dozen
rows of each frame and one bucket produces a hundred and fifty output rows. A
generator drawing keys from a wide range would produce a nearly one-to-one join
every time and would never exercise the case that breaks.

The comparison is on the pair lists themselves rather than on any frame built
from them, and it includes the order. That is deliberate: the order is a promise
this library makes, and a harness that sorted before comparing would let it drift
while still passing.

Three things are drawn to make the shape vary rather than to hit a specific bug.
Null density, because a null key matches nothing and the rows still have to
survive on the kinds that keep unmatched rows. Key count, because two keys go
through the packing path in `group_ordinals` and one does not. And an occasional
float column, because NaN and negative zero are the two values where the right
answer disagrees with `==`, and neither will ever be drawn from an integer
generator.

Usage:
    mojo run -I . tests/fuzz/join.mojo [--cases=N] [--seed=N] [--max-total-time=SECONDS]
"""

from std.sys import argv
from std.time import perf_counter_ns

from firepanda.array.any import AnyArray, borrow_columns
from firepanda.array.array import Array
from firepanda.join.pairs import JoinIndices, JoinKind, join_indices
from firepanda.join.scalar import join_nested
from firepanda.testing.rng import Rng

comptime DEFAULT_CASES = 2_000_000
"""A case is two frames of up to forty rows joined twice, once quadratically.
The default runs in roughly forty seconds."""

comptime MAX_ROWS = 41
"""Small on purpose. The quadratic side is the budget, and what is being tested
is multiplicity rather than scale."""

comptime KIND_COUNT = 7
"""Inner, left, right, outer, semi, anti and cross."""


struct Options(Copyable, Movable):
    """What the harness was asked to do."""

    var cases: Int
    """The number of joins to check."""

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


def integer_keys(
    mut rng: Rng, rows: Int, ceiling: Int, null_odds: Int
) -> AnyArray:
    """Builds an int64 key column.

    Args:
        rng: The generator.
        rows: The height.
        ceiling: Keys are drawn below this.
        null_odds: One row in this many is null, or zero for none.

    Returns:
        The column, erased.
    """
    var column = Array[DType.int64](rows)
    for i in range(rows):
        if null_odds > 0 and rng.next_below(null_odds) == 0:
            column.set_null(i)
            continue
        column[i] = Int64(rng.next_below(ceiling))
    return AnyArray(column^)


def float_keys(
    mut rng: Rng, rows: Int, ceiling: Int, null_odds: Int
) -> AnyArray:
    """Builds a float64 key column that contains NaN and negative zero.

    Both of those are values where key equality disagrees with `==`, and the
    twin keys on `key_bits` for the same reason the fast path does. A generator
    that only drew ordinary floats would leave that agreement untested.

    Args:
        rng: The generator.
        rows: The height.
        ceiling: Ordinary keys are drawn below this.
        null_odds: One row in this many is null, or zero for none.

    Returns:
        The column, erased.
    """
    var column = Array[DType.float64](rows)
    for i in range(rows):
        if null_odds > 0 and rng.next_below(null_odds) == 0:
            column.set_null(i)
            continue
        var pick = rng.next_below(8)
        if pick == 0:
            column[i] = Float64(0.0) / Float64(0.0)
        elif pick == 1:
            column[i] = -Float64(0.0)
        elif pick == 2:
            column[i] = Float64(0.0)
        else:
            column[i] = Float64(rng.next_below(ceiling))
    return AnyArray(column^)


def compare(
    fast: JoinIndices,
    slow: JoinIndices,
    step: Int,
    seed: UInt64,
    kind: JoinKind,
) raises:
    """Raises on the first place the two pairings differ.

    Args:
        fast: What `join_indices` produced.
        slow: What the nested loop produced.
        step: The case number, for the failure message.
        seed: The seed, for the failure message.
        kind: The join kind, for the failure message.

    Raises:
        If the two disagree in length or anywhere in either list.
    """
    if len(fast) != len(slow):
        raise Error(
            String(
                "step ",
                step,
                " seed ",
                seed,
                " kind ",
                kind,
                ": produced ",
                len(fast),
                " rows but the twin produced ",
                len(slow),
            )
        )
    for r in range(len(fast)):
        if fast.left_at[r] != slow.left_at[r]:
            raise Error(
                String(
                    "step ",
                    step,
                    " seed ",
                    seed,
                    " kind ",
                    kind,
                    ": row ",
                    r,
                    " came from left row ",
                    fast.left_at[r],
                    " but the twin says ",
                    slow.left_at[r],
                )
            )
        if fast.right_at[r] != slow.right_at[r]:
            raise Error(
                String(
                    "step ",
                    step,
                    " seed ",
                    seed,
                    " kind ",
                    kind,
                    ": row ",
                    r,
                    " came from right row ",
                    fast.right_at[r],
                    " but the twin says ",
                    slow.right_at[r],
                )
            )


def main() raises:
    var options = parse_options()
    print(
        "fuzzing join:",
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
        if options.max_seconds > 0.0 and step % 1024 == 0 and step > 0:
            var elapsed = Float64(perf_counter_ns() - started) / 1.0e9
            if elapsed >= options.max_seconds:
                stopped_early = True
                break

        var kind = JoinKind(UInt8(step % KIND_COUNT))
        var key_count = 1 + rng.next_below(2)
        if kind == JoinKind.CROSS:
            key_count = 0

        var left_rows = rng.next_below(MAX_ROWS)
        var right_rows = rng.next_below(MAX_ROWS)
        if kind == JoinKind.CROSS:
            # The product is the cost, so this one gets shorter frames rather
            # than a different generator.
            left_rows = rng.next_below(12)
            right_rows = rng.next_below(12)

        # Small ranges make keys collide, which is the whole point. The large
        # one is here so that the nearly one-to-one case is covered too.
        var ceiling = 2
        var spread = rng.next_below(4)
        if spread == 1:
            ceiling = 3
        elif spread == 2:
            ceiling = 7
        elif spread == 3:
            ceiling = 64

        var null_odds = 0
        if rng.next_bool():
            null_odds = 2 + rng.next_below(6)

        var floating = rng.next_below(4) == 0

        var left_columns = List[AnyArray]()
        var right_columns = List[AnyArray]()
        var left_keys = List[Int]()
        var right_keys = List[Int]()
        for k in range(key_count):
            if floating:
                left_columns.append(
                    float_keys(rng, left_rows, ceiling, null_odds)
                )
                right_columns.append(
                    float_keys(rng, right_rows, ceiling, null_odds)
                )
            else:
                left_columns.append(
                    integer_keys(rng, left_rows, ceiling, null_odds)
                )
                right_columns.append(
                    integer_keys(rng, right_rows, ceiling, null_odds)
                )
            left_keys.append(k)
            right_keys.append(k)

        var fast = join_indices(
            borrow_columns(left_columns),
            left_keys,
            left_rows,
            borrow_columns(right_columns),
            right_keys,
            right_rows,
            kind,
        )
        var slow = join_nested(
            borrow_columns(left_columns),
            left_keys,
            left_rows,
            borrow_columns(right_columns),
            right_keys,
            right_rows,
            kind,
        )
        compare(fast, slow, step, options.seed, kind)
        applied += 1

    var elapsed_ns = perf_counter_ns() - started
    var seconds = Float64(elapsed_ns) / 1.0e9
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
