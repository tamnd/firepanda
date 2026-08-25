"""Differential fuzzing of the validity bitmap against a `List[Bool]` reference.

The bitmap is the type where a bug is least likely to be caught by a unit test and
most likely to be catastrophic. Its whole job is to pack a boolean per row into a
bit, and every operation on it does something clever with byte boundaries: masked
first and last bytes, a whole-byte middle, a popcount over a tail that must stay
zero. Any of those can be wrong for exactly one length modulo eight, and a hand
written test suite will pick the wrong eight lengths.

So the reference implementation is a `List[Bool]`, written the obvious way with no
bit twiddling at all, and the harness applies the same random operation to both and
compares. A failure prints the seed and the case number, and re-running with
`--seed=N` replays the run exactly, because the generator is splitmix64 over a
single word of state rather than anything seeded from the clock.

Usage:
    mojo run -I . tests/fuzz/main.mojo [--cases=N] [--seed=N] [--max-total-time=SECONDS]

CI runs it with a time budget. A developer chasing a bug runs it with the seed the
failure printed. The M0 exit criterion is ten million cases, which is the default.
"""

from std.sys import argv
from std.time import perf_counter_ns

from firepanda.bitmap.bitmap import Bitmap
from firepanda.testing.rng import Rng

comptime DEFAULT_CASES = 10_000_000
"""The M0 exit criterion. Roughly a minute on a current desktop core."""

comptime MAX_LENGTH = 517
"""A prime, so lengths land on every offset modulo eight and modulo sixty four."""


struct Options(Copyable, Movable):
    """What the harness was asked to do."""

    var cases: Int
    """The number of operations to apply."""

    var seed: UInt64
    """The generator seed. Printed on every run so a failure can be replayed."""

    var max_seconds: Float64
    """A wall clock budget. Zero means no budget."""

    def __init__(out self):
        """Constructs the defaults."""
        self.cases = DEFAULT_CASES
        self.seed = 0x243F6A8885A308D3
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


def reference_set_range(
    mut model: List[Bool], start: Int, end: Int, value: Bool
):
    """Applies `set_range` to the reference the obvious way.

    Args:
        model: The reference bits.
        start: The first position, inclusive.
        end: The last position, exclusive.
        value: The value to write.
    """
    for i in range(start, end):
        model[i] = value


def check(
    bitmap: Bitmap, model: List[Bool], step: Int, seed: UInt64, what: String
) raises:
    """Compares the bitmap against the reference and raises on the first difference.

    This is the whole point of the harness, so it checks the derived quantities as
    well as the bits: `count_ones` reads whole bytes including the tail past the
    logical length, and is the operation most likely to disagree with a bit by bit
    walk.

    Args:
        bitmap: The bitmap under test.
        model: The reference bits.
        step: The case number, for the failure message.
        seed: The seed, for the failure message.
        what: The operation that was just applied.

    Raises:
        If the bitmap and the reference disagree anywhere.
    """
    if len(bitmap) != len(model):
        raise Error(
            String(
                "step ",
                step,
                " seed ",
                seed,
                " after ",
                what,
                ": length ",
                len(bitmap),
                " but reference has ",
                len(model),
            )
        )

    var ones = 0
    for i in range(len(model)):
        if model[i]:
            ones += 1
        if bitmap.get(i) != model[i]:
            raise Error(
                String(
                    "step ",
                    step,
                    " seed ",
                    seed,
                    " after ",
                    what,
                    ": bit ",
                    i,
                    " is ",
                    bitmap.get(i),
                    " but reference has ",
                    model[i],
                    " (length ",
                    len(model),
                    ")",
                )
            )

    if bitmap.count_ones() != ones:
        raise Error(
            String(
                "step ",
                step,
                " seed ",
                seed,
                " after ",
                what,
                ": count_ones is ",
                bitmap.count_ones(),
                " but reference has ",
                ones,
                " (length ",
                len(model),
                ")",
            )
        )
    if bitmap.null_count() != len(model) - ones:
        raise Error(
            String(
                "step ",
                step,
                " seed ",
                seed,
                " after ",
                what,
                ": null_count is ",
                bitmap.null_count(),
                " but reference has ",
                len(model) - ones,
            )
        )
    if bitmap.all_valid() != (ones == len(model)):
        raise Error(
            String(
                "step ",
                step,
                " seed ",
                seed,
                " after ",
                what,
                ": all_valid disagrees",
            )
        )
    if bitmap.any_valid() != (ones > 0):
        raise Error(
            String(
                "step ",
                step,
                " seed ",
                seed,
                " after ",
                what,
                ": any_valid disagrees",
            )
        )


def main() raises:
    var options = parse_options()
    print(
        "fuzzing bitmap:",
        options.cases,
        "cases, seed",
        options.seed,
        "max_seconds",
        options.max_seconds,
    )

    var rng = Rng(options.seed)
    var length = rng.next_range(1, MAX_LENGTH)
    var bitmap = Bitmap(length)
    var model = List[Bool](length=length, fill=True)

    var started = perf_counter_ns()
    var applied = 0
    var stopped_early = False

    for step in range(options.cases):
        # A time check is a syscall on some platforms, so it is amortized over a
        # block of cases rather than paid per case.
        if options.max_seconds > 0.0 and step % 65536 == 0 and step > 0:
            var elapsed = Float64(perf_counter_ns() - started) / 1.0e9
            if elapsed >= options.max_seconds:
                stopped_early = True
                break

        var op = rng.next_below(100)

        if op < 45:
            # The common case: a single bit, checked at a single position. Cheap
            # enough that ten million of them fit in a CI step.
            var i = rng.next_below(len(model))
            var value = rng.next_bool()
            bitmap.set(i, value)
            model[i] = value
            if bitmap.get(i) != value:
                raise Error(
                    String(
                        "step ",
                        step,
                        " seed ",
                        options.seed,
                        ": set(",
                        i,
                        ", ",
                        value,
                        ") did not take",
                    )
                )
            if step % 4096 == 0:
                check(bitmap, model, step, options.seed, "set")

        elif op < 60:
            var a = rng.next_below(len(model) + 1)
            var b = rng.next_below(len(model) + 1)
            var start = a if a <= b else b
            var end = b if a <= b else a
            var value = rng.next_bool()
            bitmap.set_range(start, end, value)
            reference_set_range(model, start, end, value)
            check(bitmap, model, step, options.seed, "set_range")

        elif op < 65:
            bitmap.set_all()
            for i in range(len(model)):
                model[i] = True
            check(bitmap, model, step, options.seed, "set_all")

        elif op < 70:
            bitmap.clear_all()
            for i in range(len(model)):
                model[i] = False
            check(bitmap, model, step, options.seed, "clear_all")

        elif op < 76:
            bitmap.invert()
            for i in range(len(model)):
                model[i] = not model[i]
            check(bitmap, model, step, options.seed, "invert")

        elif op < 84:
            var other = Bitmap(len(model), all_valid=False)
            var other_model = List[Bool](length=len(model), fill=False)
            for i in range(len(model)):
                var bit = rng.next_bool()
                other.set(i, bit)
                other_model[i] = bit

            if rng.next_bool():
                bitmap.and_with(other)
                for i in range(len(model)):
                    model[i] = model[i] and other_model[i]
                check(bitmap, model, step, options.seed, "and_with")
            else:
                bitmap.or_with(other)
                for i in range(len(model)):
                    model[i] = model[i] or other_model[i]
                check(bitmap, model, step, options.seed, "or_with")

        elif op < 92:
            var a = rng.next_below(len(model) + 1)
            var b = rng.next_below(len(model) + 1)
            var start = a if a <= b else b
            var end = b if a <= b else a
            var part = bitmap.slice(start, end)
            var part_model = List[Bool]()
            for i in range(start, end):
                part_model.append(model[i])
            check(part, part_model, step, options.seed, "slice")
            # The source must be untouched by slicing it.
            check(bitmap, model, step, options.seed, "slice source")

        elif op < 96:
            var duplicate = Bitmap(copy=bitmap)
            check(duplicate, model, step, options.seed, "copy")
            duplicate.invert()
            check(bitmap, model, step, options.seed, "copy source")

        else:
            # Start over at a new length. This is what gets every length modulo
            # eight and modulo sixty four in front of every operation.
            length = rng.next_range(1, MAX_LENGTH)
            var all_valid = rng.next_bool()
            bitmap = Bitmap(length, all_valid=all_valid)
            model = List[Bool](length=length, fill=all_valid)
            check(bitmap, model, step, options.seed, "reallocate")

        applied += 1

    var elapsed_ns = perf_counter_ns() - started
    check(bitmap, model, applied, options.seed, "final")

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
