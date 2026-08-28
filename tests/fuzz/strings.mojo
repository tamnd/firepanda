"""Differential fuzzing of the string column against a list of strings.

The string column is two buffers that have to stay in step. The views array says
where each element is and how long it is, the payload holds the bytes of the long
ones, and every operation that produces a new column rebuilds both. A unit test
picks a handful of lengths and the bug is always at the length it did not pick,
which for this layout is any length either side of twelve.

So the reference is a `List[String]` beside a `List[Bool]`, with no buffers and no
offsets in it at all, and the harness applies the same operation to both and
compares every element. Lengths are drawn from a range that straddles the inline
limit heavily, because the boundary is where the layout can be wrong.

A failure prints the seed and the case number, and re-running with `--seed=N`
replays the run exactly.

Usage:
    mojo run -I . tests/fuzz/strings.mojo [--cases=N] [--seed=N]
        [--max-total-time=SECONDS]
"""

from std.sys import argv
from std.time import perf_counter_ns

from firepanda.array.strings import StringArray, StringBuilder
from firepanda.bitmap.bitmap import Bitmap
from firepanda.testing.rng import Rng

comptime DEFAULT_CASES = 1_000_000
"""Fewer cases than the bitmap fuzzer, because a case here rebuilds a column and
then walks every element of it rather than checking one bit."""

comptime MAX_ROWS = 137
"""A prime, so a column length is rarely a multiple of anything."""

comptime MAX_BYTES = 29
"""Long enough to reach past the twelve byte inline limit twice over."""


struct Options(Copyable, Movable):
    """What the harness was asked to do."""

    var cases: Int
    """The number of rounds to run."""

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


def random_text(mut rng: Rng) -> String:
    """Draws one element's bytes.

    The length is drawn so that a third of the elements land within a byte or two
    of the inline limit, since that is where the layout decides which of its two
    storage paths an element takes.

    Args:
        rng: The generator.

    Returns:
        The text.
    """
    var length: Int
    if rng.next_below(3) == 0:
        length = rng.next_range(10, 15)
    else:
        length = rng.next_below(MAX_BYTES + 1)
    var text = String()
    for _ in range(length):
        text += chr(97 + Int(rng.next_below(26)))
    return text^


def check(
    column: StringArray,
    values: List[String],
    present: List[Bool],
    step: Int,
    seed: UInt64,
    what: String,
) raises:
    """Compares the column against the reference and raises on the first difference.

    Every element is checked three ways: its validity, its length, and its bytes.
    The length is checked separately from the bytes because it comes from the view
    while the bytes may come from the payload, and a column can get one right
    while getting the other wrong.

    Args:
        column: The column under test.
        values: The reference elements. A null is the empty string.
        present: The reference validity.
        step: The case number, for the failure message.
        seed: The seed, for the failure message.
        what: The operation that was just applied.

    Raises:
        If the column and the reference disagree anywhere.
    """
    if len(column) != len(values):
        raise Error(
            String(
                "step ",
                step,
                " seed ",
                seed,
                " after ",
                what,
                ": length ",
                len(column),
                " against ",
                len(values),
            )
        )

    var nulls = 0
    for i in range(len(values)):
        if not present[i]:
            nulls += 1
        if column.is_valid(i) != present[i]:
            raise Error(
                String(
                    "step ",
                    step,
                    " seed ",
                    seed,
                    " after ",
                    what,
                    ": validity at ",
                    i,
                    " is ",
                    column.is_valid(i),
                )
            )
        var expected = values[i] if present[i] else String("")
        if column.byte_length(i) != expected.byte_length():
            raise Error(
                String(
                    "step ",
                    step,
                    " seed ",
                    seed,
                    " after ",
                    what,
                    ": length at ",
                    i,
                    " is ",
                    column.byte_length(i),
                    " against ",
                    expected.byte_length(),
                )
            )
        if column[i] != expected:
            raise Error(
                String(
                    "step ",
                    step,
                    " seed ",
                    seed,
                    " after ",
                    what,
                    ": element ",
                    i,
                    " is '",
                    column[i],
                    "' against '",
                    expected,
                    "'",
                )
            )
        if present[i] and not column.equals(i, expected.as_bytes()):
            raise Error(
                String(
                    "step ",
                    step,
                    " seed ",
                    seed,
                    " after ",
                    what,
                    ": equals disagrees at ",
                    i,
                )
            )

    if column.null_count() != nulls:
        raise Error(
            String(
                "step ",
                step,
                " seed ",
                seed,
                " after ",
                what,
                ": null_count ",
                column.null_count(),
                " against ",
                nulls,
            )
        )


def build(
    mut rng: Rng, mut values: List[String], mut present: List[Bool]
) raises -> StringArray:
    """Builds a fresh random column and the reference beside it.

    Args:
        rng: The generator.
        values: Filled with the reference elements.
        present: Filled with the reference validity.

    Returns:
        The column.

    Raises:
        If the builder raises.
    """
    var rows = rng.next_below(MAX_ROWS + 1)
    values = List[String](capacity=rows)
    present = List[Bool](capacity=rows)
    var builder = StringBuilder(capacity=rows)
    for _ in range(rows):
        if rng.next_below(6) == 0:
            builder.append_null()
            values.append(String(""))
            present.append(False)
        else:
            var text = random_text(rng)
            builder.append(text.as_bytes())
            values.append(text)
            present.append(True)
    return builder^.finish()


def main() raises:
    var options = parse_options()
    print(
        "fuzzing strings:",
        options.cases,
        "cases, seed",
        options.seed,
        "max_seconds",
        options.max_seconds,
    )

    var rng = Rng(options.seed)
    var values = List[String]()
    var present = List[Bool]()
    var column = build(rng, values, present)
    check(column, values, present, 0, options.seed, "build")

    var started = perf_counter_ns()
    var applied = 0
    var stopped_early = False

    for step in range(options.cases):
        # A time check is a syscall on some platforms, so it is amortized over a
        # block of cases rather than paid per case.
        if options.max_seconds > 0.0 and step % 256 == 0 and step > 0:
            var elapsed = Float64(perf_counter_ns() - started) / 1.0e9
            if elapsed >= options.max_seconds:
                stopped_early = True
                break

        var op = rng.next_below(100)

        if op < 25 and len(values) > 0:
            var a = rng.next_below(len(values) + 1)
            var b = rng.next_below(len(values) + 1)
            var start = a if a <= b else b
            var end = b if a <= b else a
            var cut_values = List[String](capacity=end - start)
            var cut_present = List[Bool](capacity=end - start)
            for i in range(start, end):
                cut_values.append(values[i])
                cut_present.append(present[i])
            var cut = column.slice(start, end)
            check(cut, cut_values, cut_present, step, options.seed, "slice")
            column = cut^
            values = cut_values^
            present = cut_present^

        elif op < 50 and len(values) > 0:
            # Repeats and reversals, because a gather that walks forward through
            # the payload can be right while a gather that jumps is wrong.
            var count = rng.next_below(MAX_ROWS + 1)
            var picks = List[Int](capacity=count)
            var took_values = List[String](capacity=count)
            var took_present = List[Bool](capacity=count)
            for _ in range(count):
                var i = rng.next_below(len(values))
                picks.append(i)
                took_values.append(values[i])
                took_present.append(present[i])
            var took = column.take(picks)
            check(took, took_values, took_present, step, options.seed, "take")
            column = took^
            values = took_values^
            present = took_present^

        elif op < 70 and len(values) > 0:
            var mask = Bitmap(len(values))
            mask.clear_all()
            var kept_values = List[String]()
            var kept_present = List[Bool]()
            for i in range(len(values)):
                if rng.next_bool():
                    mask.set(i, True)
                    kept_values.append(values[i])
                    kept_present.append(present[i])
            var kept = column.filter(mask)
            check(kept, kept_values, kept_present, step, options.seed, "filter")
            column = kept^
            values = kept_values^
            present = kept_present^

        elif op < 80:
            var duplicate = StringArray(copy=column)
            check(duplicate, values, present, step, options.seed, "copy")
            # The copy has to survive the source going away, which is the whole
            # reason these operations copy instead of pointing at a payload.
            column = duplicate^
            check(column, values, present, step, options.seed, "copy source")

        elif op < 90 and len(values) > 1:
            # Every pair of a small column compared both ways, against what the
            # reference says. This is the only check that reaches the prefix
            # comparison in the view.
            var limit = len(values) if len(values) < 24 else 24
            for i in range(limit):
                for j in range(limit):
                    var expected = (
                        present[i] and present[j] and values[i] == values[j]
                    )
                    if column.element_equals(i, j) != expected:
                        raise Error(
                            String(
                                "step ",
                                step,
                                " seed ",
                                options.seed,
                                ": element_equals(",
                                i,
                                ", ",
                                j,
                                ") is ",
                                column.element_equals(i, j),
                            )
                        )

        else:
            column = build(rng, values, present)
            check(column, values, present, step, options.seed, "rebuild")

        applied += 1

    var elapsed_ns = perf_counter_ns() - started
    check(column, values, present, applied, options.seed, "final")

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
