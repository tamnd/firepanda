"""Reductions over a column: sum, min, max, count and mean.

Two shapes of loop appear here and the difference between them is the whole
reason this file is longer than it looks like it should be.

`sum_of` never reads the validity bitmap. It does not have to, because a null
holds a zero and zero is the identity for addition. So the loop is a straight
vectorized add over the values buffer, and on a million int64 rows that is 99 us
against the 471 us a per-element validity check costs.

`min_of` and `max_of` cannot do that. Zero is not an identity for either, and a
column of positive numbers with one null in it would report a minimum of zero.
They walk the validity bitmap a 64-bit word at a time instead and take one of
three paths per word: all present, vectorize the whole block; all null, skip the
whole block; mixed, fall back to a bit test per value. Real columns are mostly
one of the first two, and the third is only ever paid for on the boundary.

`count_of` is a popcount and does not look at the values at all.

On a float dtype all of them step over a NaN as well as over a null, because
that is what pandas does and a NaN is one of pandas' two spellings of missing.
The sum turns a NaN into a zero and the extremes turn it into their own identity,
both with a compare and a select per vector, so the loop shape is unchanged and
the branch is gone before the loop exists on every other dtype. The mean needs
the divisor as well as the total, so `_sum_range` hands back how many values it
stepped over and `mean_over` subtracts that from the count it was given. That is
one pass and not two, which is the whole reason the count is threaded through
rather than being asked for separately. `count_of` is the exception and still
counts bits and nothing else, because it is the Arrow answer; the pandas one is
`nulls.missing_count_any`. See #170.

Every reduction comes in two spellings. The `_of` one takes a typed column and
is what a caller with an `Array[dt]` in hand wants. The `_over` one takes the
values pointer, the validity and the row count, and is what a caller holding an
`AnyArray` wants, because turning one of those into an `Array[dt]` copies the
whole column and the reduction was supposed to be one pass over it. `reduce.mojo`
is the caller that needed the second spelling. The loop lives in the `_over` one
and the `_of` one is three lines that unpack the column.

A reduction longer than one morsel runs on every core. It is not a split the way
an elementwise kernel is a split, because there is one answer and not one answer
per row, so each morsel reduces its own rows into a slot of its own and a serial
loop combines the slots afterwards. The combine is over the number of morsels
rather than the number of rows, which is seventy six slots for ten million rows,
so it costs nothing next to the pass it replaces. A minimum needs a second slot
per morsel saying whether that morsel saw a value at all, because a morsel that
is entirely null has no value that could stand for it and the identity would be
a wrong answer if every morsel were like that.

The one thing this changes about the answers is the order the additions happen
in, which is only visible in floating point. A sum was already not adding left to
right, because the vector unit keeps one running total per lane, and now the lane
totals are per morsel as well. The combine runs in morsel order rather than in
whichever order the workers finished, so the answer is at least the same on every
run for the same input. It is not bit identical to what a single scalar loop
would produce, and neither was the vectorized version.

Every function here has a twin in `scalar.mojo` and the fuzz harness runs them
against each other. See the note on the null-is-zero invariant in `__init__.mojo`.
"""

from std.math import isnan
from std.sys.info import simd_width_of

from firepanda.array.array import Array
from firepanda.bitmap.bitmap import Bitmap
from firepanda.exec import MORSEL_ROWS, parallel_morsels

from .accum import accumulator, highest, lowest

comptime BLOCK = 64
"""Values covered by one validity word. Not tunable; it is what a word holds."""


struct AggResult[dt: DType](Copyable, Movable):
    """The value a reduction produced, and whether it produced one at all.

    A minimum over a column with no non-null values does not have an answer, and
    returning a sentinel would make the caller guess which sentinel. This says so
    instead.
    """

    var value: Scalar[Self.dt]
    """The result. Meaningless unless `valid`."""

    var valid: Bool
    """Whether the reduction had at least one non-null value to work with."""

    def __init__(out self, value: Scalar[Self.dt], valid: Bool):
        """Constructs a result.

        Args:
            value: The reduced value.
            valid: Whether the value means anything.
        """
        self.value = value
        self.valid = valid

    @staticmethod
    def none() -> Self:
        """Returns the result of a reduction that saw no values.

        Returns:
            An invalid result.
        """
        return Self(Scalar[Self.dt](0), False)


@fieldwise_init
struct SumResult[acc: DType](Copyable, Movable):
    """A total, and how many values were left out of it for being NaN.

    A mean needs both, and getting the second one any other way means a second
    pass over the values. On any dtype that is not floating point `skipped` is
    zero and the loop that would have counted is compiled away.
    """

    var total: Scalar[Self.acc]
    """The sum of everything that was not NaN."""

    var skipped: Int
    """How many values were NaN and so contributed nothing."""


def count_of[dt: DType](col: Array[dt]) -> Int:
    """Returns the number of set validity bits in a column.

    This is the Arrow answer and not the pandas one. On a float column pandas
    would also count a NaN as missing, which this does not, because an `Array` is
    the Arrow half of the library and says what is in the buffers. The pandas
    answer is `nulls.missing_count_any`, and it is what `Series.count` uses. See
    #170.

    Args:
        col: The column.

    Parameters:
        dt: The column's dtype.

    Returns:
        The count of set validity bits.
    """
    return col.data.validity.count_ones()


def sum_of[dt: DType](col: Array[dt]) raises -> AggResult[accumulator(dt)]:
    """Adds up the non-null values of a column.

    The nulls are added too. They are zero, so it makes no difference to the
    answer and it removes the validity bitmap from the inner loop entirely. See
    `__init__.mojo` for why that invariant is safe to lean on.

    A NaN is not added, on a float dtype, because pandas steps over one exactly
    as it steps over a value that was never there. It is turned into a zero on
    the way into the accumulator, which is the same trick and the same reason.

    The sum of an empty column, and the sum of a column that is entirely null, is
    zero and is valid. That is what pandas does, and it is a genuinely awkward
    choice, but disagreeing with pandas about it would be worse.

    Args:
        col: The column.

    Parameters:
        dt: The column's dtype.

    Returns:
        The total, widened to int64, uint64 or float64.

    Raises:
        Error: Only what the morsel runtime raises.
    """
    return sum_over(col.unsafe_ptr(), len(col))


def sum_over[
    dt: DType, //, origin: ImmOrigin, acc: DType = accumulator(dt)
](source: Pointer[Scalar[dt], origin], n: Int) raises -> AggResult[acc]:
    """Adds up `n` values, nulls included, because a null holds a zero.

    Past one morsel this runs on every core. Each morsel adds up its own rows
    into a slot of its own and the totals are added together afterwards, in
    order, on one thread.

    The accumulator is a parameter with the natural widening as its default, and
    it is a parameter because the mean needs a different one. A sum over int64
    accumulates in int64 and wraps, which is what pandas does and is therefore
    the answer. A mean must not be computed from that wrapped total: pandas
    converts to float64 first and divides, so the mean of a column of large
    int64 values is a large float rather than whatever the wrap happened to
    leave behind. See `mean_over`.

    Args:
        source: The values.
        n: How many of them.

    Parameters:
        dt: The value dtype.
        origin: Where the values live.
        acc: What to accumulate in. Defaults to int64, uint64 or float64.

    Returns:
        The total, in the accumulator type.

    Raises:
        Error: Only what the morsel runtime raises.
    """
    return AggResult[acc](sum_and_skipped_over[acc=acc](source, n).total, True)


def sum_and_skipped_over[
    dt: DType, //, origin: ImmOrigin, acc: DType = accumulator(dt)
](source: Pointer[Scalar[dt], origin], n: Int) raises -> SumResult[acc]:
    """Adds up `n` values and says how many of them were NaN.

    This is `sum_over` with the second number kept rather than thrown away, and
    the mean is the caller that wants it. Everything the docstring above says
    about the accumulator and about the order of the additions applies here,
    because this is the function that does the work and that one is a wrapper.

    Args:
        source: The values.
        n: How many of them.

    Parameters:
        dt: The value dtype.
        origin: Where the values live.
        acc: What to accumulate in. Defaults to int64, uint64 or float64.

    Returns:
        The total and the number of NaN values that were stepped over.

    Raises:
        Error: Only what the morsel runtime raises.
    """
    if n <= MORSEL_ROWS:
        return _sum_range[acc=acc](source, 0, n)

    var count = (n + MORSEL_ROWS - 1) // MORSEL_ROWS
    var partials = Array[acc](count)

    # One slot per morsel for the NaN count as well, which is a few hundred bytes
    # next to the column being read and is allocated on every dtype rather than
    # only on the floating point ones. Making the allocation conditional would
    # mean two spellings of this loop for a saving of one small array.
    var passed = Array[DType.int64](count)

    # The closure captures the column and takes the pointer inside itself, which
    # is not a preference. Hoisting the pointer out and capturing that instead
    # makes the compiler emit invalid code for the call into the morsel runtime,
    # and it fails in the backend rather than at the call, so it is worth naming
    # here. Every other kernel in this package captures the column the same way.
    def add_up(start: Int, stop: Int) {mut partials, mut passed, imm}:
        var found = _sum_range[acc=acc](source, start, stop)
        var at = start // MORSEL_ROWS
        partials.unsafe_ptr().unsafe_offset(at).unsafe_write(found.total)
        passed.unsafe_ptr().unsafe_offset(at).unsafe_write(Int64(found.skipped))

    parallel_morsels(add_up, n)

    var slots = partials.unsafe_ptr()
    var skips = passed.unsafe_ptr()

    # Added back in morsel order rather than in whatever order the workers
    # happened to finish, so the answer does not depend on the scheduling. It
    # still is not the order a single loop would have used, which is the note in
    # the module docstring about floating point.
    var total = Scalar[acc](0)
    var skipped = 0
    for i in range(count):
        total += slots.unsafe_offset(i).unsafe_load()
        skipped += Int(skips.unsafe_offset(i).unsafe_load())
    return SumResult[acc](total, skipped)


def _sum_range[
    dt: DType, //, origin: ImmOrigin, acc: DType = accumulator(dt)
](source: Pointer[Scalar[dt], origin], start: Int, stop: Int) -> SumResult[acc]:
    """Adds up one range of values, on one thread.

    On a float dtype a NaN is turned into a zero before it reaches the
    accumulator and is counted instead, which is two extra vector instructions on
    a loop that is waiting on memory. On every other dtype the two `comptime if`
    blocks are not there at all and the loop is what it was.

    Args:
        source: The values.
        start: The first row to add.
        stop: One past the last row to add.

    Parameters:
        dt: The value dtype.
        origin: Where the values live.
        acc: What to accumulate in.

    Returns:
        The total over the range and how many NaN values it stepped over.
    """
    comptime width = simd_width_of[dt]()

    var ptr = source
    var lanes = SIMD[acc, width](0)
    var nans = SIMD[DType.int64, width](0)
    var i = start
    while i + width <= stop:
        var chunk = ptr.unsafe_offset(i).unsafe_load[width=width]().cast[acc]()
        comptime if dt.is_floating_point():
            var missing = isnan(chunk)
            nans += missing.cast[DType.int64]()
            chunk = missing.select(SIMD[acc, width](0), chunk)
        lanes += chunk
        i += width

    var total = lanes.reduce_add()
    var skipped = Int(nans.reduce_add())
    while i < stop:
        var value = Scalar[acc](ptr.unsafe_offset(i).unsafe_load())
        comptime if dt.is_floating_point():
            if isnan(value):
                skipped += 1
                i += 1
                continue
        total += value
        i += 1
    return SumResult[acc](total, skipped)


def min_of[dt: DType](col: Array[dt]) raises -> AggResult[dt]:
    """Returns the smallest non-null value in a column.

    Args:
        col: The column.

    Parameters:
        dt: The column's dtype.

    Returns:
        The minimum, or an invalid result if every value is null.

    Raises:
        Error: Only what the morsel runtime raises.
    """
    return extreme_over[want_min=True](
        col.unsafe_ptr(), col.data.validity, len(col)
    )


def max_of[dt: DType](col: Array[dt]) raises -> AggResult[dt]:
    """Returns the largest non-null value in a column.

    Args:
        col: The column.

    Parameters:
        dt: The column's dtype.

    Returns:
        The maximum, or an invalid result if every value is null.

    Raises:
        Error: Only what the morsel runtime raises.
    """
    return extreme_over[want_min=False](
        col.unsafe_ptr(), col.data.validity, len(col)
    )


def extreme_over[
    dt: DType, //, origin: ImmOrigin, want_min: Bool
](
    source: Pointer[Scalar[dt], origin], validity: Bitmap, n: Int
) raises -> AggResult[dt]:
    """Reduces `n` values to the smallest or largest non-null one.

    Min and max differ by one comparison and nothing else, so they share a body
    parameterized on which one it is. `want_min` is a parameter rather than an
    argument so the branch is gone before the loop exists.

    On a float dtype a NaN is not a candidate, because pandas steps over one. It
    is replaced by the identity, which loses every comparison it is in, and the
    range keeps a separate note of whether it saw anything real. A column that is
    entirely NaN therefore reports that it found nothing, the same as a column
    that is entirely null, and `reduce.mojo` turns that into the NaN pandas
    would have given.

    Past one morsel this runs on every core. Each morsel reduces its own rows
    into a slot of its own, and a morsel that saw nothing but nulls says so in a
    second slot, because there is no value that could stand for it.

    Args:
        source: The values.
        validity: Which of them are present.
        n: How many of them.

    Parameters:
        dt: The value dtype.
        origin: Where the values live.
        want_min: True for a minimum, False for a maximum.

    Returns:
        The extreme value, or an invalid result if every value is null.

    Raises:
        Error: Only what the morsel runtime raises.
    """
    if n <= MORSEL_ROWS:
        return _extreme_range[want_min=want_min](source, validity, 0, n)

    var count = (n + MORSEL_ROWS - 1) // MORSEL_ROWS
    var values = Array[dt](count)
    var flags = Array[DType.uint8](count)

    def reduce_one(start: Int, stop: Int) {mut values, mut flags, imm}:
        var found = _extreme_range[want_min=want_min](
            source, validity, start, stop
        )
        var at = start // MORSEL_ROWS
        values.unsafe_ptr().unsafe_offset(at).unsafe_write(found.value)
        flags.unsafe_ptr().unsafe_offset(at).unsafe_write(
            UInt8(1) if found.valid else UInt8(0)
        )

    parallel_morsels(reduce_one, n)

    var value_slots = values.unsafe_ptr()
    var flag_slots = flags.unsafe_ptr()

    comptime identity = highest[dt]() if want_min else lowest[dt]()
    var best = identity
    var seen = False
    for i in range(count):
        if flag_slots.unsafe_offset(i).unsafe_load() == 0:
            continue
        seen = True
        best = _better[dt, want_min](
            best, value_slots.unsafe_offset(i).unsafe_load()
        )

    if not seen:
        return AggResult[dt].none()
    return AggResult[dt](best, True)


def _extreme_range[
    dt: DType, //, origin: ImmOrigin, want_min: Bool
](
    source: Pointer[Scalar[dt], origin], validity: Bitmap, start: Int, stop: Int
) -> AggResult[dt]:
    """Reduces one range of values to the smallest or largest non-null one.

    Args:
        source: The values.
        validity: Which of them are present.
        start: The first row to read. Must be a multiple of 64, which every
            morsel boundary is.
        stop: One past the last row to read.

    Parameters:
        dt: The value dtype.
        origin: Where the values live.
        want_min: True for a minimum, False for a maximum.

    Returns:
        The extreme value over the range, or an invalid result if every value in
        it is null.
    """
    comptime width = simd_width_of[dt]()
    comptime identity = highest[dt]() if want_min else lowest[dt]()

    var ptr = source
    var best = identity
    var seen = False

    # The bitmap is read through the argument rather than bound to a local. A
    # local would be a copy, and `Bitmap` owns an allocation, so binding it here
    # would allocate and memcpy the whole validity of the column before the loop
    # that is supposed to be reading it cheaply had started.
    var first = start // BLOCK
    var limit = min((stop + BLOCK - 1) // BLOCK, validity.word_count())
    for w in range(first, limit):
        var word = validity.unsafe_word(w)
        if word == 0:
            continue

        var base = w * BLOCK
        var last = base + BLOCK
        if last > stop:
            last = stop

        if word == UInt64.MAX and last == base + BLOCK:
            # The whole block is present, so the values can go through the vector
            # unit without a single bit test. On a float dtype present is not the
            # same thing as there, so the word alone no longer settles whether
            # anything was found and the loop below has to say so.
            comptime if not dt.is_floating_point():
                seen = True
            var i = base

            # Booleans skip the vector unit and fall straight through to the
            # loop below. A SIMD minimum over booleans is spelled as a floating
            # point reduction in the standard library and does not compile on
            # every target, which is a fair thing for it to do: a column with two
            # possible values has nothing to gain from sixteen lanes.
            comptime if dt != DType.bool:
                var lanes = SIMD[dt, width](identity)
                var real = SIMD[DType.bool, width](fill=False)
                while i + width <= last:
                    var chunk = ptr.unsafe_offset(i).unsafe_load[width=width]()

                    comptime if dt.is_floating_point():
                        # A NaN becomes the identity, which can never win, and
                        # the lanes that held something real are remembered on
                        # the side. They cannot be read back out of the fold
                        # afterwards, because a block of nothing but NaN folds to
                        # the identity and so does a block holding one genuine
                        # infinity.
                        var missing = isnan(chunk)
                        chunk = missing.select(SIMD[dt, width](identity), chunk)
                        real = real | ~missing

                    comptime if want_min:
                        lanes = min(lanes, chunk)
                    else:
                        lanes = max(lanes, chunk)
                    i += width

                comptime if dt.is_floating_point():
                    if real.reduce_or():
                        seen = True
                var folded = (
                    lanes.reduce_min() if want_min else lanes.reduce_max()
                )
                best = _better[dt, want_min](best, folded)

            while i < last:
                var value = ptr.unsafe_offset(i).unsafe_load()
                comptime if dt.is_floating_point():
                    if isnan(value):
                        i += 1
                        continue
                    seen = True
                best = _better[dt, want_min](best, value)
                i += 1
            continue

        for i in range(base, last):
            if (word >> UInt64(i - base)) & 1 == 0:
                continue
            var value = ptr.unsafe_offset(i).unsafe_load()
            comptime if dt.is_floating_point():
                if isnan(value):
                    continue
            seen = True
            best = _better[dt, want_min](best, value)

    if not seen:
        return AggResult[dt].none()
    return AggResult[dt](best, True)


def _better[
    dt: DType, want_min: Bool
](a: Scalar[dt], b: Scalar[dt]) -> Scalar[dt]:
    """Returns whichever of two values the reduction is looking for.

    Args:
        a: One value.
        b: The other.

    Parameters:
        dt: The dtype.
        want_min: True to keep the smaller, False to keep the larger.

    Returns:
        The kept value.
    """
    comptime if want_min:
        return a if a < b else b
    return a if a > b else b


def mean_of[dt: DType](col: Array[dt]) raises -> AggResult[DType.float64]:
    """Returns the arithmetic mean of the non-null values.

    The divisor is the non-null count, not the length, which is what pandas does
    with `skipna=True` and what everyone means by an average of a column with
    gaps in it.

    Args:
        col: The column.

    Parameters:
        dt: The column's dtype.

    Returns:
        The mean, or an invalid result if every value is null.

    Raises:
        Error: Only what the morsel runtime raises.
    """
    return mean_over(col.unsafe_ptr(), col.data.validity.count_ones(), len(col))


def mean_over[
    dt: DType, //, origin: ImmOrigin
](
    source: Pointer[Scalar[dt], origin], present: Int, n: Int
) raises -> AggResult[DType.float64]:
    """Returns the mean of the values that are present.

    The count comes in as an argument rather than being derived here, because
    every caller already knows it. A frame knows its null count and a grouped
    reduction has just counted, so recomputing a popcount over the whole bitmap
    would be a second pass bought for nothing.

    Args:
        source: The values.
        present: How many of them are not null.
        n: How many there are in total.

    Parameters:
        dt: The value dtype.
        origin: Where the values live.

    Returns:
        The mean, or an invalid result if every value is null.

    Raises:
        Error: Only what the morsel runtime raises.
    """
    if present == 0:
        return AggResult[DType.float64].none()
    # Summed in float64 and not in the natural accumulator, which for an integer
    # column is the difference between an answer and a wrapped one. A sum over
    # int64 wraps, and that is the right answer for a sum because pandas wraps
    # there too, but pandas computes a mean by converting to float64 first. Off
    # the corpus column of large int64 values the two differ by ten orders of
    # magnitude: 2040395725.875 out of the wrapped total against -4.6e18, which
    # is the mean of the values that are actually in the column.
    var total = sum_and_skipped_over[acc=DType.float64](source, n)

    # The divisor loses the NaNs as well as the nulls. Every NaN the sum stepped
    # over was in a row whose validity bit was set, because a null holds a zero
    # and a zero is not a NaN, so subtracting the one count from the other cannot
    # take the same row away twice. On any dtype that is not floating point the
    # subtraction is of zero.
    var divisor = present - total.skipped
    if divisor <= 0:
        return AggResult[DType.float64].none()
    return AggResult[DType.float64](total.total / Float64(divisor), True)
