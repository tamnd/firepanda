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

Every function here has a twin in `scalar.mojo` and the fuzz harness runs them
against each other. See the note on the null-is-zero invariant in `__init__.mojo`.
"""

from std.sys.info import simd_width_of

from firepanda.array.array import Array

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


def count_of[dt: DType](col: Array[dt]) -> Int:
    """Returns the number of non-null values in a column.

    Args:
        col: The column.

    Parameters:
        dt: The column's dtype.

    Returns:
        The count of set validity bits.
    """
    return col.data.validity.count_ones()


def sum_of[dt: DType](col: Array[dt]) -> AggResult[accumulator(dt)]:
    """Adds up the non-null values of a column.

    The nulls are added too. They are zero, so it makes no difference to the
    answer and it removes the validity bitmap from the inner loop entirely. See
    `__init__.mojo` for why that invariant is safe to lean on.

    The sum of an empty column, and the sum of a column that is entirely null, is
    zero and is valid. That is what pandas does, and it is a genuinely awkward
    choice, but disagreeing with pandas about it would be worse.

    Args:
        col: The column.

    Parameters:
        dt: The column's dtype.

    Returns:
        The total, widened to int64, uint64 or float64.
    """
    comptime acc = accumulator(dt)
    comptime width = simd_width_of[dt]()

    var ptr = col.unsafe_ptr()
    var n = len(col)
    var lanes = SIMD[acc, width](0)
    var i = 0
    while i + width <= n:
        lanes += ptr.unsafe_offset(i).unsafe_load[width=width]().cast[acc]()
        i += width

    var total = lanes.reduce_add()
    while i < n:
        total += Scalar[acc](ptr.unsafe_offset(i).unsafe_load())
        i += 1
    return AggResult[acc](total, True)


def min_of[dt: DType](col: Array[dt]) -> AggResult[dt]:
    """Returns the smallest non-null value in a column.

    Args:
        col: The column.

    Parameters:
        dt: The column's dtype.

    Returns:
        The minimum, or an invalid result if every value is null.
    """
    return _extreme[dt, want_min=True](col)


def max_of[dt: DType](col: Array[dt]) -> AggResult[dt]:
    """Returns the largest non-null value in a column.

    Args:
        col: The column.

    Parameters:
        dt: The column's dtype.

    Returns:
        The maximum, or an invalid result if every value is null.
    """
    return _extreme[dt, want_min=False](col)


def _extreme[dt: DType, want_min: Bool](col: Array[dt]) -> AggResult[dt]:
    """Reduces a column to its smallest or largest non-null value.

    Min and max differ by one comparison and nothing else, so they share a body
    parameterized on which one it is. `want_min` is a parameter rather than an
    argument so the branch is gone before the loop exists.

    Args:
        col: The column.

    Parameters:
        dt: The column's dtype.
        want_min: True for a minimum, False for a maximum.

    Returns:
        The extreme value, or an invalid result if every value is null.
    """
    comptime width = simd_width_of[dt]()
    comptime identity = highest[dt]() if want_min else lowest[dt]()

    var ptr = col.unsafe_ptr()
    var n = len(col)
    var best = identity
    var seen = False

    # The bitmap is read through the column rather than bound to a local. A local
    # would be a copy, and `Bitmap` owns an allocation, so binding it here would
    # allocate and memcpy the whole validity of the column before the loop that
    # is supposed to be reading it cheaply had started.
    for w in range(col.data.validity.word_count()):
        var word = col.data.validity.unsafe_word(w)
        if word == 0:
            continue

        var base = w * BLOCK
        var last = base + BLOCK
        if last > n:
            last = n

        if word == UInt64.MAX and last == base + BLOCK:
            # The whole block is present, so the values can go through the vector
            # unit without a single bit test.
            seen = True
            var lanes = SIMD[dt, width](identity)
            var i = base
            while i + width <= last:
                var chunk = ptr.unsafe_offset(i).unsafe_load[width=width]()

                comptime if want_min:
                    lanes = min(lanes, chunk)
                else:
                    lanes = max(lanes, chunk)
                i += width
            var folded = lanes.reduce_min() if want_min else lanes.reduce_max()
            best = _better[dt, want_min](best, folded)
            while i < last:
                best = _better[dt, want_min](
                    best, ptr.unsafe_offset(i).unsafe_load()
                )
                i += 1
            continue

        for i in range(base, last):
            if (word >> UInt64(i - base)) & 1 == 0:
                continue
            seen = True
            best = _better[dt, want_min](
                best, ptr.unsafe_offset(i).unsafe_load()
            )

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


def mean_of[dt: DType](col: Array[dt]) -> AggResult[DType.float64]:
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
    """
    var present = count_of(col)
    if present == 0:
        return AggResult[DType.float64].none()
    var total = sum_of(col)
    return AggResult[DType.float64](
        Float64(total.value) / Float64(present), True
    )
