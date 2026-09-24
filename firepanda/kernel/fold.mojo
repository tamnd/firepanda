"""A whole column sum of a column and a constant, in one pass and no column.

`Reduce` folds an operation against a constant into the reduction that reads
it, so that ninety sums of ninety shifts of one column hold one shifted column
at a time rather than ninety. That still writes the shifted column: the
operation runs over the whole chunk into a new buffer, and the sum reads the
buffer back. On ClickBench q29 that is ninety buffers per chunk, each one a
cast of the source to the common type, a pass of the operation, and a pass of
the sum.

This runs the operation inside the sum. Every row is read, converted, given
the operation and added, the same as before, and nothing is written between
the two. It is the fusion q29 exists to measure, and it is not the algebra that
query must not be answered with: `SUM(x + c)` is not rewritten as
`SUM(x) + c * COUNT(x)` here or anywhere, and each of the ninety sums does its
own additions over every row.

Only integers take this route. An integer sum wraps in its accumulator and
wrapping addition does not care about order, so the fused loop answers exactly
what the two passes answered. A float sum depends on the order its additions
happen in, and matching the two pass route bit for bit there means matching its
loop, which is a promise this would have to keep in two places.
"""

from std.sys.info import simd_width_of

from firepanda.array.any import AnyArray
from firepanda.array.array import Array
from firepanda.array.value import Value
from firepanda.dtype.logical import promote
from firepanda.exec import MORSEL_ROWS, parallel_morsels

from .accum import accumulator
from .binary import BinaryOp, binary_type, resolve_constant
from .group import AggKind

comptime INTEGRAL: List[DType] = [
    DType.int8,
    DType.int16,
    DType.int32,
    DType.int64,
    DType.uint8,
    DType.uint16,
    DType.uint32,
    DType.uint64,
]
"""The dtypes a fused sum reads and computes in. Integers only, see above."""


def reduce_value_any(
    a: AnyArray,
    b: Value,
    op: BinaryOp,
    value_on_left: Bool,
    kind: AggKind,
    as_float: Bool,
) raises -> Optional[AnyArray]:
    """Reduces `a op b` to one row without building `a op b`.

    None when the pair is one this does not fuse, and then the caller builds
    the column with `binary_value_any` and reduces it with `reduce_any`, which
    is what it did before this existed. What fuses is a sum or a count, not
    taken as a float, of an addition, a subtraction or a multiplication, over
    a flat integer column and a present constant whose common type is an
    integer and is the type of the answer.

    Args:
        a: The column.
        b: The constant.
        op: The operation.
        value_on_left: True for `5 - x` rather than `x - 5`.
        kind: The reduction.
        as_float: True if the sum is to be taken in float64, which does not
            fuse.

    Returns:
        The one row `reduce_any(binary_value_any(a, b, op, value_on_left),
        kind, as_float)` answers, or None.

    Raises:
        If the constant cannot be read at the column's type, which
        `binary_value_any` would have raised too.
    """
    if kind != AggKind.SUM and kind != AggKind.COUNT:
        return None
    if as_float:
        return None
    if op != BinaryOp.ADD and op != BinaryOp.SUB and op != BinaryOp.MUL:
        return None
    if not a.is_flat() or not a.type.is_integer():
        return None
    var scalar = resolve_constant(a.type, b, op)
    if scalar.is_null() or not scalar.type.is_integer():
        return None
    var left = a.type if not value_on_left else scalar.type
    var right = scalar.type if not value_on_left else a.type
    var answer = binary_type(op, left, right)
    # The two pass route converts the column to the common type, runs the
    # operation there and labels the result with the answer's type, so the
    # loop below is the same loop only when those two are one type.
    if not answer.is_integer() or promote(a.type, scalar.type) != answer:
        return None

    # The answer is null exactly where the column is, since the constant is
    # present and none of the three operations can fail on an integer.
    if kind == AggKind.COUNT:
        var counted = Array[DType.int64](1)
        counted[0] = Int64(len(a) - a.null_count())
        return AnyArray(counted^)

    comptime for source in INTEGRAL:
        if a.type.physical == source:
            comptime for common in INTEGRAL:
                if answer.physical == common:
                    return _sum_shifted[source, common](
                        a, scalar.as_scalar[common](), op, value_on_left
                    )
    return None


def _sum_shifted[
    source: DType, common: DType
](a: AnyArray, k: Scalar[common], op: BinaryOp, flip: Bool) raises -> AnyArray:
    """Adds up `a op k` over the present rows, converting each value on the way.

    Past one morsel this runs on every core the way `sum_over` does, one slot
    per morsel, added up in morsel order afterwards.

    Args:
        a: The column, of dtype `source`.
        k: The constant, already at the common type.
        op: An addition, a subtraction or a multiplication.
        flip: True if the constant is the left operand of the subtraction.

    Parameters:
        source: The column's dtype.
        common: The dtype the operation runs in.

    Returns:
        A column of one row in the accumulator `sum_over` would have used.

    Raises:
        Error: Only what the morsel runtime raises.
    """
    comptime acc = accumulator(common)
    var n = len(a)
    var nulls = a.null_count() > 0
    var count = max(1, (n + MORSEL_ROWS - 1) // MORSEL_ROWS)
    var partials = Array[acc](count)
    if op == BinaryOp.ADD:
        _sum_morsels[source, common, 0](a, k, flip, nulls, partials)
    elif op == BinaryOp.MUL:
        _sum_morsels[source, common, 1](a, k, flip, nulls, partials)
    else:
        _sum_morsels[source, common, 2](a, k, flip, nulls, partials)

    var total = Scalar[acc](0)
    for i in range(count):
        total += partials[i]
    var summed = Array[acc](1)
    summed[0] = total
    return AnyArray(summed^)


def _sum_morsels[
    source: DType, common: DType, code: Int
](
    a: AnyArray,
    k: Scalar[common],
    flip: Bool,
    nulls: Bool,
    mut partials: Array[accumulator(common)],
) raises:
    """Fills one partial total per morsel, on every core past one morsel.

    Args:
        a: The column.
        k: The constant.
        flip: True if the constant is the left operand of the subtraction.
        nulls: True if the column has a null anywhere.
        partials: One slot per morsel, written.

    Parameters:
        source: The column's dtype.
        common: The dtype the operation runs in.
        code: 0 for an addition, 1 for a multiplication, 2 for a subtraction.

    Raises:
        Error: Only what the morsel runtime raises.
    """
    var n = len(a)
    if n <= MORSEL_ROWS:
        partials[0] = _shifted_range[source, common, code](
            a, k, flip, nulls, 0, n
        )
        return

    def add_up(start: Int, stop: Int) {mut partials, imm}:
        var total = _shifted_range[source, common, code](
            a, k, flip, nulls, start, stop
        )
        partials.unsafe_mut_ptr().unsafe_offset(
            start // MORSEL_ROWS
        ).unsafe_write(total)

    parallel_morsels(add_up, n)


def _shifted_range[
    source: DType, common: DType, code: Int
](
    a: AnyArray,
    k: Scalar[common],
    flip: Bool,
    nulls: Bool,
    start: Int,
    stop: Int,
) -> Scalar[accumulator(common)]:
    """Adds up `a op k` over one range of rows, on one thread.

    The operation runs at the common type and wraps there, and only then is the
    result widened into the accumulator, because that is the order the two pass
    route does it in: the shifted column is `common`, and its sum widens.

    Args:
        a: The column.
        k: The constant.
        flip: True if the constant is the left operand of the subtraction.
        nulls: True if the column has a null anywhere.
        start: The first row.
        stop: One past the last row.

    Parameters:
        source: The column's dtype.
        common: The dtype the operation runs in.
        code: 0 for an addition, 1 for a multiplication, 2 for a subtraction.

    Returns:
        The total over the range.
    """
    comptime acc = accumulator(common)
    # Four registers of the common type at a time, so the widening and the
    # additions of one group are not waiting on the group before.
    comptime width = 4 * simd_width_of[common]()
    var src = a.unsafe_ptr[source]()
    var y = SIMD[common, width](k)
    var sums = SIMD[acc, width](0)
    var i = start
    if not nulls:
        while i + width <= stop:
            var x = (
                src.unsafe_offset(i).unsafe_load[width=width]().cast[common]()
            )
            var r: SIMD[common, width]
            comptime if code == 0:
                r = x + y
            elif code == 1:
                r = x * y
            else:
                r = y - x if flip else x - y
            sums += r.cast[acc]()
            i += width
    var total = sums.reduce_add()
    ref valid = a.data.validity
    while i < stop:
        if not nulls or valid.get(i):
            var x = src.unsafe_offset(i).unsafe_load().cast[common]()
            var r: Scalar[common]
            comptime if code == 0:
                r = x + k
            elif code == 1:
                r = x * k
            else:
                r = k - x if flip else x - k
            total += r.cast[acc]()
        i += 1
    return total
