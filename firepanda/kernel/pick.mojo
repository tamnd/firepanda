"""Choosing between two columns a row at a time, on a condition.

This is SQL's `CASE WHEN c THEN a ELSE b END` and polars' `when(c).then(a).
otherwise(b)`. TPC-H asks for it three times and all three are the same shape: a
condition over one column, a value on the true side, and a zero on the false
side, summed. q14 divides the promotion revenue by the total, q12 counts the
urgent orders inside a group by, and q8 picks out the one nation's volume.

The interesting thing about this kernel is how little it does. There is no
branch in the loop, because a SIMD select is one instruction; and there is no
null handling in the loop either, which is worth explaining because it looks
like an omission.

A null condition takes the false side here. That is SQL's rule, an unknown is
not a true, and it is already the rule `filter` follows when it drops a row on a
null in the mask. It is not what polars does, which gives a null, and the
binding layer is where that difference gets paid.

Getting it for free is the part worth writing down. `kernel/__init__.mojo` says a
null value is zero in the values buffer, so a null in a bool column is already a
`False` sitting there in memory. Reading the condition's values buffer and
ignoring its validity entirely therefore gives exactly the rule above, with no
mask, no extra load and no second pass. The invariant does the work.

The same invariant covers the output. A row that takes a null from either side
copies that side's zero along with it, so the output already holds a zero under
every null it has and there is nothing for `repair_range` to repair. This is the
only elementwise kernel in the package that does not call it, and that is why.

What is left is the output's validity, which is the one thing a select cannot
compute in the values registers: it is `a`'s bit where the condition is true and
`b`'s where it is false. When neither side has a null there is nothing to
compute, and that is the common case and the fast path. When one does, the bits
are built a word at a time inside the worker that just wrote those rows, so the
condition bytes they are packed out of are still in that core's cache. That is
the same argument `kernel/mask.mojo` makes for repairing in the worker rather
than in a pass of its own.

The last word of a column whose length is not a multiple of sixty four hangs over
the end, and the bits out there have to be left clear, because `count_ones` walks
whole words and a bit set past the end is a row that does not exist counted as
present. `pick` gets that for free, since a bit past the end takes the false
side's cleared tail; `pick_const` does not, since the false side is a constant
and is always valid. Both mask it anyway.
"""

from std.sys.info import simd_width_of

from firepanda.array.any import AnyArray
from firepanda.array.array import Array
from firepanda.array.strings import StringArray, StringBuilder
from firepanda.bitmap.bitmap import Bitmap
from firepanda.dtype.lists import ALL
from firepanda.exec import parallel_morsels


def _selected_word(cond: Array[DType.bool], w: Int) -> UInt64:
    """Packs sixty four condition bytes into the bits of one word.

    Reading the values buffer and not the validity is the null rule described at
    the top of the module: a null condition is a zero byte and comes back as a
    zero bit, which is the false side.

    Args:
        cond: The condition column.
        w: Which word. Rows past the end of the column come back as zero bits,
            so a caller can ask for the word a short column ends inside.

    Returns:
        The bits, lowest bit first.
    """
    var base = w * 64
    var stop = min(base + 64, len(cond))
    var src = cond.unsafe_ptr()
    var bits = UInt64(0)
    for i in range(base, stop):
        if src.unsafe_offset(i).unsafe_load():
            bits |= UInt64(1) << UInt64(i - base)
    return bits


def _tail_mask(length: Int, w: Int) -> UInt64:
    """Returns the bits of a word that are rows and not padding.

    The last word of a column that is not a multiple of sixty four rows long
    hangs over the end, and a bitmap has to keep those bits clear: `count_ones`
    walks whole words, so a bit set past the end is a row that does not exist
    counted as present. Every word but the last is all ones.

    Args:
        length: The row count.
        w: Which word.

    Returns:
        A mask to and the built word with.
    """
    var covered = min(64, length - w * 64)
    if covered >= 64:
        return ~UInt64(0)
    return (UInt64(1) << UInt64(covered)) - 1


def pick[
    dt: DType
](cond: Array[DType.bool], a: Array[dt], b: Array[dt]) raises -> Array[dt]:
    """Returns a column taking each row from `a` or from `b`.

    Args:
        cond: The condition. A null in it takes the false side, which is SQL's
            answer and `filter`'s.
        a: The true side. Must be as long as the condition.
        b: The false side. Must be as long as the condition.

    Parameters:
        dt: The dtype of both sides.

    Returns:
        A column of the same dtype and length, null at a row wherever the side
        that row came from is null.

    Raises:
        Error: If the three are not all the same length.
    """
    var n = len(cond)
    if len(a) != n or len(b) != n:
        raise Error(
            "pick: condition of "
            + String(n)
            + " rows against sides of "
            + String(len(a))
            + " and "
            + String(len(b))
        )

    var out = Array[dt](overwritten=n)
    comptime width = simd_width_of[dt]()

    # Nothing to build when neither side can contribute a null, which is the
    # case the queries are made of. The values are already right under any null
    # the sides do have, because a null holds a zero and the select copied it.
    var mixed = a.null_count() > 0 or b.null_count() > 0
    var validity = Bitmap(n if mixed else 0)

    def body(start: Int, stop: Int) {mut out, mut validity, imm}:
        var c = cond.unsafe_ptr()
        var x = a.unsafe_ptr()
        var y = b.unsafe_ptr()
        var dst = out.unsafe_ptr()
        var i = start
        while i < stop:
            var hit = c.unsafe_offset(i).unsafe_load[width=width]()
            dst.unsafe_offset(i).unsafe_store(
                hit.select(
                    x.unsafe_offset(i).unsafe_load[width=width](),
                    y.unsafe_offset(i).unsafe_load[width=width](),
                )
            )
            i += width

        if mixed:
            # In the worker that just computed the rows, for the same reason
            # `repair_range` is: the condition bytes this reads are still in
            # this core's cache. A morsel boundary is a multiple of sixty four
            # rows, so no two workers reach for the same word.
            for w in range(start // 64, (stop + 63) // 64):
                var sel = _selected_word(cond, w)
                validity.unsafe_set_word(
                    w,
                    (
                        (sel & a.data.validity.unsafe_word(w))
                        | (~sel & b.data.validity.unsafe_word(w))
                    )
                    & _tail_mask(n, w),
                )

    parallel_morsels(body, n)

    if mixed:
        out.data.validity = validity^
    return out^


def pick_const[
    dt: DType
](cond: Array[DType.bool], a: Array[dt], b: Scalar[dt]) raises -> Array[dt]:
    """Returns a column taking each row from `a` or from a constant.

    This is the shape q8 and q14 are written in, a column on the true side and a
    zero on the false side. It is a separate function rather than a call to
    `pick` against a broadcast column because the broadcast would be a second
    allocation and a second stream of loads for a value that fits in a register.

    Args:
        cond: The condition. A null in it takes the constant.
        a: The true side. Must be as long as the condition.
        b: The false side. Never null.

    Parameters:
        dt: The dtype.

    Returns:
        A column of the same dtype and length, null only where the condition
        holds and `a` is null.

    Raises:
        Error: If the condition and the column are different lengths.
    """
    var n = len(cond)
    if len(a) != n:
        raise Error(
            "pick: condition of "
            + String(n)
            + " rows against a side of "
            + String(len(a))
        )

    var out = Array[dt](overwritten=n)
    comptime width = simd_width_of[dt]()
    var fill = SIMD[dt, width](b)

    var mixed = a.null_count() > 0
    var validity = Bitmap(n if mixed else 0)

    def body(start: Int, stop: Int) {mut out, mut validity, imm}:
        var c = cond.unsafe_ptr()
        var x = a.unsafe_ptr()
        var dst = out.unsafe_ptr()
        var i = start
        while i < stop:
            var hit = c.unsafe_offset(i).unsafe_load[width=width]()
            dst.unsafe_offset(i).unsafe_store(
                hit.select(x.unsafe_offset(i).unsafe_load[width=width](), fill)
            )
            i += width

        if mixed:
            for w in range(start // 64, (stop + 63) // 64):
                # Valid wherever the condition is false, because the constant
                # is, and `a`'s own bit wherever it is true. The tail mask is
                # not optional here the way it is in `pick`: the false side is
                # a constant, so a bit past the end of the column would come
                # out set rather than picking up the other side's cleared tail.
                var sel = _selected_word(cond, w)
                validity.unsafe_set_word(
                    w,
                    (~sel | (sel & a.data.validity.unsafe_word(w)))
                    & _tail_mask(n, w),
                )

    parallel_morsels(body, n)

    if mixed:
        out.data.validity = validity^
    return out^


def pick_constants[
    dt: DType
](cond: Array[DType.bool], a: Scalar[dt], b: Scalar[dt]) raises -> Array[dt]:
    """Returns a column of one of two constants, chosen a row at a time.

    This is q12, which counts the orders whose priority is urgent by summing a
    one against a zero. Nothing here can be null, so there is no validity to
    build and no branch anywhere in it.

    Args:
        cond: The condition. A null in it takes `b`.
        a: The value where the condition holds.
        b: The value where it does not.

    Parameters:
        dt: The dtype.

    Returns:
        A column of the same length, with no nulls.

    Raises:
        Error: Only what the morsel runtime raises.
    """
    var n = len(cond)
    var out = Array[dt](overwritten=n)
    comptime width = simd_width_of[dt]()
    var yes = SIMD[dt, width](a)
    var no = SIMD[dt, width](b)

    def body(start: Int, stop: Int) {mut out, imm}:
        var c = cond.unsafe_ptr()
        var dst = out.unsafe_ptr()
        var i = start
        while i < stop:
            var hit = c.unsafe_offset(i).unsafe_load[width=width]()
            dst.unsafe_offset(i).unsafe_store(hit.select(yes, no))
            i += width

    parallel_morsels(body, n)
    return out^


def text_pick(
    cond: Array[DType.bool], a: StringArray, b: StringArray
) raises -> StringArray:
    """Returns a text column taking each row from `a` or from `b`.

    Built through a builder on one core, unlike the three above. A text output is
    a payload whose length nobody knows until the rows have been chosen, and the
    two pass shape `substr.mojo` uses to get around that, size every morsel's
    share first and then let each fill its own stretch, is the right answer here
    too. It is not done yet because no query in front of this needs it, and the
    place to spend that work first is `filter` and `take`, which run on every
    query rather than on none of them.

    Args:
        cond: The condition. A null in it takes the false side.
        a: The true side. Must be as long as the condition.
        b: The false side. Must be as long as the condition.

    Returns:
        A text column of the same length.

    Raises:
        Error: If the three are not all the same length.
    """
    var n = len(cond)
    if len(a) != n or len(b) != n:
        raise Error(
            "pick: condition of "
            + String(n)
            + " rows against sides of "
            + String(len(a))
            + " and "
            + String(len(b))
        )

    var builder = StringBuilder(capacity=n)
    var src = cond.unsafe_ptr()
    for i in range(n):
        # The values buffer and not the validity, which is the null rule.
        if src.unsafe_offset(i).unsafe_load():
            if a.is_valid(i):
                builder.append(a.unsafe_bytes(i))
            else:
                builder.append_null()
        elif b.is_valid(i):
            builder.append(b.unsafe_bytes(i))
        else:
            builder.append_null()
    return builder^.finish()


def pick_any(
    cond: Array[DType.bool], a: AnyArray, b: AnyArray
) raises -> AnyArray:
    """Returns a column taking each row from `a` or from `b`.

    Args:
        cond: The condition.
        a: The true side.
        b: The false side. Must be the same type as `a`, because promoting here
            would decide silently which side loses precision and a caller that
            meant to mix types can cast first and say so.

    Returns:
        A column of that type.

    Raises:
        Error: If the two sides are different types, if the type has no physical
            layout, or if the lengths disagree.
    """
    if a.is_string() != b.is_string() or (
        not a.is_string() and a.type.physical != b.type.physical
    ):
        raise Error(
            "pick: cannot choose between "
            + String(a.type)
            + " and "
            + String(b.type)
        )
    if a.is_string():
        return AnyArray(text_pick(cond, a.strings(), b.strings()))

    comptime for target in ALL:
        if a.type.physical == target:
            ref x = a.as_typed_view[target]()
            ref y = b.as_typed_view[target]()
            return AnyArray(pick[target](cond, x, y))
    raise Error("pick: unsupported dtype")
