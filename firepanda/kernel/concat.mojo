"""Stacking columns end to end.

`concat` is the one operation in this package whose cost is entirely memory
traffic. There is no arithmetic in it and no branch that depends on a value, so
the only thing worth designing is how few passes it makes over the bytes and how
much of the validity work it can skip.

It skips a lot. A fresh `Array` is zeroed and marked present everywhere, so a
part with no nulls needs no validity work at all: the bits it wants are already
there. On the columns most people actually concatenate, that is the whole bitmap
loop gone, and what remains is a vectorized copy of the values.

The parts that do have nulls are handled a bit at a time rather than a word at a
time, because the destination offset is the running total of everything before it
and is almost never a multiple of sixty four. Copying words would mean shifting
every word across the boundary, which is worth writing when concat shows up in a
profile and is not worth writing before that.

Dtypes must match exactly. Promoting them here would mean stacking an int32
column onto a float64 one and silently changing values on the way, and the cast
is cheap to write and belongs where a reader can see it.
"""

from std.sys.info import simd_width_of

from firepanda.array.any import AnyArray
from firepanda.array.array import Array
from firepanda.array.strings import StringArray, StringBuilder
from firepanda.bitmap.bitmap import Bitmap
from firepanda.dtype.lists import ALL


def concat_arrays[dt: DType](parts: List[Array[dt]]) -> Array[dt]:
    """Stacks columns end to end, in the order given.

    Args:
        parts: The columns. An empty list gives an empty column, which is the
            answer a caller folding over a list of frames wants.

    Parameters:
        dt: The dtype.

    Returns:
        A column as tall as the parts put together.
    """
    var total = 0
    for p in range(len(parts)):
        total += len(parts[p])

    var out = Array[dt](total)
    var at = 0
    for p in range(len(parts)):
        _copy_into(out, at, parts[p].unsafe_ptr(), parts[p].data.validity)
        at += len(parts[p])
    return out^


def concat_any(parts: List[AnyArray]) raises -> AnyArray:
    """Stacks columns whose dtype is a runtime value.

    Args:
        parts: The columns. All of them must have the same dtype.

    Returns:
        A column as tall as the parts put together.

    Raises:
        If the list is empty, if two parts disagree on dtype, or if the dtype
        has no physical layout.
    """
    if len(parts) == 0:
        raise Error("concat: at least one column is required")

    var dt = parts[0].dtype()
    var total = 0
    for p in range(len(parts)):
        # The `is_string` half is not redundant. A string column's physical
        # dtype is uint8, so a string column and a column of bytes agree on
        # `dtype()` and are not the same column at all.
        if (
            parts[p].dtype() != dt
            or parts[p].is_string() != parts[0].is_string()
        ):
            raise Error(
                "concat: every column must have the same dtype; got "
                + String(parts[0].type)
                + " and "
                + String(parts[p].type)
            )
        total += len(parts[p])

    if parts[0].is_string():
        var builder = StringBuilder(capacity=total)
        for p in range(len(parts)):
            _append_all(builder, parts[p].strings())
        return AnyArray(builder^.finish())

    comptime for candidate in ALL:
        if dt == candidate:
            var out = Array[candidate](total)
            var at = 0
            for p in range(len(parts)):
                _copy_into(
                    out,
                    at,
                    parts[p].unsafe_ptr[candidate](),
                    parts[p].data.validity,
                )
                at += len(parts[p])
            return AnyArray(out^)
    raise Error("concat: unsupported dtype " + String(dt))


def concat_two_any(a: AnyArray, b: AnyArray) raises -> AnyArray:
    """Stacks exactly two columns whose dtype is a runtime value.

    The list spelling would be the same operation, and it is not the one a join
    can use: building a `List[AnyArray]` out of two columns it only borrows means
    deep copying both, which on the key alignment path is the entire cost of the
    operation paid a second time. Two arguments borrow.

    Args:
        a: The first column.
        b: The second, of the same dtype.

    Returns:
        A column of `len(a) + len(b)` rows.

    Raises:
        If the dtypes differ or have no physical layout.
    """
    if a.dtype() != b.dtype():
        raise Error(
            "concat: every column must have the same dtype; got "
            + String(a.dtype())
            + " and "
            + String(b.dtype())
        )
    if a.is_string() != b.is_string():
        raise Error(
            "concat: every column must have the same dtype; got "
            + String(a.type)
            + " and "
            + String(b.type)
        )
    if a.is_string():
        var builder = StringBuilder(capacity=len(a) + len(b))
        _append_all(builder, a.strings())
        _append_all(builder, b.strings())
        return AnyArray(builder^.finish())

    comptime for candidate in ALL:
        if a.dtype() == candidate:
            var out = Array[candidate](len(a) + len(b))
            _copy_into(out, 0, a.unsafe_ptr[candidate](), a.data.validity)
            _copy_into(out, len(a), b.unsafe_ptr[candidate](), b.data.validity)
            return AnyArray(out^)
    raise Error("concat: unsupported dtype " + String(a.dtype()))


def _append_all(mut builder: StringBuilder, col: StringArray) raises:
    """Appends every element of a string column to a builder.

    The fixed width path memcpys a whole part into place at a known offset. This
    one cannot, because the payload offsets in a part's views are relative to
    that part's payload block, and stacking the blocks would leave every view
    after the first pointing into the wrong one. Rewriting the offsets instead of
    the elements would be faster and is worth doing the day a concat of string
    columns is on a hot path; today it is a `read_csv` of several files.

    Args:
        builder: The builder to append to.
        col: The column whose elements are appended, in order.
    """
    for i in range(len(col)):
        if col.is_valid(i):
            builder.append(col.unsafe_bytes(i))
        else:
            builder.append_null()


def _copy_into[
    dt: DType, //, origin: ImmOrigin
](
    mut out: Array[dt],
    at: Int,
    src: Pointer[Scalar[dt], origin],
    valid: Bitmap,
):
    """Writes one part into the output at a row offset.

    The values go across unconditionally, nulls included, because a null holds a
    zero and copying it preserves the invariant the rest of the kernel layer
    rests on. Only the bits need a decision, and only when there are nulls.

    Args:
        out: The output column. Must have room for the part at `at`.
        at: The row the part starts on.
        src: The part's values.
        valid: The part's validity. Its length is the part's height.

    Parameters:
        dt: The dtype.
        origin: The part's origin.
    """
    comptime width = simd_width_of[dt]()
    var n = len(valid)
    var target = out.unsafe_ptr()

    var i = 0
    while i + width <= n:
        target.unsafe_offset(at + i).unsafe_store(
            src.unsafe_offset(i).unsafe_load[width=width]()
        )
        i += width
    while i < n:
        target.unsafe_offset(at + i).unsafe_store(
            src.unsafe_offset(i).unsafe_load()
        )
        i += 1

    # The output starts all present, so a part with no nulls is already right.
    if valid.all_valid():
        return
    for w in range(valid.word_count()):
        var word = valid.unsafe_word(w)
        if word == UInt64.MAX:
            continue
        var base = w * 64
        var last = base + 64
        if last > n:
            last = n
        for r in range(base, last):
            if (word >> UInt64(r - base)) & 1 == 0:
                out.data.validity.set(at + r, False)
