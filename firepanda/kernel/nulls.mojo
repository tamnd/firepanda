"""Asking about nulls, and getting rid of them.

Four operations live here and they split cleanly in two.

`is_null` and `is_not_null` never touch the values buffer. Whether a row is
missing is entirely a fact about the validity bitmap, so both of them are a
bitmap copy and, for `is_not_null`, an invert. That is also why neither needs the
dtype: the erased spelling takes an `AnyArray` and does not raise, which is the
only kernel in this package that manages it. The result itself is never null,
because "is this row missing" has an answer on every row including the missing
ones. pandas agrees, and it is one of the few places where a three-valued logic
argument does not come up.

`coalesce` and the two directional fills do touch the values, and both of them
answer the same question: what should stand in for a missing value. `coalesce`
says another column, and a part of length one broadcasts, which is what makes
filling with a scalar the same operation as filling from a column. The fills say
the nearest present value in a direction, which is only meaningful when the rows
are in an order that means something, so unlike everything else in this package
they are order-dependent by construction.

`limit` on the fills is the longest run of nulls that will be filled, and zero
means no limit. pandas spells the no-limit case `None` and cannot then take an
integer of zero to mean anything; here zero is the natural spelling because
filling at most zero nulls is a no-op nobody asks for.

None of these promote. `coalesce` over an int32 and a float64 raises rather than
picking one, for the same reason `concat` does.
"""

from std.sys.info import simd_width_of

from firepanda.array.any import AnyArray
from firepanda.array.array import Array
from firepanda.array.strings import StringArray, StringBuilder
from firepanda.bitmap.bitmap import Bitmap
from firepanda.dtype.lists import ALL


def is_null[dt: DType](col: Array[dt]) -> Array[DType.bool]:
    """Returns true where a row is missing.

    Args:
        col: The column to look at. Its values are not read.

    Parameters:
        dt: The dtype.

    Returns:
        A bool column with no nulls of its own.
    """
    return _null_mask[wants_null=True](col.data.validity, len(col))


def is_null_any(col: AnyArray) -> Array[DType.bool]:
    """Returns true where a row is missing, for a column whose dtype is a runtime value.

    Args:
        col: The column to look at. Its values are not read, so the dtype never
            comes into it and this cannot fail.

    Returns:
        A bool column with no nulls of its own.
    """
    return _null_mask[wants_null=True](col.data.validity, len(col))


def is_not_null[dt: DType](col: Array[dt]) -> Array[DType.bool]:
    """Returns true where a row is present.

    Args:
        col: The column to look at. Its values are not read.

    Parameters:
        dt: The dtype.

    Returns:
        A bool column with no nulls of its own.
    """
    return _null_mask[wants_null=False](col.data.validity, len(col))


def is_not_null_any(col: AnyArray) -> Array[DType.bool]:
    """Returns true where a row is present, for a column whose dtype is a runtime value.

    Args:
        col: The column to look at.

    Returns:
        A bool column with no nulls of its own.
    """
    return _null_mask[wants_null=False](col.data.validity, len(col))


def all_valid_mask(columns: List[AnyArray], rows: Int) -> Array[DType.bool]:
    """Returns true on the rows where every column is present.

    The intersection is taken on the bitmaps, a word at a time, and expanded to
    a byte per row once at the end. Expanding first and combining the bool
    columns afterwards would do the same work sixty four times over.

    Args:
        columns: The columns to look at. All of them must be `rows` tall. An
            empty list gives an all-true mask, which is the right answer for
            "every one of no columns is present" and is what a frame with no
            columns needs.
        rows: The height.

    Returns:
        A bool column with no nulls of its own.
    """
    var combined = Bitmap(rows, all_valid=True)
    for c in range(len(columns)):
        combined.and_with(columns[c].data.validity)
    return _null_mask[wants_null=False](combined, rows)


def _null_mask[wants_null: Bool](valid: Bitmap, rows: Int) -> Array[DType.bool]:
    """Turns a validity bitmap into a bool column.

    A bool `Array` stores one byte per row where the bitmap stores one bit, so
    this is an expansion rather than a copy and the loop is per row. The word at
    a time shortcut still pays: an all-present or all-null word is a run of
    sixty four identical bytes and the vector unit writes those.

    Args:
        valid: The validity to read.
        rows: The height. May be shorter than the bitmap's rounded-up length.

    Parameters:
        wants_null: True to report the missing rows, false to report the present
            ones.

    Returns:
        The mask, with every row present.
    """
    comptime width = simd_width_of[DType.bool]()
    var out = Array[DType.bool](rows)
    var target = out.unsafe_ptr()

    for w in range(valid.word_count()):
        var word = valid.unsafe_word(w)
        var base = w * 64
        var last = base + 64
        if last > rows:
            last = rows

        if word == UInt64.MAX or word == 0:
            var present = word == UInt64.MAX
            var answer = Bool(present != wants_null)
            var i = base
            while i + width <= last:
                target.unsafe_offset(i).unsafe_store(
                    SIMD[DType.bool, width](fill=answer)
                )
                i += width
            while i < last:
                target.unsafe_offset(i).unsafe_store(answer)
                i += 1
            continue

        for i in range(base, last):
            var present = ((word >> UInt64(i - base)) & 1) == 1
            target.unsafe_offset(i).unsafe_store(Bool(present != wants_null))
    return out^


def coalesce[dt: DType](a: Array[dt], b: Array[dt]) -> Array[dt]:
    """Returns the first column's value where it is present and the second's otherwise.

    Args:
        a: The preferred column.
        b: The fallback. Either the same height as `a` or one row, which
            broadcasts to every row and is how filling with a scalar is spelled.

    Parameters:
        dt: The dtype.

    Returns:
        A column as tall as `a`, null only where both inputs were.
    """
    return _coalesce_core(
        a.unsafe_ptr(),
        a.data.validity,
        len(a),
        b.unsafe_ptr(),
        b.data.validity,
        len(b),
    )


def coalesce_any(a: AnyArray, b: AnyArray) raises -> AnyArray:
    """Coalesces two columns whose dtype is a runtime value.

    Args:
        a: The preferred column.
        b: The fallback, of the same dtype and either the same height or one row.

    Returns:
        A column as tall as `a`.

    Raises:
        If the dtypes differ, if `b` is neither the same height nor one row, or
        if the dtype has no physical layout.
    """
    if a.dtype() != b.dtype():
        raise Error(
            "coalesce: both columns must have the same dtype; got "
            + String(a.dtype())
            + " and "
            + String(b.dtype())
        )
    if len(b) != len(a) and len(b) != 1:
        raise Error(
            "coalesce: the fallback must have one row or as many as the column;"
            " column has "
            + String(len(a))
            + " rows and fallback has "
            + String(len(b))
        )
    if a.is_string() != b.is_string():
        raise Error(
            "coalesce: both columns must have the same dtype; got "
            + String(a.type)
            + " and "
            + String(b.type)
        )
    if a.is_string():
        return AnyArray(_coalesce_strings(a.strings(), b.strings()))

    comptime for candidate in ALL:
        if a.dtype() == candidate:
            return AnyArray(
                _coalesce_core(
                    a.unsafe_ptr[candidate](),
                    a.data.validity,
                    len(a),
                    b.unsafe_ptr[candidate](),
                    b.data.validity,
                    len(b),
                )
            )
    raise Error("coalesce: unsupported dtype " + String(a.dtype()))


def _coalesce_strings(a: StringArray, b: StringArray) raises -> StringArray:
    """Takes each element from the first column, or the second where it is null.

    Args:
        a: The preferred column.
        b: The fallback, either as tall as `a` or one row, which is how filling
            with a scalar is spelled.

    Returns:
        A column as tall as `a`, null only where both were.
    """
    var builder = StringBuilder(capacity=len(a))
    for i in range(len(a)):
        if a.is_valid(i):
            builder.append(a.unsafe_bytes(i))
            continue
        var at = i if len(b) == len(a) else 0
        if b.is_valid(at):
            builder.append(b.unsafe_bytes(at))
        else:
            builder.append_null()
    return builder^.finish()


def _coalesce_core[
    dt: DType, //, first: ImmOrigin, second: ImmOrigin
](
    a: Pointer[Scalar[dt], first],
    a_valid: Bitmap,
    a_len: Int,
    b: Pointer[Scalar[dt], second],
    b_valid: Bitmap,
    b_len: Int,
) -> Array[dt]:
    """The pick loop, over pointers and bitmaps rather than columns.

    The values from `a` go across in blocks first, all of them, and only the rows
    where `a` was missing are revisited. That is the same shape the elementwise
    kernels use: compute over everything, repair the exceptions. On a column with
    no nulls the repair pass is one word comparison per sixty four rows.
    """
    comptime width = simd_width_of[dt]()
    var out = Array[dt](a_len)
    var target = out.unsafe_ptr()

    var i = 0
    while i + width <= a_len:
        target.unsafe_offset(i).unsafe_store(
            a.unsafe_offset(i).unsafe_load[width=width]()
        )
        i += width
    while i < a_len:
        target.unsafe_offset(i).unsafe_store(a.unsafe_offset(i).unsafe_load())
        i += 1

    if a_valid.all_valid():
        return out^

    var broadcast = b_len == 1
    for w in range(a_valid.word_count()):
        var word = a_valid.unsafe_word(w)
        if word == UInt64.MAX:
            continue
        var base = w * 64
        var last = base + 64
        if last > a_len:
            last = a_len
        for r in range(base, last):
            if ((word >> UInt64(r - base)) & 1) == 1:
                continue
            var at = 0 if broadcast else r
            if at < b_len and b_valid.get(at):
                target.unsafe_offset(r).unsafe_store(
                    b.unsafe_offset(at).unsafe_load()
                )
            else:
                out.data.validity.set(r, False)
    return out^


def fill_forward[dt: DType](col: Array[dt], limit: Int = 0) -> Array[dt]:
    """Carries the last present value forward over the nulls after it.

    Args:
        col: The column.
        limit: The longest run of nulls to fill, or zero for no limit.

    Parameters:
        dt: The dtype.

    Returns:
        A column of the same height. A null before the first present value stays
        null, because there is nothing behind it to carry.
    """
    return _fill_core[forward=True](
        col.unsafe_ptr(), col.data.validity, len(col), limit
    )


def fill_backward[dt: DType](col: Array[dt], limit: Int = 0) -> Array[dt]:
    """Carries the next present value backward over the nulls before it.

    Args:
        col: The column.
        limit: The longest run of nulls to fill, or zero for no limit.

    Parameters:
        dt: The dtype.

    Returns:
        A column of the same height. A null after the last present value stays
        null.
    """
    return _fill_core[forward=False](
        col.unsafe_ptr(), col.data.validity, len(col), limit
    )


def fill_forward_any(col: AnyArray, limit: Int = 0) raises -> AnyArray:
    """Fills forward over a column whose dtype is a runtime value.

    Args:
        col: The column.
        limit: The longest run of nulls to fill, or zero for no limit.

    Returns:
        A column of the same height.

    Raises:
        If the dtype has no physical layout.
    """
    if col.is_string():
        return AnyArray(_fill_strings[forward=True](col.strings(), limit))

    comptime for candidate in ALL:
        if col.dtype() == candidate:
            return AnyArray(
                _fill_core[forward=True](
                    col.unsafe_ptr[candidate](),
                    col.data.validity,
                    len(col),
                    limit,
                )
            )
    raise Error("fill_forward: unsupported dtype " + String(col.dtype()))


def fill_backward_any(col: AnyArray, limit: Int = 0) raises -> AnyArray:
    """Fills backward over a column whose dtype is a runtime value.

    Args:
        col: The column.
        limit: The longest run of nulls to fill, or zero for no limit.

    Returns:
        A column of the same height.

    Raises:
        If the dtype has no physical layout.
    """
    if col.is_string():
        return AnyArray(_fill_strings[forward=False](col.strings(), limit))

    comptime for candidate in ALL:
        if col.dtype() == candidate:
            return AnyArray(
                _fill_core[forward=False](
                    col.unsafe_ptr[candidate](),
                    col.data.validity,
                    len(col),
                    limit,
                )
            )
    raise Error("fill_backward: unsupported dtype " + String(col.dtype()))


def _fill_core[
    dt: DType, //, forward: Bool, origin: ImmOrigin
](
    src: Pointer[Scalar[dt], origin], valid: Bitmap, rows: Int, limit: Int
) -> Array[dt]:
    """The carry loop, over pointers and bitmaps rather than columns.

    This one is a scan and cannot be anything else: row `i` depends on row
    `i - 1`, so there is no arithmetic for the vector unit to do. What it can do
    is skip, and the word at a time test does that. An all-present word has
    nothing to fill, so its values are copied in blocks and the carry is taken
    from whichever end the scan is leaving. On a column with few nulls that is
    the entire loop.
    """
    comptime width = simd_width_of[dt]()
    var out = Array[dt](rows)
    var target = out.unsafe_ptr()
    var carry = Scalar[dt](0)
    var have = False
    var run = 0

    var words = valid.word_count()
    for w in range(words):
        var wi = w if forward else words - 1 - w
        var base = wi * 64
        if base >= rows:
            continue
        var last = base + 64
        if last > rows:
            last = rows

        var word = valid.unsafe_word(wi)
        if word == UInt64.MAX:
            var i = base
            while i + width <= last:
                target.unsafe_offset(i).unsafe_store(
                    src.unsafe_offset(i).unsafe_load[width=width]()
                )
                i += width
            while i < last:
                target.unsafe_offset(i).unsafe_store(
                    src.unsafe_offset(i).unsafe_load()
                )
                i += 1
            carry = src.unsafe_offset(
                last - 1 if forward else base
            ).unsafe_load()
            have = True
            run = 0
            continue

        for step in range(base, last):
            var i = step if forward else base + last - 1 - step
            var value = src.unsafe_offset(i).unsafe_load()
            if ((word >> UInt64(i - base)) & 1) == 1:
                target.unsafe_offset(i).unsafe_store(value)
                carry = value
                have = True
                run = 0
                continue

            run += 1
            if have and (limit <= 0 or run <= limit):
                target.unsafe_offset(i).unsafe_store(carry)
            else:
                out.data.validity.set(i, False)
    return out^


def _fill_strings[
    forward: Bool
](col: StringArray, limit: Int) raises -> StringArray:
    """Carries the last present element across a run of nulls.

    The fixed width version can copy the whole column first and then patch the
    gaps, because the element it carries is a value in a register. Here the
    element is a run of bytes somewhere in the payload, so the column is built in
    one pass and the carried element is the position it came from rather than the
    element itself.

    Args:
        col: The column.
        limit: The longest run of nulls to fill, or zero for no limit.

    Parameters:
        forward: Whether to carry from earlier rows or from later ones.

    Returns:
        A column of the same height. A null with no present element on the
        carrying side stays null.
    """
    var n = len(col)
    var source = List[Int](capacity=n)
    for _ in range(n):
        source.append(-1)

    var last = -1
    var run = 0
    for k in range(n):
        var i = k if forward else n - 1 - k
        if col.is_valid(i):
            source[i] = i
            last = i
            run = 0
            continue
        if last < 0:
            continue
        run += 1
        if limit > 0 and run > limit:
            continue
        source[i] = last

    var builder = StringBuilder(capacity=n)
    for i in range(n):
        if source[i] < 0:
            builder.append_null()
        else:
            builder.append(col.unsafe_bytes(source[i]))
    return builder^.finish()
