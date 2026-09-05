"""Asking about nulls, and getting rid of them.

Four operations live here and they split cleanly in two.

`is_null` and `is_not_null` report whether a row is missing. The result itself is
never null, because "is this row missing" has an answer on every row including
the missing ones. pandas agrees, and it is one of the few places where a three
valued logic argument does not come up.

What missing means is the interesting part and it is not only the validity
bitmap. On a float column a NaN is missing too, because pandas on the numpy
backend has no separate presence bitmap there and NaN is the only missing it has.
`Series([1.0, nan]).count()` is 1 in pandas, and a library that answered 2 would
be wrong about a question nobody thinks is subtle. So on a float dtype these
functions read the values as well as the bitmap, and a row is present when its
bit is set and its value is not NaN. On every other dtype nothing changed and
nothing is read, since there is no NaN to find.

That is a divergence from Arrow, where a NaN is an ordinary float that happens to
compare false against itself, and it is taken deliberately. The values buffer
still holds the NaN and still writes it back out, so nothing is lost on the round
trip; it is only the answer to "is this missing" that follows pandas. See #170.

The cost is a read of the values buffer on a float column where there used to be
none. It is arranged so that the common shape pays almost nothing: a word of the
bitmap covers sixty four rows, those rows are scanned with a vector `isnan` and a
reduction, and only a word that turns out to hold a NaN is taken apart bit by
bit. A float column with no NaN in it therefore costs one load, one compare and
one reduce per vector, and produces a bitmap identical to the one it started
with. A row whose validity bit is already clear holds a zero rather than
whatever the producer had, which is the rule everywhere in this package, so a
cleared row can be scanned along with the rest and cannot contribute a NaN.

`Array.null_count` is not part of this and still counts cleared bits and nothing
else. That is the line between the two halves of the library: an `Array` is
Arrow and says what is in the buffers, and a `Series` is pandas and says what
pandas would say. `Series.null_count` is on the pandas side of it and counts a
NaN, which is why `Series.drop_nulls` drops one.

`coalesce` and the two directional fills do touch the values, and both of them
answer the same question: what should stand in for a missing value. `coalesce`
says another column, and a part of length one broadcasts, which is what makes
filling with a scalar the same operation as filling from a column. The fills say
the nearest present value in a direction, which is only meaningful when the rows
are in an order that means something, so unlike everything else in this package
they are order-dependent by construction.

The fills read the values on a float column for the same reason the questions
above them do, and it matters more here, because a fill does not only report what
is missing, it copies a value over it. A NaN treated as present is wrong twice:
the row is left standing where pandas would have filled it, and worse, the NaN is
picked up as the carry and runs forward over every null after it, so one NaN
turns a gap that had a perfectly good value behind it into a run of NaN. What
comes back out is dtype dependent in the same way `min` and `first` are: a row
that could not be filled is a null on an integer column and a NaN on a float one,
because that is the only missing pandas has on a float column, so a float column
never comes out of a fill carrying a null. See #170.

The expansion and the pick both run on every core. Neither carries anything from
one row to the next, so the column splits over morsels the way an elementwise
kernel does, and a morsel boundary is a multiple of sixty four rows, so the
validity words fall on one side or the other and no two workers share one. That
is what makes it safe for `coalesce` to clear a bit in the output's validity from
inside a worker. The fills are the exception and stay on one thread: the value
that stands in for a null is the nearest present one before it, which is a
dependency reaching back an unbounded distance, and splitting that needs a
different algorithm rather than a different loop.

`limit` on the fills is the longest run of nulls that will be filled, and zero
means no limit. pandas spells the no-limit case `None` and cannot then take an
integer of zero to mean anything; here zero is the natural spelling because
filling at most zero nulls is a no-op nobody asks for.

None of these promote. `coalesce` over an int32 and a float64 raises rather than
picking one, for the same reason `concat` does.
"""

from std.bit import pop_count
from std.math import isnan, nan
from std.sys.info import simd_width_of

from firepanda.array.any import AnyArray, ColumnRefs
from firepanda.array.array import Array
from firepanda.array.strings import StringArray, StringBuilder
from firepanda.bitmap.bitmap import Bitmap
from firepanda.dtype.lists import ALL, FLOAT
from firepanda.exec import MORSEL_ROWS, parallel_morsels


def present_bitmap[dt: DType](col: Array[dt]) raises -> Bitmap:
    """Returns the rows that hold a value, with a NaN counting as holding none.

    The one place the pandas rule for a float column is written down. Everything
    else here that has to know what missing means asks this.

    Args:
        col: The column to look at.

    Parameters:
        dt: The dtype.

    Returns:
        A bitmap as long as the column's validity, set where a row is present.

    Raises:
        Error: Only what the morsel runtime raises.
    """
    return present_bitmap_of(col.unsafe_ptr(), col.data.validity, len(col))


def present_bitmap_of[
    dt: DType, //, origin: ImmOrigin
](
    source: Pointer[Scalar[dt], origin], valid: Bitmap, rows: Int
) raises -> Bitmap:
    """The same as `present_bitmap`, for a caller holding a pointer.

    The grouped reductions work over a values pointer and a bitmap rather than
    over a column, because one instantiation of a core serves both the typed
    entry point and the erased one, so they cannot call the column form. See
    #170.

    Args:
        source: The values.
        valid: The validity to start from.
        rows: The height.

    Parameters:
        dt: The value dtype.
        origin: Where the values live.

    Returns:
        A new bitmap, set where a row is present.

    Raises:
        Error: Only what the morsel runtime raises.
    """
    comptime if not dt.is_floating_point():
        return Bitmap(copy=valid)
    return _drop_nans(source, valid, rows)


def present_bitmap_any(col: AnyArray) raises -> Bitmap:
    """Returns the present rows, for a column whose dtype is a runtime value.

    A string column has no float spelling and takes the copy, which is why the
    dispatch is over `FLOAT` rather than over `ALL` with a test inside it.

    Args:
        col: The column to look at.

    Returns:
        A bitmap as long as the column's validity, set where a row is present.

    Raises:
        Error: Only what the morsel runtime raises.
    """
    if not col.is_string():
        comptime for candidate in FLOAT:
            if col.dtype() == candidate:
                return _drop_nans(
                    col.unsafe_ptr[candidate](), col.data.validity, len(col)
                )
    return Bitmap(copy=col.data.validity)


def missing_count_any(col: AnyArray) raises -> Int:
    """Counts the rows that hold no value, with a NaN counting as holding none.

    This is `Array.null_count` plus the NaNs, and on anything but a float column
    it is exactly `Array.null_count` and costs nothing. On a float column it is a
    pass over the values, which is a real change from a number the column already
    knew, and it is what makes `Series.count` agree with pandas.

    Args:
        col: The column to look at.

    Returns:
        How many rows pandas would call missing.

    Raises:
        Error: Only what the morsel runtime raises.
    """
    if not col.is_string():
        comptime for candidate in FLOAT:
            if col.dtype() == candidate:
                return len(col) - _present_count(
                    col.unsafe_ptr[candidate](), col.data.validity, len(col)
                )
    return col.null_count()


def is_null[dt: DType](col: Array[dt]) raises -> Array[DType.bool]:
    """Returns true where a row is missing.

    Args:
        col: The column to look at. Its values are read on a float dtype and
            not on any other.

    Parameters:
        dt: The dtype.

    Returns:
        A bool column with no nulls of its own.

    Raises:
        Error: Only what the morsel runtime raises.
    """
    return _null_mask[wants_null=True](present_bitmap(col), len(col))


def is_null_any(col: AnyArray) raises -> Array[DType.bool]:
    """Returns true where a row is missing, for a column whose dtype is a runtime value.

    Args:
        col: The column to look at.

    Returns:
        A bool column with no nulls of its own.

    Raises:
        Error: Only what the morsel runtime raises.
    """
    return _null_mask[wants_null=True](present_bitmap_any(col), len(col))


def is_not_null[dt: DType](col: Array[dt]) raises -> Array[DType.bool]:
    """Returns true where a row is present.

    Args:
        col: The column to look at. Its values are read on a float dtype and
            not on any other.

    Parameters:
        dt: The dtype.

    Returns:
        A bool column with no nulls of its own.

    Raises:
        Error: Only what the morsel runtime raises.
    """
    return _null_mask[wants_null=False](present_bitmap(col), len(col))


def is_not_null_any(col: AnyArray) raises -> Array[DType.bool]:
    """Returns true where a row is present, for a column whose dtype is a runtime value.

    Args:
        col: The column to look at.

    Returns:
        A bool column with no nulls of its own.

    Raises:
        Error: Only what the morsel runtime raises.
    """
    return _null_mask[wants_null=False](present_bitmap_any(col), len(col))


def all_valid_mask[
    o: ImmOrigin
](columns: ColumnRefs[o], rows: Int) raises -> Array[DType.bool]:
    """Returns true on the rows where every column is present.

    The intersection is taken on the bitmaps, a word at a time, and expanded to
    a byte per row once at the end. Expanding first and combining the bool
    columns afterwards would do the same work sixty four times over.

    A float column contributes its NaNs to the intersection, since a row that a
    float column has no value in is a row `dropna` drops, so the bitmap folded in
    is `present_bitmap_any` and not the raw validity.

    Args:
        columns: The columns to look at. All of them must be `rows` tall. An
            empty list gives an all-true mask, which is the right answer for
            "every one of no columns is present" and is what a frame with no
            columns needs.
        rows: The height.

    Returns:
        A bool column with no nulls of its own.

    Raises:
        Error: Only what the morsel runtime raises.
    """
    var combined = Bitmap(rows, all_valid=True)
    for c in range(len(columns)):
        combined.and_with(present_bitmap_any(columns[c][]))
    return _null_mask[wants_null=False](combined, rows)


def _drop_nans[
    dt: DType, //, origin: ImmOrigin
](
    source: Pointer[Scalar[dt], origin], valid: Bitmap, rows: Int
) raises -> Bitmap:
    """Copies a validity bitmap with the bit cleared on every row holding a NaN.

    A word of the bitmap is sixty four rows, and the question asked of those rows
    first is whether any of them is NaN at all, which is a vector `isnan` and a
    reduction and touches no bits. Only a word that says yes is then taken apart
    one row at a time. So the price on a float column with no NaN in it is a scan
    of the values and nothing else, and the price on a column that is all NaN is
    the scan plus the bit loop, which is the shape that has to be slow.

    A row whose bit is already clear is scanned along with the rest rather than
    being skipped. It holds a zero, because every null in this package does, so
    it cannot be a NaN and cannot change an answer, and testing for it would cost
    more than reading it.

    Args:
        source: The values.
        valid: The validity to start from.
        rows: The height.

    Parameters:
        dt: The value dtype, which the caller has already established is a float
            one.
        origin: Where the values live.

    Returns:
        A new bitmap. The one passed in is not touched.

    Raises:
        Error: Only what the morsel runtime raises.
    """
    comptime width = simd_width_of[dt]()
    var out = Bitmap(copy=valid)

    def scan(start: Int, stop: Int) {mut out, imm source, imm}:
        for w in range(start // 64, (stop + 63) // 64):
            var word = out.unsafe_word(w)
            if word == 0:
                continue
            var base = w * 64
            var last = base + 64
            if last > stop:
                last = stop

            var dirty = False
            var i = base
            while i + width <= last:
                var chunk = source.unsafe_offset(i).unsafe_load[width=width]()
                if isnan(chunk).reduce_or():
                    dirty = True
                    break
                i += width
            if not dirty:
                while i < last:
                    if isnan(source.unsafe_offset(i).unsafe_load()):
                        dirty = True
                        break
                    i += 1
            if not dirty:
                continue

            for j in range(base, last):
                if isnan(source.unsafe_offset(j).unsafe_load()):
                    word &= ~(UInt64(1) << UInt64(j - base))
            out.unsafe_set_word(w, word)

    parallel_morsels(scan, rows)
    return out^


def _present_count[
    dt: DType, //, origin: ImmOrigin
](source: Pointer[Scalar[dt], origin], valid: Bitmap, rows: Int) raises -> Int:
    """Counts the rows whose bit is set and whose value is not NaN.

    The same scan as `_drop_nans` without the bitmap it would have to allocate
    and then count, since a count is all the caller wanted. The two are kept
    apart rather than one being written in terms of the other, because a `count`
    on a ten million row column should not allocate a megabyte to answer.

    Args:
        source: The values.
        valid: The validity.
        rows: The height.

    Parameters:
        dt: The value dtype, which the caller has already established is a float
            one.
        origin: Where the values live.

    Returns:
        How many rows hold a value.

    Raises:
        Error: Only what the morsel runtime raises.
    """
    if rows <= MORSEL_ROWS:
        return _present_range(source, valid, 0, rows)

    # One slot per morsel and a serial add at the end, the arrangement every
    # reduction in `agg.mojo` uses. A count of integers is associative in a way a
    # sum of floats is not, so the order this adds them back in does not matter,
    # but it is left in morsel order anyway to match.
    var count = (rows + MORSEL_ROWS - 1) // MORSEL_ROWS
    var partials = Array[DType.int64](count)

    def tally(start: Int, stop: Int) {mut partials, imm source, imm valid, imm}:
        partials.unsafe_ptr().unsafe_offset(start // MORSEL_ROWS).unsafe_write(
            Int64(_present_range(source, valid, start, stop))
        )

    parallel_morsels(tally, rows)

    var found = 0
    for i in range(count):
        found += Int(partials[i])
    return found


def _present_range[
    dt: DType, //, origin: ImmOrigin
](
    source: Pointer[Scalar[dt], origin], valid: Bitmap, start: Int, stop: Int
) -> Int:
    """Counts the present rows of one morsel.

    The range starts and ends on a multiple of sixty four rows except at the end
    of the column, so the word loop is derived from it and no two callers of this
    ever look at the same word.

    Args:
        source: The values.
        valid: The validity.
        start: The first row.
        stop: One past the last row.

    Parameters:
        dt: The value dtype.
        origin: Where the values live.

    Returns:
        How many rows in the range hold a value.
    """
    comptime width = simd_width_of[dt]()
    var found = 0

    for w in range(start // 64, (stop + 63) // 64):
        var word = valid.unsafe_word(w)
        if word == 0:
            continue
        var base = w * 64
        var last = base + 64
        if last > stop:
            last = stop

        var dirty = False
        var i = base
        while i + width <= last:
            var chunk = source.unsafe_offset(i).unsafe_load[width=width]()
            if isnan(chunk).reduce_or():
                dirty = True
                break
            i += width
        if not dirty:
            while i < last:
                if isnan(source.unsafe_offset(i).unsafe_load()):
                    dirty = True
                    break
                i += 1
        if not dirty:
            found += Int(pop_count(word))
            continue

        for j in range(base, last):
            if (word >> UInt64(j - base)) & 1 == 0:
                continue
            if not isnan(source.unsafe_offset(j).unsafe_load()):
                found += 1

    return found


def _null_mask[
    wants_null: Bool
](valid: Bitmap, rows: Int) raises -> Array[DType.bool]:
    """Turns a validity bitmap into a bool column.

    A bool `Array` stores one byte per row where the bitmap stores one bit, so
    this is an expansion rather than a copy and the loop is per row. The word at
    a time shortcut still pays: an all-present or all-null word is a run of
    sixty four identical bytes and the vector unit writes those.

    The row range a worker is handed starts and ends on a word, so the word loop
    is derived from it rather than the other way round. Words past the end of the
    bitmap read as zero, which is to say as null, so a bitmap shorter than the
    column still leaves every row written.

    Args:
        valid: The validity to read.
        rows: The height. May be shorter than the bitmap's rounded-up length.

    Parameters:
        wants_null: True to report the missing rows, false to report the present
            ones.

    Returns:
        The mask, with every row present.

    Raises:
        Error: Only what the morsel runtime raises.
    """
    comptime width = simd_width_of[DType.bool]()
    # Every row is written below, so the zeroing allocation is a wasted pass.
    var out = Array[DType.bool](overwritten=rows)

    def expand(start: Int, stop: Int) {mut out, imm valid, imm}:
        var target = out.unsafe_ptr()
        for w in range(start // 64, (stop + 63) // 64):
            var word = valid.unsafe_word(w)
            var base = w * 64
            var last = base + 64
            if last > stop:
                last = stop

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
                target.unsafe_offset(i).unsafe_store(
                    Bool(present != wants_null)
                )

    parallel_morsels(expand, rows)
    return out^


def coalesce[dt: DType](a: Array[dt], b: Array[dt]) raises -> Array[dt]:
    """Returns the first column's value where it is present and the second's otherwise.

    Args:
        a: The preferred column.
        b: The fallback. Either the same height as `a` or one row, which
            broadcasts to every row and is how filling with a scalar is spelled.

    Parameters:
        dt: The dtype.

    Returns:
        A column as tall as `a`, null only where both inputs were.

    Raises:
        Error: Only what the morsel runtime raises.
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
        if the dtype has no physical layout, or whatever the morsel runtime
        raises.
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
) raises -> Array[dt]:
    """The pick loop, over pointers and bitmaps rather than columns.

    The values from `a` go across in blocks first, all of them, and only the rows
    where `a` was missing are revisited. That is the same shape the elementwise
    kernels use: compute over everything, repair the exceptions. On a column with
    no nulls the repair pass is one word comparison per sixty four rows.

    Both halves happen inside the same worker, over the rows that worker was
    handed, so the rows it revisits are the ones it has just written and they are
    still in its cache. Clearing a bit of the output's validity from a worker is
    safe for the same reason the repair in `mask.mojo` is: the range starts and
    ends on a word.

    Raises:
        Error: Only what the morsel runtime raises.
    """
    comptime width = simd_width_of[dt]()
    # Every row is written by the copy below, so the zeroing allocation is a
    # wasted pass.
    var out = Array[dt](overwritten=a_len)

    var repairing = not a_valid.all_valid()
    var broadcast = b_len == 1

    def pick(start: Int, stop: Int) {mut out, imm a_valid, imm b_valid, imm}:
        var target = out.unsafe_ptr()
        var i = start
        while i + width <= stop:
            target.unsafe_offset(i).unsafe_store(
                a.unsafe_offset(i).unsafe_load[width=width]()
            )
            i += width
        while i < stop:
            target.unsafe_offset(i).unsafe_store(
                a.unsafe_offset(i).unsafe_load()
            )
            i += 1

        if not repairing:
            return

        for w in range(start // 64, (stop + 63) // 64):
            var word = a_valid.unsafe_word(w)
            if word == UInt64.MAX:
                continue
            var base = w * 64
            var last = base + 64
            if last > stop:
                last = stop
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

    parallel_morsels(pick, a_len)
    return out^


def fill_forward[dt: DType](col: Array[dt], limit: Int = 0) -> Array[dt]:
    """Carries the last present value forward over the nulls after it.

    Args:
        col: The column.
        limit: The longest run of nulls to fill, or zero for no limit.

    Parameters:
        dt: The dtype.

    Returns:
        A column of the same height. A row before the first present value is
        still missing afterwards, because there is nothing behind it to carry,
        and it is spelled the way the dtype spells missing: a null on an integer
        column and a NaN on a float one.
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
        A column of the same height. A row after the last present value is still
        missing afterwards, spelled the way the dtype spells missing: a null on
        an integer column and a NaN on a float one.
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

    On a float dtype a NaN is missing as well, and the two things that follow
    from that are both wanted. A NaN is filled over rather than left standing,
    and a NaN is never picked up as the carry, which is the worse of the two: one
    NaN taken as the carry runs forward over every null after it and turns a gap
    that had a perfectly good value behind it into a run of NaN.

    Neither of those costs a pass over the values, and the arrangement that gets
    there is the same one the grouped reductions use. The block copy has the
    values in registers already, so it folds a vector `isnan` into the copy and
    reduces the lanes once at the end of the block rather than once per vector.
    A block that turns out to have held a NaN falls through to the row loop,
    which rewrites it. The row loop has the value in hand too, so it tests there
    rather than in a corrected word built in front of it. Building that word
    instead was written first and thrown away, because it costs a whole extra
    read of the values. The two builds run back to back on a million rows: with
    the corrected word a clean float column filled in 682 microseconds against
    474 for the int64 column it should be level with, and folded into the copy it
    is 489 against 489, which is level to the microsecond.

    What a row that could not be filled comes back as depends on the dtype. On an
    int64 column it is a null, which is the only missing that dtype has. On a
    float column it is a NaN, because that is the only missing pandas has there,
    and a fill is an operation whose output is a statement about what is still
    missing afterwards. That is the same rule the reductions took in #179, where
    `min` and `max` and `first` and `last` keep the column's dtype and answer
    with a NaN on a float column and a null on an integer one. So a float column
    never comes out of a fill with a null in it. See #170.
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
            var nanned = False
            var lanes = SIMD[DType.bool, width](fill=False)
            var i = base
            while i + width <= last:
                var chunk = src.unsafe_offset(i).unsafe_load[width=width]()
                target.unsafe_offset(i).unsafe_store(chunk)
                comptime if dt.is_floating_point():
                    lanes |= isnan(chunk)
                i += width
            comptime if dt.is_floating_point():
                nanned = lanes.reduce_or()
            while i < last:
                var value = src.unsafe_offset(i).unsafe_load()
                target.unsafe_offset(i).unsafe_store(value)
                comptime if dt.is_floating_point():
                    if isnan(value):
                        nanned = True
                i += 1
            # Only a block that held one falls through, and the row loop below
            # writes over what was just copied. On any dtype without a NaN this
            # is always False and the branch is compiled away.
            if not nanned:
                carry = src.unsafe_offset(
                    last - 1 if forward else base
                ).unsafe_load()
                have = True
                run = 0
                continue

        for step in range(base, last):
            var i = step if forward else base + last - 1 - step
            var value = src.unsafe_offset(i).unsafe_load()
            var there = ((word >> UInt64(i - base)) & 1) == 1
            comptime if dt.is_floating_point():
                if isnan(value):
                    there = False
            if there:
                target.unsafe_offset(i).unsafe_store(value)
                carry = value
                have = True
                run = 0
                continue

            run += 1
            if have and (limit <= 0 or run <= limit):
                target.unsafe_offset(i).unsafe_store(carry)
            else:
                comptime if dt.is_floating_point():
                    target.unsafe_offset(i).unsafe_store(nan[dt]())
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
