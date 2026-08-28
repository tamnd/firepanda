"""Sorting, by producing a permutation rather than by moving values.

Everything here returns row indices. `argsort` gives you the order and `take_rows`
applies it, and the two stay separate because a frame sorted on one column has to
reorder every other column by the same permutation. Sorting values in place would
mean either redoing the comparison work once per column or keeping a permutation
anyway, and a permutation is smaller than the values it stands for on every dtype
wider than four bytes.

The algorithm is a least significant digit radix sort on eight bit digits. Radix
sorting a float sounds wrong and is not: `sort_key` maps every numeric dtype onto
an unsigned integer whose ordering is the dtype's ordering, so by the time the
digits are counted there is no such thing as a float. The transform is two
instructions and it is exact, which is more than can be said for any comparison
sort's handling of negative zero.

Two things keep the pass count honest. Every digit's histogram is counted in one
read of the keys rather than one read per digit, and a digit whose values are all
identical is skipped, which on real data is most of the high bytes of a wide
dtype. A column of small positive int64 values sorts in one pass, not eight.

Stability is not optional here. It is what makes a multi-key sort work: sorting by
the last key and then stably by each earlier one leaves the earlier keys dominant,
which is the same trick the digits themselves are sorted with, one level up.
`argsort_into` therefore refines the permutation it is given rather than starting
from the identity, and `argsort_multi` walks the keys backwards.

Measured on the reference machine, a million rows: 21.9 ms for random int64,
7.6 ms when the values fit in ten bits, 10.9 ms for random uint32, and 48.4 ms
for the standard library sorting a plain `List[Int64]` with no nulls and no
permutation to carry. The one number that does not fit the pattern is a column
that is already sorted, which costs 32.7 ms on three passes where a random column
of the same value range costs 8.7 ms on the same three. Sequential input walks the
256 write cursors in strict round robin, so each one is evicted before it comes
round again. Staging the scatter through a small per bucket buffer is the known
fix and it is not in here yet.

A string is sorted by the same radix passes, which is the whole reason the eight
byte prefix exists. `StringArray.sort_prefix` packs the first eight bytes of an
element into a `UInt64` most significant byte first, so comparing two of those as
integers gives the same answer as comparing the elements, right up to the point
where the first eight bytes agree. The radix sort then does what it does for an
int64 column and what comes out is correct except within runs of rows whose first
eight bytes are identical.

Those runs are what the comparison sort in here is for, and it only ever runs on
them. On the columns a dataframe actually sorts, a city, a currency code, a status
label, a surname, almost every run is a single row and the comparison sort is
never entered at all. When it is, it is a stable merge sort with an insertion sort
underneath, because a run is usually small and occasionally is the whole column,
which is what a column of URLs sharing a scheme and host looks like.

Eight bytes rather than the four already sitting in the view, because four leaves
far more ties than eight: "Amsterdam" and "Amersfoort" agree on four and not on
eight. The cost is that a long element's key needs its payload read once while the
keys are built, and that read is in row order rather than scattered.
"""

from std.sys.info import size_of

from firepanda.array.any import AnyArray
from firepanda.array.array import Array
from firepanda.array.strings import StringArray
from firepanda.bitmap.bitmap import Bitmap
from firepanda.buffer.buffer import Buffer
from firepanda.dtype.lists import ORDERED

comptime SMALL_RUN = 16
"""Rows below which a tied run is insertion sorted rather than merged.

A merge sort of eight rows spends more of its time on bookkeeping than on
comparisons, and the runs this sees are mostly two or three rows long.
"""

comptime KEY_BYTES = 8
"""Bytes of an element that fit in its radix key, which is `StringArray.sort_prefix`.

An element no longer than this is entirely inside its own key, which is what lets
a run of them be recognised as already settled without a single comparison.
"""

comptime RADIX_BITS = 8
"""Bits per digit. Eight means a 256 entry histogram, 2 KB of counters, which
stays in L1 next to everything else a pass is touching."""

comptime RADIX_SIZE = 1 << RADIX_BITS
"""Histogram entries per digit."""

comptime RADIX_MASK = UInt64(RADIX_SIZE - 1)
"""Mask for one digit."""


def sort_key[
    dt: DType, width: Int = 1
](value: SIMD[dt, width]) -> SIMD[DType.uint64, width]:
    """Maps values onto unsigned integers that sort the same way.

    Unsigned values are already in the right order and only widen. Signed values
    have their sign bit flipped, which moves the negatives below the positives
    where two's complement had them above. Floats flip the sign bit when it is
    clear and every bit when it is set, which is the standard IEEE trick and works
    because the exponent sits above the mantissa, so the bit pattern of a positive
    float already increases with its value.

    Two things it does not do. It does not canonicalize NaN, because unlike a hash
    key a sort key has no equality to preserve and every NaN ends up at the top
    together regardless. It does not fold negative zero into positive zero, so
    `-0.0` sorts below `0.0`, which is what numpy does and is invisible to anybody
    who is not looking for it.

    Args:
        value: The values.

    Parameters:
        dt: The dtype.
        width: The register width.

    Returns:
        One unsigned key per lane, ordered the way the input was.
    """
    comptime if dt.is_floating_point():
        # Widening to float64 first is order preserving, costs nothing on a
        # column that was float64 already, and means one key width rather than
        # one per float dtype.
        var raw = value.cast[DType.float64]().to_bits[DType.uint64]()
        # An arithmetic shift of the sign bit gives all ones for a negative and
        # all zeros for a positive. Or in the top bit and that is the mask for
        # both cases at once: flip everything, or flip only the sign.
        var negative = (raw.cast[DType.int64]() >> 63).cast[DType.uint64]()
        var top = SIMD[DType.uint64, width](UInt64(1) << 63)
        return raw ^ (negative | top)
    comptime if dt.is_signed():
        var raw = value.cast[DType.int64]().cast[DType.uint64]()
        return raw ^ SIMD[DType.uint64, width](UInt64(1) << 63)
    return value.cast[DType.uint64]()


def key_width[dt: DType]() -> Int:
    """Returns how many bytes of a sort key can differ.

    Everything is widened to 64 bits by `sort_key`, but a narrow unsigned integer
    leaves the high bytes constant and there is no reason to count histograms for
    them. Signed dtypes get no such discount, because the sign flip lands in the
    top bit of the widened value rather than the top bit of the original, so an
    int8 column spans all eight bytes. Floats span all eight too, since a float32
    is widened to a float64 bit pattern.

    Parameters:
        dt: The dtype.

    Returns:
        Bytes that need a radix pass.
    """
    comptime if dt.is_floating_point():
        return 8
    comptime if dt.is_signed():
        return 8
    return size_of[dt]()


def argsort[
    dt: DType
](col: Array[dt], descending: Bool = False, nulls_first: Bool = False) -> Array[
    DType.uint32
]:
    """Returns the row order that sorts a column.

    Stable: rows with equal keys keep their original relative order, and that
    holds in the descending direction too.

    Args:
        col: The column to sort.
        descending: Largest first.
        nulls_first: Put the nulls at the front rather than the back.

    Parameters:
        dt: The dtype.

    Returns:
        A permutation of `[0, len(col))`.
    """
    var n = len(col)
    var order = Array[DType.uint32](n)
    var out = order.unsafe_ptr()
    for i in range(n):
        out.unsafe_offset(i).unsafe_write(UInt32(i))

    argsort_into(col, order, descending, nulls_first)
    return order^


def argsort_into[
    dt: DType
](
    col: Array[dt],
    mut order: Array[DType.uint32],
    descending: Bool = False,
    nulls_first: Bool = False,
):
    """Refines an existing permutation by sorting it on a column.

    Rows that compare equal on this column keep the order they arrived in. That is
    what makes `argsort_multi` work and it is the only reason this is separate
    from `argsort`.

    Args:
        col: The column to sort on.
        order: The permutation to refine, in place. Every entry must be a row of
            `col`.
        descending: Largest first.
        nulls_first: Put the nulls at the front rather than the back.

    Parameters:
        dt: The dtype.
    """
    _argsort_core(
        col.unsafe_ptr(),
        col.data.validity,
        col.null_count() > 0,
        order,
        descending,
        nulls_first,
    )


def argsort_multi(
    cols: List[AnyArray],
    descending: List[Bool],
    nulls_first: List[Bool],
) raises -> Array[DType.uint32]:
    """Returns the row order that sorts a set of columns, first key dominant.

    Least significant digit again, one level up: sort by the last key, then
    stably by the second to last, and so on. Every pass after the first only
    reorders rows that its key distinguishes, because the rows it considers equal
    keep the order the later keys gave them. That is why `argsort_into` refines a
    permutation instead of building one.

    The alternative is a comparison sort with a tuple comparator, which is one
    pass instead of `k` but pays a branch per key per comparison and cannot radix
    anything. With two or three keys, which is what a sort in practice has, the
    passes win.

    Args:
        cols: The key columns, most significant first. All the same length.
        descending: One flag per column.
        nulls_first: One flag per column.

    Returns:
        A permutation of `[0, len(cols[0]))`.

    Raises:
        If the lists are different lengths, if there are no columns, if the
        columns are different lengths, or if a column's dtype is not sortable.
    """
    if len(cols) == 0:
        raise Error("argsort_multi needs at least one key column")
    if len(descending) != len(cols) or len(nulls_first) != len(cols):
        raise Error(
            "argsort_multi needs one descending and one nulls_first flag per"
            " column; got "
            + String(len(cols))
            + " columns, "
            + String(len(descending))
            + " descending and "
            + String(len(nulls_first))
            + " nulls_first"
        )

    var n = len(cols[0])
    for k in range(1, len(cols)):
        if len(cols[k]) != n:
            raise Error(
                "argsort_multi key columns must be the same length; column 0"
                " has "
                + String(n)
                + " rows and column "
                + String(k)
                + " has "
                + String(len(cols[k]))
            )

    var order = identity_permutation(n)
    for at in range(len(cols) - 1, -1, -1):
        argsort_any_into(cols[at], order, descending[at], nulls_first[at])
    return order^


def argsort_any(
    col: AnyArray, descending: Bool = False, nulls_first: Bool = False
) raises -> Array[DType.uint32]:
    """Returns the row order that sorts one type-erased column.

    `argsort_multi` with a single key does the same thing, but its argument is a
    `List[AnyArray]` and building one costs a deep copy of the column. A `Series`
    sorts itself often enough for that to be worth its own entry point.

    Args:
        col: The column to sort on.
        descending: Largest first.
        nulls_first: Put the nulls at the front rather than the back.

    Returns:
        A permutation of `[0, len(col))`.

    Raises:
        If the column's dtype is not one firepanda can sort.
    """
    var order = identity_permutation(len(col))
    argsort_any_into(col, order, descending, nulls_first)
    return order^


def identity_permutation(n: Int) -> Array[DType.uint32]:
    """Returns `[0, 1, ..., n - 1]` as a permutation column.

    Every erased sort starts here, because `argsort_into` refines a permutation
    rather than building one and the unsorted frame's own row order is what it
    starts from.

    Args:
        n: The number of rows.

    Returns:
        A column of `n` uint32 values counting up from zero.
    """
    var order = Array[DType.uint32](n)
    var out = order.unsafe_ptr()
    for i in range(n):
        out.unsafe_offset(i).unsafe_write(UInt32(i))
    return order^


def argsort_any_into(
    col: AnyArray,
    mut order: Array[DType.uint32],
    descending: Bool,
    nulls_first: Bool,
) raises:
    """Refines a permutation by sorting it on a type-erased column.

    This is `dispatch` written out by hand. `dispatch` takes an operation that
    returns a value and this one has to write through a mutable reference it was
    handed, which a parametric callable cannot carry.

    Args:
        col: The column to sort on.
        order: The permutation to refine, in place.
        descending: Largest first.
        nulls_first: Put the nulls at the front rather than the back.

    Raises:
        If the column's dtype is not one firepanda can sort.
    """
    # Before the dispatch, because uint8 is in ORDERED and a string column would
    # match it and sort on the first byte of each view, which is a plausible
    # looking wrong answer rather than an error.
    if col.is_string():
        _argsort_strings(
            col.strings(),
            col.data.validity,
            col.null_count() > 0,
            order,
            descending,
            nulls_first,
        )
        return
    comptime for candidate in ORDERED:
        if col.dtype() == candidate:
            _argsort_core(
                col.unsafe_ptr[candidate](),
                col.data.validity,
                col.null_count() > 0,
                order,
                descending,
                nulls_first,
            )
            return
    raise Error("unsupported dtype " + String(col.dtype()) + " for sort")


def _argsort_core[
    dt: DType, //, origin: ImmOrigin
](
    values: Pointer[Scalar[dt], origin],
    validity: Bitmap,
    has_null: Bool,
    mut order: Array[DType.uint32],
    descending: Bool,
    nulls_first: Bool,
):
    """Sorts a permutation by the values it points at.

    Takes a pointer and a bitmap rather than a column so that the typed and the
    type-erased entry points share it without either of them copying anything.

    Args:
        values: The column's values.
        validity: The column's validity bitmap.
        has_null: Whether the validity bitmap has anything to say.
        order: The permutation to refine, in place.
        descending: Largest first.
        nulls_first: Put the nulls at the front rather than the back.

    Parameters:
        dt: The dtype.
        origin: The origin of the values, inferred.
    """
    var n = len(order)
    if n < 2:
        return

    var keys = Buffer(n * 8)
    var alt_keys = Buffer(n * 8)
    var rows = Buffer(n * 4)
    var alt_rows = Buffer(n * 4)
    var digits = key_width[dt]()

    # The nulls come out first, stably, and go back in at whichever end was asked
    # for. Partitioning them here rather than inside the radix passes keeps the
    # digit loop free of a branch it would otherwise take on every row of every
    # pass, to serve a column that usually has no nulls at all.
    var nulls = List[UInt32]()
    var live = 0

    var order_ptr = order.unsafe_ptr()
    var key = keys.bitcast[DType.uint64]()
    var row = rows.bitcast[DType.uint32]()

    if has_null:
        for i in range(n):
            var at = order_ptr.unsafe_offset(i).unsafe_load()
            if validity.get(Int(at)):
                row.unsafe_offset(live).unsafe_write(at)
                key.unsafe_offset(live).unsafe_write(
                    sort_key(values.unsafe_offset(Int(at)).unsafe_load())
                )
                live += 1
            else:
                nulls.append(at)
    else:
        live = n
        for i in range(n):
            var at = order_ptr.unsafe_offset(i).unsafe_load()
            row.unsafe_offset(i).unsafe_write(at)
            key.unsafe_offset(i).unsafe_write(
                sort_key(values.unsafe_offset(Int(at)).unsafe_load())
            )

    if descending:
        # Complementing the key reverses the order without reversing the array,
        # which is the difference between a stable descending sort and one that
        # scrambles the rows it considers equal.
        for i in range(live):
            key.unsafe_offset(i).unsafe_write(
                ~key.unsafe_offset(i).unsafe_load()
            )

    var flipped = _radix_sort(keys, alt_keys, rows, alt_rows, live, digits)

    var lead = len(nulls) if nulls_first else 0
    var target = order.unsafe_ptr()
    if flipped:
        var sorted_rows = alt_rows.bitcast[DType.uint32]()
        for i in range(live):
            target.unsafe_offset(lead + i).unsafe_write(
                sorted_rows.unsafe_offset(i).unsafe_load()
            )
    else:
        var sorted_rows = rows.bitcast[DType.uint32]()
        for i in range(live):
            target.unsafe_offset(lead + i).unsafe_write(
                sorted_rows.unsafe_offset(i).unsafe_load()
            )

    var tail = 0 if nulls_first else live
    for i in range(len(nulls)):
        target.unsafe_offset(tail + i).unsafe_write(nulls[i])


def _argsort_strings(
    col: StringArray,
    validity: Bitmap,
    has_null: Bool,
    mut order: Array[DType.uint32],
    descending: Bool,
    nulls_first: Bool,
):
    """Sorts a permutation by the text it points at.

    The same shape as `_argsort_core`: partition the nulls out, build a key per
    live row, radix sort the keys with the rows riding along, put the nulls back
    at whichever end was asked for. The two differences are that the key is the
    first eight bytes of the element rather than the whole value, and that because
    the key is not the whole value the radix pass leaves runs of rows it could not
    tell apart, which `_resolve_ties` then finishes.

    The eight digits are all counted even though a column of ASCII text uses about
    ninety values in each byte, because unlike a numeric column there is no high
    byte here that is constant across the data. The all-one-bucket skip in
    `_radix_sort` still catches the case where it is, which is a column of fixed
    prefixes.

    Args:
        col: The text.
        validity: The column's validity bitmap.
        has_null: Whether the validity bitmap has anything to say.
        order: The permutation to refine, in place.
        descending: Largest first.
        nulls_first: Put the nulls at the front rather than the back.
    """
    var n = len(order)
    if n < 2:
        return

    var keys = Buffer(n * 8)
    var alt_keys = Buffer(n * 8)
    var rows = Buffer(n * 4)
    var alt_rows = Buffer(n * 4)

    var nulls = List[UInt32]()
    var live = 0

    var order_ptr = order.unsafe_ptr()
    var key = keys.bitcast[DType.uint64]()
    var row = rows.bitcast[DType.uint32]()

    if has_null:
        for i in range(n):
            var at = order_ptr.unsafe_offset(i).unsafe_load()
            if validity.get(Int(at)):
                row.unsafe_offset(live).unsafe_write(at)
                key.unsafe_offset(live).unsafe_write(col.sort_prefix(Int(at)))
                live += 1
            else:
                nulls.append(at)
    else:
        live = n
        for i in range(n):
            var at = order_ptr.unsafe_offset(i).unsafe_load()
            row.unsafe_offset(i).unsafe_write(at)
            key.unsafe_offset(i).unsafe_write(col.sort_prefix(Int(at)))

    if descending:
        for i in range(live):
            key.unsafe_offset(i).unsafe_write(
                ~key.unsafe_offset(i).unsafe_load()
            )

    var flipped = _radix_sort(keys, alt_keys, rows, alt_rows, live, 8)

    var lead = len(nulls) if nulls_first else 0
    var target = order.unsafe_ptr()
    var sorted_keys = alt_keys.bitcast[
        DType.uint64
    ]() if flipped else keys.bitcast[DType.uint64]()
    var sorted_rows = alt_rows.bitcast[
        DType.uint32
    ]() if flipped else rows.bitcast[DType.uint32]()
    for i in range(live):
        target.unsafe_offset(lead + i).unsafe_write(
            sorted_rows.unsafe_offset(i).unsafe_load()
        )

    _resolve_ties(col, sorted_keys, target, lead, live, descending)

    var tail = 0 if nulls_first else live
    for i in range(len(nulls)):
        target.unsafe_offset(tail + i).unsafe_write(nulls[i])


def _resolve_ties[
    origin: MutOrigin, key_origin: MutOrigin
](
    col: StringArray,
    keys: Pointer[UInt64, key_origin],
    target: Pointer[UInt32, origin],
    lead: Int,
    live: Int,
    descending: Bool,
):
    """Orders the rows the eight byte key could not separate.

    A run here is a maximal stretch of rows whose keys are equal, which after the
    radix passes is a contiguous stretch. Every run of one row is already right,
    and skipping those is the whole reason this is affordable: on a column of
    cities or currency codes the loop below finds nothing to do and costs one pass
    over the keys.

    Args:
        col: The text.
        keys: The sorted keys, `live` of them starting at zero.
        target: The permutation, whose live rows start at `lead`.
        lead: Where the live rows start.
        live: How many live rows there are.
        descending: Whether the comparison is reversed. The keys were complemented
            before the radix passes, which is what put the runs in descending
            order, but a complemented key says nothing about the bytes past the
            eighth, so the tie break has to know.

    Parameters:
        origin: The permutation's origin, inferred.
        key_origin: The keys' origin, inferred.
    """
    var scratch = List[UInt32]()
    var start = 0
    while start < live:
        var stop = start + 1
        var here = keys.unsafe_offset(start).unsafe_load()
        while stop < live and keys.unsafe_offset(stop).unsafe_load() == here:
            stop += 1

        var count = stop - start
        if count > 1:
            var run = target.unsafe_offset(lead + start)
            if not _run_is_settled(col, run, count):
                if count <= SMALL_RUN:
                    _insertion_sort_run(col, run, count, descending)
                else:
                    _merge_sort_run(col, run, count, descending, scratch)
        start = stop


def _run_is_settled[
    origin: MutOrigin
](col: StringArray, run: Pointer[UInt32, origin], count: Int) -> Bool:
    """Reports whether a tied run is already in its final order.

    The rows in a run agree on the packed first eight bytes. If every one of them
    is also at most eight bytes long and they are all the same length, then that
    packed key was the whole element and the rows are byte identical, so there is
    nothing left to order and the comparison sort would spend its whole time
    proving it.

    That is not a corner case, it is the shape of the columns most often sorted.
    A status, a currency, a country, a category or a ticker is a handful of short
    values repeated across every row, which puts almost the entire column into a
    few enormous runs. Without this check a sort of a hundred distinct two byte
    labels costs 403 ns a row on the reference machine against 27 for an int64
    column, and nearly all of that is a merge sort discovering that everything it
    compares is equal.

    The check is one length load per row and it stops at the first row that fails,
    so a column of long elements pays for one load in total.

    Args:
        col: The text.
        run: The rows in the run.
        count: How many.

    Returns:
        True when every row in the run holds the same bytes.
    """
    var first = col.byte_length(Int(run.unsafe_offset(0).unsafe_load()))
    if first > KEY_BYTES:
        return False
    for i in range(1, count):
        if col.byte_length(Int(run.unsafe_offset(i).unsafe_load())) != first:
            return False
    return True


def _compare_rows(
    col: StringArray, a: UInt32, b: UInt32, descending: Bool
) -> Int:
    """Orders two rows of a text column, honouring the direction.

    Args:
        col: The text.
        a: The left row.
        b: The right row.
        descending: Reverse the answer.

    Returns:
        Negative if `a` sorts first, zero if the two are identical, positive if
        `b` sorts first.
    """
    var order = col.compare_elements(Int(a), Int(b))
    return -order if descending else order


def _insertion_sort_run[
    origin: MutOrigin
](
    col: StringArray,
    run: Pointer[UInt32, origin],
    count: Int,
    descending: Bool,
):
    """Stably sorts a short run of rows by their text.

    Stable because the inner loop stops on the first element that does not sort
    strictly after the one being placed, so equal elements are never crossed. That
    matters as much here as it does in the radix passes: a multi-key sort is built
    out of stable single-key sorts and nothing else.

    Args:
        col: The text.
        run: The rows to sort, in place.
        count: How many.
        descending: Reverse the comparison.

    Parameters:
        origin: The run's origin, inferred.
    """
    for i in range(1, count):
        var value = run.unsafe_offset(i).unsafe_load()
        var j = i - 1
        while (
            j >= 0
            and _compare_rows(
                col, run.unsafe_offset(j).unsafe_load(), value, descending
            )
            > 0
        ):
            run.unsafe_offset(j + 1).unsafe_write(
                run.unsafe_offset(j).unsafe_load()
            )
            j -= 1
        run.unsafe_offset(j + 1).unsafe_write(value)


def _merge_sort_run[
    origin: MutOrigin
](
    col: StringArray,
    run: Pointer[UInt32, origin],
    count: Int,
    descending: Bool,
    mut scratch: List[UInt32],
):
    """Stably sorts a long run of rows by their text.

    Bottom up rather than recursive, with runs of `SMALL_RUN` insertion sorted
    first so the merging starts from something already ordered. This is the path a
    column of long shared prefixes takes, a URL column being the obvious one, and
    on such a column it is the whole sort rather than a correction to it, so the
    n log n matters.

    The scratch list is passed in and reused across runs rather than allocated per
    run, because a column that ties once tends to tie many times.

    Args:
        col: The text.
        run: The rows to sort, in place.
        count: How many.
        descending: Reverse the comparison.
        scratch: Working space. Resized as needed and left holding whatever the
            last merge put in it.

    Parameters:
        origin: The run's origin, inferred.
    """
    while len(scratch) < count:
        scratch.append(UInt32(0))

    var width = SMALL_RUN
    var at = 0
    while at < count:
        var here = count - at
        if here > width:
            here = width
        _insertion_sort_run(col, run.unsafe_offset(at), here, descending)
        at += here

    while width < count:
        var left = 0
        while left < count:
            var middle = left + width
            if middle >= count:
                break
            var right = middle + width
            if right > count:
                right = count

            # The two halves are each sorted and the left one comes first in the
            # original order, so taking from the left whenever the comparison is
            # not strictly greater is what keeps this stable.
            var a = left
            var b = middle
            var out = left
            while a < middle and b < right:
                var lhs = run.unsafe_offset(a).unsafe_load()
                var rhs = run.unsafe_offset(b).unsafe_load()
                if _compare_rows(col, lhs, rhs, descending) <= 0:
                    scratch[out] = lhs
                    a += 1
                else:
                    scratch[out] = rhs
                    b += 1
                out += 1
            while a < middle:
                scratch[out] = run.unsafe_offset(a).unsafe_load()
                a += 1
                out += 1
            while b < right:
                scratch[out] = run.unsafe_offset(b).unsafe_load()
                b += 1
                out += 1

            for k in range(left, right):
                run.unsafe_offset(k).unsafe_write(scratch[k])
            left = right
        width *= 2


def _radix_sort(
    mut keys: Buffer,
    mut alt_keys: Buffer,
    mut rows: Buffer,
    mut alt_rows: Buffer,
    n: Int,
    digits: Int,
) -> Bool:
    """Sorts keys and their rows, least significant digit first.

    Args:
        keys: The sort keys, `n` of them at the front.
        alt_keys: Scratch, the same size.
        rows: The row indices, `n` of them at the front.
        alt_rows: Scratch, the same size.
        n: How many rows.
        digits: How many bytes of the key can differ.

    Returns:
        True when the answer ended up in the scratch buffers, which is what an
        odd number of passes leaves behind. Copying it back would be a whole pass
        over the data to save the caller a branch it has to take anyway.
    """
    if n < 2:
        return False

    # One read of the keys fills every histogram. Counting a digit per pass reads
    # the keys as many times as there are passes for no reason, and the counters
    # are 2 KB per digit, so all eight fit in L1 together.
    var counts = Buffer(digits * RADIX_SIZE * 8)
    var count = counts.bitcast[DType.uint64]()
    var key = keys.bitcast[DType.uint64]()
    for i in range(n):
        var k = key.unsafe_offset(i).unsafe_load()
        for d in range(digits):
            var at = d * RADIX_SIZE + Int(
                (k >> UInt64(d * RADIX_BITS)) & RADIX_MASK
            )
            count.unsafe_offset(at).unsafe_write(
                count.unsafe_offset(at).unsafe_load() + 1
            )

    var flipped = False
    for d in range(digits):
        # A digit where one bucket holds every row is a digit that would copy the
        # array onto itself. That is most of the high bytes of a wide dtype
        # holding ordinary data, and skipping them is why a column of small
        # positive int64 values costs one pass rather than eight. The histogram
        # is over a multiset the passes only reorder, so it stays correct no
        # matter which buffer the data is currently in.
        var head = (
            alt_keys.bitcast[DType.uint64]()
            .unsafe_offset(0)
            .unsafe_load() if flipped else keys.bitcast[DType.uint64]()
            .unsafe_offset(0)
            .unsafe_load()
        )
        var first = Int((head >> UInt64(d * RADIX_BITS)) & RADIX_MASK)
        if Int(count.unsafe_offset(d * RADIX_SIZE + first).unsafe_load()) == n:
            continue

        if flipped:
            _radix_pass(alt_keys, alt_rows, keys, rows, counts, d, n)
        else:
            _radix_pass(keys, rows, alt_keys, alt_rows, counts, d, n)
        flipped = not flipped

    return flipped


def _radix_pass(
    mut src_keys: Buffer,
    mut src_rows: Buffer,
    mut dst_keys: Buffer,
    mut dst_rows: Buffer,
    mut counts: Buffer,
    digit: Int,
    n: Int,
):
    """Scatters one digit's worth of keys and rows into the destination.

    Args:
        src_keys: Keys to read.
        src_rows: Rows to read.
        dst_keys: Keys to write.
        dst_rows: Rows to write.
        counts: The histograms. This digit's entries become write cursors in
            place, which is safe because each digit is passed over once.
        digit: Which byte of the key.
        n: How many rows.
    """
    var count = counts.bitcast[DType.uint64]()
    var base = digit * RADIX_SIZE
    var shift = UInt64(digit * RADIX_BITS)

    # Counts become offsets: the running total is where each bucket starts.
    var total = 0
    for b in range(RADIX_SIZE):
        var at = base + b
        var here = Int(count.unsafe_offset(at).unsafe_load())
        count.unsafe_offset(at).unsafe_write(UInt64(total))
        total += here

    var from_key = src_keys.bitcast[DType.uint64]()
    var from_row = src_rows.bitcast[DType.uint32]()
    var to_key = dst_keys.bitcast[DType.uint64]()
    var to_row = dst_rows.bitcast[DType.uint32]()

    for i in range(n):
        var k = from_key.unsafe_offset(i).unsafe_load()
        var at = base + Int((k >> shift) & RADIX_MASK)
        var slot = Int(count.unsafe_offset(at).unsafe_load())
        count.unsafe_offset(at).unsafe_write(UInt64(slot + 1))
        to_key.unsafe_offset(slot).unsafe_write(k)
        to_row.unsafe_offset(slot).unsafe_write(
            from_row.unsafe_offset(i).unsafe_load()
        )


def sort_rows[
    dt: DType
](col: Array[dt], descending: Bool = False, nulls_first: Bool = False) -> Array[
    dt
]:
    """Returns a column's values in sorted order.

    A convenience over `argsort` followed by a gather. A frame wants the
    permutation, because it has other columns to reorder by the same one, so this
    is for the single column case and for tests.

    Args:
        col: The column to sort.
        descending: Largest first.
        nulls_first: Put the nulls at the front rather than the back.

    Parameters:
        dt: The dtype.

    Returns:
        A column of the same length, sorted.
    """
    var order = argsort(col, descending, nulls_first)
    var n = len(col)
    var out = Array[dt](n)
    var source = col.unsafe_ptr()
    var target = out.unsafe_ptr()
    var rows = order.unsafe_ptr()
    var validity = Bitmap(n, all_valid=False)

    # Same shape as `take_rows`: the output positions are consecutive, so the
    # validity bits go into a register and get stored once every sixty four rows.
    var word = UInt64(0)
    for i in range(n):
        var at = Int(rows.unsafe_offset(i).unsafe_load())
        target.unsafe_offset(i).unsafe_write(
            source.unsafe_offset(at).unsafe_load()
        )
        if col.data.validity.get(at):
            word |= UInt64(1) << UInt64(i & 63)
        if i & 63 == 63:
            validity.unsafe_set_word(i >> 6, word)
            word = 0

    if n & 63 != 0:
        validity.unsafe_set_word(n >> 6, word)

    out.data.validity = validity^
    return out^


def is_sorted[
    dt: DType
](col: Array[dt], descending: Bool = False, nulls_first: Bool = False) -> Bool:
    """Reports whether a column is already in sorted order.

    Worth having on its own because an already sorted column is common enough
    that the check pays for itself, and because it is the cheapest possible check
    of what `argsort` produced.

    Args:
        col: The column.
        descending: Check for largest first.
        nulls_first: Expect the nulls at the front rather than the back.

    Parameters:
        dt: The dtype.

    Returns:
        Whether the column is sorted the way the arguments describe.
    """
    var n = len(col)
    if n < 2:
        return True

    var values = col.unsafe_ptr()
    var has_null = col.null_count() > 0

    var seen_null = False
    var have_previous = False
    var previous = UInt64(0)

    for i in range(n):
        if has_null and not col.data.validity.get(i):
            if nulls_first:
                # A null belongs at the front, so one here is in order only if
                # nothing that is not a null has come before it.
                if have_previous:
                    return False
            else:
                seen_null = True
            continue

        if seen_null:
            return False
        var k = sort_key(values.unsafe_offset(i).unsafe_load())
        if descending:
            k = ~k
        if have_previous and k < previous:
            return False
        previous = k
        have_previous = True

    return True
