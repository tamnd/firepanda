"""Reordering and dropping rows: take and filter.

These two are where the vectorized style runs out. A gather reads a different
cache line per row and a compaction writes a variable number of them, so neither
loop has a shape the vector unit helps with on the targets firepanda builds for.
What can be done is to keep the branches out of the value loop, and both kernels
do that in the same way: read unconditionally, because a null holds a zero and
reading it is harmless, and build the validity bitmap separately.

What the vector unit will not do, the other cores will. A gather's output row
depends on its own index and on nothing else in the output, so `take_rows` splits
by output row, on boundaries rounded to a multiple of sixty four because the
validity bitmap is the one thing in there that is not per row. `filter_rows` has
no such shape: where a row lands depends on how many rows before it survived.

`take_rows` treats a negative index as a null. That is not a convenience, it is
how a left join reports that the row on the right did not exist, and it is why
the index list is signed.

`filter_rows` drops the rows where the mask is null. The alternative, keeping
them, would mean `filter(m)` and `filter(not m)` both contain the same row, which
no query engine does and pandas does not either.

Both kernels come in two spellings. The typed one takes an `Array[dt]` and is
what other kernels call. The erased one takes an `AnyArray` and is what a
`DataFrame` calls, because a frame holds a list of columns whose dtypes are only
known at runtime and differ from each other. They share a body: the typed entry
point hands its pointer and bitmap to the core, and the erased one walks `ALL`
and hands over the same two things. Routing the erased case through
`AnyArray.as_typed` instead would have been three lines shorter and would have
deep copied every column on the way in, which on a filter is the entire cost of
the operation paid twice.
"""

from std.memory import unsafe_memcpy

from firepanda.array.any import AnyArray
from firepanda.array.array import Array
from firepanda.array.strings import StringArray, StringBuilder
from firepanda.array.strview import VIEW_SIZE, StringView, make_long_at
from firepanda.bitmap.bitmap import Bitmap
from firepanda.buffer.buffer import Buffer
from firepanda.dtype.lists import ALL
from firepanda.exec import parallel_morsels
from firepanda.exec.parallel import parallel_for, worker_count


comptime PARALLEL_TAKE_ROWS = 1 << 16
"""Below this many gathered rows the take stays on one thread.

A gather is a cache miss a row, so it is a great deal more than a handful of
nanoseconds and the split pays off sooner than it does for a loop over
consecutive memory. Half of `join`'s threshold, and picked the same way.
"""

comptime TAKE_MORSEL_ROWS = 1 << 16
"""Output rows a worker takes at a time once the gather is on every core.

A multiple of sixty four, which is what makes the morsel boundaries land on
validity word boundaries and is the only coupling between this number and the
loop below.

Small, because the cost of a gathered row is the cost of the cache miss it takes
and that is not the same for every row: indices that walk a small region are hot
and indices that walk the whole column are not, and a join or a sort produces
both in the same call. Eight thousand rows is around ten microseconds of work,
which is four orders of magnitude more than the atomic that hands it out.
"""


def take_rows[
    dt: DType
](col: Array[dt], indices: List[Int]) raises -> Array[dt]:
    """Gathers rows by position.

    Args:
        col: The column to gather from.
        indices: The positions to gather. Each is either a valid position in
            `col` or negative, which produces a null.

    Parameters:
        dt: The dtype.

    Returns:
        A column of length `len(indices)`.
    """
    return _take_core(
        col.unsafe_ptr(), col.data.validity, col.null_count() > 0, indices
    )


def take_any(
    col: AnyArray, indices: List[Int], spread: Bool = True
) raises -> AnyArray:
    """Gathers rows by position from a column whose dtype is a runtime value.

    Args:
        col: The column to gather from.
        indices: The positions to gather, negative meaning null, as in
            `take_rows`.
        spread: Whether this gather may use more than one core. False when
            the caller is already running on a worker, because a second layer of
            tasks inside the first one is thirty two times the tasks and none of
            the parallelism.

    Returns:
        A column of length `len(indices)` with the same dtype as the input.

    Raises:
        If the column's dtype is not one firepanda has a physical layout for.
    """
    if col.is_string():
        return AnyArray(_take_strings(col.strings(), indices, spread))
    comptime for candidate in ALL:
        if col.dtype() == candidate:
            return AnyArray(
                _take_core(
                    col.unsafe_ptr[candidate](),
                    col.data.validity,
                    col.null_count() > 0,
                    indices,
                    spread,
                )
            )
    raise Error("take: unsupported dtype")


def _take_bounds(rows: Int, workers: Int) -> List[Int]:
    """Cuts a row range into pieces that share no word of the output validity.

    Args:
        rows: How many output rows there are.
        workers: How many pieces to cut them into.

    Returns:
        `workers + 1` boundaries, each a multiple of sixty four except the last.
    """
    var bounds = List[Int](capacity=workers + 1)
    for w in range(workers):
        var cut = ((rows * w // workers) + 63) // 64 * 64
        bounds.append(cut if cut < rows else rows)
    bounds.append(rows)
    return bounds^


def _take_strings(
    col: StringArray, indices: List[Int], spread: Bool = True
) raises -> StringArray:
    """Gathers variable width rows by position.

    The output element is not a fixed number of bytes, so unlike `_take_core`
    this cannot write every row unconditionally at a computed offset. What saves
    it is that a gather does not change how an element is stored: a short one
    stays short and a long one stays long and keeps its length. So an output
    row's view is sixteen bytes wide wherever it lands, and the only thing that
    depends on the rows before it is where its payload bytes go.

    That is one number a worker needs, so the split is by output row with a
    counting pass in front of it: each worker adds up the payload its own range
    will copy, the totals prefix sum into a base per worker, and then each worker
    writes its views and its payload with a cursor of its own. The counting pass
    is skipped when the column has no payload at all, which is every column of
    labels and is the case a group by's key gather actually hits.

    Args:
        col: The column to gather from.
        indices: The positions to gather. A negative index produces a null, as
            in `take_rows`, because that is how a left join reports a row that
            was not there.
        spread: Whether this gather may use more than one core. False when
            the caller is already running on a worker, because a second layer of
            tasks inside the first one is thirty two times the tasks and none of
            the parallelism.

    Returns:
        A column of length `len(indices)`.

    Raises:
        If an index is neither negative nor a position in the column, or if one
        of the workers the parallel route starts cannot be run.
    """
    var n = len(indices)
    var workers = worker_count()
    if n < PARALLEL_TAKE_ROWS or workers <= 1 or not spread:
        var builder = StringBuilder(capacity=n)
        for k in range(n):
            var at = indices[k]
            if at >= len(col):
                raise Error(
                    String(
                        "take index ", at, " is outside a column of ", len(col)
                    )
                )
            if at < 0 or not col.is_valid(at):
                builder.append_null()
            else:
                builder.append(col.unsafe_bytes(at))
        return builder^.finish()

    var most = n // PARALLEL_TAKE_ROWS
    if workers > most:
        workers = most
    var bounds = _take_bounds(n, workers)
    var height = len(col)
    var source_views = col.views.unsafe_ptr().unsafe_bitcast[StringView]()
    var source_bytes = col.payload.unsafe_ptr()

    # A column whose payload is empty has no element longer than twelve bytes,
    # so nothing is copied out of it and the counting pass has only one answer.
    var carried = List[Int](length=workers + 1, fill=0)
    if len(col.payload) > 0:
        var totals = Buffer(workers * 8)

        def measure(w: Int) raises {mut totals, imm}:
            var wide = 0
            for i in range(bounds[w], bounds[w + 1]):
                var at = indices[i]
                if at >= height:
                    raise Error(
                        String(
                            "take index ",
                            at,
                            " is outside a column of ",
                            height,
                        )
                    )
                if at < 0 or not col.is_valid(at):
                    continue
                var view = source_views.unsafe_offset(at)[]
                if not view.is_inline():
                    wide += len(view)
            totals.bitcast[DType.int64]().unsafe_offset(w).unsafe_store(
                Int64(wide)
            )

        parallel_for(measure, workers)

        var counted = totals.bitcast[DType.int64]()
        for w in range(workers):
            carried[w + 1] = carried[w] + Int(
                counted.unsafe_offset(w).unsafe_load()
            )

    var views = Buffer(overwritten=n * VIEW_SIZE)
    var payload = Buffer(overwritten=carried[workers])
    var built = Bitmap(n, all_valid=False)

    def gather(w: Int) raises {mut views, mut payload, mut built, imm}:
        var target = views.unsafe_ptr().unsafe_bitcast[StringView]()
        var into = payload.unsafe_ptr()
        var cursor = carried[w]

        # The output positions are consecutive, so the validity bits are built in
        # a register and stored once every sixty four rows. The boundaries are
        # multiples of sixty four, so no two workers write the same word.
        var word = UInt64(0)
        for i in range(bounds[w], bounds[w + 1]):
            var at = indices[i]
            if at >= height:
                raise Error(
                    String(
                        "take index ", at, " is outside a column of ", height
                    )
                )
            if at < 0 or not col.is_valid(at):
                # The view of the empty string, so that reading a null's bytes
                # gives an empty span rather than uninitialized memory, which is
                # what `StringBuilder.append_null` writes for the same reason.
                target.unsafe_offset(i)[] = StringView()
            else:
                var view = source_views.unsafe_offset(at)[]
                if view.is_inline():
                    # The bytes are already inside the sixteen, so the view is
                    # the whole element and copying it is the whole gather.
                    target.unsafe_offset(i)[] = view
                else:
                    var count = len(view)
                    var from_ = source_bytes.unsafe_offset(view.offset())
                    unsafe_memcpy(
                        dest=into.unsafe_offset(cursor),
                        src=from_,
                        count=count,
                    )
                    target.unsafe_offset(i)[] = make_long_at(
                        from_, count, 0, cursor
                    )
                    cursor += count
                word |= UInt64(1) << UInt64(i & 63)
            if i & 63 == 63:
                built.unsafe_set_word(i >> 6, word)
                word = 0

        # Only a range that ends part way through a word has anything left in the
        # register, and since the boundaries are multiples of sixty four that is
        # only ever the last one.
        var stop = bounds[w + 1]
        if stop & 63 != 0 and stop > bounds[w]:
            built.unsafe_set_word(stop >> 6, word)

    parallel_for(gather, workers)
    return StringArray(views^, payload^, built^, n)


def _take_core[
    dt: DType, //, origin: ImmOrigin
](
    source: Pointer[Scalar[dt], origin],
    validity: Bitmap,
    has_nulls: Bool,
    indices: List[Int],
    spread: Bool = True,
) raises -> Array[dt]:
    """The gather loop, over a pointer and a bitmap rather than a column."""
    var n = len(indices)

    # Not the zeroing constructor. The gather writes every output element,
    # including a zero where the index says null, so a memset in front of it is
    # a second pass over the output. It is also a pass on one thread, and it is
    # the pass that faults the output's pages in, so the whole column arrives on
    # whichever core happened to run the allocation. Writing the zero in the
    # loop instead hands each morsel's pages to the worker that is about to fill
    # them.
    var out = Array[dt](overwritten=n)
    var built = Bitmap(n, all_valid=False)

    # Output row `i` depends on `indices[i]` and on nothing else in the output,
    # so the gather splits by output row. The morsel size is a multiple of sixty
    # four so that no two workers write the same validity word, which is the only
    # thing here that is not per row.
    def gather(start: Int, stop: Int) raises {mut out, mut built, imm}:
        var target = out.unsafe_ptr()

        # The output positions are consecutive, so the validity bits can be
        # built in a register and stored once every sixty four rows instead of
        # read-modify-writing a byte per row. The input side has no such luck; a
        # gather is a gather.
        #
        # The validity probe is the second random read of the row, into a
        # different array from the values, and a column with no nulls does not
        # need it. A join gathers with a list that has negatives in it and a
        # source that usually does not have nulls, so the two halves of that
        # condition are worth keeping apart.
        var word = UInt64(0)
        for i in range(start, stop):
            var at = indices[i]
            if at >= 0 and (not has_nulls or validity.get(at)):
                target.unsafe_offset(i).unsafe_write(
                    source.unsafe_offset(at).unsafe_load()
                )
                word |= UInt64(1) << UInt64(i & 63)
            else:
                target.unsafe_offset(i).unsafe_write(Scalar[dt]())
            if i & 63 == 63:
                built.unsafe_set_word(i >> 6, word)
                word = 0

        # Only a morsel that ends part way through a word has anything left in
        # the register, and since the morsel is a multiple of sixty four that is
        # only ever the last one.
        if stop & 63 != 0 and stop > start:
            built.unsafe_set_word(stop >> 6, word)

    if n < PARALLEL_TAKE_ROWS or not spread:
        gather(0, n)
    else:
        parallel_morsels(gather, n, TAKE_MORSEL_ROWS)

    out.data.validity = built^
    return out^


def filter_rows[
    dt: DType
](col: Array[dt], mask: Array[DType.bool]) -> Array[dt]:
    """Keeps the rows where the mask is true.

    Two passes. The first counts the kept rows so the output can be allocated
    once at the right size, the second copies. Growing a buffer instead would
    save the counting pass and cost a reallocation and a copy of everything
    already written, several times, on a column that is usually large.

    The second pass comes in two versions and the split is on whether the column
    being filtered has any nulls. It usually does not, and that case is worth a
    lot: with no validity to carry across, the copy loop has no branch left in it.

    Args:
        col: The column to filter.
        mask: The mask. Must be the same length as `col`.

    Parameters:
        dt: The dtype.

    Returns:
        A column holding the kept rows, in their original order.
    """
    return _filter_core(
        col.unsafe_ptr(), col.data.validity, col.null_count() > 0, mask
    )


def filter_any(col: AnyArray, mask: Array[DType.bool]) raises -> AnyArray:
    """Keeps the rows where the mask is true, for a runtime dtype.

    Args:
        col: The column to filter.
        mask: The mask. Must be the same length as `col`.

    Returns:
        A column holding the kept rows, with the same dtype as the input.

    Raises:
        If the column's dtype is not one firepanda has a physical layout for.
    """
    if col.is_string():
        return AnyArray(_filter_strings(col.strings(), mask))
    comptime for candidate in ALL:
        if col.dtype() == candidate:
            return AnyArray(
                _filter_core(
                    col.unsafe_ptr[candidate](),
                    col.data.validity,
                    col.null_count() > 0,
                    mask,
                )
            )
    raise Error("filter: unsupported dtype")


def _filter_strings(
    col: StringArray, mask: Array[DType.bool]
) raises -> StringArray:
    """Keeps the variable width rows where the mask is true.

    A null in the mask drops the row, the same rule `_filter_core` follows and
    for the same reason. There is no branch free version of this one: the
    trick in `_filter_core` is to write every row and advance the cursor by the
    mask bit, which only works when a row that nobody keeps costs a fixed number
    of bytes that the next row overwrites.

    Args:
        col: The column to filter.
        mask: The mask. Must be as tall as the column.

    Returns:
        A column holding the kept rows in their original order.

    Raises:
        If the mask is not as tall as the column.
    """
    if len(mask) != len(col):
        raise Error(
            String(
                "filter mask has ",
                len(mask),
                " rows and the column has ",
                len(col),
            )
        )
    var values = mask.unsafe_ptr()
    var builder = StringBuilder(capacity=len(col))
    for i in range(len(col)):
        if not mask.data.validity.get(i):
            continue
        if not Bool(values.unsafe_offset(i).unsafe_load()):
            continue
        if col.is_valid(i):
            builder.append(col.unsafe_bytes(i))
        else:
            builder.append_null()
    return builder^.finish()


def _filter_core[
    dt: DType, //, origin: ImmOrigin
](
    source: Pointer[Scalar[dt], origin],
    validity: Bitmap,
    has_null: Bool,
    mask: Array[DType.bool],
) -> Array[dt]:
    """The compaction loop, over a pointer and a bitmap rather than a column."""
    var n = len(mask)
    var mask_values = mask.unsafe_ptr()

    var kept = 0
    for i in range(n):
        if not mask.data.validity.get(i):
            continue
        if Bool(mask_values.unsafe_offset(i).unsafe_load()):
            kept += 1

    # Every one of the kept positions is written below, on both routes, so this
    # does not need the zeroing constructor either. The branchless loop writes
    # the row before it decides whether to keep it, which means it writes every
    # output slot at least once and the last write to a slot is the row that
    # belongs there.
    var out = Array[dt](overwritten=kept)
    var target = out.unsafe_ptr()

    if not has_null:
        # Nothing to record, because a filter of a column with no nulls has no
        # nulls, and `Array` starts out all present. That leaves a loop with no
        # branch in it at all: every row is written at the output cursor and the
        # cursor advances by the mask bit, so a dropped row is simply overwritten
        # by the next one. The mask is data and the branch predictor cannot learn
        # it, which is why removing the branch is worth writing a row nobody
        # keeps.
        var written = 0
        var i = 0
        while written < kept:
            var present = mask.data.validity.get(i)
            var truthy = Bool(mask_values.unsafe_offset(i).unsafe_load())
            target.unsafe_offset(written).unsafe_write(
                source.unsafe_offset(i).unsafe_load()
            )
            written += Int(present and truthy)
            i += 1
        return out^

    var built = Bitmap(kept, all_valid=False)

    # As in `take_rows`, the output positions are consecutive and the validity
    # goes down a word at a time. Here it matters more, because the row being
    # written is not the row being read and the byte the bit lives in would be a
    # second unpredictable memory reference per kept row.
    var at = 0
    var word = UInt64(0)
    for i in range(n):
        if not mask.data.validity.get(i):
            continue
        if not Bool(mask_values.unsafe_offset(i).unsafe_load()):
            continue
        target.unsafe_offset(at).unsafe_write(
            source.unsafe_offset(i).unsafe_load()
        )
        if validity.get(i):
            word |= UInt64(1) << UInt64(at & 63)
        at += 1
        if at & 63 == 0:
            built.unsafe_set_word((at >> 6) - 1, word)
            word = 0

    if at & 63 != 0:
        built.unsafe_set_word(at >> 6, word)

    out.data.validity = built^
    return out^


def take_range(start: Int, indices: List[Int]) raises -> Array[DType.int64]:
    """Gathers rows out of the arithmetic range `start`, `start + 1`, and so on.

    A row label is `start + at` and depends on nothing that has to exist first,
    so this never builds the range it gathers from. That is the whole reason it
    is a separate entry point: the obvious route, materializing the range into a
    column and handing it to `take_rows`, costs an allocation and a pass
    proportional to the height going in rather than the height coming out, and on
    a million rows it cost more than the gather it was decorating.

    Everything else here is `_take_core` with the random read taken out. The
    output is written unconditionally, the validity is built a word at a time in
    a register rather than a bit at a time through the bitmap, and the loop goes
    on every core above the same threshold, which matters more than it looks like
    it should: this is the pass that faults in the output's pages, and a serial
    one hangs the whole column off whichever core ran the allocation while the
    columns beside it are being gathered in parallel.

    Args:
        start: The first label of the range.
        indices: The positions to gather. Each is either a position in the range
            or negative, which produces a null.

    Returns:
        The gathered labels, of length `len(indices)`.

    Raises:
        Error: If the output cannot be allocated.
    """
    var n = len(indices)
    var out = Array[DType.int64](overwritten=n)
    var built = Bitmap(n, all_valid=False)
    var base = Int64(start)

    def gather(begin: Int, stop: Int) raises {mut out, mut built, imm}:
        var target = out.unsafe_ptr()
        var word = UInt64(0)
        for i in range(begin, stop):
            var at = indices[i]
            if at >= 0:
                target.unsafe_offset(i).unsafe_write(base + Int64(at))
                word |= UInt64(1) << UInt64(i & 63)
            else:
                target.unsafe_offset(i).unsafe_write(Int64(0))
            if i & 63 == 63:
                built.unsafe_set_word(i >> 6, word)
                word = 0

        if stop & 63 != 0 and stop > begin:
            built.unsafe_set_word(stop >> 6, word)

    if n < PARALLEL_TAKE_ROWS:
        gather(0, n)
    else:
        parallel_morsels(gather, n, TAKE_MORSEL_ROWS)

    out.data.validity = built^
    return out^


def filter_range(start: Int, mask: Array[DType.bool]) -> Array[DType.int64]:
    """Keeps the labels of the rows a mask keeps, out of an arithmetic range.

    The counterpart of `take_range` and separate for the same reason. It is
    `_filter_core`'s no-null route with the read replaced by arithmetic, and it
    is always that route rather than sometimes the other one, because a range has
    no missing label to carry: the label of row `i` is `start + i` for every row
    there is. So there is no validity to build and the copy loop has no branch in
    it, the label is written at the output cursor and the cursor advances by the
    mask bit, and a row that is dropped is simply overwritten by the next one.

    A null in the mask drops the row, which is the rule `filter_rows` follows and
    is not the same as the label being null.

    Args:
        start: The first label of the range.
        mask: The mask. Must be as long as the range.

    Returns:
        The labels of the kept rows, in their original order.
    """
    var n = len(mask)
    var mask_values = mask.unsafe_ptr()

    var kept = 0
    for i in range(n):
        if not mask.data.validity.get(i):
            continue
        if Bool(mask_values.unsafe_offset(i).unsafe_load()):
            kept += 1

    var out = Array[DType.int64](overwritten=kept)
    var target = out.unsafe_ptr()
    var base = Int64(start)

    var written = 0
    var i = 0
    while written < kept:
        var present = mask.data.validity.get(i)
        var truthy = Bool(mask_values.unsafe_offset(i).unsafe_load())
        target.unsafe_offset(written).unsafe_write(base + Int64(i))
        written += Int(present and truthy)
        i += 1

    return out^
