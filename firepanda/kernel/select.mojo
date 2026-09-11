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
no such shape, because where a row lands depends on how many rows before it
survived, so it counts first: one parallel pass over the mask gives every morsel
the number of rows it keeps, a prefix sum turns those into the output positions
the morsels start at, and the second pass is then as independent as a gather's.

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
from std.sys.intrinsics import PrefetchOptions, prefetch

from firepanda.array.any import AnyArray
from firepanda.array.array import Array
from firepanda.array.strings import StringArray, StringBuilder
from firepanda.array.strview import VIEW_SIZE, StringView, make_long_at
from firepanda.bitmap.bitmap import Bitmap
from firepanda.buffer.buffer import Buffer
from firepanda.dtype.lists import ALL
from firepanda.exec import parallel_morsels
from firepanda.exec.parallel import parallel_for, worker_count
from firepanda.kernel.dictionary import with_categories


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

comptime PARALLEL_FILTER_ROWS = 1 << 16
"""Below this many input rows a filter stays on one thread.

Higher than it would be for a gather, because a filter reads its input in order
and a gather does not, so a filtered row is a few nanoseconds rather than a cache
miss. The same number as the take threshold in the end, which is a coincidence of
two different arguments landing in the same place rather than a shared constant.

Both routes use it, the fixed width one and the variable width one, and they pay
different things for the split. The variable width route has to size its payload
before it can write it, the fixed width one has to count its kept rows before it
knows where each morsel starts, and both of those are a pass over the mask that
the serial version does not make. They come out close enough that a second
threshold would be a number with nothing behind it.
"""

comptime FILTER_MORSEL_ROWS = 1 << 16
"""Input rows a worker counts, and then compacts, at a time.

Unlike the take's morsel this one does not have to be a multiple of sixty four,
because a filter's output positions do not line up with its input positions and
no rounding of the input would make them. It is the same size anyway, which on
`lineitem` at sf1 is ninety two morsels for thirty two cores, enough that a
morsel whose mask happens to be all true does not hold up the end.
"""

comptime FILTER_PACK_WORDS = 1 << 10
"""Validity words a worker packs at a time, when the filtered column has nulls.

A word is sixty four output rows, so this is the same sixty five thousand rows
as the morsel above, and the packing pass is over the output rather than the
input.
"""

comptime TAKE_LOOKAHEAD = 8
"""Rows the gather runs ahead of itself when issuing prefetches.

The index list is in memory before the loop starts, so where row `i` will read
from is known long before row `i` is reached, and the only reason the load is
late is that nothing asked for it early. Reading `indices[i + 8]` and prefetching
the line it points at turns eight misses that would have been taken one after
another into eight that are outstanding at once, which is the same trick the hash
table's batch probe plays and for the same reason.

Eight, matching `PROBE_LOOKAHEAD`, and picked the same way: far enough ahead that
a miss to memory has time to land, close enough that the lines prefetched are
still there when the loop arrives.

What it is worth is entirely a question of whether the column being gathered from
fits in cache, and the join queries in db-benchmark sit on both sides of that. The
big join gathers from a hundred million rows, eight hundred megabytes against a
thirty six megabyte L3, and there it is worth three per cent: four ABBA passes a
side on a 13900K, every run with the prefetch below every run without it, 0.659
to 0.667 seconds against 0.673 to 0.690, and 45.1 CPU seconds against 46.6. The
medium join gathers from a million rows, eight megabytes, which is L3 resident,
and there it is worth nothing measurable: the same eight passes come out fully
interleaved. The join microbenchmarks gather from a hundred thousand rows and are
interleaved too.

So it never pays for itself on a small gather and it never costs anything either,
which is what makes it worth having unconditionally rather than behind a size
test. A branch and a load per row against a memory latency that a large gather
takes on every row is not a trade that needs tuning.
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
        return AnyArray(_take_strings(col.strings(), indices, spread)).retyped(
            col.type
        )
    comptime for candidate in ALL:
        if col.dtype() == candidate:
            return with_categories(
                AnyArray(
                    _take_core(
                        col.unsafe_ptr[candidate](),
                        col.data.validity,
                        col.null_count() > 0,
                        indices,
                        spread,
                    )
                ).retyped(col.type),
                col,
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
            totals.mut_bitcast[DType.int64]().unsafe_offset(w).unsafe_store(
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
        var target = views.unsafe_mut_ptr().unsafe_bitcast[StringView]()
        var into = payload.unsafe_mut_ptr()
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
        var target = out.unsafe_mut_ptr()

        # A run of consecutive ascending indices is a copy, and it is not a rare
        # shape. An inner join that matches every probe row once hands the left
        # side an index list that is `0, 1, 2, ...`, because the output is in
        # probe order and every probe row produced exactly one output row, so
        # every column taken from the probe side is being gathered by the
        # identity. A limit and a slice are the same thing offset.
        #
        # The check is a pass over the indices and it is almost free on the
        # columns it does not help: an index list that is not a run stops the
        # loop at the first row that is not, which for a genuinely scattered
        # list is the second row. What it saves when it does hold is the eight
        # byte index load and the branch on every row, and it hands the copy to
        # a memcpy that moves a cache line at a time instead of an element.
        #
        # Four ABBA passes a side on a 13900K, join microbenchmarks at ten
        # million rows, milliseconds without the run check against with it.
        # `join/inner_1000`, four gathered columns, 24.06 23.96 24.06 24.36
        # against 22.49 22.19 22.20 22.54. `join/inner_projected`, two of them,
        # 15.13 15.12 15.14 15.48 against 14.73 14.55 14.51 14.75.
        # `join/two_keys` 43.8 42.4 42.2 47.9 against 40.1 40.1 40.6 40.8.
        # `join/outer` 37.4 37.8 37.8 40.8 against 36.2 36.1 36.8 37.1. Every
        # run with it below every run without it in all four.
        #
        # The controls say the check costs nothing where it fails.
        # `join/indices_1000` pairs without gathering anything and comes out
        # 7.24 7.21 7.18 7.23 against 7.29 7.17 7.19 7.26, `join/semi` and
        # `join/anti` are interleaved the same way, and `join/many_to_many`,
        # whose left indices repeat and so are never a run, is interleaved too.
        #
        # The db-benchmark joins at a hundred million rows move less, because at
        # that size the gather is waiting on memory rather than on instructions.
        # CPU seconds for three timed runs, without against with: j1 6.22 6.27
        # 6.23 6.16 against 5.94 5.81 5.97 6.11, j2 12.49 12.74 12.84 12.55
        # against 11.98 12.10 12.16 12.34, j3 12.61 12.81 11.81 12.59 against
        # 12.01 12.10 12.22 12.25. Four per cent of the CPU on j1 and j2, and
        # one to two per cent of the wall clock, which is what is left once the
        # eight hundred megabytes still have to be read either way.
        #
        # Only when the source has no nulls. With nulls the output's validity is
        # a bit shifted copy of a range of the input's, which is a different and
        # more delicate loop than the two here, and a join gathering from a
        # column with nulls is not the case this is for.
        if not has_nulls and stop > start and indices[start] >= 0:
            var first = indices[start]
            var run = True
            for i in range(start + 1, stop):
                if indices[i] != first + (i - start):
                    run = False
                    break
            if run:
                unsafe_memcpy(
                    dest=target.unsafe_offset(start),
                    src=source.unsafe_offset(first),
                    count=stop - start,
                )
                # Every row of the run is valid, so the words are filled rather
                # than accumulated. The last word of the last morsel is the only
                # partial one, and the bits above `stop` in it belong to no row.
                var w = start >> 6
                while (w + 1) << 6 <= stop:
                    built.unsafe_set_word(w, UInt64.MAX)
                    w += 1
                if stop & 63 != 0:
                    built.unsafe_set_word(
                        w, (UInt64(1) << UInt64(stop & 63)) - 1
                    )
                return

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
        var ahead = min(start + TAKE_LOOKAHEAD, stop)
        for i in range(start, stop):
            # The line row `i + 8` is going to read, asked for now. Only the
            # values array, not the validity bitmap: a bitmap covering the whole
            # column is a sixty fourth of its size and the row that misses on the
            # values usually hits on the bits, so prefetching both would double
            # the instructions to halve a cost that is already small.
            if ahead < stop:
                var next = indices[ahead]
                if next >= 0:
                    prefetch[PrefetchOptions().for_read().high_locality()](
                        source.unsafe_offset(next)
                    )
                ahead += 1

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
](col: Array[dt], mask: Array[DType.bool]) raises -> Array[dt]:
    """Keeps the rows where the mask is true.

    Two passes. The first counts the kept rows so the output can be allocated
    once at the right size, the second copies. Growing a buffer instead would
    save the counting pass and cost a reallocation and a copy of everything
    already written, several times, on a column that is usually large.

    The second pass comes in two versions and the split is on whether the column
    being filtered has any nulls. It usually does not, and that case is worth a
    lot: with no validity to carry across, the copy loop has no branch left in it.

    Above `PARALLEL_FILTER_ROWS` both passes run on every core. Counting is what
    makes that possible, since it is the count that tells a worker starting in
    the middle of the mask where its output begins; see `_filter_spread`.

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
        return AnyArray(_filter_strings(col.strings(), mask)).retyped(col.type)
    comptime for candidate in ALL:
        if col.dtype() == candidate:
            return with_categories(
                AnyArray(
                    _filter_core(
                        col.unsafe_ptr[candidate](),
                        col.data.validity,
                        col.null_count() > 0,
                        mask,
                    )
                ).retyped(col.type),
                col,
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

    What it does have is the split `_take_strings` uses, and it needs it more.
    A filtered row is sixteen bytes of view wherever it lands, so the only thing
    a worker has to be told is how many rows and how many payload bytes the
    workers before it produced. Both come out of a counting pass, and then every
    worker writes its own stretch of the output with no coordination at all.
    Before this the loop went through `StringBuilder`, which appends a view and a
    null flag to two growing lists and then copies both into the finished column,
    and on six million single character labels that was ninety milliseconds
    against eight for the same filter over a column of doubles.

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
    var n = len(col)
    var values = mask.unsafe_ptr()
    var workers = worker_count()
    if n < PARALLEL_FILTER_ROWS or workers <= 1:
        var builder = StringBuilder(capacity=n)
        for i in range(n):
            if not mask.data.validity.get(i):
                continue
            if not Bool(values.unsafe_offset(i).unsafe_load()):
                continue
            if col.is_valid(i):
                builder.append(col.unsafe_bytes(i))
            else:
                builder.append_null()
        return builder^.finish()

    var most = n // PARALLEL_FILTER_ROWS
    if workers > most:
        workers = most
    var bounds = _take_bounds(n, workers)
    var source_views = col.views.unsafe_ptr().unsafe_bitcast[StringView]()
    var source_bytes = col.payload.unsafe_ptr()
    var wide_payload = len(col.payload) > 0

    # Where a worker's rows land depends on how many rows the workers before it
    # kept, and where their bytes land depends on how many bytes those workers
    # copied, so both are counted before anything is written. The byte count is
    # skipped when the column has no payload at all, which is every column of
    # labels and is the case a filter over a status column actually hits.
    var kept_totals = Buffer(workers * 8)
    var byte_totals = Buffer(workers * 8)

    def measure(w: Int) raises {mut kept_totals, mut byte_totals, imm}:
        var rows = 0
        var wide = 0
        for i in range(bounds[w], bounds[w + 1]):
            if not mask.data.validity.get(i):
                continue
            if not Bool(values.unsafe_offset(i).unsafe_load()):
                continue
            rows += 1
            if wide_payload and col.is_valid(i):
                var view = source_views.unsafe_offset(i)[]
                if not view.is_inline():
                    wide += len(view)
        kept_totals.mut_bitcast[DType.int64]().unsafe_offset(w).unsafe_store(
            Int64(rows)
        )
        byte_totals.mut_bitcast[DType.int64]().unsafe_offset(w).unsafe_store(
            Int64(wide)
        )

    parallel_for(measure, workers)

    var rows_before = List[Int](length=workers + 1, fill=0)
    var bytes_before = List[Int](length=workers + 1, fill=0)
    var counted_rows = kept_totals.bitcast[DType.int64]()
    var counted_bytes = byte_totals.bitcast[DType.int64]()
    for w in range(workers):
        rows_before[w + 1] = rows_before[w] + Int(
            counted_rows.unsafe_offset(w).unsafe_load()
        )
        bytes_before[w + 1] = bytes_before[w] + Int(
            counted_bytes.unsafe_offset(w).unsafe_load()
        )

    var kept = rows_before[workers]
    var views = Buffer(overwritten=kept * VIEW_SIZE)
    var payload = Buffer(overwritten=bytes_before[workers])
    var nulls = not col.validity.all_valid()
    var built = Bitmap(kept, all_valid=not nulls)

    def compact(w: Int) raises {mut views, mut payload, imm}:
        var target = views.unsafe_mut_ptr().unsafe_bitcast[StringView]()
        var into = payload.unsafe_mut_ptr()
        var at = rows_before[w]
        var cursor = bytes_before[w]
        for i in range(bounds[w], bounds[w + 1]):
            if not mask.data.validity.get(i):
                continue
            if not Bool(values.unsafe_offset(i).unsafe_load()):
                continue
            if not col.is_valid(i):
                # The view of the empty string, so that reading a null's bytes
                # gives an empty span rather than uninitialized memory, which is
                # what `StringBuilder.append_null` writes for the same reason.
                target.unsafe_offset(at)[] = StringView()
            else:
                var view = source_views.unsafe_offset(i)[]
                if view.is_inline():
                    target.unsafe_offset(at)[] = view
                else:
                    var count = len(view)
                    var from_ = source_bytes.unsafe_offset(view.offset())
                    unsafe_memcpy(
                        dest=into.unsafe_offset(cursor),
                        src=from_,
                        count=count,
                    )
                    target.unsafe_offset(at)[] = make_long_at(
                        from_, count, 0, cursor
                    )
                    cursor += count
            at += 1

    parallel_for(compact, workers)

    # A worker's first output row can land in the middle of a validity word that
    # the worker before it also writes to, which is the one thing the take route
    # does not have to worry about because there the output row is the input row.
    # Rather than lock a word or align the cuts, the bits go down in one pass
    # here, and only when there is a null to record.
    if nulls:
        var at = 0
        var word = UInt64(0)
        for i in range(n):
            if not mask.data.validity.get(i):
                continue
            if not Bool(values.unsafe_offset(i).unsafe_load()):
                continue
            if col.is_valid(i):
                word |= UInt64(1) << UInt64(at & 63)
            at += 1
            if at & 63 == 0:
                built.unsafe_set_word((at >> 6) - 1, word)
                word = 0
        if at & 63 != 0:
            built.unsafe_set_word(at >> 6, word)

    return StringArray(views^, payload^, built^, kept)


def _filter_core[
    dt: DType, //, origin: ImmOrigin
](
    source: Pointer[Scalar[dt], origin],
    validity: Bitmap,
    has_null: Bool,
    mask: Array[DType.bool],
) raises -> Array[dt]:
    """The compaction loop, over a pointer and a bitmap rather than a column."""
    var n = len(mask)
    var mask_values = mask.unsafe_ptr()

    if n >= PARALLEL_FILTER_ROWS:
        return _filter_spread(source, validity, has_null, mask)

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
    var target = out.unsafe_mut_ptr()

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


def _filter_spread[
    dt: DType, //, origin: ImmOrigin
](
    source: Pointer[Scalar[dt], origin],
    validity: Bitmap,
    has_null: Bool,
    mask: Array[DType.bool],
) raises -> Array[dt]:
    """The compaction loop on every core.

    The serial version is two passes, one to count and one to copy, and the
    reason it stayed serial for so long is that the copy looks unsplittable:
    where a row lands depends on how many rows before it survived, so a worker
    starting in the middle of the mask does not know where to write. Counting
    first answers exactly that question. The first pass gives every morsel the
    number of rows it keeps, a prefix sum over those turns into the output
    position each morsel begins at, and the copy is then as independent as a
    gather's, each worker writing a run of output nobody else touches.

    That leaves the mask read twice rather than once, which is why this waits
    for `PARALLEL_FILTER_ROWS`. A mask is a byte a row against eight for a
    double and sixteen for a text view, so the extra pass is a small fraction of
    what the copy moves, and it buys the whole rest of the machine.

    The copy loop is the serial one unchanged, including the trick of writing
    every row and advancing the cursor by the mask bit rather than branching on
    it, and including the bound: a worker stops when it has written the number
    of rows it counted, which is what keeps its last speculative write inside
    its own run and out of the next worker's first slot.

    Nulls in the filtered column are the one thing that does not fall out for
    free, because sixty four output rows share a validity word and two morsels
    can land in the same one. Rather than synchronize on it, the copy records a
    byte a kept row and a third pass packs those bytes into words, which is
    independent again because a word is sixty four consecutive output rows. The
    byte buffer is the size of the answer, not the size of the input.

    Args:
        source: The values to filter.
        validity: The values' validity.
        has_null: Whether `validity` has anything in it worth reading.
        mask: The mask. A null in it drops the row.

    Parameters:
        dt: The dtype.
        origin: Where `source` is borrowed from.

    Returns:
        A column holding the kept rows, in their original order.

    Raises:
        Error: If a worker fails, or if the output cannot be allocated.
    """
    var n = len(mask)
    var mask_values = mask.unsafe_ptr()
    var dense = mask.data.validity.all_valid()
    var morsels = (n + FILTER_MORSEL_ROWS - 1) // FILTER_MORSEL_ROWS
    var offsets = List[Int](length=morsels + 1, fill=0)

    def count(w: Int) raises {mut offsets, imm}:
        var begin = w * FILTER_MORSEL_ROWS
        var stop = begin + FILTER_MORSEL_ROWS
        if stop > n:
            stop = n
        var seen = 0
        if dense:
            for i in range(begin, stop):
                seen += Int(Bool(mask_values.unsafe_offset(i).unsafe_load()))
        else:
            for i in range(begin, stop):
                var truthy = Bool(mask_values.unsafe_offset(i).unsafe_load())
                seen += Int(truthy and mask.data.validity.get(i))
        offsets[w + 1] = seen

    parallel_for(count, morsels)

    for m in range(morsels):
        offsets[m + 1] += offsets[m]
    var kept = offsets[morsels]

    var out = Array[dt](overwritten=kept)
    var flags = Buffer(overwritten=kept if has_null else 0)

    def compact(w: Int) raises {mut out, mut flags, imm}:
        var target = out.unsafe_mut_ptr()
        var marks = flags.mut_bitcast[DType.uint8]()
        var limit = offsets[w + 1]
        var written = offsets[w]
        var i = w * FILTER_MORSEL_ROWS
        while written < limit:
            target.unsafe_offset(written).unsafe_write(
                source.unsafe_offset(i).unsafe_load()
            )
            if has_null:
                marks.unsafe_offset(written).unsafe_write(
                    UInt8(1) if validity.get(i) else UInt8(0)
                )
            var truthy = Bool(mask_values.unsafe_offset(i).unsafe_load())
            if dense:
                written += Int(truthy)
            else:
                written += Int(truthy and mask.data.validity.get(i))
            i += 1

    parallel_for(compact, morsels)

    if not has_null:
        return out^

    var built = Bitmap(kept, all_valid=False)
    var words = built.word_count()

    def pack(w: Int) raises {mut built, imm}:
        var marks = flags.bitcast[DType.uint8]()
        var first = w * FILTER_PACK_WORDS
        var last = first + FILTER_PACK_WORDS
        if last > words:
            last = words
        for word in range(first, last):
            var base = word << 6
            var stop = base + 64
            if stop > kept:
                stop = kept
            var bits = UInt64(0)
            for i in range(base, stop):
                if marks.unsafe_offset(i).unsafe_load() != 0:
                    bits |= UInt64(1) << UInt64(i - base)
            built.unsafe_set_word(word, bits)

    if words > 0:
        parallel_for(pack, (words + FILTER_PACK_WORDS - 1) // FILTER_PACK_WORDS)

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
        var target = out.unsafe_mut_ptr()
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
    var target = out.unsafe_mut_ptr()
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
