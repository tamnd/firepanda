"""Turning two frames and a set of key columns into a list of row pairs.

A join is two questions and this file answers only the first one. Which rows on
the left go with which rows on the right, and then, separately, what the output
columns should be. Keeping them apart means the answer to the first is a pair of
index lists, which the frame layer feeds straight to `take_rows`, and which every
join kind can produce without knowing anything about dtypes or schemas.

## The key alignment

Matching rows means comparing key tuples across two frames, and the two frames
have nothing in common: a code from `factorize` on the left is a number in the
left column's own space and means nothing on the right. `keys.mojo` is what puts
them in one space, either by building a dictionary on the smaller side and
probing the larger with it, or by concatenating both sides and factorizing the
lot. This file takes the ordinals and does not care which route produced them.

What it does care about is that a row matching nothing is an ordinal rather than
a special case. Whatever route ran, an unmatched row comes back holding an
ordinal that no row of the other side holds, so its bucket is empty and the emit
reads it the same way it reads every other row rather than testing for it.

## Nulls do not match nulls

A row whose key contains a null matches nothing, including another null. That is
what SQL does and what Polars does by default. pandas `merge` disagrees and joins
NaN keys together, which is a decision it made when NaN was the only missing
value it had; firepanda has a validity bitmap and does not need to.

The rows are not dropped. A null key on the left is an unmatched left row, so a
left join keeps it with nulls on the right and an anti join keeps it too.

## The build side and the output order

The right side is scanned once into buckets, one per code, holding the row
numbers in increasing order. Then the left side is walked in order and each row
emits one pair per row in its bucket. So the output is in left row order, and
within a left row, in right row order. Nothing about that is required by
anything, and it is fixed rather than incidental, because a join whose row order
moves between runs cannot be tested against another engine.

A right join is the same operation with the sides exchanged, so it is
implemented that way: swap, run a left join, swap the result back. That makes a
right join come out in right row order, which is what pandas does and what
somebody asking for a right join instead of a left one is asking for.

Buckets are the general answer and most joins do not need the general answer. A
join onto a primary key, which is most of them, has one right row per code, and
then the counts, the prefix sum, the cursor walk and the bucket array itself are
all machinery for saying "one". So the build starts by assuming the right key is
unique and filling one table from code to row, and the first code it finds
already taken abandons that and runs the general build from the top. The cost of
being wrong is part of one scan of the right side; the cost of not trying was
two extra walks of a table as long as the frame, plus an array as long as it
again.

## Splitting the emit

The left walk is the tallest part of the pairing on a wide result, and what a
left row emits depends on that row and on the finished buckets and on nothing
else, so it splits by left row across cores. The catch is that a worker cannot
append: it has to know where in the output its piece begins, which is the sum of
what every piece before it emits. So the walk is done twice, once counting and
once writing, with a prefix sum in between, and both passes cut the left side the
same way so that a piece is counted and written by the same arithmetic.

The pieces are morsels rather than one per worker, because what a left row costs
is the size of the bucket it matches and that is not a constant. A left side
whose hot keys sit together, which is what a fact table sorted by date looks
like, hands one worker the whole expensive range and leaves the rest waiting.
Morsels cost a longer prefix sum, over the morsel count rather than the worker
count, and that is a few hundred additions on a join large enough to be split at
all.

An outer join is the exception and stays on one thread. It has to remember which
right rows were paired so that it can emit the rest afterwards, and that memory
is a bitmap, whose set is a read modify write of a word eight rows share.

## The two lists

Both index lists are the same length and both use a negative to mean "no row
here". That is not a sentinel invented for the occasion: `take_rows` already
treats a negative index as a null, which is exactly what an outer join needs to
put in the columns of the side that had nothing.
"""

from firepanda.array.any import AnyArray
from firepanda.array.array import Array
from firepanda.array.strings import StringArray, StringBuilder
from firepanda.bitmap.bitmap import Bitmap
from firepanda.dtype.lists import ALL
from firepanda.exec import parallel_morsels

from .keys import align_keys


comptime PARALLEL_LEFT_ROWS = 1 << 17
"""Below this many left rows the emit stays on one thread.

The walk is a handful of nanoseconds a row and a fork and join is tens of
microseconds, so the split has to be paying for itself over at least that many
rows before it is offered. Same shape of constant as `factorize`'s
`PARALLEL_ROWS` and picked the same way, from where the two costs cross.
"""

comptime LEFT_MORSEL_ROWS = 1 << 15
"""Left rows a worker takes at a time once the emit is on every core.

What a left row costs is the size of the bucket it matches, and in a real join
that is not a constant. An order line joined against a customer table pairs with
one row; the same probe against a table with one enormous key pairs with
millions. Cut the left side into one piece per worker up front and the worker
holding the hot range decides when the join finishes.

The count pass and the emit pass have to agree on the boundaries, since the emit
starts writing where the count said it would, so both walk the same morsels.
"""


struct JoinKind(Equatable, ImplicitlyCopyable, Movable, Writable):
    """Which rows a join keeps.

    A runtime tag rather than a parameter, on the same grounds as `AggKind`: the
    kind picks which rows are emitted and the emitting loop is the same loop, so
    making it a parameter would produce seven copies of one function to save one
    comparison per row.
    """

    var code: UInt8
    """Which join."""

    def __init__(out self, code: UInt8):
        """Constructs a kind.

        Args:
            code: Which join.
        """
        self.code = code

    comptime INNER = Self(0)
    """Rows that matched, and nothing else."""

    comptime LEFT = Self(1)
    """Every left row, with nulls where the right had no match."""

    comptime RIGHT = Self(2)
    """Every right row, with nulls where the left had no match."""

    comptime OUTER = Self(3)
    """Every row from both sides, matched where possible."""

    comptime SEMI = Self(4)
    """Left rows that had at least one match, once each, with no right columns."""

    comptime ANTI = Self(5)
    """Left rows that had no match, with no right columns."""

    comptime CROSS = Self(6)
    """Every left row against every right row, with no keys."""

    def __eq__(self, other: Self) -> Bool:
        """Compares two kinds.

        Args:
            other: The kind to compare against.

        Returns:
            True if they are the same join.
        """
        return self.code == other.code

    def __ne__(self, other: Self) -> Bool:
        """Compares two kinds for inequality.

        Args:
            other: The kind to compare against.

        Returns:
            True if they are different joins.
        """
        return self.code != other.code

    def keeps_right_columns(self) -> Bool:
        """Reports whether the result carries the right frame's columns.

        Returns:
            False for the two filtering joins, which use the right side to
            decide which left rows survive and then discard it.
        """
        return self != Self.SEMI and self != Self.ANTI

    def keeps_unmatched_left(self) -> Bool:
        """Reports whether a left row with no match still produces a row.

        Returns:
            True for left, outer and anti.
        """
        return self == Self.LEFT or self == Self.OUTER or self == Self.ANTI

    def write_to(self, mut writer: Some[Writer]):
        """Writes the name a user would recognise.

        Args:
            writer: The sink.
        """
        if self == Self.INNER:
            writer.write("inner")
        elif self == Self.LEFT:
            writer.write("left")
        elif self == Self.RIGHT:
            writer.write("right")
        elif self == Self.OUTER:
            writer.write("outer")
        elif self == Self.SEMI:
            writer.write("semi")
        elif self == Self.ANTI:
            writer.write("anti")
        else:
            writer.write("cross")


struct JoinIndices(Movable, Sized):
    """One row of the result per entry, as a pair of source row numbers."""

    var left_at: List[Int]
    """Which left row each output row comes from, or negative for none."""

    var right_at: List[Int]
    """Which right row each output row comes from, or negative for none."""

    def __init__(out self, var left_at: List[Int], var right_at: List[Int]):
        """Constructs a pairing.

        Args:
            left_at: The left row numbers.
            right_at: The right row numbers. Must be the same length.
        """
        self.left_at = left_at^
        self.right_at = right_at^

    def __len__(self) -> Int:
        """Returns the number of output rows.

        Returns:
            The row count.
        """
        return len(self.left_at)

    def swapped(deinit self) -> Self:
        """Exchanges the two sides.

        Returns:
            The same pairing with left and right reversed.
        """
        var left = self.left_at^
        var right = self.right_at^
        return Self(right^, left^)


def join_indices(
    left_columns: List[AnyArray],
    left_keys: List[Int],
    left_rows: Int,
    right_columns: List[AnyArray],
    right_keys: List[Int],
    right_rows: Int,
    kind: JoinKind,
) raises -> JoinIndices:
    """Pairs the rows of two frames on a set of key columns.

    Args:
        left_columns: The left frame's columns.
        left_keys: Which of them are keys, most significant first.
        left_rows: The left frame's height.
        right_columns: The right frame's columns.
        right_keys: Which of them are keys, matched positionally with
            `left_keys`.
        right_rows: The right frame's height.
        kind: Which rows to keep.

    Returns:
        One entry per output row, in left row order for every kind but right,
        which comes out in right row order.

    Raises:
        If the key lists disagree in length, if they are empty for a kind that
        needs keys or non-empty for a cross join, if a key pair has different
        dtypes, or if a key dtype has no physical layout.
    """
    if kind == JoinKind.CROSS:
        if len(left_keys) != 0 or len(right_keys) != 0:
            raise Error("cross join: takes no key columns")
        return _cross(left_rows, right_rows)

    if len(left_keys) != len(right_keys):
        raise Error(
            "join: needs the same number of keys on each side; got "
            + String(len(left_keys))
            + " on the left and "
            + String(len(right_keys))
            + " on the right"
        )
    if len(left_keys) == 0:
        raise Error("join: at least one key column is required")

    if kind == JoinKind.RIGHT:
        # The same operation with the sides exchanged, which is also what makes
        # the result come out in right row order.
        return join_indices(
            right_columns,
            right_keys,
            right_rows,
            left_columns,
            left_keys,
            left_rows,
            JoinKind.LEFT,
        ).swapped()

    var aligned = align_keys(
        left_columns,
        left_keys,
        left_rows,
        right_columns,
        right_keys,
        right_rows,
    )
    var codes = aligned.codes.unsafe_ptr()
    var groups = aligned.groups
    var has_nulls = aligned.has_nulls
    ref absent = aligned.absent

    # Most joins are onto a key that is unique on the right, and a bucket per
    # code is the wrong shape for those: every bucket holds one row, so the
    # counts, the prefix sum, the cursor walk and the bucket array itself all
    # exist to express "one". A single table from code to row says the same
    # thing in one pass and one allocation, and the allocation is `int32` rather
    # than `Int`, so the widest join in the suite carries forty megabytes here
    # where the general shape carries a hundred and sixty.
    #
    # Uniqueness is not asked in advance, it is assumed and then contradicted.
    # The first right row whose code is already taken ends the pass and the
    # general build runs from the top, having lost that much of one scan and
    # nothing else. A key with duplicates usually reaches its first one early,
    # and a key without them was never going to pay.
    var unique = right_rows <= Int(Int32.MAX)
    var only = List[Int32]()
    if unique:
        only = List[Int32](length=groups, fill=-1)
        var seat = only.unsafe_ptr()
        for r in range(right_rows):
            if has_nulls and absent[left_rows + r]:
                continue
            var g = Int(codes.unsafe_offset(left_rows + r).unsafe_load())
            if seat.unsafe_offset(g).unsafe_load() >= 0:
                unique = False
                break
            seat.unsafe_offset(g).unsafe_write(Int32(r))
        if not unique:
            only = List[Int32]()

    # Bucket the right side by code: count, prefix sum, scatter. Scanning in
    # increasing row order is what puts each bucket in increasing row order,
    # which is what fixes the output order within a left row.
    #
    # The group table is as long as the number of distinct key tuples, which on
    # a join between two frames that are mostly one to one is as long as the
    # frames themselves. So it is worth not walking it more times than the three
    # this needs, and the loops below go through the pointer rather than through
    # `List.__setitem__` for the same reason.
    var starts = List[Int](length=0 if unique else groups + 1, fill=0)
    var bucket = List[Int]()
    if not unique:
        var edge = starts.unsafe_ptr()
        for r in range(right_rows):
            if has_nulls and absent[left_rows + r]:
                continue
            var g = Int(codes.unsafe_offset(left_rows + r).unsafe_load()) + 1
            edge.unsafe_offset(g).unsafe_write(
                edge.unsafe_offset(g).unsafe_load() + 1
            )
        for g in range(groups):
            edge.unsafe_offset(g + 1).unsafe_write(
                edge.unsafe_offset(g + 1).unsafe_load()
                + edge.unsafe_offset(g).unsafe_load()
            )

        # The scatter uses the group table as its own cursor rather than a copy
        # of it. Group `g` is written from `starts[g]` up to `starts[g + 1]`, so
        # when the scatter finishes every entry holds what its successor held,
        # and one backwards pass puts them back. That is a sequential walk over
        # the table instead of another allocation of it and a copy into it, and
        # `starts[g]` for an empty group already equals `starts[g + 1]`, so a
        # group nothing was scattered into comes out the same way.
        bucket = List[Int](
            unsafe_uninit_length=edge.unsafe_offset(groups).unsafe_load()
        )
        var into = bucket.unsafe_ptr()
        for r in range(right_rows):
            if has_nulls and absent[left_rows + r]:
                continue
            var g = Int(codes.unsafe_offset(left_rows + r).unsafe_load())
            var at = edge.unsafe_offset(g).unsafe_load()
            into.unsafe_offset(at).unsafe_write(r)
            edge.unsafe_offset(g).unsafe_write(at + 1)
        for g in range(groups, 0, -1):
            edge.unsafe_offset(g).unsafe_write(
                edge.unsafe_offset(g - 1).unsafe_load()
            )
        edge.unsafe_offset(0).unsafe_write(0)

    var wants_right = kind == JoinKind.OUTER
    var matched = Bitmap(right_rows if wants_right else 0, all_valid=False)

    # The left side is walked in morsels, and a morsel is counted and then
    # emitted over the same rows. What a left row emits depends on that row and
    # on the buckets, and the buckets are finished, so nothing here is shared
    # between two morsels except where each of them writes, which the counts
    # settle before any of them writes anything.
    #
    # An outer join stays on one thread. Its emit marks every right row it pairs
    # in `matched`, and setting a bit is a read modify write of a word that eight
    # neighbouring rows share, so two workers doing it at once would drop marks
    # and invent unmatched right rows. No other kind touches anything shared.
    var parallel = left_rows >= PARALLEL_LEFT_ROWS and not wants_right
    var chunk = LEFT_MORSEL_ROWS if parallel else max(left_rows, 1)
    var pieces = (left_rows + chunk - 1) // chunk
    if pieces == 0:
        # An empty left side rounds down to no morsels at all, and the passes
        # below still run once and still write where their morsel ends. One
        # empty morsel is the shape that costs nothing and reads correctly.
        pieces = 1

    # Count the output rows before emitting any, so the two lists are allocated
    # once at the right size. Same trade as `_filter_core`: a second pass over
    # the left rows against a reallocation and a copy of everything already
    # written, several times, on a result that a many-to-many join makes far
    # taller than either input. Here it buys the split as well, because a morsel
    # can only write where the morsels before it stopped.
    #
    # `counts[m + 1]` is what morsel `m` emits until the prefix sum turns it into
    # where morsel `m + 1` starts, which leaves the total in `counts[pieces]`.
    # The sum is serial over the morsel count rather than the row count, which at
    # thirty two thousand rows a morsel is a few hundred additions on a join
    # large enough to be here at all.
    var counts = List[Int](length=pieces + 1, fill=0)

    # The unique route is written as its own pair of loops rather than as a
    # branch inside these two. The branch would be the same answer on every one
    # of ten million rows and it is still a compare and a jump on every one of
    # them, and the loop it sits in does about four other things. Splitting them
    # measured the difference between the widest joins moving and the narrow
    # ones going backwards.
    #
    # The pointer is taken again inside each body rather than captured, because
    # its origin names `aligned` and a capture list cannot carry that.
    def tally_one(start: Int, stop: Int) raises {mut counts, imm}:
        var code_at = aligned.codes.unsafe_ptr()
        var seat = only.unsafe_ptr()
        var here = 0
        for i in range(start, stop):
            var hit = False
            if not has_nulls or not absent[i]:
                var g = Int(code_at.unsafe_offset(i).unsafe_load())
                hit = seat.unsafe_offset(g).unsafe_load() >= 0
            if not hit:
                here += Int(kind.keeps_unmatched_left())
            elif kind != JoinKind.ANTI:
                here += 1
        counts[start // chunk + 1] = here

    def tally(start: Int, stop: Int) raises {mut counts, imm}:
        var code_at = aligned.codes.unsafe_ptr()
        var here = 0
        for i in range(start, stop):
            var width = 0
            if not has_nulls or not absent[i]:
                var g = Int(code_at.unsafe_offset(i).unsafe_load())
                width = starts[g + 1] - starts[g]
            if width == 0:
                here += Int(kind.keeps_unmatched_left())
            elif kind == JoinKind.ANTI:
                continue
            elif kind == JoinKind.SEMI:
                here += 1
            else:
                here += width
        counts[start // chunk + 1] = here

    if not parallel:
        if unique:
            tally_one(0, left_rows)
        else:
            tally(0, left_rows)
    elif unique:
        parallel_morsels(tally_one, left_rows, chunk)
    else:
        parallel_morsels(tally, left_rows, chunk)
    for m in range(pieces):
        counts[m + 1] += counts[m]

    # An outer join's unmatched right rows are appended after the split, because
    # knowing how many of them there are means having already done the probe.
    var out_left = List[Int](unsafe_uninit_length=counts[pieces])
    var out_right = List[Int](unsafe_uninit_length=counts[pieces])

    # The unique twin of the loop below. A left row pairs with at most one right
    # row here, so the inner loop over a bucket is gone and with it the reason
    # to look up where the bucket started.
    def spill_one(
        start: Int, stop: Int
    ) raises {mut out_left, mut out_right, mut matched, imm}:
        var code_at = aligned.codes.unsafe_ptr()
        var seat = only.unsafe_ptr()
        var left_out = out_left.unsafe_ptr()
        var right_out = out_right.unsafe_ptr()
        var put = counts[start // chunk]
        for i in range(start, stop):
            var r = -1
            if not has_nulls or not absent[i]:
                var g = Int(code_at.unsafe_offset(i).unsafe_load())
                r = Int(seat.unsafe_offset(g).unsafe_load())

            if r < 0:
                if kind.keeps_unmatched_left():
                    left_out.unsafe_offset(put).unsafe_write(i)
                    right_out.unsafe_offset(put).unsafe_write(-1)
                    put += 1
                continue

            if kind == JoinKind.ANTI:
                continue
            if kind == JoinKind.SEMI:
                left_out.unsafe_offset(put).unsafe_write(i)
                right_out.unsafe_offset(put).unsafe_write(-1)
                put += 1
                continue

            left_out.unsafe_offset(put).unsafe_write(i)
            right_out.unsafe_offset(put).unsafe_write(r)
            put += 1
            if wants_right:
                matched.set(r, True)

    def spill(
        start: Int, stop: Int
    ) raises {mut out_left, mut out_right, mut matched, imm}:
        var code_at = aligned.codes.unsafe_ptr()
        var left_out = out_left.unsafe_ptr()
        var right_out = out_right.unsafe_ptr()
        var put = counts[start // chunk]
        for i in range(start, stop):
            var first = -1
            var last = -1
            if not has_nulls or not absent[i]:
                var g = Int(code_at.unsafe_offset(i).unsafe_load())
                first = starts[g]
                last = starts[g + 1]

            if first == last:
                if kind.keeps_unmatched_left():
                    left_out.unsafe_offset(put).unsafe_write(i)
                    right_out.unsafe_offset(put).unsafe_write(-1)
                    put += 1
                continue

            if kind == JoinKind.ANTI:
                continue
            if kind == JoinKind.SEMI:
                left_out.unsafe_offset(put).unsafe_write(i)
                right_out.unsafe_offset(put).unsafe_write(-1)
                put += 1
                continue

            for p in range(first, last):
                left_out.unsafe_offset(put).unsafe_write(i)
                right_out.unsafe_offset(put).unsafe_write(bucket[p])
                put += 1
                if wants_right:
                    matched.set(bucket[p], True)

    if not parallel:
        if unique:
            spill_one(0, left_rows)
        else:
            spill(0, left_rows)
    elif unique:
        parallel_morsels(spill_one, left_rows, chunk)
    else:
        parallel_morsels(spill, left_rows, chunk)

    if wants_right:
        for r in range(right_rows):
            if not matched.get(r):
                out_left.append(-1)
                out_right.append(r)

    return JoinIndices(out_left^, out_right^)


def take_pair(
    a: AnyArray, b: AnyArray, a_at: List[Int], b_at: List[Int]
) raises -> AnyArray:
    """Gathers from one column where a row exists in it and from the other where
    it does not.

    This is what a shared key column needs after an outer join. `country` came
    from both frames and the output has one of it, so for a row that only the
    right side had, the value has to come from the right side. Taking it from the
    left would produce a null in the very column the row was matched on.

    Args:
        a: The preferred source.
        b: The fallback.
        a_at: One row number per output row, negative to fall back.
        b_at: The fallback row numbers, the same length.

    Returns:
        A column of `len(a_at)` rows.

    Raises:
        If the dtypes differ or have no physical layout.
    """
    if a.is_string() != b.is_string():
        raise Error(
            "join: cannot combine a text key column with one of "
            + String(b.dtype() if a.is_string() else a.dtype())
        )
    if a.dtype() != b.dtype():
        raise Error(
            "join: cannot combine key columns of "
            + String(a.dtype())
            + " and "
            + String(b.dtype())
        )
    # Before the dispatch, because uint8 is in ALL and two text columns would
    # match it and produce a key column holding the first byte of each view.
    if a.is_string():
        return AnyArray(_pair_strings(a.strings(), b.strings(), a_at, b_at))
    comptime for candidate in ALL:
        if a.dtype() == candidate:
            return AnyArray(
                _pair_core(
                    a.unsafe_ptr[candidate](),
                    a.data.validity,
                    b.unsafe_ptr[candidate](),
                    b.data.validity,
                    a_at,
                    b_at,
                )
            )
    raise Error("join: unsupported key dtype")


def _pair_core[
    dt: DType, //, first: ImmOrigin, second: ImmOrigin
](
    a: Pointer[Scalar[dt], first],
    a_valid: Bitmap,
    b: Pointer[Scalar[dt], second],
    b_valid: Bitmap,
    a_at: List[Int],
    b_at: List[Int],
) -> Array[dt]:
    """The two-source gather, over pointers and bitmaps rather than columns."""
    var n = len(a_at)
    var out = Array[dt](n)
    var target = out.unsafe_ptr()

    # A fresh column is zeroed and all present, so the only work per row is the
    # value when there is one and the bit when there is not.
    for i in range(n):
        var at = a_at[i]
        if at < 0:
            at = b_at[i]
            if at >= 0 and b_valid.get(at):
                target.unsafe_offset(i).unsafe_store(
                    b.unsafe_offset(at).unsafe_load()
                )
            else:
                out.data.validity.set(i, False)
            continue
        if a_valid.get(at):
            target.unsafe_offset(i).unsafe_store(
                a.unsafe_offset(at).unsafe_load()
            )
        else:
            out.data.validity.set(i, False)
    return out^


def _pair_strings(
    a: StringArray,
    b: StringArray,
    a_at: List[Int],
    b_at: List[Int],
) -> StringArray:
    """The two-source gather for text.

    Appending rather than writing into a fresh column, because the output's
    payload size is not known until the rows have been chosen and a string
    column is built once and not edited.
    """
    var n = len(a_at)
    var builder = StringBuilder(capacity=n)
    for i in range(n):
        var at = a_at[i]
        if at < 0:
            at = b_at[i]
            if at >= 0 and b.is_valid(at):
                builder.append(b.unsafe_bytes(at))
            else:
                builder.append_null()
            continue
        if a.is_valid(at):
            builder.append(a.unsafe_bytes(at))
        else:
            builder.append_null()
    return builder^.finish()


def _cross(left_rows: Int, right_rows: Int) -> JoinIndices:
    """Pairs every left row with every right row.

    No keys, no hashing, no null rule. The only thing worth saying about it is
    the size: the result is the product of the two heights, so a cross join of
    two hundred thousand row frames is forty billion rows and will not finish.
    The size is the caller's problem because there is no honest default limit,
    and refusing above some threshold would be an arbitrary number in the way of
    the one legitimate use, which is a small table on one side.

    Args:
        left_rows: The left frame's height.
        right_rows: The right frame's height.

    Returns:
        The pairing, in left row order.
    """
    var total = left_rows * right_rows
    var out_left = List[Int](capacity=total)
    var out_right = List[Int](capacity=total)
    for i in range(left_rows):
        for r in range(right_rows):
            out_left.append(i)
            out_right.append(r)
    return JoinIndices(out_left^, out_right^)
