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

An inner join exchanges them too, but for a different reason and with the order
put back. The scan is of the right side and the walk is of the left, so a call
written with the dimension table first buckets the whole fact table in order to
walk fifteen hundred rows, and the same join written the other way round does
almost no work at all. Nothing about the data says which side is which, only how
the caller spelled the call, and an inner join is symmetric in the rows it
produces. So when the left is enough shorter, the sides are exchanged, and the
pairing that comes back in right row order is counting sorted into left row
order before it is returned. `BUILD_SIDE_MARGIN` is how much shorter, and
`_by_left_row` is the sort. This is the first of the planner decisions to move
out of the caller's hands and it does not need a plan to make.

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

An outer join splits too, which took a second look. It has to remember which
right rows were paired so that it can emit the rest afterwards, and that memory
was a bitmap, whose set is a read modify write of a word eight rows share, so
two workers marking at once would drop marks and invent unmatched rows. That is
a fact about the bitmap and not about the remembering. What it marks now is a
byte per code, and two threads storing to two bytes are storing to two memory
locations however close together they sit, so what they share is a cache line,
which costs speed and never an answer. The codes are turned back into rows by
one walk of the built side after the emit, which is exact because a code is
paired whole or not at all.

## Two halves, kept apart

The scan that buckets one side and the walk that pairs the other are two
functions rather than one, `bucket_side` and `pair_probe`, with `ProbeTable`
between them. A whole frame join calls them in order and has no use for the
seam, but a join in a pipeline builds the table once from the frame it holds and
then walks every chunk that arrives against it, which is only expressible if the
table is a value and the walk takes a stretch of rows rather than a frame.

The walk numbers its output rows from the start of the stretch it was given, so
a chunk gets rows numbered within that chunk, and the caller shifts them if it
wants absolute ones. The row numbers from the built side are absolute either
way, since that side is one frame whichever chunk is going past.

## The two lists

Both index lists are the same length and both use a negative to mean "no row
here". That is not a sentinel invented for the occasion: `take_rows` already
treats a negative index as a null, which is exactly what an outer join needs to
put in the columns of the side that had nothing.
"""

from firepanda.array.any import AnyArray, ColumnRefs
from firepanda.array.array import Array
from firepanda.array.strings import StringArray, StringBuilder
from firepanda.bitmap.bitmap import Bitmap
from firepanda.dtype.lists import ALL
from firepanda.exec.morsel import parallel_morsels

from .keys import align_keys


comptime PARALLEL_LEFT_ROWS = 1 << 17
"""Below this many left rows the emit stays on one thread.

The walk is a handful of nanoseconds a row and a fork and join is tens of
microseconds, so the split has to be paying for itself over at least that many
rows before it is offered. Same shape of constant as `factorize`'s
`PARALLEL_ROWS` and picked the same way, from where the two costs cross.
"""

comptime BUILD_SIDE_MARGIN = 4
"""How much shorter the left has to be before an inner join exchanges the sides.

Building on the shorter side is the right answer on cost and the wrong answer on
order, so the exchange is not free: the pairing comes back in right row order
and has to be sorted back. The sort is a counting sort over the output, so the
saving has to cover one extra pass over the result before the exchange is worth
making, and that is what the margin is for rather than any property of the join.

Four rather than two because the count that matters is the output height and not
the left height, and a join that fans out pays the sort on every row it produces
while it only saves on the rows it scanned. Four is where the two crossed on a
dimension against fact join at sf1, and the crossover is shallow either side of
it. See `_by_left_row` for what the sort costs.
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
    making it a parameter would produce a copy of one function per kind to save one
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

    comptime MARK = Self(7)
    """Every left row, with a boolean saying whether it matched.

    A semi join answers the same question by keeping the rows that matched and
    dropping the rest, which is the answer a `WHERE` wants. Written anywhere
    else, an `IN` over a subquery is a value rather than a filter, and a value
    has to arrive on every row, including the rows that matched nothing. So the
    right side is kept out of the output the same way and what replaces it is
    one column.
    """

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
            False for the three that use the right side to decide something
            about the left rows and then discard it.
        """
        return self != Self.SEMI and self != Self.ANTI and self != Self.MARK

    def keeps_unmatched_left(self) -> Bool:
        """Reports whether a left row with no match still produces a row.

        Returns:
            True for left, outer, anti and mark.
        """
        return (
            self == Self.LEFT
            or self == Self.OUTER
            or self == Self.ANTI
            or self == Self.MARK
        )

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
        elif self == Self.MARK:
            writer.write("mark")
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


struct ProbeTable(Movable):
    """One side of a join scanned into a table from ordinal to row number.

    Two shapes in one struct, because most joins are onto a key that is unique
    on this side and a bucket per ordinal is the wrong shape for those. The
    unused shape is an empty `List`.
    """

    var unique: Bool
    """Whether every ordinal on this side belongs to exactly one row."""

    var only: List[Int32]
    """The unique shape: this side's row per ordinal, or -1, or empty."""

    var starts: List[Int]
    """The general shape: where each ordinal's run of rows begins, or empty.

    One longer than the ordinal count, so an ordinal's run is `starts[g]` up to
    `starts[g + 1]` and the last entry is the total.
    """

    var bucket: List[Int]
    """The general shape: this side's row numbers, grouped by ordinal.

    In increasing row order within an ordinal, which is what fixes the output
    order within a probe row.
    """

    var rows: Int
    """How many rows this side has."""

    def __init__(
        out self,
        unique: Bool,
        var only: List[Int32],
        var starts: List[Int],
        var bucket: List[Int],
        rows: Int,
    ):
        """Constructs a table.

        Args:
            unique: Whether the unique shape was taken.
            only: The unique shape's rows per ordinal, or an empty list.
            starts: The general shape's edges, or an empty list.
            bucket: The general shape's row numbers, or an empty list.
            rows: How many rows this side has.
        """
        self.unique = unique
        self.only = only^
        self.starts = starts^
        self.bucket = bucket^
        self.rows = rows

    def __init__(out self):
        """Constructs a table with no rows in it.

        For a caller that has to hold the field before it has the side to fill
        it from, which is what a pipeline node planning its schema does. Pairing
        against it is not an error, it is a join against an empty frame.
        """
        self.unique = True
        self.only = List[Int32]()
        self.starts = List[Int]()
        self.bucket = List[Int]()
        self.rows = 0


def join_indices[
    l: ImmOrigin, r: ImmOrigin
](
    left_columns: ColumnRefs[l],
    left_keys: List[Int],
    left_rows: Int,
    right_columns: ColumnRefs[r],
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
        dtypes, if a key dtype has no physical layout, or if the kind is the
        mark join, which answers a column rather than a pairing.
    """
    if kind == JoinKind.MARK:
        raise Error(
            "join: a mark join answers a column of booleans rather than a"
            " pairing, so it is not a pair of index lists and there is nothing"
            " for take_rows to do with it; it runs as a pipeline operator"
        )
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

    if (
        kind == JoinKind.INNER
        and left_rows * BUILD_SIDE_MARGIN <= right_rows
        and left_rows != 0
    ):
        # The build is a scan of the right side and the probe is a walk of the
        # left, so calling the dimension table `left` and the fact table `right`
        # buckets six million rows in order to walk fifteen hundred. Which side
        # is which is a fact about how the caller wrote the call and not about
        # the data, and an inner join is the one kind where exchanging them
        # changes nothing about which rows come out. So exchange them, and put
        # the order back afterwards.
        return _by_left_row(
            join_indices(
                right_columns,
                right_keys,
                right_rows,
                left_columns,
                left_keys,
                left_rows,
                JoinKind.INNER,
            ).swapped(),
            left_rows,
        )

    var aligned = align_keys(
        left_columns,
        left_keys,
        left_rows,
        right_columns,
        right_keys,
        right_rows,
    )
    var table = bucket_side(
        aligned.codes,
        left_rows,
        right_rows,
        aligned.absent,
        left_rows,
        aligned.has_nulls,
        aligned.groups,
    )

    # An outer join's unmatched right rows are appended after the pairing,
    # because knowing how many of them there are means having already done it.
    var wants_right = kind == JoinKind.OUTER
    var matched = Bitmap(right_rows if wants_right else 0, all_valid=False)
    var paired = pair_probe(
        table,
        aligned.codes,
        0,
        left_rows,
        aligned.absent,
        0,
        aligned.has_nulls,
        kind,
        matched,
    )
    if wants_right:
        for r in range(right_rows):
            if not matched.get(r):
                paired.left_at.append(-1)
                paired.right_at.append(r)
    return paired^


def _by_left_row(var paired: JoinIndices, left_rows: Int) -> JoinIndices:
    """Puts a pairing that came back in right row order into left row order.

    A counting sort rather than a comparison sort, because the key is a left row
    number and left row numbers are exactly the integers below `left_rows`, so
    there is nothing to compare. One pass to count, one prefix sum over the left
    height, one pass to scatter.

    The sort is stable and that is not a detail. The pairing arrives in right row
    order, so the entries for one left row arrive with their right rows already
    increasing, and a stable sort by left row leaves them that way. That is the
    order a join without the exchange produces, which is what makes the exchange
    invisible to a caller.

    Only reachable for an inner join, where every entry names a real row on both
    sides. A kind that can emit a negative left row would need a bucket for it
    and would have to decide where those rows go, and none of them come through
    here.

    Args:
        paired: The pairing, in right row order.
        left_rows: The left frame's height, which bounds the key.

    Returns:
        The same pairs in left row order, and in right row order within a left
        row.
    """
    var pairs = len(paired)
    if pairs == 0:
        return paired^

    # Offset by one so that the prefix sum below lands each row's run start in
    # its own slot rather than the next one's.
    var starts = List[Int](length=left_rows + 1, fill=0)
    for i in range(pairs):
        starts[paired.left_at[i] + 1] += 1
    for row in range(left_rows):
        starts[row + 1] += starts[row]

    var left = List[Int](length=pairs, fill=0)
    var right = List[Int](length=pairs, fill=0)
    for i in range(pairs):
        var row = paired.left_at[i]
        var at = starts[row]
        starts[row] = at + 1
        left[at] = row
        right[at] = paired.right_at[i]
    return JoinIndices(left^, right^)


def bucket_side(
    codes: Array[DType.uint32],
    side_at: Int,
    rows: Int,
    absent: List[Bool],
    absent_at: Int,
    has_nulls: Bool,
    groups: Int,
) raises -> ProbeTable:
    """Scans one side's ordinals into a table from ordinal to row number.

    Split out of `join_indices` because a streaming join builds this once from
    the frame it holds and then pairs every chunk that arrives against it, so it
    has to be a value that outlives the pass that filled it. Nothing in here
    looks at the other side.

    Args:
        codes: The ordinals of both sides, as `align_keys` returns them.
        side_at: Where this side's ordinals start in `codes`.
        rows: How many rows this side has.
        absent: The null key flags, or an empty list.
        absent_at: Where this side's flags start in `absent`.
        has_nulls: Whether `absent` was filled.
        groups: How many ordinals there are.

    Returns:
        The table, ready to be paired against.

    Raises:
        Error: If a read raises, which it does not.
    """
    var code_at = codes.unsafe_ptr()

    # Most joins are onto a key that is unique on the side being scanned, and a
    # bucket per code is the wrong shape for those: every bucket holds one row,
    # so the counts, the prefix sum, the cursor walk and the bucket array itself
    # all exist to express "one". A single table from code to row says the same
    # thing in one pass and one allocation, and the allocation is `int32` rather
    # than `Int`, so the widest join in the suite carries forty megabytes here
    # where the general shape carries a hundred and sixty.
    #
    # Uniqueness is not asked in advance, it is assumed and then contradicted.
    # The first row whose code is already taken ends the pass and the
    # general build runs from the top, having lost that much of one scan and
    # nothing else. A key with duplicates usually reaches its first one early,
    # and a key without them was never going to pay.
    var unique = rows <= Int(Int32.MAX)
    var only = List[Int32]()
    if unique:
        only = List[Int32](length=groups, fill=-1)
        var seat = only.unsafe_ptr()
        for r in range(rows):
            if has_nulls and absent[absent_at + r]:
                continue
            var g = Int(code_at.unsafe_offset(side_at + r).unsafe_load())
            if seat.unsafe_offset(g).unsafe_load() >= 0:
                unique = False
                break
            seat.unsafe_offset(g).unsafe_write(Int32(r))
        if not unique:
            only = List[Int32]()

    # Bucket this side by code: count, prefix sum, scatter. Scanning in
    # increasing row order is what puts each bucket in increasing row order,
    # which is what fixes the output order within a probe row.
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
        for r in range(rows):
            if has_nulls and absent[absent_at + r]:
                continue
            var g = Int(code_at.unsafe_offset(side_at + r).unsafe_load()) + 1
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
        for r in range(rows):
            if has_nulls and absent[absent_at + r]:
                continue
            var g = Int(code_at.unsafe_offset(side_at + r).unsafe_load())
            var at = edge.unsafe_offset(g).unsafe_load()
            into.unsafe_offset(at).unsafe_write(r)
            edge.unsafe_offset(g).unsafe_write(at + 1)
        for g in range(groups, 0, -1):
            edge.unsafe_offset(g).unsafe_write(
                edge.unsafe_offset(g - 1).unsafe_load()
            )
        edge.unsafe_offset(0).unsafe_write(0)

    return ProbeTable(unique, only^, starts^, bucket^, rows)


def pair_probe(
    table: ProbeTable,
    codes: Array[DType.uint32],
    probe_at: Int,
    probe_rows: Int,
    absent: List[Bool],
    absent_at: Int,
    has_nulls: Bool,
    kind: JoinKind,
    mut matched: Bitmap,
    spread: Bool = True,
) raises -> JoinIndices:
    """Walks one side against a built table and emits the pairs.

    The output row numbers on the probe side are counted from `probe_at`, so a
    caller pairing a chunk gets rows numbered within that chunk. The row numbers
    on the built side are absolute, since the table holds one frame whichever
    chunk is going past.

    Args:
        table: The other side, bucketed by ordinal.
        codes: The ordinals of both sides, as `align_keys` returns them.
        probe_at: Where this side's ordinals start in `codes`.
        probe_rows: How many rows this side has.
        absent: The null key flags, or an empty list.
        absent_at: Where this side's flags start in `absent`.
        has_nulls: Whether `absent` was filled.
        kind: Which rows to keep.
        matched: For an outer join, every built side row that paired is set.
            Sized to the built side's height, or empty for any other kind.
        spread: Whether this pairing may use more than one core. False when the
            caller is already running on a worker, because a second layer of
            tasks inside the first one is thirty two times the tasks and none of
            the parallelism.

    Returns:
        One entry per output row, in probe row order.

    Raises:
        Error: If the parallel emit raises, or if the kind is the mark join,
            which answers a column rather than a pairing.
    """
    if kind == JoinKind.MARK:
        raise Error(
            "join: a mark join answers a column rather than a pairing, since"
            " every probe row produces one output row and that row is the row"
            " that arrived; call mark_probe"
        )
    var wants_right = kind == JoinKind.OUTER

    # The probe side is walked in morsels, and a morsel is counted and then
    # emitted over the same rows. What a row emits depends on that row and on
    # the buckets, and the buckets are finished, so nothing here is shared
    # between two morsels except where each of them writes, which the counts
    # settle before any of them writes anything.
    #
    # An outer join used to stay on one thread, because its emit marks every
    # built side row it pairs and setting a bit is a read modify write of a word
    # that eight neighbouring rows share, so two workers doing it at once would
    # drop marks and invent unmatched rows. That was true of the bitmap and not
    # of the marking, and the marking is what the emit has to do. A byte per
    # group is written instead, below, and two threads storing to two bytes are
    # storing to two memory locations however close together they are, so the
    # only thing they share is a cache line, which costs speed and never an
    # answer. Nothing else in either pass is shared except where each morsel
    # writes, which the counts settle before any of them writes anything.
    var parallel = spread and probe_rows >= PARALLEL_LEFT_ROWS

    # The marks go against the group rather than against the row that group
    # holds, which is both what makes them safe and what takes them out of the
    # inner loop: a group is either paired or not, so a probe row that lands on
    # a bucket of nine marks once instead of nine times. Turning the groups back
    # into rows afterwards is one walk of the built side, against the ten million
    # bit writes it replaces.
    #
    # The emit reads the byte before writing it, which looks like a wasted load
    # and is the opposite. A store is what takes a cache line away from the other
    # cores, and a load is not, so a join onto a thousand keys whose marks are a
    # thousand bytes is thirty two cores fighting over sixteen lines ten million
    # times if it stores every time, and marking each of a thousand groups once
    # if it looks first. Two workers reading zero and both storing one is a race
    # that writes the same value twice, so there is nothing to protect.
    #
    # `join/outer` at ten million rows on an i9-13900K, three runs a side in ABBA
    # order, is 104.5, 109.8 and 115.3 ms on one thread against 34.9, 37.1 and
    # 37.1 with all of this, so just under three times. Storing the byte every
    # time instead of looking first lands at 56.3, 58.5 and 60.0, which is the
    # difference between splitting the work and splitting the work while thirty
    # two cores pass sixteen cache lines around. `join/inner_1000` at 25 ms and
    # `join/left_1000` at 25.7 ms are unmoved by any of it, which is what says
    # the sessions were comparable.
    var groups = len(table.only) if table.unique else max(
        len(table.starts) - 1, 0
    )
    var hit = List[UInt8](length=groups if wants_right else 0, fill=0)
    var chunk = LEFT_MORSEL_ROWS if parallel else max(probe_rows, 1)
    var pieces = (probe_rows + chunk - 1) // chunk
    if pieces == 0:
        # An empty left side rounds down to no morsels at all, and the passes
        # below still run once and still write where their morsel ends. One
        # empty morsel is the shape that costs nothing and reads correctly.
        pieces = 1

    # Count the output rows before emitting any, so the two lists are allocated
    # once at the right size. Same trade as `_filter_core`: a second pass over
    # the probe rows against a reallocation and a copy of everything already
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
    # its origin names `codes` and a capture list cannot carry that.
    def tally_one(start: Int, stop: Int) raises {mut counts, imm}:
        var code_at = codes.unsafe_ptr()
        var seat = table.only.unsafe_ptr()
        var here = 0
        for i in range(start, stop):
            var hit = False
            if not has_nulls or not absent[absent_at + i]:
                var g = Int(code_at.unsafe_offset(probe_at + i).unsafe_load())
                hit = seat.unsafe_offset(g).unsafe_load() >= 0
            if not hit:
                here += Int(kind.keeps_unmatched_left())
            elif kind != JoinKind.ANTI:
                here += 1
        counts[start // chunk + 1] = here

    def tally(start: Int, stop: Int) raises {mut counts, imm}:
        var code_at = codes.unsafe_ptr()
        var here = 0
        for i in range(start, stop):
            var width = 0
            if not has_nulls or not absent[absent_at + i]:
                var g = Int(code_at.unsafe_offset(probe_at + i).unsafe_load())
                width = table.starts[g + 1] - table.starts[g]
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
        if table.unique:
            tally_one(0, probe_rows)
        else:
            tally(0, probe_rows)
    elif table.unique:
        parallel_morsels(tally_one, probe_rows, chunk)
    else:
        parallel_morsels(tally, probe_rows, chunk)
    for m in range(pieces):
        counts[m + 1] += counts[m]

    var out_left = List[Int](unsafe_uninit_length=counts[pieces])
    var out_right = List[Int](unsafe_uninit_length=counts[pieces])

    # The unique twin of the loop below. A probe row pairs with at most one row
    # here, so the inner loop over a bucket is gone and with it the reason to
    # look up where the bucket started.
    def spill_one(
        start: Int, stop: Int
    ) raises {mut out_left, mut out_right, mut hit, imm}:
        var code_at = codes.unsafe_ptr()
        var seat = table.only.unsafe_ptr()
        var left_out = out_left.unsafe_ptr()
        var right_out = out_right.unsafe_ptr()
        var mark = hit.unsafe_ptr()
        var put = counts[start // chunk]
        for i in range(start, stop):
            var r = -1
            var g = -1
            if not has_nulls or not absent[absent_at + i]:
                g = Int(code_at.unsafe_offset(probe_at + i).unsafe_load())
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
            if wants_right and mark.unsafe_offset(g).unsafe_load() == 0:
                mark.unsafe_offset(g).unsafe_write(1)

    def spill(
        start: Int, stop: Int
    ) raises {mut out_left, mut out_right, mut hit, imm}:
        var code_at = codes.unsafe_ptr()
        var left_out = out_left.unsafe_ptr()
        var right_out = out_right.unsafe_ptr()
        var mark = hit.unsafe_ptr()
        var put = counts[start // chunk]
        for i in range(start, stop):
            var first = -1
            var last = -1
            var g = -1
            if not has_nulls or not absent[absent_at + i]:
                g = Int(code_at.unsafe_offset(probe_at + i).unsafe_load())
                first = table.starts[g]
                last = table.starts[g + 1]

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
                right_out.unsafe_offset(put).unsafe_write(table.bucket[p])
                put += 1
            if wants_right and mark.unsafe_offset(g).unsafe_load() == 0:
                mark.unsafe_offset(g).unsafe_write(1)

    if not parallel:
        if table.unique:
            spill_one(0, probe_rows)
        else:
            spill(0, probe_rows)
    elif table.unique:
        parallel_morsels(spill_one, probe_rows, chunk)
    else:
        parallel_morsels(spill, probe_rows, chunk)

    # Turn the paired groups back into paired rows. A group is emitted whole or
    # not at all, every row in its bucket going out against the probe row that
    # landed on it, so a marked group means every row it holds paired and an
    # unmarked one means none of them did. That is what makes this exact rather
    # than approximate, and it is why the mark could move off the row in the
    # first place.
    #
    # It is one thread, but it is one thread over the built side rather than
    # over the output, and it only writes where a group was hit.
    if wants_right:
        var mark = hit.unsafe_ptr()
        if table.unique:
            var seat = table.only.unsafe_ptr()
            for g in range(groups):
                if mark.unsafe_offset(g).unsafe_load() != 0:
                    matched.set(Int(seat.unsafe_offset(g).unsafe_load()), True)
        else:
            for g in range(groups):
                if mark.unsafe_offset(g).unsafe_load() != 0:
                    for p in range(table.starts[g], table.starts[g + 1]):
                        matched.set(table.bucket[p], True)

    return JoinIndices(out_left^, out_right^)


def mark_probe(
    table: ProbeTable,
    codes: Array[DType.uint32],
    probe_at: Int,
    probe_rows: Int,
    absent: List[Bool],
    absent_at: Int,
    has_nulls: Bool,
    built_has_nulls: Bool,
    spread: Bool = True,
) raises -> Array[DType.bool]:
    """Walks one side against a built table and answers whether each row matched.

    This is the mark join's whole emit. Nothing is gathered, because every probe
    row produces exactly one output row and that row is the row that arrived, so
    the caller keeps its chunk and puts this column beside it.

    ## The nulls are the point

    The question a mark join is asked is `x IN (SELECT k FROM ...)`, and SQL
    answers that with three values rather than two. A row that matched is true.
    A row that did not is false only when it is known to have matched nothing,
    and it is not known when a null was involved, because a null is a value
    nobody wrote down rather than a value that differs from everything.

    So a probe row whose own key is null is null, since it might have equalled
    whatever the missing value was. A probe row whose key is not null and that
    matched nothing is null as well when the built side holds a null key
    anywhere, for the same reason and on the other side of the comparison. Only
    when neither is true is the answer false.

    That rule is why `NOT IN` over a column with a null in it keeps no rows at
    all, which surprises people and is what every engine does. `NOT` of null is
    null, and a `WHERE` keeps a row on true.

    Whether the built side holds a null key is a property of the whole build
    side rather than of anything in this chunk, so the caller works it out once
    and passes it in.

    Args:
        table: The other side, bucketed by ordinal.
        codes: The ordinals of both sides, as `align_keys` returns them.
        probe_at: Where this side's ordinals start in `codes`.
        probe_rows: How many rows this side has.
        absent: The null key flags, or an empty list.
        absent_at: Where this side's flags start in `absent`.
        has_nulls: Whether `absent` was filled.
        built_has_nulls: Whether the built side holds a null key anywhere.
        spread: Whether this may use more than one core. False when the caller
            is already running on a worker, for the reason `pair_probe` gives.

    Returns:
        One boolean per probe row, null where the answer is unknown.

    Raises:
        Error: If the parallel walk raises.
    """
    var out = Array[DType.bool](overwritten=probe_rows)
    var valid = Bitmap(probe_rows, all_valid=True)
    var parallel = spread and probe_rows >= PARALLEL_LEFT_ROWS

    # The two routes the table has, kept apart the way `pair_probe` keeps them:
    # the unique shape asks whether the seat holds a row and the general shape
    # asks whether the bucket is wider than nothing, and the branch between them
    # is the same answer on every row.
    def walk_one(start: Int, stop: Int) raises {mut out, mut valid, imm}:
        var code_at = codes.unsafe_ptr()
        var seat = table.only.unsafe_ptr()
        var dst = out.unsafe_mut_ptr()
        for i in range(start, stop):
            if has_nulls and absent[absent_at + i]:
                dst.unsafe_offset(i).unsafe_write(False)
                valid.set(i, False)
                continue
            var g = Int(code_at.unsafe_offset(probe_at + i).unsafe_load())
            var hit = seat.unsafe_offset(g).unsafe_load() >= 0
            dst.unsafe_offset(i).unsafe_write(hit)
            if not hit and built_has_nulls:
                valid.set(i, False)

    def walk(start: Int, stop: Int) raises {mut out, mut valid, imm}:
        var code_at = codes.unsafe_ptr()
        var dst = out.unsafe_mut_ptr()
        for i in range(start, stop):
            if has_nulls and absent[absent_at + i]:
                dst.unsafe_offset(i).unsafe_write(False)
                valid.set(i, False)
                continue
            var g = Int(code_at.unsafe_offset(probe_at + i).unsafe_load())
            var hit = table.starts[g + 1] > table.starts[g]
            dst.unsafe_offset(i).unsafe_write(hit)
            if not hit and built_has_nulls:
                valid.set(i, False)

    # A morsel is a run of rows and a validity bit is one of eight in a byte, so
    # two morsels meeting inside a byte would be a read modify write of the same
    # location from two cores. The morsel length is a multiple of eight, which
    # is what keeps them out of each other's bytes, so the split is safe without
    # the byte per group trick `pair_probe` needs.
    if not parallel:
        if table.unique:
            walk_one(0, probe_rows)
        else:
            walk(0, probe_rows)
    elif table.unique:
        parallel_morsels(walk_one, probe_rows, LEFT_MORSEL_ROWS)
    else:
        parallel_morsels(walk, probe_rows, LEFT_MORSEL_ROWS)

    out.data.validity = valid^
    return out^


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
    var target = out.unsafe_mut_ptr()

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
