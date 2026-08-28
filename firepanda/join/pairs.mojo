"""Turning two frames and a set of key columns into a list of row pairs.

A join is two questions and this file answers only the first one. Which rows on
the left go with which rows on the right, and then, separately, what the output
columns should be. Keeping them apart means the answer to the first is a pair of
index lists, which the frame layer feeds straight to `take_rows`, and which every
join kind can produce without knowing anything about dtypes or schemas.

## The key alignment trick

Matching rows means comparing key tuples across two frames, and the two frames
have nothing in common: a code from `factorize` on the left is a number in the
left column's own space and means nothing on the right.

So this does not factorize the two sides separately. It concatenates each key
column with its opposite number into one column of `left_rows + right_rows`
values, hands the whole set to `group_ordinals`, and slices the codes back apart
afterwards. Two rows share a code exactly when they share a key tuple, whichever
side they came from, and the multi-key packing, the densifying and the small
integer fast path all come along for free because they are already in there.

The bill is one extra pass over each key column plus the memory to hold the
copy. That is real and it is worth it here, because the alternative is a second
implementation of key handling that has to agree with the first one about null
keys, float normalization and integer ranges, and the two would drift.

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
from firepanda.hash.grouping import group_ordinals
from firepanda.kernel.concat import concat_two_any


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

    var rows = left_rows + right_rows
    var merged = List[AnyArray](capacity=len(left_keys))
    for k in range(len(left_keys)):
        merged.append(
            _concat_any(
                left_columns[left_keys[k]], right_columns[right_keys[k]]
            )
        )

    var at = List[Int](capacity=len(merged))
    for k in range(len(merged)):
        at.append(k)

    var grouping = group_ordinals(merged, at, rows)
    var codes = grouping.codes.unsafe_ptr()
    var groups = grouping.groups

    # A row whose key tuple contains a null matches nothing. Reading that off the
    # concatenated columns rather than the originals means one loop covers both
    # sides and the row numbering matches the codes.
    var absent = List[Bool](capacity=rows)
    for i in range(rows):
        var missing = False
        for k in range(len(merged)):
            if not merged[k].is_valid(i):
                missing = True
                break
        absent.append(missing)

    # Bucket the right side by code: count, prefix sum, scatter. Scanning in
    # increasing row order is what puts each bucket in increasing row order,
    # which is what fixes the output order within a left row.
    var starts = List[Int](capacity=groups + 1)
    for _ in range(groups + 1):
        starts.append(0)
    for r in range(right_rows):
        if absent[left_rows + r]:
            continue
        starts[Int(codes.unsafe_offset(left_rows + r).unsafe_load()) + 1] += 1
    for g in range(groups):
        starts[g + 1] += starts[g]

    var total = starts[groups]
    var bucket = List[Int](capacity=total)
    for _ in range(total):
        bucket.append(0)
    var cursor = List[Int](capacity=groups)
    for g in range(groups):
        cursor.append(starts[g])
    for r in range(right_rows):
        if absent[left_rows + r]:
            continue
        var g = Int(codes.unsafe_offset(left_rows + r).unsafe_load())
        bucket[cursor[g]] = r
        cursor[g] += 1

    var wants_right = kind == JoinKind.OUTER
    var matched = Bitmap(right_rows if wants_right else 0, all_valid=False)

    # Count the output rows before emitting any, so the two lists are allocated
    # once at the right size. Same trade as `_filter_core`: a second pass over
    # the left rows against a reallocation and a copy of everything already
    # written, several times, on a result that a many-to-many join makes far
    # taller than either input. An outer join's right side contributes an upper
    # bound rather than a count, because knowing how many right rows went
    # unmatched means having already done the probe.
    var expected = 0
    for i in range(left_rows):
        var width = 0
        if not absent[i]:
            var g = Int(codes.unsafe_offset(i).unsafe_load())
            width = starts[g + 1] - starts[g]
        if width == 0:
            expected += Int(kind.keeps_unmatched_left())
        elif kind == JoinKind.ANTI:
            continue
        elif kind == JoinKind.SEMI:
            expected += 1
        else:
            expected += width
    if wants_right:
        expected += right_rows

    var out_left = List[Int](capacity=expected)
    var out_right = List[Int](capacity=expected)

    for i in range(left_rows):
        var first = -1
        var last = -1
        if not absent[i]:
            var g = Int(codes.unsafe_offset(i).unsafe_load())
            first = starts[g]
            last = starts[g + 1]

        if first == last:
            if kind.keeps_unmatched_left():
                out_left.append(i)
                out_right.append(-1)
            continue

        if kind == JoinKind.ANTI:
            continue
        if kind == JoinKind.SEMI:
            out_left.append(i)
            out_right.append(-1)
            continue

        for p in range(first, last):
            out_left.append(i)
            out_right.append(bucket[p])
            if wants_right:
                matched.set(bucket[p], True)

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


def _concat_any(a: AnyArray, b: AnyArray) raises -> AnyArray:
    """Appends one column to another, for columns whose dtype is a runtime value.

    The two dtypes have to match exactly. Promoting them here instead would mean
    a join silently comparing an int32 column against a float64 one and finding
    fewer matches than either side expected, so the cast is the caller's to write
    and is visible where it happens.

    Args:
        a: The first column.
        b: The second column.

    Returns:
        A column of `len(a) + len(b)` rows.

    Raises:
        If the dtypes differ or have no physical layout.
    """
    if a.dtype() != b.dtype():
        raise Error(
            "join: key columns must have the same dtype; got "
            + String(a.dtype())
            + " and "
            + String(b.dtype())
        )
    return concat_two_any(a, b)
