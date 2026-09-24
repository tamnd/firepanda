"""Ranks, one float a row, over the whole column or within groups of rows.

This is `Series.rank`, `DataFrame.rank` and `groupby(...).rank()`. All three
are the same walk with a different grouping, so they share it: a group by hands
in a group number for every row and a plain rank hands in none, which is one
group holding every row.

The walk is a sort and a pass. The column is sorted once, stably, in the
direction asked for. The sorted rows are then dealt into their groups by a
counting pass that keeps the sorted order inside each group, so no second sort
is needed, and each group's rows are walked in order. Equal values are adjacent
in the order, and whether two neighbours are equal is read off the column's
factorize codes rather than by comparing the values, which is one integer
comparison whatever the type is, text and categories included.

The rules below were measured against pandas 3.0.

- A missing value is a null or a NaN. With `na_option="keep"` it ranks as NaN.
  With `"top"` the missing rows of a group rank before every value in it, and
  with `"bottom"` they rank after. The missing rows of a group tie with each
  other either way.
- A tie is settled by the method: `average` answers the mean of the ranks the
  tie spans, `min` and `max` its ends, `dense` the tie's own position among the
  distinct values, and `first` the order the rows appear in, which holds even
  when the ranking is descending.
- `pct` divides by the number of rows ranked in the group, missing rows
  included under `top` and `bottom`, or by the number of distinct values for
  `dense`.
- An ordered category ranks by its place in the categories and an unordered
  one by its value, and a group by refuses to rank an unordered one.
- The answer is float64 for every input type, and a row whose group key is
  missing under `dropna` answers NaN.
"""

from std.math import isnan, nan

from firepanda.array.any import AnyArray
from firepanda.array.array import Array
from firepanda.dtype.lists import FLOAT
from firepanda.dtype.logical import TypeKind
from firepanda.hash.grouping import factorize_any
from firepanda.kernel.dictionary import decode_dictionary, dictionary_codes
from firepanda.kernel.sort import argsort_any

comptime RANK_AVERAGE = 0
"""The mean of the ranks a tie spans."""
comptime RANK_MIN = 1
"""The lowest rank a tie spans."""
comptime RANK_MAX = 2
"""The highest rank a tie spans."""
comptime RANK_FIRST = 3
"""Ties broken by the order the rows appear in."""
comptime RANK_DENSE = 4
"""The position of the value among the distinct values, with no gaps."""

comptime NA_KEEP = 0
"""A missing value ranks as NaN."""
comptime NA_TOP = 1
"""Missing values rank before every value."""
comptime NA_BOTTOM = 2
"""Missing values rank after every value."""


def rank_method(name: String) raises -> Int:
    """Reads a method by the name pandas gives it.

    Args:
        name: One of `average`, `min`, `max`, `first` or `dense`.

    Returns:
        The method's code.

    Raises:
        Error: For any other name.
    """
    if name == "average":
        return RANK_AVERAGE
    if name == "min":
        return RANK_MIN
    if name == "max":
        return RANK_MAX
    if name == "first":
        return RANK_FIRST
    if name == "dense":
        return RANK_DENSE
    raise Error("rank: unknown method '" + name + "'")


def rank_missing(name: String) raises -> Int:
    """Reads a placement for the missing values by the name pandas gives it.

    Args:
        name: One of `keep`, `top` or `bottom`.

    Returns:
        The placement's code.

    Raises:
        Error: For any other name.
    """
    if name == "keep":
        return NA_KEEP
    if name == "top":
        return NA_TOP
    if name == "bottom":
        return NA_BOTTOM
    raise Error("rank: unknown na_option '" + name + "'")


def _missing_rows(col: AnyArray) raises -> List[Bool]:
    """Marks every row that is a null or, in a float column, a NaN.

    Args:
        col: The column.

    Returns:
        One flag a row.

    Raises:
        Error: Only what reading a float column raises.
    """
    var n = len(col)
    var out = List[Bool](capacity=n)
    for i in range(n):
        out.append(not col.is_valid(i))
    if col.type.kind == TypeKind.FLOAT_KIND:
        comptime for target in FLOAT:
            if col.type.physical == target:
                ref view = col.as_typed_view[target]()
                var values = view.unsafe_ptr()
                for i in range(n):
                    if isnan(values.unsafe_offset(i).unsafe_load()):
                        out[i] = True
    return out^


def rank_any(
    col: AnyArray,
    groups: List[Int],
    group_count: Int,
    method: Int,
    ascending: Bool,
    na: Int,
    pct: Bool,
) raises -> AnyArray:
    """Ranks every row of a column within its group.

    Args:
        col: The column to rank.
        groups: The group number of every row, in `[0, group_count)`, or -1
            for a row in no group, which answers NaN. Empty means one group
            holding every row.
        group_count: How many group numbers there are.
        method: One of the `RANK_` codes.
        ascending: Rank the smallest value first.
        na: One of the `NA_` codes.
        pct: Answer the rank as a fraction of the group.

    Returns:
        A float64 column as tall as `col`.

    Raises:
        Error: If the column's type cannot be sorted or factorized.
    """
    if col.is_dictionary():
        # An ordered category ranks by its place in the categories, which is
        # its code, and an unordered one by its value, which is what pandas
        # does. A group by refuses the unordered one, as pandas' does.
        var flat: AnyArray
        if col.type.ordered:
            flat = AnyArray(dictionary_codes(col))
        elif len(groups) > 0:
            raise Error("Cannot perform rank with non-ordered Categorical")
        else:
            flat = AnyArray(decode_dictionary(col))
        return rank_any(flat, groups, group_count, method, ascending, na, pct)
    var n = len(col)
    var whole = len(groups) == 0
    var count = 1 if whole else group_count
    var missing = _missing_rows(col)
    var codes = Array[DType.uint32](0)
    var firsts = List[Int]()
    factorize_any(col).into_parts(codes, firsts)
    var order = argsort_any(col, not ascending, False)
    var sorted_rows = order.unsafe_ptr()
    var equal = codes.unsafe_ptr()

    # Every group's segment is its values in sorted order, with its missing
    # rows before them or after them or nowhere, in the order they appear.
    var values = List[Int](length=count, fill=0)
    var gaps = List[Int](length=count, fill=0)
    for i in range(n):
        var g = 0 if whole else groups[i]
        if g < 0:
            continue
        if missing[i]:
            gaps[g] += 1
        else:
            values[g] += 1
    var starts = List[Int](capacity=count + 1)
    var total = 0
    for g in range(count):
        starts.append(total)
        total += values[g] + (0 if na == NA_KEEP else gaps[g])
    starts.append(total)
    var next_value = List[Int](capacity=count)
    var next_gap = List[Int](capacity=count)
    for g in range(count):
        var before = gaps[g] if na == NA_TOP else 0
        next_value.append(starts[g] + before)
        next_gap.append(starts[g] if na == NA_TOP else starts[g] + values[g])
    var laid = List[Int](length=total, fill=0)
    for p in range(n):
        var i = Int(sorted_rows.unsafe_offset(p).unsafe_load())
        var g = 0 if whole else groups[i]
        if g < 0 or missing[i]:
            continue
        laid[next_value[g]] = i
        next_value[g] += 1
    if na != NA_KEEP:
        for i in range(n):
            var g = 0 if whole else groups[i]
            if g < 0 or not missing[i]:
                continue
            laid[next_gap[g]] = i
            next_gap[g] += 1

    var out = Array[DType.float64](overwritten=n)
    var answer = out.unsafe_mut_ptr()
    for i in range(n):
        answer.unsafe_offset(i).unsafe_store(nan[DType.float64]())

    for g in range(count):
        var first = starts[g]
        var last = starts[g + 1]
        var dense = 0
        var p = first
        while p < last:
            var row = laid[p]
            var q = p + 1
            while q < last:
                var other = laid[q]
                if missing[row] != missing[other]:
                    break
                if not missing[row] and (
                    equal.unsafe_offset(row).unsafe_load()
                    != equal.unsafe_offset(other).unsafe_load()
                ):
                    break
                q += 1
            dense += 1
            for t in range(p, q):
                var rank: Float64
                if method == RANK_AVERAGE:
                    rank = Float64(p - first) + Float64(q - p + 1) / 2.0
                elif method == RANK_MIN:
                    rank = Float64(p - first + 1)
                elif method == RANK_MAX:
                    rank = Float64(q - first)
                elif method == RANK_FIRST:
                    rank = Float64(t - first + 1)
                else:
                    rank = Float64(dense)
                answer.unsafe_offset(laid[t]).unsafe_store(rank)
            p = q
        if pct and last > first:
            var size = Float64(dense if method == RANK_DENSE else last - first)
            for t in range(first, last):
                var at = answer.unsafe_offset(laid[t])
                at.unsafe_store(at.unsafe_load() / size)
    return AnyArray(out^)
