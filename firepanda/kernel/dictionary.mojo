"""Turning a column of values into codes and a list of categories.

This is what `astype("category")` is, and it is a conversion rather than a cast.
Everything else in `cast.mojo` reads a value and writes the same value in another
layout, row by row, and a row can be converted without looking at any other row.
Dictionary encoding cannot: what code row 4 gets depends on every row before it,
because the code is a position in a list that is being built while the column is
being read.

So it lives here rather than there, and `cast_any` calls into it.

The categories come out sorted, because that is what pandas does and because the
ordering is not an implementation detail a caller cannot see. `Series.cat.
categories` prints them, a groupby over a category column produces its groups in
category order, and an ordered categorical compares by position, so a library
that encoded in first appearance order would give a different answer to `min()`
on the same data depending on how the rows happened to be arranged.

Nulls do not get a category. Arrow says a dictionary column carries its missing
values in the codes buffer's validity, exactly as an integer column does, and
pandas agrees: a `NaN` in a categorical is not a category, does not appear in
`.cat.categories`, and is not counted by `value_counts` unless asked for. So a
null row comes out a null code and the categories are the distinct values that
were actually there.

The index width is int32. Arrow permits all eight integer widths and pandas
writes int8 where it can, which saves three bytes a row on a column that is
usually being encoded to save memory in the first place. That is a real
difference and it is deliberate for now: the width is visible through the Arrow
export, so narrowing it later changes what a consumer sees, and picking the width
from the cardinality means the same column encodes to different types on
different data, which is a conversation about the dtype vocabulary rather than
about this kernel.
"""

from firepanda.array.any import AnyArray
from firepanda.array.array import Array
from firepanda.array.strings import StringArray, StringBuilder
from firepanda.dtype.logical import LogicalType
from firepanda.hash.factorize import factorize_strings
from firepanda.kernel.sort import argsort_any


def encode_dictionary(
    col: StringArray, ordered: Bool = False
) raises -> AnyArray:
    """Rewrites a text column as codes into a sorted list of its distinct values.

    Args:
        col: The column to encode.
        ordered: Whether the categories are to have a meaning to their order.

    Returns:
        A dictionary column over int32 codes, null where the input was null.

    Raises:
        Error: If the distinct values cannot be sorted, which cannot happen for
            text and is the sort kernel's promise rather than this one's.
    """
    var rows = len(col)
    var found = factorize_strings(col)
    # `firsts` is one representative row per non-null group, in the order the
    # groups were first seen, so taking them gives the distinct values without
    # a second pass over the column.
    var distinct = col.take(found.firsts)
    var order = argsort_any(AnyArray(StringArray(copy=distinct)))

    # `order` says which distinct value belongs at each sorted position, and the
    # codes need the other direction: given the ordinal factorize handed a row,
    # where does that value sit once sorted.
    var groups = len(found.firsts)
    var rank = List[Int32](length=groups, fill=Int32(0))
    for at in range(groups):
        rank[Int(order[at])] = Int32(at)

    # Nulls took ordinal zero and pushed every real group up by one, or there
    # were none and the ordinals start at zero. Either way this is the amount to
    # take off an ordinal to index `rank`.
    var shift = 1 if found.null_group >= 0 else 0

    var codes = Array[DType.int32](rows)
    for i in range(rows):
        if not col.is_valid(i):
            codes.set_null(i)
            continue
        codes.set_valid(i, rank[Int(found.codes[i]) - shift])

    var categories = distinct.take(_positions(order))
    return AnyArray.dictionary[DType.int32](codes^, categories^, ordered)


def _positions(order: Array[DType.uint32]) -> List[Int]:
    """Turns a permutation column into the list of rows `take` wants.

    Args:
        order: The permutation.

    Returns:
        The same numbers as a list of `Int`.
    """
    var out = List[Int](capacity=len(order))
    for i in range(len(order)):
        out.append(Int(order[i]))
    return out^


def decode_dictionary(col: AnyArray) raises -> StringArray:
    """Reads a dictionary column back as the text it stands for.

    The other direction, which is what `astype(str)` on a category column has to
    do and what anything that wants the values rather than the codes needs. It
    materialises: a column of ten million rows over four categories becomes ten
    million strings, which is the cost of asking for the values and is the whole
    reason the encoding exists.

    Args:
        col: The dictionary column.

    Returns:
        The values, null where the codes were null.

    Raises:
        Error: If the column is not a dictionary column.
    """
    if not col.is_dictionary():
        raise Error(String("cast: ", col.type, " is not a dictionary column"))
    ref categories = col.categories()
    var codes = col.codes[DType.int32]()
    var out = StringBuilder(capacity=len(col))
    for i in range(len(col)):
        if not col.is_valid(i):
            out.append_null()
            continue
        out.append(categories.unsafe_bytes(Int(codes[i])))
    return out^.finish()
