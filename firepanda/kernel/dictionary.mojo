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
about this kernel. Everything in here reads codes through `dictionary_codes`,
which widens, so a column that arrived over Arrow at int8 is read like any other.

The rest of the file is what a caller does to a categorical after it exists.
Pandas puts eleven names on `Series.cat` and there are three operations under
them. A rename is decided by position and leaves every code where it is. Setting
the categories is decided by value, and a row whose value is not in the new list
becomes null. Dropping the unused ones is a question about the codes rather than
about a list the caller passed, which is why it is a third door and not a call to
the second one. The eight remaining pandas names are arithmetic over these three,
and that arithmetic is in the Python layer where the pandas surface lives.
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
    var codes = dictionary_codes(col)
    var out = StringBuilder(capacity=len(col))
    for i in range(len(col)):
        if not col.is_valid(i):
            out.append_null()
            continue
        out.append(categories.unsafe_bytes(Int(codes[i])))
    return out^.finish()


def dictionary_codes(col: AnyArray) raises -> Array[DType.int32]:
    """Reads a dictionary column's codes as int32, at whatever width they sit.

    The encoder above writes int32 and nothing else in firepanda builds a
    dictionary column, but a column that arrived over Arrow carries whatever
    width its producer chose, and pandas chooses int8 whenever the cardinality
    lets it. A pandas categorical handed to firepanda over the C data interface
    therefore has int8 codes, which is the common case rather than an exotic one.

    So every reader below comes through here and widens once, instead of the
    whole file being parametrised on the index type for a difference that stops
    mattering the moment the codes have been read. The cost is a pass and a
    buffer on the columns that are not already int32.

    Args:
        col: The dictionary column.

    Returns:
        The codes as int32, null where the column is null.

    Raises:
        Error: If the column is not a dictionary column, or if it carries an
            index type that is not an integer one.
    """
    if not col.is_dictionary():
        raise Error(
            String("category: ", col.type, " is not a dictionary column")
        )
    var index = col.type.physical
    if index == DType.int32:
        return col.codes[DType.int32]()
    if index == DType.int8:
        return _widened[DType.int8](col)
    if index == DType.int16:
        return _widened[DType.int16](col)
    if index == DType.int64:
        return _widened[DType.int64](col)
    if index == DType.uint8:
        return _widened[DType.uint8](col)
    if index == DType.uint16:
        return _widened[DType.uint16](col)
    if index == DType.uint32:
        return _widened[DType.uint32](col)
    if index == DType.uint64:
        return _widened[DType.uint64](col)
    raise Error(
        String(
            "category: an index of ",
            index,
            " is not an integer, and a code is a position",
        )
    )


def _widened[dt: DType](col: AnyArray) raises -> Array[DType.int32]:
    """Copies codes of one width into int32.

    Args:
        col: The dictionary column.

    Parameters:
        dt: The width the codes are held at.

    Returns:
        The same codes as int32.

    Raises:
        Error: If the column's index type is not `dt`.
    """
    var codes = col.codes[dt]()
    var rows = len(codes)
    var out = Array[DType.int32](rows)
    for i in range(rows):
        if not codes.is_valid(i):
            out.set_null(i)
            continue
        out.set_valid(i, Int32(Int(codes[i])))
    return out^


def set_ordered(col: AnyArray, ordered: Bool) raises -> AnyArray:
    """Says whether the categories have a meaning to their order.

    This is a change of type and not of data, so it copies the column and
    rewrites one flag on it. It is a door of its own rather than a call to
    `set_categories` with the categories the column already has, because that
    would walk every row to arrive at the codes it started with.

    Args:
        col: The dictionary column.
        ordered: The flag to carry.

    Returns:
        The same column under a type that says so.

    Raises:
        Error: If the column is not a dictionary column.
    """
    if not col.is_dictionary():
        raise Error(
            String("category: ", col.type, " is not a dictionary column")
        )
    var out = col.copy()
    out.type = LogicalType.dictionary(out.type.physical, ordered)
    return out^


def rename_categories(
    col: AnyArray, var names: StringArray, ordered: Bool
) raises -> AnyArray:
    """Gives the categories new labels, leaving every code where it is.

    The one operation on a categorical that is decided by position. Row 4 held
    category 2 before and holds category 2 after, and what changed is what
    category 2 is called. Everything else in this file is decided by value, and
    the difference is the whole reason there are two doors rather than one: a
    rename that went through the value route would look up each old label in the
    new list and find nothing, and would null the column.

    The count is not checked here, because the two pandas methods that reach this
    disagree about it. `rename_categories` insists on one label per category and
    checks that itself. `set_categories(rename=True)` does not: a shorter list
    nulls the rows whose category has fallen off the end, and a longer one leaves
    the extra labels sitting there as categories nothing uses. Both of those are
    written out below, so the check belongs to the caller that wants it.

    Args:
        col: The dictionary column.
        names: The new labels, in the order the column holds its categories.
        ordered: Whether the order is to mean anything.

    Returns:
        The same codes under the new labels.

    Raises:
        Error: If the column is not a dictionary column, or if the new labels
            are not distinct and present.
    """
    if not col.is_dictionary():
        raise Error(
            String("category: ", col.type, " is not a dictionary column")
        )
    _check_distinct(names)
    var have = len(col.categories())
    var wanted = len(names)
    if wanted >= have:
        var out = col.copy()
        out.type = LogicalType.dictionary(out.type.physical, ordered)
        out.dict_values = names^
        return out^

    var codes = dictionary_codes(col)
    var rows = len(col)
    var out = Array[DType.int32](rows)
    for i in range(rows):
        if not col.is_valid(i) or Int(codes[i]) >= wanted:
            out.set_null(i)
            continue
        out.set_valid(i, codes[i])
    return AnyArray.dictionary[DType.int32](out^, names^, ordered)


def set_categories(
    col: AnyArray, var names: StringArray, ordered: Bool
) raises -> AnyArray:
    """Rewrites the column against a new list of categories, matched by value.

    The other door, and the one the other five pandas methods are arithmetic
    over. Every row keeps the value it had if that value is in the new list, and
    becomes null if it is not, so dropping a category is how a categorical loses
    rows to missing and pandas does the same thing. Adding one that nothing uses
    is legal and leaves the codes alone except for the shuffling any reordering
    forces.

    Args:
        col: The dictionary column.
        names: The categories to hold, in the order to hold them.
        ordered: Whether that order is to mean anything.

    Returns:
        A dictionary column over int32 codes and the given categories.

    Raises:
        Error: If the column is not a dictionary column, or if the given
            categories are not distinct and present.
    """
    if not col.is_dictionary():
        raise Error(
            String("category: ", col.type, " is not a dictionary column")
        )
    var moved = _moved_to(col.categories(), names)
    var codes = dictionary_codes(col)
    var rows = len(col)
    var out = Array[DType.int32](rows)
    for i in range(rows):
        if not col.is_valid(i):
            out.set_null(i)
            continue
        var to = moved[Int(codes[i])]
        if to < 0:
            out.set_null(i)
            continue
        out.set_valid(i, Int32(to))
    return AnyArray.dictionary[DType.int32](out^, names^, ordered)


def drop_unused_categories(col: AnyArray) raises -> AnyArray:
    """Drops the categories nothing in the column uses, keeping the order.

    A third door rather than arithmetic over the other two, because which
    categories are used is a question about the codes and the other two doors
    take the answer as an argument. Asking the caller to work it out would mean
    reading every code out into Python to build the list to hand back.

    Args:
        col: The dictionary column.

    Returns:
        The same values over the categories that appear in them.

    Raises:
        Error: If the column is not a dictionary column.
    """
    if not col.is_dictionary():
        raise Error(
            String("category: ", col.type, " is not a dictionary column")
        )
    ref categories = col.categories()
    var have = len(categories)
    var codes = dictionary_codes(col)
    var rows = len(col)

    var used = List[Bool](length=have, fill=False)
    for i in range(rows):
        if col.is_valid(i):
            used[Int(codes[i])] = True

    var moved = List[Int](length=have, fill=-1)
    var kept = List[Int]()
    for at in range(have):
        if used[at]:
            moved[at] = len(kept)
            kept.append(at)

    var out = Array[DType.int32](rows)
    for i in range(rows):
        if not col.is_valid(i):
            out.set_null(i)
            continue
        out.set_valid(i, Int32(moved[Int(codes[i])]))
    var left = categories.take(kept)
    return AnyArray.dictionary[DType.int32](out^, left^, col.type.ordered)


def _check_distinct(names: StringArray) raises:
    """Refuses a category list with a repeat or a hole in it.

    Args:
        names: The categories to check.

    Raises:
        Error: If a label is missing or appears twice.
    """
    var found = factorize_strings(names)
    if found.null_group >= 0:
        raise Error(
            "category: a category cannot be missing, because a missing value is"
            " the absence of a category rather than one of them"
        )
    if len(found.firsts) != len(names):
        raise Error("category: the categories must be distinct")


def _moved_to(categories: StringArray, names: StringArray) raises -> List[Int]:
    """Says where each existing category lands in a new list, or that it does not.

    The lookup is one factorize over the two lists laid end to end rather than a
    hash table built by hand. Two labels that are equal land in the same group
    whichever list they came from, so the group ordinal is the join key, and the
    machinery that does it is already written and already parallel.

    Args:
        categories: The categories the column has, in its order.
        names: The categories it is to have, in the caller's order.

    Returns:
        One entry per existing category, holding its position in `names`, and
        -1 for a category that is not in `names` at all.

    Raises:
        Error: If the new categories are not distinct and present.
    """
    var wanted = len(names)
    var have = len(categories)
    var both = StringBuilder(capacity=wanted + have)
    for j in range(wanted):
        if not names.is_valid(j):
            raise Error(
                "category: a category cannot be missing, because a missing"
                " value is the absence of a category rather than one of them"
            )
        both.append(names.unsafe_bytes(j))
    for at in range(have):
        if not categories.is_valid(at):
            both.append_null()
            continue
        both.append(categories.unsafe_bytes(at))
    var joined = both^.finish()
    var found = factorize_strings(joined)

    # A null in the joined column takes ordinal zero and pushes every real group
    # up by one. Only an existing category can be null, since the new ones were
    # refused above, and those rows are skipped, so this is the amount to take
    # off an ordinal that belongs to a label.
    var shift = 1 if found.null_group >= 0 else 0

    var slot = List[Int](length=len(found.firsts), fill=-1)
    for j in range(wanted):
        var group = Int(found.codes[j]) - shift
        if slot[group] >= 0:
            raise Error("category: the categories must be distinct")
        slot[group] = j

    var out = List[Int](length=have, fill=-1)
    for at in range(have):
        if not categories.is_valid(at):
            continue
        out[at] = slot[Int(found.codes[wanted + at]) - shift]
    return out^
