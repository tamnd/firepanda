"""The obviously correct factorize, and the one we are trying to beat.

Two functions that both do what `factorize` does, for two different reasons.

`factorize_linear` is the twin, on the same terms as `firepanda/kernel/scalar.mojo`.
It keeps the distinct keys in a list and scans the whole list for every row. That
is quadratic and it is never called in production. It is here because it is too
simple to be wrong, and when it and `factorize` disagree, `factorize` is wrong.

`factorize_dict` is not a twin. It is the alternative implementation the M1 exit
criteria ask about: if our own hash table is not meaningfully better than the
language's `Dict`, then writing one was a mistake and it is much cheaper to find
that out now than at M8. It is kept in the library rather than in the benchmark
file so that the comparison stays runnable and stays honest as both sides change.

Both key on the same `key_bits` the fast path keys on, so the comparison is
between the maps and not between two different ideas of what a key is. In
particular `factorize_dict` would otherwise put every NaN in a group of its own,
which would make it look bad for a reason that has nothing to do with `Dict`.
"""

from firepanda.array.array import Array

from .function import key_bits


def factorize_linear[dt: DType](col: Array[dt]) -> Array[DType.uint32]:
    """Assigns group ordinals by scanning the keys found so far.

    Returns only the codes. The keys are recoverable from them and the column,
    and a twin that returns less is a twin with less to be wrong about.

    Args:
        col: The column.

    Parameters:
        dt: The column's dtype.

    Returns:
        One ordinal per row, matching `factorize(col).codes`.
    """
    var n = len(col)
    var codes = Array[DType.uint32](n)
    var seen = List[UInt64]()

    var has_null = col.null_count() > 0
    var offset = 1 if has_null else 0

    for i in range(n):
        if has_null and not col.is_valid(i):
            codes[i] = UInt32(0)
            continue

        var bits = key_bits(col[i])
        var at = -1
        for j in range(len(seen)):
            if seen[j] == bits:
                at = j
                break
        if at < 0:
            at = len(seen)
            seen.append(bits)
        codes[i] = UInt32(at + offset)

    return codes^


def factorize_dict[dt: DType](col: Array[dt]) -> Array[DType.uint32]:
    """Assigns group ordinals using the language's `Dict`.

    Args:
        col: The column.

    Parameters:
        dt: The column's dtype.

    Returns:
        One ordinal per row, matching `factorize(col).codes`.
    """
    var n = len(col)
    var codes = Array[DType.uint32](n)
    var seen = Dict[UInt64, Int]()

    var has_null = col.null_count() > 0
    var offset = 1 if has_null else 0

    for i in range(n):
        if has_null and not col.is_valid(i):
            codes[i] = UInt32(0)
            continue

        # One lookup rather than a containment test followed by a read. The
        # comparison is only worth anything if the `Dict` side is written the way
        # somebody who wanted it to be fast would write it.
        var bits = key_bits(col[i])
        var found = seen.get(bits)
        if found:
            codes[i] = UInt32(found.value() + offset)
        else:
            var assigned = len(seen)
            seen[bits] = assigned
            codes[i] = UInt32(assigned + offset)

    return codes^
