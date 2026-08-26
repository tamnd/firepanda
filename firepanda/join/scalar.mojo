"""The obviously correct join, and the one the fast path has to agree with.

Same terms as `firepanda/kernel/scalar.mojo` and `firepanda/hash/scalar.mojo`.
`join_nested` compares every left row against every right row and emits the pairs
in the order it finds them. It is quadratic, it is never called in production,
and it is here because there is nothing in it to be wrong about: no codes, no
buckets, no concatenation, no densifying.

That last part is what makes it worth having. `join_indices` gets its answer by
routing both frames through `group_ordinals`, which is a lot of machinery to put
between a caller and the question "are these two rows the same". A twin that
asks the question directly is the only way to be sure the machinery is not
quietly changing the answer.

Two rows are the same when every key pair has the same `key_bits`, which is the
same predicate the hash path keys on. Comparing values with `==` instead would
make the twin disagree with the fast path on NaN, and it would be the twin that
was wrong: a column of NaNs joining to nothing is not what pandas does and not
what anyone wants.

A null key matches nothing, checked before the bits are looked at, because a null
holds a zero and its bits are the bits of a zero.
"""

from firepanda.array.any import AnyArray
from firepanda.bitmap.bitmap import Bitmap
from firepanda.dtype.lists import ALL
from firepanda.hash.function import key_bits

from .pairs import JoinIndices, JoinKind


def join_nested(
    left_columns: List[AnyArray],
    left_keys: List[Int],
    left_rows: Int,
    right_columns: List[AnyArray],
    right_keys: List[Int],
    right_rows: Int,
    kind: JoinKind,
) raises -> JoinIndices:
    """Pairs the rows of two frames by comparing all of them against all of them.

    Args:
        left_columns: The left frame's columns.
        left_keys: Which of them are keys.
        left_rows: The left frame's height.
        right_columns: The right frame's columns.
        right_keys: Which of them are keys, matched positionally.
        right_rows: The right frame's height.
        kind: Which rows to keep.

    Returns:
        The same pairing `join_indices` produces, in the same order.

    Raises:
        On the same inputs `join_indices` refuses.
    """
    if kind == JoinKind.RIGHT:
        return join_nested(
            right_columns,
            right_keys,
            right_rows,
            left_columns,
            left_keys,
            left_rows,
            JoinKind.LEFT,
        ).swapped()

    var out_left = List[Int]()
    var out_right = List[Int]()
    var matched = Bitmap(right_rows, all_valid=False)

    for i in range(left_rows):
        var hits = 0
        for r in range(right_rows):
            var same = True
            if kind != JoinKind.CROSS:
                for k in range(len(left_keys)):
                    if not _same_key(
                        left_columns[left_keys[k]],
                        i,
                        right_columns[right_keys[k]],
                        r,
                    ):
                        same = False
                        break
            if not same:
                continue

            hits += 1
            matched.set(r, True)
            if kind == JoinKind.ANTI:
                break
            if kind == JoinKind.SEMI:
                out_left.append(i)
                out_right.append(-1)
                break
            out_left.append(i)
            out_right.append(r)

        if hits == 0 and kind.keeps_unmatched_left():
            out_left.append(i)
            out_right.append(-1)

    if kind == JoinKind.OUTER:
        for r in range(right_rows):
            if not matched.get(r):
                out_left.append(-1)
                out_right.append(r)

    return JoinIndices(out_left^, out_right^)


def _same_key(a: AnyArray, i: Int, b: AnyArray, r: Int) raises -> Bool:
    """Reports whether two rows hold the same key value.

    Args:
        a: The left column.
        i: The left row.
        b: The right column.
        r: The right row.

    Returns:
        False if either side is null, whatever the other side holds.

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
    if not a.is_valid(i) or not b.is_valid(r):
        return False
    comptime for candidate in ALL:
        if a.dtype() == candidate:
            return key_bits(
                a.unsafe_ptr[candidate]().unsafe_offset(i).unsafe_load()
            ) == key_bits(
                b.unsafe_ptr[candidate]().unsafe_offset(r).unsafe_load()
            )
    raise Error("join: unsupported key dtype")
