"""One byte string per row that stands for a whole key tuple.

A join on more than one column has to compare tuples, and every table in this
library compares one column. `keys.mojo` has one answer to that already: take
each key's range over both frames at once, lay the keys out in positional
notation and pack the tuple into a single integer. It is the better answer where
it applies, and what it needs is both frames, because the two sides have to pack
the same way and a range over one of them is not a range over the other.

A streaming join does not have both frames. It has the build side whole and the
probe side arriving a chunk at a time, and a packing decided from the build side
alone would have to do something with a probe value the build side never saw.

So this packs into bytes rather than into a number. Each key contributes its own
bytes to one buffer and the buffer is the row's key. A fixed width value
contributes the bytes it is stored as, which is the same count for every row of
that column, and a string contributes its length as four bytes followed by its
bytes. Either way the reader of the concatenation could tell where one key ends
and the next begins, which is the whole of what makes the packing injective: two
rows have the same bytes exactly when they have the same tuple.

Nothing about that needs a range, a minimum or a second frame. The build side
packs, the probe side packs the same way because the layout comes from the
dtypes and not from the values, and the text key route in `keys.mojo` does the
rest: build over the right frame's packed column, probe with each chunk's packed
column as it arrives, compare the bytes on a hash match. A hash is not an exact
answer for a byte string and that route already knew it.

A row with a null in any key packs to a null element. That is not a shortcut: a
null key matches nothing in SQL, so a row with one cannot pair whatever its
other keys say, and the join's own null handling drops a null key already.

What this costs is a pass over the key columns and a byte string per row. A pair
of eight byte keys packs to sixteen bytes, which is four past what fits inside a
`StringView`, so it lands in the payload and the pass is a copy. A pair of four
byte keys fits inside the view and the payload is never touched. An integer
tuple narrow enough to share a uint32 would be better off going through
`_pair_plan`, and a streaming join cannot ask for that, so the case is noted
here rather than handled.
"""

from firepanda.array.any import AnyArray
from firepanda.array.strings import StringBuilder
from firepanda.dtype.lists import dtype_size


def pack_keys(
    columns: List[AnyArray], keys: List[Int], rows: Int
) raises -> AnyArray:
    """Packs a key tuple per row into one byte string per row.

    Args:
        columns: The frame's or the chunk's columns.
        keys: Which of them are keys, in key order. Both sides have to pass the
            same order and the matching dtypes, which is what makes two packed
            columns comparable.
        rows: How many rows to pack.

    Returns:
        A text column of `rows` elements, null wherever a key is null.

    Raises:
        Error: If a key column is neither text nor a dtype with a fixed width.
    """
    var count = len(keys)
    var text = List[Bool](capacity=count)
    var widths = List[Int](capacity=count)
    for k in range(count):
        ref column = columns[keys[k]]
        if column.is_string():
            text.append(True)
            widths.append(0)
            continue
        var width = dtype_size(column.dtype())
        if width == 0:
            raise Error(
                String(
                    (
                        "join: a packed key is the bytes each key is stored as,"
                        " and key "
                    ),
                    k,
                    " is ",
                    column.dtype(),
                    ", which has no fixed width layout",
                )
            )
        text.append(False)
        widths.append(width)

    var built = StringBuilder(capacity=rows)
    # One buffer for every row rather than one per row. It grows to the widest
    # tuple in the column and then stops.
    var scratch = List[UInt8]()
    for i in range(rows):
        var missing = False
        for k in range(count):
            if not columns[keys[k]].is_valid(i):
                missing = True
                break
        if missing:
            built.append_null()
            continue

        scratch.clear()
        for k in range(count):
            ref column = columns[keys[k]]
            if text[k]:
                var bytes = column.strings().unsafe_bytes(i)
                # Little endian by hand rather than by reinterpreting a UInt32,
                # so that two machines that disagree about byte order still pack
                # a tuple the same way. Nothing here reads the length back, but
                # a packed column that travels is a packed column that has to.
                var length = UInt32(len(bytes))
                scratch.append(UInt8(length & 0xFF))
                scratch.append(UInt8((length >> 8) & 0xFF))
                scratch.append(UInt8((length >> 16) & 0xFF))
                scratch.append(UInt8((length >> 24) & 0xFF))
                for b in range(len(bytes)):
                    scratch.append(bytes[b])
            else:
                var width = widths[k]
                var at = column.unsafe_ptr[DType.uint8]()
                for b in range(width):
                    scratch.append(
                        at.unsafe_offset(i * width + b).unsafe_load()
                    )
        built.append(Span(scratch))
    return AnyArray(built^.finish())
