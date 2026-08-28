"""Turning a column value into the 64 bits a hash table works with.

Two steps, deliberately separate. `key_bits` decides what two keys being equal
means, and `mix` decides where a key lands. Keeping them apart is what lets a
narrow dtype and a wide one agree on a key, and it is what lets the batch path
hash a whole register of values at a time.

`mix` is the splitmix64 finalizer with the seed folded in. For a fixed seed it is
a bijection on 64 bits, which is worth more than it looks. It introduces no
collisions of its own, so every collision the table sees comes from the mask
rather than from the function. And because it is a bijection, two keys have the
same hash exactly when they have the same bits, which is why the table stores
hashes and nothing else: the key bits would be a second copy of the same
information. It costs two multiplies and three shifts and it vectorizes, which
matters because the batch path hashes a thousand keys before it probes anything.

The seed is per query and not per build. A published seed is a published
collision recipe, and a group by on user supplied keys is exactly the place
somebody would use one.

Text is the exception to all of this and it is worth being explicit about why.
`key_bits` cannot reduce a string to 64 bits without losing information, so
`hash_bytes` is a real hash rather than a relabelling, two different strings can
land on the same 64 bits, and everything the table gets to assume about a fixed
width key stops being true. That is why the string route through the table
compares the bytes on a hash match instead of taking the match as proof. See
`HashTable.build_strings`.
"""

from std.collections.span import Span
from std.math import isnan
from std.sys.info import simd_width_of

from firepanda.array.array import Array
from firepanda.array.strings import StringArray
from firepanda.buffer.buffer import Buffer

comptime DEFAULT_SEED = UInt64(0x243F6A8885A308D3)
"""The seed used when a caller does not supply one.

The first 64 bits of the fractional part of pi, which is the same constant
`firepanda/testing/rng.mojo` starts from, for the same reason: a constant with no
structure rather than a constant somebody might accidentally match.
"""

comptime CANONICAL_NAN = UInt64(0x7FF8000000000000)
"""The bit pattern every NaN is normalized to.

There are 2^52 - 1 distinct float64 NaNs and IEEE says none of them equals any
other, including itself. A group by has to disagree with IEEE here or a column of
NaNs produces a group per row.
"""


def key_bits[
    dt: DType, width: Int = 1
](value: SIMD[dt, width]) -> SIMD[DType.uint64, width]:
    """Reduces values to the bits that decide whether two keys are the same.

    Integers widen. Narrow signed values sign-extend, so `Int8(-1)` and
    `Int64(-1)` produce the same bits, which is what you want the day a join key
    is int32 on one side and int64 on the other.

    Floats go to their float64 bit pattern with two corrections. NaN becomes one
    canonical pattern, so that every NaN in a column is the same key. Zero becomes
    positive zero, so that `-0.0` and `0.0` are the same key, which is what pandas
    does and what anyone summing a price column expects.

    Args:
        value: The values.

    Parameters:
        dt: The dtype.
        width: The register width.

    Returns:
        One 64-bit key per lane.
    """
    comptime if dt.is_floating_point():
        var wide = value.cast[DType.float64]()
        var bits = wide.to_bits[DType.uint64]()
        bits = wide.eq(SIMD[DType.float64, width](0)).select(
            SIMD[DType.uint64, width](0), bits
        )
        return isnan(wide).select(
            SIMD[DType.uint64, width](CANONICAL_NAN), bits
        )
    return value.cast[DType.uint64]()


def mix[
    width: Int = 1
](bits: SIMD[DType.uint64, width], seed: UInt64) -> SIMD[DType.uint64, width]:
    """Spreads key bits across the whole 64-bit range.

    Args:
        bits: The key bits.
        seed: The per-query seed.

    Parameters:
        width: The register width.

    Returns:
        One hash per lane.
    """
    var h = bits ^ SIMD[DType.uint64, width](seed)
    h = (h ^ (h >> SIMD[DType.uint64, width](30))) * SIMD[DType.uint64, width](
        0xBF58476D1CE4E5B9
    )
    h = (h ^ (h >> SIMD[DType.uint64, width](27))) * SIMD[DType.uint64, width](
        0x94D049BB133111EB
    )
    return h ^ (h >> SIMD[DType.uint64, width](31))


def hash_of[
    dt: DType
](value: Scalar[dt], seed: UInt64 = DEFAULT_SEED) -> UInt64:
    """Hashes a single value.

    Args:
        value: The value.
        seed: The per-query seed.

    Parameters:
        dt: The dtype.

    Returns:
        The hash.
    """
    return mix(key_bits(value), seed)


def hash_into[dt: DType](col: Array[dt], seed: UInt64, mut hashes: Buffer):
    """Hashes a whole column into an output buffer.

    Args:
        col: The column.
        seed: The per-query seed.
        hashes: Where to write the hashes. Must be at least `8 * len(col)` bytes.

    Parameters:
        dt: The dtype.
    """
    hash_chunk(col, 0, len(col), seed, hashes)


def hash_chunk[
    dt: DType
](col: Array[dt], start: Int, count: Int, seed: UInt64, mut hashes: Buffer,):
    """Hashes a run of rows into the front of an output buffer.

    The nulls are hashed along with everything else; they hold a zero and the
    caller is going to ignore the answer, and branching on validity here would
    cost more than the wasted lanes.

    The run exists so that the caller can work in pieces small enough to stay in
    cache. Hashing a million rows into an eight megabyte buffer and then reading
    it back is sixteen megabytes of traffic that a few kilobytes at a time does
    not have to move, and on a low cardinality column that traffic was most of
    the operation.

    Args:
        col: The column.
        start: The first row to hash.
        count: How many rows to hash.
        seed: The per-query seed.
        hashes: Where to write the hashes, starting at its own first element.
            Must be at least `8 * count` bytes.

    Parameters:
        dt: The dtype.
    """
    # The register width is chosen for the output rather than the input. A hash
    # is 64 bits wide whatever went in, so stepping by the uint64 width means a
    # narrow input dtype does a partial load and everything downstream stays in
    # whole registers.
    #
    # The tail is a whole register too, past the last row and into the padding
    # rather than masked off. `Buffer` rounds every allocation up to 64 bytes,
    # which is eight of these lanes, so a register that starts inside the buffer
    # ends inside the allocation. Same argument as the kernels; see
    # `firepanda/buffer/buffer.mojo`.
    comptime width = simd_width_of[DType.uint64]()

    var ptr = col.unsafe_ptr()
    var out = hashes.bitcast[DType.uint64]()
    var i = 0
    while i < count:
        var k = key_bits(
            ptr.unsafe_offset(start + i).unsafe_load[width=width]()
        )
        out.unsafe_offset(i).unsafe_store(mix(k, seed))
        i += width


comptime BYTE_WORD = 8
"""Bytes folded into the running hash at a time. One 64-bit word."""

comptime BYTE_SHIFTS = SIMD[DType.uint64, BYTE_WORD](
    0, 8, 16, 24, 32, 40, 48, 56
)
"""Where each byte of a word goes when eight of them are packed into one.

The order is little endian and it does not matter which order it is, only that it
is the same one every time. This is a hash rather than a comparison, and unlike
`StringArray.sort_prefix`, which packs the other way round because there the
integer order has to be the byte order.
"""


def hash_bytes(bytes: Span[UInt8, _], seed: UInt64 = DEFAULT_SEED) -> UInt64:
    """Hashes a run of bytes.

    A word at a time through `mix`, which is two multiplies per eight bytes and
    is where nearly all of the time goes on the short fields a dataframe is
    actually full of. A dedicated wide hash would win on paragraphs and lose on
    names.

    The length is folded in before any bytes are, because without it the tail
    cannot tell "ab" from "ab\\0": both leave the same bits in a word that was
    zero to begin with. Folding it at the start rather than the end also means a
    column of same-length keys, which is most categorical data, starts from a
    value the seed has already spread out.

    The word loop loads eight lanes of one byte rather than one lane of eight,
    because an element's bytes start wherever the payload put them and a 64-bit
    load off an odd address is an unaligned access. The pack afterwards is a
    shift and a reduction and it vectorizes.

    Args:
        bytes: The bytes to hash.
        seed: The per-query seed.

    Returns:
        The hash. Unlike the fixed width path this is a real hash, so equal
        hashes do not mean equal keys and a caller that needs equality has to
        compare the bytes.
    """
    var count = len(bytes)
    var ptr = bytes.unsafe_ptr()
    var h = UInt64(count)

    var i = 0
    while i + BYTE_WORD <= count:
        var chunk = ptr.unsafe_offset(i).unsafe_load[width=BYTE_WORD]()
        h = mix(
            h ^ (chunk.cast[DType.uint64]() << BYTE_SHIFTS).reduce_or(), seed
        )
        i += BYTE_WORD

    var tail = UInt64(0)
    var shift = UInt64(0)
    while i < count:
        tail |= UInt64(ptr.unsafe_offset(i).unsafe_load()) << shift
        shift += 8
        i += 1
    return mix(h ^ tail, seed)


def hash_strings_chunk(
    col: StringArray, start: Int, count: Int, seed: UInt64, mut hashes: Buffer
):
    """Hashes a run of a string column's elements into the front of a buffer.

    The counterpart of `hash_chunk` for text, and it is a loop rather than
    anything clever because every element is a different length and there is no
    register shape that covers them.

    A null hashes as the empty string. The caller ignores the answer for a null
    row the same way it does on the fixed width path, and branching here to skip
    the work would cost more than the work.

    Args:
        col: The column.
        start: The first element to hash.
        count: How many to hash.
        seed: The per-query seed.
        hashes: Where to write them, from its own first element. Must be at least
            `8 * count` bytes.
    """
    var out = hashes.bitcast[DType.uint64]()
    for i in range(count):
        out.unsafe_offset(i).unsafe_store(
            hash_bytes(col.unsafe_bytes(start + i), seed)
        )
