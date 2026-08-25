"""Splitting rows across workers by hash, so their tables never touch.

The parallel build this is for lives in the executor and does not exist yet. The
partitioning does, because the layout it produces is what the rest of the design
assumes: each worker owns a contiguous run of row indices whose keys can only
belong to that worker, so the tables are private, there is no contention, no
atomics, and the merge at the end is a concatenation rather than a reduction.
There is no race detector for this language and that is a large part of why the
design is arranged to have nothing to detect.

Partitions come from the **high** bits of the hash and the table buckets come
from the low bits. Those two have to be different bits. Taking both from the
bottom would mean every key in a partition hashed to the same region of that
worker's table, which turns linear probing into a linked list and is a bug that
looks like a performance problem.

This is a counting sort. Two passes over the hashes, one to count and one to
place, with an exclusive prefix sum in between. It is stable, which matters
because first-appearance ordering of groups is what pandas reports and a stable
partition is what lets a parallel build reproduce it.
"""

from firepanda.buffer.buffer import Buffer


struct Partitioning(Movable):
    """Row indices grouped by hash partition."""

    var order: List[Int]
    """Row indices, grouped by partition and in their original order within one."""

    var offsets: List[Int]
    """Where each partition starts in `order`, with a final entry holding the
    total. Partition `p` is `order[offsets[p]:offsets[p + 1]]`."""

    def __init__(out self, var order: List[Int], var offsets: List[Int]):
        """Constructs a partitioning.

        Args:
            order: The grouped row indices.
            offsets: The partition boundaries.
        """
        self.order = order^
        self.offsets = offsets^

    def count(self) -> Int:
        """Returns the number of partitions.

        Returns:
            The partition count.
        """
        return len(self.offsets) - 1


def radix_partition(hashes: Buffer, n: Int, bits: Int) -> Partitioning:
    """Groups row indices by the top bits of their hash.

    Args:
        hashes: One 64-bit hash per row, as written by `hash_into`.
        n: The number of rows.
        bits: How many high bits to partition on, between 1 and 63. `2 ** bits`
            partitions.

    Returns:
        The row indices grouped by partition, and the partition boundaries.
    """
    # Zero would shift by 64, which is undefined rather than zero, and a caller
    # asking for one partition wants the identity rather than a crash.
    debug_assert(bits >= 1 and bits <= 63, "partition bits out of range ", bits)

    var parts = 1 << bits
    var shift = UInt64(64 - bits)
    var values = hashes.bitcast[DType.uint64]()

    var counts = List[Int]()
    for _ in range(parts + 1):
        counts.append(0)

    for i in range(n):
        var p = Int(values.unsafe_offset(i).unsafe_load() >> shift)
        counts[p + 1] += 1

    for p in range(parts):
        counts[p + 1] += counts[p]

    # `counts` is now the exclusive prefix sum, which is also the write cursor
    # for each partition, so the placing pass advances it in place and the copy
    # taken first is what gets returned as the offsets.
    var offsets = List[Int]()
    for p in range(parts + 1):
        offsets.append(counts[p])

    var order = List[Int]()
    for _ in range(n):
        order.append(0)

    for i in range(n):
        var p = Int(values.unsafe_offset(i).unsafe_load() >> shift)
        order[counts[p]] = i
        counts[p] += 1

    return Partitioning(order^, offsets^)
