"""A key column shaped like a web analytics dump rather than like a generator.

Everything this repository generates is uniform. `Rng.next_below` draws every
value with the same probability, every benchmark column is either a cycle or a
uniform draw, and every number we have tuned the hash table against came from one
of those. Real group by keys are not like that, and the query that says so is
ClickBench q31, which groups a hundred million rows by `ClientIP`.

Two things are wrong with a uniform key and they are independent.

**The frequencies.** A handful of addresses account for a large share of the rows
and most addresses are seen once. A table meeting that has a few slots read
millions of times and millions of slots read once, which is a completely
different cache story from every slot read the same number of times, and it is
not visible in a mean.

**The values.** An address is not a random 32 bit number. It sits inside a
network, so the high bits take a few dozen values across the whole column and the
low bits carry almost all of the entropy. A hash that leans on the high bits of
its input, or a table that indexes with the wrong end of the hash, meets a key
set where a third of the bits are nearly constant. Uniform keys never ask that
question because uniform keys have entropy everywhere.

So the generator here draws a host inside one of a few networks, and draws the
host from a distribution with a head. `Skewed` is the draw and `skewed_int64` is
the column. Both are deterministic from the seed, on the same terms as `Rng`: a
measurement that cannot be replayed is not a measurement.

What this is not is a copy of the real column. The real one is not redistributable
and its exact shape is not the point; what is wanted is a key set that has a head
and has structure, so that a number measured on it can be compared against the
same number measured on a uniform column of the same height and the difference
attributed to one of those two things.
"""

from std.math import exp, log

from firepanda.array.array import Array

from .rng import Rng

comptime HOST_BITS = 24
"""Bits of an address that vary inside one network.

Eight bits of network and twenty four of host, which is an IPv4 address split the
way the largest blocks are split. The split matters more than the widths do: what
is being reproduced is that the top of the key is drawn from a small set and the
bottom is not, and any split with that property asks the hash the same question.
"""

comptime HOST_MASK = (1 << HOST_BITS) - 1
"""The host part of an address."""

comptime DEFAULT_NETWORKS = 8
"""Networks the addresses are drawn from.

Small enough that the high byte of every key in the column takes one of eight
values, which is the structure this is here to produce. A real dump has more than
eight and the ones past the first few carry very little of the traffic, so eight
is the part of it that changes a measurement.
"""

comptime DEFAULT_HEADS = 1024
"""Frequently seen hosts per network, by default.

A thousand addresses per network with half the rows between them is a head heavy
enough to show up in a cache miss count and light enough that it is not the whole
column.
"""

comptime DEFAULT_SHARE = 0.5
"""Share of the rows the head takes, by default.

Half is a deliberate middle. At one the column has no tail and is a low
cardinality group by; at zero every row is its own group and the column is the
nearly unique case. Both ends are worth measuring and neither is the default.
"""


struct Skewed(Copyable, Movable):
    """Draws addresses with a heavy head out of a few networks."""

    var rng: Rng
    """The stream. Print the seed, replay the column."""

    var nets: List[UInt64]
    """The network numbers, already shifted into the top of the key."""

    var heads: Int
    """Hosts per network that are drawn more than once."""

    var share: Float64
    """Probability that a row is drawn from the head rather than from the tail."""

    var issued: Int
    """Tail hosts handed out so far, which is what makes them distinct."""

    def __init__(
        out self,
        heads: Int = DEFAULT_HEADS,
        share: Float64 = DEFAULT_SHARE,
        networks: Int = DEFAULT_NETWORKS,
        seed: UInt64 = 0xC11E_47A5,
    ):
        """Constructs a generator.

        Args:
            heads: Hosts per network that repeat. Zero gives a column with no
                head at all, where every row is its own group.
            share: Fraction of rows drawn from the head, in `[0, 1]`.
            networks: How many networks the addresses come from. At least one.
            seed: The starting state, printed by anything that uses this.
        """
        self.rng = Rng(seed)
        self.heads = heads
        self.share = share
        self.issued = 0

        # Drawn rather than taken in order, so the high byte is not 0 through 7
        # and a hash that happens to be fine on small numbers is not flattered.
        # Distinct by rejection, because two of the eight landing on the same
        # network would quietly make it seven and nothing downstream would say so.
        self.nets = List[UInt64](capacity=networks)
        while len(self.nets) < networks:
            var candidate = (self.rng.next_u64() & 0xFF) << HOST_BITS
            var seen = False
            for i in range(len(self.nets)):
                if self.nets[i] == candidate:
                    seen = True
            if not seen:
                self.nets.append(candidate)

    def next_key(mut self) -> Int64:
        """Draws one address.

        The head is drawn Zipf with an exponent of one, by inverse transform:
        `heads` raised to a uniform power is a rank whose density falls off as
        one over the rank, which is the law a frequency ranked list of anything
        people do tends to follow. It costs a log and an exp rather than a table
        of `heads` cumulative weights, which matters because this is called once
        per row and the row counts here are in the hundred millions.

        The tail is a counter, so a tail host is new until the counter has gone
        round the host space. After that it repeats, which is not a flaw to work
        around: a real address space is finite too, and a hundred million rows of
        a thirty two bit address cannot all be distinct.

        Returns:
            An address in one of the networks, as the int64 a key column holds.
        """
        var net = self.nets[self.rng.next_below(len(self.nets))]
        if self.heads > 0 and self.rng.next_float64() < self.share:
            var rank = Int(
                exp(self.rng.next_float64() * log(Float64(self.heads)))
            )
            if rank > self.heads:
                rank = self.heads
            return Int64(net | UInt64(rank - 1))
        var host = UInt64(self.heads + self.issued) & HOST_MASK
        self.issued += 1
        return Int64(net | host)


def skewed_int64(
    rows: Int,
    heads: Int = DEFAULT_HEADS,
    share: Float64 = DEFAULT_SHARE,
    networks: Int = DEFAULT_NETWORKS,
    seed: UInt64 = 0xC11E_47A5,
) -> Array[DType.int64]:
    """Builds a column of addresses.

    Args:
        rows: The column's height.
        heads: Hosts per network that repeat.
        share: Fraction of rows drawn from the head.
        networks: How many networks to draw from.
        seed: The generator seed.

    Returns:
        The column, with no nulls in it. Nulls are a separate axis and a caller
        that wants them can set them.
    """
    var out = Array[DType.int64](rows)
    var gen = Skewed(heads, share, networks, seed)
    for i in range(rows):
        out[i] = gen.next_key()
    return out^
