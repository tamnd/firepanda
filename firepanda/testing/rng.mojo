"""A deterministic random number generator for tests, fuzzing and benchmarks.

The stdlib generator is seeded from the clock and its stream is not something we
control, which is the wrong shape for a fuzzer. A fuzz failure is only useful if
the case that produced it can be replayed, so the generator here is a plain
splitmix64: the whole state is one 64-bit word, the seed is printed by every
harness that uses it, and the same seed replays the same run on every machine and
every platform.

Splitmix64 is not cryptographic and does not need to be. It passes BigCrush, it
has a period of 2^64, and it costs a multiply and three shifts, which matters when
the harness is expected to do ten million operations inside a CI time budget.
"""


struct Rng(Copyable, Movable):
    """A splitmix64 generator."""

    var state: UInt64
    """The entire generator state. Copy it to fork a stream, print it to replay one."""

    def __init__(out self, seed: UInt64 = 0x243F6A8885A308D3):
        """Constructs a generator.

        Args:
            seed: The starting state. The default is the first 64 bits of the
                fractional part of pi, which is a constant with no structure
                rather than a constant someone might accidentally match.
        """
        self.state = seed

    def next_u64(mut self) -> UInt64:
        """Advances the generator and returns the next word.

        Returns:
            A uniformly distributed 64-bit value.
        """
        self.state += 0x9E3779B97F4A7C15
        var z = self.state
        z = (z ^ (z >> UInt64(30))) * 0xBF58476D1CE4E5B9
        z = (z ^ (z >> UInt64(27))) * 0x94D049BB133111EB
        return z ^ (z >> UInt64(31))

    def next_below(mut self, bound: Int) -> Int:
        """Returns a value in `[0, bound)`.

        This uses a remainder, so the low values are very slightly more likely
        than the high ones. For a bound under a few million the bias is on the
        order of 2^-40 and does not affect what the fuzzer covers.

        Args:
            bound: The exclusive upper bound. Must be positive.

        Returns:
            A value in the half-open range.
        """
        return Int(self.next_u64() % UInt64(bound))

    def next_range(mut self, low: Int, high: Int) -> Int:
        """Returns a value in `[low, high)`.

        Args:
            low: The inclusive lower bound.
            high: The exclusive upper bound. Must be greater than `low`.

        Returns:
            A value in the half-open range.
        """
        return low + self.next_below(high - low)

    def next_bool(mut self) -> Bool:
        """Returns a coin flip.

        Returns:
            True half the time.
        """
        return (self.next_u64() >> UInt64(63)) != 0

    def next_float64(mut self) -> Float64:
        """Returns a value in `[0, 1)`.

        Uses the top 53 bits, which is exactly the mantissa width, so every
        representable value in the range is reachable and none is favoured.

        Returns:
            A uniformly distributed double.
        """
        return Float64(self.next_u64() >> UInt64(11)) * (
            1.0 / 9007199254740992.0
        )
