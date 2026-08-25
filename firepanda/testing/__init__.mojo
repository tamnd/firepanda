"""Helpers the test, fuzz, stress and benchmark harnesses share.

Nothing in here is meant for callers of the library. It ships inside the package
rather than beside the tests because the benchmarks and the differential suite
are separate binaries that all need the same generator, and duplicating it three
times is how the three copies drift apart.
"""

from .rng import Rng
