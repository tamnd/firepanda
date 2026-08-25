"""Joins: pairing the rows of two frames.

This is a package of its own rather than a file in `firepanda/kernel` because a
join needs both halves of what is underneath it. It reads key columns through
`firepanda/hash` and it builds its result with `firepanda/kernel`, and `hash`
already imports `kernel`, so a join living in either one would close a cycle.
Sitting above both is the honest arrangement and it says something true about the
operation: a join is the first thing in the library that is not a kernel.

`pairs.mojo` answers only which rows go with which. The frame layer takes that
answer and decides what the columns should be, which is a separate problem with
its own rules about names and coalescing, and it is in `firepanda/frame`.
"""

from .pairs import JoinIndices, JoinKind, join_indices, take_pair
