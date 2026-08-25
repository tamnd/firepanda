"""The kernel layer: one generic implementation per operation, plus its twin.

Two rules hold across every file in this package, and both are load bearing.

**Every kernel has a scalar twin in `scalar.mojo`.** The twin is the slowest
correct implementation anybody could write: a loop, one element at a time, no
SIMD, no null fast path, no cleverness of any kind. It is never called in
production. Its only job is to be so obviously right that when it and the real
kernel disagree, the real kernel is wrong. `tests/fuzz` runs the two against each
other on random data forever. docs/specs/09-quality-bar.md section 2.

**A null value is zero in the values buffer.** `Array.set_null` zeroes the slot
it invalidates, `Array.slice` copies the zeros along with everything else, and a
freshly allocated column starts zeroed. That invariant is what lets `sum_of` add
straight through a column without ever looking at the validity bitmap, which on a
million rows is the difference between 99 us and 471 us.

The invariant has one sharp edge and it is worth knowing about: `Array.__setitem__`
writes a value without touching validity, so writing a non-zero value into a
position that is currently null breaks it. Use `set_valid`, which writes both.
`tests/test_kernel.mojo` has a test that constructs exactly that situation and
asserts what `sum_of` does with it, so the behaviour is at least pinned down
rather than folklore.

`scalar.mojo` is deliberately not re-exported here. Importing it should take the
extra line, because production code has no business reaching for it and the
import is where that shows up in review.
"""

from .accum import accumulator, highest, lowest
from .agg import AggResult, count_of, max_of, mean_of, min_of, sum_of
from .arith import add, divide, multiply, subtract
from .cast import cast_to
from .compare import equal, greater, greater_equal, less, less_equal, not_equal
from .mask import apply_validity, combined_validity
from .select import filter_rows, take_rows
from .sort import (
    argsort,
    argsort_into,
    argsort_multi,
    is_sorted,
    sort_key,
    sort_rows,
)
