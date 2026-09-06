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
from .agg import (
    AggResult,
    count_of,
    extreme_over,
    max_of,
    mean_of,
    mean_over,
    min_of,
    sum_of,
    sum_over,
)
from .arith import (
    add,
    arith_const,
    divide,
    divide_const,
    floor_divide,
    floor_divide_const,
    modulo,
    modulo_const,
    multiply,
    power,
    power_const,
    subtract,
)
from .binary import BinaryOp, binary_any, binary_type, binary_value_any
from .cast import cast_any, cast_strings_to, cast_to, cast_to_strings
from .compare import (
    compare_const,
    equal,
    greater,
    greater_equal,
    less,
    less_equal,
    not_equal,
)
from .concat import (
    column_ref,
    concat_any,
    concat_arrays,
    concat_refs_any,
    concat_strings,
    concat_two_any,
)
from .group import (
    AggKind,
    aggregate_group,
    aggregate_group_any,
    aggregate_group_pair_any,
    aggregate_group_strings,
    group_corr,
    group_count,
    group_cov,
    group_first,
    group_last,
    group_max,
    group_mean,
    group_median,
    group_min,
    group_nunique,
    group_quantile,
    group_sem,
    group_size,
    group_skew,
    group_std,
    group_sum,
    group_var,
)
from .mask import apply_validity, combined_validity
from .nulls import (
    all_valid_mask,
    coalesce,
    coalesce_any,
    fill_backward,
    fill_backward_any,
    fill_forward,
    fill_forward_any,
    is_not_null,
    is_not_null_any,
    is_null,
    is_null_any,
    missing_count_any,
    present_bitmap,
    present_bitmap_any,
)
from .reduce import reduce_any
from .select import (
    filter_any,
    filter_range,
    filter_rows,
    take_any,
    take_range,
    take_rows,
)
from .sort import (
    argsort,
    argsort_any,
    argsort_into,
    argsort_multi,
    is_sorted,
    sort_key,
    sort_rows,
)
from .text import compare_text, compare_text_const
from .topn import GroupTop, group_top_rows, group_top_rows_any
from .unary import UnaryOp, absolute, invert, negate, unary_any, unary_type
