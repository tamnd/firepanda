"""Compile-time dtype lists.

Every generic kernel in firepanda is monomorphized by walking one of these lists
with `comptime for`. The lists are therefore the single knob that controls how
much code the compiler emits: adding one dtype to `NUMERIC` adds one full copy of
every kernel that dispatches over `NUMERIC`. See docs/specs/03-dtype-dispatch.md.

The lists are written out as literals rather than composed from each other.
Composition would be tidier to read and much harder to audit, and auditing is the
point: `tools/compile_budget.py` reads these lengths back out of the source and
graphs the instantiation count over time.
"""

from std.sys.info import size_of


# Physical integer types, split by signedness because promotion rules care.
comptime SIGNED: List[DType] = [
    DType.int8,
    DType.int16,
    DType.int32,
    DType.int64,
]

comptime UNSIGNED: List[DType] = [
    DType.uint8,
    DType.uint16,
    DType.uint32,
    DType.uint64,
]

comptime INTEGER: List[DType] = [
    DType.int8,
    DType.int16,
    DType.int32,
    DType.int64,
    DType.uint8,
    DType.uint16,
    DType.uint32,
    DType.uint64,
]

# float16 is included even though almost nobody stores a dataframe column in it.
# Leaving it out would mean `Array[DType.float16]` compiles and then fails at the
# first dispatch, which is a worse failure than a slightly larger binary.
comptime FLOAT: List[DType] = [DType.float16, DType.float32, DType.float64]

comptime NUMERIC: List[DType] = [
    DType.int8,
    DType.int16,
    DType.int32,
    DType.int64,
    DType.uint8,
    DType.uint16,
    DType.uint32,
    DType.uint64,
    DType.float16,
    DType.float32,
    DType.float64,
]

# Bool participates in comparison and sorting but not in arithmetic promotion,
# so it is in ORDERED and not in NUMERIC.
comptime ORDERED: List[DType] = [
    DType.bool,
    DType.int8,
    DType.int16,
    DType.int32,
    DType.int64,
    DType.uint8,
    DType.uint16,
    DType.uint32,
    DType.uint64,
    DType.float16,
    DType.float32,
    DType.float64,
]

# Hashing a float is legal and is what pandas does for groupby keys, NaN included.
# The hash kernel is responsible for normalizing negative zero and NaN payloads.
comptime HASHABLE: List[DType] = [
    DType.bool,
    DType.int8,
    DType.int16,
    DType.int32,
    DType.int64,
    DType.uint8,
    DType.uint16,
    DType.uint32,
    DType.uint64,
    DType.float16,
    DType.float32,
    DType.float64,
]

# Every physical dtype a firepanda column can be backed by. String columns are
# backed by uint8 payload bytes plus a separate offsets array, so DType.uint8
# appearing here covers them.
comptime ALL: List[DType] = [
    DType.bool,
    DType.int8,
    DType.int16,
    DType.int32,
    DType.int64,
    DType.uint8,
    DType.uint16,
    DType.uint32,
    DType.uint64,
    DType.float16,
    DType.float32,
    DType.float64,
]


def contains[list: List[DType]](dt: DType) -> Bool:
    """Reports whether a runtime dtype appears in a compile-time dtype list.

    The list is a parameter rather than an argument because a `comptime` list
    does not materialize into a runtime value. Unrolling it here is what the
    caller wants anyway: the whole thing collapses to a chain of comparisons
    against a value in a register.

    Args:
        dt: The dtype to look for.

    Parameters:
        list: The dtype list to search.

    Returns:
        True if `dt` is a member of `list`.
    """
    comptime for candidate in list:
        if dt == candidate:
            return True
    return False


def dtype_size(dt: DType) -> Int:
    """Returns the byte width of a dtype known only at runtime.

    The standard library's `size_of` takes the dtype as a parameter, so it cannot
    answer for a value. This walks `ALL` instead, which the compiler turns into a
    comparison chain over a value that is already in a register.

    Args:
        dt: The dtype.

    Returns:
        The number of bytes one element occupies in a values buffer, or 0 if the
        dtype is not one firepanda supports. Bool reports 1: validity is packed a
        bit per value, but a bool column's values are a byte per value.
    """
    comptime for candidate in ALL:
        if dt == candidate:
            return size_of[candidate]()
    return 0
