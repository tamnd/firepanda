"""Differential tests against the Python libraries firepanda is copying.

Mojo can import CPython in process, so the reference implementation is not a
description of pandas behaviour written down by hand, it is pandas. That matters
most for the parts of the semantics nobody would guess correctly: type promotion
is the one that exists at M0, and it has rules that look like mistakes until you
find the issue thread that explains them.

What is compared here at M0:

- `promote` against `numpy.promote_types` for every pair of dtypes we support.
- the physical layout of a validity bitmap against what pyarrow produces for the
  same nulls, because a buffer that does not match Arrow's bit order is useless
  to every consumer outside this process.

Later milestones add the operator by operator comparison against pandas and the
query by query comparison against DuckDB.

Usage:
    pixi run differential
"""

from std.python import Python, PythonObject

from firepanda.array.array import Array
from firepanda.bitmap.bitmap import Bitmap
from firepanda.dtype.logical import LogicalType, logical_for, promote
from firepanda.testing.rng import Rng

comptime BITMAP_CASES = 500
"""Random null patterns to compare against pyarrow."""


def as_int(value: PythonObject) raises -> Int:
    """Returns a Python integer as a Mojo one.

    `Int(value)` does not compile. `PythonObject` conforms to neither `Intable`
    nor `IntableRaising` in Mojo 1.0, and `__int__` is no help either because it
    hands back another `PythonObject`. The conversion that actually crosses the
    boundary is the CPython one.

    Args:
        value: A Python object that is already an int.

    Returns:
        The value as an `Int`.

    Raises:
        If the object is not an integer.
    """
    return Python.py_long_as_ssize_t(value)


def our_types() -> List[LogicalType]:
    """Returns the fixed-width types the promotion table covers.

    Returns:
        The twelve types, bool first.
    """
    return [
        LogicalType.BOOL,
        LogicalType.INT8,
        LogicalType.INT16,
        LogicalType.INT32,
        LogicalType.INT64,
        LogicalType.UINT8,
        LogicalType.UINT16,
        LogicalType.UINT32,
        LogicalType.UINT64,
        LogicalType.FLOAT16,
        LogicalType.FLOAT32,
        LogicalType.FLOAT64,
    ]


def numpy_name(type: LogicalType) -> String:
    """Returns the NumPy dtype name for one of our types.

    Args:
        type: The type.

    Returns:
        The name numpy.dtype accepts.
    """
    if type == LogicalType.BOOL:
        return "bool"
    return String(type)


def compare_promotion() raises -> Int:
    """Compares every promotion against numpy.promote_types.

    Returns:
        The number of pairs compared.

    Raises:
        If any pair disagrees with numpy.
    """
    var np = Python.import_module("numpy")
    var types = our_types()
    var compared = 0

    for i in range(len(types)):
        for j in range(len(types)):
            var ours = promote(types[i], types[j])
            var theirs = String(
                np.promote_types(
                    numpy_name(types[i]), numpy_name(types[j])
                ).name
            )
            if numpy_name(ours) != theirs:
                raise Error(
                    String(
                        "promote(",
                        numpy_name(types[i]),
                        ", ",
                        numpy_name(types[j]),
                        ") gives ",
                        numpy_name(ours),
                        " but numpy gives ",
                        theirs,
                    )
                )
            compared += 1

    return compared


def compare_bitmap_layout() raises -> Int:
    """Compares our validity bitmap bytes against the ones pyarrow writes.

    Arrow packs validity least significant bit first and pads the last byte with
    zeros. Writing it the other way round is invisible inside this process and
    silently corrupts every buffer that leaves it, so the check is on the bytes
    rather than on the accessors.

    Returns:
        The number of null patterns compared.

    Raises:
        If any byte differs from pyarrow's.
    """
    var pa = Python.import_module("pyarrow")
    var rng = Rng(0x5DEECE66D)
    var compared = 0

    for pattern in range(BITMAP_CASES):
        var length = rng.next_range(1, 300)
        var bitmap = Bitmap(length, all_valid=False)
        var values = Python.list()
        for i in range(length):
            var valid = rng.next_below(4) != 0
            bitmap.set(i, valid)
            if valid:
                values.append(i)
            else:
                values.append(Python.none())

        var array = pa.array(values, type=pa.int64())
        if as_int(array.null_count) != bitmap.null_count():
            raise Error(
                String(
                    "case ",
                    pattern,
                    ": pyarrow counts ",
                    as_int(array.null_count),
                    " nulls but we count ",
                    bitmap.null_count(),
                    " over ",
                    length,
                    " rows",
                )
            )

        var theirs = array.buffers()[0]
        if theirs is Python.none():
            # pyarrow omits the validity buffer when nothing is null, which only
            # happens here when the generator produced no nulls at all.
            if bitmap.null_count() != 0:
                raise Error(
                    String(
                        "case ",
                        pattern,
                        ": pyarrow dropped a needed validity buffer",
                    )
                )
            compared += 1
            continue

        var their_bytes = theirs.to_pybytes()
        for byte_index in range(bitmap.byte_length()):
            var ours = (
                bitmap.unsafe_ptr().unsafe_offset(byte_index).unsafe_load()
            )
            var them = UInt8(as_int(their_bytes[byte_index]))
            if ours != them:
                raise Error(
                    String(
                        "case ",
                        pattern,
                        ": validity byte ",
                        byte_index,
                        " is ",
                        Int(ours),
                        " but pyarrow wrote ",
                        Int(them),
                        " (length ",
                        length,
                        ")",
                    )
                )
        compared += 1

    return compared


def main() raises:
    var pairs = compare_promotion()
    print("ok: promotion matches numpy for", pairs, "type pairs")

    var patterns = compare_bitmap_layout()
    print("ok: validity layout matches pyarrow for", patterns, "null patterns")
