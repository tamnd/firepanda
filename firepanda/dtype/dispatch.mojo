"""The bridge between a runtime dtype and a compile-time one.

Everything above this file works with a dtype as a value. Everything below works
with a dtype as a parameter. `dispatch` is the one place the two meet, and it
does so by unrolling a compile-time list of candidate dtypes into a chain of
comparisons and calling the operation with the matching one bound as a parameter.

The cost model is worth stating plainly. `dispatch[NUMERIC](col, op)` compiles
eleven copies of `op`. Two nested dispatches over `NUMERIC` compile one hundred
and twenty one. That is why the lists in `lists.mojo` are narrow, why binary size
is graphed from this milestone rather than thresholded later, and why a kernel
that does not need a dtype parameter should not take one.

See docs/specs/03-dtype-dispatch.md.
"""

from firepanda.array.any import AnyArray
from firepanda.array.array import Array

from .lists import ALL, contains


def dispatch[
    R: Movable,
    op: def[dt: DType](AnyArray) raises -> R,
    //,
    list: List[DType],
](col: AnyArray, operation: op) raises -> R:
    """Calls an operation with the column's dtype bound as a parameter.

    Args:
        col: The type-erased column.
        operation: The operation to run. It is instantiated once per dtype in
            `list`, so the list length is a direct multiplier on compile time and
            binary size.

    Parameters:
        R: The operation's return type.
        op: The type of the operation, inferred from `operation`.
        list: The dtypes the operation supports.

    Returns:
        Whatever the operation returns.

    Raises:
        If the column's dtype is not in `list`. This is not a bug in the caller's
        data, it is a gap in the kernel, and the message says which dtype is
        missing so that the fix is obvious.
    """
    comptime for candidate in list:
        if col.dtype() == candidate:
            return operation[candidate](col)
    raise Error(
        "unsupported dtype "
        + String(col.dtype())
        + " for this operation; supported: "
        + list_names[list]()
    )


def dispatch_typed[
    R: Movable,
    op: def[dt: DType](Array[dt]) raises -> R,
    //,
    list: List[DType],
](col: AnyArray, operation: op) raises -> R:
    """Calls an operation with a typed copy of the column.

    Use this when the operation wants an `Array[dt]` rather than the erased
    column. It costs a deep copy of the buffers, so kernels on the hot path take
    the erased column and read through `unsafe_ptr` after dispatch has already
    proved the dtype.

    Args:
        col: The type-erased column.
        operation: The operation to run.

    Parameters:
        R: The operation's return type.
        op: The type of the operation, inferred from `operation`.
        list: The dtypes the operation supports.

    Returns:
        Whatever the operation returns.

    Raises:
        If the column's dtype is not in `list`.
    """
    comptime for candidate in list:
        if col.dtype() == candidate:
            return operation[candidate](col.as_typed[candidate]())
    raise Error(
        "unsupported dtype "
        + String(col.dtype())
        + " for this operation; supported: "
        + list_names[list]()
    )


def list_names[list: List[DType]]() -> String:
    """Renders a dtype list for an error message.

    Parameters:
        list: The dtype list.

    Returns:
        A comma separated list of dtype names.
    """
    var out = String()
    comptime for candidate in list:
        if out.byte_length() > 0:
            out += ", "
        out += String(candidate)
    return out^
