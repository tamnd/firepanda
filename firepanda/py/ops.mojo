"""Reading an arithmetic call's arguments, on the way in from Python.

A series has twenty six operators and named forms and a frame has twenty four,
and every one of them is the same call with a different operation in it. Binding
each one separately would be fifty bound methods that differ by three letters,
which is fifty entries in the registration table, fifty stub lines and fifty
places for the null rule to be written down slightly differently.

So the boundary is narrow instead. There is one entry point per shape, `binary`,
`compare` and `unary`, and the operation crosses as a string. That is the same
trick `IndexMixin._set_operation` already plays on the Python side for the three
set operations, and it is the reason the generated table can carry fifty rows
that are each one expression.

The string is a word rather than a symbol. `"add"` rather than `"+"`, because the
word is what pandas calls the named form, so `s.add(t)` and `s + t` reach the
same string from both directions and there is no second spelling to keep in step.

### Why the operand is not read here

A `binary` takes a series, a frame or a constant and the three do different
things, so the reader that tells them apart has to live next to the type that
holds them. `series.mojo` and `frame.mojo` each do their own downcast. What is
here is only what both of them read the same way.

### Why a failure out of the core is tagged `dtype`

The core does not tag its errors, because a tag is a Python class and the core
does not know there is a Python. An untagged error that reaches `translate` on
the other side comes out a `RuntimeError`, which is the right answer for a
failure nobody classified and the wrong one here, since every one of these
raises for the same reason: two dtypes that have no operation between them, or a
constant of a type the column cannot take. pandas raises `TypeError` for all of
those, so each call site catches what the core raised, keeps the message, and
puts `dtype` on it.

The comparisons are the exception and `compare_series` says why.
"""

from std.collections import Optional
from std.python import Python, PythonObject

from firepanda.array.value import Value
from firepanda.dtype.logical import LogicalType, TypeKind
from firepanda.kernel.binary import (
    BinaryOp,
    resolve_constant,
    unsupported_on_bool,
)
from firepanda.kernel.unary import UnaryOp
from firepanda.py.errors import DTYPE, OVERFLOW, UNSUPPORTED, VALUE, tagged


def binary_op(name: String) raises -> BinaryOp:
    """Reads the name of a binary operation.

    The thirteen names are the pandas method names, so the list here is the list
    of methods the Python layer exposes and a name that is not on it is a bug in
    the generated table rather than something a user typed.

    Args:
        name: The operation, such as `add` or `lt`.

    Returns:
        The operation.

    Raises:
        Error: Tagged `value`, if the name is not one of the thirteen.
    """
    if name == "add":
        return BinaryOp.ADD
    if name == "sub":
        return BinaryOp.SUB
    if name == "mul":
        return BinaryOp.MUL
    if name == "truediv":
        return BinaryOp.DIV
    if name == "floordiv":
        return BinaryOp.FLOORDIV
    if name == "mod":
        return BinaryOp.MOD
    if name == "pow":
        return BinaryOp.POW
    if name == "eq":
        return BinaryOp.EQ
    if name == "ne":
        return BinaryOp.NE
    if name == "lt":
        return BinaryOp.LT
    if name == "le":
        return BinaryOp.LE
    if name == "gt":
        return BinaryOp.GT
    if name == "ge":
        return BinaryOp.GE
    raise tagged(
        VALUE,
        String(
            "unknown operation ",
            name,
            (
                "; expected one of add, sub, mul, truediv, floordiv, mod, pow,"
                " eq, ne, lt, le, gt or ge"
            ),
        ),
    )


def unary_op(name: String) raises -> UnaryOp:
    """Reads the name of a unary operation.

    Args:
        name: The operation, such as `neg` or `abs`.

    Returns:
        The operation.

    Raises:
        Error: Tagged `value`, if the name is not one of the four.
    """
    if name == "neg":
        return UnaryOp.NEG
    if name == "pos":
        return UnaryOp.POS
    if name == "abs":
        return UnaryOp.ABS
    if name == "invert":
        return UnaryOp.INVERT
    raise tagged(
        VALUE,
        String(
            "unknown operation ",
            name,
            "; expected one of neg, pos, abs or invert",
        ),
    )


def constant(value: PythonObject, name: String) raises -> Value:
    """Reads a Python scalar as a value a kernel can use.

    The order the four types are tried in matters, because `bool` is a subclass
    of `int` in Python and `isinstance(True, int)` is therefore True. A boolean
    read as an integer would turn `s == True` into `s == 1`, which is the same
    answer on an integer column and a different one on a boolean column, so the
    narrower type is tested first.

    A `None` is not read here. It is a missing value rather than a constant and
    every operation involving one answers a column of nulls, which is a thing the
    caller has to decide it wants rather than something this can decide for it.

    Args:
        value: What Python passed.
        name: The parameter name, for the message.

    Returns:
        The constant.

    Raises:
        Error: Tagged `dtype`, if the object is not one of the four scalars.
    """
    var builtins = Python.import_module("builtins")
    if Bool(builtins.isinstance(value, builtins.bool)):
        return Value(Bool(py=value)).weakened()
    if Bool(builtins.isinstance(value, builtins.int)):
        return Value(_as_int64(value)).weakened()
    if Bool(builtins.isinstance(value, builtins.float)):
        return Value(Float64(py=value)).weakened()
    if Bool(builtins.isinstance(value, builtins.str)):
        return Value(String(value))
    raise tagged(
        DTYPE,
        String(
            name,
            " must be a series, a frame or a number, got ",
            Python.type(value).__name__,
            " ",
            value.__repr__(),
        ),
    )


def _as_int64(value: PythonObject) raises -> Int64:
    """Reads a Python integer, refusing one that will not fit a machine word.

    Python integers are unbounded and int64 is not, so there is a range where
    the number is a perfectly good Python object and there is nothing here to
    put it in. pandas answers the same way and its message is copied exactly,
    because that is the string somebody who hits this will search for.

    It is the narrowest of the three refusals a scalar can meet. This one is
    about the machine and does not know what column the value is headed for; the
    one in `resolve_constant` is about the column's dtype and has a different
    message for that reason.

    Args:
        value: The Python integer.

    Returns:
        The number.

    Raises:
        Error: Tagged `overflow`, if it does not fit in an int64.
    """
    try:
        return Int64(Int(py=value))
    except:
        raise tagged(OVERFLOW, "Python int too large to convert to C long")


def binary_tag(
    op: BinaryOp, left: List[LogicalType], right: List[LogicalType]
) -> String:
    """Says which tag a failed operation between two columns should carry.

    Almost every failure on this path is a pair of dtypes with no operation
    between them, which is `dtype`. The exception is `/`, `//` and `**` on two
    bools, which pandas declines with a `NotImplementedError` rather than a
    `TypeError`, and which the core raises untagged like everything else.

    The question is asked about the two sides rather than about the aligned
    pairs. A frame carries one dtype per column and two frames line their
    columns up by name, so a frame whose bool column does not meet the other
    frame's bool column would be named `unsupported` here for a failure that was
    something else. That is a wrong word on a call that was going to raise
    anyway, and the alternative is doing the alignment a second time in the
    binding to improve it.

    A bool on one side only is not enough and is not a near miss. `bool / int8`
    promotes to int8 and answers a float64, so nothing about it raises, and a
    failure on a pair like that is a failure about something else.

    Args:
        op: The operation.
        left: The left operand's dtypes, which is one for a series and one per
            column for a frame.
        right: The right operand's dtypes, read the same way.

    Returns:
        The `unsupported` prefix if the operation is one of the three and both
        sides carry a bool, and the `dtype` prefix otherwise.
    """
    if not unsupported_on_bool(op):
        return DTYPE
    if not _holds_bool(left) or not _holds_bool(right):
        return DTYPE
    return UNSUPPORTED


def _holds_bool(dtypes: List[LogicalType]) -> Bool:
    """Whether any of the dtypes is bool.

    Args:
        dtypes: One dtype for a series, one per column for a frame.

    Returns:
        True if at least one is bool.
    """
    for dtype in dtypes:
        if dtype.kind == TypeKind.BOOL:
            return True
    return False


def constant_tag(
    dtypes: List[LogicalType], value: Value, op: BinaryOp
) -> String:
    """Says which tag a failed constant operation should carry.

    An operation against a constant has three ways to fail and they are three
    different kinds. The dtypes can have no answer between them, which is
    `dtype`; the constant can be a number too large for the column it was headed
    for, which is `overflow`; and it can be one of the three operations pandas
    declines on a bool column, which is `unsupported`. The core knows which one
    happened and cannot say so, because it raises untagged, which is the rule
    `firepanda/py/errors.mojo` states.

    The bool case is asked first and is asked as a question about the operation
    rather than by catching anything, because it is the one failure that is not
    a failure of the constant. `s / True` on a bool column raises for the same
    reason `s / t` does, which is that pandas puts a dtype check in front of
    `/`, `//` and `**` on bools, so nothing about resolving the constant went
    wrong and there is nothing to re-run to find out.

    So the binding asks the question again on the failure path, the way
    `PySeries.compare_series` compares its labels again to find out which of its
    two failures it had. Resolving a constant is a handful of comparisons
    against a dtype and no data is touched, so asking twice costs nothing, and
    it is only ever reached once something has already gone wrong.

    Args:
        dtypes: The dtypes the constant would meet, which is one for a series
            and one per column for a frame.
        value: The constant, as `constant` built it.
        op: The operation.

    Returns:
        The `unsupported` prefix if the operation is one of the three pandas
        declines on a bool column, the `overflow` prefix if resolving the
        constant against any of those dtypes is what failed, and
        the `dtype` prefix otherwise. A frame that fails on its third column is
        still an overflow, so any is the right question rather than all, and
        pandas answers a mixed frame the same way: one bool column among the
        integer ones is enough for `df / True` to raise.
    """
    if value.type.kind == TypeKind.BOOL:
        var one = List[LogicalType](capacity=1)
        one.append(value.type)
        if binary_tag(op, dtypes, one) == UNSUPPORTED:
            return UNSUPPORTED
    for dtype in dtypes:
        try:
            _ = resolve_constant(dtype, value, op)
        except:
            return OVERFLOW
    return DTYPE


def fill(value: PythonObject) raises -> Optional[Value]:
    """Reads the `fill_value` argument, which is usually absent.

    Args:
        value: What Python passed, which is `None` on almost every call.

    Returns:
        The constant, or nothing.

    Raises:
        Error: Tagged `dtype`, if it is present and is not a scalar.
    """
    if value is Python.none():
        return None
    return constant(value, "fill_value")
