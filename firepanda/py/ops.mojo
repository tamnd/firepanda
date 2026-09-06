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
from firepanda.kernel.binary import BinaryOp
from firepanda.kernel.unary import UnaryOp
from firepanda.py.errors import DTYPE, VALUE, tagged


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
        return Value(Bool(py=value))
    if Bool(builtins.isinstance(value, builtins.int)):
        return Value(Int64(Int(py=value)))
    if Bool(builtins.isinstance(value, builtins.float)):
        return Value(Float64(py=value))
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
