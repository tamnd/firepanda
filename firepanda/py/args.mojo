"""Reading the arguments a bound method was handed.

Everything crossing into a binding is a `PythonObject`, so every method starts by
turning some of them into Mojo values, and the interesting part is what happens
when that fails. `Int(py=value)` raises `invalid literal for int() with base 10:
'x'`, which is the right complaint and names neither the argument nor the
function. A user with three integer arguments cannot tell from it which one they
got wrong.

So each reader here takes the parameter name and puts it in the message. That is
the whole content of the module and it is the reason it exists as one: the same
two line helper had been written twice by the time there were two bound types,
and a third copy is where a private convention starts drifting into three
slightly different messages.

Nothing here validates a range or a set of allowed words. A method that only
accepts `left` and `right` says so itself, in its own message, because the list
belongs next to the behaviour rather than in a table over here.
"""

from std.python import Python, PythonObject

from firepanda.py.errors import DTYPE, tagged


def whole(value: PythonObject, name: String) raises -> Int:
    """Reads a Python integer, and says which argument was wrong if it is not one.

    Args:
        value: What Python passed.
        name: The parameter name, for the message.

    Returns:
        The integer.

    Raises:
        Error: Tagged `dtype`, if the value is not an integer.
    """
    try:
        return Int(py=value)
    except:
        raise tagged(
            DTYPE,
            String(
                name,
                " must be an integer, got ",
                Python.type(value).__name__,
                " ",
                value.__repr__(),
            ),
        )


def maybe_whole(value: PythonObject, name: String) raises -> Optional[Int]:
    """Reads a Python integer that is allowed to be absent.

    `None` is a position in a slice and not a missing argument, which is why
    this exists rather than a default. `s.str.slice(None, None, -1)` reverses
    every row and `s.str.slice(0, 0, -1)` empties them, so an absent bound has
    to survive the crossing as an absence and cannot be turned into a number on
    either side of it.

    Args:
        value: What Python passed.
        name: The parameter name, for the message.

    Returns:
        The integer, or nothing if Python passed `None`.

    Raises:
        Error: Tagged `dtype`, if the value is neither an integer nor `None`.
    """
    if value is Python.none():
        return None
    return whole(value, name)


def flag(value: PythonObject, name: String) raises -> Bool:
    """Reads a Python boolean.

    Anything truthy is not accepted. pandas takes `sort=1` and means `sort=True`,
    and copying that would mean `sort=None`, which pandas gives a third meaning
    to, quietly becoming `False` here. Refusing is the only answer that cannot be
    wrong in a way nobody notices.

    Args:
        value: What Python passed.
        name: The parameter name, for the message.

    Returns:
        The boolean.

    Raises:
        Error: Tagged `dtype`, if the value is not a `bool`.
    """
    var builtins = Python.import_module("builtins")
    if not Bool(builtins.isinstance(value, builtins.bool)):
        raise tagged(
            DTYPE,
            String(
                name,
                " must be True or False, got ",
                Python.type(value).__name__,
                " ",
                value.__repr__(),
            ),
        )
    return Bool(py=value)


def words(value: PythonObject, name: String) raises -> String:
    """Reads a Python string.

    `String(value)` would accept anything at all, since every Python object has a
    `__str__`, so an argument that was meant to be a name and arrived as a list
    would become the text of the list rather than an error.

    Args:
        value: What Python passed.
        name: The parameter name, for the message.

    Returns:
        The string.

    Raises:
        Error: Tagged `dtype`, if the value is not a `str`.
    """
    var builtins = Python.import_module("builtins")
    if not Bool(builtins.isinstance(value, builtins.str)):
        raise tagged(
            DTYPE,
            String(
                name,
                " must be a string, got ",
                Python.type(value).__name__,
                " ",
                value.__repr__(),
            ),
        )
    return String(value)


def number(value: PythonObject, name: String) raises -> Float64:
    """Reads a Python number as a float.

    An integer is accepted, because the two arguments that reach this are a
    delta degrees of freedom and a quantile and a caller writes `ddof=1` rather
    than `ddof=1.0`. A boolean is accepted for the same reason `flag` gives in
    reverse: Python makes a `bool` an `int` and refusing it here would be
    stricter than the library being copied.

    Args:
        value: What Python passed.
        name: The parameter name, for the message.

    Returns:
        The number.

    Raises:
        Error: Tagged `dtype`, if the value is not a number.
    """
    var builtins = Python.import_module("builtins")
    var numeric = Bool(builtins.isinstance(value, builtins.float)) or Bool(
        builtins.isinstance(value, builtins.int)
    )
    if not numeric:
        raise tagged(
            DTYPE,
            String(
                name,
                " must be a number, got ",
                Python.type(value).__name__,
                " ",
                value.__repr__(),
            ),
        )
    return Float64(py=builtins.float(value))
