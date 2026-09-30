"""Prepared statements, and the values an `EXECUTE` puts where the parameters are.

`PREPARE q AS SELECT ...` keeps the statement under a name and `EXECUTE q(...)`
runs it with a value for each parameter. A parameter is written `?`, `$1` or
`$name`, and each form names the value it takes: `$n` takes the `n`th value
whatever order it is written in, a `?` takes the one after the highest any
parameter before it took, so the `n`th `?` of a statement with nothing else
takes the `n`th, and `$name` takes the value passed as `name := ...`. Two
parameters that name the same value both take it, and a statement may name its
values by number or by name but not both, as in DuckDB.

An argument is an expression, as it is in DuckDB, so `EXECUTE q(1 + 1)` passes
two and `EXECUTE q(DATE '2020-01-01')` passes a date. The arguments are parsed
into the same arenas as the statement they are for, and each parameter node is
then made a copy of the node its argument's expression starts at. The copy
points at the argument's own children, so the statement reads the argument
exactly where the parameter was written, and a parameter is typed the way the
expression passed for it is typed, which is DuckDB's rule too: `9999999999`
passed for `?` is a `BIGINT` and `-5` is an `INTEGER`.

The check is DuckDB's, in its order. A parameter no value was passed for is
named first, and then a value passed for no parameter, each with DuckDB's
message. A statement with a parameter in it that is not run by an `EXECUTE` at
all is refused, because there is nothing to put there.
"""

from .ast import EXPR_PARAMETER, Ast


struct Arguments(Copyable, Movable):
    """The values an `EXECUTE` passed, as the text of each expression."""

    var names: List[String]
    """What each value is for: `1`, `2` and on for a value passed by position,
    and the name folded for one passed as `name := value`."""

    var texts: List[String]
    """Each value's expression, as the `EXECUTE` wrote it."""

    var bound: Bool
    """Whether an `EXECUTE` is running, which is the only time a parameter has
    anything to take."""

    def __init__(out self):
        """No `EXECUTE` running."""
        self.names = List[String]()
        self.texts = List[String]()
        self.bound = False

    def __init__(out self, var names: List[String], var texts: List[String]):
        """The values one `EXECUTE` passed.

        Args:
            names: What each value is for.
            texts: Each value's expression.
        """
        self.names = names^
        self.texts = texts^
        self.bound = True


def _identifier(ast: Ast, node: UInt32, mut counted: Int) -> String:
    """The value a parameter takes: what follows the `$`, folded, or for a `?`
    the number one past the highest any parameter before it took, which is
    how DuckDB counts them."""
    var name = ast.text(ast.exprs[Int(node)].payload)
    if name.byte_length() == 0:
        counted += 1
        return String(counted)
    try:
        counted = max(counted, Int(name))
    except:
        pass
    return name.lower()


def _numbered(name: String) -> Bool:
    for byte in name.as_bytes():
        if byte < Byte(ord("0")) or byte > Byte(ord("9")):
            return False
    return True


def _listed(names: List[String]) -> String:
    var text = String()
    for i in range(len(names)):
        if i > 0:
            text += ", "
        text += names[i]
    return text^


def taken(ast: Ast, arguments: Arguments) raises -> List[Int]:
    """Which argument each parameter of a statement takes, checked.

    Args:
        ast: The statement's arenas.
        arguments: The values the running `EXECUTE` passed, or none.

    Returns:
        One entry per expression node: the position in `arguments` of the
        value that node takes, or -1 for a node that is not a parameter.
        Empty when the statement has no parameter.

    Raises:
        If the statement has a parameter and no `EXECUTE` is running, if a
        parameter has no value, or if a value has no parameter, each with
        DuckDB's message.
    """
    var parameters = List[UInt32]()
    for i in range(len(ast.exprs)):
        if ast.exprs[i].kind == EXPR_PARAMETER:
            parameters.append(UInt32(i))
    if len(parameters) == 0 and not arguments.bound:
        return List[Int]()
    if len(parameters) > 0 and not arguments.bound:
        raise Error(
            "Invalid Input Error: Prepared statement parameters cannot be used"
            " directly\nTo use prepared statement parameters, use PREPARE to"
            " prepare a statement, followed by EXECUTE"
        )
    # The nodes are in the order they were built and a `?` counts in the order
    # it was written, which the token each one starts at says.
    for i in range(1, len(parameters)):
        var j = i
        while j > 0 and (
            ast.exprs[Int(parameters[j])].token
            < ast.exprs[Int(parameters[j - 1])].token
        ):
            var held = parameters[j]
            parameters[j] = parameters[j - 1]
            parameters[j - 1] = held
            j -= 1
    var counted = 0
    var wants = List[String]()
    for node in parameters:
        wants.append(_identifier(ast, node, counted))
    var named = 0
    for want in wants:
        if not _numbered(want):
            named += 1
    if named > 0 and named < len(wants):
        raise Error(
            "Not implemented Error: Mixing named and positional parameters is"
            " not supported yet"
        )
    var missing = List[String]()
    for want in wants:
        if want not in arguments.names and want not in missing:
            missing.append(want)
    if len(missing) > 0:
        raise Error(
            String(
                (
                    "Invalid Input Error: Values were not provided for the"
                    " following prepared statement parameters: "
                ),
                _listed(missing),
            )
        )
    var excess = List[String]()
    for name in arguments.names:
        if name not in wants:
            excess.append(name)
    if len(excess) > 0:
        raise Error(
            String(
                (
                    "Invalid Input Error: Parameter argument/count mismatch,"
                    " identifiers of the excess parameters: "
                ),
                _listed(excess),
            )
        )
    if len(parameters) == 0:
        return List[Int]()
    var out = List[Int](length=len(ast.exprs), fill=-1)
    for i in range(len(parameters)):
        var at = 0
        while arguments.names[at] != wants[i]:
            at += 1
        out[Int(parameters[i])] = at
    return out^
