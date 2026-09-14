"""What a call comes out as, once an overload has won.

`resolve.mojo` picks the signature and says at the bottom of its own file that
answering the type is somebody else's job. This is that job. Most signatures
answer it themselves, `length(VARCHAR) -> BIGINT` being every one of them, and
for those there is nothing to derive and this hands back what the catalog says.

The ones worth a file are the signatures whose return is a rule. There are
three shapes of rule and DuckDB writes them the same way, as a word in the
catalog that is not a type, so the shape has to be read off the signature
rather than looked up.

The first is `ANY` and the single letter templates, which mean the type the
argument came in as. `first(ANY) -> ANY` over a `DECIMAL(5,2)` column is a
`DECIMAL(5,2)`, and over a `UUID` column it is a `UUID`. Where several
parameters carry the letter the question is which one wins, and the answer is
the first, measured: `arg_max(ANY, ANY)` over a boolean and an interval is a
boolean and the other way round it is an interval, so the second argument is
the one being ordered by and has nothing to do with the result.

The exception is a variadic, where the letter is in the trailing slot. Then
every argument is one of them and the result is the type they all agree on,
because that is a common type DuckDB works out and casts to rather than a type
any one argument brought: `greatest(DECIMAL(5,2), INTEGER, TINYINT)` is a
`DECIMAL(12,2)`, which is none of the three.

The second is a bare `DECIMAL`, which is a promise about the family and not
about the width. A bare `DECIMAL` parameter is a cast target, so every argument
landing on one is cast to what they agree on and the width and scale come from
that: `mod(DECIMAL(5,2), INTEGER)` is a `DECIMAL(12,2)`. That is the default,
and four names then do something else with the result. `sum` widens to 38
digits and keeps the scale. `avg` returns a `DOUBLE` even though the catalog
says `DECIMAL`, which is the bind function disagreeing with the signature and
is DuckDB's to explain, not this file's. `ceil`, `ceiling` and `floor` keep the
width and drop the scale to nothing.

The third is the containers, and of those it is the lists that are answered.
The element is read off the return the same way the whole return is, either
said outright as in `str_split(VARCHAR, VARCHAR) -> VARCHAR[]` or as a letter
standing for an argument as in `list(T) -> T[]`. `MAP` is the one left, and it
is left because a map needs two element types and `SqlType` carries one. That
returns an invalid type, which is this saying it does not know rather than
saying anything wrong.

The name is a parameter because DuckDB's own catalog is not enough to answer
this. A bind function per name is how DuckDB does it and `sum` and `avg` really
do declare the same return and produce different types. Keying on the name is
that fact written down rather than a shortcut around it.
"""

from .cast import common_type
from .generated.functions import KIND_MACRO
from .registry import (
    NO_SLOT,
    ROLE_ANY,
    ROLE_EXACT,
    ROLE_LIST,
    ROLE_TEMPLATE,
    Overload,
    Registry,
)
from .types import (
    DECIMAL_MAX_WIDTH,
    INVALID,
    TYPE_ARRAY,
    TYPE_DATE,
    TYPE_DECIMAL,
    TYPE_DOUBLE,
    TYPE_LIST,
    TYPE_MAP,
    TYPE_STRUCT,
    TYPE_TIMESTAMP,
    TYPE_UNION,
    TYPE_VARCHAR,
    SqlType,
    decimal,
)


def result_type(
    registry: Registry,
    name: StringSlice,
    overload: Overload,
    arguments: List[SqlType],
) raises -> SqlType:
    """The type a resolved call comes out as.

    Args:
        registry: The catalog.
        name: The name the call was written with, which is what tells `sum`
            from `avg` where the catalog does not.
        overload: The signature that won.
        arguments: The argument types, in the order they were written.

    Returns:
        The type, or an invalid type where it cannot be said. That is a macro,
        whose body is not in the catalog, and a `MAP`, which needs two element
        types where `SqlType` carries one.

    Raises:
        Error: Never, but the decimal it builds checks its own width.
    """
    if overload.kind == KIND_MACRO or overload.returns == NO_SLOT:
        return INVALID

    var slot = Int(overload.returns)
    var role = registry.roles[slot]

    if role == ROLE_ANY or role == ROLE_TEMPLATE:
        if name == "concat":
            return _concatenated(arguments)
        var carried = _carried(
            registry, overload, arguments, registry.spellings[slot]
        )
        if name == "median":
            return _interpolated(carried)
        return carried

    if role == ROLE_LIST:
        return _listed(registry, overload, arguments, slot)

    if role != ROLE_EXACT:
        return INVALID

    var declared = registry.types[slot]
    if (
        declared.id == TYPE_LIST
        or declared.id == TYPE_ARRAY
        or declared.id == TYPE_STRUCT
        or declared.id == TYPE_MAP
        or declared.id == TYPE_UNION
    ):
        return INVALID
    if declared.id != TYPE_DECIMAL:
        return declared

    var agreed = _unified(registry, overload, arguments, "DECIMAL")
    if agreed.id != TYPE_DECIMAL:
        return INVALID
    if name == "avg":
        return SqlType(TYPE_DOUBLE)
    if name == "sum":
        return decimal(DECIMAL_MAX_WIDTH, agreed.scale)
    if name == "ceil" or name == "ceiling" or name == "floor":
        return decimal(agreed.width, 0)
    if name == "round":
        # `round(x)` is the same rounding to nothing that `floor` does. With a
        # scale argument the answer is that scale, or the argument's own where
        # that is smaller, and DuckDB reads it off a constant and refuses the
        # call outright where the second argument is a column. Nothing here
        # knows which it was given, so the one argument form is answered and
        # the other is left alone.
        if overload.arity != 1:
            return INVALID
        return decimal(agreed.width, 0)
    return agreed


def _carried(
    registry: Registry,
    overload: Overload,
    arguments: List[SqlType],
    spelling: StringSlice,
) -> SqlType:
    """The argument type a template spelling stands for.

    The first parameter carrying the spelling is the one that answers, unless
    the trailing slot carries it too, in which case every argument does and
    what they agree on answers.

    Args:
        registry: The catalog.
        overload: The signature that won.
        arguments: The argument types.
        spelling: The return's spelling, the one to look for.

    Returns:
        The type, or an invalid type where no parameter carries the spelling,
        which is a container like `MAP(K, V)` that this does not answer.
    """
    if overload.variadic() and registry.spelling(overload.varargs) == spelling:
        return _unified(registry, overload, arguments, spelling)
    for at in range(Int(overload.arity)):
        var slot = Int(registry.parameters[Int(overload.first) + at])
        if registry.spellings[slot] == spelling and at < len(arguments):
            return arguments[at]
    return INVALID


def _unified(
    registry: Registry,
    overload: Overload,
    arguments: List[SqlType],
    spelling: StringSlice,
) -> SqlType:
    """What every argument landing on one spelling agrees on.

    Args:
        registry: The catalog.
        overload: The signature that won.
        arguments: The argument types.
        spelling: The parameter spelling to gather.

    Returns:
        The common type of those arguments, or an invalid type where they do
        not mix or where none of them carried the spelling.
    """
    var out = INVALID
    for at in range(len(arguments)):
        var slot: Int
        if at < Int(overload.arity):
            slot = Int(registry.parameters[Int(overload.first) + at])
        elif overload.variadic():
            slot = Int(overload.varargs)
        else:
            break
        if registry.spellings[slot] != spelling:
            continue
        if out == INVALID:
            out = arguments[at]
        else:
            out = common_type(out, arguments[at])
    return out


def _listed(
    registry: Registry,
    overload: Overload,
    arguments: List[SqlType],
    slot: Int,
) -> SqlType:
    """What a signature returning a list comes out as.

    The element is read the same way the whole return is read anywhere else.
    `VARCHAR[]` says its element outright, and `T[]` and `ANY[]` mean the type
    of the argument the letter stands for, which for `list(T) -> T[]` is the
    only argument there is and for `max(ANY, BIGINT) -> ANY[]` is the first of
    the two.

    Args:
        registry: The catalog.
        overload: The signature that won.
        arguments: The argument types.
        slot: The return's slot.

    Returns:
        The list type, or an invalid type where the element cannot be said.
    """
    var element = registry.elements[slot]
    if element == ROLE_EXACT:
        return SqlType.list_of(registry.element_types[slot])
    if element != ROLE_ANY and element != ROLE_TEMPLATE:
        return INVALID
    ref spelling = registry.spellings[slot]
    var carried = _carried(
        registry,
        overload,
        arguments,
        spelling[byte = 0 : spelling.byte_length() - 2],
    )
    if carried == INVALID:
        return INVALID
    return SqlType.list_of(carried)


def _interpolated(type: SqlType) -> SqlType:
    """What `median` comes out as over one argument type.

    It is declared over `ANY` and returns `ANY`, and for most types that is
    true: a median over a `VARCHAR` column is one of the strings in it. But a
    median over an even number of rows is the midpoint of the two in the
    middle, and for a type where that midpoint is not a value of the type the
    result widens to one where it is. Ten integer types give a `DOUBLE`, a
    `DATE` gives a `TIMESTAMP`, and a `FLOAT` stays a `FLOAT` because the
    midpoint of two floats is one.

    Args:
        type: What the argument came in as.

    Returns:
        The type of the median over it.
    """
    if type.is_integer():
        return SqlType(TYPE_DOUBLE)
    if type.id == TYPE_DATE:
        return SqlType(TYPE_TIMESTAMP)
    return type


def _concatenated(arguments: List[SqlType]) -> SqlType:
    """What `concat` comes out as, which its signature does not say.

    It is declared over `ANY` and returns `ANY` and is neither. Over anything
    that is not a list it is a `VARCHAR`, which is the whole of what everyday
    use of it means. Over lists it joins them end to end instead and comes out
    as a list over what their elements agree on, so two `INTEGER[]` give an
    `INTEGER[]` and an `INTEGER[]` beside a `BIGINT[]` gives a `BIGINT[]`.
    Mixing a list with something that is not one is an error DuckDB refuses
    outright rather than a type, so there is nothing to answer for it here.

    Args:
        arguments: The argument types.

    Returns:
        `VARCHAR`, or the list, or an invalid type where an element is not
        known.
    """
    var lists = 0
    for type in arguments:
        if type.id == TYPE_LIST or type.id == TYPE_ARRAY:
            lists += 1
    if lists == 0:
        return SqlType(TYPE_VARCHAR)
    if lists != len(arguments):
        return INVALID

    var element = INVALID
    for type in arguments:
        var one = type.element_type()
        if one == INVALID:
            return INVALID
        if element == INVALID:
            element = one
        else:
            element = common_type(element, one)
    return SqlType.list_of(element)
