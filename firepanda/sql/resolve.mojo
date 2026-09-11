"""Picking one overload out of several, the way DuckDB picks.

A name in the catalog is a list of signatures and a call has to become one of
them. DuckDB does that by price: every candidate that could take the arguments
at all is charged for the casts it would need, and the cheapest one wins. An
argument that is already the right type is free, so a call that needs no casts
costs nothing and beats everything. See docs/specs/sql/07-functions.md
section 3.

The prices are `casts.mojo`, which is DuckDB's own lattice measured rather than
invented. What is here is the two things that lattice cannot answer, because
neither of them is a cast.

The first is `ANY`, and the single letter templates that behave like it. A
parameter that takes the argument as it stands inserts no cast and so has no
row in the matrix, but it is not free either: `first` has an `ANY` overload and
a `DECIMAL` one and takes the `ANY` one for an integer column, which only says
something if both cost something. `ANY_COST` is what it costs, solved along
with the rest.

A letter is not `ANY` though, and the two cannot cost the same, because `first`
has one of each and DuckDB binds it over any column there is rather than
refusing it as ambiguous. So a letter costs `ANY_COST + 1`. It could as easily
have been one less, since nothing DuckDB does says which way round: the only
name that carries both returns the same type from both and does the same thing
with the argument, so the choice moves which overload `Resolution.at` names and
nothing else. Up rather than down is the safe direction, because `ANY_COST` is
already the cheapest a cast can be and going under it would put a template
level with an exact match.

The second is the containers. `ANY[]`, `T[]` and `MAP(K, V)` are shapes rather
than types, and firepanda's `SqlType` carries no element type yet, so a list
argument against a list parameter is as far as this can look. It matches on the
shape and charges `ANY_COST`, which is right for the case where the elements
already agree and optimistic for the case where they do not. Nothing in tier 1
has two list overloads to choose between, so the optimism costs nothing today,
and the day `SqlType` grows an element type is the day this gets to be exact.

A macro resolves on its argument count alone. It declares parameters and no
types for them, so there is nothing to charge, and DuckDB's own refusal for a
macro says as much by listing parameter names where a function's lists types.

A tie is a case, and DuckDB's answer to one is to refuse the call. `century` at
one argument is the shortest example: a `NULL` costs the same to make an
`INTERVAL` as it does to make a `DATE`, and rather than pick one DuckDB says it
cannot. The generator reads 68 of those refusals off the tier 1 catalog and
they are worth more than the binds are, since the candidate list in a refusal
is the only place DuckDB ever says out loud that two signatures cost the same.
`Resolution` carries the tied overloads and `ambiguity` writes the sentence.

What is not here is what the call's type comes out as. A concrete signature
says, and `registry.spelling(overload.returns)` is it, but a template says `T`
and answering that means substituting what `T` bound to, which is the binder's
job and not the scorer's.
"""

from .cast import common_type
from .casts import Casts
from .generated.casts import ANY_COST, NO_CAST
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
    TYPE_ARRAY,
    TYPE_INVALID,
    TYPE_LIST,
    SqlType,
    type_name,
)


comptime NO_MATCH: Int = -1
"""What a call that fits an overload not at all scores, and what a call that
fits none of them resolves to."""

comptime TEMPLATE_COST: Int = Int(ANY_COST) + 1
"""What a single letter template parameter costs, which is a shade more than
`ANY`, for the reason at the top of this file."""


struct Resolution(Copyable, Movable):
    """Which overload a call resolved to, and what it cost."""

    var at: Int
    """Which overload of the name, counting from the first in catalog order, or
    `NO_MATCH` when none of them fit. When several tied this is the first of
    them, which is not a decision, only somewhere to point."""

    var cost: Int
    """What the casts total, zero when every argument was already the right
    type, and `NO_MATCH` when nothing fit."""

    var tied: List[Int]
    """The overloads that all cost `cost`, in catalog order, for a call DuckDB
    would refuse to decide. Empty when one overload won outright, which is the
    ordinary case."""

    def __init__(out self, at: Int, cost: Int, var tied: List[Int]):
        """One resolution.

        Args:
            at: Which overload, or `NO_MATCH`.
            cost: What it cost.
            tied: The overloads that cost that, or nothing when one won.
        """
        self.at = at
        self.cost = cost
        self.tied = tied^

    def matched(self) -> Bool:
        """Whether the call resolved to anything.

        Returns:
            True when an overload took it.
        """
        return self.at != NO_MATCH

    def ambiguous(self) -> Bool:
        """Whether several overloads cost the same, which DuckDB refuses.

        Returns:
            True when the call has no cheapest overload, in which case
            `ambiguity` has the message to refuse it with.
        """
        return len(self.tied) != 0


def resolve(
    registry: Registry, casts: Casts, at: Int, arguments: List[SqlType]
) raises -> Resolution:
    """Picks the overload of a name that a call costs the least to become.

    Args:
        registry: The catalog.
        casts: The cast lattice.
        at: The name's position, which `registry.find` gives.
        arguments: The argument types, in order.

    Returns:
        The resolution, whose `at` is `NO_MATCH` when no overload fits, which is
        when `registry.no_match` has the sentence to say so, and whose `tied`
        is not empty when several fit equally well, which is when `ambiguity`
        has one instead.

    Raises:
        Error: Never, but the registry lookups it makes can.
    """
    var best = NO_MATCH
    var cost = NO_MATCH
    var tied = List[Int]()
    var overloads = registry.signatures(at)
    for which in range(len(overloads)):
        var total = score(registry, casts, overloads[which], arguments)
        if total == NO_MATCH:
            continue
        if best == NO_MATCH or total < cost:
            best = which
            cost = total
            tied.clear()
        elif total == cost:
            if len(tied) == 0:
                tied.append(best)
            tied.append(which)
    if best != NO_MATCH and not _combines(registry.names[at], arguments):
        return Resolution(NO_MATCH, NO_MATCH, List[Int]())
    return Resolution(best, cost, tied^)


def _combines(name: String, arguments: List[SqlType]) -> Bool:
    """Whether a name that wants one type from all of its arguments gets one.

    A handful of names take `ANY` and then insist the arguments agree with
    each other. `greatest` is the shape: the catalog says `greatest(ANY)` and
    takes any number of them, and DuckDB then refuses `greatest(a, b)` over a
    `TINYINT` and a `VARCHAR` with the same sentence `CASE` gives for the same
    pair, because both of them are asking the lattice the same question. The
    signature cannot say that, so it is said here.

    What firepanda prints when this refuses is the sentence for a call that
    matches no overload, and DuckDB prints the one about combining types.

    Args:
        name: The name the query wrote.
        arguments: The argument types, in order.

    Returns:
        Whether the call is allowed, which is true for every name not in this
        handful.
    """
    if name != "greatest" and name != "least":
        return True
    if len(arguments) == 0:
        return True
    var joined = arguments[0]
    for which in range(1, len(arguments)):
        joined = common_type(joined, arguments[which])
        if joined.id == TYPE_INVALID:
            return False
    return True


def ambiguity(
    registry: Registry,
    name: StringSlice,
    at: Int,
    arguments: List[String],
    resolved: Resolution,
) raises -> String:
    """DuckDB's error for a call with no cheapest overload.

    A different sentence from the one a call that matches nothing gets, and a
    shorter list under it: the overloads that tied and not every overload of
    the name, because the ones that lost are not what the query has to choose
    between.

    The list is not in catalog order either. DuckDB keeps the first overload to
    reach the cheapest price in one place and the ones that later match it in
    another, and writes the second lot out before the first, so the overload a
    reader would expect at the top of the list is at the bottom of it. That is
    the shape of its loop showing through rather than anything meant, and a
    message that puts them the other way round is a message that does not
    match, so this puts them the way DuckDB does.

    Args:
        registry: The catalog.
        name: The name the query wrote.
        at: The name's position, which `registry.find` gives.
        arguments: The argument types, spelled as DuckDB spells them, the same
            way `registry.no_match` wants them.
        resolved: What `resolve` said, whose `tied` is the candidate list.

    Returns:
        The message, candidate list and all.

    Raises:
        Error: Never, but the registry lookups it makes can.
    """
    var written = String()
    for argument in arguments:
        if written.byte_length() != 0:
            written += ", "
        written += argument
    var out = String(
        (
            "Binder Error: Could not choose a best candidate function for the"
            ' function call "'
        ),
        name,
        "(",
        written,
        (
            ')". In order to select one, please add explicit type'
            " casts.\n\tCandidate functions:\n"
        ),
    )
    var overloads = registry.signatures(at)
    for index in range(1, len(resolved.tied)):
        var which = resolved.tied[index]
        out += String(
            "\t", registry.signature_text(name, overloads[which]), "\n"
        )
    if len(resolved.tied) != 0:
        var first = resolved.tied[0]
        out += String(
            "\t", registry.signature_text(name, overloads[first]), "\n"
        )
    return out^


def score(
    registry: Registry,
    casts: Casts,
    overload: Overload,
    arguments: List[SqlType],
) raises -> Int:
    """What one call would cost as one signature.

    Args:
        registry: The catalog.
        casts: The cast lattice.
        overload: The signature to price.
        arguments: The argument types, in order.

    Returns:
        The total, zero for an exact match, and `NO_MATCH` when the signature
        cannot take these arguments at all.

    Raises:
        Error: Never, but the registry lookups it makes can.
    """
    if not overload.accepts(len(arguments)):
        return NO_MATCH
    # A macro has declared no types, so the count was the whole question and it
    # has already been answered.
    if overload.kind == KIND_MACRO:
        return 0
    var total = 0
    for which in range(len(arguments)):
        var price = _slot(
            registry,
            casts,
            registry.parameter(overload, which),
            arguments[which],
        )
        if price == NO_MATCH:
            return NO_MATCH
        total += price
    return total


def _slot(
    registry: Registry, casts: Casts, slot: Int32, argument: SqlType
) raises -> Int:
    """What one argument costs against one parameter.

    Args:
        registry: The catalog, which is where a slot's role is.
        casts: The cast lattice.
        slot: The parameter's type slot.
        argument: The type the call passed.

    Returns:
        The cost, or `NO_MATCH` when the parameter cannot take the argument.

    Raises:
        Error: Never, but the registry lookups it makes can.
    """
    if slot == NO_SLOT:
        return NO_MATCH
    var role = registry.roles[Int(slot)]
    if role == ROLE_ANY:
        return Int(ANY_COST)
    if role == ROLE_TEMPLATE:
        var text = registry.spelling(slot)
        var open = text.find("(")
        if open == -1:
            # A bare letter, `T` in `first(T)`, which takes anything and costs
            # a shade more than `ANY` does for taking it.
            return TEMPLATE_COST
        # A letter inside a container, `MAP(K, V)`, which takes anything of
        # that container's shape and nothing else. The word in front of the
        # bracket is the shape, and it is compared against the argument's own
        # name rather than against a list of containers written out here.
        var shape = text[byte=0:open]
        if shape == type_name(argument.id):
            return Int(ANY_COST)
        return NO_MATCH
    if role == ROLE_LIST:
        # A fixed size array counts, because DuckDB casts one to a list of the
        # same element without being asked.
        if argument.id == TYPE_LIST or argument.id == TYPE_ARRAY:
            return Int(ANY_COST)
        return NO_MATCH
    if role != ROLE_EXACT:
        # A spelling that names no type, which is a macro's parameter name and
        # nothing a function has.
        return NO_MATCH
    var price = casts.cost(argument.id, registry.types[Int(slot)].id)
    if price == NO_CAST:
        return NO_MATCH
    return Int(price)
