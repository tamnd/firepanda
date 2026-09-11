"""What an implicit cast costs, which is how one overload beats another.

DuckDB picks between the overloads of a name by totalling what it would cost to
cast each argument to each candidate and taking the cheapest. This is the table
those totals are read out of: for every pair of scalar types, whether an
implicit cast exists at all and what it is worth. See
docs/specs/sql/07-functions.md section 3.

The numbers are not DuckDB's own, because DuckDB publishes none. They are the
answer to the inequalities its choices imply, solved by tools/gen_casts.py and
checked by replaying every one of the 2300 decisions it collected. What matters
is not that they are the same numbers but that they order the same way, and the
ordering is what the generator checks on every CI run.

A cost belongs to the target type alone, which was a guess when this started and
is now measured: one number per target satisfies every inequality and a number
per pair of types buys nothing. Whether the cast exists at all is a property of
the pair and there is no guessing it. `UTINYINT` reaches `SMALLINT` and not
`TINYINT`, `TINYINT` reaches no unsigned type and not `VARCHAR` either, and
`VARIANT` reaches every type there is while only `NULL` reaches `VARIANT`.

Scalar types only, the same line firepanda/sql/types.mojo draws. A list, an
array, a struct, a map and a union cast by their elements, and firepanda's
`SqlType` carries no element type yet, so there is nothing here to say about
them and `covers` says so.

What is not here is the scoring itself, which is `resolve.mojo`, because a
signature can say `ANY` or a template letter and neither of those is a cast.
"""

from .generated.casts import CAST_COUNT, NO_CAST, SCALAR_COUNT, TABLE
from .types import TYPE_COUNT, type_name


struct Casts(Movable, Sized):
    """The implicit cast lattice, read out of the generated table."""

    var costs: List[Int16]
    """One entry per ordered pair of type identifiers, target major, so the
    cost of casting `source` to `target` is at `target * TYPE_COUNT + source`.
    A pair with no implicit cast holds `NO_CAST` and a type paired with itself
    holds zero."""

    var count: Int
    """How many identifiers the table speaks for, which is the scalars."""

    def __init__(out self) raises:
        """Reads the generated table.

        Raises:
            Error: If the table does not say what its header says it does, or
                names a type the identifier it sits on does not mean, which
                would mean the generator and types.mojo have drifted apart.
        """
        self.costs = List[Int16]()
        self.count = 0

        var lines = List[String]()
        for line in TABLE.split("\n"):
            if line.byte_length() != 0:
                lines.append(String(line))
        if len(lines) == 0:
            raise Error("the cast table is empty")

        var width = Int(TYPE_COUNT)
        if lines[0] != String("X ", width):
            raise Error("the cast table is not as wide as types.mojo is")
        if len(lines) != 1 + width:
            raise Error("the cast table has a line per identifier or it is not")

        var covered = List[Bool](length=width, fill=False)
        for identifier in range(width):
            var name = _read(lines[1 + identifier], width, self.costs)
            if name == "-":
                continue
            if name != String(type_name(UInt8(identifier))):
                raise Error(
                    String(
                        "the cast table calls identifier ",
                        identifier,
                        " ",
                        name,
                        " and types.mojo calls it ",
                        type_name(UInt8(identifier)),
                    )
                )
            covered[identifier] = True
            self.count += 1
        if self.count != SCALAR_COUNT:
            raise Error("the cast table disagrees with its own scalar count")

        # A type the table does not speak for is neither a source nor a target,
        # so its row and its column are both refusals. Without this the two
        # halves of `covers` could disagree and a caller would have to know
        # which one it was asking about.
        var edges = 0
        for target in range(width):
            for source in range(width):
                var price = self.costs[target * width + source]
                if not covered[target] or not covered[source]:
                    if price != NO_CAST:
                        raise Error(
                            "the cast table prices a type it does not cover"
                        )
                    continue
                if (target == source) != (price == 0):
                    raise Error("the cast table has no zero on its diagonal")
                if price > 0:
                    edges += 1
        if edges != CAST_COUNT:
            raise Error("the cast table disagrees with its own cast count")

    def __len__(self) -> Int:
        """How many types it speaks for.

        Returns:
            The count, which is the scalar types and nothing else.
        """
        return self.count

    def covers(self, id: UInt8) -> Bool:
        """Whether the table says anything about a type.

        Args:
            id: One of the `TYPE_` constants.

        Returns:
            Whether it is a scalar the table holds a row for. A list, a struct
            and the rest are not, and neither is `TYPE_INVALID`.
        """
        if id >= TYPE_COUNT:
            return False
        return self.costs[Int(id) * Int(TYPE_COUNT) + Int(id)] == 0

    def cost(self, source: UInt8, target: UInt8) -> Int16:
        """What it costs to pass a `source` value where a `target` is wanted.

        A decimal is one row here whatever its width, because DuckDB's decimal
        parameter is a template that takes the width of the argument, so the
        answer for one pair of decimals is the answer for every pair.

        Args:
            source: The type the argument has.
            target: The type the parameter wants.

        Returns:
            The cost, zero when the two are the same type, and `NO_CAST` when
            there is no implicit cast between them or either one is a type the
            table does not cover.
        """
        if source >= TYPE_COUNT or target >= TYPE_COUNT:
            return NO_CAST
        return self.costs[Int(target) * Int(TYPE_COUNT) + Int(source)]

    def reaches(self, source: UInt8, target: UInt8) -> Bool:
        """Whether a `source` value can be passed where a `target` is wanted.

        Args:
            source: The type the argument has.
            target: The type the parameter wants.

        Returns:
            Whether an implicit cast exists, which a type has to itself.
        """
        return self.cost(source, target) != NO_CAST


def _read(
    line: StringSlice, width: Int, mut costs: List[Int16]
) raises -> String:
    """Reads one line of the table: a name, and then `width` numbers.

    The name is not what comes before the first space, because three of them
    hold one and `TIME WITH TIME ZONE` is a type rather than four fields. The
    numbers are a fixed count, so the line is read from its end instead.

    Args:
        line: The line, with no newline on it.
        width: How many numbers to expect.
        costs: The list to append them to.

    Returns:
        The name, which is `-` for an identifier the table does not cover.

    Raises:
        Error: If the line is too short or holds a field that is not a number.
    """
    var pieces = List[String]()
    for piece in line.split(" "):
        pieces.append(String(piece))
    if len(pieces) < width + 1:
        raise Error("a cast table line with too few fields")

    var first = len(pieces) - width
    var name = String()
    for at in range(first):
        if at > 0:
            name += " "
        name += pieces[at]
    for at in range(first, len(pieces)):
        costs.append(_value(pieces[at]))
    return name^


def _value(text: StringSlice) raises -> Int16:
    """Reads one number, which the table writes as -1 where it has no cast.

    Args:
        text: The digits, with a leading minus or without one.

    Returns:
        The value.

    Raises:
        Error: If it is not a number, which would mean the line was read at the
            wrong offset and every number after it is somebody else's.
    """
    var negative = text.startswith("-")
    var digits = text[byte = 1 : text.byte_length()] if negative else text
    if digits.byte_length() == 0:
        raise Error("a cast table field with no digits in it")
    var value = 0
    for digit in digits.as_bytes():
        if digit < 48 or digit > 57:
            raise Error(
                String("a cast table field that is not a number: ", text)
            )
        value = value * 10 + Int(digit - 48)
    return Int16(-value) if negative else Int16(value)
