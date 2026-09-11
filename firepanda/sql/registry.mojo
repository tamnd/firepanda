"""The tier 1 function catalog: what a name is, and what its overloads are.

Read once out of the generated table into a flat set of lists, looked up by a
binary search over sorted names. The table is DuckDB's own catalog rather than
a set of signatures somebody typed in, which matters because the signatures
that would be wrong are the ones nobody would check. `sum(BOOLEAN)` gives back
a `HUGEINT`, `min` has a second overload that takes a count and gives back a
list, `length` accepts a `BIT`, and `round(DECIMAL)` keeps the decimal rather
than going to a double. See docs/specs/sql/07-functions.md section 2.

Document 07 says the table should be built at comptime. It is one `StaticString`
read once instead, for the reason `rules.mojo` is: a literal holding 802 structs
is compile time that every build of firepanda pays. The read costs 390 us
against the grammar table's 300 on the same machine, and it happens once rather
than per query, so it is nowhere near document 01's budget, which is a warm
statement end to end. The cost that document was arguing against is a dictionary
rebuilt in every process, and this is not one.

What is here is the table and the questions it can answer on its own: does a
name exist, what is it, which overloads does it have, and what does the message
say when the answer is no. What is not here is picking one overload out of
several, because that is a scoring problem over the cast lattice and it is the
next thing in document 07.

Three names document 07 lists are not in the catalog at all and are not here.
`coalesce` and `ifnull` are grammar rules, which is why the wrong number of
arguments to either one comes back as `Parser Error: Wrong number of arguments
to IFNULL.` rather than as a binder error, and `current_timestamp` is a keyword.
All three belong to the transformer.

A name and its aliases both carry the whole overload list, because DuckDB's
catalog does and because the candidate list in an error message prints the name
the query wrote. Asking about `substr` gets `substr(VARCHAR, BIGINT) -> VARCHAR`
and not `substring`'s spelling of it.

Three shapes of refusal, and they are three different sentences rather than one
with the name swapped. An unknown name is a catalog error that says `Scalar
Function` whether the name resembles a scalar function or an aggregate. A call
that matches no overload is a binder error listing every candidate for the name,
in catalog order. A call to a macro is a third sentence again, listing the
macro's parameter names rather than any types, because a macro has none.
"""

from .catalog import NOT_FOUND, edit_distance, fold
from .generated.functions import (
    KIND_AGGREGATE,
    KIND_MACRO,
    KIND_SCALAR,
    NAME_COUNT,
    OVERLOAD_COUNT,
    TABLE,
    TYPE_COUNT,
)
from .types import INVALID, SqlType, parse_type


comptime NO_SLOT: Int32 = -1
"""What a signature writes where it has no type, which is every slot of a macro.
"""

comptime ROLE_EXACT: UInt8 = 0
"""A concrete type, which is most of them."""

comptime ROLE_ANY: UInt8 = 1
"""`ANY`, which every type matches."""

comptime ROLE_TEMPLATE: UInt8 = 2
"""A one letter name, or a spelling holding one, which binds to whatever the
call passes and has to agree with the other slots that share the letter."""

comptime ROLE_LIST: UInt8 = 3
"""A spelling ending in `[]`, whose element is the role beside it."""

comptime ROLE_UNKNOWN: UInt8 = 4
"""A spelling that names no type, which is a macro's parameter name and nothing
else."""


struct Overload(Copyable, ImplicitlyCopyable, Movable):
    """One signature of one name."""

    var kind: UInt8
    """One of the `KIND_` constants."""

    var returns: Int32
    """The return type's slot, or `NO_SLOT` for a macro."""

    var varargs: Int32
    """The trailing type's slot, or `NO_SLOT` when the arity is fixed."""

    var first: UInt32
    """Where its parameter slots start in the registry's flat list."""

    var arity: UInt32
    """How many parameters it declares, before any varargs."""

    def __init__(
        out self,
        kind: UInt8,
        returns: Int32,
        varargs: Int32,
        first: UInt32,
        arity: UInt32,
    ):
        """One signature.

        Args:
            kind: One of the `KIND_` constants.
            returns: The return type's slot.
            varargs: The trailing type's slot, or `NO_SLOT`.
            first: Where its parameters start.
            arity: How many it declares.
        """
        self.kind = kind
        self.returns = returns
        self.varargs = varargs
        self.first = first
        self.arity = arity

    def variadic(self) -> Bool:
        """Whether it takes any number of trailing arguments.

        Returns:
            True when it declares a varargs type.
        """
        return self.varargs != NO_SLOT

    def accepts(self, count: Int) -> Bool:
        """Whether an argument count could be this overload's.

        A macro is included, because a macro declares its parameters even
        though it declares no types for them, so the count is the one thing
        about a macro that can be checked here.

        Args:
            count: How many arguments the call wrote.

        Returns:
            Whether the count is possible. It says nothing about the types.
        """
        if self.variadic():
            return count >= Int(self.arity)
        return count == Int(self.arity)


struct Registry(Movable, Sized):
    """The whole tier 1 catalog, read out of the generated table."""

    var names: List[String]
    """Every name, sorted, which is what makes a lookup a binary search."""

    var kinds: List[UInt8]
    """What each name is, one of the `KIND_` constants."""

    var firsts: List[UInt32]
    """Where each name's overloads start."""

    var counts: List[UInt32]
    """How many overloads each name has."""

    var aliases: List[String]
    """The name each one is an alias of, or an empty string."""

    var spellings: List[String]
    """Every distinct type spelling, DuckDB's own words."""

    var roles: List[UInt8]
    """What each spelling is, one of the `ROLE_` constants."""

    var types: List[SqlType]
    """The parsed type of each spelling, invalid unless the role is exact."""

    var elements: List[UInt8]
    """For a list spelling, the role of its element, and `ROLE_UNKNOWN`
    otherwise."""

    var element_types: List[SqlType]
    """For a list spelling of a concrete element, that element's type."""

    var overloads: List[Overload]
    """Every signature, in catalog order within each name."""

    var parameters: List[Int32]
    """Every signature's parameter slots, run together."""

    def __init__(out self) raises:
        """Reads the generated table.

        Raises:
            Error: If the table does not say what its header says it does,
                which would mean the generator and this reader disagree.
        """
        self.names = List[String]()
        self.kinds = List[UInt8]()
        self.firsts = List[UInt32]()
        self.counts = List[UInt32]()
        self.aliases = List[String]()
        self.spellings = List[String]()
        self.roles = List[UInt8]()
        self.types = List[SqlType]()
        self.elements = List[UInt8]()
        self.element_types = List[SqlType]()
        self.overloads = List[Overload]()
        self.parameters = List[Int32]()

        var lines = List[String]()
        for line in TABLE.split("\n"):
            if line.byte_length() != 0:
                lines.append(String(line))
        if len(lines) == 0:
            raise Error("the function table is empty")

        var header = _fields(lines[0])
        if len(header) != 4 or header[0] != "F":
            raise Error("the function table has no header")
        var names = _number(header[1])
        var overloads = _number(header[2])
        var types = _number(header[3])
        if (
            names != NAME_COUNT
            or overloads != OVERLOAD_COUNT
            or types != TYPE_COUNT
        ):
            raise Error("the function table disagrees with its own counts")
        if len(lines) != 1 + types + names + overloads:
            raise Error("the function table is not as long as it says")

        for at in range(1, 1 + types):
            self._add_spelling(lines[at])

        var last = String()
        for at in range(1 + types, 1 + types + names):
            var fields = _fields(lines[at])
            if len(fields) != 5:
                raise Error("a name line with the wrong number of fields")
            if len(self.names) != 0 and fields[0] <= last:
                raise Error("the function table's names are not sorted")
            last = String(fields[0])
            self.names.append(String(fields[0]))
            self.kinds.append(UInt8(_number(fields[1])))
            self.firsts.append(UInt32(_number(fields[2])))
            self.counts.append(UInt32(_number(fields[3])))
            self.aliases.append(
                String() if fields[4] == "-" else String(fields[4])
            )

        for at in range(1 + types + names, len(lines)):
            var fields = _fields(lines[at])
            if len(fields) < 4:
                raise Error("an overload line with too few fields")
            var arity = _number(fields[3])
            if len(fields) != 4 + arity:
                raise Error("an overload line that miscounts its parameters")
            self.overloads.append(
                Overload(
                    UInt8(_number(fields[0])),
                    Int32(_signed(fields[1])),
                    Int32(_signed(fields[2])),
                    UInt32(len(self.parameters)),
                    UInt32(arity),
                )
            )
            for slot in range(4, len(fields)):
                self.parameters.append(Int32(_signed(fields[slot])))

    def __len__(self) -> Int:
        """How many names it holds.

        Returns:
            The count, aliases included.
        """
        return len(self.names)

    def find(self, name: StringSlice) -> Int:
        """Looks a name up.

        Args:
            name: The name the query wrote.

        Returns:
            Its position, or `NOT_FOUND`.
        """
        var probe = fold(name)
        var low = 0
        var high = len(self.names)
        while low < high:
            var middle = (low + high) // 2
            if self.names[middle] < probe:
                low = middle + 1
            elif self.names[middle] > probe:
                high = middle
            else:
                return middle
        return NOT_FOUND

    def contains(self, name: StringSlice) -> Bool:
        """Whether a name is one the registry knows.

        Args:
            name: The name the query wrote.

        Returns:
            Whether it is there.
        """
        return self.find(name) != NOT_FOUND

    def kind_of(self, at: Int) -> UInt8:
        """What a name is.

        Args:
            at: The name's position.

        Returns:
            One of the `KIND_` constants.
        """
        return self.kinds[at]

    def is_aggregate(self, name: StringSlice) -> Bool:
        """Whether a name is an aggregate.

        `classify.mojo` answers the same question off a list of names, because
        it runs over an expression before anything has read a catalog. This is
        the same answer from the table it was copied out of, and a test holds
        the two together.

        Args:
            name: The name the query wrote.

        Returns:
            Whether the catalog calls it an aggregate.
        """
        var at = self.find(name)
        return at != NOT_FOUND and self.kinds[at] == KIND_AGGREGATE

    def signatures(self, at: Int) -> List[Overload]:
        """Every overload of one name, in catalog order.

        Args:
            at: The name's position.

        Returns:
            The overloads.
        """
        var out = List[Overload]()
        var first = Int(self.firsts[at])
        for offset in range(Int(self.counts[at])):
            out.append(self.overloads[first + offset])
        return out^

    def parameter(self, overload: Overload, index: Int) -> Int32:
        """One parameter slot of one overload.

        Args:
            overload: The signature.
            index: Which parameter, from 0.

        Returns:
            The slot, or `NO_SLOT` when the index is past the declared ones and
            the overload is not variadic.
        """
        if index < Int(overload.arity):
            return self.parameters[Int(overload.first) + index]
        return overload.varargs

    def spelling(self, slot: Int32) -> String:
        """How DuckDB writes one type.

        Args:
            slot: The slot, which may be `NO_SLOT`.

        Returns:
            The spelling, or an empty string for `NO_SLOT`.
        """
        if slot == NO_SLOT:
            return String()
        return self.spellings[Int(slot)]

    def nearest(self, name: StringSlice) -> Int:
        """The known name closest to one that is not known.

        The threshold is `catalog.mojo`'s, an edit distance below half the
        shorter name, for the same reason: below it is a typo and above it is a
        different word. DuckDB has no threshold at all and will answer a name
        that resembles nothing with whatever sorted nearest, which is how
        `zzzzzzqq` gets suggested `finalize`. That is not worth copying.

        Args:
            name: The name the query wrote.

        Returns:
            The position of the closest name, or `NOT_FOUND`.
        """
        var probe = fold(name)
        var best = NOT_FOUND
        var best_distance = 0
        for at in range(len(self.names)):
            var candidate = self.names[at]
            var shorter = min(probe.byte_length(), candidate.byte_length())
            var limit = shorter // 2 + 1
            var distance = edit_distance(probe, candidate, limit)
            if distance >= limit:
                continue
            if best == NOT_FOUND or distance < best_distance:
                best = at
                best_distance = distance
        return best

    def unknown(self, name: StringSlice) -> String:
        """DuckDB's error for a name that is in no catalog.

        It says `Scalar Function` whatever the name looks like, so a misspelled
        aggregate is reported as a missing scalar function. That is DuckDB's
        and it is kept.

        Args:
            name: The name the query wrote.

        Returns:
            The message, with a `Did you mean` line when something is close.
        """
        var message = String(
            "Catalog Error: Scalar Function with name ",
            name,
            " does not exist!",
        )
        var near = self.nearest(name)
        if near == NOT_FOUND:
            return message
        return String(message, '\nDid you mean "', self.names[near], '"?')

    def signature_text(self, name: StringSlice, overload: Overload) -> String:
        """One candidate line, the way DuckDB prints it.

        A variadic writes its trailing type in brackets with an ellipsis, so
        `concat` reads `concat(ANY, [ANY...]) -> ANY`. A macro has no types and
        writes its parameter names instead, and no return at all.

        Args:
            name: The name the query wrote, since an alias prints as itself.
            overload: The signature.

        Returns:
            The line, with no leading tab and no newline.
        """
        var out = String(name, "(")
        for at in range(Int(overload.arity)):
            if at > 0:
                out += ", "
            out += self.spelling(self.parameters[Int(overload.first) + at])
        if overload.variadic():
            if overload.arity > 0:
                out += ", "
            out += String("[", self.spelling(overload.varargs), "...]")
        out += ")"
        if overload.kind == KIND_MACRO:
            return out
        return String(out, " -> ", self.spelling(overload.returns))

    def no_match(
        self, name: StringSlice, at: Int, arguments: List[String]
    ) -> String:
        """DuckDB's error for a call that matches no overload of a known name.

        Args:
            name: The name the query wrote.
            at: The name's position.
            arguments: The argument types, spelled as DuckDB spells them, which
                for an untyped literal is `STRING_LITERAL` or `INTEGER_LITERAL`
                rather than a type.

        Returns:
            The message, candidate list and all.
        """
        if self.kinds[at] == KIND_MACRO:
            return self._no_macro_match(name, at)
        var written = String()
        for argument in arguments:
            if written.byte_length() != 0:
                written += ", "
            written += argument
        var out = String(
            (
                "Binder Error: No function matches the given name and argument"
                " types '"
            ),
            name,
            "(",
            written,
            (
                ")'. You might need to add explicit type casts.\n\tCandidate"
                " functions:\n"
            ),
        )
        for overload in self.signatures(at):
            out += String("\t", self.signature_text(name, overload), "\n")
        return out

    def _no_macro_match(self, name: StringSlice, at: Int) -> String:
        """DuckDB's error for a call to a macro that does not fit.

        A different sentence from the one a function gets, with the candidates
        under a heading that carries no leading tab and with no trailing blank
        line. Three tier 1 names are macros: `date_add`, `nullif` and
        `split_part`.

        Args:
            name: The name the query wrote.
            at: The name's position.

        Returns:
            The message.
        """
        var out = String(
            "Binder Error: Macro ",
            name,
            (
                "() does not support the supplied arguments. You might need to"
                " add explicit type casts.\nCandidate macros:"
            ),
        )
        for overload in self.signatures(at):
            out += String("\n\t", self.signature_text(name, overload))
        return out

    def _add_spelling(mut self, text: StringSlice) raises:
        """Classifies one type spelling and records it.

        Args:
            text: The spelling, DuckDB's own words.

        Raises:
            Error: Never, but the type parse it calls can.
        """
        self.spellings.append(String(text))
        var element = text
        var list = text.endswith("[]")
        if list:
            element = text[byte = 0 : text.byte_length() - 2]
        var role = _role(element)
        var parsed = INVALID
        if role == ROLE_EXACT:
            parsed = parse_type(element)
        if list:
            self.roles.append(ROLE_LIST)
            self.types.append(INVALID)
            self.elements.append(role)
            self.element_types.append(parsed)
            return
        self.roles.append(role)
        self.types.append(parsed)
        self.elements.append(ROLE_UNKNOWN)
        self.element_types.append(INVALID)


def _role(text: StringSlice) raises -> UInt8:
    """What one spelling is, ignoring any `[]` already taken off it.

    Args:
        text: The spelling.

    Returns:
        One of the `ROLE_` constants, never `ROLE_LIST`.

    Raises:
        Error: Never, but it is written as a raising function because the type
            parse it uses to decide can raise and is caught here.
    """
    if text == "ANY":
        return ROLE_ANY
    if text.byte_length() == 1:
        return ROLE_TEMPLATE
    if (
        text.find("(") != -1
        and (text.find("K") != -1 or text.find("V") != -1)
        and text.find('"') == -1
    ):
        # `MAP(K, V)` is a container over two template letters, which is a
        # different thing from `STRUCT("year" BIGINT, ...)` where the letters
        # are part of a column name.
        return ROLE_TEMPLATE
    try:
        _ = parse_type(text)
        return ROLE_EXACT
    except:
        return ROLE_UNKNOWN


def _fields(line: StringSlice) -> List[String]:
    """Splits one table line on single spaces.

    Args:
        line: The line, with no newline on it.

    Returns:
        The fields. A spelling holding a space comes back as several, which is
        why only the sections with a fixed field count are read this way.
    """
    var out = List[String]()
    for piece in line.split(" "):
        out.append(String(piece))
    return out^


def _number(text: StringSlice) -> Int:
    """Reads one non negative integer.

    Args:
        text: The digits.

    Returns:
        The value, or 0 for anything that is not digits.
    """
    var value = 0
    for digit in text.as_bytes():
        if digit < 48 or digit > 57:
            return 0
        value = value * 10 + Int(digit - 48)
    return value


def _signed(text: StringSlice) -> Int:
    """Reads one integer, which the table writes as -1 where it has no value.

    Args:
        text: The digits, possibly with a leading minus.

    Returns:
        The value.
    """
    if text.startswith("-"):
        return -_number(text[byte=1:])
    return _number(text)
