"""What an expression's type is, asked of both binders.

The corpus differential next door asks whether a statement parses. This asks the
question after that one, and it is the question the compatibility claim actually
turns on: given columns of known types, what type does an expression over them
come out as. A parser that agrees with DuckDB everywhere and a binder that makes
`sum` a `BIGINT` where DuckDB makes it a `HUGEINT` is a library that returns
wrong answers with no error attached to them, and `typeof()` is in the corpus,
so the difference is not even hidden.

Four things are compared, which between them are every type decision the binder
makes today.

Arithmetic, over every ordered pair in the matrix and all six operators. This is
`firepanda/sql/arith.mojo`, and it is the one with derived results in it rather
than a table lookup: `DECIMAL(4,2) * DECIMAL(5,3)` is a `DECIMAL(9,5)` and
nothing was looked up to get there.

Negation, which is a short list and a different rule from subtraction.

The lattice, through `CASE`, over every ordered pair. This is
`firepanda/sql/cast.mojo`, the type two branches of one expression agree on.

Calls, over the tier 1 catalog. This is `firepanda/sql/registry.mojo` and
`resolve.mojo` together, and it is the resolution fuzzer document 07 asks for:
every name, at every arity it declares up to two, over every combination of a
smaller matrix. Where the winning signature names a concrete return type, the
type is compared. Where it says `ANY`, a template letter or a bare `DECIMAL`,
only the choice between binding and refusing is, because substituting a template
and deriving a decimal's precision are the binder's job and the binder does not
do them yet.

The oracle is `tools/semantics.py`, which builds a table with one column per
type in the matrix and asks DuckDB for `typeof` of each expression over it. The
columns are what keep DuckDB from folding the expression before it can be read,
and the single row in the table is all `typeof` needs, since it is answered
before a value is ever looked at.

Usage:
    pixi run differential-semantics
"""

from std.python import Python, PythonObject

from firepanda.sql import (
    OP_ADD,
    OP_DIVIDE,
    OP_INT_DIVIDE,
    OP_MODULO,
    OP_MULTIPLY,
    OP_SUBTRACT,
    Casts,
    Registry,
    SqlType,
    arithmetic_type,
    common_type,
    decimal,
    negation_type,
    window_only_names,
)
from firepanda.sql.arith import operator_name
from firepanda.sql.generated.functions import KIND_MACRO
from firepanda.sql.registry import NO_SLOT, ROLE_EXACT, Overload
from firepanda.sql.resolve import resolve
from firepanda.sql.types import (
    BLOB,
    BOOLEAN,
    DATE,
    DOUBLE,
    FLOAT,
    HUGEINT,
    INTERVAL,
    TYPE_ARRAY,
    TYPE_DECIMAL,
    TYPE_LIST,
    TYPE_MAP,
    TYPE_STRUCT,
    TYPE_UBIGINT,
    TYPE_UHUGEINT,
    TYPE_UINTEGER,
    TYPE_UNION,
    TYPE_USMALLINT,
    TYPE_UTINYINT,
    TYPE_UUID,
    BIGINT,
    INTEGER,
    SMALLINT,
    TIME,
    TIMESTAMP,
    TIMESTAMP_TZ,
    TINYINT,
    VARCHAR,
)

comptime REFUSED = "!"
"""What DuckDB's answer starts with when it would not bind the expression.

`tools/semantics.py` writes it, and the rest of the line is the first line of
DuckDB's own message, so a disagreement reads without running the query again by
hand.
"""

comptime OURS_REFUSED = ""
"""What firepanda's answer is when it would not bind the expression."""

comptime OURS_UNKNOWN = "*"
"""What firepanda's answer is when it binds the expression and has nothing to
say about the type.

Only the call section produces one. A signature that returns `ANY`, a template
letter or a bare `DECIMAL` has a return type that is a rule over the arguments
rather than a type, and those rules are the binder's and are not written yet. It
counts as agreement against any type DuckDB gives, because what is being
compared there is the choice of overload and not the type.
"""

comptime MOST_ARGUMENTS = 2
"""How many arguments a call is asked about at most.

Three would be 1,728 combinations per name against 144, for a catalog where
three argument overloads are a few dozen out of 802. `tools/gen_casts.py` goes
to three because the costs it solves for need the coverage; this needs the pairs
and would spend an hour on the triples.
"""

comptime SHOWN = 30
"""How many disagreements of each kind to print before summarizing the rest."""

comptime SECTION_ARITHMETIC: UInt8 = 0
"""A binary arithmetic operator over two columns."""

comptime SECTION_NEGATION: UInt8 = 1
"""A unary minus over one column."""

comptime SECTION_LATTICE: UInt8 = 2
"""A `CASE` with two branches, which is the type the two agree on."""

comptime SECTION_CALL: UInt8 = 3
"""A call to a name in the tier 1 catalog."""

comptime SECTION_COUNT: Int = 4
"""How many sections there are."""


def section_name(which: UInt8) -> StaticString:
    """What a section is called in the report.

    Args:
        which: One of the `SECTION_` constants.

    Returns:
        The name.
    """
    if which == SECTION_ARITHMETIC:
        return "arithmetic"
    if which == SECTION_NEGATION:
        return "negation"
    if which == SECTION_LATTICE:
        return "the lattice"
    return "calls"


struct Probe(ImplicitlyCopyable, Movable):
    """One expression, and what firepanda says its type is."""

    var section: UInt8
    """One of the `SECTION_` constants."""

    var expression: String
    """The expression, written over the probe table's columns."""

    var ours: String
    """The type firepanda gives it, `OURS_REFUSED` when it refuses, or
    `OURS_UNKNOWN` when it binds and cannot name the type."""

    def __init__(
        out self, section: UInt8, var expression: String, var ours: String
    ):
        """One probe.

        Args:
            section: Which section it belongs to.
            expression: The expression.
            ours: Firepanda's answer.
        """
        self.section = section
        self.expression = expression^
        self.ours = ours^


def matrix() raises -> List[SqlType]:
    """The types the probe table has a column of.

    Twenty seven, which is every type arithmetic and the lattice have a rule
    for plus the six decimals that make the derived precision rules visible. A
    decimal at six different widths is not thoroughness for its own sake. Every
    ceiling there is sits between two of them: `DECIMAL(18,6) *
    DECIMAL(38,10)` overflows the 38 digit one, `DECIMAL(18,6) +
    DECIMAL(18,6)` is narrowed by the 18 digit one that nobody sees coming,
    `DECIMAL(18,9) * DECIMAL(18,9)` is not narrowed because the product has no
    room left in front of the point, and `DECIMAL(18,18) + DECIMAL(18,6)` is
    narrowed anyway because addition does not ask for that room.

    Returns:
        The types, in the order the columns are declared.

    Raises:
        Error: Never, but the decimal constructor it calls can.
    """
    return [
        BOOLEAN,
        TINYINT,
        SMALLINT,
        INTEGER,
        BIGINT,
        HUGEINT,
        SqlType(TYPE_UTINYINT),
        SqlType(TYPE_USMALLINT),
        SqlType(TYPE_UINTEGER),
        SqlType(TYPE_UBIGINT),
        SqlType(TYPE_UHUGEINT),
        FLOAT,
        DOUBLE,
        decimal(4, 2),
        decimal(9, 4),
        decimal(18, 6),
        decimal(18, 9),
        decimal(18, 18),
        decimal(38, 10),
        VARCHAR,
        BLOB,
        DATE,
        TIME,
        TIMESTAMP,
        TIMESTAMP_TZ,
        INTERVAL,
        SqlType(TYPE_UUID),
    ]


def call_columns(types: List[SqlType]) raises -> List[Int]:
    """Which columns a call is asked about, as positions in the matrix.

    Twelve rather than the whole matrix, because a call section over the matrix
    is 729 combinations per name per arity and this one is 144. What is dropped
    is the types that only differ from one that is kept by width, since the
    interesting thing about a call is which family the argument is in and the
    widths are what the arithmetic section is for.

    They are looked up by name rather than written down as positions, so that
    adding a column to the matrix cannot quietly move this section onto a
    different set of types than the one it says it uses.

    Args:
        types: The matrix.

    Returns:
        The positions.

    Raises:
        Error: If one of them is not in the matrix.
    """
    var wanted = [
        BOOLEAN,
        TINYINT,
        INTEGER,
        BIGINT,
        HUGEINT,
        SqlType(TYPE_UBIGINT),
        DOUBLE,
        decimal(9, 4),
        VARCHAR,
        DATE,
        TIMESTAMP,
        INTERVAL,
    ]
    var out = List[Int]()
    for want in wanted:
        var found = -1
        for at in range(len(types)):
            if types[at].name() == want.name():
                found = at
                break
        if found == -1:
            raise Error(String("the matrix has no ", want.name(), " column"))
        out.append(found)
    return out^


def operators() -> List[UInt8]:
    """The binary arithmetic operators, in the order the report lists them.

    Returns:
        The six.
    """
    return [
        OP_ADD,
        OP_SUBTRACT,
        OP_MULTIPLY,
        OP_DIVIDE,
        OP_INT_DIVIDE,
        OP_MODULO,
    ]


def answer(type: SqlType) -> String:
    """Turns a type the binder gave back into an answer to compare.

    Args:
        type: What the binder said, which is invalid when it refused.

    Returns:
        The type's name, or `OURS_REFUSED`.
    """
    if type.id == 0:
        return String(OURS_REFUSED)
    return type.name()


def arithmetic_probes(types: List[SqlType]) raises -> List[Probe]:
    """Every operator over every ordered pair.

    Args:
        types: The matrix.

    Returns:
        The probes.

    Raises:
        Error: Never, but the binder calls it makes can.
    """
    var out = List[Probe]()
    for operator in operators():
        for left in range(len(types)):
            for right in range(len(types)):
                out.append(
                    Probe(
                        SECTION_ARITHMETIC,
                        String(
                            "c",
                            left,
                            " ",
                            operator_name(operator),
                            " c",
                            right,
                        ),
                        answer(
                            arithmetic_type(operator, types[left], types[right])
                        ),
                    )
                )
    return out^


def negation_probes(types: List[SqlType]) raises -> List[Probe]:
    """A unary minus over every type in the matrix.

    Args:
        types: The matrix.

    Returns:
        The probes.

    Raises:
        Error: Never, but the binder calls it makes can.
    """
    var out = List[Probe]()
    for at in range(len(types)):
        out.append(
            Probe(
                SECTION_NEGATION,
                String("-c", at),
                answer(negation_type(types[at])),
            )
        )
    return out^


def lattice_probes(types: List[SqlType]) raises -> List[Probe]:
    """A two branch `CASE` over every ordered pair.

    The condition is column zero, which is the matrix's boolean. A literal
    condition would let DuckDB fold the whole `CASE` away and answer about one
    branch rather than about both.

    Args:
        types: The matrix.

    Returns:
        The probes.

    Raises:
        Error: Never, but the binder calls it makes can.
    """
    var out = List[Probe]()
    for left in range(len(types)):
        for right in range(len(types)):
            out.append(
                Probe(
                    SECTION_LATTICE,
                    String(
                        "case when c0 then c",
                        left,
                        " else c",
                        right,
                        " end",
                    ),
                    answer(common_type(types[left], types[right])),
                )
            )
    return out^


def returned(registry: Registry, overload: Overload) -> String:
    """What a signature says its result type is, when it says one.

    Args:
        registry: The catalog.
        overload: The winning signature.

    Returns:
        The type's name, or `OURS_UNKNOWN` where the signature names a rule
        rather than a type.
    """
    if overload.kind == KIND_MACRO or overload.returns == NO_SLOT:
        return String(OURS_UNKNOWN)
    var slot = Int(overload.returns)
    if registry.roles[slot] != ROLE_EXACT:
        return String(OURS_UNKNOWN)
    var type = registry.types[slot]
    # A bare `DECIMAL` in the catalog is a promise about the family and not
    # about the width. `sum(DECIMAL(5,2))` is a `DECIMAL(38,2)` and the
    # signature says neither number. A container is the same promise about its
    # elements: `histogram` is declared to return `MAP` and returns
    # `MAP(BOOLEAN, UBIGINT)`, and the two words in there come from the
    # argument and from what the aggregate does, neither of which the catalog
    # writes down.
    if (
        type.id == TYPE_DECIMAL
        or type.id == TYPE_LIST
        or type.id == TYPE_ARRAY
        or type.id == TYPE_STRUCT
        or type.id == TYPE_MAP
        or type.id == TYPE_UNION
    ):
        return String(OURS_UNKNOWN)
    return type.name()


def arities(registry: Registry, at: Int) -> List[Int]:
    """Which argument counts a name is worth asking about.

    Args:
        registry: The catalog.
        at: The name's position.

    Returns:
        Every count up to `MOST_ARGUMENTS` that some overload of the name would
        take, in ascending order.
    """
    var out = List[Int]()
    for count in range(MOST_ARGUMENTS + 1):
        for overload in registry.signatures(at):
            if overload.accepts(count):
                out.append(count)
                break
    return out^


def call_probes(
    registry: Registry, casts: Casts, types: List[SqlType]
) raises -> List[Probe]:
    """Every tier 1 name, at every arity it declares, over the call matrix.

    The thirteen window only names are left out. `row_number()` without an
    `OVER` clause is a refusal about the clause rather than about the overload,
    so asking would compare two binders on a question neither of them is being
    tested for here.

    The macros are left out too, and that one is a real limit rather than a
    tidy one. A macro in DuckDB's catalog is a body and not a signature:
    `date_add` takes its arguments by arity and then binds the expression it
    expands to, so `date_add(c1, c2)` over two integers is accepted here and
    refused by DuckDB with a message about `+` rather than about `date_add`.
    Getting that right means expanding the body, which the registry does not
    carry. Until it does, a macro's arguments are checked by whatever the body
    does with them and this section would only be measuring the gap.

    Args:
        registry: The catalog.
        casts: The cast lattice.
        types: The matrix, which the call columns index into.

    Returns:
        The probes.

    Raises:
        Error: Never, but the binder calls it makes can.
    """
    var windows = window_only_names()
    var columns = call_columns(types)
    var out = List[Probe]()

    for at in range(len(registry)):
        ref name = registry.names[at]
        var window_only = False
        for other in windows:
            if other == name:
                window_only = True
                break
        if window_only:
            continue

        var overloads = registry.signatures(at)
        var macro = False
        for overload in overloads:
            if overload.kind == KIND_MACRO:
                macro = True
                break
        if macro:
            continue

        for count in arities(registry, at):
            var chosen = List[Int](length=count, fill=0)
            while True:
                var arguments = List[SqlType]()
                var written = String(name, "(")
                for slot in range(count):
                    if slot > 0:
                        written += ", "
                    var column = columns[chosen[slot]]
                    written += String("c", column)
                    arguments.append(types[column])
                written += ")"

                var resolved = resolve(registry, casts, at, arguments)
                var ours = String(OURS_REFUSED)
                if resolved.matched() and not resolved.ambiguous():
                    ours = returned(registry, overloads[resolved.at])
                out.append(Probe(SECTION_CALL, written^, ours^))

                var slot = count - 1
                while slot >= 0:
                    chosen[slot] += 1
                    if chosen[slot] < len(columns):
                        break
                    chosen[slot] = 0
                    slot -= 1
                if slot < 0:
                    break
    return out^


def ask_duckdb(
    types: List[SqlType], probes: List[Probe]
) raises -> List[String]:
    """Asks DuckDB for the type of every probe at once.

    One call across the whole batch rather than one call each, because the
    bridge costs more per crossing than DuckDB costs per bind.

    Args:
        types: The matrix, which becomes the probe table's columns.
        probes: The expressions.

    Returns:
        One answer per probe, in order.

    Raises:
        Error: If the helper could not be reached, or answered the wrong number
            of times.
    """
    var columns = Python.list()
    for type in types:
        columns.append(PythonObject(type.name()))
    var batch = Python.list()
    for probe in probes:
        batch.append(PythonObject(probe.expression))

    var python_path = Python.import_module("sys").path
    python_path.insert(0, "tools")
    var helper = Python.import_module("semantics")

    var out = List[String]()
    for line in String(helper.types_of(columns, batch)).split("\n"):
        if line.byte_length() != 0:
            out.append(String(line))
    if len(out) != len(probes):
        raise Error(
            String(
                "DuckDB answered about ",
                len(out),
                " expressions and there are ",
                len(probes),
            )
        )
    return out^


def known(probe: Probe, theirs: StringSlice) -> Bool:
    """Says whether a disagreement is one this repository has already decided.

    Every entry is a case where firepanda's answer is the one it means to give
    and DuckDB's is not something to copy, or where the difference is about
    something other than the type. They are listed here rather than left in the
    count so that the ceilings can stay at zero and mean something.

    Args:
        probe: The expression and firepanda's answer.
        theirs: DuckDB's answer.

    Returns:
        Whether this is one of them.
    """
    # A function whose argument has to be a constant is refused over a column
    # whatever its type, so what comes back says nothing about the overload.
    # `quantile(x, 0.5)` is the shape, and the message names the requirement.
    if theirs.find("constant") != -1:
        return True
    # The same name at one argument, where the requirement is that the second
    # one exists at all. The catalog declares a one argument `quantile` and
    # DuckDB's bind function then asks for the quantile to take, so the
    # refusal is about the argument that is not there rather than about the
    # one that is.
    if theirs.find("QUANTILE requires") != -1:
        return True
    # A name DuckDB publishes in the catalog and then refuses on sight.
    # `sum_no_overflow` is the only one in tier 1, and nothing in the catalog
    # marks it, so a resolver reading the catalog binds it. Refusing it would
    # mean keeping a list of names DuckDB will not let anybody call, which is
    # a list worth having the day it has a second entry.
    if theirs.find("internal use only") != -1:
        return True
    # A name the grammar spells with a keyword rather than a comma.
    # `position(a in b)` is how DuckDB writes it, so `position(a, b)` never
    # reaches a binder there and what comes back is the parser talking. The
    # corpus differential is where the grammar is compared.
    if theirs.find("Parser Error") != -1:
        return True
    return False


def report(
    kind: StringSlice, probes: List[Probe], answers: List[String], of: Int
) -> None:
    """Prints one kind of disagreement, broken down by section.

    Args:
        kind: What this list is, for the heading.
        probes: The probes that disagreed this way.
        answers: What DuckDB said about each, parallel to it.
        of: How many probes were compared in total.
    """
    if len(probes) == 0:
        return

    print()
    print(kind, len(probes), "of", of)

    var counts = List[Int](length=SECTION_COUNT, fill=0)
    for probe in probes:
        counts[Int(probe.section)] += 1
    for which in range(SECTION_COUNT):
        if counts[which] != 0:
            print("   ", counts[which], "in", section_name(UInt8(which)))

    print()
    for at in range(min(SHOWN, len(probes))):
        ref probe = probes[at]
        var ours = probe.ours
        if ours == OURS_REFUSED:
            ours = String("refused")
        print(
            "   ",
            probe.expression,
            "  firepanda",
            ours,
            "  DuckDB",
            answers[at],
        )
    if len(probes) > SHOWN:
        print("   ", len(probes) - SHOWN, "more")


def main() raises:
    var types = matrix()
    var registry = Registry()
    var casts = Casts()

    var probes = List[Probe]()
    probes.extend(arithmetic_probes(types))
    probes.extend(negation_probes(types))
    probes.extend(lattice_probes(types))
    probes.extend(call_probes(registry, casts, types))
    print(
        "asking DuckDB about",
        len(probes),
        "expressions over",
        len(types),
        "columns",
    )

    var answers = ask_duckdb(types, probes)

    var explained = 0
    var wrong_type = List[Probe]()
    var wrong_type_answers = List[String]()
    var we_bind = List[Probe]()
    var we_bind_answers = List[String]()
    var we_refuse = List[Probe]()
    var we_refuse_answers = List[String]()

    for at in range(len(probes)):
        ref probe = probes[at]
        ref theirs = answers[at]
        var they_refused = theirs.startswith(REFUSED)
        var we_refused = probe.ours == OURS_REFUSED

        if we_refused and they_refused:
            continue
        if not we_refused and not they_refused:
            if probe.ours == OURS_UNKNOWN or probe.ours == theirs:
                continue

        if known(probe, theirs):
            explained += 1
            continue

        if we_refused:
            we_refuse.append(probe)
            we_refuse_answers.append(theirs)
        elif they_refused:
            we_bind.append(probe)
            we_bind_answers.append(theirs)
        else:
            wrong_type.append(probe)
            wrong_type_answers.append(theirs)

    var disagreements = len(wrong_type) + len(we_bind) + len(we_refuse)
    print(
        "compared",
        len(probes),
        "expressions,",
        explained,
        "were known cases",
    )

    report(
        "these come out a different type:",
        wrong_type,
        wrong_type_answers,
        len(probes),
    )
    report(
        "firepanda binds these and DuckDB does not:",
        we_bind,
        we_bind_answers,
        len(probes),
    )
    report(
        "DuckDB binds these and firepanda does not:",
        we_refuse,
        we_refuse_answers,
        len(probes),
    )

    print()
    print(
        "agreement",
        (len(probes) - disagreements) * 10000 // len(probes),
        "in ten thousand,",
        disagreements,
        "disagreements",
    )

    if disagreements != 0:
        raise Error(
            String(
                "firepanda and DuckDB disagree about the type of ",
                disagreements,
                " expressions, against a ceiling of 0",
            )
        )
