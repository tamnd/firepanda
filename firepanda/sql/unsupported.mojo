"""The refusal table, and the shape a refusal takes.

The grammar accepts everything DuckDB accepts and the engine executes rather
less than that, so there is a line somewhere and this file is where it is
written down. A query on the far side of it gets a refusal, which is neither a
syntax error nor a silent no-op, because both of those leave the user with no
way to find out what happened. See docs/specs/sql/05-ast-and-binder.md section
4.

A refusal has four parts and all four are mandatory:

    Not Implemented Error: firepanda does not support a slice or a subscript.
    LINE 1: SELECT a[1] FROM t
                    ^
    firepanda reads a list element with a function rather than with brackets.
    See https://github.com/tamnd/firepanda/issues/13

The feature by name, the position with a caret under it, what firepanda is
instead, and where to go next. The third is the one that gets dropped and it is
the one that stops the user filing the bug.

They are entries in one table rather than a `raise` wherever the transformer
happens to run out of cases, which is what makes the set enumerable.
`sql_support()` returns it, so a user can ask what firepanda does not do without
reading the source, and the conformance harness can tell a corpus file that
failed for a refusal from one that failed for a crash. The two are very
different failures and a harness that cannot separate them reports the wrong
number.

A message may hold one `{}`, which is where the text from the query goes, so
`{} on a call` refuses `ORDER BY` and `FILTER` in the same entry without either
one losing its name.

An entry leaves the table when the feature lands, and the constants after it
move down by one. The numbers are an index into this list and nothing outside
this file keeps one, so what a caller matches on is the feature name, which is
why the name is the part that may not change.
"""


@fieldwise_init
struct Refusal(Copyable, ImplicitlyCopyable, Movable):
    """One thing firepanda does not do, and what it says about it."""

    var feature: StaticString
    """A stable name for the thing, for a caller matching on it.

    Not shown to the user. It is what a test or a harness compares against, so
    it may not change once it is in a release even if the message does.
    """

    var message: StaticString
    """What follows `firepanda does not support` in the error.

    Holds at most one `{}`, filled with the text from the query.
    """

    var explanation: StaticString
    """What firepanda is instead, in one or two sentences."""

    var issue: UInt32
    """The issue to read, or to file against."""

    def rendered(self) -> String:
        """The entry as `sql_support` lists it.

        Returns:
            The message, the explanation and the link, on three lines.
        """
        return String(
            self.message,
            "\n    ",
            self.explanation,
            "\n    ",
            issue_link(self.issue),
        )


comptime SQL_ISSUE: UInt32 = 13
"""The SQL tracking issue, for a feature with no nearer home."""

comptime STAGE_ISSUE: UInt32 = 304
"""The dialect milestone, for a feature that is coming and has not landed."""


comptime OPERATOR: UInt16 = 0
"""An infix operator the expression arena has no kind for."""

comptime CUSTOM_OPERATOR: UInt16 = 1
"""`OPERATOR(...)` written in front of an operand."""

comptime LIKE_ESCAPE: UInt16 = 2
"""`ESCAPE` after an operator with no escaping form, which is `SIMILAR TO` and
the regular expression spellings. A `LIKE` and an `ILIKE` become a call."""

comptime FIELD_ACCESS: UInt16 = 3
"""`.name` after something that is not a name."""

comptime SUBSCRIPT: UInt16 = 4
"""`x[1]` and `x[1:2]`."""

comptime POSTFIX_OPERATOR: UInt16 = 5
"""An operator written after its operand."""

comptime CALL_MODIFIER: UInt16 = 6
"""`WITHIN GROUP` or `EXPORT_STATE` after a call."""

comptime CALL_ARGUMENT: UInt16 = 7
"""`ORDER BY` or a null treatment inside a call."""

comptime ARRAY_SUBQUERY: UInt16 = 8
"""`ARRAY(SELECT ...)`."""

comptime DOTTED_NAME: UInt16 = 9
"""A qualified name where only a plain one fits."""

comptime QUOTED_NAME: UInt16 = 10
"""Anything but a bare name where only a bare name fits."""

comptime NOT_SUBQUERY: UInt16 = 11
"""`NOT (SELECT ...)` where a subquery is the whole operand."""

comptime SELECT_CLAUSE: UInt16 = 12
"""A clause on a `SELECT` that the query node has no slot for."""

comptime SELECT_SAMPLE: UInt16 = 13
"""`USING SAMPLE` on a `SELECT`."""

comptime TABLE_SAMPLE: UInt16 = 14
"""`TABLESAMPLE` or `USING SAMPLE` on one table."""

comptime TABLE_MODIFIER: UInt16 = 15
"""Something after a table that is neither a join nor a pivot."""

comptime TABLE_AT: UInt16 = 16
"""`AT` after a table, which reads it as of a version or a timestamp."""

comptime JOIN_FORM: UInt16 = 17
"""A join written in a form the reference arena has no kind for."""

comptime WITH_ORDINALITY: UInt16 = 18
"""`WITH ORDINALITY` after a table function."""

comptime WITH_USING_KEY: UInt16 = 19
"""`USING KEY` on a `WITH` entry."""

comptime STATEMENT_LATER: UInt16 = 20
"""A statement that maps onto something a dataframe already does."""

comptime STATEMENT_NEVER: UInt16 = 21
"""A statement that asks for something a dataframe library does not have."""

comptime ROW_VALUE: UInt16 = 22
"""`(a, b)` or `ROW(a, b)`, several values written as one."""

comptime INTERVAL: UInt16 = 23
"""`INTERVAL '1 day'` and the other spellings of a duration."""

comptime SPECIAL_CALL: UInt16 = 24
"""`TRY` or `UNPACK`, which are written as calls and are not functions."""

comptime LAMBDA: UInt16 = 25
"""`lambda x: x + 1`, a function written in the query."""

comptime LIST_COMPREHENSION: UInt16 = 26
"""`[x + 1 FOR x IN l]`, a list built by running an expression."""

comptime NAMED_ARGUMENT: UInt16 = 27
"""`f(a := 1)`, an argument passed by name."""

comptime COLUMNS: UInt16 = 28
"""`COLUMNS(...)`, a set of columns written where one expression goes."""

comptime MAP_LITERAL: UInt16 = 29
"""`MAP {'a': 1}`, a map written out in the query."""

comptime GROUPING: UInt16 = 30
"""`GROUPING(a)`, which reports the grouping set a row came from."""

comptime POSITIONAL: UInt16 = 31
"""`#1`, a column named by its place in the select list."""

comptime DEFAULT_VALUE: UInt16 = 32
"""`DEFAULT` where a value goes."""

comptime UNPIVOT_NULLS: UInt16 = 33
"""`INCLUDE NULLS` on an `UNPIVOT`."""

comptime UNPIVOT_GROUPS: UInt16 = 34
"""More than one `FOR` group on an `UNPIVOT`."""

comptime QUANTIFIED_VALUE: UInt16 = 35
"""`ANY` or `ALL` over a value rather than over a subquery."""

comptime NO_CASE: UInt16 = 36
"""A grammar rule the transformer has no case for at all."""

comptime AGGREGATE_FILTER: UInt16 = 37
"""`FILTER` on a fold the clause cannot be rewritten into an argument of."""

comptime LIST_VALUE: UInt16 = 38
"""A list written out, `[1, 2]` or `ARRAY[1, 2]`."""


def sql_support() -> List[Refusal]:
    """Everything firepanda's SQL front end refuses, in one list.

    The order is the order of the constants above and a test checks that it
    is, because the alternative is a message that names the wrong feature,
    which reads as if it were right and is worse than no message at all.

    Returns:
        The whole table.
    """
    return [
        Refusal(
            "operator",
            "{} as an operator",
            (
                "firepanda holds an operator as the words it was written with"
                " and runs the ones it has a kernel for, and this is not one of"
                " them."
            ),
            SQL_ISSUE,
        ),
        Refusal(
            "custom-operator",
            "OPERATOR(...) as a prefix operator",
            (
                "An operator named this way is resolved against the catalog,"
                " and firepanda has no catalog of operators to resolve it"
                " against."
            ),
            SQL_ISSUE,
        ),
        Refusal(
            "like-escape",
            "ESCAPE on an operator that has no escaping form",
            (
                "A LIKE and an ILIKE take one. SIMILAR TO does not, which"
                " DuckDB says too."
            ),
            SQL_ISSUE,
        ),
        Refusal(
            "field-access",
            "a field access",
            (
                "It reads and prints back. A field belongs to a struct and a"
                " firepanda column holds one scalar, so there is nothing here"
                " with a field in it to reach into."
            ),
            STAGE_ISSUE,
        ),
        Refusal(
            "subscript",
            "a slice or a subscript",
            (
                "firepanda reads a list element and a substring with a function"
                " rather than with brackets."
            ),
            STAGE_ISSUE,
        ),
        Refusal(
            "postfix-operator",
            "a postfix operator",
            (
                "The two firepanda reads after an operand are a cast and a"
                " dotted name, and this is neither."
            ),
            SQL_ISSUE,
        ),
        Refusal(
            "call-modifier",
            "{} on a call",
            (
                "Both read and print back where they were written. WITHIN GROUP"
                " gives the fold an order to see the rows in and EXPORT_STATE"
                " asks for its state rather than its answer, and firepanda has"
                " no fold that can be told either."
            ),
            STAGE_ISSUE,
        ),
        Refusal(
            "call-argument",
            "{} inside a call",
            (
                "Both read and print back where they were written. An ordered"
                " aggregate gives the fold an order to see the rows in and a"
                " null treatment gives it a rule for what to do with a null,"
                " and firepanda has no fold that can be told either."
            ),
            STAGE_ISSUE,
        ),
        Refusal(
            "array-subquery",
            "ARRAY over a subquery",
            (
                "It reads and prints, and lowering stops on it, since"
                " collecting a whole column into one list value is a way of"
                " running a subquery that firepanda has no node for."
            ),
            SQL_ISSUE,
        ),
        Refusal(
            "dotted-name",
            "a dotted name here",
            "Only a plain name fits in this position.",
            SQL_ISSUE,
        ),
        Refusal(
            "quoted-name",
            "anything but a name here",
            (
                "Only a name fits in this position. Dots in it are fine, since"
                " that is how two collations are composed."
            ),
            SQL_ISSUE,
        ),
        Refusal(
            "not-subquery",
            "NOT in front of a subquery",
            (
                "Write NOT EXISTS or NOT IN, which say which of the two this"
                " means."
            ),
            SQL_ISSUE,
        ),
        Refusal(
            "select-clause",
            "{} in a SELECT",
            (
                "The query node has a slot for each clause firepanda runs, and"
                " none for this one yet."
            ),
            STAGE_ISSUE,
        ),
        Refusal(
            "select-sample",
            "a sample on a SELECT",
            (
                "The sample reads and prints, and lowering stops on it, since"
                " taking a share of the rows is a row source of its own and"
                " firepanda has no node for one yet."
            ),
            STAGE_ISSUE,
        ),
        Refusal(
            "table-sample",
            "a sample on a table",
            (
                "The sample reads and prints, and lowering stops on it, since"
                " taking a share of the rows is a row source of its own and"
                " firepanda has no node for one yet."
            ),
            STAGE_ISSUE,
        ),
        Refusal(
            "table-modifier",
            "{} on a table",
            (
                "A join, a PIVOT and an UNPIVOT are the three things that go"
                " here and all three are read, so this is a fourth one the"
                " grammar grew and the transformer has no case for."
            ),
            STAGE_ISSUE,
        ),
        Refusal(
            "table-at",
            "AT on a table",
            (
                "It reads and prints, and lowering turns it down, because it"
                " asks for a table as of a version or a moment and firepanda"
                " has no storage that keeps either one."
            ),
            SQL_ISSUE,
        ),
        Refusal(
            "join-form",
            "this kind of join",
            (
                "firepanda runs the joins that name a condition or take none."
                " POSITIONAL, NEAREST and JOIN BY are not among them."
            ),
            SQL_ISSUE,
        ),
        Refusal(
            "with-ordinality",
            "WITH ORDINALITY",
            (
                "It reads and prints, and lowering stops on it, since it adds"
                " a column to what a table function produces and the two are"
                " built together."
            ),
            SQL_ISSUE,
        ),
        Refusal(
            "with-using-key",
            "USING KEY on a WITH",
            (
                "It says to keep one row per key and replace that row as the"
                " query runs, which is a way of running a WITH that firepanda"
                " has no node for."
            ),
            SQL_ISSUE,
        ),
        Refusal(
            "statement-later",
            "the {} statement yet",
            (
                "It maps onto something a dataframe already does and it is"
                " coming. firepanda runs SELECT today."
            ),
            STAGE_ISSUE,
        ),
        Refusal(
            "statement-never",
            "the {} statement",
            (
                "It asks for a catalog, a transaction or an extension, and"
                " firepanda is a dataframe library rather than a database."
                " Read the data with SELECT and do the rest in Mojo."
            ),
            SQL_ISSUE,
        ),
        Refusal(
            "row-value",
            "a row value",
            (
                "Several expressions in one pair of parentheses make a single"
                " value with fields in it, and a firepanda column holds one"
                " scalar. Select the parts as separate columns."
            ),
            STAGE_ISSUE,
        ),
        Refusal(
            "interval",
            "an INTERVAL literal",
            (
                "A duration is its own type with its own arithmetic, and"
                " firepanda has no column type for one yet. It arrives with the"
                " date and time work."
            ),
            STAGE_ISSUE,
        ),
        Refusal(
            "special-call",
            "{} yet",
            (
                "TRY and UNPACK are written the way a call is written and read"
                " as calls, and neither one is a function. TRY answers NULL"
                " where the expression would have raised and UNPACK spreads a"
                " list across the arguments of the call around it, so both are"
                " shapes the plan would have to grow."
            ),
            STAGE_ISSUE,
        ),
        Refusal(
            "lambda",
            "a lambda",
            (
                "A function written inside the query has to be compiled along"
                " with the query, and firepanda runs the functions it already"
                " has. Pass a named one."
            ),
            SQL_ISSUE,
        ),
        Refusal(
            "list-comprehension",
            "a list comprehension",
            (
                "It runs an expression once for every element, which is a"
                " lambda in different brackets, and firepanda runs the"
                " functions it already has."
            ),
            SQL_ISSUE,
        ),
        Refusal(
            "named-argument",
            "an argument passed by name",
            (
                "firepanda matches arguments by position, so f(a := 1) has"
                " nowhere to put the name. Pass it in order."
            ),
            STAGE_ISSUE,
        ),
        Refusal(
            "columns",
            "COLUMNS",
            (
                "It stands for however many columns it matches, so the shape of"
                " the result is not known until the table is, and firepanda"
                " works out the shape first. Name the columns."
            ),
            STAGE_ISSUE,
        ),
        Refusal(
            "map-literal",
            "a MAP literal",
            (
                "A map holds keys and values in one value and a firepanda"
                " column holds one scalar. It arrives with the nested types."
            ),
            STAGE_ISSUE,
        ),
        Refusal(
            "grouping",
            "GROUPING",
            (
                "It reports which grouping set a row came from, which only"
                " means anything next to ROLLUP, CUBE and GROUPING SETS, and"
                " firepanda does not carry that number out of the aggregate"
                " yet."
            ),
            STAGE_ISSUE,
        ),
        Refusal(
            "positional",
            "a column written as #1",
            (
                "firepanda reads a column by name. Write the name, or the"
                " expression the column was built from."
            ),
            STAGE_ISSUE,
        ),
        Refusal(
            "default-value",
            "DEFAULT where a value goes",
            (
                "It stands for whatever a table declares as the default for a"
                " column, and that lives in a catalog. firepanda is a dataframe"
                " library and has no catalog to ask."
            ),
            SQL_ISSUE,
        ),
        Refusal(
            "unpivot-nulls",
            "INCLUDE NULLS on an UNPIVOT",
            (
                "The statement spelling of an UNPIVOT has no way to write it,"
                " and that is the spelling the node records, so there is"
                " nowhere to keep it. EXCLUDE NULLS is the default and is"
                " read."
            ),
            STAGE_ISSUE,
        ),
        Refusal(
            "unpivot-groups",
            "more than one FOR group on an UNPIVOT",
            (
                "One UNPIVOT node holds one name column and one set of value"
                " columns, so a second group has nowhere to go."
            ),
            STAGE_ISSUE,
        ),
        Refusal(
            "quantified-value",
            "ANY or ALL over a value",
            (
                "DuckDB unnests the right side when it is a list, so x = ANY"
                " ([1, 2]) is a membership test written the long way. Write it"
                " as IN, or put a SELECT on the right."
            ),
            STAGE_ISSUE,
        ),
        Refusal(
            "no-case",
            "grammar rule {}",
            (
                "The grammar accepts more than firepanda runs, and this is a"
                " rule the transformer has no case for. Please file it."
            ),
            STAGE_ISSUE,
        ),
        Refusal(
            "aggregate-filter",
            "FILTER on {}",
            (
                "A filter is read as a CASE around the argument, which answers"
                " the same number for a fold that passes over a null. first,"
                " last and any_value read a null as a value, so it would not,"
                " no fold here reads more than one argument, and a name that is"
                " not a fold has nothing for the CASE to go inside."
            ),
            SQL_ISSUE,
        ),
        Refusal(
            "list-value",
            "a list written out",
            (
                "A literal in the plan is one scalar, and a list is a value"
                " with a length. LIST is a type firepanda has and a column can"
                " hold one, so what is missing is the constant rather than the"
                " type."
            ),
            STAGE_ISSUE,
        ),
    ]


comptime UNKNOWN = Refusal(
    "unknown",
    "this",
    (
        "firepanda ran out of cases here and has nothing better to say about"
        " it, which is a bug in firepanda rather than in the query. Please"
        " file it."
    ),
    SQL_ISSUE,
)
"""What an index with no entry gets.

Nothing should ever reach it, and a test checks that the last constant is the
last entry, so this exists to keep the lookup from raising. A refusal is
already the unhappy path and an error raised while building an error is the
one shape nobody can read.
"""


def refusal(feature: UInt16) -> Refusal:
    """Looks one entry up.

    Args:
        feature: One of the constants in this file.

    Returns:
        The entry, or `UNKNOWN` if there is no such entry.
    """
    var table = sql_support()
    if Int(feature) >= len(table):
        return UNKNOWN
    return table[Int(feature)]


comptime NO_REFUSAL: UInt16 = 65535
"""What `feature_of` gives back for text that is not a refusal at all."""

comptime _PREFIX = "Not Implemented Error: firepanda does not support "
"""How the first line of every refusal starts."""


def feature_of(message: StringSlice) -> UInt16:
    """Reads a refusal back and says which entry it came from.

    A raised error is text by the time anything catches it, so a caller that
    wants to count refusals by feature has to get from the text back to the
    table. This is the way back. The conformance harness is the caller that
    needs it: a directory full of one refusal is one missing feature, and the
    same count spread over ten of them is ten, and telling those apart is the
    whole reason the table is a table.

    An entry holding a `{}` is matched on the text either side of it, because
    what went in the middle came from the query and is not known here. Two
    entries could in principle both fit, so the longer match wins, which is the
    more specific of the two.

    Args:
        message: The error, or its first line.

    Returns:
        The index into `sql_support()`, or `NO_REFUSAL` if the text is not one.
    """
    if not message.startswith(_PREFIX):
        return NO_REFUSAL

    # Only the first line says what the feature was. The rest is the caret
    # block and the explanation, and both can hold anything.
    var end = message.find("\n")
    if end < 0:
        end = message.byte_length()
    var line = message[byte = _PREFIX.byte_length() : end]
    if not line.endswith("."):
        return NO_REFUSAL
    line = line[byte = 0 : line.byte_length() - 1]

    var found = NO_REFUSAL
    var best = -1
    var table = sql_support()
    for i in range(len(table)):
        var text = table[i].message
        var hole = text.find("{}")
        if hole < 0:
            if line == text:
                return UInt16(i)
            continue
        var before = text[byte=0:hole]
        var after = text[byte = hole + 2 : text.byte_length()]
        # The two halves have to fit without overlapping, or `{} on a call`
        # would match `on a call` with nothing where the query text goes.
        if line.byte_length() < before.byte_length() + after.byte_length():
            continue
        if not line.startswith(before) or not line.endswith(after):
            continue
        var length = before.byte_length() + after.byte_length()
        if length > best:
            best = length
            found = UInt16(i)
    return found


def support_table() -> String:
    """The whole table as markdown, for the README.

    A list of what a library will not do goes stale the week after it is
    written, every time, because nobody remembers the README when they are
    adding a table entry. So the README does not have a list, it has this, and
    a test fails when the file and the table disagree.

    Returns:
        A markdown table with a header row, ending in a newline.
    """
    var out = String(
        "| Name | firepanda does not support | Instead | Issue |\n",
        "| --- | --- | --- | --- |\n",
    )
    for entry in sql_support():
        out += String(
            "| `",
            entry.feature,
            "` | ",
            entry.message,
            " | ",
            entry.explanation,
            " | [#",
            entry.issue,
            "](https://github.com/tamnd/firepanda/issues/",
            entry.issue,
            ") |\n",
        )
    return out^


def issue_link(issue: UInt32) -> String:
    """The URL a refusal points the reader at.

    Args:
        issue: The issue number.

    Returns:
        The link, as it appears in the message.
    """
    return String("See https://github.com/tamnd/firepanda/issues/", issue)


def not_implemented(
    feature: UInt16, detail: StringSlice, position: String
) -> Error:
    """Builds a refusal.

    Args:
        feature: One of the constants in this file.
        detail: The text from the query the message holds a `{}` for, empty
            for an entry with no `{}` in it.
        position: The caret block, or empty for a refusal with no position.

    Returns:
        The error, ready to raise.
    """
    var entry = refusal(feature)
    var out = String(
        "Not Implemented Error: firepanda does not support ",
        filled(entry.message, detail),
        ".",
    )
    if position.byte_length() > 0:
        out += String("\n", position)
    return Error(
        String(out, "\n", entry.explanation, " ", issue_link(entry.issue))
    )


def filled(message: StaticString, detail: StringSlice) -> String:
    """Puts the text from the query into a message's `{}`.

    Args:
        message: The message, with at most one `{}` in it.
        detail: What goes there.

    Returns:
        The message, with nothing left to fill.
    """
    var at = message.find("{}")
    if at < 0:
        return String(message)
    var before = message[byte=0:at]
    var after = message[byte = at + 2 : message.byte_length()]
    return String(before, detail, after)
