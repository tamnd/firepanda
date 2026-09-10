"""The expression AST: arenas, node layout and the text they carry.

The parse tree from `matcher.mojo` has one node per grammar rule, which for an
ordinary comparison is a dozen nodes of pure syntax. Binding against that would
tie every later stage to grammar rule names, so a grammar bump would break the
binder rather than a small translation layer. This is the shape that layer
produces instead. See docs/specs/sql/05-ast-and-binder.md section 2.

Three things decide the layout.

Nodes are fixed size and live in an arena, referenced by index rather than by
pointer. The alternative, a variant with an inline `List` of children, allocates
per node and fights ownership for no benefit. A node that needs a variable
number of children stores a run in a side list instead, and a run is a count
followed by that many entries, so the node itself stays one size.

Index 0 is the null node in every arena, the same convention `matcher.mojo`
uses, so a missing operand is 0 and no caller needs a separate flag to say a
field is absent. Building a node returns an index that is never 0.

The AST owns its text. A token points into the query string and keeps its
quotes, its escapes and its original case, because that is what makes a token
twelve bytes. Nothing downstream wants any of that, so text is decoded once on
the way in and interned here, and an `Ast` is therefore independent of the
query it came from.

Every node carries a token index. Not for debugging: every binder and runtime
error renders a caret, and a node with no position produces the error message
nobody can act on.

What is here is the expression grammar. Statements and table references are the
other two arenas and they arrive with the transformer, along with the three
expression kinds that hold a statement inside them, which are a scalar
subquery, `EXISTS` and the subquery form of `IN`.
"""

comptime NO_NODE: UInt32 = 0
"""The null node, and the empty run.

Index 0 of every arena is a node nobody builds, so an absent operand is 0 and a
node with no variadic children has a `children` of 0.
"""


comptime EXPR_LITERAL: UInt8 = 1
"""A constant.

`payload` is the interned text of the value and `b` is one of the `LITERAL_`
tags saying how to print it. The text is the decoded value, so a string literal
holds what it means rather than what was typed, and the printer puts the quotes
back.
"""

comptime EXPR_COLUMN: UInt8 = 2
"""A column reference, qualified or not.

`children` is a run of interned name parts, outermost first, so `a` is one part
and `s.t.c` is three. Nothing here says whether a part named a table or a
struct field, because deciding that needs the catalog and this stage has none.
"""

comptime EXPR_STAR: UInt8 = 3
"""`*`, or `t.*`, with the three modifiers DuckDB allows on it.

`children` is a run of name parts qualifying the star, empty for a bare one.
`a` is a run of interned names for `EXCLUDE`. `b` is a run of alternating name
and expression for `REPLACE`. `payload` is a run of alternating name and name
for `RENAME`. All four are runs and all four may be empty, which is the whole
of `SELECT * EXCLUDE (a) REPLACE (x + 1 AS b) RENAME (c AS d)`.
"""

comptime EXPR_FUNCTION: UInt8 = 4
"""A call, `f(x)`.

`payload` is a run of interned name parts, so a qualified `main.f` keeps its
qualification. `children` is a run of argument expressions. `a` is a bit set of
the `CALL_` flags. `b` is reserved for the modifier node that carries `ORDER
BY`, `FILTER` and `OVER`, and is 0 until that node exists.

An operator is not one of these even where the grammar spells it as one. The
printer has to know that `+` goes between its operands and `f` goes before
them, and that is the difference the kind records.
"""

comptime EXPR_UNARY: UInt8 = 5
"""A prefix operator, `-x` or `NOT x`.

`payload` is the interned operator text and `a` is the operand. The text is the
operator as SQL spells it, so the printer needs no table to print it back, and
a word operator like `NOT` prints with a space after it because the text says
it is a word.
"""

comptime EXPR_BINARY: UInt8 = 6
"""An infix operator, `a + b` or `a AND b`.

`payload` is the interned operator text, `a` is the left operand and `b` is the
right one. Precedence is not stored, because by the time a node exists the tree
already has the shape precedence gave it, and the printer parenthesizes every
operand rather than recomputing what it could leave out.
"""

comptime EXPR_CAST: UInt8 = 7
"""`CAST(x AS t)`, and `TRY_CAST` when `b` is 1.

`a` is the operand and `payload` is the interned type text. The type is text
and not a parsed type, because resolving a type name needs the type registry
and this stage resolves nothing.
"""

comptime EXPR_CASE: UInt8 = 8
"""`CASE`, both the searched form and the simple one.

`a` is the operand a simple `CASE x WHEN` compares against, or 0 for the
searched form. `children` is a run of alternating condition and result, so it
always has an even length. `b` is the `ELSE` result, or 0.
"""

comptime EXPR_BETWEEN: UInt8 = 9
"""`x BETWEEN lo AND hi`.

`a` is the operand, `children` is a run of exactly two, the bounds in order,
and `payload` is 1 for `NOT BETWEEN`. It is its own kind rather than two
comparisons joined by `AND` because rewriting it here would evaluate the
operand twice and print back something the user did not write.
"""

comptime EXPR_IN: UInt8 = 10
"""`x IN (a, b, c)`.

`a` is the operand, `children` is a run of candidates and `payload` is 1 for
`NOT IN`. The form where the right side is a subquery is a different kind and
arrives with the statement arena.
"""

comptime EXPR_LIST: UInt8 = 11
"""A list constructor, `[1, 2, 3]`.

`children` is a run of elements, which may be empty.
"""

comptime EXPR_STRUCT: UInt8 = 12
"""A struct constructor, `{'a': 1}`.

`children` is a run of alternating interned field name and value expression, so
it always has an even length. The names are stored as text rather than as
literal nodes because a struct field name is not an expression, and letting it
be one would mean the printer had to decide when a literal was really a name.
"""

comptime EXPR_COLLATE: UInt8 = 13
"""`x COLLATE c`.

`a` is the operand and `payload` is the interned collation name.
"""

comptime EXPR_PARAMETER: UInt8 = 14
"""A prepared statement parameter, `?`, `$1` or `$name`.

`payload` is the interned text after the sigil, empty for a bare `?`, and `b`
is the interned sigil itself, so the printer puts back the form that was
written. The two forms are not interchangeable in DuckDB: a statement may use
positional parameters or named ones, not both.
"""


comptime LITERAL_NULL: UInt32 = 0
"""`NULL`, whose interned text is empty because the kind is the whole value."""

comptime LITERAL_BOOLEAN: UInt32 = 1
"""`TRUE` or `FALSE`, interned as the word itself in upper case."""

comptime LITERAL_NUMBER: UInt32 = 2
"""A number, interned as the digits with the underscores already taken out.

Whether it is an integer, a decimal or a double is a typing decision and this
stage makes none, so the text is kept and the binder reads it.
"""

comptime LITERAL_STRING: UInt32 = 3
"""A string, interned as the decoded value with no quotes and no escapes."""


comptime CALL_DISTINCT: UInt32 = 1
"""`f(DISTINCT x)`."""

comptime CALL_STAR: UInt32 = 2
"""`count(*)`, which has no arguments rather than one star argument."""


struct Expr(ImplicitlyCopyable, Movable):
    """One expression node, twenty one bytes, holding no pointers.

    What `a`, `b`, `children` and `payload` mean depends on `kind`, and every
    kind says so in its own docstring. Four fields is what the widest kind
    needs, which is `EXPR_STAR` with its three modifier runs and its
    qualification.
    """

    var kind: UInt8
    """One of the `EXPR_` constants."""

    var a: UInt32
    """An operand index, a run or a flag set, by kind."""

    var b: UInt32
    """A second operand index, a run or a tag, by kind."""

    var children: UInt32
    """A run in the side list, or 0 for none."""

    var token: UInt32
    """The token this node starts at, for the caret in an error."""

    var payload: UInt32
    """A string index or a run, by kind."""

    def __init__(
        out self,
        kind: UInt8,
        token: UInt32,
        a: UInt32 = NO_NODE,
        b: UInt32 = NO_NODE,
        children: UInt32 = NO_NODE,
        payload: UInt32 = NO_NODE,
    ):
        """Builds a node.

        Args:
            kind: One of the `EXPR_` constants.
            token: The token the node starts at.
            a: By kind.
            b: By kind.
            children: A run, or 0.
            payload: By kind.
        """
        self.kind = kind
        self.a = a
        self.b = b
        self.children = children
        self.token = token
        self.payload = payload


struct Ast(Movable):
    """The arenas, the side list and the string pool.

    One of these owns a whole statement's worth of nodes and the text they
    refer to, so it outlives the query string it was built from.
    """

    var exprs: List[Expr]
    """The expression arena. Index 0 is the null node."""

    var runs: List[UInt32]
    """The side list holding every variadic child run.

    A run is a count followed by that many entries, which is why a run index of
    0 can mean empty: index 0 holds a count of zero and nothing follows it.
    """

    var strings: List[String]
    """The pool. Index 0 is the empty string."""

    var interned: Dict[String, UInt32]
    """Text to pool index, so the same name is stored once.

    Names repeat heavily in a real query, since every column reference in a
    `SELECT` list and every mention of it in `GROUP BY` and `ORDER BY` is the
    same handful of words.
    """

    def __init__(out self):
        """Builds an empty AST, with the null node and the empty run in place.
        """
        self.exprs = List[Expr]()
        self.exprs.append(Expr(kind=0, token=0))
        self.runs = List[UInt32]()
        self.runs.append(0)
        self.strings = List[String]()
        self.strings.append(String())
        self.interned = Dict[String, UInt32]()

    def intern(mut self, text: StringSlice) -> UInt32:
        """Puts text in the pool and returns its index.

        Args:
            text: The decoded text, with no quotes and no escapes left in it.

        Returns:
            The pool index, which is 0 for the empty string.
        """
        if text.byte_length() == 0:
            return 0
        var key = String(text)
        var found = self.interned.get(key)
        if found:
            return found.value()
        var at = UInt32(len(self.strings))
        self.strings.append(key)
        self.interned[key^] = at
        return at

    def text(ref self, index: UInt32) -> ref[self.strings[Int(index)]] String:
        """Returns the text a pool index refers to.

        A reference rather than a slice, because a short `String` may keep its
        bytes inside itself, and a slice into one would dangle the moment the
        pool grew and moved it. A reference is rebound by the same move and
        stays correct.

        Args:
            index: A pool index, where 0 is the empty string.

        Returns:
            The text, borrowed from the pool.
        """
        return self.strings[Int(index)]

    def run(mut self, items: List[UInt32]) -> UInt32:
        """Stores a run of child indices and returns a handle to it.

        Args:
            items: The children, in order.

        Returns:
            The run handle, which is 0 for an empty run.
        """
        if len(items) == 0:
            return NO_NODE
        var at = UInt32(len(self.runs))
        self.runs.append(UInt32(len(items)))
        for item in items:
            self.runs.append(item)
        return at

    def length(self, run: UInt32) -> Int:
        """Returns how many entries a run has.

        Args:
            run: A run handle, or 0.

        Returns:
            The count, which is 0 for an empty run.
        """
        if run == NO_NODE:
            return 0
        return Int(self.runs[Int(run)])

    def at(self, run: UInt32, index: Int) -> UInt32:
        """Returns one entry of a run.

        Args:
            run: A run handle, which must not be 0.
            index: Which entry, from 0.

        Returns:
            The entry.
        """
        return self.runs[Int(run) + 1 + index]

    def items(self, run: UInt32) -> List[UInt32]:
        """Collects a whole run.

        Args:
            run: A run handle, or 0.

        Returns:
            The entries in order, empty for an empty run.
        """
        var out = List[UInt32]()
        for i in range(self.length(run)):
            out.append(self.at(run, i))
        return out^

    def add(mut self, var node: Expr) -> UInt32:
        """Puts a node in the expression arena.

        Args:
            node: The node.

        Returns:
            Its index, which is never 0.
        """
        var at = UInt32(len(self.exprs))
        self.exprs.append(node^)
        return at

    def literal(
        mut self, tag: UInt32, value: StringSlice, token: UInt32 = 0
    ) -> UInt32:
        """Builds a constant.

        Args:
            tag: One of the `LITERAL_` constants.
            value: The decoded value, empty for `NULL`.
            token: The token it starts at.

        Returns:
            The node index.
        """
        var text = self.intern(value)
        return self.add(
            Expr(kind=EXPR_LITERAL, token=token, b=tag, payload=text)
        )

    def column(mut self, parts: List[String], token: UInt32 = 0) -> UInt32:
        """Builds a column reference.

        Args:
            parts: The name parts, outermost first, at least one.
            token: The token it starts at.

        Returns:
            The node index.
        """
        var interned = List[UInt32]()
        for part in parts:
            interned.append(self.intern(part))
        var run = self.run(interned)
        return self.add(Expr(kind=EXPR_COLUMN, token=token, children=run))

    def binary(
        mut self,
        operator: StringSlice,
        left: UInt32,
        right: UInt32,
        token: UInt32 = 0,
    ) -> UInt32:
        """Builds an infix operator node.

        Args:
            operator: The operator as SQL spells it.
            left: The left operand.
            right: The right operand.
            token: The token the operator is at.

        Returns:
            The node index.
        """
        var text = self.intern(operator)
        return self.add(
            Expr(kind=EXPR_BINARY, token=token, a=left, b=right, payload=text)
        )

    def unary(
        mut self, operator: StringSlice, operand: UInt32, token: UInt32 = 0
    ) -> UInt32:
        """Builds a prefix operator node.

        Args:
            operator: The operator as SQL spells it.
            operand: What it applies to.
            token: The token the operator is at.

        Returns:
            The node index.
        """
        var text = self.intern(operator)
        return self.add(
            Expr(kind=EXPR_UNARY, token=token, a=operand, payload=text)
        )

    def call(
        mut self,
        name: StringSlice,
        arguments: List[UInt32],
        flags: UInt32 = 0,
        token: UInt32 = 0,
    ) -> UInt32:
        """Builds a function call with an unqualified name.

        Args:
            name: The function name.
            arguments: The argument expressions, in order.
            flags: A bit set of the `CALL_` constants.
            token: The token the name is at.

        Returns:
            The node index.
        """
        var parts = List[UInt32]()
        parts.append(self.intern(name))
        var named = self.run(parts)
        var args = self.run(arguments)
        return self.add(
            Expr(
                kind=EXPR_FUNCTION,
                token=token,
                a=flags,
                children=args,
                payload=named,
            )
        )

    def cast(
        mut self,
        operand: UInt32,
        type_name: StringSlice,
        tries: Bool = False,
        token: UInt32 = 0,
    ) -> UInt32:
        """Builds a cast.

        Args:
            operand: What is being cast.
            type_name: The target type, as written.
            tries: Whether this is `TRY_CAST`.
            token: The token it starts at.

        Returns:
            The node index.
        """
        var text = self.intern(type_name)
        return self.add(
            Expr(
                kind=EXPR_CAST,
                token=token,
                a=operand,
                b=UInt32(1) if tries else UInt32(0),
                payload=text,
            )
        )

    def case(
        mut self,
        arms: List[UInt32],
        otherwise: UInt32 = NO_NODE,
        operand: UInt32 = NO_NODE,
        token: UInt32 = 0,
    ) raises -> UInt32:
        """Builds a `CASE`.

        Args:
            arms: Alternating condition and result, so an even length.
            otherwise: The `ELSE` result, or 0 for none.
            operand: What a simple `CASE x WHEN` compares against, or 0.
            token: The token it starts at.

        Returns:
            The node index.

        Raises:
            Error: If `arms` is empty or has an odd length.
        """
        if len(arms) == 0 or len(arms) % 2 != 0:
            raise Error(
                String(
                    (
                        "a CASE wants an even number of condition and result"
                        " entries, and at least two, but got "
                    ),
                    len(arms),
                )
            )
        var run = self.run(arms)
        return self.add(
            Expr(
                kind=EXPR_CASE,
                token=token,
                a=operand,
                b=otherwise,
                children=run,
            )
        )

    def between(
        mut self,
        operand: UInt32,
        low: UInt32,
        high: UInt32,
        negated: Bool = False,
        token: UInt32 = 0,
    ) -> UInt32:
        """Builds `x BETWEEN lo AND hi`.

        Args:
            operand: What is being tested.
            low: The lower bound.
            high: The upper bound.
            negated: Whether this is `NOT BETWEEN`.
            token: The token it starts at.

        Returns:
            The node index.
        """
        var bounds = List[UInt32]()
        bounds.append(low)
        bounds.append(high)
        var run = self.run(bounds)
        return self.add(
            Expr(
                kind=EXPR_BETWEEN,
                token=token,
                a=operand,
                children=run,
                payload=UInt32(1) if negated else UInt32(0),
            )
        )

    def in_list(
        mut self,
        operand: UInt32,
        candidates: List[UInt32],
        negated: Bool = False,
        token: UInt32 = 0,
    ) -> UInt32:
        """Builds `x IN (a, b, c)`.

        Named with a trailing underscore because `in` is a keyword.

        Args:
            operand: What is being tested.
            candidates: The values on the right, in order.
            negated: Whether this is `NOT IN`.
            token: The token it starts at.

        Returns:
            The node index.
        """
        var run = self.run(candidates)
        return self.add(
            Expr(
                kind=EXPR_IN,
                token=token,
                a=operand,
                children=run,
                payload=UInt32(1) if negated else UInt32(0),
            )
        )

    def list_of(mut self, elements: List[UInt32], token: UInt32 = 0) -> UInt32:
        """Builds a list constructor, `[1, 2, 3]`.

        Args:
            elements: The elements, in order, possibly none.
            token: The token it starts at.

        Returns:
            The node index.
        """
        var run = self.run(elements)
        return self.add(Expr(kind=EXPR_LIST, token=token, children=run))

    def struct_of(
        mut self,
        names: List[String],
        values: List[UInt32],
        token: UInt32 = 0,
    ) raises -> UInt32:
        """Builds a struct constructor, `{'a': 1}`.

        Args:
            names: The field names, in order.
            values: The field values, in the same order.
            token: The token it starts at.

        Returns:
            The node index.

        Raises:
            Error: If there are not as many values as names.
        """
        if len(names) != len(values):
            raise Error(
                String(
                    "a struct wants one value per field name, but got ",
                    len(names),
                    " names and ",
                    len(values),
                    " values",
                )
            )
        var flat = List[UInt32]()
        for i in range(len(names)):
            flat.append(self.intern(names[i]))
            flat.append(values[i])
        var run = self.run(flat)
        return self.add(Expr(kind=EXPR_STRUCT, token=token, children=run))

    def collate(
        mut self, operand: UInt32, collation: StringSlice, token: UInt32 = 0
    ) -> UInt32:
        """Builds `x COLLATE c`.

        Args:
            operand: What is being collated.
            collation: The collation name.
            token: The token it starts at.

        Returns:
            The node index.
        """
        var text = self.intern(collation)
        return self.add(
            Expr(kind=EXPR_COLLATE, token=token, a=operand, payload=text)
        )

    def parameter(
        mut self, sigil: StringSlice, name: StringSlice = "", token: UInt32 = 0
    ) -> UInt32:
        """Builds a prepared statement parameter.

        Args:
            sigil: Either `?` or `$`.
            name: The number or name after it, empty for a bare `?`.
            token: The token it starts at.

        Returns:
            The node index.
        """
        return self.add(
            Expr(
                kind=EXPR_PARAMETER,
                token=token,
                b=self.intern(sigil),
                payload=self.intern(name),
            )
        )

    def star(
        mut self,
        qualifier: List[String] = List[String](),
        exclude: List[String] = List[String](),
        token: UInt32 = 0,
    ) -> UInt32:
        """Builds `*`, or `t.*`, with an optional `EXCLUDE`.

        `REPLACE` and `RENAME` are the other two modifiers the node has room
        for. They take expressions and name pairs rather than plain names, so
        they are built directly rather than through this.

        Args:
            qualifier: The name parts before the star, empty for a bare one.
            exclude: The names to leave out.
            token: The token the star is at.

        Returns:
            The node index.
        """
        var parts = List[UInt32]()
        for part in qualifier:
            parts.append(self.intern(part))
        var excluded = List[UInt32]()
        for name in exclude:
            excluded.append(self.intern(name))
        return self.add(
            Expr(
                kind=EXPR_STAR,
                token=token,
                a=self.run(excluded),
                children=self.run(parts),
            )
        )
