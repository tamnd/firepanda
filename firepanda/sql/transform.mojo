"""The parse tree to the expression AST.

`matcher.mojo` gives back one node per grammar rule, which is the shape the
grammar has rather than the shape SQL means. This turns that into the arenas in
`ast.mojo`, and it is the only place in the engine that knows a grammar rule
name. A grammar bump therefore breaks this file or nothing. See
docs/specs/sql/05-ast-and-binder.md section 3.

Three properties the spec asks of it, and where each one lives here.

Dispatch is a jump table. `Transform` builds one byte per rule index once, at
the time the grammar is loaded, and the walk is a lookup into that array
followed by a switch. Nothing compares a rule name while a query is being
transformed. A rule with no entry is not a fallthrough: it refuses and names
itself, which is what keeps the coverage gap visible instead of turning it into
a wrong answer. The refusal table that gives those refusals their message and
their tracking link is the next piece of #307, and until it lands they raise
with the rule name in them.

Nothing is resolved. `a.b` becomes a two part name and not a column, `count(x)`
becomes a call with a name and not an overload. Every decision that needs to
know what exists belongs to the binder.

Precedence is flattened. The grammar spells sixteen levels as a chain of
`X <- Y Tail*` rules, so `1 + 2 * 3` arrives as a tower of single child pass
throughs. Every one of those levels is folded here by the same routine, left to
right, which is the associativity the chain shape asks for. Getting this wrong
is a wrong answer that no syntax test catches, which is why the printer landed
with the AST and why the test for this is a round trip rather than a comparison
against an expected tree.

Where a sub structure has to be told apart from its siblings, this reads the
first token of the node rather than looking its rule up. A star's three
modifiers all arrive as children of the same node and are told apart by
`EXCLUDE`, `REPLACE` and `RENAME`, which is the word a reader would use too.
That is a byte comparison on a word already in hand, not a dispatch, and the
dispatch above it is still an array lookup.

The walk itself is a loop and not a recursion. `Work` holds one slot per parse
node, and a form that needs the value of a child asks for it, which either
hands back what the child became or records the request and raises so the loop
can build the child and run the form again. Nothing in here calls itself, so
depth costs heap and not stack. What refuses a deeply nested query today is the
matcher's own depth limit, and when that goes this file will not be the next
wall. Do not fold this back into a recursion. Besides the depth, the loop is
also the shape the Mojo 1.0 compiler will build: a recursive transformer with
this much control flow in it hangs the compiler outright, which is #369.
"""

from .ast import (
    Ast,
    CALL_DISTINCT,
    CALL_STAR,
    EXPR_COLUMN,
    EXPR_FUNCTION,
    EXPR_STAR,
    Expr,
    LITERAL_BOOLEAN,
    LITERAL_NULL,
    LITERAL_NUMBER,
    LITERAL_STRING,
    NO_NODE,
)
from .matcher import Parse, parse_rule
from .table import Grammar
from .token import (
    FLAG_DOLLAR,
    FLAG_ESCAPE,
    TOKEN_IDENTIFIER,
    TOKEN_KEYWORD,
    TOKEN_NUMBER,
    TOKEN_QUOTED_IDENTIFIER,
    TOKEN_STRING,
    Token,
    token_text,
)

comptime _NO_CASE: UInt8 = 0
"""No case for this rule, so it refuses and names itself."""

comptime _DESCEND: UInt8 = 1
"""One child, which is the expression. The whole of the pass through rules."""

comptime _FOLD: UInt8 = 2
"""`X <- Y Tail*`, folded left to right into binary nodes."""

comptime _COLLATE: UInt8 = 3
"""`CollateExpression`, a fold whose right side is a name and not a value."""

comptime _NOT: UInt8 = 4
"""`LogicalNotExpression`, an optional run of `NOT` in front of an operand."""

comptime _IS: UInt8 = 5
"""`IsExpression`, a run of `IS` tests after an operand."""

comptime _BETWEEN_IN_LIKE: UInt8 = 6
"""`BetweenInLikeExpression`, the three postfix forms that share a level."""

comptime _PREFIX: UInt8 = 7
"""`PrefixExpression`, a run of prefix operators in front of an operand."""

comptime _BASE: UInt8 = 8
"""`BaseExpression`, an operand followed by casts, field accesses and slices."""

comptime _COLUMN: UInt8 = 9
"""`ColumnReference`, one to five dotted name parts."""

comptime _FUNCTION: UInt8 = 10
"""`FunctionExpression`, a call."""

comptime _STAR: UInt8 = 11
"""`StarExpression`, with `EXCLUDE`, `REPLACE` and `RENAME`."""

comptime _CASE: UInt8 = 12
"""`CaseExpression`, both the searched form and the simple one."""

comptime _CAST: UInt8 = 13
"""`CastExpression`, the `CAST(x AS t)` spelling."""

comptime _LIST: UInt8 = 14
"""`ListExpression`, a list constructor."""

comptime _STRUCT: UInt8 = 15
"""`StructExpression`, a struct constructor."""

comptime _PARAMETER: UInt8 = 16
"""`Parameter`, one of the four prepared statement spellings."""

comptime _COALESCE: UInt8 = 17
"""`CoalesceExpression`, which is a call written with its own syntax."""

comptime _NULLIF: UInt8 = 18
"""`NullIfExpression`, the same."""

comptime _STRING: UInt8 = 19
comptime _NUMBER: UInt8 = 20
comptime _NULL: UInt8 = 21
comptime _TRUE: UInt8 = 22
comptime _FALSE: UInt8 = 23

comptime _LEFT_PAREN = Byte(ord("("))
comptime _LEFT_BRACKET = Byte(ord("["))
comptime _DOT = Byte(ord("."))
comptime _COLON = Byte(ord(":"))
comptime _SINGLE_QUOTE = Byte(ord("'"))
comptime _DOUBLE_QUOTE = Byte(ord('"'))
comptime _DOLLAR = Byte(ord("$"))
comptime _UNDERSCORE = Byte(ord("_"))

comptime _PENDING = "the transformer is waiting for a child"
"""What a form raises when a value it asked for is not built yet.

This never reaches a caller. `Transform.expression` catches it, sees the
requests the form left behind in `Work.wants`, builds those first and runs the
form again. It is spelled as an error because a form asks for a value in the
middle of an expression, where there is nothing sensible to return instead.
"""


struct Work(Movable):
    """What one walk over one expression has built so far.

    A form asks for the value of a child through `value` and gets one of two
    things: the expression node the child became, or a request recorded in
    `wants` and a raise. That is what lets the walk be a loop rather than a
    recursion, so the depth of an expression costs heap and not stack.

    The cost of the retry is one extra pass over the form, which reads the
    parse tree and builds nothing, so a form that has more than one operand
    asks for all of them at once through `warm` before it builds anything.
    """

    var results: List[UInt32]
    """One entry per parse node. `NO_NODE` means the node is not built yet."""

    var wants: List[UInt32]
    """The children the attempt in progress asked for and did not have."""

    def __init__(out self, count: Int):
        """Starts a walk with nothing built.

        Args:
            count: How many nodes the parse has.
        """
        self.results = List[UInt32](length=count, fill=NO_NODE)
        self.wants = List[UInt32]()

    def value(mut self, node: UInt32) raises -> UInt32:
        """The expression a child became, or a request for it.

        Args:
            node: The child parse node.

        Returns:
            The expression node it became.

        Raises:
            Error: If it is not built yet, having recorded the request.
        """
        if node == NO_NODE:
            raise Error("the transformer was handed the null node")
        var built = self.results[Int(node)]
        if built != NO_NODE:
            return built
        self.wants.append(node)
        raise Error(_PENDING)

    def warm(mut self, nodes: List[UInt32]) raises:
        """Asks for several children at once, so one retry covers them all.

        Args:
            nodes: The children, `NO_NODE` for the ones with nothing to ask
                for.

        Raises:
            Error: If any of them is not built yet, having recorded every one
                that is not.
        """
        var missing = False
        for node in nodes:
            if node == NO_NODE:
                continue
            if self.results[Int(node)] == NO_NODE:
                self.wants.append(node)
                missing = True
        if missing:
            raise Error(_PENDING)


struct Transform(Movable):
    """The jump table, built once against a loaded grammar.

    One of these is as reusable as the `Grammar` it was built from, holds no
    query state, and costs one byte per grammar rule, which is 1,187 bytes.
    """

    var actions: List[UInt8]
    """One action per rule index. `_NO_CASE` means the rule refuses."""

    var expression_rule: Int
    """The index of `Expression`, so a caller can parse one directly."""

    def __init__(out self, grammar: Grammar) raises:
        """Builds the table.

        Args:
            grammar: A loaded grammar.

        Raises:
            Error: If the grammar has no rule by a name this expects, which
                means a bump renamed something and this file has to follow.
        """
        self.actions = List[UInt8](length=len(grammar.names), fill=_NO_CASE)
        self.expression_rule = -1

        # The pass throughs. Every one of these is a rule that exists so
        # another rule could name it, and it has exactly one child.
        var descends: List[StaticString] = [
            "Expression",
            "SingleExpression",
            "ParensExpression",
            "Parens_Expression",
            "LiteralExpression",
            "ConstantLiteral",
            "IsLiteralValue",
            "SpecialFunctionExpression",
            "FunctionArgument",
            "PositionalFunctionArgument",
        ]
        for name in descends:
            self._set(grammar, name, _DESCEND)

        # The precedence chain, every level of it the same shape.
        var folds: List[StaticString] = [
            "LambdaArrowExpression",
            "LogicalOrExpression",
            "LogicalAndExpression",
            "IsDistinctFromExpression",
            "ComparisonExpression",
            "OtherOperatorExpression",
            "BitwiseExpression",
            "AdditiveExpression",
            "MultiplicativeExpression",
            "ExponentiationExpression",
            "AtTimeZoneExpression",
        ]
        for name in folds:
            self._set(grammar, name, _FOLD)

        self._set(grammar, "CollateExpression", _COLLATE)
        self._set(grammar, "LogicalNotExpression", _NOT)
        self._set(grammar, "IsExpression", _IS)
        self._set(grammar, "BetweenInLikeExpression", _BETWEEN_IN_LIKE)
        self._set(grammar, "PrefixExpression", _PREFIX)
        self._set(grammar, "BaseExpression", _BASE)
        self._set(grammar, "ColumnReference", _COLUMN)
        self._set(grammar, "FunctionExpression", _FUNCTION)
        self._set(grammar, "StarExpression", _STAR)
        self._set(grammar, "CaseExpression", _CASE)
        self._set(grammar, "CastExpression", _CAST)
        self._set(grammar, "ListExpression", _LIST)
        self._set(grammar, "StructExpression", _STRUCT)
        self._set(grammar, "Parameter", _PARAMETER)
        self._set(grammar, "CoalesceExpression", _COALESCE)
        self._set(grammar, "NullIfExpression", _NULLIF)
        self._set(grammar, "StringLiteral", _STRING)
        self._set(grammar, "NumberLiteral", _NUMBER)
        self._set(grammar, "NullLiteral", _NULL)
        self._set(grammar, "TrueLiteral", _TRUE)
        self._set(grammar, "FalseLiteral", _FALSE)

        self.expression_rule = grammar.rule("Expression")

    def _set(
        mut self, grammar: Grammar, name: StaticString, action: UInt8
    ) raises:
        """Points one rule at one action.

        Args:
            grammar: A loaded grammar.
            name: The rule name, spelled the way the grammar spells it.
            action: What to do with a node of that rule.

        Raises:
            Error: If there is no such rule.
        """
        var index = grammar.rule(name)
        if index < 0:
            raise Error(
                String(
                    "the transformer dispatches on a rule named ",
                    name,
                    (
                        " and the grammar has no such rule, so a grammar bump"
                        " renamed it and firepanda/sql/transform.mojo has to"
                        " follow"
                    ),
                )
            )
        self.actions[index] = action

    def parse_expression(
        self, sql: StringSlice, grammar: Grammar, mut ast: Ast
    ) raises -> UInt32:
        """Parses one expression and transforms it, which is what tests want.

        Args:
            sql: The expression text on its own, with no `SELECT` around it.
            grammar: The grammar it was parsed against.
            ast: Where to put the nodes.

        Returns:
            The root expression node.

        Raises:
            Error: If the text is not an expression, or holds something this
                does not transform yet.
        """
        var tree = parse_rule(sql, grammar, self.expression_rule)
        return self.expression(tree, sql, tree.root, ast)

    def expression(
        self, tree: Parse, sql: StringSlice, node: UInt32, mut ast: Ast
    ) raises -> UInt32:
        """Transforms one parse node, and everything under it, into the AST.

        The walk is a stack and a loop and not a recursion. A form that needs
        the value of a child asks for it, and if it is not built yet the form
        raises, the child goes on the stack, and the form runs again once the
        child is there. Two things come out of that. Depth costs heap and not
        stack, so there is no depth at which this stops working before the
        matcher's own limit does. And a form stays written in the order a
        reader would write it, which a hand written state machine would not
        be.

        Args:
            tree: The parse the node lives in.
            sql: The query the parse came from, for token text.
            node: The parse node.
            ast: Where to put the nodes.

        Returns:
            The expression node index, which is never 0.

        Raises:
            Error: If the rule has no case, or a part of it has none.
        """
        if node == NO_NODE:
            raise Error("the transformer was handed the null node")

        var work = Work(len(tree.nodes))
        var stack = List[UInt32]()
        stack.append(node)

        while len(stack) > 0:
            var top = stack[len(stack) - 1]
            if work.results[Int(top)] != NO_NODE:
                _ = stack.pop()
                continue

            work.wants.clear()
            try:
                var built = self._build(tree, sql, top, ast, work)
                work.results[Int(top)] = built
                _ = stack.pop()
            except e:
                # A form that left no request behind failed for a reason of its
                # own, and that is the reason the caller should see.
                if len(work.wants) == 0:
                    raise e
                for want in work.wants:
                    stack.append(want)

        return work.results[Int(node)]

    def _build(
        self,
        tree: Parse,
        sql: StringSlice,
        node: UInt32,
        mut ast: Ast,
        mut work: Work,
    ) raises -> UInt32:
        """Runs the one form a node's rule asks for.

        Args:
            tree: The parse the node lives in.
            sql: The query the parse came from, for token text.
            node: The parse node.
            ast: Where to put the nodes.
            work: The walk, for the values of children.

        Returns:
            The expression node index, which is never 0.

        Raises:
            Error: If the rule has no case, or a value it needs is not built
                yet.
        """
        var rule = Int(tree.nodes[Int(node)].rule)
        var at = tree.nodes[Int(node)].token_start
        var action = self.actions[rule]

        if action == _DESCEND:
            return work.value(self._only(tree, node))

        if action == _FOLD:
            return self._fold(tree, sql, node, ast, work)

        if action == _COLLATE:
            return self._collate(tree, sql, node, ast, work)

        if action == _NOT:
            var kids = tree.children(node)
            var operand = work.value(kids[len(kids) - 1])
            if len(kids) == 1:
                return operand
            return self._negate(tree, kids[0], ast, operand)

        if action == _IS:
            return self._is(tree, sql, node, ast, work)

        if action == _BETWEEN_IN_LIKE:
            return self._between_in_like(tree, sql, node, ast, work)

        if action == _PREFIX:
            return self._prefix(tree, sql, node, ast, work)

        if action == _BASE:
            return self._base(tree, sql, node, ast, work)

        if action == _COLUMN:
            return ast.column(
                self._parts(tree, sql, self._only(tree, node)), at
            )

        if action == _FUNCTION:
            return self._function(tree, sql, node, ast, work)

        if action == _STAR:
            return self._star(tree, sql, node, ast, work)

        if action == _CASE:
            return self._case(tree, sql, node, ast, work)

        if action == _CAST:
            return self._cast(tree, sql, node, ast, work)

        if action == _LIST:
            return self._list(tree, sql, node, ast, work)

        if action == _STRUCT:
            return self._struct(tree, sql, node, ast, work)

        if action == _PARAMETER:
            return self._parameter(tree, sql, node, ast)

        if action == _COALESCE:
            # `COALESCE Parens(List(Expression))`.
            return self._named_call(
                ast,
                work,
                self._items(tree, self._only(tree, node)),
                "coalesce",
                at,
            )

        if action == _NULLIF:
            # `NULLIF Parens(NullIfArguments)`, and the arguments rule holds
            # the two expressions itself rather than a list.
            return self._named_call(
                ast,
                work,
                tree.children(self._only(tree, self._only(tree, node))),
                "nullif",
                at,
            )

        if action == _STRING:
            return ast.literal(
                LITERAL_STRING, _string_value(sql, tree.tokens[Int(at)]), at
            )

        if action == _NUMBER:
            return ast.literal(
                LITERAL_NUMBER, _number_value(sql, tree.tokens[Int(at)]), at
            )

        if action == _NULL:
            return ast.literal(LITERAL_NULL, "", at)

        if action == _TRUE:
            return ast.literal(LITERAL_BOOLEAN, "TRUE", at)

        if action == _FALSE:
            return ast.literal(LITERAL_BOOLEAN, "FALSE", at)

        raise _no_case(tree, sql, node)

    def _fold(
        self,
        tree: Parse,
        sql: StringSlice,
        node: UInt32,
        mut ast: Ast,
        mut work: Work,
    ) raises -> UInt32:
        """Folds one precedence level, left to right.

        Args:
            tree: The parse.
            sql: The query.
            node: An `X <- Y Tail*` node.
            ast: Where to put the nodes.
            work: The walk, for the values of the operands.

        Returns:
            The expression node.

        Raises:
            Error: If a tail is shaped in a way this has no case for.
        """
        var kids = tree.children(node)
        if len(kids) == 0:
            raise _malformed(tree, sql, node, "a precedence level with nothing")

        # Every operand is asked for before anything is built, so a level with
        # a long run of tails is built once and not once per tail.
        var operands = List[UInt32]()
        operands.append(kids[0])
        for i in range(1, len(kids)):
            var inner = tree.children(kids[i])
            if len(inner) == 0:
                raise _malformed(tree, sql, kids[i], "a tail with no operand")
            operands.append(inner[len(inner) - 1])
        work.warm(operands)

        var left = work.value(kids[0])
        for i in range(1, len(kids)):
            left = self._tail(tree, sql, kids[i], ast, work, left)
        return left

    def _tail(
        self,
        tree: Parse,
        sql: StringSlice,
        tail: UInt32,
        mut ast: Ast,
        mut work: Work,
        left: UInt32,
    ) raises -> UInt32:
        """Applies one `Op Right` tail to what is on its left.

        The operator is whatever tokens sit between the start of the tail and
        the start of its operand, which is one rule for every level whether the
        grammar spells the operator as a literal in the tail or as a rule of
        its own. A `NOT` between the two belongs to the operand and not to the
        operator, which is the one thing the span rule cannot see on its own.

        Args:
            tree: The parse.
            sql: The query.
            tail: The tail node.
            ast: Where to put the nodes.
            work: The walk, for the value of the operand.
            left: What the tail applies to.

        Returns:
            The expression node.

        Raises:
            Error: If the operator is one this has no case for.
        """
        var kids = tree.children(tail)
        if len(kids) == 0:
            raise _malformed(tree, sql, tail, "a tail with no operand")

        var operand = kids[len(kids) - 1]
        var start = tree.nodes[Int(tail)].token_start
        var cut = tree.nodes[Int(operand)].token_start
        var negation = NO_NODE
        for i in range(len(kids) - 1):
            if _word(tree, sql, kids[i]) == "NOT":
                negation = kids[i]
                cut = tree.nodes[Int(kids[i])].token_start
                break

        var operator = _span(tree, sql, start, cut)
        if Int(cut) - Int(start) > 1 and not _multiword(operator):
            raise _unsupported(
                tree, sql, tail, String(operator, " as an operator")
            )

        var right = work.value(operand)
        if negation != NO_NODE:
            right = self._negate(tree, negation, ast, right)
        return ast.binary(operator, left, right, start)

    def _collate(
        self,
        tree: Parse,
        sql: StringSlice,
        node: UInt32,
        mut ast: Ast,
        mut work: Work,
    ) raises -> UInt32:
        """Folds `x COLLATE c`, whose right side names a collation.

        The grammar puts a whole expression on the right of `COLLATE` because
        that is what the level below it produces, but only a bare name is
        meaningful there, so anything else refuses rather than being stored as
        a collation nobody can look up.

        Args:
            tree: The parse.
            sql: The query.
            node: The `CollateExpression` node.
            ast: Where to put the nodes.
            work: The walk, for the values of the operands.

        Returns:
            The expression node.

        Raises:
            Error: If the right side is not a plain name.
        """
        var kids = tree.children(node)
        if len(kids) == 0:
            raise _malformed(tree, sql, node, "a collate level with nothing")

        var operands = List[UInt32]()
        operands.append(kids[0])
        for i in range(1, len(kids)):
            var inner = tree.children(kids[i])
            if len(inner) == 0:
                raise _malformed(tree, sql, kids[i], "a collate with no name")
            operands.append(inner[len(inner) - 1])
        work.warm(operands)

        var left = work.value(kids[0])
        for i in range(1, len(kids)):
            var tail = tree.children(kids[i])
            var name_node = tail[len(tail) - 1]
            var names = self._name_of(tree, sql, name_node, ast, work)
            left = ast.collate(
                left, names, tree.nodes[Int(kids[i])].token_start
            )
        return left

    def _name_of(
        self,
        tree: Parse,
        sql: StringSlice,
        node: UInt32,
        mut ast: Ast,
        mut work: Work,
    ) raises -> String:
        """Reads a sub tree that has to be a single unqualified name.

        Args:
            tree: The parse.
            sql: The query.
            node: The node.
            ast: Where the expression it transforms to goes, and is discarded.
            work: The walk, for the value of the node.

        Returns:
            The name.

        Raises:
            Error: If it is anything but one name part.
        """
        var built = work.value(node)
        ref item = ast.exprs[Int(built)]
        if item.kind != EXPR_COLUMN or ast.length(item.children) != 1:
            raise _unsupported(
                tree, sql, node, "anything but a plain name here"
            )
        return String(ast.text(ast.at(item.children, 0)))

    def _is(
        self,
        tree: Parse,
        sql: StringSlice,
        node: UInt32,
        mut ast: Ast,
        mut work: Work,
    ) raises -> UInt32:
        """Applies a run of `IS` tests to an operand.

        `IS NULL`, `NOTNULL` and `ISNULL` all mean the same test, so all three
        become the same node and the printer writes back the one spelling. `IS
        UNKNOWN` does not, because the value it compares against has no literal
        kind, and inventing one would be a typing decision this stage does not
        get to make.

        Args:
            tree: The parse.
            sql: The query.
            node: The `IsExpression` node.
            ast: Where to put the nodes.
            work: The walk, for the values on the right of a test.

        Returns:
            The expression node.

        Raises:
            Error: If a test is one this has no case for.
        """
        var kids = tree.children(node)
        if len(kids) == 0:
            raise _malformed(tree, sql, node, "an is level with nothing")

        # Every test is read first and every value it needs asked for, so a run
        # of them is built once and not once per test. A value of `NO_NODE`
        # means the test compares against `NULL`, which is spelled by the
        # operator itself and has nothing to ask for.
        var operators = List[String]()
        var values = List[UInt32]()
        var wheres = List[UInt32]()

        for i in range(1, len(kids)):
            var test = self._only(tree, kids[i])
            var at = tree.nodes[Int(test)].token_start
            var lead = _word(tree, sql, test)

            if lead == "ISNULL":
                operators.append("IS")
                values.append(NO_NODE)
                wheres.append(at)
                continue
            if lead == "NOTNULL" or lead == "NOT":
                operators.append("IS NOT")
                values.append(NO_NODE)
                wheres.append(at)
                continue
            if lead != "IS":
                raise _no_case(tree, sql, test)

            # `IsLiteral <- 'IS' 'NOT'? IsLiteralValue`, so the one child is
            # the value and the `NOT` is a literal in between.
            var value = self._only(tree, test)
            var negated = Int(tree.nodes[Int(value)].token_start) - Int(at) > 1
            if _word(tree, sql, value) == "UNKNOWN":
                raise _unsupported(tree, sql, test, "IS UNKNOWN")
            operators.append("IS NOT" if negated else "IS")
            values.append(value)
            wheres.append(at)

        var operands = values.copy()
        operands.append(kids[0])
        work.warm(operands)

        var left = work.value(kids[0])
        for i in range(len(operators)):
            var at = wheres[i]
            var right = ast.literal(LITERAL_NULL, "", at) if values[
                i
            ] == NO_NODE else work.value(values[i])
            left = ast.binary(operators[i], left, right, at)
        return left

    def _between_in_like(
        self,
        tree: Parse,
        sql: StringSlice,
        node: UInt32,
        mut ast: Ast,
        mut work: Work,
    ) raises -> UInt32:
        """Applies `BETWEEN`, `IN` or a `LIKE` family operator to an operand.

        Args:
            tree: The parse.
            sql: The query.
            node: The `BetweenInLikeExpression` node.
            ast: Where to put the nodes.
            work: The walk, for the values on either side.

        Returns:
            The expression node.

        Raises:
            Error: If the right side is one this has no case for.
        """
        var kids = tree.children(node)
        var left = work.value(kids[0])
        if len(kids) == 1:
            return left

        var op = kids[1]
        var at = tree.nodes[Int(op)].token_start
        var negated = _word(tree, sql, op) == "NOT"
        var which = self._only(tree, self._only(tree, op))
        var lead = _word(tree, sql, which)
        var inner = tree.children(which)

        if lead == "BETWEEN":
            if len(inner) != 2:
                raise _malformed(
                    tree, sql, which, "a between without two bounds"
                )
            work.warm(inner)
            return ast.between(
                left,
                work.value(inner[0]),
                work.value(inner[1]),
                negated,
                at,
            )

        if lead == "IN":
            # `InExpression` has three forms and only the parenthesized list is
            # representable here. The subquery form refuses inside the walk,
            # naming the statement rule, and the unparenthesized form refuses
            # here because `x IN y` over a list column is not `x IN (y)`.
            var right = self._only(tree, self._only(tree, which))
            if _first_byte(tree, sql, right) != _LEFT_PAREN:
                raise _unsupported(
                    tree, sql, right, "IN over an unparenthesized value"
                )
            var candidates = List[UInt32]()
            var items = self._items(tree, self._only(tree, right))
            work.warm(items)
            for item in items:
                candidates.append(work.value(item))
            return ast.in_list(left, candidates, negated, at)

        # `LikeClause <- LikeVariations OtherOperatorExpression EscapeClause?`.
        if len(inner) > 2:
            raise _unsupported(tree, sql, inner[2], "ESCAPE on a LIKE")
        if len(inner) != 2:
            raise _malformed(tree, sql, which, "a like without an operand")
        var operator = _span(
            tree,
            sql,
            tree.nodes[Int(inner[0])].token_start,
            tree.nodes[Int(inner[1])].token_start,
        )
        var built = ast.binary(operator, left, work.value(inner[1]), at)
        if negated:
            built = ast.unary("NOT", built, at)
        return built

    def _prefix(
        self,
        tree: Parse,
        sql: StringSlice,
        node: UInt32,
        mut ast: Ast,
        mut work: Work,
    ) raises -> UInt32:
        """Wraps an operand in whatever prefix operators are in front of it.

        Args:
            tree: The parse.
            sql: The query.
            node: The `PrefixExpression` node.
            ast: Where to put the nodes.
            work: The walk, for the value of the operand.

        Returns:
            The expression node.

        Raises:
            Error: If an operator is one this has no case for.
        """
        var kids = tree.children(node)
        var last = len(kids) - 1
        var built = work.value(kids[last])
        for i in range(last - 1, -1, -1):
            var operator = kids[i]
            if _tokens(tree, operator) != 1:
                raise _unsupported(
                    tree, sql, operator, "OPERATOR(...) as a prefix"
                )
            var at = tree.nodes[Int(operator)].token_start
            built = ast.unary(_span(tree, sql, at, at + 1), built, at)
        return built

    def _base(
        self,
        tree: Parse,
        sql: StringSlice,
        node: UInt32,
        mut ast: Ast,
        mut work: Work,
    ) raises -> UInt32:
        """Applies casts and field accesses that follow an operand.

        Args:
            tree: The parse.
            sql: The query.
            node: The `BaseExpression` node.
            ast: Where to put the nodes.
            work: The walk, for the value of the operand.

        Returns:
            The expression node.

        Raises:
            Error: If an indirection is one this has no case for.
        """
        var kids = tree.children(node)
        var built = work.value(kids[0])
        if len(kids) == 1:
            return built

        for step in tree.children(kids[1]):
            var what = self._only(tree, step)
            var at = tree.nodes[Int(what)].token_start
            var lead = _first_byte(tree, sql, what)

            if lead == _DOT:
                # `.name` on a name is one longer name. `.name` on anything
                # else is a struct field access, which needs a kind of its own
                # and a binder that can tell a field from a column.
                if _tokens(tree, what) != 2:
                    raise _unsupported(tree, sql, what, "a method call")
                if ast.exprs[Int(built)].kind != EXPR_COLUMN:
                    raise _unsupported(tree, sql, what, "a field access")
                var names = ast.exprs[Int(built)].children
                var parts = List[String]()
                for i in range(ast.length(names)):
                    parts.append(String(ast.text(ast.at(names, i))))
                parts.append(_identifier(sql, tree.tokens[Int(at) + 1]))
                built = ast.column(parts, tree.nodes[Int(node)].token_start)
                continue

            if lead == _COLON:
                # `::` reaches the tokenizer as two punctuation tokens, since a
                # colon is punctuation on its own in a struct and a slice, so
                # this looks at the byte rather than at a joined operator.
                built = ast.cast(
                    built,
                    _type_text(tree, sql, self._only(tree, what)),
                    False,
                    at,
                )
                continue

            if lead == _LEFT_BRACKET:
                raise _unsupported(tree, sql, what, "a slice or a subscript")

            raise _unsupported(tree, sql, what, "a postfix operator")
        return built

    def _function(
        self,
        tree: Parse,
        sql: StringSlice,
        node: UInt32,
        mut ast: Ast,
        mut work: Work,
    ) raises -> UInt32:
        """Builds a call.

        Args:
            tree: The parse.
            sql: The query.
            node: The `FunctionExpression` node.
            ast: Where to put the nodes.
            work: The walk, for the values of the arguments.

        Returns:
            The expression node.

        Raises:
            Error: If the call carries a modifier this has no case for.
        """
        var kids = tree.children(node)
        var at = tree.nodes[Int(node)].token_start
        if len(kids) > 2:
            raise _unsupported(
                tree,
                sql,
                kids[2],
                String(_word(tree, sql, kids[2]), " on a call"),
            )

        var parts = self._parts(tree, sql, self._only(tree, kids[0]))
        var names = List[UInt32]()
        for part in parts:
            names.append(ast.intern(part))

        var flags = UInt32(0)
        var arguments = List[UInt32]()

        # `FunctionExpressionArguments <- Parens(FunctionExpressionArgumentList)`
        # and the list rule holds up to four groups: the `DISTINCT` or `ALL` in
        # front, the arguments themselves, an `ORDER BY` and a null treatment.
        var groups = self._only(tree, self._only(tree, kids[1]))
        for group in tree.children(groups):
            var lead = _word(tree, sql, group)
            if lead == "DISTINCT":
                flags |= CALL_DISTINCT
                continue
            if lead == "ALL":
                # The default, so it carries nothing and is not written back.
                continue
            if lead == "ORDER" or lead == "IGNORE" or lead == "RESPECT":
                raise _unsupported(
                    tree, sql, group, String(lead, " inside a call")
                )
            var items = self._items(tree, group)
            work.warm(items)
            for item in items:
                arguments.append(work.value(item))

        # `count(*)` is a call with no arguments and a flag, not a call with one
        # star argument, because the star there is a spelling and not a value.
        if len(arguments) == 1:
            ref only = ast.exprs[Int(arguments[0])]
            var bare = (
                only.kind == EXPR_STAR
                and only.a == NO_NODE
                and only.b == NO_NODE
                and only.children == NO_NODE
                and only.payload == NO_NODE
            )
            if bare:
                flags |= CALL_STAR
                arguments = List[UInt32]()

        return ast.add(
            Expr(
                kind=EXPR_FUNCTION,
                token=at,
                a=flags,
                children=ast.run(arguments),
                payload=ast.run(names),
            )
        )

    def _named_call(
        self,
        mut ast: Ast,
        mut work: Work,
        items: List[UInt32],
        name: StaticString,
        at: UInt32,
    ) raises -> UInt32:
        """Builds a call out of a form the grammar gives its own syntax.

        `COALESCE` and `NULLIF` are functions that happen to be spelled in the
        grammar, so they become ordinary calls and the printer writes them back
        as calls, which parses to the same thing.

        Args:
            ast: Where to put the nodes.
            work: The walk, for the values of the arguments.
            items: The argument nodes, in order.
            name: The function name to give it.
            at: The token the call starts at.

        Returns:
            The expression node.

        Raises:
            Error: If an argument is one this has no case for.
        """
        var arguments = List[UInt32]()
        work.warm(items)
        for item in items:
            arguments.append(work.value(item))
        return ast.call(name, arguments, 0, at)

    def _star(
        self,
        tree: Parse,
        sql: StringSlice,
        node: UInt32,
        mut ast: Ast,
        mut work: Work,
    ) raises -> UInt32:
        """Builds a star and its three modifiers.

        Args:
            tree: The parse.
            sql: The query.
            node: The `StarExpression` node.
            ast: Where to put the nodes.
            work: The walk, for the values a `REPLACE` holds.

        Returns:
            The expression node.

        Raises:
            Error: If a modifier holds something this has no case for.
        """
        var qualifier = List[UInt32]()
        var exclude = List[UInt32]()
        var replace = List[UInt32]()
        var rename = List[UInt32]()

        var kids = tree.children(node)
        for index in range(len(kids)):
            var part = kids[index]

            # The qualifier is the only thing that can come before the star, so
            # it is found by position. Looking for a word would misread
            # `replace.*`, since `REPLACE` is an unreserved keyword and so is a
            # legal name for a table.
            if index == 0 and (
                tree.nodes[Int(part)].token_start
                == tree.nodes[Int(node)].token_start
            ):
                for step in tree.children(part):
                    qualifier.append(
                        ast.intern(
                            _identifier(
                                sql,
                                tree.tokens[
                                    Int(tree.nodes[Int(step)].token_start)
                                ],
                            )
                        )
                    )
                continue

            var lead = _word(tree, sql, part)

            if lead == "EXCLUDE" or lead == "EXCEPT":
                var holder = tree.children(part)
                for name in self._entries(tree, sql, holder[len(holder) - 1]):
                    exclude.append(ast.intern(self._plain(tree, sql, name)))
                continue

            if lead == "REPLACE":
                var entries = self._entries(tree, sql, self._only(tree, part))
                var pairs = List[UInt32]()
                for entry in entries:
                    pairs.append(tree.children(entry)[0])
                work.warm(pairs)
                for entry in entries:
                    var pair = tree.children(entry)
                    var value = work.value(pair[0])
                    replace.append(ast.intern(self._plain(tree, sql, pair[1])))
                    replace.append(value)
                continue

            if lead == "RENAME":
                for entry in self._entries(tree, sql, self._only(tree, part)):
                    var pair = tree.children(entry)
                    rename.append(ast.intern(self._plain(tree, sql, pair[0])))
                    rename.append(ast.intern(self._plain(tree, sql, pair[1])))
                continue

            raise _no_case(tree, sql, part)

        return ast.add(
            Expr(
                kind=EXPR_STAR,
                token=tree.nodes[Int(node)].token_start,
                a=ast.run(exclude),
                b=ast.run(replace),
                children=ast.run(qualifier),
                payload=ast.run(rename),
            )
        )

    def _case(
        self,
        tree: Parse,
        sql: StringSlice,
        node: UInt32,
        mut ast: Ast,
        mut work: Work,
    ) raises -> UInt32:
        """Builds a `CASE`.

        Args:
            tree: The parse.
            sql: The query.
            node: The `CaseExpression` node.
            ast: Where to put the nodes.
            work: The walk, for the values in the arms.

        Returns:
            The expression node.

        Raises:
            Error: If an arm holds something this has no case for.
        """
        var operand = NO_NODE
        var otherwise = NO_NODE
        var arms = List[UInt32]()

        var wanted = List[UInt32]()
        for part in tree.children(node):
            var lead = _word(tree, sql, part)
            if lead == "WHEN":
                var pair = tree.children(part)
                wanted.append(pair[0])
                wanted.append(pair[1])
            elif lead == "ELSE":
                wanted.append(self._only(tree, part))
            else:
                wanted.append(part)
        work.warm(wanted)

        for part in tree.children(node):
            var lead = _word(tree, sql, part)
            if lead == "WHEN":
                var pair = tree.children(part)
                arms.append(work.value(pair[0]))
                arms.append(work.value(pair[1]))
            elif lead == "ELSE":
                otherwise = work.value(self._only(tree, part))
            else:
                operand = work.value(part)

        return ast.case(
            arms, otherwise, operand, tree.nodes[Int(node)].token_start
        )

    def _cast(
        self,
        tree: Parse,
        sql: StringSlice,
        node: UInt32,
        mut ast: Ast,
        mut work: Work,
    ) raises -> UInt32:
        """Builds `CAST(x AS t)` or `TRY_CAST(x AS t)`.

        Args:
            tree: The parse.
            sql: The query.
            node: The `CastExpression` node.
            ast: Where to put the nodes.
            work: The walk, for the value being cast.

        Returns:
            The expression node.

        Raises:
            Error: If the operand is one this has no case for.
        """
        var kids = tree.children(node)
        var tries = _word(tree, sql, kids[0]) == "TRY_CAST"

        # `CastExpression <- CastOrTryCast Parens(CastArguments)` and
        # `CastArguments <- Expression 'AS' Type`.
        var arguments = tree.children(self._only(tree, kids[1]))
        return ast.cast(
            work.value(arguments[0]),
            _type_text(tree, sql, arguments[1]),
            tries,
            tree.nodes[Int(node)].token_start,
        )

    def _list(
        self,
        tree: Parse,
        sql: StringSlice,
        node: UInt32,
        mut ast: Ast,
        mut work: Work,
    ) raises -> UInt32:
        """Builds a list constructor.

        Args:
            tree: The parse.
            sql: The query.
            node: The `ListExpression` node.
            ast: Where to put the nodes.
            work: The walk, for the values of the elements.

        Returns:
            The expression node.

        Raises:
            Error: If it is the `ARRAY(SELECT ...)` form, or an element has no
                case.
        """
        var bounded = self._only(tree, self._only(tree, node))
        if _first_byte(tree, sql, bounded) != _LEFT_BRACKET:
            raise _unsupported(tree, sql, bounded, "ARRAY over a subquery")
        var elements = List[UInt32]()
        var items = self._items(tree, bounded)
        work.warm(items)
        for item in items:
            elements.append(work.value(item))
        return ast.list_of(elements, tree.nodes[Int(node)].token_start)

    def _struct(
        self,
        tree: Parse,
        sql: StringSlice,
        node: UInt32,
        mut ast: Ast,
        mut work: Work,
    ) raises -> UInt32:
        """Builds a struct constructor.

        Args:
            tree: The parse.
            sql: The query.
            node: The `StructExpression` node.
            ast: Where to put the nodes.
            work: The walk, for the values of the fields.

        Returns:
            The expression node.

        Raises:
            Error: If a field value is one this has no case for.
        """
        var names = List[String]()
        var values = List[UInt32]()
        var fields = self._items(tree, node)
        var wanted = List[UInt32]()
        for field in fields:
            wanted.append(tree.children(field)[1])
        work.warm(wanted)
        for field in fields:
            var pair = tree.children(field)
            names.append(self._plain(tree, sql, pair[0]))
            values.append(work.value(pair[1]))
        return ast.struct_of(names, values, tree.nodes[Int(node)].token_start)

    def _parameter(
        self, tree: Parse, sql: StringSlice, node: UInt32, mut ast: Ast
    ) raises -> UInt32:
        """Builds a prepared statement parameter.

        Args:
            tree: The parse.
            sql: The query.
            node: The `Parameter` node.
            ast: Where to put the nodes.

        Returns:
            The expression node.

        Raises:
            Error: If the form is one this has no case for.
        """
        var form = self._only(tree, node)
        var at = tree.nodes[Int(form)].token_start
        var sigil = _span(tree, sql, at, at + 1)
        if _tokens(tree, form) == 1:
            return ast.parameter(sigil, "", at)
        return ast.parameter(
            sigil, _identifier(sql, tree.tokens[Int(at) + 1]), at
        )

    def _negate(
        self, tree: Parse, node: UInt32, mut ast: Ast, operand: UInt32
    ) -> UInt32:
        """Wraps an operand once for every `NOT` in a `NotExpression`.

        `NOT NOT x` is not `x`. It is a cast to boolean and then a double
        negation, and DuckDB keeps both, so this keeps both too.

        Args:
            tree: The parse.
            node: The `NotExpression` node.
            ast: Where to put the nodes.
            operand: What the negations apply to.

        Returns:
            The expression node.
        """
        var at = tree.nodes[Int(node)].token_start
        var built = operand
        for _ in range(_tokens(tree, node)):
            built = ast.unary("NOT", built, at)
        return built

    def _parts(
        self, tree: Parse, sql: StringSlice, node: UInt32
    ) raises -> List[String]:
        """Reads a dotted name, one part per child.

        Every qualification rule in the grammar is `Name '.'` and every leaf
        one is a bare `Identifier`, so in both cases the part is the node's
        first token and the dot after it can be ignored.

        Args:
            tree: The parse.
            sql: The query.
            node: The node whose children are the parts.

        Returns:
            The parts, outermost first.

        Raises:
            Error: If there are no parts, or a part is not a name.
        """
        var out = List[String]()
        for part in tree.children(node):
            var at = tree.nodes[Int(part)].token_start
            out.append(_identifier(sql, tree.tokens[Int(at)]))
        if len(out) == 0:
            raise _malformed(tree, sql, node, "a name with no parts")
        return out^

    def _plain(
        self, tree: Parse, sql: StringSlice, node: UInt32
    ) raises -> String:
        """Reads a node that has to be exactly one name and nothing else.

        Args:
            tree: The parse.
            sql: The query.
            node: The node.

        Returns:
            The name.

        Raises:
            Error: If it is a dotted name, which the modifiers on a star have
                nowhere to put.
        """
        if _tokens(tree, node) != 1:
            raise _unsupported(tree, sql, node, "a dotted name here")
        return _identifier(
            sql, tree.tokens[Int(tree.nodes[Int(node)].token_start)]
        )

    def _items(self, tree: Parse, node: UInt32) raises -> List[UInt32]:
        """The items of a `List(X)` that sits under a rule of its own.

        The generator turns `List(X)` into a `List_X` node with one child per
        item, and every rule that uses one has it as its only child, so this
        steps over that child and hands back what is under it. A node with no
        children at all is the empty list, which is how `[]` and `{}` arrive.

        Args:
            tree: The parse.
            node: The rule the list sits under.

        Returns:
            One node per item, in order.

        Raises:
            Error: If the node under it is not the list it should be.
        """
        if tree.nodes[Int(node)].first_child == NO_NODE:
            return List[UInt32]()
        return tree.children(self._only(tree, node))

    def _entries(
        self, tree: Parse, sql: StringSlice, node: UInt32
    ) raises -> List[UInt32]:
        """The entries of one of the three modifiers a star can carry.

        All three are spelled the same way, as one entry on its own or a
        parenthesized list of them, so the byte the node starts with is what
        tells the two apart.

        Args:
            tree: The parse.
            sql: The query.
            node: The `ExcludeNames`, `ReplaceEntries` or `RenameEntries` node.

        Returns:
            One node per entry, in order.

        Raises:
            Error: If it is shaped like neither.
        """
        var inside = self._only(tree, node)
        if _first_byte(tree, sql, inside) == _LEFT_PAREN:
            return self._items(tree, self._only(tree, inside))
        var out = List[UInt32]()
        out.append(self._only(tree, inside))
        return out^

    def _only(self, tree: Parse, node: UInt32) raises -> UInt32:
        """Returns the one child a pass through rule has.

        Args:
            tree: The parse.
            node: The node.

        Returns:
            The child.

        Raises:
            Error: If it does not have exactly one.
        """
        var first = tree.nodes[Int(node)].first_child
        if first == NO_NODE:
            raise Error(
                String(
                    "the transformer expected one child under rule ",
                    tree.nodes[Int(node)].rule,
                    " and found none",
                )
            )
        return first


def _no_case(tree: Parse, sql: StringSlice, node: UInt32) -> Error:
    """Builds the error a rule with no case raises.

    Args:
        tree: The parse.
        sql: The query.
        node: The node.

    Returns:
        The error.
    """
    return Error(
        String(
            "firepanda does not support this yet: grammar rule ",
            tree.nodes[Int(node)].rule,
            " at ",
            _here(tree, sql, node),
            ". See https://github.com/tamnd/firepanda/issues/307",
        )
    )


def _unsupported(
    tree: Parse, sql: StringSlice, node: UInt32, what: StringSlice
) -> Error:
    """Builds the error a form with no case raises.

    Args:
        tree: The parse.
        sql: The query.
        node: The node.
        what: The form, named the way somebody reading the query would name it.

    Returns:
        The error.
    """
    return Error(
        String(
            "firepanda does not support ",
            what,
            " yet, at ",
            _here(tree, sql, node),
            ". See https://github.com/tamnd/firepanda/issues/307",
        )
    )


def _malformed(
    tree: Parse, sql: StringSlice, node: UInt32, what: StringSlice
) -> Error:
    """Builds the error a parse tree that cannot happen raises.

    Args:
        tree: The parse.
        sql: The query.
        node: The node.
        what: What was found.

    Returns:
        The error.
    """
    return Error(
        String(
            "the parse tree holds ",
            what,
            " at ",
            _here(tree, sql, node),
            (
                ", which the grammar should not allow, so this is a bug in"
                " firepanda rather than in the query"
            ),
        )
    )


def _here(tree: Parse, sql: StringSlice, node: UInt32) -> String:
    """Quotes the text a node covers, for an error message.

    Args:
        tree: The parse.
        sql: The query.
        node: The node.

    Returns:
        The text, cut short if it is long.
    """
    var start = tree.nodes[Int(node)].token_start
    var end = tree.nodes[Int(node)].token_end
    if end <= start:
        return String("the end of the query")
    var first = tree.tokens[Int(start)]
    var last = tree.tokens[Int(end) - 1]
    var from_byte = Int(first.start)
    var to_byte = Int(last.start) + Int(last.length)
    var text = StringSlice(unsafe_from_utf8=sql.as_bytes()[from_byte:to_byte])
    if text.byte_length() > 60:
        return String(text[byte=0:60], " ...")
    return String(text)


def _tokens(tree: Parse, node: UInt32) -> Int:
    """Returns how many tokens a node covers.

    Args:
        tree: The parse.
        node: The node.

    Returns:
        The count.
    """
    return Int(tree.nodes[Int(node)].token_end) - Int(
        tree.nodes[Int(node)].token_start
    )


def _first_byte(tree: Parse, sql: StringSlice, node: UInt32) -> Byte:
    """Returns the first byte of a node's first token.

    Args:
        tree: The parse.
        sql: The query.
        node: The node.

    Returns:
        The byte, or 0 for a node that covers nothing.
    """
    if _tokens(tree, node) == 0:
        return 0
    var token = tree.tokens[Int(tree.nodes[Int(node)].token_start)]
    if token.length == 0:
        return 0
    return sql.as_bytes()[Int(token.start)]


def _word(tree: Parse, sql: StringSlice, node: UInt32) -> String:
    """Returns a node's first token in upper case, for telling siblings apart.

    Args:
        tree: The parse.
        sql: The query.
        node: The node.

    Returns:
        The word, empty for a node that covers nothing.
    """
    if _tokens(tree, node) == 0:
        return String()
    var token = tree.tokens[Int(tree.nodes[Int(node)].token_start)]
    return String(token_text(sql, token)).upper()


def _span(tree: Parse, sql: StringSlice, start: UInt32, end: UInt32) -> String:
    """Joins a run of tokens into one operator or type name.

    Keywords come back in upper case and everything else comes back as it was
    written, so `is not distinct from` and `IS NOT DISTINCT FROM` reach the AST
    as the same text and two queries that differ only in case give the same
    tree.

    Args:
        tree: The parse.
        sql: The query.
        start: The first token.
        end: One past the last.

    Returns:
        The joined text, with one space between tokens.
    """
    var out = String()
    for i in range(Int(start), Int(end)):
        if i > Int(start):
            out += " "
        var token = tree.tokens[i]
        var text = token_text(sql, token)
        if token.kind == TOKEN_KEYWORD:
            out += String(text).upper()
        else:
            out += text
    return out^


def _multiword(operator: StringSlice) -> Bool:
    """Says whether a many token operator is one the AST can hold.

    Everything else made of more than one token is `OPERATOR(...)` or an
    `ANY` or `ALL` comparison, both of which need something this stage does not
    have, so they refuse rather than being stored as their own text.

    Args:
        operator: The joined operator text.

    Returns:
        Whether it is allowed.
    """
    return (
        operator == "IS DISTINCT FROM"
        or operator == "IS NOT DISTINCT FROM"
        or operator == "AT TIME ZONE"
    )


def _type_text(tree: Parse, sql: StringSlice, node: UInt32) -> String:
    """Reads a type as text, because this stage resolves no type names.

    Args:
        tree: The parse.
        sql: The query.
        node: The `Type` node.

    Returns:
        The type as written, with its tokens joined by single spaces.
    """
    return _span(
        tree,
        sql,
        tree.nodes[Int(node)].token_start,
        tree.nodes[Int(node)].token_end,
    )


def _identifier(sql: StringSlice, token: Token) raises -> String:
    """Decodes one token that is being used as a name.

    Args:
        sql: The query.
        token: The token.

    Returns:
        The name, folded down if it was bare and unwrapped if it was quoted.

    Raises:
        Error: If the token is not one that can be a name.
    """
    var text = token_text(sql, token)
    if token.kind == TOKEN_IDENTIFIER or token.kind == TOKEN_KEYWORD:
        return String(text).lower()
    if token.kind == TOKEN_QUOTED_IDENTIFIER:
        return _unwrapped(text, _DOUBLE_QUOTE)
    if token.kind == TOKEN_STRING:
        return _string_value(sql, token)
    if token.kind == TOKEN_NUMBER:
        return String(text)
    raise Error(
        String("the transformer wanted a name and found ", text, " instead")
    )


def _number_value(sql: StringSlice, token: Token) -> String:
    """Reads a number literal, with the digit separators taken out.

    Args:
        sql: The query.
        token: The token.

    Returns:
        The digits, which stay text because whether this is an integer, a
        decimal or a double is a typing decision and this stage makes none.
    """
    var bytes = token_text(sql, token).as_bytes()
    var out = List[Byte](capacity=len(bytes))
    for i in range(len(bytes)):
        if bytes[i] != _UNDERSCORE:
            out.append(bytes[i])
    return String(StringSlice(unsafe_from_utf8=Span(out)))


def _string_value(sql: StringSlice, token: Token) raises -> String:
    """Decodes a string literal into the value it means.

    Three of the four spellings are here. A plain `'...'` doubles a quote to
    hold one. A `$tag$...$tag$` string has no escapes at all. Two literals with
    only whitespace and a newline between them are one literal, which is how a
    long string is written across lines, and the tokenizer already hands that
    back as one token. An `E'...'` string is the fourth and it refuses, because
    decoding it means implementing every backslash escape and half of that is
    worse than none of it.

    Args:
        sql: The query.
        token: The token.

    Returns:
        The value, with no quotes and no escapes left in it.

    Raises:
        Error: If it is an `E'...'` string.
    """
    var text = token_text(sql, token)
    if token.flags & FLAG_ESCAPE != 0:
        raise Error(
            String(
                "firepanda does not support an E'...' string yet, at ",
                text,
                ". See https://github.com/tamnd/firepanda/issues/307",
            )
        )

    var bytes = text.as_bytes()
    if token.flags & FLAG_DOLLAR != 0:
        var tag = 1
        while tag < len(bytes) and bytes[tag] != _DOLLAR:
            tag += 1
        tag += 1
        return String(
            StringSlice(unsafe_from_utf8=bytes[tag : len(bytes) - tag])
        )

    # One pass over however many quoted runs the token holds, which is one for
    # an ordinary literal and more for a continued one.
    var out = List[Byte](capacity=len(bytes))
    var i = 0
    while i < len(bytes):
        if bytes[i] != _SINGLE_QUOTE:
            i += 1
            continue
        i += 1
        while i < len(bytes):
            var c = bytes[i]
            if c == _SINGLE_QUOTE:
                if i + 1 < len(bytes) and bytes[i + 1] == _SINGLE_QUOTE:
                    out.append(c)
                    i += 2
                    continue
                i += 1
                break
            out.append(c)
            i += 1
    return String(StringSlice(unsafe_from_utf8=Span(out)))


def _unwrapped(text: StringSlice, quote: Byte) -> String:
    """Takes the quotes off a wrapped token and undoubles the quote inside it.

    Args:
        text: The token text, quotes and all.
        quote: The character it is wrapped in.

    Returns:
        The value.
    """
    var bytes = text.as_bytes()
    var out = List[Byte](capacity=len(bytes))
    var i = 1
    var last = len(bytes) - 1
    while i < last:
        out.append(bytes[i])
        if bytes[i] == quote and i + 1 < last and bytes[i + 1] == quote:
            i += 2
        else:
            i += 1
    return String(StringSlice(unsafe_from_utf8=Span(out)))
