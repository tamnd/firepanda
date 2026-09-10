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
    GROUP_ALL,
    GROUP_CUBE,
    GROUP_EMPTY,
    GROUP_EXPRESSION,
    GROUP_ROLLUP,
    GROUP_SETS,
    GROUP_TUPLE,
    LIMIT_ALL,
    LIMIT_PERCENT,
    LITERAL_BOOLEAN,
    LITERAL_NULL,
    LITERAL_NUMBER,
    LITERAL_STRING,
    MATERIALIZE_DEFAULT,
    MATERIALIZE_NO,
    MATERIALIZE_YES,
    NO_NODE,
    NULLS_DEFAULT,
    NULLS_FIRST,
    NULLS_LAST,
    SELECT_ALL,
    SELECT_DISTINCT,
    SORT_ASCENDING,
    SORT_DEFAULT,
    SORT_DESCENDING,
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
    caret_at,
    token_text,
)
from .unsupported import (
    ALIAS_COLON,
    ARRAY_SUBQUERY,
    CALL_ARGUMENT,
    CALL_MODIFIER,
    COLUMNS,
    CUSTOM_OPERATOR,
    DEFAULT_VALUE,
    DOTTED_NAME,
    ESCAPE_STRING,
    FIELD_ACCESS,
    GROUPING,
    INTERVAL,
    IN_BARE_VALUE,
    IS_UNKNOWN,
    JOIN_FORM,
    LAMBDA,
    LIKE_ESCAPE,
    LIST_COMPREHENSION,
    MAP_LITERAL,
    METHOD_CALL,
    NAMED_ARGUMENT,
    NOT_SUBQUERY,
    NO_CASE,
    OPERATOR,
    POSITIONAL,
    POSTFIX_OPERATOR,
    QUOTED_NAME,
    ROW_VALUE,
    SELECT_CLAUSE,
    SELECT_SAMPLE,
    SPECIAL_CALL,
    STATEMENT_LATER,
    STATEMENT_NEVER,
    SUBSCRIPT,
    TABLE_AT,
    TABLE_MODIFIER,
    TABLE_SAMPLE,
    TYPE_LITERAL,
    WITH_ORDINALITY,
    WITH_USING_KEY,
    not_implemented,
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

comptime _SUBQUERY: UInt8 = 24
"""`SubqueryExpression`, a scalar subquery or an `EXISTS`."""

comptime _SELECT: UInt8 = 25
"""`SelectStatementInternal`, a whole statement with its `WITH` and its tail."""

comptime _SETOP: UInt8 = 26
"""`SelectSetOpChain` and `IntersectChain`, folded left to right."""

comptime _SIMPLE_SELECT: UInt8 = 27
"""`SimpleSelect`, the `SELECT ... FROM ... WHERE ...` block itself."""

comptime _BLOCK: UInt8 = 28
"""`SelectFromClause` and `FromSelectClause`, the two orders of the two."""

comptime _SELECT_CLAUSE: UInt8 = 29
"""`SelectClause`, the `SELECT` word with its `DISTINCT` and its targets."""

comptime _TARGETS: UInt8 = 30
"""`TargetList`, one `STMT_ITEM` per entry."""

comptime _ITEM: UInt8 = 31
"""`ExpressionAsCollabel` and `ExpressionOptIdentifier`, value then name."""

comptime _ITEM_COLON: UInt8 = 32
"""`ColIdExpression`, the `name: value` spelling, which is name then value."""

comptime _FROM: UInt8 = 33
"""`FromClause`, one table reference per entry."""

comptime _TABLE_REF: UInt8 = 34
"""`TableRef`, a reference with however many joins hang off it."""

comptime _BASE_TABLE: UInt8 = 35
comptime _TABLE_SUBQUERY: UInt8 = 36
comptime _PARENS_TABLE: UInt8 = 37
comptime _VALUES_REF: UInt8 = 38
comptime _TABLE_FUNCTION: UInt8 = 39

comptime _GROUP_LIST: UInt8 = 40
comptime _GROUP_ALL: UInt8 = 41
comptime _GROUP_ITEM: UInt8 = 42
comptime _GROUP_EMPTY: UInt8 = 43
comptime _GROUP_NESTED: UInt8 = 44
comptime _GROUP_SETS: UInt8 = 45

comptime _ORDER_LIST: UInt8 = 46
comptime _ORDER_ALL: UInt8 = 47
comptime _ORDER_ITEM: UInt8 = 48

comptime _MODIFIERS: UInt8 = 49
"""`ResultModifiers`, the `ORDER BY`, `LIMIT` and `OFFSET` on the end."""

comptime _TAIL: UInt8 = 50
"""The four orders `LimitOffset` allows, all of them the same three parts."""

comptime _LIMIT: UInt8 = 51
comptime _OFFSET: UInt8 = 52
comptime _FETCH: UInt8 = 53

comptime _WITH: UInt8 = 54
comptime _CTE: UInt8 = 55
comptime _VALUES: UInt8 = 56
comptime _VALUES_ROW: UInt8 = 57
comptime _TABLE_STATEMENT: UInt8 = 58

comptime _STATEMENT_LATER: UInt8 = 59
"""A statement firepanda will run and does not run yet, refused by name."""

comptime _STATEMENT_NEVER: UInt8 = 60
"""A statement firepanda will not run, refused by name.

The two are separate actions rather than one because a user who reads `not yet`
waits and a user who reads `not this` writes their query another way, and
telling them apart is the whole reason `Statement` is what the transformer is
aimed at. A statement that fell out of the matcher instead would say `syntax
error` about text that is perfectly good SQL.
"""

comptime _TOP_LEVEL: UInt8 = 61
"""One statement and the semicolons after it, if there are any.

This is where a statement starts, rather than `Statement` itself, because a
query copied out of a file or a shell ends in a semicolon and a reader who is
told that is a syntax error will not believe it. The rule also matches nothing
at all, which is how a file of nothing but whitespace parses, so this is the one
place the transformer has to look at a missing child instead of trusting the
grammar to have provided one.
"""

comptime _REFUSE: UInt8 = 62
"""A rule that refuses by name, with the name in `refusals`.

There are a lot of these and there will be more, and a feature whose whole
implementation is one sentence of English does not need an action byte and a
dispatch arm of its own. It needs a row in a table.
"""

# A marker is a rule that builds nothing. The rule above it reads it directly,
# and the byte is here so that rule can pick it out of its siblings with the
# same array lookup the dispatch uses. The alternative is guessing from a
# keyword, and `FROM t AS at` is enough to show why that is not good enough.
comptime _MARK_TABLE_ALIAS: UInt8 = 200
comptime _MARK_ALIAS_COLON: UInt8 = 201
comptime _MARK_AT: UInt8 = 202
comptime _MARK_SAMPLE: UInt8 = 203
comptime _MARK_LATERAL: UInt8 = 204
comptime _MARK_ORDINALITY: UInt8 = 205
comptime _MARK_RECURSIVE: UInt8 = 206
comptime _MARK_MATERIALIZED: UInt8 = 207
comptime _MARK_USING_KEY: UInt8 = 208
comptime _MARK_CTE_COLUMNS: UInt8 = 209
comptime _MARK_IN_SELECT: UInt8 = 210
comptime _MARK_JOIN: UInt8 = 211
comptime _MARK_JOIN_ON: UInt8 = 212
comptime _MARK_JOIN_PLAIN: UInt8 = 213

# Where the parts of a statement sit in the small runs that carry them up. A
# rule that has more than one thing to hand its parent puts them in a run of
# named slots, because a walk that holds one result per parse node has one slot
# to put an answer in. These never reach the AST: the rule above unpacks them.
comptime _SELECT_FLAGS: Int = 0
comptime _SELECT_DISTINCT: Int = 1
comptime _SELECT_TARGETS: Int = 2
comptime _SELECT_SLOTS: Int = 3

comptime _BLOCK_TARGETS: Int = 0
comptime _BLOCK_TABLES: Int = 1
comptime _BLOCK_SLOTS: Int = 2

comptime _TAIL_LIMIT: Int = 0
comptime _TAIL_OFFSET: Int = 1
comptime _TAIL_FLAGS: Int = 2
comptime _TAIL_SLOTS: Int = 3

comptime _WITH_RECURSIVE: Int = 0
comptime _WITH_ENTRIES: Int = 1
comptime _WITH_SLOTS: Int = 2

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

This never reaches a caller. `Transform.walk` catches it, sees the
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

    var refusals: List[UInt16]
    """The refusal a `_REFUSE` rule raises, by rule index, 0 for the rest.

    A second table rather than a second action byte per feature. A rule that
    does nothing but refuse carries no code, only a table entry, and there are
    enough of them that giving each one an action byte and a dispatch arm would
    be a hundred lines saying the same thing a hundred times.
    """

    var expression_rule: Int
    """The index of `Expression`, so a caller can parse one directly."""

    var statement_rule: Int
    """The index of `SelectStatement`, for the same reason."""

    var parens_rule: Int
    """The index of `ParenthesisExpression`, which a grouping entry looks for.
    """

    def __init__(out self, grammar: Grammar) raises:
        """Builds the table.

        Args:
            grammar: A loaded grammar.

        Raises:
            Error: If the grammar has no rule by a name this expects, which
                means a bump renamed something and this file has to follow.
        """
        self.actions = List[UInt8](length=len(grammar.names), fill=_NO_CASE)
        self.refusals = List[UInt16](length=len(grammar.names), fill=0)
        self.expression_rule = -1
        self.statement_rule = -1
        self.parens_rule = -1

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
            # The statement side. Every one of these is a rule the grammar
            # needed a name for and the language does not, so it is one child
            # and nothing else.
            "SelectStatement",
            "SelectAtom",
            "SelectParens",
            "SelectStatementType",
            "OptionalParensSimpleSelect",
            "SimpleSelectParens",
            "SelectFrom",
            "AliasedExpression",
            "SubqueryReference",
            "InnerTableRef",
            "TableFunction",
            "CTEBody",
            "CTESelectBody",
            "WhereClause",
            "HavingClause",
            "QualifyClause",
            "OnClause",
            "GroupByClause",
            "GroupByExpressions",
            "GroupByExpression",
            "OrderByClause",
            "OrderByExpressions",
            "LimitOffset",
            "FetchValue",
            "Parens_SelectStatementInternal",
            "Parens_SimpleSelect",
            "Parens_TableRef",
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

        self._set(grammar, "SubqueryExpression", _SUBQUERY)
        self._set(grammar, "SelectStatementInternal", _SELECT)
        self._set(grammar, "SelectSetOpChain", _SETOP)
        self._set(grammar, "IntersectChain", _SETOP)
        self._set(grammar, "SimpleSelect", _SIMPLE_SELECT)
        self._set(grammar, "SelectFromClause", _BLOCK)
        self._set(grammar, "FromSelectClause", _BLOCK)
        self._set(grammar, "SelectClause", _SELECT_CLAUSE)
        self._set(grammar, "TargetList", _TARGETS)
        self._set(grammar, "ExpressionAsCollabel", _ITEM)
        self._set(grammar, "ExpressionOptIdentifier", _ITEM)
        self._set(grammar, "ColIdExpression", _ITEM_COLON)
        self._set(grammar, "FromClause", _FROM)
        self._set(grammar, "TableRef", _TABLE_REF)
        self._set(grammar, "BaseTableRef", _BASE_TABLE)
        self._set(grammar, "TableSubquery", _TABLE_SUBQUERY)
        self._set(grammar, "ParensTableRef", _PARENS_TABLE)
        self._set(grammar, "ValuesRef", _VALUES_REF)
        self._set(grammar, "TableFunctionLateralOpt", _TABLE_FUNCTION)
        self._set(grammar, "GroupByList", _GROUP_LIST)
        self._set(grammar, "GroupByAll", _GROUP_ALL)
        self._set(grammar, "GroupByBaseExpression", _GROUP_ITEM)
        self._set(grammar, "EmptyGroupingItem", _GROUP_EMPTY)
        self._set(grammar, "CubeOrRollupClause", _GROUP_NESTED)
        self._set(grammar, "GroupingSetsClause", _GROUP_SETS)
        self._set(grammar, "OrderByExpressionList", _ORDER_LIST)
        self._set(grammar, "OrderByAll", _ORDER_ALL)
        self._set(grammar, "OrderByExpression", _ORDER_ITEM)
        self._set(grammar, "ResultModifiers", _MODIFIERS)
        self._set(grammar, "LimitOffsetClause", _TAIL)
        self._set(grammar, "OffsetLimitClause", _TAIL)
        self._set(grammar, "OffsetFetchClause", _TAIL)
        self._set(grammar, "FetchOnlyClause", _TAIL)
        self._set(grammar, "LimitClause", _LIMIT)
        self._set(grammar, "OffsetClause", _OFFSET)
        self._set(grammar, "FetchClause", _FETCH)
        self._set(grammar, "WithClause", _WITH)
        self._set(grammar, "WithStatement", _CTE)
        self._set(grammar, "ValuesClause", _VALUES)
        self._set(grammar, "ValuesExpressions", _VALUES_ROW)
        self._set(grammar, "TableStatement", _TABLE_STATEMENT)

        # `Statement` is thirty seven alternatives and one of them is the one
        # firepanda runs. It descends into whichever matched, and every other
        # one carries a refusal, so a `CREATE INDEX` gets a sentence naming it
        # rather than a syntax error about text that is perfectly good SQL.
        # docs/specs/sql/05-ast-and-binder.md section 4, tiers two and three.
        self._set(grammar, "Statement", _DESCEND)
        self._set(grammar, "TopLevelStatement", _TOP_LEVEL)

        # Tier two. Each of these is a frame operation with SQL spelling, so it
        # says `not yet` and points at the milestone.
        var later: List[StaticString] = [
            "CreateStatement",
            "InsertStatement",
            "CopyStatement",
            "ExplainStatement",
            "PrepareStatement",
            "ExecuteStatement",
            "DeallocateStatement",
            "SetStatement",
            "ResetStatement",
            "PragmaStatement",
        ]
        for name in later:
            self._set(grammar, name, _STATEMENT_LATER)

        # Tier three. Each of these wants a catalog, a transaction, a file on
        # disk that outlives the process, or an extension, and firepanda has
        # none of those. `UPDATE` and `DELETE` are here rather than above on
        # purpose: they are expressible over an immutable frame as a rewrite,
        # they are not what a dataframe user reaches for, and half of them is
        # worse than none of them.
        var never: List[StaticString] = [
            "AlterStatement",
            "AnalyzeStatement",
            "AttachStatement",
            "CallStatement",
            "CheckpointStatement",
            "CommentStatement",
            "ConnectStatement",
            "DeleteStatement",
            "DetachStatement",
            "DisconnectStatement",
            "DropStatement",
            "ExportStatement",
            "ExpressionStatement",
            "ExtensionRepositoryStatement",
            "ExternalResourceStatement",
            "ImportStatement",
            "InstallStatement",
            "LoadStatement",
            "MergeIntoStatement",
            "TransactionStatement",
            "TruncateStatement",
            "UpdateExtensionsStatement",
            "UpdateStatement",
            "UseStatement",
            "VacuumStatement",
        ]
        for name in never:
            self._set(grammar, name, _STATEMENT_NEVER)

        # `DescribeStatement`, `PivotStatement` and `UnpivotStatement` hang off
        # `SelectStatementType` rather than off `Statement`, because all three
        # produce rows, so the loop above never reached them. They are the same
        # kind of thing it refuses: a query firepanda will run and does not run
        # yet.
        self._set(grammar, "DescribeStatement", _STATEMENT_LATER)
        self._set(grammar, "PivotStatement", _STATEMENT_LATER)
        self._set(grammar, "UnpivotStatement", _STATEMENT_LATER)

        # `CTEDMLBody <- Parens(Statement)` is `WITH x AS (INSERT ...)`. Walking
        # into it costs nothing and the statement inside says its own name, so
        # the user reads `the INSERT statement yet` rather than the name of the
        # wrapper, which is a rule they did not write and cannot look up. Two
        # rules because the parentheses are a rule of their own.
        self._set(grammar, "CTEDMLBody", _DESCEND)
        self._set(grammar, "Parens_Statement", _DESCEND)

        # Features with no form in the arena yet. Each of these is one sentence
        # of English in `unsupported.mojo` and no code at all, which is what the
        # refusal table is for.
        self._refuse(grammar, "ParenthesisExpression", ROW_VALUE)
        self._refuse(grammar, "RowExpression", ROW_VALUE)
        self._refuse(grammar, "IntervalLiteral", INTERVAL)
        self._refuse(grammar, "TypeLiteral", TYPE_LITERAL)
        self._refuse(grammar, "LambdaExpression", LAMBDA)
        self._refuse(grammar, "ListComprehensionExpression", LIST_COMPREHENSION)
        self._refuse(grammar, "NamedFunctionArgument", NAMED_ARGUMENT)
        self._refuse(grammar, "ColumnsExpression", COLUMNS)
        self._refuse(grammar, "MapExpression", MAP_LITERAL)
        self._refuse(grammar, "GroupingExpression", GROUPING)
        self._refuse(grammar, "PositionalExpression", POSITIONAL)
        self._refuse(grammar, "DefaultExpression", DEFAULT_VALUE)
        self._refuse(grammar, "TableFunctionAliasColon", ALIAS_COLON)

        # The functions SQL spells with keywords inside the parentheses. The
        # message fills in whichever one it was, so they share an entry.
        var special: List[StaticString] = [
            "ExtractExpression",
            "SubstringExpression",
            "TrimExpression",
            "PositionExpression",
            "OverlayExpression",
            "TryExpression",
            "UnpackExpression",
        ]
        for name in special:
            self._refuse(grammar, name, SPECIAL_CALL)

        # The markers, which build nothing and are only ever recognized.
        self._set(grammar, "TableAlias", _MARK_TABLE_ALIAS)
        self._set(grammar, "TableAliasColon", _MARK_ALIAS_COLON)
        self._set(grammar, "AtClause", _MARK_AT)
        self._set(grammar, "SampleClause", _MARK_SAMPLE)
        self._set(grammar, "Lateral", _MARK_LATERAL)
        self._set(grammar, "WithOrdinality", _MARK_ORDINALITY)
        self._set(grammar, "Recursive", _MARK_RECURSIVE)
        self._set(grammar, "Materialized", _MARK_MATERIALIZED)
        self._set(grammar, "UsingKey", _MARK_USING_KEY)
        self._set(grammar, "InsertColumnList", _MARK_CTE_COLUMNS)
        self._set(grammar, "InSelectStatement", _MARK_IN_SELECT)
        self._set(grammar, "JoinClause", _MARK_JOIN)
        self._set(grammar, "RegularJoinClause", _MARK_JOIN_ON)
        self._set(grammar, "JoinWithoutOnClause", _MARK_JOIN_PLAIN)

        self.expression_rule = grammar.rule("Expression")
        self.statement_rule = grammar.rule("TopLevelStatement")
        self.parens_rule = grammar.rule("ParenthesisExpression")

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

    def _refuse(
        mut self, grammar: Grammar, name: StaticString, feature: UInt16
    ) raises:
        """Points one rule at one refusal.

        Args:
            grammar: A loaded grammar.
            name: The rule name, spelled the way the grammar spells it.
            feature: A key into the refusal table in `unsupported.mojo`.

        Raises:
            Error: If there is no such rule.
        """
        self._set(grammar, name, _REFUSE)
        self.refusals[grammar.rule(name)] = feature

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
        return self.walk(tree, sql, tree.root, ast)

    def parse_statement(
        self, sql: StringSlice, grammar: Grammar, mut ast: Ast
    ) raises -> UInt32:
        """Parses one statement and transforms it.

        Args:
            sql: The statement text, with or without a trailing semicolon.
            grammar: The grammar it was parsed against.
            ast: Where to put the nodes.

        Returns:
            The root statement node.

        Raises:
            Error: If the text is not one statement, or is a statement
                firepanda does not run, or holds something this does not
                transform yet.
        """
        var tree = parse_rule(sql, grammar, self.statement_rule)
        return self.walk(tree, sql, tree.root, ast)

    def walk(
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

        if action == _TOP_LEVEL:
            var only = tree.nodes[Int(node)].first_child
            if only == NO_NODE:
                raise Error("Parser Error: syntax error at end of input")
            return work.value(only)

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

        if action == _SUBQUERY:
            return self._subquery(tree, sql, node, ast, work)

        if action == _SELECT:
            return self._select(tree, sql, node, ast, work)

        if action == _SETOP:
            return self._setop(tree, sql, node, ast, work)

        if action == _SIMPLE_SELECT:
            return self._simple_select(tree, sql, node, ast, work)

        if action == _BLOCK:
            return self._block(tree, node, ast, work)

        if action == _SELECT_CLAUSE:
            return self._select_clause(tree, sql, node, ast, work)

        if action == _TARGETS:
            return self._collect(tree, self._items(tree, node), ast, work)

        if action == _ITEM:
            return self._item(tree, sql, node, ast, work, False)

        if action == _ITEM_COLON:
            return self._item(tree, sql, node, ast, work, True)

        if action == _FROM:
            return self._collect(tree, self._items(tree, node), ast, work)

        if action == _TABLE_REF:
            return self._table_ref(tree, sql, node, ast, work)

        if action == _BASE_TABLE:
            return self._base_table(tree, sql, node, ast)

        if action == _TABLE_SUBQUERY:
            return self._table_subquery(tree, sql, node, ast, work)

        if action == _PARENS_TABLE:
            return self._parens_table(tree, sql, node, ast, work)

        if action == _VALUES_REF:
            return self._values_ref(tree, sql, node, ast, work)

        if action == _TABLE_FUNCTION:
            return self._table_function(tree, sql, node, ast, work)

        if action == _GROUP_LIST:
            return self._collect(tree, self._items(tree, node), ast, work)

        if action == _GROUP_ALL:
            var only = List[UInt32]()
            only.append(ast.group(GROUP_ALL, NO_NODE, List[UInt32](), at))
            return ast.run(only)

        if action == _GROUP_ITEM:
            return self._group_item(tree, node, ast, work)

        if action == _GROUP_EMPTY:
            return ast.group(GROUP_EMPTY, NO_NODE, List[UInt32](), at)

        if action == _GROUP_NESTED:
            return self._group_nested(tree, sql, node, ast, work)

        if action == _GROUP_SETS:
            var entries = self._items(tree, self._only(tree, node))
            work.warm(entries)
            var nested = List[UInt32]()
            for entry in entries:
                nested.append(work.value(entry))
            return ast.group(GROUP_SETS, NO_NODE, nested, at)

        if action == _ORDER_LIST:
            return self._collect(tree, self._items(tree, node), ast, work)

        if action == _ORDER_ALL or action == _ORDER_ITEM:
            return self._order(tree, sql, node, ast, work, action == _ORDER_ALL)

        if action == _MODIFIERS:
            return self._modifiers(tree, sql, node, ast, work)

        if action == _TAIL:
            return self._tail_parts(tree, node, ast, work)

        if action == _LIMIT:
            return self._limit(tree, node, ast, work)

        if action == _OFFSET:
            # `OffsetClause <- 'OFFSET' OffsetValue` and
            # `OffsetValue <- Expression RowOrRows?`, so the `ROWS` is a
            # spelling and the expression is the first child under it.
            return work.value(tree.children(self._only(tree, node))[0])

        if action == _FETCH:
            var kids = tree.children(node)
            if len(kids) < 2:
                raise _malformed(tree, sql, node, "a FETCH with no count")
            var parts = List[UInt32](length=_TAIL_SLOTS, fill=NO_NODE)
            parts[_TAIL_LIMIT] = work.value(kids[1])
            return ast.run(parts)

        if action == _WITH:
            return self._with(tree, node, ast, work)

        if action == _CTE:
            return self._cte(tree, sql, node, ast, work)

        if action == _VALUES:
            var rows = List[List[UInt32]]()
            var items = self._items(tree, node)
            work.warm(items)
            for item in items:
                rows.append(ast.items(work.value(item)))
            return ast.values(rows, at)

        if action == _VALUES_ROW:
            return self._collect(
                tree, self._items(tree, self._only(tree, node)), ast, work
            )

        if action == _TABLE_STATEMENT:
            return ast.table_statement(
                self._name_parts(tree, sql, self._only(tree, node)), at
            )

        if action == _STATEMENT_LATER:
            raise _unsupported(
                tree, sql, node, STATEMENT_LATER, _word(tree, sql, node)
            )

        if action == _STATEMENT_NEVER:
            raise _unsupported(
                tree, sql, node, STATEMENT_NEVER, _word(tree, sql, node)
            )

        if action == _REFUSE:
            # The first word goes along whether the message has a slot for it or
            # not, because `filled` drops it when there is no `{}` and most of
            # these messages name the feature themselves.
            raise _unsupported(
                tree, sql, node, self.refusals[rule], _word(tree, sql, node)
            )

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
            raise _unsupported(tree, sql, tail, OPERATOR, operator)

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
            raise _unsupported(tree, sql, node, QUOTED_NAME)
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
                raise _unsupported(tree, sql, test, IS_UNKNOWN)
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
            # `InExpression` has three forms. The subquery form is a kind of
            # its own because the right side lives in the statement arena, the
            # parenthesized list is an ordinary list, and the unparenthesized
            # form refuses because `x IN y` over a list column is not
            # `x IN (y)`.
            var right = self._only(tree, self._only(tree, which))
            if self._marked(tree, right, _MARK_IN_SELECT):
                return ast.in_subquery(
                    left, work.value(self._only(tree, right)), negated, at
                )
            if _first_byte(tree, sql, right) != _LEFT_PAREN:
                raise _unsupported(tree, sql, right, IN_BARE_VALUE)
            var candidates = List[UInt32]()
            var items = self._items(tree, self._only(tree, right))
            work.warm(items)
            for item in items:
                candidates.append(work.value(item))
            return ast.in_list(left, candidates, negated, at)

        # `LikeClause <- LikeVariations OtherOperatorExpression EscapeClause?`.
        if len(inner) > 2:
            raise _unsupported(tree, sql, inner[2], LIKE_ESCAPE)
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
                raise _unsupported(tree, sql, operator, CUSTOM_OPERATOR)
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
                    raise _unsupported(tree, sql, what, METHOD_CALL)
                if ast.exprs[Int(built)].kind != EXPR_COLUMN:
                    raise _unsupported(tree, sql, what, FIELD_ACCESS)
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
                raise _unsupported(tree, sql, what, SUBSCRIPT)

            raise _unsupported(tree, sql, what, POSTFIX_OPERATOR)
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
                CALL_MODIFIER,
                _word(tree, sql, kids[2]),
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
                raise _unsupported(tree, sql, group, CALL_ARGUMENT, lead)
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
            raise _unsupported(tree, sql, bounded, ARRAY_SUBQUERY)
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
            raise _unsupported(tree, sql, node, DOTTED_NAME)
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

    def _action(self, tree: Parse, node: UInt32) -> UInt8:
        """The action byte a node's rule carries.

        Args:
            tree: The parse.
            node: The node.

        Returns:
            The action, `_NO_CASE` for a rule with none.
        """
        return self.actions[Int(tree.nodes[Int(node)].rule)]

    def _marked(self, tree: Parse, node: UInt32, mark: UInt8) -> Bool:
        """Says whether a node is the rule a marker stands for.

        Args:
            tree: The parse.
            node: The node.
            mark: The marker.

        Returns:
            Whether it matches.
        """
        return self._action(tree, node) == mark

    def _collect(
        self, tree: Parse, nodes: List[UInt32], mut ast: Ast, mut work: Work
    ) raises -> UInt32:
        """Builds every node in a list and puts the results in one run.

        This is what a rule that is a list and nothing else does, and there are
        six of them, so they share it.

        Args:
            tree: The parse.
            nodes: The item nodes, in order.
            ast: Where to put the nodes.
            work: The walk, for the values.

        Returns:
            The run handle, which is 0 for an empty list.

        Raises:
            Error: If an item is not built yet, having asked for all of them.
        """
        work.warm(nodes)
        var out = List[UInt32]()
        for item in nodes:
            out.append(work.value(item))
        return ast.run(out)

    def _subquery(
        self,
        tree: Parse,
        sql: StringSlice,
        node: UInt32,
        mut ast: Ast,
        mut work: Work,
    ) raises -> UInt32:
        """Builds a scalar subquery or an `EXISTS`.

        Args:
            tree: The parse.
            sql: The query.
            node: The `SubqueryExpression` node.
            ast: Where to put the nodes.
            work: The walk, for the statement.

        Returns:
            The expression node.

        Raises:
            Error: If it is a `NOT` in front of a subquery that is not an
                `EXISTS`, which is not something the AST holds.
        """
        var at = tree.nodes[Int(node)].token_start
        var negated = False
        var exists = False
        var reference = NO_NODE
        for kid in tree.children(node):
            var lead = _word(tree, sql, kid)
            if lead == "NOT":
                negated = True
            elif lead == "EXISTS":
                exists = True
            else:
                reference = kid
        if reference == NO_NODE:
            raise _malformed(tree, sql, node, "a subquery with no statement")

        var statement = work.value(reference)
        if exists:
            return ast.exists(statement, negated, at)
        if negated:
            raise _unsupported(tree, sql, node, NOT_SUBQUERY)
        return ast.subquery(statement, at)

    def _select(
        self,
        tree: Parse,
        sql: StringSlice,
        node: UInt32,
        mut ast: Ast,
        mut work: Work,
    ) raises -> UInt32:
        """Builds a whole statement, with its `WITH` and its trailing clauses.

        Args:
            tree: The parse.
            sql: The query.
            node: The `SelectStatementInternal` node.
            ast: Where to put the nodes.
            work: The walk, for the parts.

        Returns:
            The statement node.

        Raises:
            Error: If there is no query under it, or a part is not built yet.
        """
        var with_clause = NO_NODE
        var chain = NO_NODE
        var tail = NO_NODE
        for kid in tree.children(node):
            var action = self._action(tree, kid)
            if action == _WITH:
                with_clause = kid
            elif action == _MODIFIERS:
                # `ResultModifiers` matches the empty string, so the node is
                # there even on a statement that ends at the query, and an
                # empty one has nothing to build.
                if tree.nodes[Int(kid)].first_child != NO_NODE:
                    tail = kid
            else:
                chain = kid
        if chain == NO_NODE:
            raise _malformed(tree, sql, node, "a statement with no query")

        var wanted = List[UInt32]()
        wanted.append(chain)
        if with_clause != NO_NODE:
            wanted.append(with_clause)
        if tail != NO_NODE:
            wanted.append(tail)
        work.warm(wanted)

        var ctes = List[UInt32]()
        var recursive = False
        if with_clause != NO_NODE:
            var carried = work.value(with_clause)
            recursive = ast.slot(carried, _WITH_RECURSIVE) == 1
            ctes = ast.items(ast.slot(carried, _WITH_ENTRIES))
        var modifiers = NO_NODE
        if tail != NO_NODE:
            modifiers = work.value(tail)
        return ast.select(
            work.value(chain),
            modifiers,
            ctes,
            recursive,
            tree.nodes[Int(node)].token_start,
        )

    def _setop(
        self,
        tree: Parse,
        sql: StringSlice,
        node: UInt32,
        mut ast: Ast,
        mut work: Work,
    ) raises -> UInt32:
        """Folds a chain of `UNION`, `EXCEPT` or `INTERSECT` left to right.

        The two chain rules are the two precedence levels a set operation has,
        and they are the same shape as the expression chains, so this is the
        same fold with a statement on each side.

        Args:
            tree: The parse.
            sql: The query.
            node: The `SelectSetOpChain` or `IntersectChain` node.
            ast: Where to put the nodes.
            work: The walk, for the operands.

        Returns:
            The statement node.

        Raises:
            Error: If a tail has no operand, or one is not built yet.
        """
        var kids = tree.children(node)
        var wanted = List[UInt32]()
        wanted.append(kids[0])
        for i in range(1, len(kids)):
            var tail = tree.children(kids[i])
            if len(tail) != 2:
                raise _malformed(
                    tree, sql, kids[i], "a set operation with one side"
                )
            wanted.append(tail[1])
        work.warm(wanted)

        var built = work.value(kids[0])
        for i in range(1, len(kids)):
            var tail = tree.children(kids[i])
            var start = tree.nodes[Int(tail[0])].token_start
            # The operator is kept as the words that were written, so that
            # `UNION ALL BY NAME` needs no flags and prints back as itself.
            var operator = _span(
                tree, sql, start, tree.nodes[Int(tail[0])].token_end
            )
            built = ast.set_operation(
                operator, built, work.value(tail[1]), start
            )
        return built

    def _simple_select(
        self,
        tree: Parse,
        sql: StringSlice,
        node: UInt32,
        mut ast: Ast,
        mut work: Work,
    ) raises -> UInt32:
        """Builds one `SELECT ... FROM ... WHERE ...` block.

        Args:
            tree: The parse.
            sql: The query.
            node: The `SimpleSelect` node.
            ast: Where to put the nodes.
            work: The walk, for the clauses.

        Returns:
            The statement node.

        Raises:
            Error: If it carries a clause this has no case for.
        """
        var kids = tree.children(node)
        var block = kids[0]
        var filter = NO_NODE
        var grouping = NO_NODE
        var having = NO_NODE
        var qualify = NO_NODE
        for i in range(1, len(kids)):
            var lead = _word(tree, sql, kids[i])
            if lead == "WHERE":
                filter = kids[i]
            elif lead == "GROUP":
                grouping = kids[i]
            elif lead == "HAVING":
                having = kids[i]
            elif lead == "QUALIFY":
                qualify = kids[i]
            elif self._marked(tree, kids[i], _MARK_SAMPLE):
                raise _unsupported(tree, sql, kids[i], SELECT_SAMPLE)
            else:
                raise _unsupported(tree, sql, kids[i], SELECT_CLAUSE, lead)

        var wanted = List[UInt32]()
        wanted.append(block)
        if filter != NO_NODE:
            wanted.append(filter)
        if grouping != NO_NODE:
            wanted.append(grouping)
        if having != NO_NODE:
            wanted.append(having)
        if qualify != NO_NODE:
            wanted.append(qualify)
        work.warm(wanted)

        var parts = work.value(block)
        var targets = ast.slot(parts, _BLOCK_TARGETS)
        var flags = UInt32(0)
        var distinct_on = List[UInt32]()
        var projection = List[UInt32]()
        if targets != NO_NODE:
            flags = ast.slot(targets, _SELECT_FLAGS)
            distinct_on = ast.items(ast.slot(targets, _SELECT_DISTINCT))
            projection = ast.items(ast.slot(targets, _SELECT_TARGETS))

        var grouped = List[UInt32]()
        if grouping != NO_NODE:
            grouped = ast.items(work.value(grouping))
        return ast.query(
            projection,
            ast.items(ast.slot(parts, _BLOCK_TABLES)),
            work.value(filter) if filter != NO_NODE else NO_NODE,
            grouped,
            work.value(having) if having != NO_NODE else NO_NODE,
            work.value(qualify) if qualify != NO_NODE else NO_NODE,
            flags,
            distinct_on,
            tree.nodes[Int(node)].token_start,
        )

    def _block(
        self, tree: Parse, node: UInt32, mut ast: Ast, mut work: Work
    ) raises -> UInt32:
        """Carries the `SELECT` list and the `FROM` up in one run.

        The two rules for this are the two orders DuckDB allows them in, and
        both mean the same thing, so both come out of here the same way round.

        Args:
            tree: The parse.
            node: The `SelectFromClause` or `FromSelectClause` node.
            ast: Where to put the run.
            work: The walk, for the two parts.

        Returns:
            A run of `_BLOCK_SLOTS` entries.

        Raises:
            Error: If a part is not built yet.
        """
        var targets = NO_NODE
        var tables = NO_NODE
        for kid in tree.children(node):
            if self._action(tree, kid) == _FROM:
                tables = kid
            else:
                targets = kid

        var wanted = List[UInt32]()
        if targets != NO_NODE:
            wanted.append(targets)
        if tables != NO_NODE:
            wanted.append(tables)
        work.warm(wanted)

        var parts = List[UInt32](length=_BLOCK_SLOTS, fill=NO_NODE)
        if targets != NO_NODE:
            parts[_BLOCK_TARGETS] = work.value(targets)
        if tables != NO_NODE:
            parts[_BLOCK_TABLES] = work.value(tables)
        return ast.run(parts)

    def _select_clause(
        self,
        tree: Parse,
        sql: StringSlice,
        node: UInt32,
        mut ast: Ast,
        mut work: Work,
    ) raises -> UInt32:
        """Carries the `DISTINCT` and the target list up in one run.

        Args:
            tree: The parse.
            sql: The query.
            node: The `SelectClause` node.
            ast: Where to put the run.
            work: The walk, for the targets.

        Returns:
            A run of `_SELECT_SLOTS` entries.

        Raises:
            Error: If a part is not built yet.
        """
        var flags = UInt32(0)
        var on = NO_NODE
        var targets = NO_NODE
        for kid in tree.children(node):
            if self._action(tree, kid) == _TARGETS:
                targets = kid
                continue
            # `DistinctClause <- DistinctOn / DistinctAll`, and `ALL` is the
            # default, so it is kept only because writing it back is what the
            # query said.
            var which = self._only(tree, kid)
            if _word(tree, sql, which) == "ALL":
                flags |= SELECT_ALL
                continue
            flags |= SELECT_DISTINCT
            var inner = tree.children(which)
            if len(inner) > 0:
                on = inner[0]

        var wanted = List[UInt32]()
        var chosen = List[UInt32]()
        if on != NO_NODE:
            chosen = self._items(tree, self._only(tree, on))
            for item in chosen:
                wanted.append(item)
        if targets != NO_NODE:
            wanted.append(targets)
        work.warm(wanted)

        var picked = List[UInt32]()
        for item in chosen:
            picked.append(work.value(item))

        var parts = List[UInt32](length=_SELECT_SLOTS, fill=NO_NODE)
        parts[_SELECT_FLAGS] = flags
        parts[_SELECT_DISTINCT] = ast.run(picked)
        if targets != NO_NODE:
            parts[_SELECT_TARGETS] = work.value(targets)
        return ast.run(parts)

    def _item(
        self,
        tree: Parse,
        sql: StringSlice,
        node: UInt32,
        mut ast: Ast,
        mut work: Work,
        name_first: Bool,
    ) raises -> UInt32:
        """Builds one entry of a `SELECT` list.

        Args:
            tree: The parse.
            sql: The query.
            node: The `AliasedExpression` alternative.
            ast: Where to put the nodes.
            work: The walk, for the value.
            name_first: Whether this is the `name: value` spelling.

        Returns:
            The statement node.

        Raises:
            Error: If the value is not built yet.
        """
        var kids = tree.children(node)
        var at = tree.nodes[Int(node)].token_start
        if name_first:
            if len(kids) != 2:
                raise _malformed(tree, sql, node, "a name: with no value")
            return ast.item(
                work.value(kids[1]), self._plain(tree, sql, kids[0]), at
            )
        var value = work.value(kids[0])
        if len(kids) == 1:
            return ast.item(value, "", at)
        return ast.item(value, self._plain(tree, sql, kids[1]), at)

    def _table_ref(
        self,
        tree: Parse,
        sql: StringSlice,
        node: UInt32,
        mut ast: Ast,
        mut work: Work,
    ) raises -> UInt32:
        """Builds a table reference and folds however many joins hang off it.

        `TableRef <- InnerTableRef JoinOrPivot*`, so a chain of joins is left
        nested and the fold matches the shape the grammar gives it. That is
        also why the printer never has to put parentheses around a join side.

        Args:
            tree: The parse.
            sql: The query.
            node: The `TableRef` node.
            ast: Where to put the nodes.
            work: The walk, for the sides and the conditions.

        Returns:
            The table reference node.

        Raises:
            Error: If a join is one this has no case for, or a side is not
                built yet.
        """
        var kids = tree.children(node)
        var wanted = List[UInt32]()
        wanted.append(kids[0])
        for i in range(1, len(kids)):
            var form = self._join_form(tree, sql, kids[i])
            wanted.append(self._join_right(tree, form))
            var on = self._join_on(tree, sql, form)
            if on != NO_NODE:
                wanted.append(on)
        work.warm(wanted)

        var built = work.value(kids[0])
        for i in range(1, len(kids)):
            var form = self._join_form(tree, sql, kids[i])
            var at = tree.nodes[Int(form)].token_start
            var text = _join_text(tree, sql, form)
            var right = work.value(self._join_right(tree, form))
            var on = self._join_on(tree, sql, form)
            if on != NO_NODE:
                built = ast.join(text, built, right, work.value(on), at)
                continue
            var using = self._join_using(tree, form)
            if using == NO_NODE:
                built = ast.join(text, built, right, NO_NODE, at)
                continue
            var columns = List[String]()
            for name in self._items(tree, self._only(tree, using)):
                columns.append(self._plain(tree, sql, name))
            built = ast.join_using(text, built, right, columns, at)
        return built

    def _join_form(
        self, tree: Parse, sql: StringSlice, node: UInt32
    ) raises -> UInt32:
        """Steps from a `JoinOrPivot` down to the join form itself.

        Args:
            tree: The parse.
            sql: The query.
            node: The `JoinOrPivot` node.

        Returns:
            The `RegularJoinClause` or `JoinWithoutOnClause` node.

        Raises:
            Error: If it is a pivot, or a join form this has no case for.
        """
        var clause = self._only(tree, node)
        if not self._marked(tree, clause, _MARK_JOIN):
            raise _unsupported(
                tree, sql, node, TABLE_MODIFIER, _word(tree, sql, node)
            )
        var form = self._only(tree, clause)
        if self._marked(tree, form, _MARK_JOIN_ON) or self._marked(
            tree, form, _MARK_JOIN_PLAIN
        ):
            return form
        raise _unsupported(tree, sql, form, JOIN_FORM)

    def _join_right(self, tree: Parse, form: UInt32) raises -> UInt32:
        """The right side of a join.

        `JoinWithoutOnClause` ends with its table and `RegularJoinClause` ends
        with its qualifier, and everything in front of either is a keyword the
        join text already carries.

        Args:
            tree: The parse.
            form: The join form node.

        Returns:
            The node for the right side.

        Raises:
            Error: If the form has no right side, which the grammar forbids.
        """
        var kids = tree.children(form)
        var back = 1 if self._marked(tree, form, _MARK_JOIN_PLAIN) else 2
        if len(kids) < back:
            raise Error(
                "the parse tree holds a join with no right side, which is a"
                " bug in firepanda rather than in the query"
            )
        return kids[len(kids) - back]

    def _join_on(
        self, tree: Parse, sql: StringSlice, form: UInt32
    ) raises -> UInt32:
        """The `ON` clause of a join, or nothing.

        Args:
            tree: The parse.
            sql: The query.
            form: The join form node.

        Returns:
            The `OnClause` node, or 0 for a `USING` or a join that takes none.

        Raises:
            Error: If the form is shaped in a way the grammar forbids.
        """
        if self._marked(tree, form, _MARK_JOIN_PLAIN):
            return NO_NODE
        var kids = tree.children(form)
        var which = self._only(tree, kids[len(kids) - 1])
        return which if _word(tree, sql, which) == "ON" else NO_NODE

    def _join_using(self, tree: Parse, form: UInt32) raises -> UInt32:
        """The `USING` clause of a join, or nothing.

        Args:
            tree: The parse.
            form: The join form node.

        Returns:
            The `UsingClause` node, or 0.

        Raises:
            Error: If the form is shaped in a way the grammar forbids.
        """
        if self._marked(tree, form, _MARK_JOIN_PLAIN):
            return NO_NODE
        var kids = tree.children(form)
        return self._only(tree, kids[len(kids) - 1])

    def _base_table(
        self, tree: Parse, sql: StringSlice, node: UInt32, mut ast: Ast
    ) raises -> UInt32:
        """Builds a named table reference.

        Args:
            tree: The parse.
            sql: The query.
            node: The `BaseTableRef` node.
            ast: Where to put the nodes.

        Returns:
            The table reference node.

        Raises:
            Error: If it carries a modifier this has no case for.
        """
        var name = NO_NODE
        var named = NO_NODE
        for kid in tree.children(node):
            if self._marked(tree, kid, _MARK_TABLE_ALIAS):
                named = kid
            elif self._marked(tree, kid, _MARK_AT):
                raise _unsupported(tree, sql, kid, TABLE_AT)
            elif self._marked(tree, kid, _MARK_SAMPLE):
                raise _unsupported(tree, sql, kid, TABLE_SAMPLE)
            elif self._marked(tree, kid, _MARK_ALIAS_COLON):
                raise _unsupported(tree, sql, kid, ALIAS_COLON)
            else:
                name = kid
        if name == NO_NODE:
            raise _malformed(tree, sql, node, "a table with no name")
        return ast.table(
            self._name_parts(tree, sql, name),
            self._alias_name(tree, sql, named),
            self._alias_columns(tree, sql, named),
            tree.nodes[Int(node)].token_start,
        )

    def _table_subquery(
        self,
        tree: Parse,
        sql: StringSlice,
        node: UInt32,
        mut ast: Ast,
        mut work: Work,
    ) raises -> UInt32:
        """Builds a subquery in a `FROM`.

        Args:
            tree: The parse.
            sql: The query.
            node: The `TableSubquery` node.
            ast: Where to put the nodes.
            work: The walk, for the statement.

        Returns:
            The table reference node.

        Raises:
            Error: If it carries a modifier this has no case for.
        """
        var reference = NO_NODE
        var named = NO_NODE
        var lateral = False
        for kid in tree.children(node):
            if self._marked(tree, kid, _MARK_TABLE_ALIAS):
                named = kid
            elif self._marked(tree, kid, _MARK_LATERAL):
                lateral = True
            elif self._marked(tree, kid, _MARK_ALIAS_COLON):
                raise _unsupported(tree, sql, kid, ALIAS_COLON)
            else:
                reference = kid
        if reference == NO_NODE:
            raise _malformed(tree, sql, node, "a subquery with no statement")
        return ast.subquery_ref(
            work.value(reference),
            self._alias_name(tree, sql, named),
            self._alias_columns(tree, sql, named),
            lateral,
            tree.nodes[Int(node)].token_start,
        )

    def _parens_table(
        self,
        tree: Parse,
        sql: StringSlice,
        node: UInt32,
        mut ast: Ast,
        mut work: Work,
    ) raises -> UInt32:
        """Builds a parenthesized table reference.

        This node is the reason the printer can print a join's sides bare. The
        parentheses a query wrote are here, and nowhere else, so writing them
        back here is enough.

        Args:
            tree: The parse.
            sql: The query.
            node: The `ParensTableRef` node.
            ast: Where to put the nodes.
            work: The walk, for the reference inside.

        Returns:
            The table reference node.

        Raises:
            Error: If it carries a modifier this has no case for.
        """
        var inner = NO_NODE
        var named = NO_NODE
        for kid in tree.children(node):
            if self._marked(tree, kid, _MARK_TABLE_ALIAS):
                named = kid
            elif self._marked(tree, kid, _MARK_SAMPLE):
                raise _unsupported(tree, sql, kid, TABLE_SAMPLE)
            elif self._marked(tree, kid, _MARK_ALIAS_COLON):
                raise _unsupported(tree, sql, kid, ALIAS_COLON)
            else:
                inner = kid
        if inner == NO_NODE:
            raise _malformed(tree, sql, node, "empty parentheses in a FROM")
        return ast.parens_ref(
            work.value(inner),
            self._alias_name(tree, sql, named),
            self._alias_columns(tree, sql, named),
            tree.nodes[Int(node)].token_start,
        )

    def _values_ref(
        self,
        tree: Parse,
        sql: StringSlice,
        node: UInt32,
        mut ast: Ast,
        mut work: Work,
    ) raises -> UInt32:
        """Builds a `VALUES` used as a table.

        It becomes a subquery reference holding a `VALUES` statement, because
        that is what it is and it prints back the way it was written.

        Args:
            tree: The parse.
            sql: The query.
            node: The `ValuesRef` node.
            ast: Where to put the nodes.
            work: The walk, for the rows.

        Returns:
            The table reference node.

        Raises:
            Error: If it carries a modifier this has no case for.
        """
        var rows = NO_NODE
        var named = NO_NODE
        for kid in tree.children(node):
            if self._marked(tree, kid, _MARK_TABLE_ALIAS):
                named = kid
            elif self._marked(tree, kid, _MARK_ALIAS_COLON):
                raise _unsupported(tree, sql, kid, ALIAS_COLON)
            else:
                rows = kid
        if rows == NO_NODE:
            raise _malformed(tree, sql, node, "a VALUES with no rows")
        return ast.subquery_ref(
            work.value(rows),
            self._alias_name(tree, sql, named),
            self._alias_columns(tree, sql, named),
            False,
            tree.nodes[Int(node)].token_start,
        )

    def _table_function(
        self,
        tree: Parse,
        sql: StringSlice,
        node: UInt32,
        mut ast: Ast,
        mut work: Work,
    ) raises -> UInt32:
        """Builds a table function call in a `FROM`.

        Args:
            tree: The parse.
            sql: The query.
            node: The `TableFunctionLateralOpt` node.
            ast: Where to put the nodes.
            work: The walk, for the arguments.

        Returns:
            The table reference node.

        Raises:
            Error: If it carries a modifier this has no case for.
        """
        var name = NO_NODE
        var arguments = NO_NODE
        var named = NO_NODE
        var lateral = False
        for kid in tree.children(node):
            if self._marked(tree, kid, _MARK_TABLE_ALIAS):
                named = kid
            elif self._marked(tree, kid, _MARK_LATERAL):
                lateral = True
            elif self._marked(tree, kid, _MARK_ORDINALITY):
                raise _unsupported(tree, sql, kid, WITH_ORDINALITY)
            elif name == NO_NODE:
                name = kid
            else:
                arguments = kid
        if name == NO_NODE or arguments == NO_NODE:
            raise _malformed(tree, sql, node, "a table function with no call")

        var items = self._items(tree, self._only(tree, arguments))
        work.warm(items)
        var values = List[UInt32]()
        for item in items:
            values.append(work.value(item))
        return ast.function_ref(
            self._parts(tree, sql, name),
            values,
            self._alias_name(tree, sql, named),
            self._alias_columns(tree, sql, named),
            lateral,
            tree.nodes[Int(node)].token_start,
        )

    def _group_item(
        self, tree: Parse, node: UInt32, mut ast: Ast, mut work: Work
    ) raises -> UInt32:
        """Builds one ordinary `GROUP BY` entry.

        `GROUPING SETS ((a, b))` writes a set of columns the way SQL writes a
        row, so the grammar hands that back as an expression and the tuple has
        to be recognized here. Everywhere else a row is still refused, because
        the AST has no expression kind for one.

        Args:
            tree: The parse.
            node: The `GroupByBaseExpression` node.
            ast: Where to put the nodes.
            work: The walk, for the expression.

        Returns:
            The statement node.

        Raises:
            Error: If the expression is not built yet.
        """
        var at = tree.nodes[Int(node)].token_start
        var tuple = self._tuple(tree, self._only(tree, node))
        if tuple == NO_NODE:
            return ast.group(
                GROUP_EXPRESSION,
                work.value(self._only(tree, node)),
                List[UInt32](),
                at,
            )

        # `ParenthesisExpression <- Parens(List(Expression)?)`.
        var items = self._items(tree, self._only(tree, tuple))
        # `GROUPING SETS ((a, ))` is a set of one column, and so is
        # `GROUPING SETS (a)`, so the one entry row builds as the entry. Leaving
        # the row there would print as `(a)`, which reads back as the plain
        # entry it already was and makes the print of a print differ from the
        # print. The empty row is not this: `()` is the grand total and has to
        # stay a row.
        if len(items) == 1:
            return ast.group(
                GROUP_EXPRESSION, work.value(items[0]), List[UInt32](), at
            )
        work.warm(items)
        var nested = List[UInt32]()
        for item in items:
            nested.append(
                ast.group(
                    GROUP_EXPRESSION, work.value(item), List[UInt32](), at
                )
            )
        return ast.group(GROUP_TUPLE, NO_NODE, nested, at)

    def _tuple(self, tree: Parse, node: UInt32) raises -> UInt32:
        """Finds the row a grouping entry is, if it is one.

        An entry that is a row is a chain of pass through rules down to a
        `ParenthesisExpression` and nothing else, so following the chain while
        it stays one child wide either lands on that rule or does not.

        Args:
            tree: The parse.
            node: The entry's expression node.

        Returns:
            The `ParenthesisExpression` node, or 0 for an entry that is not a
            row.
        """
        var here = node
        while True:
            if Int(tree.nodes[Int(here)].rule) == self.parens_rule:
                return here
            var first = tree.nodes[Int(here)].first_child
            if first == NO_NODE or tree.nodes[Int(first)].next_sibling != 0:
                return NO_NODE
            here = first

    def _group_nested(
        self,
        tree: Parse,
        sql: StringSlice,
        node: UInt32,
        mut ast: Ast,
        mut work: Work,
    ) raises -> UInt32:
        """Builds a `CUBE` or a `ROLLUP`.

        Both hold plain expressions rather than grouping entries, so each one
        is wrapped here to give the entry list one shape.

        Args:
            tree: The parse.
            sql: The query.
            node: The `CubeOrRollupClause` node.
            ast: Where to put the nodes.
            work: The walk, for the expressions.

        Returns:
            The statement node.

        Raises:
            Error: If an expression is not built yet.
        """
        var kids = tree.children(node)
        var at = tree.nodes[Int(node)].token_start
        var tag = (
            GROUP_CUBE if _word(tree, sql, kids[0]) == "CUBE" else GROUP_ROLLUP
        )
        var items = self._items(tree, kids[1])
        work.warm(items)
        var nested = List[UInt32]()
        for item in items:
            nested.append(
                ast.group(
                    GROUP_EXPRESSION, work.value(item), List[UInt32](), at
                )
            )
        return ast.group(tag, NO_NODE, nested, at)

    def _order(
        self,
        tree: Parse,
        sql: StringSlice,
        node: UInt32,
        mut ast: Ast,
        mut work: Work,
        every: Bool,
    ) raises -> UInt32:
        """Builds one `ORDER BY` entry, or the run that `ORDER BY ALL` is.

        Args:
            tree: The parse.
            sql: The query.
            node: The `OrderByExpression` or `OrderByAll` node.
            ast: Where to put the nodes.
            work: The walk, for the expression.
            every: Whether this is `ALL`.

        Returns:
            The statement node, or a run of one for `ALL`.

        Raises:
            Error: If the expression is not built yet.
        """
        var kids = tree.children(node)
        var at = tree.nodes[Int(node)].token_start
        var start = 0 if every else 1
        var expression = NO_NODE
        if not every:
            expression = work.value(kids[0])
        var built = ast.order(
            expression,
            self._direction(tree, sql, kids, start),
            self._nulls(tree, sql, kids, start),
            at,
        )
        if not every:
            return built
        var only = List[UInt32]()
        only.append(built)
        return ast.run(only)

    def _direction(
        self, tree: Parse, sql: StringSlice, kids: List[UInt32], start: Int
    ) -> UInt32:
        """Reads the `ASC` or `DESC` off an `ORDER BY` entry.

        Args:
            tree: The parse.
            sql: The query.
            kids: The entry's children.
            start: The first child that can be a modifier.

        Returns:
            One of the `SORT_` constants.
        """
        for i in range(start, len(kids)):
            var lead = _word(tree, sql, kids[i])
            if lead == "DESC" or lead == "DESCENDING":
                return SORT_DESCENDING
            if lead == "ASC" or lead == "ASCENDING":
                return SORT_ASCENDING
        return SORT_DEFAULT

    def _nulls(
        self, tree: Parse, sql: StringSlice, kids: List[UInt32], start: Int
    ) -> UInt32:
        """Reads the null placement off an `ORDER BY` entry.

        Args:
            tree: The parse.
            sql: The query.
            kids: The entry's children.
            start: The first child that can be a modifier.

        Returns:
            One of the `NULLS_` constants.
        """
        for i in range(start, len(kids)):
            if _word(tree, sql, kids[i]) != "NULLS":
                continue
            var text = _span(
                tree,
                sql,
                tree.nodes[Int(kids[i])].token_start,
                tree.nodes[Int(kids[i])].token_end,
            )
            return NULLS_LAST if text == "NULLS LAST" else NULLS_FIRST
        return NULLS_DEFAULT

    def _modifiers(
        self,
        tree: Parse,
        sql: StringSlice,
        node: UInt32,
        mut ast: Ast,
        mut work: Work,
    ) raises -> UInt32:
        """Builds the `ORDER BY`, `LIMIT` and `OFFSET` that trail a statement.

        Args:
            tree: The parse.
            sql: The query.
            node: The `ResultModifiers` node.
            ast: Where to put the nodes.
            work: The walk, for the parts.

        Returns:
            The statement node.

        Raises:
            Error: If a part is not built yet.
        """
        var order = NO_NODE
        var tail = NO_NODE
        for kid in tree.children(node):
            if _word(tree, sql, kid) == "ORDER":
                order = kid
            else:
                tail = kid

        var wanted = List[UInt32]()
        if order != NO_NODE:
            wanted.append(order)
        if tail != NO_NODE:
            wanted.append(tail)
        work.warm(wanted)

        var entries = List[UInt32]()
        if order != NO_NODE:
            entries = ast.items(work.value(order))
        var limit = NO_NODE
        var offset = NO_NODE
        var flags = UInt32(0)
        if tail != NO_NODE:
            var parts = work.value(tail)
            limit = ast.slot(parts, _TAIL_LIMIT)
            offset = ast.slot(parts, _TAIL_OFFSET)
            flags = ast.slot(parts, _TAIL_FLAGS)
        return ast.modifiers(
            entries, limit, offset, flags, tree.nodes[Int(node)].token_start
        )

    def _tail_parts(
        self, tree: Parse, node: UInt32, mut ast: Ast, mut work: Work
    ) raises -> UInt32:
        """Carries a limit and an offset up in one run, in either order.

        `FETCH FIRST n ROWS ONLY` lands on the same slot `LIMIT n` does, so it
        prints back as `LIMIT n`. Both spellings reparse to the node they came
        from, and the printer is not a formatter.

        Args:
            tree: The parse.
            node: One of the four `LimitOffset` alternatives.
            ast: Where to put the run.
            work: The walk, for the two parts.

        Returns:
            A run of `_TAIL_SLOTS` entries.

        Raises:
            Error: If a part is not built yet.
        """
        var kids = tree.children(node)
        work.warm(kids)
        var parts = List[UInt32](length=_TAIL_SLOTS, fill=NO_NODE)
        for kid in kids:
            if self._action(tree, kid) == _OFFSET:
                parts[_TAIL_OFFSET] = work.value(kid)
                continue
            var carried = work.value(kid)
            parts[_TAIL_LIMIT] = ast.slot(carried, _TAIL_LIMIT)
            parts[_TAIL_FLAGS] = ast.slot(carried, _TAIL_FLAGS)
        return ast.run(parts)

    def _limit(
        self, tree: Parse, node: UInt32, mut ast: Ast, mut work: Work
    ) raises -> UInt32:
        """Builds a `LIMIT`, in its three spellings.

        Args:
            tree: The parse.
            node: The `LimitClause` node.
            ast: Where to put the run.
            work: The walk, for the count.

        Returns:
            A run of `_TAIL_SLOTS` entries.

        Raises:
            Error: If the count is not built yet.
        """
        var form = self._only(tree, self._only(tree, node))
        var parts = List[UInt32](length=_TAIL_SLOTS, fill=NO_NODE)

        # `LimitAll` is the only one of the three with nothing under it.
        if tree.nodes[Int(form)].first_child == NO_NODE:
            parts[_TAIL_FLAGS] = LIMIT_ALL
            return ast.run(parts)

        var inner = self._only(tree, form)
        parts[_TAIL_LIMIT] = work.value(inner)
        if self._action(tree, inner) == _NUMBER:
            # `LimitLiteralPercent <- NumberLiteral 'PERCENT'`, which is the
            # only spelling that reaches a bare literal.
            parts[_TAIL_FLAGS] = LIMIT_PERCENT
        elif _tokens(tree, form) > _tokens(tree, inner):
            # `LimitExpression <- Expression '%'?`, and the extra token is the
            # per cent sign.
            parts[_TAIL_FLAGS] = LIMIT_PERCENT
        return ast.run(parts)

    def _with(
        self, tree: Parse, node: UInt32, mut ast: Ast, mut work: Work
    ) raises -> UInt32:
        """Carries a `WITH` and its `RECURSIVE` up in one run.

        Args:
            tree: The parse.
            node: The `WithClause` node.
            ast: Where to put the run.
            work: The walk, for the entries.

        Returns:
            A run of `_WITH_SLOTS` entries.

        Raises:
            Error: If an entry is not built yet.
        """
        var recursive = False
        var listed = NO_NODE
        for kid in tree.children(node):
            if self._marked(tree, kid, _MARK_RECURSIVE):
                recursive = True
            else:
                listed = kid

        var entries = List[UInt32]()
        if listed != NO_NODE:
            entries = tree.children(listed)
        var parts = List[UInt32](length=_WITH_SLOTS, fill=NO_NODE)
        parts[_WITH_RECURSIVE] = UInt32(1) if recursive else UInt32(0)
        parts[_WITH_ENTRIES] = self._collect(tree, entries, ast, work)
        return ast.run(parts)

    def _cte(
        self,
        tree: Parse,
        sql: StringSlice,
        node: UInt32,
        mut ast: Ast,
        mut work: Work,
    ) raises -> UInt32:
        """Builds one entry of a `WITH`.

        Args:
            tree: The parse.
            sql: The query.
            node: The `WithStatement` node.
            ast: Where to put the nodes.
            work: The walk, for the statement.

        Returns:
            The statement node.

        Raises:
            Error: If it carries a modifier this has no case for.
        """
        var kids = tree.children(node)
        var columns = List[String]()
        var materialize = MATERIALIZE_DEFAULT
        var body = NO_NODE
        for i in range(1, len(kids)):
            if self._marked(tree, kids[i], _MARK_CTE_COLUMNS):
                # `InsertColumnList <- Parens(ColumnList)` and
                # `ColumnList <- List(ColId)`.
                var listed = self._only(tree, self._only(tree, kids[i]))
                for name in self._items(tree, listed):
                    columns.append(self._plain(tree, sql, name))
            elif self._marked(tree, kids[i], _MARK_USING_KEY):
                raise _unsupported(tree, sql, kids[i], WITH_USING_KEY)
            elif self._marked(tree, kids[i], _MARK_MATERIALIZED):
                # `Materialized <- 'NOT'? 'MATERIALIZED'`.
                materialize = (
                    MATERIALIZE_NO if _tokens(tree, kids[i])
                    == 2 else MATERIALIZE_YES
                )
            else:
                body = kids[i]
        if body == NO_NODE:
            raise _malformed(tree, sql, node, "a WITH entry with no statement")
        return ast.cte(
            self._plain(tree, sql, kids[0]),
            work.value(body),
            columns,
            materialize,
            tree.nodes[Int(node)].token_start,
        )

    def _name_parts(
        self, tree: Parse, sql: StringSlice, node: UInt32
    ) raises -> List[String]:
        """Reads a `BaseTableName`, which is one level deeper when qualified.

        `UnqualifiedBaseTableName <- TableName` holds its one part directly and
        `QualifiedTableName` holds a rule that holds the parts, so the parts
        are under whichever of the two has children of its own.

        Args:
            tree: The parse.
            sql: The query.
            node: The `BaseTableName` node.

        Returns:
            The name parts, outermost first.

        Raises:
            Error: If there are no parts.
        """
        var inner = self._only(tree, node)
        var holder = inner
        var below = self._only(tree, inner)
        if tree.nodes[Int(below)].first_child != NO_NODE:
            holder = below
        return self._parts(tree, sql, holder)

    def _alias_name(
        self, tree: Parse, sql: StringSlice, node: UInt32
    ) raises -> String:
        """Reads the name off a `TableAlias`.

        Args:
            tree: The parse.
            sql: The query.
            node: The `TableAlias` node, or 0 for none.

        Returns:
            The name, empty for none.

        Raises:
            Error: If the alias is not one name.
        """
        if node == NO_NODE:
            return String()
        return self._plain(tree, sql, tree.children(self._only(tree, node))[0])

    def _alias_columns(
        self, tree: Parse, sql: StringSlice, node: UInt32
    ) raises -> List[String]:
        """Reads the column aliases off a `TableAlias`.

        Args:
            tree: The parse.
            sql: The query.
            node: The `TableAlias` node, or 0 for none.

        Returns:
            The column names, in order, empty for none.

        Raises:
            Error: If one of them is not one name.
        """
        var out = List[String]()
        if node == NO_NODE:
            return out^
        var kids = tree.children(self._only(tree, node))
        if len(kids) < 2:
            return out^
        # `ColumnAliases <- Parens(List(ColIdOrString))`.
        for name in self._items(tree, self._only(tree, kids[1])):
            out.append(self._plain(tree, sql, name))
        return out^


def _no_case(tree: Parse, sql: StringSlice, node: UInt32) -> Error:
    """Builds the error a rule with no case raises.

    Args:
        tree: The parse.
        sql: The query.
        node: The node.

    Returns:
        The error.
    """
    return _unsupported(
        tree, sql, node, NO_CASE, String(tree.nodes[Int(node)].rule)
    )


def _unsupported(
    tree: Parse,
    sql: StringSlice,
    node: UInt32,
    feature: UInt16,
    detail: StringSlice = "",
) -> Error:
    """Builds the error a form with no case raises.

    The words are the refusal table's rather than this file's, because a
    refusal is worth nothing unless somebody can find out what the whole set of
    them is, and a `raise` written where the cases ran out is not in any set.
    All this adds is the position, which is the one part of the message the
    table cannot know.

    Args:
        tree: The parse.
        sql: The query.
        node: The node the caret goes under.
        feature: The entry in firepanda/sql/unsupported.mojo.
        detail: The text from the query the entry holds a `{}` for.

    Returns:
        The error.
    """
    return not_implemented(feature, detail, _at(tree, sql, node))


def _at(tree: Parse, sql: StringSlice, node: UInt32) -> String:
    """The caret block pointing at where a node starts.

    Args:
        tree: The parse.
        sql: The query.
        node: The node.

    Returns:
        The two lines a refusal quotes, empty if the node starts past the end
        of the query.
    """
    var start = Int(tree.nodes[Int(node)].token_start)
    if start >= len(tree.tokens):
        return String()
    return caret_at(sql.as_bytes(), Int(tree.tokens[start].start))


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


def _join_text(tree: Parse, sql: StringSlice, form: UInt32) raises -> String:
    """Reads a join back as the words it was written with.

    Everything a join says about itself sits in front of the word `JOIN`, so
    the text runs from the form's first token through that word. That is what
    makes `LEFT OUTER JOIN`, `CROSS JOIN` and `NATURAL LEFT JOIN` all come out
    whole with no flags anywhere.

    Args:
        tree: The parse.
        sql: The query.
        form: The join form node.

    Returns:
        The join as SQL spells it.

    Raises:
        Error: If there is no `JOIN` in it, which the grammar forbids.
    """
    var start = Int(tree.nodes[Int(form)].token_start)
    for i in range(start, Int(tree.nodes[Int(form)].token_end)):
        if String(token_text(sql, tree.tokens[i])).upper() != "JOIN":
            continue
        return _span(tree, sql, UInt32(start), UInt32(i + 1))
    raise Error(
        "the parse tree holds a join with no JOIN in it, which is a bug in"
        " firepanda rather than in the query"
    )


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
        raise not_implemented(
            ESCAPE_STRING, "", caret_at(sql.as_bytes(), Int(token.start))
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
