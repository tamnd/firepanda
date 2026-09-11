"""An AST back to SQL text.

Written at the same time as the AST rather than after it, because it pays for
itself three times. Round trip is the transformer's main test: parse, print,
reparse, compare, over the whole corpus, and it catches precedence mistakes that
nothing else catches. It is the generator's oracle, since a random AST can be
printed, parsed and compared. And it is how `EXPLAIN` shows filter and
projection expressions, which is how somebody works out why a pushdown did not
happen. See docs/specs/sql/05-ast-and-binder.md section 6.

It is not a formatter and it does not try to give back the text that was typed.
Two rules follow from that and both are deliberate.

It parenthesizes every operand of an operator rather than working out which
parentheses it could leave out. A printer that minimizes parentheses is a second
implementation of precedence, and therefore a second place to get precedence
wrong, inside the component whose whole job is to check the first one.

Two things are the exception to that, and both are deliberate. A set operation
prints its operands bare and a join prints its sides bare, because the
parentheses a query wrote around either one are already a node of their own.
`STMT_SELECT` under a set operation and `REF_PARENS` under a join sit exactly
where the text had parentheses, so putting them back there and nowhere else is
what makes the printed text read back as the tree it came from.

It quotes an identifier whenever the bare text would not read back as itself.
The tokenizer folds an unquoted word to lower case, so a name with a capital in
it can only have arrived quoted and has to leave quoted. The same goes for a
name holding a character no bare identifier may hold, and for a name that is a
keyword in a class that cannot stand where the name stands.
"""

from .ast import (
    Ast,
    EXPR_BETWEEN,
    EXPR_BINARY,
    EXPR_CASE,
    EXPR_CAST,
    EXPR_COLLATE,
    EXPR_COLUMN,
    EXPR_EXISTS,
    EXPR_FRAME,
    EXPR_FUNCTION,
    EXPR_IN,
    EXPR_IN_SUBQUERY,
    EXPR_LIST,
    EXPR_LITERAL,
    EXPR_PARAMETER,
    EXPR_QUANTIFIED,
    EXPR_STAR,
    EXPR_STRUCT,
    EXPR_SUBQUERY,
    EXPR_UNARY,
    EXPR_WINDOW,
    BOUND_CURRENT_ROW,
    BOUND_FOLLOWING,
    BOUND_NONE,
    BOUND_PRECEDING,
    BOUND_UNBOUNDED_FOLLOWING,
    BOUND_UNBOUNDED_PRECEDING,
    CALL_DISTINCT,
    CALL_STAR,
    CLAUSE_FROM,
    CLAUSE_GROUP,
    CLAUSE_HAVING,
    CLAUSE_PROJECTION,
    CLAUSE_QUALIFY,
    CLAUSE_WHERE,
    CLAUSE_WINDOW,
    EXCLUDE_CURRENT_ROW,
    EXCLUDE_GROUP,
    EXCLUDE_NONE,
    EXCLUDE_NO_OTHERS,
    EXCLUDE_TIES,
    FRAME_GROUPS,
    FRAME_RANGE,
    FRAME_ROWS,
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
    MATERIALIZE_NO,
    MATERIALIZE_YES,
    frame_end,
    frame_exclude,
    frame_mode,
    frame_start,
    NO_NODE,
    NULLS_FIRST,
    NULLS_LAST,
    REF_FUNCTION,
    REF_JOIN,
    REF_JOIN_USING,
    REF_PARENS,
    REF_SUBQUERY,
    REF_TABLE,
    SELECT_ALL,
    SELECT_DISTINCT,
    SORT_ASCENDING,
    SORT_DESCENDING,
    STMT_CTE,
    STMT_GROUP,
    STMT_ITEM,
    STMT_MODIFIERS,
    STMT_ORDER,
    STMT_PIVOT,
    STMT_PIVOT_ON,
    STMT_QUERY,
    STMT_SELECT,
    STMT_SET_OPERATION,
    STMT_TABLE,
    STMT_UNPIVOT,
    STMT_VALUES,
    STMT_WINDOW,
)
from .generated.keywords import (
    KEYWORD_COLUMN_NAME,
    KEYWORD_FUNC_NAME,
    KEYWORD_TYPE_NAME,
    KEYWORD_UNRESERVED,
)
from .table import Grammar

comptime DOUBLE_QUOTE = Byte(ord('"'))
"""What an identifier is wrapped in when it needs wrapping."""

comptime SINGLE_QUOTE = Byte(ord("'"))
"""What a string literal is wrapped in."""


def print_expr(ast: Ast, node: UInt32, grammar: Grammar) raises -> String:
    """Prints one expression.

    Args:
        ast: The AST the node lives in.
        node: The node index.
        grammar: A loaded grammar, for the keyword table the quoting rule
            consults.

    Returns:
        SQL text that parses back to the same shape.

    Raises:
        Error: If the node is the null node, or carries a kind or a tag this
            does not know how to print.
    """
    var out = String()
    _write(ast, node, grammar, out)
    return out^


def print_stmt(ast: Ast, node: UInt32, grammar: Grammar) raises -> String:
    """Prints one statement.

    Args:
        ast: The AST the node lives in.
        node: The index in the statement arena.
        grammar: A loaded grammar, for the keyword table the quoting rule
            consults.

    Returns:
        SQL text that parses back to the same shape.

    Raises:
        Error: If the node is the null node, or carries a kind or a tag this
            does not know how to print.
    """
    var out = String()
    _write_stmt(ast, node, grammar, out)
    return out^


def print_ref(ast: Ast, node: UInt32, grammar: Grammar) raises -> String:
    """Prints one table reference.

    Args:
        ast: The AST the node lives in.
        node: The index in the table reference arena.
        grammar: A loaded grammar, for the keyword table.

    Returns:
        SQL text that parses back to the same shape, as it would read after a
        `FROM`.

    Raises:
        Error: If the node is the null node, or carries a kind this does not
            know how to print.
    """
    var out = String()
    _write_ref(ast, node, grammar, out)
    return out^


def quote_name(
    name: StringSlice, grammar: Grammar, calling: Bool = False
) -> String:
    """Quotes an identifier if the bare text would not read back as itself.

    Args:
        name: The name, as the AST holds it.
        grammar: A loaded grammar, for the keyword table.
        calling: Whether this is the name of a function being called, which is
            a position two more keyword classes may stand in.

    Returns:
        The name, in double quotes when it needs them, with any double quote in
        it doubled.
    """
    if not needs_quoting(name, grammar, calling):
        return String(name)
    return _wrapped(name, DOUBLE_QUOTE)


def quote_string(value: StringSlice) -> String:
    """Puts a string literal back in single quotes.

    Doubling the quote rather than escaping it with a backslash, because a
    doubled quote is the plain `'...'` form and a backslash needs the `E'...'`
    prefix in front of it. Both are in the dialect and only one of them needs
    anything extra.

    Args:
        value: The decoded value, with no quotes on it.

    Returns:
        The literal as SQL spells it.
    """
    return _wrapped(value, SINGLE_QUOTE)


def needs_quoting(
    name: StringSlice, grammar: Grammar, calling: Bool = False
) -> Bool:
    """Says whether an identifier has to be written in double quotes.

    Args:
        name: The name.
        grammar: A loaded grammar, for the keyword table.
        calling: Whether this is the name of a function being called.

    Returns:
        Whether printing it bare would read back as something else.
    """
    var bytes = name.as_bytes()
    if len(bytes) == 0:
        return True
    # By the time a name reaches here a bare one is `[a-z_][a-z0-9_]*`, because
    # the tokenizer folded it on the way in. Anything else was quoted going in
    # and has to be quoted going out.
    var first = bytes[0]
    if not (
        (first >= Byte(ord("a")) and first <= Byte(ord("z")))
        or first == Byte(ord("_"))
    ):
        return True
    for i in range(1, len(bytes)):
        var c = bytes[i]
        var ordinary = (
            (c >= Byte(ord("a")) and c <= Byte(ord("z")))
            or (c >= Byte(ord("0")) and c <= Byte(ord("9")))
            or c == Byte(ord("_"))
        )
        if not ordinary:
            return True
    # Which keywords may stand bare is a property of the position and not of the
    # word, and DuckDB's own classes are what say which. An unreserved keyword
    # and a column name keyword may be a name anywhere, so `coalesce(a, b)`
    # keeps its bare name. A function name keyword and a type name keyword may
    # only be a name where a function is being called, so `left(a, 1)` is bare
    # and the same word as a column is not. A reserved keyword may be neither.
    #
    # Getting this wrong in the quoting direction costs quotes nobody asked for
    # and getting it wrong the other way costs a query that does not parse. The
    # corpus found the second kind: `SELECT "inner" FROM t`, in
    # copy/parquet/parquet_1618_struct_strings.test, printed bare and came back
    # as a syntax error at the FROM.
    var allowed = KEYWORD_UNRESERVED | KEYWORD_COLUMN_NAME
    if calling:
        allowed |= KEYWORD_FUNC_NAME | KEYWORD_TYPE_NAME
    var classes = grammar.keyword_class(name)
    return classes != 0 and classes & allowed == 0


def _wrapped(value: StringSlice, quote: Byte) -> String:
    """Wraps text in a quote character and doubles that character inside it.

    Byte by byte rather than character by character, because the only thing
    being looked for is ASCII and every byte of a multi byte character is above
    127, so no encoding needs decoding to do this correctly.

    Args:
        value: The text.
        quote: The character to wrap it in.

    Returns:
        The quoted text.
    """
    var bytes = value.as_bytes()
    var out = List[Byte](capacity=len(bytes) + 2)
    out.append(quote)
    for i in range(len(bytes)):
        var c = bytes[i]
        out.append(c)
        if c == quote:
            out.append(c)
    out.append(quote)
    return String(StringSlice(unsafe_from_utf8=Span(out)))


def _is_word(text: StringSlice) -> Bool:
    """Says whether an operator is spelled with letters rather than symbols.

    Args:
        text: The operator.

    Returns:
        Whether it needs a space between it and its operand.
    """
    if text.byte_length() == 0:
        return False
    var first = text.as_bytes()[0]
    return (first >= Byte(ord("a")) and first <= Byte(ord("z"))) or (
        first >= Byte(ord("A")) and first <= Byte(ord("Z"))
    )


def _names(
    ast: Ast, run: UInt32, grammar: Grammar, calling: Bool = False
) -> String:
    """Joins a run of interned name parts with dots.

    Args:
        ast: The AST.
        run: A run of pool indices.
        grammar: A loaded grammar, for the quoting rule.
        calling: Whether the last part is the name of a function being called.
            Only the last part is, since everything before it is a schema or a
            catalog and stands where an ordinary name stands.

    Returns:
        The dotted name, empty for an empty run.
    """
    var out = String()
    var count = ast.length(run)
    for i in range(count):
        if i > 0:
            out += "."
        out += quote_name(
            ast.text(ast.at(run, i)), grammar, calling and i == count - 1
        )
    return out^


@fieldwise_init
struct _Step(ImplicitlyCopyable, Movable):
    """A node, and which part of it is being written.

    Eight bytes, so a chain two thousand deep is sixteen kilobytes of heap
    rather than two thousand call frames of stack.
    """

    var node: UInt32
    """The expression node."""

    var phase: UInt32
    """How much of it has been written already.

    Phase 0 is the text before the first child. After that the meaning is the
    node kind's own, and for the kinds with a run of children it is the index
    into that run plus a fixed offset, which is what lets one counter stand for
    a loop that has been turned inside out.
    """


def _write(ast: Ast, node: UInt32, grammar: Grammar, mut out: String) raises:
    """Appends one expression to a buffer.

    The walk is a loop over an explicit stack rather than a function that calls
    itself once per child, for the reason the transformer's walk is. Expression
    depth is not bounded by anything the user cannot type: `x + x + x` folds to
    the left, so two thousand terms is a tree two thousand deep, and the corpus
    has exactly that in `overflow/expression_tree_depth.test`. A recursive
    printer runs out of stack on it and takes the process with it, which is a
    crash rather than an error somebody can read.

    Writing a node is split into phases. Phase 0 writes the text before the
    first child and pushes two things: the same node again at phase 1, and the
    first child at phase 0. The child is on top so it is written first, and when
    it is done the node comes back at the phase after it and writes the text
    between that child and the next. So the text that a recursive printer would
    have written after the recursive call is written by the node's next visit
    instead, and nothing is held on the stack between the two.

    The parts that reach into another arena are still recursive calls, and that
    is deliberate. A subquery goes through `_write_stmt` and a star modifier
    goes through `_write_replace`, and getting to either one costs a set of
    parentheses in the query text, which the matcher caps. Those are bounded by
    what somebody typed, and this is not.

    Args:
        ast: The AST the node lives in.
        node: The node index.
        grammar: A loaded grammar.
        out: The buffer.

    Raises:
        Error: If the node is null, or its kind is not one this knows, or it is
            missing a part the kind says it always has.
    """
    var stack = List[_Step]()
    stack.append(_Step(node, 0))
    while len(stack) > 0:
        var step = stack.pop()
        _write_step(ast, step, grammar, out, stack)


def _write_step(
    ast: Ast,
    step: _Step,
    grammar: Grammar,
    mut out: String,
    mut stack: List[_Step],
) raises:
    """Writes one phase of one node, and pushes whatever comes after it.

    Args:
        ast: The AST the node lives in.
        step: The node and the phase.
        grammar: A loaded grammar.
        out: The buffer.
        stack: The work stack, which this appends to.

    Raises:
        Error: If the node is null, or its kind is not one this knows, or it is
            missing a part the kind says it always has.
    """
    var node = step.node
    var phase = Int(step.phase)

    if node == NO_NODE:
        raise Error("the printer was handed the null node")

    ref item = ast.exprs[Int(node)]
    var kind = item.kind

    if kind == EXPR_LITERAL:
        if item.b == LITERAL_NULL:
            out += "NULL"
        elif item.b == LITERAL_BOOLEAN or item.b == LITERAL_NUMBER:
            out += ast.text(item.payload)
        elif item.b == LITERAL_STRING:
            out += quote_string(ast.text(item.payload))
        else:
            raise Error(String("the printer has no case for literal ", item.b))
        return

    if kind == EXPR_COLUMN:
        if ast.length(item.children) == 0:
            raise Error("a column reference with no name parts in it")
        out += _names(ast, item.children, grammar)
        return

    if kind == EXPR_STAR:
        var qualifier = _names(ast, item.children, grammar)
        if qualifier.byte_length() > 0:
            out += qualifier
            out += "."
        out += "*"
        _write_exclude(ast, item.a, grammar, out)
        _write_replace(ast, item.b, grammar, out)
        _write_rename(ast, item.payload, grammar, out)
        return

    if kind == EXPR_FUNCTION:
        # Phase 0 is the name and the opening parenthesis, and phase 1 onwards
        # walks the argument run, one argument per phase. The same shape does
        # the list, the struct and the two IN forms below.
        var count = ast.length(item.children)
        if phase == 0:
            if ast.length(item.payload) == 0:
                raise Error("a function call with no name on it")
            out += _names(ast, item.payload, grammar, calling=True)
            out += "("
            if item.a & CALL_STAR != 0:
                out += "*)"
                _write_over(ast, item.b, grammar, out)
                return
            if item.a & CALL_DISTINCT != 0:
                out += "DISTINCT "
            stack.append(_Step(node, 1))
            return
        if phase == count + 1:
            out += ")"
            _write_over(ast, item.b, grammar, out)
            return
        var argument = phase - 1
        if argument > 0:
            out += ", "
        stack.append(_Step(node, UInt32(phase + 1)))
        stack.append(_Step(ast.at(item.children, argument), 0))
        return

    if kind == EXPR_UNARY:
        if phase == 0:
            ref operator = ast.text(item.payload)
            out += "("
            out += operator
            if _is_word(operator):
                out += " "
            stack.append(_Step(node, 1))
            stack.append(_Step(item.a, 0))
            return
        out += ")"
        return

    if kind == EXPR_BINARY:
        if phase == 0:
            out += "("
            stack.append(_Step(node, 1))
            stack.append(_Step(item.a, 0))
            return
        if phase == 1:
            out += " "
            out += ast.text(item.payload)
            out += " "
            stack.append(_Step(node, 2))
            stack.append(_Step(item.b, 0))
            return
        out += ")"
        return

    if kind == EXPR_CAST:
        if phase == 0:
            out += "TRY_CAST(" if item.b == 1 else "CAST("
            stack.append(_Step(node, 1))
            stack.append(_Step(item.a, 0))
            return
        out += " AS "
        out += ast.text(item.payload)
        out += ")"
        return

    if kind == EXPR_CASE:
        var arms = ast.length(item.children)
        if phase == 0:
            if arms == 0 or arms % 2 != 0:
                raise Error(String("a CASE with ", arms, " arm entries in it"))
            out += "CASE"
            stack.append(_Step(node, 1))
            if item.a != NO_NODE:
                out += " "
                stack.append(_Step(item.a, 0))
            return
        if phase == arms + 1:
            if item.b == NO_NODE:
                out += " END"
                return
            out += " ELSE "
            stack.append(_Step(node, UInt32(phase + 1)))
            stack.append(_Step(item.b, 0))
            return
        if phase == arms + 2:
            out += " END"
            return
        var entry = phase - 1
        out += " WHEN " if entry % 2 == 0 else " THEN "
        stack.append(_Step(node, UInt32(phase + 1)))
        stack.append(_Step(ast.at(item.children, entry), 0))
        return

    if kind == EXPR_BETWEEN:
        if phase == 0:
            if ast.length(item.children) != 2:
                raise Error("a BETWEEN without exactly two bounds on it")
            out += "("
            stack.append(_Step(node, 1))
            stack.append(_Step(item.a, 0))
            return
        if phase == 1:
            out += " NOT BETWEEN " if item.payload == 1 else " BETWEEN "
            stack.append(_Step(node, 2))
            stack.append(_Step(ast.at(item.children, 0), 0))
            return
        if phase == 2:
            out += " AND "
            stack.append(_Step(node, 3))
            stack.append(_Step(ast.at(item.children, 1), 0))
            return
        out += ")"
        return

    if kind == EXPR_IN:
        var count = ast.length(item.children)
        if phase == 0:
            out += "("
            stack.append(_Step(node, 1))
            stack.append(_Step(item.a, 0))
            return
        if phase == 1:
            out += " NOT IN (" if item.payload == 1 else " IN ("
            stack.append(_Step(node, 2))
            return
        if phase == count + 2:
            out += "))"
            return
        var entry = phase - 2
        if entry > 0:
            out += ", "
        stack.append(_Step(node, UInt32(phase + 1)))
        stack.append(_Step(ast.at(item.children, entry), 0))
        return

    if kind == EXPR_LIST:
        var count = ast.length(item.children)
        if phase == 0:
            out += "["
            stack.append(_Step(node, 1))
            return
        if phase == count + 1:
            out += "]"
            return
        var entry = phase - 1
        if entry > 0:
            out += ", "
        stack.append(_Step(node, UInt32(phase + 1)))
        stack.append(_Step(ast.at(item.children, entry), 0))
        return

    if kind == EXPR_STRUCT:
        # The run alternates an interned key and a value, so one phase covers a
        # pair and the phase count is half the run length.
        var entries = ast.length(item.children)
        var pairs = entries // 2
        if phase == 0:
            if entries % 2 != 0:
                raise Error(String("a struct with ", entries, " entries in it"))
            out += "{"
            stack.append(_Step(node, 1))
            return
        if phase == pairs + 1:
            out += "}"
            return
        var pair = phase - 1
        if pair > 0:
            out += ", "
        out += quote_string(ast.text(ast.at(item.children, pair * 2)))
        out += ": "
        stack.append(_Step(node, UInt32(phase + 1)))
        stack.append(_Step(ast.at(item.children, pair * 2 + 1), 0))
        return

    if kind == EXPR_COLLATE:
        if phase == 0:
            out += "("
            stack.append(_Step(node, 1))
            stack.append(_Step(item.a, 0))
            return
        out += " COLLATE "
        out += quote_name(ast.text(item.payload), grammar)
        out += ")"
        return

    if kind == EXPR_PARAMETER:
        out += ast.text(item.b)
        out += ast.text(item.payload)
        return

    if kind == EXPR_SUBQUERY:
        out += "("
        _write_stmt(ast, item.a, grammar, out)
        out += ")"
        return

    if kind == EXPR_EXISTS:
        out += "(NOT EXISTS (" if item.b == 1 else "(EXISTS ("
        _write_stmt(ast, item.a, grammar, out)
        out += "))"
        return

    if kind == EXPR_IN_SUBQUERY:
        if phase == 0:
            out += "("
            stack.append(_Step(node, 1))
            stack.append(_Step(item.a, 0))
            return
        out += " NOT IN (" if item.payload == 1 else " IN ("
        _write_stmt(ast, item.b, grammar, out)
        out += "))"
        return

    if kind == EXPR_QUANTIFIED:
        if phase == 0:
            out += "("
            stack.append(_Step(node, 1))
            stack.append(_Step(item.a, 0))
            return
        out += " "
        out += ast.text(item.payload)
        # SOME comes back as ANY, which is the word DuckDB prints too.
        out += " ALL (" if item.children == 1 else " ANY ("
        _write_stmt(ast, item.b, grammar, out)
        out += "))"
        return

    if kind == EXPR_WINDOW:
        _write_window(ast, node, grammar, out)
        return

    if kind == EXPR_FRAME:
        _write_frame(ast, node, grammar, out)
        return

    raise Error(String("the printer has no case for expression kind ", kind))


def _write_over(
    ast: Ast, node: UInt32, grammar: Grammar, mut out: String
) raises:
    """Appends the `OVER` a call carries, if it carries one.

    Args:
        ast: The AST.
        node: The `EXPR_WINDOW`, or 0 for a call with no `OVER`.
        grammar: A loaded grammar.
        out: The buffer.

    Raises:
        Error: If the window could not be printed.
    """
    if node == NO_NODE:
        return
    out += " OVER "
    _write_window(ast, node, grammar, out)


def _write_window(
    ast: Ast, node: UInt32, grammar: Grammar, mut out: String
) raises:
    """Appends one window specification, parentheses and all.

    The calls back into `_write` here are recursive, for the reason the star
    modifiers are: getting to a window costs a parenthesis in the query text
    and the matcher caps how many of those there can be, so the depth is
    bounded by what somebody typed.

    `OVER w` comes out as `OVER (w)`, because the AST records that a window is
    a name and not which of the two spellings it was written in, and the two
    mean the same thing.

    Args:
        ast: The AST.
        node: The index in the expression arena.
        grammar: A loaded grammar.
        out: The buffer.

    Raises:
        Error: If the node is not a window, or could not be printed.
    """
    if node == NO_NODE:
        raise Error("the printer was handed the null expression")
    ref item = ast.exprs[Int(node)]
    if item.kind != EXPR_WINDOW:
        raise Error(String("a window holding expression kind ", item.kind))
    out += "("
    var written = False
    if item.payload != NO_NODE:
        out += quote_name(ast.text(item.payload), grammar)
        written = True
    for i in range(ast.length(item.children)):
        if i == 0:
            out += " PARTITION BY " if written else "PARTITION BY "
            written = True
        else:
            out += ", "
        _write(ast, ast.at(item.children, i), grammar, out)
    for i in range(ast.length(item.a)):
        if i == 0:
            out += " ORDER BY " if written else "ORDER BY "
            written = True
        else:
            out += ", "
        _write_order(ast, ast.at(item.a, i), grammar, out)
    if item.b != NO_NODE:
        if written:
            out += " "
        _write_frame(ast, item.b, grammar, out)
    out += ")"


def _write_frame(
    ast: Ast, node: UInt32, grammar: Grammar, mut out: String
) raises:
    """Appends the `ROWS`, `RANGE` or `GROUPS` clause of a window.

    Args:
        ast: The AST.
        node: The index in the expression arena.
        grammar: A loaded grammar.
        out: The buffer.

    Raises:
        Error: If the node is not a frame, or carries a tag this does not know.
    """
    if node == NO_NODE:
        raise Error("the printer was handed the null expression")
    ref item = ast.exprs[Int(node)]
    if item.kind != EXPR_FRAME:
        raise Error(String("a frame holding expression kind ", item.kind))
    var tags = item.payload
    var mode = frame_mode(tags)
    if mode == FRAME_ROWS:
        out += "ROWS "
    elif mode == FRAME_RANGE:
        out += "RANGE "
    elif mode == FRAME_GROUPS:
        out += "GROUPS "
    else:
        raise Error(String("the printer has no case for framing ", mode))

    var end = frame_end(tags)
    if end != BOUND_NONE:
        out += "BETWEEN "
    _write_bound(ast, frame_start(tags), item.a, grammar, out)
    if end != BOUND_NONE:
        out += " AND "
        _write_bound(ast, end, item.b, grammar, out)

    var exclude = frame_exclude(tags)
    if exclude == EXCLUDE_CURRENT_ROW:
        out += " EXCLUDE CURRENT ROW"
    elif exclude == EXCLUDE_GROUP:
        out += " EXCLUDE GROUP"
    elif exclude == EXCLUDE_TIES:
        out += " EXCLUDE TIES"
    elif exclude == EXCLUDE_NO_OTHERS:
        out += " EXCLUDE NO OTHERS"
    elif exclude != EXCLUDE_NONE:
        raise Error(String("the printer has no case for exclusion ", exclude))


def _write_bound(
    ast: Ast, tag: UInt32, at: UInt32, grammar: Grammar, mut out: String
) raises:
    """Appends one end of a frame.

    Args:
        ast: The AST.
        tag: One of the `BOUND_` constants.
        at: The bound's expression, for the two tags that have one.
        grammar: A loaded grammar.
        out: The buffer.

    Raises:
        Error: If the tag is not one this knows, or the count is missing.
    """
    if tag == BOUND_CURRENT_ROW:
        out += "CURRENT ROW"
        return
    if tag == BOUND_UNBOUNDED_PRECEDING:
        out += "UNBOUNDED PRECEDING"
        return
    if tag == BOUND_UNBOUNDED_FOLLOWING:
        out += "UNBOUNDED FOLLOWING"
        return
    if tag != BOUND_PRECEDING and tag != BOUND_FOLLOWING:
        raise Error(String("the printer has no case for frame bound ", tag))
    if at == NO_NODE:
        raise Error("a counted frame bound with no count on it")
    _write(ast, at, grammar, out)
    out += " PRECEDING" if tag == BOUND_PRECEDING else " FOLLOWING"


def _write_stmt(
    ast: Ast, node: UInt32, grammar: Grammar, mut out: String
) raises:
    """Appends one statement to a buffer.

    Args:
        ast: The AST the node lives in.
        node: The index in the statement arena.
        grammar: A loaded grammar.
        out: The buffer.

    Raises:
        Error: If the node is null, or its kind is not one this knows, or it is
            missing a part the kind says it always has.
    """
    if node == NO_NODE:
        raise Error("the printer was handed the null statement")

    ref item = ast.stmts[Int(node)]
    var kind = item.kind

    if kind == STMT_SELECT:
        var ctes = ast.length(item.children)
        if ctes > 0:
            out += "WITH RECURSIVE " if item.payload == 1 else "WITH "
            for i in range(ctes):
                if i > 0:
                    out += ", "
                _write_cte(ast, ast.at(item.children, i), grammar, out)
            out += " "
        _write_nested(ast, item.a, grammar, out)
        if item.b != NO_NODE:
            _write_modifiers(ast, item.b, grammar, out)
        return

    if kind == STMT_QUERY:
        _write_query(ast, node, grammar, out)
        return

    if kind == STMT_SET_OPERATION:
        _write_nested(ast, item.a, grammar, out)
        out += " "
        out += ast.text(item.payload)
        out += " "
        _write_nested(ast, item.b, grammar, out)
        return

    if kind == STMT_VALUES:
        var rows = ast.length(item.children)
        if rows == 0:
            raise Error("a VALUES with no rows in it")
        out += "VALUES "
        for i in range(rows):
            if i > 0:
                out += ", "
            out += "("
            _write_list(ast, ast.at(item.children, i), grammar, out)
            out += ")"
        return

    if kind == STMT_TABLE:
        if ast.length(item.children) == 0:
            raise Error("a TABLE statement with no name on it")
        out += "TABLE "
        out += _names(ast, item.children, grammar)
        return

    if kind == STMT_PIVOT:
        _write_pivot(ast, node, grammar, out)
        return

    if kind == STMT_UNPIVOT:
        _write_unpivot(ast, node, grammar, out)
        return

    raise Error(String("the printer has no case for statement kind ", kind))


def _write_pivot(
    ast: Ast, node: UInt32, grammar: Grammar, mut out: String
) raises:
    """Appends a `PIVOT` statement.

    The three lists are all optional in the grammar, so each one is written
    only when it has something in it. A `PIVOT` with none of them is `PIVOT t`,
    which parses and asks DuckDB to work the whole thing out.

    Args:
        ast: The AST.
        node: The `STMT_PIVOT` index.
        grammar: A loaded grammar.
        out: The buffer.

    Raises:
        Error: If it has no table under it, or a part could not be printed.
    """
    ref item = ast.stmts[Int(node)]
    if item.a == NO_NODE:
        raise Error("a PIVOT with no table under it")
    out += "PIVOT "
    _write_ref(ast, item.a, grammar, out)

    for i in range(ast.length(item.b)):
        out += ", " if i > 0 else " ON "
        _write_pivot_on(ast, ast.at(item.b, i), grammar, out)

    for i in range(ast.length(item.children)):
        out += ", " if i > 0 else " USING "
        _write_item(ast, ast.at(item.children, i), grammar, out)

    for i in range(ast.length(item.payload)):
        out += ", " if i > 0 else " GROUP BY "
        out += quote_name(ast.text(ast.at(item.payload, i)), grammar)


def _write_pivot_on(
    ast: Ast, node: UInt32, grammar: Grammar, mut out: String
) raises:
    """Appends one pivot column.

    Args:
        ast: The AST.
        node: The `STMT_PIVOT_ON` index.
        grammar: A loaded grammar.
        out: The buffer.

    Raises:
        Error: If the node is not a pivot column, or has no header on it.
    """
    if node == NO_NODE:
        raise Error("the printer was handed the null statement")
    ref item = ast.stmts[Int(node)]
    if item.kind != STMT_PIVOT_ON:
        raise Error(String("an ON list holding statement kind ", item.kind))
    if item.a == NO_NODE:
        raise Error("a pivot column with no expression on it")
    _write(ast, item.a, grammar, out)

    if item.payload != NO_NODE:
        out += " IN "
        out += quote_name(ast.text(item.payload), grammar)
        return
    if item.b != NO_NODE:
        out += " IN ("
        _write_stmt(ast, item.b, grammar, out)
        out += ")"
        return
    var values = ast.length(item.children)
    if values == 0:
        return
    out += " IN ("
    for i in range(values):
        if i > 0:
            out += ", "
        _write_item(ast, ast.at(item.children, i), grammar, out)
    out += ")"


def _write_unpivot(
    ast: Ast, node: UInt32, grammar: Grammar, mut out: String
) raises:
    """Appends an `UNPIVOT` statement.

    `VALUE` and `VALUES` are the same word to the grammar, so the count picks
    which one to write and a query that wrote the other one gets this one back.

    Args:
        ast: The AST.
        node: The `STMT_UNPIVOT` index.
        grammar: A loaded grammar.
        out: The buffer.

    Raises:
        Error: If it has no table or no `ON` list, or a part could not be
            printed.
    """
    ref item = ast.stmts[Int(node)]
    if item.a == NO_NODE:
        raise Error("an UNPIVOT with no table under it")
    out += "UNPIVOT "
    _write_ref(ast, item.a, grammar, out)

    var columns = ast.length(item.children)
    if columns == 0:
        raise Error("an UNPIVOT with no columns to fold up")
    for i in range(columns):
        out += ", " if i > 0 else " ON "
        _write_item(ast, ast.at(item.children, i), grammar, out)

    if item.payload == NO_NODE:
        return
    out += " INTO NAME "
    out += quote_name(ast.text(item.payload), grammar)
    var values = ast.length(item.b)
    out += " VALUES " if values > 1 else " VALUE "
    for i in range(values):
        if i > 0:
            out += ", "
        out += quote_name(ast.text(ast.at(item.b, i)), grammar)


def _write_nested(
    ast: Ast, node: UInt32, grammar: Grammar, mut out: String
) raises:
    """Appends a query node, in parentheses when it is a whole statement.

    A `STMT_SELECT` in this position is what a parenthesized select in the text
    became, so it is the one thing that gets its parentheses back.

    Args:
        ast: The AST.
        node: The index in the statement arena.
        grammar: A loaded grammar.
        out: The buffer.

    Raises:
        Error: If the node could not be printed.
    """
    if node == NO_NODE:
        raise Error("the printer was handed the null statement")
    if ast.stmts[Int(node)].kind == STMT_SELECT:
        out += "("
        _write_stmt(ast, node, grammar, out)
        out += ")"
        return
    _write_stmt(ast, node, grammar, out)


def _write_query(
    ast: Ast, node: UInt32, grammar: Grammar, mut out: String
) raises:
    """Appends one `SELECT ... FROM ... WHERE ...` block.

    The `SELECT` is left out when there is nothing to put in it and there is a
    `FROM`, because that is the `FROM` first form and leaving it in would mean
    inventing a star the query did not write.

    Args:
        ast: The AST.
        node: The index in the statement arena.
        grammar: A loaded grammar.
        out: The buffer.

    Raises:
        Error: If a clause could not be printed.
    """
    ref item = ast.stmts[Int(node)]
    var clauses = item.children
    var projection = ast.slot(clauses, CLAUSE_PROJECTION)
    var tables = ast.slot(clauses, CLAUSE_FROM)

    var bare = ast.length(projection) == 0 and item.a == 0
    var wrote_select = not (bare and ast.length(tables) > 0)
    if wrote_select:
        out += "SELECT"
        if item.a & SELECT_DISTINCT != 0:
            out += " DISTINCT"
            if ast.length(item.b) > 0:
                out += " ON ("
                _write_list(ast, item.b, grammar, out)
                out += ")"
        elif item.a & SELECT_ALL != 0:
            out += " ALL"
        for i in range(ast.length(projection)):
            out += ", " if i > 0 else " "
            _write_item(ast, ast.at(projection, i), grammar, out)

    for i in range(ast.length(tables)):
        if i > 0:
            out += ", "
        elif wrote_select:
            out += " FROM "
        else:
            out += "FROM "
        _write_ref(ast, ast.at(tables, i), grammar, out)

    var filter = ast.slot(clauses, CLAUSE_WHERE)
    if filter != NO_NODE:
        out += " WHERE "
        _write(ast, filter, grammar, out)

    var grouping = ast.slot(clauses, CLAUSE_GROUP)
    for i in range(ast.length(grouping)):
        out += ", " if i > 0 else " GROUP BY "
        _write_group(ast, ast.at(grouping, i), grammar, out)

    var having = ast.slot(clauses, CLAUSE_HAVING)
    if having != NO_NODE:
        out += " HAVING "
        _write(ast, having, grammar, out)

    # Before the `QUALIFY` and not after it, because that is the order
    # `SimpleSelect` puts them in and the printed text has to parse again.
    var windows = ast.slot(clauses, CLAUSE_WINDOW)
    for i in range(ast.length(windows)):
        out += ", " if i > 0 else " WINDOW "
        _write_window_definition(ast, ast.at(windows, i), grammar, out)

    var qualify = ast.slot(clauses, CLAUSE_QUALIFY)
    if qualify != NO_NODE:
        out += " QUALIFY "
        _write(ast, qualify, grammar, out)


def _write_window_definition(
    ast: Ast, node: UInt32, grammar: Grammar, mut out: String
) raises:
    """Appends one entry of a `WINDOW` clause.

    Args:
        ast: The AST.
        node: The index in the statement arena.
        grammar: A loaded grammar.
        out: The buffer.

    Raises:
        Error: If the node is not a window definition, or could not be printed.
    """
    if node == NO_NODE:
        raise Error("the printer was handed the null statement")
    ref item = ast.stmts[Int(node)]
    if item.kind != STMT_WINDOW:
        raise Error(String("a WINDOW holding statement kind ", item.kind))
    out += quote_name(ast.text(item.payload), grammar)
    out += " AS "
    _write_window(ast, item.a, grammar, out)


def _write_item(
    ast: Ast, node: UInt32, grammar: Grammar, mut out: String
) raises:
    """Appends one entry of a `SELECT` list.

    Args:
        ast: The AST.
        node: The index in the statement arena.
        grammar: A loaded grammar.
        out: The buffer.

    Raises:
        Error: If the node is not a `SELECT` list entry, or could not be
            printed.
    """
    if node == NO_NODE:
        raise Error("the printer was handed the null statement")
    ref item = ast.stmts[Int(node)]
    if item.kind != STMT_ITEM:
        raise Error(String("a SELECT list holding statement kind ", item.kind))
    _write(ast, item.a, grammar, out)
    if item.payload != NO_NODE:
        out += " AS "
        out += quote_name(ast.text(item.payload), grammar)


def _write_cte(
    ast: Ast, node: UInt32, grammar: Grammar, mut out: String
) raises:
    """Appends one entry of a `WITH`.

    Args:
        ast: The AST.
        node: The index in the statement arena.
        grammar: A loaded grammar.
        out: The buffer.

    Raises:
        Error: If the node is not a `WITH` entry, or could not be printed.
    """
    if node == NO_NODE:
        raise Error("the printer was handed the null statement")
    ref item = ast.stmts[Int(node)]
    if item.kind != STMT_CTE:
        raise Error(String("a WITH holding statement kind ", item.kind))
    out += quote_name(ast.text(item.payload), grammar)
    var columns = ast.length(item.children)
    if columns > 0:
        out += " ("
        for i in range(columns):
            if i > 0:
                out += ", "
            out += quote_name(ast.text(ast.at(item.children, i)), grammar)
        out += ")"
    out += " AS "
    if item.b == MATERIALIZE_YES:
        out += "MATERIALIZED "
    elif item.b == MATERIALIZE_NO:
        out += "NOT MATERIALIZED "
    out += "("
    _write_stmt(ast, item.a, grammar, out)
    out += ")"


def _write_modifiers(
    ast: Ast, node: UInt32, grammar: Grammar, mut out: String
) raises:
    """Appends the `ORDER BY`, `LIMIT` and `OFFSET` that trail a statement.

    Args:
        ast: The AST.
        node: The index in the statement arena.
        grammar: A loaded grammar.
        out: The buffer.

    Raises:
        Error: If the node is not a modifiers node, or could not be printed.
    """
    ref item = ast.stmts[Int(node)]
    if item.kind != STMT_MODIFIERS:
        raise Error(String("a statement trailed by statement kind ", item.kind))
    for i in range(ast.length(item.children)):
        out += ", " if i > 0 else " ORDER BY "
        _write_order(ast, ast.at(item.children, i), grammar, out)
    if item.payload & LIMIT_ALL != 0:
        out += " LIMIT ALL"
    elif item.a != NO_NODE:
        out += " LIMIT "
        _write(ast, item.a, grammar, out)
        if item.payload & LIMIT_PERCENT != 0:
            out += "%"
    if item.b != NO_NODE:
        out += " OFFSET "
        _write(ast, item.b, grammar, out)


def _write_order(
    ast: Ast, node: UInt32, grammar: Grammar, mut out: String
) raises:
    """Appends one entry of an `ORDER BY`.

    Args:
        ast: The AST.
        node: The index in the statement arena.
        grammar: A loaded grammar.
        out: The buffer.

    Raises:
        Error: If the node is not an `ORDER BY` entry, or could not be printed.
    """
    if node == NO_NODE:
        raise Error("the printer was handed the null statement")
    ref item = ast.stmts[Int(node)]
    if item.kind != STMT_ORDER:
        raise Error(String("an ORDER BY holding statement kind ", item.kind))
    if item.a == NO_NODE:
        out += "ALL"
    else:
        _write(ast, item.a, grammar, out)
    if item.b == SORT_ASCENDING:
        out += " ASC"
    elif item.b == SORT_DESCENDING:
        out += " DESC"
    if item.payload == NULLS_FIRST:
        out += " NULLS FIRST"
    elif item.payload == NULLS_LAST:
        out += " NULLS LAST"


def _write_group(
    ast: Ast, node: UInt32, grammar: Grammar, mut out: String
) raises:
    """Appends one entry of a `GROUP BY`.

    Args:
        ast: The AST.
        node: The index in the statement arena.
        grammar: A loaded grammar.
        out: The buffer.

    Raises:
        Error: If the node is not a `GROUP BY` entry, or carries a tag this
            does not know.
    """
    if node == NO_NODE:
        raise Error("the printer was handed the null statement")
    ref item = ast.stmts[Int(node)]
    if item.kind != STMT_GROUP:
        raise Error(String("a GROUP BY holding statement kind ", item.kind))

    if item.b == GROUP_EXPRESSION:
        _write(ast, item.a, grammar, out)
        return
    if item.b == GROUP_ALL:
        out += "ALL"
        return
    if item.b == GROUP_EMPTY:
        out += "()"
        return

    if item.b == GROUP_SETS:
        out += "GROUPING SETS ("
    elif item.b == GROUP_CUBE:
        out += "CUBE ("
    elif item.b == GROUP_ROLLUP:
        out += "ROLLUP ("
    elif item.b == GROUP_TUPLE:
        out += "("
    else:
        raise Error(String("the printer has no case for grouping ", item.b))
    for i in range(ast.length(item.children)):
        if i > 0:
            out += ", "
        _write_group(ast, ast.at(item.children, i), grammar, out)
    out += ")"


def _write_ref(
    ast: Ast, node: UInt32, grammar: Grammar, mut out: String
) raises:
    """Appends one table reference to a buffer.

    Args:
        ast: The AST the node lives in.
        node: The index in the table reference arena.
        grammar: A loaded grammar.
        out: The buffer.

    Raises:
        Error: If the node is null, or its kind is not one this knows, or it is
            missing a part the kind says it always has.
    """
    if node == NO_NODE:
        raise Error("the printer was handed the null table reference")

    ref item = ast.refs[Int(node)]
    var kind = item.kind

    if kind == REF_TABLE:
        if ast.length(item.children) == 0:
            raise Error("a table reference with no name parts in it")
        out += _names(ast, item.children, grammar)
        _write_alias(ast, item.payload, grammar, out)
        return

    if kind == REF_SUBQUERY:
        if item.b == 1:
            out += "LATERAL "
        out += "("
        _write_stmt(ast, item.a, grammar, out)
        out += ")"
        _write_alias(ast, item.payload, grammar, out)
        return

    if kind == REF_FUNCTION:
        if item.b == 1:
            out += "LATERAL "
        if ast.length(item.a) == 0:
            raise Error("a table function with no name on it")
        out += _names(ast, item.a, grammar, calling=True)
        out += "("
        _write_list(ast, item.children, grammar, out)
        out += ")"
        _write_alias(ast, item.payload, grammar, out)
        return

    if kind == REF_PARENS:
        out += "("
        _write_ref(ast, item.a, grammar, out)
        out += ")"
        _write_alias(ast, item.payload, grammar, out)
        return

    if kind == REF_JOIN:
        _write_ref(ast, item.a, grammar, out)
        out += " "
        out += ast.text(item.payload)
        out += " "
        _write_ref(ast, item.b, grammar, out)
        var condition = ast.length(item.children)
        if condition > 1:
            raise Error(String("a join with ", condition, " conditions on it"))
        if condition == 1:
            out += " ON "
            _write(ast, ast.at(item.children, 0), grammar, out)
        return

    if kind == REF_JOIN_USING:
        _write_ref(ast, item.a, grammar, out)
        out += " "
        out += ast.text(item.payload)
        out += " "
        _write_ref(ast, item.b, grammar, out)
        var names = ast.length(item.children)
        if names == 0:
            raise Error("a USING join with no column names on it")
        out += " USING ("
        for i in range(names):
            if i > 0:
                out += ", "
            out += quote_name(ast.text(ast.at(item.children, i)), grammar)
        out += ")"
        return

    raise Error(String("the printer has no case for reference kind ", kind))


def _write_alias(ast: Ast, run: UInt32, grammar: Grammar, mut out: String):
    """Appends the alias a table reference carries, if it has one.

    Args:
        ast: The AST.
        run: The alias run, which is the alias name followed by the column
            aliases, and may be empty.
        grammar: A loaded grammar.
        out: The buffer.
    """
    var count = ast.length(run)
    if count == 0:
        return
    out += " AS "
    out += quote_name(ast.text(ast.at(run, 0)), grammar)
    if count == 1:
        return
    out += " ("
    for i in range(1, count):
        if i > 1:
            out += ", "
        out += quote_name(ast.text(ast.at(run, i)), grammar)
    out += ")"


def _write_list(
    ast: Ast, run: UInt32, grammar: Grammar, mut out: String
) raises:
    """Appends a run of expressions, separated by commas.

    Args:
        ast: The AST.
        run: The run, which may be empty.
        grammar: A loaded grammar.
        out: The buffer.

    Raises:
        Error: If one of the entries could not be printed.
    """
    for i in range(ast.length(run)):
        if i > 0:
            out += ", "
        _write(ast, ast.at(run, i), grammar, out)


def _write_exclude(ast: Ast, run: UInt32, grammar: Grammar, mut out: String):
    """Appends the `EXCLUDE` modifier of a star, if it has one.

    Args:
        ast: The AST.
        run: A run of interned names, which may be empty.
        grammar: A loaded grammar.
        out: The buffer.
    """
    var count = ast.length(run)
    if count == 0:
        return
    out += " EXCLUDE ("
    for i in range(count):
        if i > 0:
            out += ", "
        out += quote_name(ast.text(ast.at(run, i)), grammar)
    out += ")"


def _write_replace(
    ast: Ast, run: UInt32, grammar: Grammar, mut out: String
) raises:
    """Appends the `REPLACE` modifier of a star, if it has one.

    Args:
        ast: The AST.
        run: A run of alternating interned name and expression, which may be
            empty.
        grammar: A loaded grammar.
        out: The buffer.

    Raises:
        Error: If the run has an odd length, or one of the expressions could
            not be printed.
    """
    var count = ast.length(run)
    if count == 0:
        return
    if count % 2 != 0:
        raise Error(String("a star REPLACE with ", count, " entries in it"))
    out += " REPLACE ("
    for i in range(0, count, 2):
        if i > 0:
            out += ", "
        _write(ast, ast.at(run, i + 1), grammar, out)
        out += " AS "
        out += quote_name(ast.text(ast.at(run, i)), grammar)
    out += ")"


def _write_rename(
    ast: Ast, run: UInt32, grammar: Grammar, mut out: String
) raises:
    """Appends the `RENAME` modifier of a star, if it has one.

    `RenameEntry <- ExcludeName 'AS' Identifier`, so it is `AS` and not `TO`
    even though `ALTER TABLE ... RENAME ... TO` a few rules away spells it the
    other way.

    Args:
        ast: The AST.
        run: A run of alternating interned old name and new name, which may be
            empty.
        grammar: A loaded grammar.
        out: The buffer.

    Raises:
        Error: If the run has an odd length.
    """
    var count = ast.length(run)
    if count == 0:
        return
    if count % 2 != 0:
        raise Error(String("a star RENAME with ", count, " entries in it"))
    out += " RENAME ("
    for i in range(0, count, 2):
        if i > 0:
            out += ", "
        out += quote_name(ast.text(ast.at(run, i)), grammar)
        out += " AS "
        out += quote_name(ast.text(ast.at(run, i + 1)), grammar)
    out += ")"
