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

It quotes an identifier whenever the bare text would not read back as itself.
The tokenizer folds an unquoted word to lower case, so a name with a capital in
it can only have arrived quoted and has to leave quoted. The same goes for a
name holding a character no bare identifier may hold, and for a name that is a
reserved keyword.
"""

from .ast import (
    Ast,
    EXPR_BETWEEN,
    EXPR_BINARY,
    EXPR_CASE,
    EXPR_CAST,
    EXPR_COLLATE,
    EXPR_COLUMN,
    EXPR_FUNCTION,
    EXPR_IN,
    EXPR_LIST,
    EXPR_LITERAL,
    EXPR_PARAMETER,
    EXPR_STAR,
    EXPR_STRUCT,
    EXPR_UNARY,
    CALL_DISTINCT,
    CALL_STAR,
    LITERAL_BOOLEAN,
    LITERAL_NULL,
    LITERAL_NUMBER,
    LITERAL_STRING,
    NO_NODE,
)
from .generated.keywords import KEYWORD_RESERVED
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


def quote_name(name: StringSlice, grammar: Grammar) -> String:
    """Quotes an identifier if the bare text would not read back as itself.

    Args:
        name: The name, as the AST holds it.
        grammar: A loaded grammar, for the keyword table.

    Returns:
        The name, in double quotes when it needs them, with any double quote in
        it doubled.
    """
    if not needs_quoting(name, grammar):
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


def needs_quoting(name: StringSlice, grammar: Grammar) -> Bool:
    """Says whether an identifier has to be written in double quotes.

    Args:
        name: The name.
        grammar: A loaded grammar, for the keyword table.

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
    # A reserved keyword cannot stand where a name is wanted. An unreserved one
    # can, so it is left alone rather than quoted for the sake of it.
    return grammar.keyword_class(name) & KEYWORD_RESERVED != 0


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


def _names(ast: Ast, run: UInt32, grammar: Grammar) -> String:
    """Joins a run of interned name parts with dots.

    Args:
        ast: The AST.
        run: A run of pool indices.
        grammar: A loaded grammar, for the quoting rule.

    Returns:
        The dotted name, empty for an empty run.
    """
    var out = String()
    for i in range(ast.length(run)):
        if i > 0:
            out += "."
        out += quote_name(ast.text(ast.at(run, i)), grammar)
    return out^


def _write(ast: Ast, node: UInt32, grammar: Grammar, mut out: String) raises:
    """Appends one expression to a buffer.

    Args:
        ast: The AST the node lives in.
        node: The node index.
        grammar: A loaded grammar.
        out: The buffer.

    Raises:
        Error: If the node is null, or its kind is not one this knows, or it is
            missing a part the kind says it always has.
    """
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
        if ast.length(item.payload) == 0:
            raise Error("a function call with no name on it")
        out += _names(ast, item.payload, grammar)
        out += "("
        if item.a & CALL_STAR != 0:
            out += "*"
        else:
            if item.a & CALL_DISTINCT != 0:
                out += "DISTINCT "
            _write_list(ast, item.children, grammar, out)
        out += ")"
        return

    if kind == EXPR_UNARY:
        ref operator = ast.text(item.payload)
        out += "("
        out += operator
        if _is_word(operator):
            out += " "
        _write(ast, item.a, grammar, out)
        out += ")"
        return

    if kind == EXPR_BINARY:
        out += "("
        _write(ast, item.a, grammar, out)
        out += " "
        out += ast.text(item.payload)
        out += " "
        _write(ast, item.b, grammar, out)
        out += ")"
        return

    if kind == EXPR_CAST:
        out += "TRY_CAST(" if item.b == 1 else "CAST("
        _write(ast, item.a, grammar, out)
        out += " AS "
        out += ast.text(item.payload)
        out += ")"
        return

    if kind == EXPR_CASE:
        out += "CASE"
        if item.a != NO_NODE:
            out += " "
            _write(ast, item.a, grammar, out)
        var arms = ast.length(item.children)
        if arms == 0 or arms % 2 != 0:
            raise Error(String("a CASE with ", arms, " arm entries in it"))
        for i in range(0, arms, 2):
            out += " WHEN "
            _write(ast, ast.at(item.children, i), grammar, out)
            out += " THEN "
            _write(ast, ast.at(item.children, i + 1), grammar, out)
        if item.b != NO_NODE:
            out += " ELSE "
            _write(ast, item.b, grammar, out)
        out += " END"
        return

    if kind == EXPR_BETWEEN:
        if ast.length(item.children) != 2:
            raise Error("a BETWEEN without exactly two bounds on it")
        out += "("
        _write(ast, item.a, grammar, out)
        out += " NOT BETWEEN " if item.payload == 1 else " BETWEEN "
        _write(ast, ast.at(item.children, 0), grammar, out)
        out += " AND "
        _write(ast, ast.at(item.children, 1), grammar, out)
        out += ")"
        return

    if kind == EXPR_IN:
        out += "("
        _write(ast, item.a, grammar, out)
        out += " NOT IN (" if item.payload == 1 else " IN ("
        _write_list(ast, item.children, grammar, out)
        out += "))"
        return

    if kind == EXPR_LIST:
        out += "["
        _write_list(ast, item.children, grammar, out)
        out += "]"
        return

    if kind == EXPR_STRUCT:
        var entries = ast.length(item.children)
        if entries % 2 != 0:
            raise Error(String("a struct with ", entries, " entries in it"))
        out += "{"
        for i in range(0, entries, 2):
            if i > 0:
                out += ", "
            out += quote_string(ast.text(ast.at(item.children, i)))
            out += ": "
            _write(ast, ast.at(item.children, i + 1), grammar, out)
        out += "}"
        return

    if kind == EXPR_COLLATE:
        out += "("
        _write(ast, item.a, grammar, out)
        out += " COLLATE "
        out += quote_name(ast.text(item.payload), grammar)
        out += ")"
        return

    if kind == EXPR_PARAMETER:
        out += ast.text(item.b)
        out += ast.text(item.payload)
        return

    raise Error(String("the printer has no case for expression kind ", kind))


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
