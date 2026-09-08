"""Query text to a flat token vector.

One pass, no backtracking beyond a two byte peek, no allocation per token. The
output is a `List[Token]`, each token a kind, a few flags, a keyword index and a
byte range into the original text. Nothing is decoded here: a string keeps its
quotes and its escapes, a number keeps its underscores, and an identifier keeps
its original case. Decoding is the transformer's job, and leaving it there is
what keeps a token at twelve bytes and keeps the whole vector in cache.

There is no vendored artifact for any of this. The grammar has five character
level rules and four of them are placeholders that DuckDB's own matcher ignores:
`NumberLiteral <- < [+-]?[0-9]*([.][0-9]*)? >` does not describe `1e5` and
`StringLiteral <- '\\'' [^\\']* '\\''` does not describe `''` doubling, let alone
dollar quoting. See firepanda/sql/grammar/matcher_overrides.list. So the rules
below come from behaviour, measured against DuckDB 1.5.5 on this machine, and
every one of the awkward ones has a test naming the query that established it.

The pin in firepanda/sql/grammar/VENDOR is v2.0-cyanoptera, which is a newer
parser than the 1.5.5 the probes ran against. That gap is real and it is what the
differential harness is for.
"""

from .table import Grammar

comptime TOKEN_END: UInt8 = 0
"""One past the last token, so the matcher never checks a bound."""

comptime TOKEN_IDENTIFIER: UInt8 = 1
"""An unquoted word that is not in any keyword class."""

comptime TOKEN_QUOTED_IDENTIFIER: UInt8 = 2
"""A `"..."` word, which keeps its case and is never a keyword."""

comptime TOKEN_KEYWORD: UInt8 = 3
"""An unquoted word that is in at least one keyword class."""

comptime TOKEN_NUMBER: UInt8 = 4

comptime TOKEN_STRING: UInt8 = 5

comptime TOKEN_OPERATOR: UInt8 = 6
"""A run of operator characters, such as `+`, `||` or `!~~*`."""

comptime TOKEN_PUNCTUATION: UInt8 = 7
"""One of `( ) [ ] { } , ; . :`, which carry structure rather than meaning."""

comptime TOKEN_PARAMETER: UInt8 = 8
"""The `?` or `$` that introduces a parameter, one byte on its own.

The number or the name after it is a token of its own, because that is how the
grammar spells it: `QuestionMarkNumberedParameter <- '?' NumberLiteral` and
`ColLabelParameter <- '$' ColLabel`. A tokenizer that handed back `$1` whole
would leave the matcher with a token the grammar has no node for.

The `$` still needs the scan that a whole parameter would need, because a dollar
opens a dollar quoted string when what follows it is a tag, and `$1` is a
parameter only because `1` is not one.
"""

comptime FLAG_DECIMAL: UInt8 = 1
"""The number has a decimal point and no exponent, so its type is DECIMAL.

This is the tokenizer reaching into the type system, and it has to. `1.1 + 2.2`
is exactly `3.3` in DuckDB because both literals are DECIMAL(2,1), and a
tokenizer that called them DOUBLE would give `3.3000000000000003` from a query
that has no floating point in it.
"""

comptime FLAG_EXPONENT: UInt8 = 2
"""The number has an exponent, so its type is DOUBLE even with no point."""

comptime FLAG_ESCAPE: UInt8 = 4
"""An `E'...'` string, where backslash escapes apply."""

comptime FLAG_DOLLAR: UInt8 = 8
"""A `$tag$...$tag$` string, which has no escapes at all."""

comptime FLAG_UNICODE: UInt8 = 16
"""A `U&'...'` string or a `U&"..."` identifier, with unicode escapes."""

comptime FLAG_CONTINUED: UInt8 = 32
"""Two or more string literals joined across a newline, see `_continues`."""

comptime NO_KEYWORD: UInt16 = 65535
"""The keyword index of a token that is not a keyword. There are 499 of them."""

comptime _TAB = UInt8(9)
comptime _NEWLINE = UInt8(10)
comptime _FORM_FEED = UInt8(12)
comptime _RETURN = UInt8(13)
comptime _SPACE = UInt8(32)
comptime _BANG = UInt8(33)
comptime _QUOTE = UInt8(39)
comptime _AMPERSAND = UInt8(38)
comptime _STAR = UInt8(42)
comptime _PLUS = UInt8(43)
comptime _MINUS = UInt8(45)
comptime _DOT = UInt8(46)
comptime _SLASH = UInt8(47)
comptime _ZERO = UInt8(48)
comptime _NINE = UInt8(57)
comptime _COLON = UInt8(58)
comptime _EQUALS = UInt8(61)
comptime _QUESTION = UInt8(63)
comptime _UPPER_A = UInt8(65)
comptime _UPPER_E = UInt8(69)
comptime _UPPER_U = UInt8(85)
comptime _UPPER_Z = UInt8(90)
comptime _UNDERSCORE = UInt8(95)
comptime _LOWER_A = UInt8(97)
comptime _LOWER_E = UInt8(101)
comptime _LOWER_U = UInt8(117)
comptime _LOWER_Z = UInt8(122)
comptime _DOUBLE_QUOTE = UInt8(34)
comptime _DOLLAR = UInt8(36)
comptime _CASE_BIT = UInt8(32)

# The longest keyword is fifteen bytes, so a longer word cannot be one and the
# fold buffer never has to grow. tools/gen_grammar.py emits KEYWORD_MAX_LENGTH
# and tests/test_sql_grammar.mojo checks the table against it.
comptime _FOLD_CAPACITY = 16


@fieldwise_init
struct Token(ImplicitlyCopyable, Movable):
    """One token, twelve bytes, holding no text of its own."""

    var kind: UInt8
    """One of the TOKEN_ constants."""

    var flags: UInt8
    """The FLAG_ constants that apply, or 0."""

    var keyword: UInt16
    """An index into the grammar's keyword table, or NO_KEYWORD."""

    var start: UInt32
    """Byte offset of the first byte, into the query text."""

    var length: UInt32
    """Byte length, so `start + length` is one past the end."""


def token_text(sql: StringSlice, token: Token) -> StringSlice[sql.origin]:
    """Returns the text a token covers.

    Args:
        sql: The query the token was produced from.
        token: The token.

    Returns:
        The slice, which for a string literal still has its quotes on it.
    """
    var start = Int(token.start)
    return StringSlice(
        unsafe_from_utf8=sql.as_bytes()[start:][: Int(token.length)]
    )


def tokenize(sql: StringSlice, grammar: Grammar) raises -> List[Token]:
    """Turns a query into tokens.

    Args:
        sql: The query text, which the tokens point into and which therefore has
            to outlive them.
        grammar: A loaded grammar, for the keyword table.

    Returns:
        The tokens, always ending in one TOKEN_END.

    Raises:
        Error: On an unterminated string, identifier or block comment, which is
            the only thing the tokenizer can fail on. Everything else it does
            not recognize comes out as a token the matcher then rejects, which
            is what puts the caret in the right place.
    """
    var scan = _Scan(sql.as_bytes(), grammar)
    # One token per five bytes is a little over what real SQL comes out at, so
    # this is one allocation for almost every query and never a wasteful one.
    var out = List[Token](capacity=sql.byte_length() // 5 + 2)
    while True:
        var token = scan.next()
        out.append(token)
        if token.kind == TOKEN_END:
            return out^


struct _Scan[origin: ImmOrigin, table: ImmOrigin](Movable):
    """The state machine. One position, one borrowed grammar, nothing else.

    Two origins because the query and the grammar come from different places and
    live for different lengths of time. A grammar is built once and kept, and a
    query is usually a temporary.
    """

    var src: Span[UInt8, Self.origin]
    var at: Int
    var grammar: Pointer[Grammar, Self.table]

    def __init__(
        out self,
        src: Span[UInt8, Self.origin],
        ref[Self.table] grammar: Grammar,
    ):
        self.src = src
        self.at = 0
        self.grammar = Pointer(to=grammar)

    def next(mut self) raises -> Token:
        """Produces the next token, skipping whatever leads up to it."""
        self._skip_blanks()
        var start = self.at
        if self.at >= len(self.src):
            return Token(TOKEN_END, 0, NO_KEYWORD, UInt32(start), 0)

        var c = self.src[self.at]

        if _is_digit(c):
            return self._number(start)
        if c == _DOT and self._digit_at(self.at + 1):
            return self._number(start)
        if c == _QUOTE:
            return self._string(start, 0)
        if c == _DOUBLE_QUOTE:
            return self._quoted_identifier(start, 0)
        if c == _DOLLAR:
            return self._dollar(start)

        # `E'...'`, `U&'...'` and `U&"..."` are prefixes only when the quote is
        # the very next byte. `SELECT e 'a'` is an identifier and a string.
        if (c | _CASE_BIT) == _LOWER_E and self._byte_at(self.at + 1) == _QUOTE:
            self.at += 1
            return self._string(start, FLAG_ESCAPE)
        if (c | _CASE_BIT) == _LOWER_U and self._byte_at(
            self.at + 1
        ) == _AMPERSAND:
            var quote = self._byte_at(self.at + 2)
            if quote == _QUOTE:
                self.at += 2
                return self._string(start, FLAG_UNICODE)
            if quote == _DOUBLE_QUOTE:
                self.at += 2
                return self._quoted_identifier(start, FLAG_UNICODE)

        if _is_word_start(c):
            return self._word(start)
        if c == _QUESTION:
            return self._question(start)
        if _is_operator_char(c):
            return self._operator(start)
        if c == _COLON:
            # `::` and `:=` are operators and a lone `:` is the slice separator
            # in `list[1:2]`, so this one character needs the lookahead.
            var after = self._byte_at(self.at + 1)
            if after == _COLON or after == _EQUALS:
                self.at += 2
                return Token(TOKEN_OPERATOR, 0, NO_KEYWORD, UInt32(start), 2)
            self.at += 1
            return Token(TOKEN_PUNCTUATION, 0, NO_KEYWORD, UInt32(start), 1)
        if _is_punctuation(c):
            self.at += 1
            return Token(TOKEN_PUNCTUATION, 0, NO_KEYWORD, UInt32(start), 1)

        # Anything left is a byte no DuckDB token starts with. It becomes a one
        # byte token rather than an error, so that the matcher reports it with
        # the same caret as every other syntax error instead of the tokenizer
        # inventing a second error format for it.
        self.at += 1
        return Token(TOKEN_OPERATOR, 0, NO_KEYWORD, UInt32(start), 1)

    # -----------------------------------------------------------------------
    # Whitespace and comments
    # -----------------------------------------------------------------------

    def _skip_blanks(mut self) raises:
        """Skips whitespace, line comments and nested block comments."""
        while self.at < len(self.src):
            var c = self.src[self.at]
            if _is_blank(c):
                self.at += 1
                continue
            if c == _MINUS and self._byte_at(self.at + 1) == _MINUS:
                while self.at < len(self.src) and self.src[self.at] != _NEWLINE:
                    self.at += 1
                continue
            if c == _SLASH and self._byte_at(self.at + 1) == _STAR:
                self._block_comment()
                continue
            return

    def _block_comment(mut self) raises:
        """Skips one `/* */`, counting nested opens.

        Nesting is not decoration. `/* SELECT /* x */ 1 */` is one comment in
        DuckDB and two thirds of a comment in a scanner that stops at the first
        `*/`, and the remaining third is then parsed as SQL.
        """
        var start = self.at
        var depth = 0
        while self.at < len(self.src):
            var c = self.src[self.at]
            var after = self._byte_at(self.at + 1)
            if c == _SLASH and after == _STAR:
                depth += 1
                self.at += 2
                continue
            if c == _STAR and after == _SLASH:
                depth -= 1
                self.at += 2
                if depth == 0:
                    return
                continue
            self.at += 1
        raise error_at(self.src, start, "unterminated /* comment")

    # -----------------------------------------------------------------------
    # Words
    # -----------------------------------------------------------------------

    def _word(mut self, start: Int) -> Token:
        """Reads an unquoted word and looks it up in the keyword table."""
        self.at += 1
        while self.at < len(self.src) and _is_word_byte(self.src[self.at]):
            self.at += 1
        var length = self.at - start

        # Fold into a fixed buffer rather than allocating a String per word. A
        # word longer than the longest keyword cannot be a keyword, so the
        # buffer never has to grow and the common long identifier skips the
        # bisection entirely.
        if length < _FOLD_CAPACITY:
            var folded = InlineArray[UInt8, _FOLD_CAPACITY](fill=0)
            for i in range(length):
                folded[i] = _fold(self.src[start + i])
            var found = self.grammar[].keyword_index(Span(folded)[:length])
            if found >= 0:
                return Token(
                    TOKEN_KEYWORD,
                    0,
                    UInt16(found),
                    UInt32(start),
                    UInt32(length),
                )
        return Token(
            TOKEN_IDENTIFIER, 0, NO_KEYWORD, UInt32(start), UInt32(length)
        )

    def _quoted_identifier(mut self, start: Int, flags: UInt8) raises -> Token:
        """Reads a `"..."`, where `""` is one embedded quote."""
        self.at += 1
        while self.at < len(self.src):
            if self.src[self.at] == _DOUBLE_QUOTE:
                if self._byte_at(self.at + 1) == _DOUBLE_QUOTE:
                    self.at += 2
                    continue
                self.at += 1
                var length = self.at - start
                # DuckDB rejects `""` outright rather than treating it as an
                # empty name, and so does every Postgres derived parser.
                if length == 2 and flags == 0:
                    raise error_at(
                        self.src, start, "zero-length delimited identifier"
                    )
                return Token(
                    TOKEN_QUOTED_IDENTIFIER,
                    flags,
                    NO_KEYWORD,
                    UInt32(start),
                    UInt32(length),
                )
            self.at += 1
        raise error_at(self.src, start, "unterminated quoted identifier")

    # -----------------------------------------------------------------------
    # Numbers
    # -----------------------------------------------------------------------

    def _number(mut self, start: Int) -> Token:
        """Reads a number.

        The shapes, each with a query behind it:

            1_000       an underscore is a separator only between two digits,
                        because `SELECT 1_` is `1` aliased `_`
            1.          valid, DECIMAL(1,0)
            .5          valid, DECIMAL(1,1)
            1.2.3       is `1.2` then `.3`, so at most one point
            1e5         DOUBLE, and so is `1.e5`
            1e          is `1` aliased `e`, so an exponent with no digits after
                        it is not an exponent and the `e` is given back
        """
        var flags = UInt8(0)
        self._digits()
        if self._byte_at(self.at) == _DOT:
            self.at += 1
            self._digits()
            flags |= FLAG_DECIMAL
        if (self._byte_at(self.at) | _CASE_BIT) == _LOWER_E:
            var save = self.at
            self.at += 1
            var sign = self._byte_at(self.at)
            if sign == _PLUS or sign == _MINUS:
                self.at += 1
            if self._digit_at(self.at):
                self._digits()
                # An exponent makes it DOUBLE whatever the point did.
                flags = (flags & ~FLAG_DECIMAL) | FLAG_EXPONENT
            else:
                self.at = save
        return Token(
            TOKEN_NUMBER,
            flags,
            NO_KEYWORD,
            UInt32(start),
            UInt32(self.at - start),
        )

    def _digits(mut self):
        """Reads digits, taking an underscore only when digits surround it."""
        while self.at < len(self.src):
            var c = self.src[self.at]
            if _is_digit(c):
                self.at += 1
            elif (
                c == _UNDERSCORE
                and self.at > 0
                and _is_digit(self.src[self.at - 1])
                and self._digit_at(self.at + 1)
            ):
                self.at += 1
            else:
                return

    # -----------------------------------------------------------------------
    # Strings
    # -----------------------------------------------------------------------

    def _string(mut self, start: Int, flags: UInt8) raises -> Token:
        """Reads a `'...'`, including any parts continued onto later lines."""
        var all_flags = flags
        while True:
            self._one_quoted(start)
            if not self._continues():
                break
            all_flags |= FLAG_CONTINUED
        return Token(
            TOKEN_STRING,
            all_flags,
            NO_KEYWORD,
            UInt32(start),
            UInt32(self.at - start),
        )

    def _one_quoted(mut self, start: Int) raises:
        """Reads one `'...'`, where `''` is one embedded quote."""
        self.at += 1
        while self.at < len(self.src):
            if self.src[self.at] == _QUOTE:
                if self._byte_at(self.at + 1) == _QUOTE:
                    self.at += 2
                    continue
                self.at += 1
                return
            self.at += 1
        raise error_at(self.src, start, "unterminated quoted string")

    def _continues(mut self) -> Bool:
        """Says whether another string literal joins onto the one just read.

        Postgres's rule, which DuckDB inherits: two string literals separated by
        whitespace containing at least one newline are one string. It is not
        whitespace in general. `SELECT 'a' 'b'` is a syntax error and
        `SELECT 'a'\\n'b'` is `'ab'`. A line comment in between keeps the join
        and a block comment breaks it, which is the shape of Postgres's scanner
        rules and is not a simplification.
        """
        var save = self.at
        var newline = False
        while self.at < len(self.src):
            var c = self.src[self.at]
            if c == _NEWLINE:
                newline = True
                self.at += 1
            elif _is_blank(c):
                self.at += 1
            elif c == _MINUS and self._byte_at(self.at + 1) == _MINUS:
                while self.at < len(self.src) and self.src[self.at] != _NEWLINE:
                    self.at += 1
            else:
                break
        if newline and self._byte_at(self.at) == _QUOTE:
            return True
        self.at = save
        return False

    def _dollar(mut self, start: Int) raises -> Token:
        """Reads a `$`, which opens a dollar quoted string or a parameter.

        The disambiguation is the whole reason this is its own method. A tag is
        a word that does not start with a digit, so `$tag$` and `$$` open a
        string and `$1` is a parameter. `SELECT $1$a$1$` proves it: DuckDB reads
        `$1` as a parameter and then fails on the rest with `unterminated
        dollar-quoted string`, which is not what it would say if `$1$` opened
        one.

        A string comes back whole and a parameter comes back as the `$` alone,
        because the grammar spells a parameter as two nodes and a string as one.
        """
        var tag_start = self.at + 1
        var cursor = tag_start
        while cursor < len(self.src) and _is_tag_byte(self.src[cursor]):
            cursor += 1
        var opens = self._byte_at(cursor) == _DOLLAR and not self._digit_at(
            tag_start
        )
        if opens:
            var tag_length = cursor - tag_start
            self.at = cursor + 1
            while self.at < len(self.src):
                if self.src[self.at] == _DOLLAR and self._tag_at(
                    self.at + 1, tag_start, tag_length
                ):
                    self.at += tag_length + 2
                    return Token(
                        TOKEN_STRING,
                        FLAG_DOLLAR,
                        NO_KEYWORD,
                        UInt32(start),
                        UInt32(self.at - start),
                    )
                self.at += 1
            raise error_at(self.src, start, "unterminated dollar-quoted string")

        # Not a string, so the dollar introduces a parameter and the number or
        # the name after it is the next token, read the ordinary way.
        self.at = tag_start
        return Token(TOKEN_PARAMETER, 0, NO_KEYWORD, UInt32(start), 1)

    def _tag_at(self, at: Int, tag_start: Int, tag_length: Int) -> Bool:
        """Says whether the closing `tag$` of a dollar quote is at `at`."""
        if at + tag_length >= len(self.src):
            return False
        for i in range(tag_length):
            if self.src[at + i] != self.src[tag_start + i]:
                return False
        return self.src[at + tag_length] == _DOLLAR

    # -----------------------------------------------------------------------
    # Operators and parameters
    # -----------------------------------------------------------------------

    def _question(mut self, start: Int) -> Token:
        """Reads the `?` of `?` or `?1`, leaving any number for the next call.
        """
        self.at += 1
        return Token(TOKEN_PARAMETER, 0, NO_KEYWORD, UInt32(start), 1)

    def _operator(mut self, start: Int) -> Token:
        """Reads a run of operator characters, longest first, with one caveat.

        The caveat is Postgres's, and DuckDB has it too. A multi character
        operator that ends in `+` or `-` keeps those characters only if it also
        contains one of `~ ! @ # ^ & | ` `, and otherwise gives them back. That
        is why `SELECT 1 =- 1` is `1 = -1` and `SELECT 1 !=- 1` asks the catalog
        for an operator named `!=-`. Without the rule, `x=-1` is a call to an
        operator nobody defined.
        """
        while self.at < len(self.src) and _is_operator_char(self.src[self.at]):
            self.at += 1
        var end = self.at
        if end - start > 1:
            var special = False
            for i in range(start, end):
                if _is_operator_marker(self.src[i]):
                    special = True
                    break
            if not special:
                while end - start > 1 and (
                    self.src[end - 1] == _PLUS or self.src[end - 1] == _MINUS
                ):
                    end -= 1
        self.at = end
        return Token(
            TOKEN_OPERATOR, 0, NO_KEYWORD, UInt32(start), UInt32(end - start)
        )

    # -----------------------------------------------------------------------
    # Peeking
    # -----------------------------------------------------------------------

    def _byte_at(self, at: Int) -> UInt8:
        """The byte at `at`, or 0 past the end, which no token contains."""
        if at < 0 or at >= len(self.src):
            return 0
        return self.src[at]

    def _digit_at(self, at: Int) -> Bool:
        return _is_digit(self._byte_at(at))


def error_at(src: Span[UInt8, _], offset: Int, message: StringSlice) -> Error:
    """Builds a parser error in DuckDB's shape.

        Parser Error: unterminated quoted string

        LINE 1: SELECT 'abc
                       ^

    The blank line after the message is DuckDB's and not a typo here.

    Shared with the matcher, because there is one error shape for the whole
    parser and document 11's corpus matches DuckDB's text by substring. The hard
    part is choosing the offset, not rendering it: the tokenizer knows exactly
    where the thing it could not finish started, and document 04 section 5 says
    what the matcher does instead.

    Args:
        src: The query text.
        offset: The byte the token started at.
        message: What went wrong, in DuckDB's words.

    Returns:
        The error, ready to raise.
    """
    var bytes = src
    var line_start = 0
    var line = 1
    for i in range(min(offset, len(bytes))):
        if bytes[i] == _NEWLINE:
            line_start = i + 1
            line += 1
    var line_end = line_start
    while line_end < len(bytes) and bytes[line_end] != _NEWLINE:
        line_end += 1

    var prefix = String("LINE ", line, ": ")
    var text = StringSlice(unsafe_from_utf8=bytes[line_start:line_end])
    var caret = String()
    for _ in range(prefix.byte_length() + offset - line_start):
        caret += " "
    return Error(
        String(
            "Parser Error: ", message, "\n\n", prefix, text, "\n", caret, "^"
        )
    )


def _is_blank(c: UInt8) -> Bool:
    """Says whether a byte separates tokens without being one.

    The grammar's `%whitespace <- [ \\t\\n\\r]*` is missing the form feed that
    DuckDB actually accepts, and it does not accept the vertical tab that a
    reader might assume goes with it. `SELECT 1\\f+\\f1` is 2 and
    `SELECT 1\\v+\\v1` is a syntax error, so this list is the binary's and not
    the grammar's.
    """
    return (
        c == _SPACE
        or c == _TAB
        or c == _NEWLINE
        or c == _RETURN
        or (c == _FORM_FEED)
    )


def _is_digit(c: UInt8) -> Bool:
    return c >= _ZERO and c <= _NINE


def _is_word_start(c: UInt8) -> Bool:
    """Says whether a byte can start an unquoted word.

    The grammar writes `[a-z_]i`, which is ASCII only, and DuckDB accepts
    `SELECT café` all the same. Rejecting a byte the real parser accepts is the
    exact failure this project promised not to have, so anything above ASCII
    starts a word and the differential harness gets to argue with it.
    """
    var lower = c | _CASE_BIT
    return (
        (lower >= _LOWER_A and lower <= _LOWER_Z)
        or c == _UNDERSCORE
        or (c >= 128)
    )


def _is_word_byte(c: UInt8) -> Bool:
    """Says whether a byte continues an unquoted word.

    `$` is here and not in `_is_word_start`, because `SELECT a$b` is one column
    and `$b` on its own is a parameter.
    """
    return _is_word_start(c) or _is_digit(c) or c == _DOLLAR


def _is_tag_byte(c: UInt8) -> Bool:
    """Says whether a byte belongs to a dollar quote tag.

    A word byte less the dollar, because the dollar is what ends the tag. With
    `_is_word_byte` here instead, `$$abc$$` reads its own closing dollar as part
    of an empty tag and the whole thing falls apart.
    """
    return _is_word_start(c) or _is_digit(c)


def _is_operator_char(c: UInt8) -> Bool:
    """Postgres's operator character set, less `?`.

    `?` is a parameter in DuckDB rather than an operator: `SELECT 1 ?? 2` fails
    at the first `?` rather than at a two character operator, so it does not
    take part in the run.
    """
    return (
        c == _PLUS
        or c == _MINUS
        or c == _STAR
        or c == _SLASH
        or c == UInt8(37)  # %
        or c == UInt8(60)  # <
        or c == _EQUALS
        or c == UInt8(62)  # >
        or _is_operator_marker(c)
    )


def _is_operator_marker(c: UInt8) -> Bool:
    """The operator characters that let a run keep a trailing `+` or `-`."""
    return (
        c == UInt8(126)  # ~
        or c == _BANG
        or c == UInt8(64)  # @
        or c == UInt8(35)  # #
        or c == UInt8(94)  # ^
        or c == _AMPERSAND
        or c == UInt8(124)  # |
        or c == UInt8(96)  # `
    )


def _is_punctuation(c: UInt8) -> Bool:
    """The characters that carry structure rather than meaning."""
    return (
        c == UInt8(40)  # (
        or c == UInt8(41)  # )
        or c == UInt8(91)  # [
        or c == UInt8(93)  # ]
        or c == UInt8(123)  # {
        or c == UInt8(125)  # }
        or c == UInt8(44)  # ,
        or c == UInt8(59)  # ;
        or c == _DOT
    )


def _fold(c: UInt8) -> UInt8:
    """Lower cases one ASCII byte and leaves everything else alone.

    Down and not up. DuckDB folds unquoted identifiers to lower case where
    standard SQL folds to upper, which is why `SELECT A` and `select a` name the
    same column and `"A"` names a different one.
    """
    if c >= _UPPER_A and c <= _UPPER_Z:
        return c | _CASE_BIT
    return c
