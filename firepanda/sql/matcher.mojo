"""The PEG matcher: a token vector and a rule table in, a parse tree out.

Recursive descent over the flat node array from `generated/rules.mojo`. It knows
what a sequence is and what an ordered choice is, and it knows nothing about SQL.
Every SQL specific decision it makes comes out of the table it is walking, which
is the property document 03 exists to buy: the dialect is vendored data, and a
grammar bump is a regeneration rather than a rewrite.

PEG semantics, which are not context free grammar semantics and which the grammar
is written expecting:

- An ordered choice takes the first alternative that matches. There is no longest
  match, no ambiguity and no conflict, so the order the alternatives are written
  in is part of the language.
- Repetition is greedy and never gives anything back, so `A* A` never matches.
- A sequence fails as a unit and leaves the position where it found it.
- Lookahead matches without consuming.

Twenty four rules are matched from code rather than from their bodies, because
their bodies are placeholders upstream's matcher also ignores. See `_overridden`,
and `firepanda/sql/grammar/matcher_overrides.list` for where the list comes from.

See docs/specs/sql/04-the-parser.md sections 3 to 5.
"""

from .generated.keywords import (
    KEYWORD_COLUMN_NAME,
    KEYWORD_FUNC_NAME,
    KEYWORD_TYPE_NAME,
    KEYWORD_UNRESERVED,
)
from .generated.rules import (
    FLAG_WORD,
    MATCHER_NUMBER_LITERAL,
    MATCHER_OPERATOR,
    MATCHER_RESERVED_IDENTIFIER,
    MATCHER_STRING_LITERAL,
    NODE_CAPTURE,
    NODE_CHOICE,
    NODE_CLASS,
    NODE_KEYWORDS,
    NODE_LIT,
    NODE_NOT,
    NODE_OPT,
    NODE_PLUS,
    NODE_REF,
    NODE_SEQ,
    NODE_STAR,
    RULE_END_OF_INPUT,
    RULE_PROGRAM,
    SUGGEST_SCALAR_FUNCTION_NAME,
    SUGGEST_TABLE_FUNCTION_NAME,
    SUGGEST_TABLE_NAME,
    SUGGEST_TYPE_NAME,
)
from .table import Grammar
from .token import (
    TOKEN_END,
    TOKEN_IDENTIFIER,
    TOKEN_KEYWORD,
    TOKEN_NUMBER,
    TOKEN_OPERATOR,
    TOKEN_PARAMETER,
    TOKEN_PUNCTUATION,
    TOKEN_QUOTED_IDENTIFIER,
    TOKEN_STRING,
    Token,
    error_at,
    tokenize,
)

comptime MAX_DEPTH = 500
"""How many rules deep a parse may go before it gives up.

The guard is here because this matcher recurses and the native stack does not
grow. The number came from measuring rather than from taste. A statement costs
forty frames before any nesting, and then twenty one frames per nested
parenthesis, twenty one per nested `CASE` and sixteen per nested subquery, so
five hundred is about twenty two parentheses or twenty eight subqueries. An
unoptimized build of this file runs out of native stack at around eight hundred
and twenty frames and an optimized one at around two thousand, so the guard fires
with room to spare on the build that has the least of it.

DuckDB needs no guard: its matcher keeps an explicit stack and runs until it runs
out of heap, which is why `SELECT ((((1))))` is accepted five thousand
parentheses deep there. The error text is the same on both sides. Closing the gap
means turning this into an explicit stack machine, which is the same change that
the parse would want for speed, so the two belong in one piece of work rather
than two.
"""

comptime NO_NODE: UInt32 = 0
"""The null node. Index 0 of the arena is a placeholder nobody reads."""

comptime _NO_SLOT: UInt8 = 255
"""The memo slot of a rule that is not memoized. There are 22 that are."""

comptime _DEFINED_OPERATORS = StaticString(
    "-> ->> <= >= != == <> ~~ ~~* ~~~ ~* !~~ !~~* !~ !~*"
)
"""The operators the grammar writes out itself, which are not catalog lookups.

`OperatorLiteral` is the rule behind `SELECT 1 @@ 2`, which asks the catalog for
an operator by that name. It has to refuse everything on this list, because the
grammar has its own nodes for them and a rule that read `<=` as a catalog lookup
would take the wrong branch of an ordered choice.

One space separated string rather than a list of them, because a list of string
literals is a comptime value that will not materialize and the scan is over fifty
bytes on the rare token that gets this far.
"""


@fieldwise_init
struct ParseNode(ImplicitlyCopyable, Movable):
    """One matched rule, twenty bytes, holding no pointers.

    Children are a first child index and a next sibling index rather than a list,
    so that a whole parse is one arena and one allocation. A parent is always
    built after its children, so a parent's index is always greater than its
    children's, which is what makes discarding a failed attempt a truncation.
    """

    var rule: UInt16
    """The rule that matched, indexing `Grammar.names`."""

    var token_start: UInt32
    """The first token this rule covers."""

    var token_end: UInt32
    """One past the last token, so an empty match has the two equal."""

    var first_child: UInt32
    """The first rule matched inside this one, or 0 for none."""

    var next_sibling: UInt32
    """The next rule matched inside this one's parent, or 0 for none."""


struct Parse(Movable):
    """A parsed statement: the tokens it came from and the tree over them."""

    var tokens: List[Token]
    """The token vector, which the nodes index into."""

    var nodes: List[ParseNode]
    """The arena. Index 0 is the null node and is never a real match."""

    var root: UInt32
    """The node for the rule the parse started at."""

    def __init__(
        out self,
        var tokens: List[Token],
        var nodes: List[ParseNode],
        root: UInt32,
    ):
        """Takes ownership of a finished parse.

        Args:
            tokens: The token vector.
            nodes: The arena.
            root: The root node index.
        """
        self.tokens = tokens^
        self.nodes = nodes^
        self.root = root

    def children(self, node: UInt32) -> List[UInt32]:
        """Collects one node's children in order.

        Args:
            node: The node index.

        Returns:
            The child indices, empty for a leaf.
        """
        var out = List[UInt32]()
        var cursor = self.nodes[Int(node)].first_child
        while cursor != NO_NODE:
            out.append(cursor)
            cursor = self.nodes[Int(cursor)].next_sibling
        return out^


def parse(sql: StringSlice, grammar: Grammar) raises -> Parse:
    """Parses a whole query.

    Args:
        sql: The query text, which the tokens point into and which therefore has
            to outlive the result.
        grammar: A loaded grammar.

    Returns:
        The parse, rooted at the grammar's `Program` rule.

    Raises:
        Error: On a tokenizer failure or a syntax error, in DuckDB's shape.
    """
    return parse_rule(sql, grammar, RULE_PROGRAM)


def parse_rule(sql: StringSlice, grammar: Grammar, rule: Int) raises -> Parse:
    """Parses a query against one rule, which then has to cover all of it.

    This is how a test asks a question about one corner of the grammar without
    wrapping it in a whole statement, and it is what a fuzzer aims at a rule.

    Args:
        sql: The query text.
        grammar: A loaded grammar.
        rule: The rule index to start at.

    Returns:
        The parse, rooted at that rule.

    Raises:
        Error: On a tokenizer failure or a syntax error, in DuckDB's shape.
    """
    var tokens = tokenize(sql, grammar)
    var run = _Run(sql.as_bytes(), tokens, grammar)
    var matched = run.rule(rule)
    # Program is `TopLevelStatement*`, so it matches the empty string and cannot
    # fail. What says the query was bad is that the parse did not reach the end,
    # and the furthest token a terminal was tried at is where a reader points.
    if not matched or run.at != len(tokens) - 1:
        raise run.syntax_error()
    var root = run.pending[len(run.pending) - 1]
    # The run borrows the tokens, so it has to be consumed before they can be
    # moved into the result.
    var nodes = run^.finish()
    return Parse(tokens^, nodes^, root)


struct _Run[origin: ImmOrigin, words: ImmOrigin, table: ImmOrigin](Movable):
    """One parse in progress. Built, used once and dropped.

    Three origins because the query text, the token vector and the grammar come
    from three places and live for three different lengths of time.
    """

    var src: Span[UInt8, Self.origin]
    """The query text, for rendering an error."""

    var tokens: Pointer[List[Token], Self.words]
    """The token vector, which always ends in one TOKEN_END."""

    var grammar: Pointer[Grammar, Self.table]
    """The rule table being walked."""

    var at: Int
    """The current token."""

    var nodes: List[ParseNode]
    """The arena, growing as rules succeed and truncating as they fail."""

    var pending: List[UInt32]
    """Nodes that have matched but have not been given a parent yet.

    A rule adopts everything above its own mark when it succeeds, so this is a
    stack and not a queue, and a failed rule truncates it back to the mark rather
    than unlinking anything.
    """

    var furthest: Int
    """The furthest token a terminal was tried at, which is where an error
    goes."""

    var quiet: Int
    """How many negative lookaheads we are inside.

    A terminal that fails inside `!X` failed on purpose, so it must not move the
    furthest position. Without this the error for a near miss points at whatever
    the grammar was checking was absent.
    """

    var depth: Int
    """How many rules deep we are, against MAX_DEPTH."""

    var memo: List[UInt8]
    """One bit per memoized rule per token position, set when it failed there.

    Failures only. A PEG rule is a pure function of the grammar and the position,
    so a rule that failed once at a position fails every time, and that is the
    whole of the exponential blowup DuckDB measured, which was 10.640 seconds for
    nineteen unmatched parentheses. Successes are not memoized, because a
    memoized success is a subtree in the arena and a failing ancestor may have
    truncated it away since.

    Empty until the first memoized rule fails, so a query that never backtracks
    never allocates it.
    """

    def __init__(
        out self,
        src: Span[UInt8, Self.origin],
        ref[Self.words] tokens: List[Token],
        ref[Self.table] grammar: Grammar,
    ):
        """Sets up a parse over one query.

        Args:
            src: The query text.
            tokens: The token vector.
            grammar: The rule table.
        """
        self.src = src
        self.tokens = Pointer(to=tokens)
        self.grammar = Pointer(to=grammar)
        self.at = 0
        # A node per matched rule, and real SQL matches a few per token. This is
        # a starting size and not a bound.
        self.nodes = List[ParseNode](capacity=len(tokens) * 4 + 8)
        self.nodes.append(ParseNode(0, 0, 0, NO_NODE, NO_NODE))
        self.pending = List[UInt32]()
        self.furthest = 0
        self.quiet = 0
        self.depth = 0
        self.memo = List[UInt8]()

    def finish(deinit self) -> List[ParseNode]:
        """Hands the arena over and ends the run.

        Consuming, because the arena is the one thing here worth keeping and
        copying it would double the cost of a parse for nothing.

        Returns:
            The node arena.
        """
        return self.nodes^

    # -----------------------------------------------------------------------
    # Rules
    # -----------------------------------------------------------------------

    def rule(mut self, index: Int) raises -> Bool:
        """Matches one rule at the current position.

        On success one node is pushed onto `pending` and the position has moved.
        On failure nothing has changed at all, which is the invariant the whole
        matcher rests on.

        Args:
            index: The rule index.

        Returns:
            Whether it matched.

        Raises:
            Error: If the parse is too deeply nested, or if the table holds a
                node this build has no case for, which is a build problem.
        """
        var slot = Int(self.grammar[].memo_slots[index])
        if slot != Int(_NO_SLOT) and self._failed_before(slot, self.at):
            return False

        var start_at = self.at
        var mark_nodes = len(self.nodes)
        var mark_pending = len(self.pending)

        self.depth += 1
        if self.depth > MAX_DEPTH:
            self.depth -= 1
            raise self._exhausted()
        var matched = self._body(index)
        self.depth -= 1

        if not matched:
            self.at = start_at
            self._truncate(mark_nodes, mark_pending)
            if slot != Int(_NO_SLOT):
                self._remember_failure(slot, start_at)
            return False

        # Adopt everything the body left behind, in order, and put one node in
        # its place.
        var node = UInt32(len(self.nodes))
        var first = NO_NODE
        if len(self.pending) > mark_pending:
            first = self.pending[mark_pending]
            for i in range(mark_pending, len(self.pending) - 1):
                self.nodes[Int(self.pending[i])].next_sibling = self.pending[
                    i + 1
                ]
        self.pending.resize(mark_pending, NO_NODE)
        self.nodes.append(
            ParseNode(
                UInt16(index), UInt32(start_at), UInt32(self.at), first, NO_NODE
            )
        )
        self.pending.append(node)
        return True

    def _body(mut self, index: Int) raises -> Bool:
        """Matches a rule's body, or runs its hand written matcher instead.

        Args:
            index: The rule index.

        Returns:
            Whether it matched.

        Raises:
            Error: As `rule`.
        """
        var matcher = self.grammar[].matchers[index]
        if matcher != 0:
            return self._overridden(matcher, self.grammar[].suggestions[index])
        return self._node(Int(self.grammar[].roots[index]))

    # -----------------------------------------------------------------------
    # Nodes
    # -----------------------------------------------------------------------

    def _node(mut self, node: Int) raises -> Bool:
        """Matches one node of a rule body.

        Every combinator that can consume before it fails puts the position back
        itself, so a caller never has to.

        Args:
            node: The node index.

        Returns:
            Whether it matched.

        Raises:
            Error: As `rule`.
        """
        var it = self.grammar[].nodes[node]

        if it.kind == NODE_REF:
            var target = Int(it.payload)
            if target == RULE_END_OF_INPUT:
                return self._terminal(self._kind(self.at) == TOKEN_END)
            return self.rule(target)

        if it.kind == NODE_LIT:
            return self._terminal(self._literal(it.flags, Int(it.payload)))

        if it.kind == NODE_KEYWORDS:
            return self._terminal(self._keyword_in(UInt8(it.payload)))

        if it.kind == NODE_SEQ:
            var start_at = self.at
            var mark_nodes = len(self.nodes)
            var mark_pending = len(self.pending)
            var cursor = Int(it.child)
            while cursor != 0:
                if not self._node(cursor):
                    self.at = start_at
                    self._truncate(mark_nodes, mark_pending)
                    return False
                cursor = Int(self.grammar[].nodes[cursor].sibling)
            return True

        if it.kind == NODE_CHOICE:
            var cursor = Int(it.child)
            while cursor != 0:
                if self._node(cursor):
                    return True
                cursor = Int(self.grammar[].nodes[cursor].sibling)
            return False

        if it.kind == NODE_OPT:
            _ = self._node(Int(it.child))
            return True

        if it.kind == NODE_STAR:
            self._repeat(Int(it.child))
            return True

        if it.kind == NODE_PLUS:
            if not self._node(Int(it.child)):
                return False
            self._repeat(Int(it.child))
            return True

        if it.kind == NODE_NOT:
            var start_at = self.at
            var mark_nodes = len(self.nodes)
            var mark_pending = len(self.pending)
            self.quiet += 1
            var found = self._node(Int(it.child))
            self.quiet -= 1
            self.at = start_at
            self._truncate(mark_nodes, mark_pending)
            return not found

        # A character class or a capture. Both live only inside %whitespace,
        # NumberLiteral, PlainIdentifier, QuotedIdentifier and StringLiteral, and
        # every one of those is either never referenced or reached only through a
        # rule the override list replaces. So getting here means the grammar grew
        # a shape the tokenizer does not cover, and a wrong parse would be worse
        # than a loud stop.
        if it.kind == NODE_CLASS or it.kind == NODE_CAPTURE:
            raise Error(
                "grammar: the matcher reached a character level node, which"
                " means a rule the tokenizer handles is no longer overridden"
            )
        raise Error(
            "grammar: the table holds a node kind this build has no case for"
        )

    def _repeat(mut self, node: Int) raises:
        """Matches a node as many times as it will go.

        Greedy with no backtracking into it, which is PEG and not regex. A run
        that matched nothing has to stop, or a node that can match the empty
        string spins forever.

        Args:
            node: The node index.

        Raises:
            Error: As `rule`.
        """
        while True:
            var before = self.at
            if not self._node(node):
                return
            if self.at == before:
                return

    def _truncate(mut self, mark_nodes: Int, mark_pending: Int):
        """Throws away everything a failed attempt built.

        A parent is always appended after its children, so nothing below the mark
        can point above it and there is nothing to unlink.

        Args:
            mark_nodes: The arena length at the start of the attempt.
            mark_pending: The pending stack length at the start of the attempt.
        """
        if len(self.nodes) != mark_nodes:
            self.nodes.resize(mark_nodes, ParseNode(0, 0, 0, NO_NODE, NO_NODE))
        if len(self.pending) != mark_pending:
            self.pending.resize(mark_pending, NO_NODE)

    # -----------------------------------------------------------------------
    # Terminals
    # -----------------------------------------------------------------------

    def _terminal(mut self, matched: Bool) -> Bool:
        """Records where a terminal was tried, and consumes it if it matched.

        Args:
            matched: Whether the token at the current position was the one
                wanted.

        Returns:
            The same answer, so a caller can return this directly.
        """
        if matched:
            # End of input matches without consuming, because there is nothing
            # there to consume and the vector's last token has to stay put.
            if self._kind(self.at) != TOKEN_END:
                self.at += 1
            return True
        if self.quiet == 0 and self.at > self.furthest:
            self.furthest = self.at
        return False

    def _literal(self, flags: UInt8, text: Int) -> Bool:
        """Says whether the current token is one the grammar writes out.

        Args:
            flags: The node's flags, carrying FLAG_WORD for a word literal.
            text: The interned string index, holding the literal upper cased.

        Returns:
            Whether it matches.
        """
        var kind = self._kind(self.at)
        var wanted = self.grammar[].strings[text].as_bytes()
        if flags & FLAG_WORD != 0:
            # A word literal is a keyword in the grammar's eyes whether or not
            # the keyword lists have it, and it never matches a quoted name:
            # `SELECT "select"` names a column called select.
            if kind != TOKEN_KEYWORD and kind != TOKEN_IDENTIFIER:
                return False
            return self._token_equals(wanted, fold=True)
        if (
            kind != TOKEN_OPERATOR
            and kind != TOKEN_PUNCTUATION
            and kind != TOKEN_PARAMETER
        ):
            return False
        return self._token_equals(wanted, fold=False)

    def _keyword_in(self, classes: UInt8) -> Bool:
        """Says whether the current token is a keyword in one of some classes.

        Args:
            classes: The class mask the grammar node carries.

        Returns:
            Whether it matches.
        """
        var token = self.tokens[][self.at]
        if token.kind != TOKEN_KEYWORD:
            return False
        return self.grammar[].keyword_classes[Int(token.keyword)] & classes != 0

    # -----------------------------------------------------------------------
    # The rules that are matched from code
    # -----------------------------------------------------------------------

    def _overridden(mut self, matcher: UInt8, suggestion: UInt8) -> Bool:
        """Matches one of the twenty four rules whose body is a placeholder.

        Args:
            matcher: The matcher class from the override list.
            suggestion: The suggestion it was built with, which is what tells
                the two identifier matchers what counts as a name here.

        Returns:
            Whether it matched.
        """
        if matcher == MATCHER_NUMBER_LITERAL:
            return self._terminal(self._kind(self.at) == TOKEN_NUMBER)
        if matcher == MATCHER_STRING_LITERAL:
            return self._terminal(self._kind(self.at) == TOKEN_STRING)
        if matcher == MATCHER_OPERATOR:
            return self._terminal(self._is_operator())
        return self._terminal(
            self._is_identifier(
                matcher == MATCHER_RESERVED_IDENTIFIER, suggestion
            )
        )

    def _is_identifier(self, reserved: Bool, suggestion: UInt8) -> Bool:
        """Says whether the current token can be a name in this position.

        Twelve of the twenty four overridden rules run the identifier matcher and
        nine run the reserved one, and the only difference between them is the
        keyword check. Dropping it is what makes `db.select` a legal column
        reference while a bare `select` is not a column name.

        Args:
            reserved: True for the reserved matcher, which takes any keyword.
            suggestion: Which kind of name the position wants.

        Returns:
            Whether it matches.
        """
        var token = self.tokens[][self.at]
        if token.kind == TOKEN_QUOTED_IDENTIFIER:
            return True
        if token.kind == TOKEN_STRING:
            # `FROM 'data.parquet'` parses because the table name position, and
            # only that one, counts a single quoted string as a name. There is no
            # grammar rule for it, and `CAST(x AS 'int')` is a syntax error for
            # the same reason there is not.
            return suggestion == SUGGEST_TABLE_NAME
        if token.kind == TOKEN_IDENTIFIER:
            return True
        if token.kind != TOKEN_KEYWORD:
            return False
        if reserved:
            return True
        var classes = self.grammar[].keyword_classes[Int(token.keyword)]
        if classes & KEYWORD_UNRESERVED != 0:
            return True
        return classes & _allowed_classes(suggestion) != 0

    def _is_operator(self) -> Bool:
        """Says whether the current token is a user defined operator.

        Deliberately narrow. A single character operator is spelled out in the
        grammar and so is every operator the dialect defines itself, so both are
        excluded here and would otherwise make an ordered choice take the wrong
        branch.

        Returns:
            Whether it matches.
        """
        var token = self.tokens[][self.at]
        if token.kind != TOKEN_OPERATOR or token.length < 2:
            return False
        var text = self._text(self.at)
        for i in range(len(text)):
            if not _is_operator_byte(text[i]):
                return False
        var listed = _DEFINED_OPERATORS.as_bytes()
        var start = 0
        for i in range(len(listed) + 1):
            if i == len(listed) or listed[i] == UInt8(32):
                if _same(text, listed[start:i]):
                    return False
                start = i + 1
        return True

    # -----------------------------------------------------------------------
    # Memoization
    # -----------------------------------------------------------------------

    def _failed_before(self, slot: Int, position: Int) -> Bool:
        """Says whether a memoized rule has already failed at a position.

        Args:
            slot: The rule's memo slot.
            position: The token position.

        Returns:
            Whether the bit is set.
        """
        if len(self.memo) == 0:
            return False
        var bit = position * self.grammar[].memo_count + slot
        return self.memo[bit >> 3] & (UInt8(1) << UInt8(bit & 7)) != 0

    def _remember_failure(mut self, slot: Int, position: Int):
        """Records that a memoized rule failed at a position.

        Args:
            slot: The rule's memo slot.
            position: The token position.
        """
        if len(self.memo) == 0:
            var bits = len(self.tokens[]) * self.grammar[].memo_count
            self.memo.resize((bits + 7) >> 3, 0)
        var bit = position * self.grammar[].memo_count + slot
        self.memo[bit >> 3] |= UInt8(1) << UInt8(bit & 7)

    # -----------------------------------------------------------------------
    # Errors and peeking
    # -----------------------------------------------------------------------

    def syntax_error(self) -> Error:
        """Builds the error for a parse that did not reach the end.

        PEG reports failure at the top level choice, having thrown away
        everything it learned, so the naive message is always at position zero.
        The furthest token a terminal was tried at is almost always where a
        reader would point, and it is what DuckDB uses too.

        Returns:
            The error, ready to raise.
        """
        var position = max(self.furthest, self.at)
        if self._kind(position) == TOKEN_END:
            # Nothing to name and nothing to point at, and DuckDB drops the LINE
            # and the caret here rather than pointing past the end.
            return Error("Parser Error: syntax error at end of input")
        return error_at(
            self.src,
            self._offset(position),
            String("syntax error at or near ", self._quoted(position)),
        )

    def _exhausted(self) -> Error:
        """Builds the error for a query that nests deeper than MAX_DEPTH.

        Returns:
            The error, in the words DuckDB uses when its own matcher runs out.
        """
        var position = min(self.at, len(self.tokens[]) - 1)
        return error_at(
            self.src,
            self._offset(position),
            String("memory exhausted at or near ", self._quoted(position)),
        )

    def _quoted(self, position: Int) -> String:
        """Renders a token the way an error message names it.

        Args:
            position: The token position.

        Returns:
            The token's own bytes in double quotes.
        """
        return String(
            '"', StringSlice(unsafe_from_utf8=self._text(position)), '"'
        )

    def _offset(self, position: Int) -> Int:
        """The byte a token starts at.

        Args:
            position: The token position.

        Returns:
            The byte offset into the query.
        """
        return Int(self.tokens[][position].start)

    def _kind(self, position: Int) -> UInt8:
        """The kind of the token at a position.

        Args:
            position: The token position, which is always in range because the
                vector ends in a TOKEN_END nobody consumes.

        Returns:
            The kind.
        """
        return self.tokens[][position].kind

    def _text(self, position: Int) -> Span[UInt8, Self.origin]:
        """The bytes a token covers.

        Args:
            position: The token position.

        Returns:
            The slice of the query text.
        """
        var token = self.tokens[][position]
        var start = Int(token.start)
        return self.src[start : start + Int(token.length)]

    def _token_equals(self, wanted: Span[UInt8, _], *, fold: Bool) -> Bool:
        """Compares the current token's bytes with a literal's.

        Args:
            wanted: The literal, which the generator interned upper cased.
            fold: Whether to compare case insensitively, which a word literal
                does and a punctuation literal has no reason to.

        Returns:
            Whether they are the same.
        """
        var text = self._text(self.at)
        if len(text) != len(wanted):
            return False
        for i in range(len(text)):
            var left = text[i]
            if fold and left >= 97 and left <= 122:
                left -= 32
            if left != wanted[i]:
                return False
        return True


def _allowed_classes(suggestion: UInt8) -> UInt8:
    """Says which keyword class a position tolerates as a name.

    This is the whole of how the grammar's five keyword classes turn into a
    decision. It comes from `IdentifierMatcher` in DuckDB, which reads the
    suggestion it was constructed with and nothing else, and the type and
    function case is upstream's `KEYWORD_TYPE_FUNC`, which is the two together.

    Args:
        suggestion: The suggestion the rule was built with.

    Returns:
        The class mask a keyword has to be in to be a name here.
    """
    if suggestion == SUGGEST_TYPE_NAME:
        return KEYWORD_TYPE_NAME
    if (
        suggestion == SUGGEST_SCALAR_FUNCTION_NAME
        or suggestion == SUGGEST_TABLE_FUNCTION_NAME
    ):
        return KEYWORD_TYPE_NAME | KEYWORD_FUNC_NAME
    return KEYWORD_COLUMN_NAME


def _is_operator_byte(c: UInt8) -> Bool:
    """Says whether a byte may appear in a user defined operator.

    Narrower than the tokenizer's run, which also takes `#` and a backtick.
    Those two can make a token and they cannot make an operator name.

    Args:
        c: The byte.

    Returns:
        Whether DuckDB would allow it.
    """
    return (
        c == UInt8(43)  # +
        or c == UInt8(45)  # -
        or c == UInt8(42)  # *
        or c == UInt8(47)  # /
        or c == UInt8(37)  # %
        or c == UInt8(94)  # ^
        or c == UInt8(60)  # <
        or c == UInt8(62)  # >
        or c == UInt8(61)  # =
        or c == UInt8(126)  # ~
        or c == UInt8(33)  # !
        or c == UInt8(64)  # @
        or c == UInt8(38)  # &
        or c == UInt8(124)  # |
    )


def _same(left: Span[UInt8, _], right: Span[UInt8, _]) -> Bool:
    """Compares two byte runs.

    Args:
        left: The first.
        right: The second.

    Returns:
        Whether they hold the same bytes.
    """
    if len(left) != len(right):
        return False
    for i in range(len(left)):
        if left[i] != right[i]:
            return False
    return True
