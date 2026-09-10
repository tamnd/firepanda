"""Sentences out of the same table the matcher walks.

A PEG grammar is a generator as well as a recognizer. The matcher reads a rule
and asks whether the tokens in front of it fit. Walking the same rule the other
way, choosing an alternative instead of trying each one and writing a token
instead of consuming it, produces a string built out of the grammar's own rules.
That is worth having for one reason above all the others: the vendored grammar
is DuckDB's, so a string built out of it is DuckDB SQL, and it reaches corners of
the dialect that nobody's test corpus visits because nobody has needed them yet.

Not everything it writes parses, and the reason is negative lookahead. Making
`!X` come out true means knowing what X would have matched, which is the problem
the parser exists to solve, so the walk skips those nodes and sometimes writes
the very thing the rule was there to forbid. About four in ten of the output is
refused somewhere, mostly there. That is a cost and not a flaw, because the
question a harness asks of this is whether two parsers agree about a string, not
whether the string was any good.

Three things make the walk terminate. Every node has a cost, meaning the fewest
tokens that finish it, computed once by fixpoint before any generation. A budget
counts tokens down, and once it is spent every choice takes its cheapest
alternative. And a depth cap stops the walk long before the matcher's own guard,
because a string this generator cannot get the matcher to accept is a string it
should not have written.

What comes out is a single statement with one space between every token. Spaces
everywhere are what makes the output safe to reassemble: two operator characters
that would have merged into one token stay two, and `-` followed by `-` is not
the start of a comment.

See docs/specs/sql/11-conformance.md section 4.
"""

from ..testing.rng import Rng
from .generated.rules import (
    MATCHER_NONE,
    MATCHER_NUMBER_LITERAL,
    MATCHER_OPERATOR,
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
)
from .table import Grammar

comptime MAX_TOKENS = 48
"""How many tokens a statement may spend before every choice goes cheapest.

Long statements are not more interesting than short ones. The bugs a generator
finds are combinations of rules rather than lengths, and a shorter case is a
better bug report.

This is where the walk stops spending, not where it stops writing. Running out
mid statement leaves frames open, and closing them costs whatever their cheapest
ending costs, so the finished statement is longer than the budget. The overshoot
is bounded, because there are at most `MAX_DEPTH` frames open and every one of
them has a finite cheapest ending, but the bound is loose. In practice a run
averages thirty tokens and the worst in twenty thousand was 182.
"""

comptime MAX_DEPTH = 40
"""How deep the walk may nest before it starts taking the way out.

Far below the matcher's own five hundred frame guard, because a string this
generator writes and the matcher then refuses for running out of stack tells
nobody anything.
"""

comptime MAX_REPEAT = 2
"""How many times a repetition repeats at most.

Two is enough to tell a list from a single item, which is the thing that goes
wrong. Ten is the same bug with more typing.
"""

comptime _SPACE = UInt8(32)
"""The byte the word pools are separated by."""

comptime _UNREACHABLE = UInt32(0xFFFF)
"""The cost of a node no finite string finishes. Saturating, so it spreads."""

comptime _IDENTIFIERS = StaticString("a b c t x y col tbl val n1 k2")
"""Names to write where the grammar wants one.

None of them is a keyword in any class, so every one of them is legal in every
name position, and they are short enough that a failing case stays readable.
"""

comptime _NUMBERS = StaticString("1 7 42 2.5 1e2")
"""Numbers to write where the grammar wants one."""

comptime _STRINGS = StaticString("'x' 'y'")
"""Strings to write where the grammar wants one.

No spaces inside one, because the pools are space separated and a literal with a
space in it would come back out as two words.
"""

comptime _OPERATOR = StaticString("@@")
"""What to write for `OperatorLiteral`.

That rule is a catalog lookup, meaning any operator token the grammar does not
spell out itself, so the one thing it must not be is an operator the grammar has
a node for. See `_DEFINED_OPERATORS` in the matcher.
"""


struct Generator[table: ImmOrigin](Movable):
    """A walk over the rule table that writes SQL instead of reading it.

    Deterministic in its seed. A harness prints the seed it started from and the
    case number it failed on, and that pair replays the exact statement.

    Parameters:
        table: The grammar, which outlives the generator.
    """

    var grammar: Pointer[Grammar, Self.table]
    """The rule table, read only."""

    var rng: Rng
    """The choices. One word of state, printed to replay a run."""

    var cost: List[UInt32]
    """The fewest tokens each node still needs, parallel to `grammar.nodes`.

    This is what makes the walk terminate rather than hope. A choice with a
    budget of three takes an alternative that costs three or less, and a choice
    with no budget left takes the cheapest one it has, which for any reachable
    rule is a finite string.
    """

    var out: String
    """The statement being written."""

    var budget: Int
    """Tokens left before every choice goes cheapest."""

    var depth: Int
    """How deep the walk is, against MAX_DEPTH."""

    var start: Int
    """The `Statement` rule, looked up once rather than once per statement."""

    def __init__(
        out self, ref[Self.table] grammar: Grammar, seed: UInt64
    ) raises:
        """Builds a generator over one grammar.

        The cost table is computed here, once, because it depends on nothing but
        the grammar and it costs a few sweeps over four thousand nodes.

        Args:
            grammar: The rule table.
            seed: The seed, which is the whole reproducibility story.

        Raises:
            Error: If the grammar has no `Statement` rule, which means the
                vendored files moved under us.
        """
        self.grammar = Pointer(to=grammar)
        self.rng = Rng(seed)
        self.cost = _costs(grammar)
        self.out = String()
        self.budget = MAX_TOKENS
        self.depth = 0
        self.start = -1
        for i in range(len(grammar.names)):
            if grammar.names[i] == "Statement":
                self.start = i
                break
        if self.start < 0:
            raise Error("grammar: no Statement rule to generate from")

    def statement(mut self) raises -> String:
        """Writes one statement.

        Returns:
            The SQL, one space between every token.

        Raises:
            Error: If the walk reached a node kind that belongs to the tokenizer.
        """
        return self.rule_text(self.start)

    def rule_text(mut self, rule: Int) raises -> String:
        """Writes one string that the given rule accepts.

        Aiming at one rule is how a fuzzer asks about one corner of the grammar,
        the same way `parse_rule` reads one.

        Args:
            rule: The rule index.

        Returns:
            The text.

        Raises:
            Error: If the walk reaches a node kind that belongs to the tokenizer,
                which means a rule stopped being overridden.
        """
        self.out = String()
        self.budget = MAX_TOKENS
        self.depth = 0
        self._rule(rule)
        return self.out.copy()

    def _rule(mut self, rule: Int) raises:
        """Writes one rule.

        Args:
            rule: The rule index.

        Raises:
            Error: As `rule_text`.
        """
        if rule == RULE_END_OF_INPUT:
            return
        var matcher = self.grammar[].matchers[rule]
        if matcher != MATCHER_NONE:
            self._token_for(matcher)
            return
        self.depth += 1
        if self.depth < MAX_DEPTH:
            self._node(Int(self.grammar[].roots[rule]))
        else:
            # Out of room. Every rule that is reachable at all has a cheapest
            # string, so take it and unwind.
            self._cheapest(Int(self.grammar[].roots[rule]))
        self.depth -= 1

    def _node(mut self, node: Int) raises:
        """Writes one node of a rule body.

        Args:
            node: The node index.

        Raises:
            Error: As `rule_text`.
        """
        if node == 0:
            return
        var it = self.grammar[].nodes[node]

        if it.kind == NODE_SEQ or it.kind == NODE_CAPTURE:
            var child = Int(it.child)
            while child != 0:
                self._node(child)
                child = Int(self.grammar[].nodes[child].sibling)
            return

        if it.kind == NODE_CHOICE:
            self._node(self._pick(node))
            return

        if it.kind == NODE_OPT:
            if self._affordable(Int(it.child)) and self.rng.next_bool():
                self._node(Int(it.child))
            return

        if it.kind == NODE_STAR or it.kind == NODE_PLUS:
            var least = 1 if it.kind == NODE_PLUS else 0
            var times = least
            if self._affordable(Int(it.child)):
                times = self.rng.next_range(least, MAX_REPEAT)
            for _ in range(times):
                self._node(Int(it.child))
            if it.kind == NODE_PLUS and times == 0:
                self._node(Int(it.child))
            return

        if it.kind == NODE_NOT:
            # A negative lookahead is a constraint on what comes next rather
            # than something to write. Nothing here checks it, so a generated
            # string can break one, and that is the one way this generator
            # produces SQL the matcher then refuses. The harness counts those
            # rather than pretending they cannot happen.
            return

        if it.kind == NODE_REF:
            self._rule(Int(it.payload))
            return

        if it.kind == NODE_LIT:
            self._write(self.grammar[].strings[Int(it.payload)])
            return

        if it.kind == NODE_KEYWORDS:
            self._keyword(UInt8(it.payload))
            return

        raise Error(
            "grammar: the generator reached a character level node, which means"
            " a rule the tokenizer handles is no longer overridden"
        )

    def _cheapest(mut self, node: Int) raises:
        """Writes the shortest string a node accepts.

        Args:
            node: The node index.

        Raises:
            Error: As `rule_text`.
        """
        var saved = self.budget
        self.budget = 0
        self._node(node)
        self.budget = saved

    def _pick(mut self, node: Int) -> Int:
        """Chooses one alternative of an ordered choice.

        Every alternative the budget can pay for is equally likely, which is not
        the distribution the grammar's order implies and is on purpose: the
        alternatives written last are the ones a corpus reaches least.

        Args:
            node: The choice node.

        Returns:
            The chosen child, or 0 if the choice has no children at all.
        """
        var best = 0
        var best_cost = _UNREACHABLE
        var affordable = 0
        var child = Int(self.grammar[].nodes[node].child)
        while child != 0:
            var each = self.cost[child]
            if each < best_cost:
                best_cost = each
                best = child
            if Int(each) <= self.budget:
                affordable += 1
            child = Int(self.grammar[].nodes[child].sibling)
        if affordable == 0:
            return best
        var wanted = self.rng.next_below(affordable)
        child = Int(self.grammar[].nodes[node].child)
        while child != 0:
            if Int(self.cost[child]) <= self.budget:
                if wanted == 0:
                    return child
                wanted -= 1
            child = Int(self.grammar[].nodes[child].sibling)
        return best

    def _affordable(self, node: Int) -> Bool:
        """Says whether there is budget left for a node.

        Args:
            node: The node index.

        Returns:
            Whether taking it keeps the walk finite and short.
        """
        return node != 0 and Int(self.cost[node]) <= self.budget

    def _keyword(mut self, wanted: UInt8):
        """Writes a keyword from one of the five classes.

        Args:
            wanted: The class mask the node carries.
        """
        var count = len(self.grammar[].keywords)
        var at = self.rng.next_below(count)
        for _ in range(count):
            if self.grammar[].keyword_classes[at] & wanted != 0:
                self._write(self.grammar[].keywords[at])
                return
            at += 1
            if at == count:
                at = 0

    def _token_for(mut self, matcher: UInt8):
        """Writes a token for one of the rules the tokenizer handles.

        Args:
            matcher: The matcher class the rule was overridden with.
        """
        if matcher == MATCHER_NUMBER_LITERAL:
            self._write(_one_of(_NUMBERS, self.rng.next_u64()))
        elif matcher == MATCHER_STRING_LITERAL:
            self._write(_one_of(_STRINGS, self.rng.next_u64()))
        elif matcher == MATCHER_OPERATOR:
            self._write(_OPERATOR)
        else:
            self._write(_one_of(_IDENTIFIERS, self.rng.next_u64()))

    def _write(mut self, text: StringSlice):
        """Appends one token.

        Args:
            text: The token's bytes.
        """
        if self.out:
            self.out += " "
        self.out += text
        self.budget -= 1


def _one_of(pool: StringSlice, draw: UInt64) -> StringSlice[pool.origin]:
    """Picks one space separated word out of a pool.

    A string rather than a list because a list of string literals is a comptime
    value that will not materialize, which is the same reason the matcher spells
    its operator list out this way.

    Args:
        pool: The words, space separated.
        draw: A random word to choose with.

    Returns:
        One of them.
    """
    var bytes = pool.as_bytes()
    var count = 1
    for i in range(len(bytes)):
        if bytes[i] == _SPACE:
            count += 1
    var wanted = Int(draw % UInt64(count))
    var start = 0
    for i in range(len(bytes)):
        if bytes[i] == _SPACE:
            if wanted == 0:
                return pool[byte=start:i]
            wanted -= 1
            start = i + 1
    return pool[byte=start:]


def _costs(grammar: Grammar) -> List[UInt32]:
    """Works out the fewest tokens each node still needs.

    A fixpoint, because a rule's cost is written in terms of the rules it calls
    and those call back. Everything starts unreachable and sweeps until a sweep
    changes nothing, which for this grammar is a handful of passes over four
    thousand nodes and happens once per generator.

    Args:
        grammar: The rule table.

    Returns:
        A cost per node, parallel to `grammar.nodes`.
    """
    var cost = List[UInt32]()
    cost.resize(len(grammar.nodes), _UNREACHABLE)
    var rules = List[UInt32]()
    rules.resize(len(grammar.roots), _UNREACHABLE)

    var moved = True
    while moved:
        moved = False
        for node in range(1, len(grammar.nodes)):
            var it = grammar.nodes[node]
            var now = _UNREACHABLE

            if it.kind == NODE_LIT or it.kind == NODE_KEYWORDS:
                now = 1
            elif it.kind == NODE_CLASS:
                # Never walked, because the rules that hold one are overridden.
                now = 1
            elif it.kind == NODE_NOT:
                now = 0
            elif it.kind == NODE_OPT or it.kind == NODE_STAR:
                now = 0
            elif it.kind == NODE_PLUS or it.kind == NODE_CAPTURE:
                now = cost[Int(it.child)]
            elif it.kind == NODE_REF:
                var target = Int(it.payload)
                if target == RULE_END_OF_INPUT:
                    now = 0
                elif grammar.matchers[target] != MATCHER_NONE:
                    now = 1
                else:
                    now = rules[target]
            elif it.kind == NODE_SEQ:
                now = 0
                var child = Int(it.child)
                while child != 0:
                    now = _plus(now, cost[child])
                    child = Int(grammar.nodes[child].sibling)
            elif it.kind == NODE_CHOICE:
                var child = Int(it.child)
                while child != 0:
                    if cost[child] < now:
                        now = cost[child]
                    child = Int(grammar.nodes[child].sibling)

            if now < cost[node]:
                cost[node] = now
                moved = True

        for rule in range(len(grammar.roots)):
            var root = Int(grammar.roots[rule])
            if cost[root] < rules[rule]:
                rules[rule] = cost[root]
                moved = True

    return cost^


def _plus(left: UInt32, right: UInt32) -> UInt32:
    """Adds two costs without wrapping.

    Args:
        left: One cost.
        right: The other.

    Returns:
        The sum, or unreachable if either side was.
    """
    if left >= _UNREACHABLE or right >= _UNREACHABLE:
        return _UNREACHABLE
    var sum = left + right
    return sum if sum < _UNREACHABLE else _UNREACHABLE
