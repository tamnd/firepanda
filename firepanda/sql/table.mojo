"""Reading the generated grammar table back into something walkable.

`firepanda/sql/generated/rules.mojo` holds the whole DuckDB grammar as one text
blob: 1,187 rules over a flat array of nodes. This file turns that blob into a
`Grammar`. Nothing here knows what SQL is, it knows what a node is.

The table is a string rather than a Mojo list literal because a list literal with
thousands of entries is a compile time cost paid by everyone who builds
firepanda, whether or not they ever run a query, and because a string is a
readable diff when the grammar is bumped. The price of that choice is this file,
and one pass over 90 KB the first time a program asks for a grammar.

`Grammar` is a value with no global behind it. Whoever wants one holds one, and a
program that never mentions SQL never builds one. See
docs/specs/sql/03-the-grammar.md section 5.
"""

from .generated.keywords import KEYWORD_COUNT, KEYWORDS
from .generated.rules import (
    MEMOIZED_COUNT,
    NODE_CAPTURE,
    NODE_CHOICE,
    NODE_CLASS,
    NODE_COUNT,
    NODE_KEYWORDS,
    NODE_LIT,
    NODE_NOT,
    NODE_OPT,
    NODE_PLUS,
    NODE_REF,
    NODE_SEQ,
    NODE_STAR,
    RULE_COUNT,
    STRING_COUNT,
    TABLE,
)

comptime NEWLINE = UInt8(10)
comptime SPACE = UInt8(32)
comptime ZERO = UInt8(48)
comptime NINE = UInt8(57)


@fieldwise_init
struct GrammarNode(ImplicitlyCopyable, Movable):
    """One node of one rule.

    A rule body is a tree, and the tree is stored as first child plus next
    sibling so that the whole grammar is one array with no allocation per node.
    Index 0 is the null node, so a `child` or `sibling` of 0 means there is none.
    """

    var kind: UInt8
    """One of the `NODE_*` constants in the generated table."""

    var flags: UInt8
    """`FLAG_WORD` on a literal whose text is a word rather than punctuation."""

    var payload: UInt32
    """A rule index, a string index or a keyword class mask, by kind."""

    var child: UInt32
    """The first child, or 0 for a leaf."""

    var sibling: UInt32
    """The next child of this node's parent, or 0 if this is the last one."""


struct Grammar(Movable):
    """The whole dialect: every rule, every node, every literal, every keyword.

    Built once from the generated table and then read only. Holding one costs a
    couple of hundred kilobytes, which is the price of not reparsing the grammar
    per query.
    """

    var nodes: List[GrammarNode]
    """Every node of every rule, indexed by the `child` and `sibling` fields."""

    var strings: List[String]
    """Literal and character class text, indexed by a node's payload."""

    var names: List[String]
    """Rule names, in the order the generator assigns rule indices."""

    var roots: List[UInt32]
    """The root node of each rule, indexed the same way as `names`."""

    var memoized: List[Bool]
    """Whether each rule is on DuckDB's packrat list. See document 04."""

    var keywords: List[String]
    """Every keyword, lower case and sorted, so a lookup can bisect."""

    var keyword_classes: List[UInt8]
    """The classes each keyword is in, as a mask, parallel to `keywords`."""

    def __init__(out self) raises:
        """Reads the generated tables.

        Raises:
            Error: If a table is malformed, which means the generator and this
                reader have gone out of step. That is a build problem rather
                than anything a caller can do something about.
        """
        self.nodes = List[GrammarNode]()
        self.strings = List[String]()
        self.names = List[String]()
        self.roots = List[UInt32]()
        self.memoized = List[Bool]()
        self.keywords = List[String]()
        self.keyword_classes = List[UInt8]()

        var reader = _Reader(TABLE.as_bytes())

        var node_count = reader.section(UInt8(ord("N")))
        if node_count != NODE_COUNT:
            raise Error("grammar table: node count disagrees with NODE_COUNT")
        self.nodes.reserve(node_count)
        for _ in range(node_count):
            var kind = UInt8(reader.number())
            var flags = UInt8(reader.number())
            var payload = UInt32(reader.number())
            var child = UInt32(reader.number())
            var sibling = UInt32(reader.number())
            reader.end_of_line()
            self.nodes.append(GrammarNode(kind, flags, payload, child, sibling))

        var string_count = reader.section(UInt8(ord("S")))
        if string_count != STRING_COUNT:
            raise Error(
                "grammar table: string count disagrees with STRING_COUNT"
            )
        self.strings.reserve(string_count)
        for _ in range(string_count):
            # Length prefixed, because a literal can be a single space and a
            # space is also the field separator.
            var length = reader.number()
            self.strings.append(reader.text(length))
            reader.end_of_line()

        var rule_count = reader.section(UInt8(ord("R")))
        if rule_count != RULE_COUNT:
            raise Error("grammar table: rule count disagrees with RULE_COUNT")
        self.names.reserve(rule_count)
        self.roots.reserve(rule_count)
        self.memoized.reserve(rule_count)
        for _ in range(rule_count):
            self.roots.append(UInt32(reader.number()))
            self.memoized.append(reader.number() == 1)
            self.names.append(reader.rest_of_line())

        reader.expect_end()

        var words = _Reader(KEYWORDS.as_bytes())
        self.keywords.reserve(KEYWORD_COUNT)
        self.keyword_classes.reserve(KEYWORD_COUNT)
        while not words.at_end():
            self.keyword_classes.append(UInt8(words.number()))
            self.keywords.append(words.rest_of_line())
        if len(self.keywords) != KEYWORD_COUNT:
            raise Error(
                "keyword table: word count disagrees with KEYWORD_COUNT"
            )

    def rule(self, name: StringSlice) -> Int:
        """Finds a rule by name.

        Linear, because this is for tests and for error messages. The matcher
        works in rule indices from the moment it starts and never asks.

        Args:
            name: The rule name, spelled the way the grammar spells it.

        Returns:
            The rule index, or -1 if there is no such rule.
        """
        for i in range(len(self.names)):
            if self.names[i] == name:
                return i
        return -1

    def keyword_class(self, word: StringSlice) -> UInt8:
        """Looks a word up in the keyword table.

        One bisection over one sorted table, rather than one lookup per class.
        The classes overlap, 26 words are both a function name and a type name
        keyword, and most words in a query are not keywords at all, so five
        tables would mean five misses for the common case.

        Args:
            word: The word, already lower cased by the caller.

        Returns:
            The mask of classes the word is in, or 0 if it is not a keyword.
        """
        var low = 0
        var high = len(self.keywords)
        while low < high:
            var middle = (low + high) >> 1
            var order = _compare(
                self.keywords[middle].as_bytes(), word.as_bytes()
            )
            if order < 0:
                low = middle + 1
            elif order > 0:
                high = middle
            else:
                return self.keyword_classes[middle]
        return 0

    def children(self, node: Int) -> List[Int]:
        """Collects one node's children in order.

        Args:
            node: The node index.

        Returns:
            The child indices, empty for a leaf.
        """
        var out = List[Int]()
        var cursor = Int(self.nodes[node].child)
        while cursor != 0:
            out.append(cursor)
            cursor = Int(self.nodes[cursor].sibling)
        return out^

    def write_rule(self, index: Int) -> String:
        """Renders one rule back as PEG, fully parenthesized.

        This is how a test checks that the table still says what the grammar
        says, and it is also what a parse failure gets printed next to.

        Args:
            index: The rule index.

        Returns:
            The rule body as PEG text.
        """
        return self._write_node(Int(self.roots[index]))

    def _write_node(self, node: Int) -> String:
        """Renders one node and everything under it.

        Args:
            node: The node index.

        Returns:
            The node as PEG text.
        """
        var it = self.nodes[node]
        if it.kind == NODE_REF:
            var target = Int(it.payload)
            if target >= len(self.names):
                return String("EndOfInput")
            return self.names[target]
        if it.kind == NODE_LIT:
            return String("'", self.strings[Int(it.payload)], "'")
        if it.kind == NODE_CLASS or it.kind == NODE_CAPTURE:
            return self.strings[Int(it.payload)]
        if it.kind == NODE_KEYWORDS:
            return String("<keywords ", Int(it.payload), ">")

        var parts = self.children(node)
        if it.kind == NODE_SEQ or it.kind == NODE_CHOICE:
            var separator = StaticString(
                " "
            ) if it.kind == NODE_SEQ else StaticString(" / ")
            var out = String("(")
            for i in range(len(parts)):
                if i > 0:
                    out += separator
                out += self._write_node(parts[i])
            return out + ")"

        var inner = self._write_node(parts[0])
        if it.kind == NODE_OPT:
            return inner + "?"
        if it.kind == NODE_STAR:
            return inner + "*"
        if it.kind == NODE_PLUS:
            return inner + "+"
        if it.kind == NODE_NOT:
            return String("!") + inner
        return inner


def memoized_rules(grammar: Grammar) raises -> List[Int]:
    """Lists the rules the matcher memoizes.

    Copied from DuckDB rather than chosen, because the list is a decision
    somebody made with a profiler on a workload we do not have yet. Document 04
    says what would have to be true before we change it.

    Args:
        grammar: A loaded grammar.

    Returns:
        The rule indices, ascending.

    Raises:
        Error: If the count disagrees with the generated constant.
    """
    var out = List[Int]()
    for i in range(len(grammar.memoized)):
        if grammar.memoized[i]:
            out.append(i)
    if len(out) != MEMOIZED_COUNT:
        raise Error(
            "grammar table: memoized rule count disagrees with the table"
        )
    return out^


def _compare(left: Span[UInt8, _], right: Span[UInt8, _]) -> Int:
    """Orders two byte strings the way the generator sorted them.

    Args:
        left: The first string.
        right: The second string.

    Returns:
        Negative if left sorts first, positive if right does, zero if equal.
    """
    var shared = min(len(left), len(right))
    for i in range(shared):
        if left[i] != right[i]:
            return -1 if left[i] < right[i] else 1
    return len(left) - len(right)


struct _Reader[origin: ImmOrigin](Movable):
    """A cursor over one generated table.

    Deliberately dumb. The text it reads is written by a script in this
    repository and read only here, so the only errors it can meet are build
    errors, and the right answer to one is a clear message rather than recovery.
    """

    var bytes: Span[UInt8, Self.origin]
    """The table text."""

    var at: Int
    """How far in we are, in bytes."""

    def __init__(out self, bytes: Span[UInt8, Self.origin]):
        """Starts at the beginning.

        Args:
            bytes: The table text, which begins with the newline that follows
                the opening triple quote.
        """
        self.bytes = bytes
        self.at = 0

    def at_end(mut self) -> Bool:
        """Steps over blank lines and reports whether anything is left.

        Returns:
            True if the table is fully read.
        """
        while self.at < len(self.bytes) and self.bytes[self.at] == NEWLINE:
            self.at += 1
        return self.at >= len(self.bytes)

    def section(mut self, header: UInt8) raises -> Int:
        """Reads a `<header> <count>` line.

        Args:
            header: The single letter that names the section.

        Returns:
            The number of records the section declares.

        Raises:
            Error: If the next line does not start that section.
        """
        _ = self.at_end()
        if self.at >= len(self.bytes) or self.bytes[self.at] != header:
            raise Error(
                "grammar table: expected a section header at byte ", self.at
            )
        self.at += 1
        var count = self.number()
        self.end_of_line()
        return count

    def number(mut self) raises -> Int:
        """Reads one non negative integer, and any space in front of it.

        Returns:
            The value.

        Raises:
            Error: If there is no digit where one was expected.
        """
        while self.at < len(self.bytes) and self.bytes[self.at] == SPACE:
            self.at += 1
        var start = self.at
        var value = 0
        while self.at < len(self.bytes):
            var b = self.bytes[self.at]
            if b < ZERO or b > NINE:
                break
            value = value * 10 + Int(b - ZERO)
            self.at += 1
        if self.at == start:
            raise Error("grammar table: expected a number at byte ", start)
        return value

    def text(mut self, length: Int) raises -> String:
        """Reads a fixed number of bytes as text.

        Args:
            length: The byte length, which the table writes in front of the
                text so that a literal space survives the round trip.

        Returns:
            The text.

        Raises:
            Error: If the table ends inside the text.
        """
        if self.at < len(self.bytes) and self.bytes[self.at] == SPACE:
            self.at += 1
        if self.at + length > len(self.bytes):
            raise Error(
                "grammar table: a string runs past the end of the table"
            )
        var out = String(
            StringSlice(unsafe_from_utf8=self.bytes[self.at : self.at + length])
        )
        self.at += length
        return out^

    def rest_of_line(mut self) -> String:
        """Reads everything up to the next newline, and steps over it.

        Returns:
            The rest of the line, without the newline.
        """
        if self.at < len(self.bytes) and self.bytes[self.at] == SPACE:
            self.at += 1
        var start = self.at
        while self.at < len(self.bytes) and self.bytes[self.at] != NEWLINE:
            self.at += 1
        var out = String(
            StringSlice(unsafe_from_utf8=self.bytes[start : self.at])
        )
        if self.at < len(self.bytes):
            self.at += 1
        return out^

    def end_of_line(mut self) raises:
        """Consumes the newline that ends a record.

        Raises:
            Error: If anything else is left on the line.
        """
        if self.at < len(self.bytes) and self.bytes[self.at] != NEWLINE:
            raise Error("grammar table: unread text at byte ", self.at)
        self.at += 1

    def expect_end(mut self) raises:
        """Checks that the whole table was read.

        Raises:
            Error: If any record is left over.
        """
        if not self.at_end():
            raise Error("grammar table: unread records at byte ", self.at)
