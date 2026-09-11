"""Turns DuckDB's vendored PEG grammar into the table firepanda's matcher walks.

Input is firepanda/sql/grammar/, which tools/vendor_grammar.sh writes and nobody
edits by hand. Output is rules.mojo and keywords.mojo in firepanda/sql/generated/,
which are checked in so that a contributor with no Python and no network can still
build, and which CI regenerates and diffs so that nobody hand edits them either.
The package's __init__.mojo beside them is not written here, because other
generators put their tables in that directory too and the file that re-exports
them all belongs to none of them.

The output is not written inside the vendored directory, even though that reads
better, because a Mojo subpackage needs an __init__.mojo and putting one there
would mean the vendored tree has a file in it that upstream did not write. The
rule that the vendored tree is byte for byte upstream is worth more than the
tidier path.

The output is data rather than code. One flat array of nodes, walked by the
matcher, instead of 1,087 generated Mojo functions: 1,087 functions is a compile
time problem in a language whose compile times tools/compile_budget.py already
tracks, and an array can be read in a diff by a human. See
docs/specs/sql/03-the-grammar.md section 5.

Three things happen here that are not obvious from reading the .gram files.

The keyword lists are not grammar files, and the rules that refer to them do not
exist until something synthesizes them. Upstream's build turns each .list file
into one rule named after the file, so reserved_keyword.list becomes
`ReservedKeyword <- 'all' / 'analyse' / ...`. This does the same, except that the
alternatives never reach the node array: a five hundred way ordered choice is a
hash lookup written out longhand, so those five rules become a single keyword
class node and the words go into the tables in keywords.mojo.

Parameterized rules are expanded at their call sites. `List(Expression)` becomes
a rule named `List_Expression` with the parameter substituted, which costs a few
hundred extra rules and buys a matcher with no environment to thread through it.

Rule boundaries are whitespace sensitive, and getting them wrong changes the
language rather than failing. A rule ends at the first newline that is outside
brackets and does not follow a trailing `/`. The tokenizer below is a
reimplementation of upstream's scripts/parser/inline_grammar.py for exactly this
reason: the segmentation has to agree with theirs token for token.

Usage:
    python tools/gen_grammar.py            rewrite the generated files
    python tools/gen_grammar.py --check    fail if the checked in files are stale
    python tools/gen_grammar.py --stats    print the table shape and write nothing
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
GRAMMAR = ROOT / "firepanda" / "sql" / "grammar"
GENERATED = ROOT / "firepanda" / "sql" / "generated"

# Each keyword list becomes one rule, named after the file in upper camel case,
# matching what upstream's inline_grammar.py does. The name on the right is the
# class the tokenizer resolves a word against, and the order here is the order
# the classes are numbered in, so it is also the order in keywords.mojo.
KEYWORD_CLASSES = [
    ("reserved_keyword.list", "ReservedKeyword"),
    ("unreserved_keyword.list", "UnreservedKeyword"),
    ("column_name_keyword.list", "ColumnNameKeyword"),
    ("func_name_keyword.list", "FuncNameKeyword"),
    ("type_name_keyword.list", "TypeNameKeyword"),
]

# Referenced by the grammar and never defined in it, because upstream supplies it
# from the matcher rather than from the text. It gets an index one past the last
# real rule and a named constant in the generated file.
BUILTIN_RULES = ["EndOfInput"]

# The matchers matcher_overrides.list can name. Index 0 is "no override, walk
# the body", so the list order is also the wire encoding. The names are
# upstream's class names in snake case. Adding one here without writing the
# matcher for it in firepanda/sql/ is how a rule silently starts matching
# nothing, so the loader checks the count.
MATCHERS = [
    "",
    "identifier",
    "reserved_identifier",
    "number_literal",
    "string_literal",
    "operator",
]

# The suggestion each override was constructed with, in the same encoding. Index
# 0 is the literal string "none", which is what the two literal matchers and the
# operator matcher are built without.
#
# This looks like autocomplete trivia and is not. IdentifierMatcher reads the
# suggestion to pick which keyword category the position tolerates, and whether
# a single quoted string counts as a name there, so a table name accepts
# 'path.csv' and a type name does not. Keeping upstream's name rather than the
# two bits we derive from it means the next person can check the mapping against
# identifier_matcher.hpp without guessing what got thrown away.
SUGGESTIONS = [
    "none",
    "variable",
    "catalog_name",
    "schema_name",
    "table_name",
    "column_name",
    "type_name",
    "scalar_function_name",
    "table_function_name",
    "pragma_name",
    "setting_name",
]

# Defined in common.gram, but applied implicitly between tokens rather than by
# reference, so the matcher has to be told which rule it is.
WHITESPACE_RULE = "%whitespace"

# The first rule in common.gram, and the one a parse starts at. Rule indices move
# whenever the grammar is bumped, so the entry point is emitted as a constant
# rather than looked up by name at run time.
START_RULE = "Program"


class GrammarError(Exception):
    pass


# ---------------------------------------------------------------------------
# Tokenizing, and where one rule stops and the next begins.
# ---------------------------------------------------------------------------

LITERAL = "literal"
REFERENCE = "reference"
CALL = "call"
REGEX = "regex"
OPERATOR = "operator"

OPERATORS = "/?()*+!"


@dataclass
class Token:
    kind: str
    text: str
    line: int


@dataclass
class RawRule:
    name: str
    params: list[str]
    tokens: list[Token]
    source: str
    line: int


def tokenize(text: str, source: str) -> list[RawRule]:
    """Splits one .gram file into rules, each carrying a flat token list.

    Mirrors upstream's parse_peg_grammar. The state machine is theirs; the value
    of copying it rather than writing a tidier one is that rule boundaries and
    accepted characters are then the same by construction, and a divergence in
    either would be a silent change to the dialect.
    """
    rules: list[RawRule] = []
    name: str | None = None
    name_line = 0
    tokens: list[Token] = []
    state = "name"
    depth = 0
    after_slash = False
    params: list[str] = []
    i = 0
    line = 1

    def die(message: str) -> None:
        raise GrammarError(f"{source}:{line}: {message}")

    while i < len(text):
        c = text[i]
        if c == "#":
            while i < len(text) and text[i] != "\n":
                i += 1
            continue
        if state == "body" and c in "\n\r" and depth == 0 and not after_slash and tokens:
            rules.append(RawRule(name or "", params, tokens, source, name_line))
            name, tokens, params, state = None, [], [], "name"
            i += 1
            line += 1
            continue
        if c.isspace():
            if c == "\n":
                line += 1
            i += 1
            continue

        if state == "name":
            start = i
            if c == "%":
                i += 1
            while i < len(text) and text[i].isalnum():
                i += 1
            if i == start:
                die("expected a rule name")
            name, name_line, state = text[start:i], line, "separator"
            continue

        if state == "separator":
            if c == "(":
                i += 1
                start = i
                while i < len(text) and text[i].isalnum():
                    i += 1
                if start == i:
                    die(f"expected a parameter name in {name}")
                params.append(text[start:i])
                if i >= len(text) or text[i] != ")":
                    die(f"expected ')' after the parameter of {name}")
                i += 1
                continue
            if text[i : i + 2] != "<-":
                die(f"expected '<-' after {name}")
            i += 2
            state = "body"
            continue

        # state == "body"
        after_slash = False
        if c == "'":
            i += 1
            start = i
            while i < len(text) and text[i] != "'":
                if text[i] == "\\":
                    i += 1
                i += 1
            if i >= len(text):
                die(f"unterminated literal in {name}")
            tokens.append(Token(LITERAL, text[start:i], line))
            i += 1
            # Upstream rejects a case insensitive literal outright rather than
            # implementing it, so a 'FOO'i appearing upstream is a change we have
            # to notice rather than quietly accept.
            if i < len(text) and text[i] == "i":
                die(f"case insensitive literal in {name}, which upstream rejects too")
        elif c.isalnum():
            start = i
            while i < len(text) and text[i].isalnum():
                i += 1
            word = text[start:i]
            if i < len(text) and text[i] == "(":
                i += 1
                depth += 1
                tokens.append(Token(CALL, word, line))
            else:
                tokens.append(Token(REFERENCE, word, line))
        elif c in "[<":
            start = i
            closer = "]" if c == "[" else ">"
            while i < len(text) and text[i] != closer:
                if text[i] == "\\":
                    i += 1
                if i < len(text):
                    i += 1
            if i >= len(text):
                die(f"unterminated {c}{closer} in {name}")
            i += 1
            # A quantifier or the case insensitive marker binds to the class it
            # follows, and inside a capture it is part of the capture's own text.
            while i < len(text) and text[i] in "i":
                i += 1
            tokens.append(Token(REGEX, text[start:i], line))
        elif c in OPERATORS:
            if c == "(":
                depth += 1
            elif c == ")":
                if depth == 0:
                    die(f"unbalanced ')' in {name}")
                depth -= 1
            elif c == "/":
                after_slash = True
            tokens.append(Token(OPERATOR, c, line))
            i += 1
        else:
            die(f"unexpected {c!r} in {name}")

    if state == "separator":
        die(f"{name} has no definition")
    if state == "body":
        if not tokens:
            die(f"{name} is empty")
        rules.append(RawRule(name or "", params, tokens, source, name_line))
    return rules


# ---------------------------------------------------------------------------
# The tree, and the recursive descent that builds it.
# ---------------------------------------------------------------------------

SEQ = "seq"
CHOICE = "choice"
OPT = "opt"
STAR = "star"
PLUS = "plus"
NOT = "not"
REF = "ref"
LIT = "lit"
CLASS = "class"
CAPTURE = "capture"
KEYWORDS = "keywords"


@dataclass
class Node:
    kind: str
    text: str = ""
    children: list["Node"] = field(default_factory=list)


class Parser:
    """Turns one rule's token list into a tree.

    The grammar of the grammar, which is small enough to state in full:

        body     <- choice
        choice   <- sequence ('/' sequence)*
        sequence <- prefixed+
        prefixed <- '!'? suffixed
        suffixed <- primary ('?' / '*' / '+')*
        primary  <- '(' choice ')' / call / reference / literal / regex
    """

    def __init__(self, rule: RawRule):
        self.rule = rule
        self.tokens = rule.tokens
        self.pos = 0

    def die(self, message: str) -> None:
        line = self.tokens[min(self.pos, len(self.tokens) - 1)].line
        raise GrammarError(f"{self.rule.source}:{line}: in {self.rule.name}, {message}")

    def peek(self) -> Token | None:
        return self.tokens[self.pos] if self.pos < len(self.tokens) else None

    def at_operator(self, *chars: str) -> bool:
        t = self.peek()
        return t is not None and t.kind == OPERATOR and t.text in chars

    def parse(self) -> Node:
        node = self.choice()
        if self.pos != len(self.tokens):
            self.die("trailing tokens")
        return node

    def choice(self) -> Node:
        alternatives = [self.sequence()]
        while self.at_operator("/"):
            self.pos += 1
            alternatives.append(self.sequence())
        if len(alternatives) == 1:
            return alternatives[0]
        return Node(CHOICE, children=alternatives)

    def sequence(self) -> Node:
        items = []
        while True:
            t = self.peek()
            if t is None or (t.kind == OPERATOR and t.text in "/)"):
                break
            items.append(self.prefixed())
        if not items:
            self.die("empty alternative")
        if len(items) == 1:
            return items[0]
        return Node(SEQ, children=items)

    def prefixed(self) -> Node:
        if self.at_operator("!"):
            self.pos += 1
            return Node(NOT, children=[self.suffixed()])
        return self.suffixed()

    def suffixed(self) -> Node:
        node = self.primary()
        while self.at_operator("?", "*", "+"):
            suffix = self.tokens[self.pos].text
            self.pos += 1
            node = Node({"?": OPT, "*": STAR, "+": PLUS}[suffix], children=[node])
        return node

    def primary(self) -> Node:
        t = self.peek()
        if t is None:
            self.die("expected an expression")
            raise AssertionError  # unreachable, keeps the type checker happy
        if t.kind == OPERATOR and t.text == "(":
            self.pos += 1
            inner = self.choice()
            if not self.at_operator(")"):
                self.die("expected ')'")
            self.pos += 1
            return inner
        if t.kind == CALL:
            self.pos += 1
            argument = self.choice()
            if not self.at_operator(")"):
                self.die(f"expected ')' closing {t.text}(")
            self.pos += 1
            return Node(CALL, t.text, [argument])
        if t.kind == REFERENCE:
            self.pos += 1
            return Node(REF, t.text)
        if t.kind == LITERAL:
            self.pos += 1
            return Node(LIT, t.text)
        if t.kind == REGEX:
            self.pos += 1
            kind = CAPTURE if t.text.startswith("<") else CLASS
            return Node(kind, t.text)
        self.die(f"unexpected {t.text!r}")
        raise AssertionError  # unreachable


# ---------------------------------------------------------------------------
# Assembling every file into one rule set.
# ---------------------------------------------------------------------------


@dataclass
class Rule:
    name: str
    body: Node
    source: str
    memoized: bool = False
    matcher: str = ""
    suggestion: str = "none"


def load_rules() -> tuple[dict[str, Rule], dict[str, RawRule], dict[str, list[str]]]:
    statements = GRAMMAR / "statements"
    keywords_dir = GRAMMAR / "keywords"
    if not statements.is_dir() or not keywords_dir.is_dir():
        raise GrammarError(
            f"{GRAMMAR} does not look vendored; run tools/vendor_grammar.sh first"
        )

    # common.gram first and the rest sorted, which is upstream's order. Nothing
    # depends on it, but a stable order means the generated diff tracks the
    # grammar diff instead of the filesystem.
    files = [statements / "common.gram"] + sorted(
        p for p in statements.glob("*.gram") if p.name != "common.gram"
    )

    raw: dict[str, RawRule] = {}
    for path in files:
        for rule in tokenize(path.read_text(), path.name):
            if rule.name in raw:
                raise GrammarError(
                    f"{path.name}: duplicate rule {rule.name},"
                    f" already defined in {raw[rule.name].source}"
                )
            raw[rule.name] = rule

    words: dict[str, list[str]] = {}
    for filename, rule_name in KEYWORD_CLASSES:
        path = keywords_dir / filename
        if not path.is_file():
            raise GrammarError(f"missing keyword list {path}")
        words[rule_name] = [
            line.strip().lower() for line in path.read_text().splitlines() if line.strip()
        ]
        if rule_name in raw:
            raise GrammarError(
                f"{rule_name} is defined in {raw[rule_name].source} and would be"
                " overwritten by the keyword list of the same name"
            )

    rules: dict[str, Rule] = {}
    for name, rule in raw.items():
        if rule.params:
            continue  # expanded at its call sites instead
        rules[name] = Rule(name, Parser(rule).parse(), rule.source)

    # The payload is the class bit rather than the class number, so that the
    # matcher can test a word's mask against a node with one and.
    for index, (_, rule_name) in enumerate(KEYWORD_CLASSES):
        rules[rule_name] = Rule(rule_name, Node(KEYWORDS, str(1 << index)), "keywords")

    return rules, raw, words


def expand_calls(rules: dict[str, Rule], raw: dict[str, RawRule]) -> None:
    """Replaces every `List(X)` with a reference to a synthesized `List_X` rule.

    Done here rather than in the matcher because a matcher that understands
    parameters has to carry an environment through every recursion, and the
    grammar only ever passes one argument to one of two rules. Expansion is by
    substituted body, not by name, so `List(A / B)` and `List(A)` are different
    rules and an inline argument still works.
    """
    templates = {name: rule for name, rule in raw.items() if rule.params}
    for name, template in templates.items():
        if len(template.params) != 1:
            raise GrammarError(
                f"{template.source}: {name} takes {len(template.params)} parameters,"
                " and this generator only handles one"
            )

    synthesized: dict[str, str] = {}

    def label(node: Node) -> str:
        if node.kind == REF:
            return node.text
        if node.kind == CALL:
            return f"{node.text}_{label(node.children[0])}"
        # An inline argument such as `List(A / B)` has no name of its own, so it
        # gets a stable one derived from its shape. There are a handful of these
        # and a readable name beats a serial number in a stack trace.
        return "".join(ch for ch in describe(node) if ch.isalnum())[:48] or "Anon"

    def describe(node: Node) -> str:
        if node.kind in (LIT, CLASS, CAPTURE):
            return node.text
        if node.kind == REF:
            return node.text
        return node.kind + "".join(describe(c) for c in node.children)

    def substitute(node: Node, param: str, argument: Node) -> Node:
        if node.kind == REF and node.text == param:
            return clone(argument)
        return Node(node.kind, node.text, [substitute(c, param, argument) for c in node.children])

    def clone(node: Node) -> Node:
        return Node(node.kind, node.text, [clone(c) for c in node.children])

    def rewrite(node: Node) -> Node:
        children = [rewrite(c) for c in node.children]
        if node.kind != CALL:
            return Node(node.kind, node.text, children)
        template = templates.get(node.text)
        if template is None:
            raise GrammarError(f"call to {node.text}(), which is not a parameterized rule")
        argument = children[0]
        name = f"{node.text}_{label(argument)}"
        key = describe(argument)
        existing = synthesized.get(name)
        if existing is None:
            synthesized[name] = key
            body = substitute(Parser(template).parse(), template.params[0], argument)
            rules[name] = Rule(name, rewrite(body), template.source)
        elif existing != key:
            raise GrammarError(
                f"two different arguments to {node.text}() both name themselves {name}"
            )
        return Node(REF, name)

    for name in list(rules):
        rules[name] = Rule(
            name, rewrite(rules[name].body), rules[name].source, rules[name].memoized
        )


def check_references(rules: dict[str, Rule]) -> None:
    missing: list[str] = []

    def walk(node: Node, owner: str) -> None:
        if node.kind == REF and node.text not in rules and node.text not in BUILTIN_RULES:
            missing.append(f"{owner} references undefined rule {node.text}")
        for child in node.children:
            walk(child, owner)

    for name, rule in rules.items():
        walk(rule.body, name)
    if missing:
        raise GrammarError("\n".join(sorted(set(missing))))


def mark_matchers(rules: dict[str, Rule]) -> dict[str, str]:
    """Attaches DuckDB's matcher overrides to the rules they belong to.

    An override says the matcher does not walk this rule's body. It is not a
    convenience: `OperatorLiteral <- Identifier` in the grammar text, so without
    the override a bare `+` parses as an identifier. The body is kept anyway, so
    that the round trip below still compares the table against the grammar.
    """
    path = GRAMMAR / "matcher_overrides.list"
    pairs: dict[str, str] = {}
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        fields = line.split()
        if len(fields) != 3:
            raise GrammarError(
                f"matcher_overrides.list wants a rule, a matcher and a suggestion"
                f" on every line, and this line has {len(fields)}: {line!r}"
            )
        name, matcher, suggestion = fields
        if name not in rules:
            raise GrammarError(f"matcher_overrides.list names a rule that does not exist: {name}")
        if matcher not in MATCHERS:
            raise GrammarError(
                f"{name} wants matcher {matcher!r}, which this generator does not"
                f" know. Known matchers: {MATCHERS[1:]}. Read"
                " docs/specs/sql/04-the-parser.md section 2 before adding one."
            )
        if suggestion not in SUGGESTIONS:
            raise GrammarError(
                f"{name} was built with suggestion {suggestion!r}, which this"
                f" generator does not know. Known suggestions: {SUGGESTIONS}. Read"
                " docs/specs/sql/04-the-parser.md section 2 before adding one."
            )
        pairs[name] = matcher
        rules[name].matcher = matcher
        rules[name].suggestion = suggestion
    return pairs


def mark_memoized(rules: dict[str, Rule]) -> list[str]:
    path = GRAMMAR / "memoized_rules.list"
    names = [line.strip() for line in path.read_text().splitlines() if line.strip()]
    unknown = [n for n in names if n not in rules]
    if unknown:
        raise GrammarError(f"memoized_rules.list names rules that do not exist: {unknown}")
    for name in names:
        rules[name].memoized = True
    return names


# ---------------------------------------------------------------------------
# Flattening, which is the whole point.
# ---------------------------------------------------------------------------

# Kind codes. These are written into the generated table, so appending is free
# and reordering is not.
KIND_CODES = {
    SEQ: 1,
    CHOICE: 2,
    OPT: 3,
    STAR: 4,
    PLUS: 5,
    NOT: 6,
    REF: 7,
    LIT: 8,
    CLASS: 9,
    CAPTURE: 10,
    KEYWORDS: 11,
}

# A literal made only of letters is a word, and a word must not match the front
# of a longer identifier: SELECT selects, and `selection` is not SELECT followed
# by `ion`. Punctuation has no such rule, which is why the two are told apart
# here rather than in the matcher.
def is_word(literal: str) -> bool:
    return literal[:1].isalpha() and all(c.isalnum() or c == "_" for c in literal)


def unescape(literal: str) -> str:
    out, i = [], 0
    while i < len(literal):
        if literal[i] == "\\" and i + 1 < len(literal):
            out.append(literal[i + 1])
            i += 2
        else:
            out.append(literal[i])
            i += 1
    return "".join(out)


@dataclass
class Flat:
    nodes: list[tuple[int, int, int, int, int]]  # kind, flags, payload, child, sibling
    strings: list[str]
    rule_names: list[str]
    rule_roots: list[int]
    rule_memoized: list[bool]
    rule_matcher: list[int]
    rule_suggestion: list[int]


def flatten(rules: dict[str, Rule], order: list[str]) -> Flat:
    index_of = {name: i for i, name in enumerate(order)}
    for i, name in enumerate(BUILTIN_RULES):
        index_of.setdefault(name, len(order) + i)

    # Node 0 is the null node, so that a child or sibling of 0 means "none" and
    # the table needs no separate sentinel.
    nodes: list[list[int]] = [[0, 0, 0, 0, 0]]
    strings: list[str] = []
    string_index: dict[str, int] = {}

    def intern(text: str) -> int:
        if text not in string_index:
            string_index[text] = len(strings)
            strings.append(text)
        return string_index[text]

    def emit(node: Node) -> int:
        me = len(nodes)
        nodes.append([KIND_CODES[node.kind], 0, 0, 0, 0])
        if node.kind == REF:
            nodes[me][2] = index_of[node.text]
        elif node.kind == LIT:
            text = unescape(node.text)
            nodes[me][1] = 1 if is_word(text) else 0
            nodes[me][2] = intern(text.upper() if is_word(text) else text)
        elif node.kind in (CLASS, CAPTURE):
            nodes[me][2] = intern(node.text)
        elif node.kind == KEYWORDS:
            nodes[me][2] = int(node.text)
        else:
            previous = 0
            for child in node.children:
                emitted = emit(child)
                if previous == 0:
                    nodes[me][3] = emitted
                else:
                    nodes[previous][4] = emitted
                previous = emitted
        return me

    roots = []
    for name in order:
        roots.append(emit(rules[name].body))

    return Flat(
        [tuple(n) for n in nodes],
        strings,
        order,
        roots,
        [rules[name].memoized for name in order],
        [MATCHERS.index(rules[name].matcher) for name in order],
        [SUGGESTIONS.index(rules[name].suggestion) for name in order],
    )


# ---------------------------------------------------------------------------
# The first token filter.
# ---------------------------------------------------------------------------

# A token's key, which is what a node's filter is a set of. A keyword is its own
# key, because telling one keyword led alternative from another is the whole
# point, and there are 499 of them. The other token kinds get a key each, except
# that punctuation, operators and parameters are keyed on their first byte: a
# literal made of punctuation has to match every byte anyway, so the first one is
# already enough to tell two of them apart.
KEY_IDENTIFIER = 512
KEY_QUOTED_IDENTIFIER = 513
KEY_NUMBER = 514
KEY_STRING = 515
KEY_END = 516
KEY_BYTE = 640

# How wide the filter is. One word per node and one per token position, so
# widening it costs table size and a slower check and buys a lower false accept
# rate. See docs/specs/sql/04-the-parser.md section 4.
FILTER_BITS = 64
FILTER_ALL = (1 << FILTER_BITS) - 1

# The bytes _is_operator_byte in matcher.mojo takes, which is therefore what the
# operator matcher can start with.
OPERATOR_BYTES = "+-*/%^<>=~!@&|"

# The keyword class bits, in the order KEYWORD_CLASSES is written above. The
# reserved class is not here because no position takes a reserved word as a
# name: the matcher that would takes every keyword and does not look at classes.
CLASS_UNRESERVED = 2
CLASS_COLUMN = 4
CLASS_FUNC = 8
CLASS_TYPE = 16


def key_bit(key: int) -> int:
    """One token key as a bit in the filter word."""
    return 1 << (key % FILTER_BITS)


def allowed_classes(suggestion: str) -> int:
    """The keyword class a position tolerates as a name.

    The same table as _allowed_classes in matcher.mojo, and the two have to say
    the same thing or the filter rejects a token the matcher would have taken.
    """
    if suggestion == "type_name":
        return CLASS_TYPE
    if suggestion in ("scalar_function_name", "table_function_name"):
        return CLASS_TYPE | CLASS_FUNC
    return CLASS_COLUMN


def matcher_first(matcher: str, suggestion: str, keyword_classes: list[int]) -> int:
    """The keys one of the twenty four code matched rules can start with.

    Read off _overridden and _is_identifier in matcher.mojo rather than off the
    rule body, because the body is a placeholder there for the same reason it is
    a placeholder upstream.
    """
    if matcher == "number_literal":
        return key_bit(KEY_NUMBER)
    if matcher == "string_literal":
        return key_bit(KEY_STRING)
    if matcher == "operator":
        mask = 0
        for ch in OPERATOR_BYTES:
            mask |= key_bit(KEY_BYTE + ord(ch))
        return mask

    mask = key_bit(KEY_IDENTIFIER) | key_bit(KEY_QUOTED_IDENTIFIER)
    if suggestion == "table_name":
        # Only the table name position reads a single quoted string as a name,
        # which is what makes FROM 'data.parquet' parse.
        mask |= key_bit(KEY_STRING)
    if matcher == "reserved_identifier":
        # The reserved matcher is the one with the keyword check dropped, which
        # is what makes db.select a legal column reference and a bare select
        # not one. Every keyword is a name to it.
        for index in range(len(keyword_classes)):
            mask |= key_bit(index)
        return mask
    wanted = CLASS_UNRESERVED | allowed_classes(suggestion)
    for index, classes in enumerate(keyword_classes):
        if classes & wanted:
            mask |= key_bit(index)
    return mask


def compute_first(flat: Flat, members: dict[str, int]) -> list[int]:
    """One filter word per node, saying which tokens it can possibly start with.

    The matcher checks this before it walks a node, and a node whose word does
    not have the current token's bit cannot match, so it fails without
    recursing. That is worth doing because an ordered choice in this grammar is
    routinely fifty alternatives long and the token in hand rules out nearly all
    of them, and the matcher would otherwise find that out one recursion at a
    time.

    A node that can match the empty string gets every bit, so it is never
    filtered: it can succeed without looking at the token at all. Everything
    else gets the union over what its first consuming terminal can be, which is
    computed to a fixed point because the rule graph has cycles in it.

    The filter is allowed to be too generous and is never allowed to be too
    tight, so anything unclear here resolves to every bit.
    """
    by_code = {code: kind for kind, code in KIND_CODES.items()}
    nodes = flat.nodes
    rule_count = len(flat.rule_names)
    words = sorted(members)
    keyword_of = {word: index for index, word in enumerate(words)}
    keyword_classes = [members[word] for word in words]

    rule_nullable = [False] * rule_count
    rule_first = [0] * rule_count
    for index, matcher in enumerate(flat.rule_matcher):
        if matcher:
            rule_first[index] = matcher_first(
                MATCHERS[matcher],
                SUGGESTIONS[flat.rule_suggestion[index]],
                keyword_classes,
            )

    def nullable(node: int) -> bool:
        kind, _flags, payload, child, _sibling = nodes[node]
        code = by_code[kind]
        if code in (OPT, STAR, NOT):
            return True
        if code == PLUS:
            return nullable(child)
        if code == SEQ:
            cursor = child
            while cursor:
                if not nullable(cursor):
                    return False
                cursor = nodes[cursor][4]
            return True
        if code == CHOICE:
            cursor = child
            while cursor:
                if nullable(cursor):
                    return True
                cursor = nodes[cursor][4]
            return False
        if code == REF:
            # EndOfInput consumes nothing, but it is still a check against the
            # token in hand, so it filters like a terminal rather than like an
            # empty match.
            return payload < rule_count and rule_nullable[payload]
        if code in (CLASS, CAPTURE):
            # Unreachable: every rule holding one is overridden. Saying yes here
            # means never filtering one away, so the matcher still gets to stop
            # loudly if the grammar ever grows one somewhere reachable.
            return True
        return False

    def first(node: int) -> int:
        kind, flags, payload, child, _sibling = nodes[node]
        code = by_code[kind]
        if code == SEQ:
            mask = 0
            cursor = child
            while cursor:
                mask |= first(cursor)
                if not nullable(cursor):
                    break
                cursor = nodes[cursor][4]
            return mask
        if code == CHOICE:
            mask = 0
            cursor = child
            while cursor:
                mask |= first(cursor)
                cursor = nodes[cursor][4]
            return mask
        if code == NOT:
            # A negative lookahead never consumes, so it contributes nothing to
            # what the node around it can start with. Saying 0 here rather than
            # the child's keys is what lets `!X Y` filter on Y.
            return 0
        if code in (OPT, STAR, PLUS):
            return first(child)
        if code == REF:
            if payload >= rule_count:
                return key_bit(KEY_END)
            return rule_first[payload]
        if code == LIT:
            text = flat.strings[payload]
            if flags & 1:
                # A word literal the keyword lists do not have tokenizes as an
                # identifier, and one they do have never tokenizes as anything
                # but that keyword.
                found = keyword_of.get(text.lower())
                return key_bit(KEY_IDENTIFIER if found is None else found)
            return key_bit(KEY_BYTE + ord(text[0]))
        if code == KEYWORDS:
            mask = 0
            for index, classes in enumerate(keyword_classes):
                if classes & payload:
                    mask |= key_bit(index)
            return mask
        return FILTER_ALL

    # Two fixed points rather than one, because first() reads nullable() and
    # would otherwise chase a moving target. Both only ever grow, so both stop.
    changed = True
    while changed:
        changed = False
        for index, root in enumerate(flat.rule_roots):
            if flat.rule_matcher[index]:
                continue
            value = nullable(root)
            if value != rule_nullable[index]:
                rule_nullable[index] = value
                changed = True

    changed = True
    while changed:
        changed = False
        for index, root in enumerate(flat.rule_roots):
            if flat.rule_matcher[index]:
                continue
            value = first(root)
            if value != rule_first[index]:
                rule_first[index] = value
                changed = True

    out = [FILTER_ALL] * len(nodes)
    for node in range(1, len(nodes)):
        out[node] = FILTER_ALL if nullable(node) else first(node)
    return out


def unflatten(flat: Flat) -> dict[str, Node]:
    """Rebuilds the tree from the array, which is what makes the check possible."""
    by_code = {code: kind for kind, code in KIND_CODES.items()}
    names = list(flat.rule_names) + BUILTIN_RULES

    def read(index: int) -> Node:
        kind_code, flags, payload, child, _ = flat.nodes[index]
        kind = by_code[kind_code]
        if kind == REF:
            return Node(REF, names[payload])
        if kind == LIT:
            return Node(LIT, flat.strings[payload])
        if kind in (CLASS, CAPTURE):
            return Node(kind, flat.strings[payload])
        if kind == KEYWORDS:
            return Node(KEYWORDS, str(payload))
        children = []
        cursor = child
        while cursor:
            children.append(read(cursor))
            cursor = flat.nodes[cursor][4]
        return Node(kind, "", children)

    return {name: read(root) for name, root in zip(flat.rule_names, flat.rule_roots)}


def render_peg(node: Node) -> str:
    """Prints one node as PEG, fully parenthesized, for comparison and for --dump."""
    if node.kind == REF:
        return node.text
    if node.kind == LIT:
        return f"'{node.text}'"
    if node.kind in (CLASS, CAPTURE):
        return node.text
    if node.kind == KEYWORDS:
        return f"<keyword class {node.text}>"
    if node.kind == SEQ:
        return "(" + " ".join(render_peg(c) for c in node.children) + ")"
    if node.kind == CHOICE:
        return "(" + " / ".join(render_peg(c) for c in node.children) + ")"
    suffix = {OPT: "?", STAR: "*", PLUS: "+"}.get(node.kind)
    if suffix:
        return render_peg(node.children[0]) + suffix
    return "!" + render_peg(node.children[0])


def normalize(node: Node) -> Node:
    """The tree as flatten() will store it, so that a round trip can be compared.

    Flattening is lossy in exactly one way on purpose: a keyword literal is
    uppercased, because the matcher compares an uppercased word and doing it here
    means it is not done once per token at parse time.
    """
    if node.kind == LIT:
        text = unescape(node.text)
        return Node(LIT, text.upper() if is_word(text) else text)
    return Node(node.kind, node.text, [normalize(c) for c in node.children])


def verify_round_trip(rules: dict[str, Rule], flat: Flat) -> None:
    """Every rule in the 40 files is in the table, and says the same thing.

    This is the exit criterion for the vendoring stage, and it is worth having as
    code rather than as a claim: the flattening is the one step here where a bug
    would produce a table that still loads, still parses most SQL, and quietly
    accepts a different language.
    """
    rebuilt = unflatten(flat)
    if set(rebuilt) != set(rules):
        missing = sorted(set(rules) - set(rebuilt))
        extra = sorted(set(rebuilt) - set(rules))
        raise GrammarError(f"round trip lost {missing} and invented {extra}")
    for name in sorted(rules):
        want = render_peg(normalize(rules[name].body))
        got = render_peg(rebuilt[name])
        if want != got:
            raise GrammarError(f"round trip changed {name}:\n  in:  {want}\n  out: {got}")


# ---------------------------------------------------------------------------
# Emitting.
# ---------------------------------------------------------------------------

BANNER = """# Generated by tools/gen_grammar.py from firepanda/sql/grammar/.
# Do not edit. Run `python tools/gen_grammar.py` and commit the result.
"""


def escape_mojo(text: str) -> str:
    out = []
    for ch in text:
        if ch == "\\":
            out.append("\\\\")
        elif ch == '"':
            out.append('\\"')
        elif ch == "\n":
            out.append("\\n")
        elif ch == "\t":
            out.append("\\t")
        elif ch == "\r":
            out.append("\\r")
        elif 32 <= ord(ch) < 127:
            out.append(ch)
        else:
            out.append("\\x%02x" % ord(ch))
    return "".join(out)


def escape_table(line: str) -> str:
    """Escapes one line of the table for a triple quoted Mojo string.

    Newlines stay real newlines, because the diff on this file is the review of a
    grammar bump and a single sixty kilobyte line is not a diff. Only a backslash
    needs escaping: a lone `"` is fine inside `\"\"\"` as long as three never
    appear in a row, and the only quotes in the grammar are single ones inside
    character classes.
    """
    if '"""' in line:
        raise GrammarError(f"triple quote in the grammar would end the table: {line!r}")
    out = []
    for ch in line:
        if ch == "\\":
            out.append("\\\\")
        elif 32 <= ord(ch) < 127:
            out.append(ch)
        else:
            raise GrammarError(f"unprintable byte {ord(ch)} in the grammar: {line!r}")
    return "".join(out)


def render_rules(flat: Flat, first: list[int], grammar_sha: str) -> str:
    lines = [
        '"""The DuckDB grammar, flattened into one array of nodes.',
        "",
        BANNER.replace("# ", "").strip(),
        "",
        f"Vendored grammar: {grammar_sha}.",
        f"{len(flat.rule_names)} rules, {len(flat.nodes)} nodes, {len(flat.strings)} strings.",
        "",
        "The table is one string rather than a list of structs on purpose. A list",
        "literal with tens of thousands of entries is a compile time cost paid by",
        "everyone who builds firepanda, whether or not they use SQL, and the table",
        "is read once into a Grammar at startup. See docs/specs/sql/03-the-grammar.md",
        "section 5.",
        '"""',
        "",
        "",
        "# Node kinds, in the order tools/gen_grammar.py writes them.",
    ]
    for name, code in sorted(KIND_CODES.items(), key=lambda kv: kv[1]):
        lines.append(f"comptime NODE_{name.upper()}: UInt8 = {code}")
    lines += [
        "",
        "# Set on a literal node whose text is a word, meaning it must not match the",
        "# front of a longer identifier.",
        "comptime FLAG_WORD: UInt8 = 1",
        "",
        f"comptime RULE_COUNT: Int = {len(flat.rule_names)}",
        f"comptime NODE_COUNT: Int = {len(flat.nodes)}",
        f"comptime STRING_COUNT: Int = {len(flat.strings)}",
        "",
        "# Rules the matcher memoizes, by index. Copied from DuckDB rather than",
        "# chosen, see docs/specs/sql/04-the-parser.md.",
        f"comptime MEMOIZED_COUNT: Int = {sum(flat.rule_memoized)}",
        "",
        "# Rules the matcher matches itself rather than by walking the body, and",
        "# how many rules carry each. Vendored from the rule overrides in DuckDB's",
        "# compiled_grammar.cpp, because OperatorLiteral reads `Identifier` in the",
        "# grammar text and a bare `+` is not an identifier.",
    ] + [
        f"comptime MATCHER_{(name or 'none').upper()}: UInt8 = {i}"
        for i, name in enumerate(MATCHERS)
    ] + [
        f"comptime MATCHER_COUNT: Int = {len(MATCHERS)}",
        f"comptime OVERRIDDEN_COUNT: Int = {sum(1 for m in flat.rule_matcher if m)}",
        "",
        "# The suggestion each overridden rule was built with. The identifier",
        "# matcher reads it to decide which keyword class the position tolerates",
        "# and whether a single quoted string is a name there, so a table name and",
        "# a type name do not accept the same words.",
    ] + [
        f"comptime SUGGEST_{name.upper()}: UInt8 = {i}"
        for i, name in enumerate(SUGGESTIONS)
    ] + [
        f"comptime SUGGESTION_COUNT: Int = {len(SUGGESTIONS)}",
        "",
        "# Three rules the matcher has to know by name. EndOfInput is referenced",
        "# by the grammar and defined nowhere in it, and gets the index one past",
        "# the last real rule. %whitespace is defined but never referenced,",
        "# because it is applied between tokens rather than called. Program is",
        "# where a parse starts.",
        f"comptime RULE_END_OF_INPUT: Int = {flat.rule_names.index(BUILTIN_RULES[0]) if BUILTIN_RULES[0] in flat.rule_names else len(flat.rule_names)}",
        f"comptime RULE_WHITESPACE: Int = {flat.rule_names.index(WHITESPACE_RULE)}",
        f"comptime RULE_PROGRAM: Int = {flat.rule_names.index(START_RULE)}",
        "",
        "# How wide the first token filter is, and how many distinct words the",
        "# nodes share between them. See tools/gen_grammar.py and",
        "# docs/specs/sql/04-the-parser.md section 4.",
        f"comptime FILTER_BITS: Int = {FILTER_BITS}",
        f"comptime FILTER_COUNT: Int = {len(dict.fromkeys(first))}",
        "",
    ]

    # The table, in a line oriented text format so that a regeneration diff is
    # readable. Fields are space separated because no field can contain a space
    # except a string, and strings are length prefixed.
    body = []
    # The filter words come first because a node line names one by index. There
    # are 4,422 nodes and 237 distinct words between them, so the palette is
    # what keeps a sixth column on every node line from being a sixth table.
    palette = list(dict.fromkeys(first))
    slot_of = {word: index for index, word in enumerate(palette)}
    body.append("F " + str(len(palette)))
    for word in palette:
        # In two halves, because the reader reads decimal into an Int and a full
        # word does not fit in a signed one.
        body.append(f"{word >> 32} {word & 0xFFFFFFFF}")
    body.append("N " + str(len(flat.nodes)))
    for node, (kind, flags, payload, child, sibling) in enumerate(flat.nodes):
        body.append(
            f"{kind} {flags} {payload} {child} {sibling} {slot_of[first[node]]}"
        )
    body.append("S " + str(len(flat.strings)))
    for text in flat.strings:
        body.append(f"{len(text.encode())} {text}")
    body.append("R " + str(len(flat.rule_names)))
    for name, root, memoized in zip(flat.rule_names, flat.rule_roots, flat.rule_memoized):
        body.append(f"{root} {1 if memoized else 0} {name}")
    # The overrides get their own section rather than two more columns on every
    # R line, because 24 rules out of 1,187 have one and writing `0 0` on the
    # other 1,163 costs four kilobytes to say nothing.
    overrides = [
        (i, m, s)
        for i, (m, s) in enumerate(zip(flat.rule_matcher, flat.rule_suggestion))
        if m
    ]
    body.append("O " + str(len(overrides)))
    for index, matcher, suggestion in overrides:
        body.append(f"{index} {matcher} {suggestion}")

    lines.append('comptime TABLE: StaticString = """')
    lines.extend(escape_table(line) for line in body)
    lines.append('"""')
    lines.append("")
    return "\n".join(lines)


def render_keywords(words: dict[str, list[str]]) -> str:
    """One sorted table of every keyword, each with the classes it belongs to.

    Five separate tables would follow the five files, and would mean up to five
    lookups for a word that is in none of them, which is the common case because
    most words in a query are identifiers. The classes are not disjoint anyway:
    26 words are both function name and type name keywords, so five tables would
    also store those words twice.
    """
    members = keyword_members(words)
    lines = [
        '"""Every SQL keyword, with the classes it belongs to.',
        "",
        BANNER.replace("# ", "").strip(),
        "",
        "Sorted, deduplicated, lower case, one table. The tokenizer lower cases a",
        "word and bisects this once, which is what makes SELECT and select the same",
        "keyword and is also why an identifier that needs its case kept has to be",
        "quoted. A word can be in more than one class, so the classes are bits in a",
        "mask rather than an enum.",
        '"""',
        "",
        "",
        "# The classes, in the order tools/gen_grammar.py writes them. A keyword",
        "# class node in the rule table carries one of these as its payload, so a",
        "# word matches that node when its mask below and the payload overlap.",
    ]
    for index, (_, rule_name) in enumerate(KEYWORD_CLASSES):
        constant = "KEYWORD_" + camel_to_upper(rule_name.removesuffix("Keyword"))
        lines.append(f"comptime {constant}: UInt8 = {1 << index}")
    lines += [
        "",
        f"comptime KEYWORD_CLASS_COUNT: Int = {len(KEYWORD_CLASSES)}",
        f"comptime KEYWORD_COUNT: Int = {len(members)}",
        f"comptime KEYWORD_MAX_LENGTH: Int = {max(len(w) for w in members)}",
        "",
        "# One word per line, `mask word`, sorted by word so the table can be",
        "# bisected as it is read.",
        'comptime KEYWORDS: StaticString = """',
    ]
    for word in sorted(members):
        lines.append(escape_table(f"{members[word]} {word}"))
    lines += ['"""', ""]
    return "\n".join(lines)


def keyword_members(words: dict[str, list[str]]) -> dict[str, int]:
    """Every keyword with the mask of classes it is in.

    Sorting this by word gives the order the tokenizer bisects and therefore the
    number each keyword answers to, which the first token filter keys on.
    """
    members: dict[str, int] = {}
    for index, (_, rule_name) in enumerate(KEYWORD_CLASSES):
        for word in words[rule_name]:
            members[word] = members.get(word, 0) | (1 << index)
    return members


def camel_to_upper(name: str) -> str:
    out = []
    for i, ch in enumerate(name):
        if ch.isupper() and i:
            out.append("_")
        out.append(ch.upper())
    return "".join(out)


def vendored_sha() -> str:
    """One digest over every vendored input, so the output says what made it."""
    digest = hashlib.sha256()
    for path in sorted(GRAMMAR.rglob("*")):
        if path.is_dir() or path.name == "VENDOR":
            continue
        digest.update(path.relative_to(GRAMMAR).as_posix().encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()[:16]


def build() -> tuple[dict[str, str], Flat, dict[str, Rule], list[int]]:
    rules, raw, words = load_rules()
    expand_calls(rules, raw)
    check_references(rules)
    mark_matchers(rules)
    mark_memoized(rules)
    order = sorted(rules)
    flat = flatten(rules, order)
    verify_round_trip(rules, flat)
    first = compute_first(flat, keyword_members(words))
    # The package's `__init__.mojo` is not written here, although this generator
    # was once the only thing in the directory and did write it. More than one
    # generator puts a table in `generated/` now, so the file that re-exports
    # them all belongs to none of them, and a generator that rewrites it deletes
    # whatever the others put there. It is checked in and edited by hand, and
    # the imports in it fail to compile if a name it lists stops existing, which
    # is the check that matters.
    return {
        "rules.mojo": render_rules(flat, first, vendored_sha()),
        "keywords.mojo": render_keywords(words),
    }, flat, rules, first


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="fail if the output is stale")
    parser.add_argument("--stats", action="store_true", help="print the table shape only")
    parser.add_argument(
        "--dump",
        metavar="RULE",
        help="print one rule as the table stores it, and write nothing",
    )
    args = parser.parse_args()

    try:
        files, flat, rules, first = build()
    except GrammarError as error:
        print(f"error: {error}", file=sys.stderr)
        return 1

    if args.dump:
        rebuilt = unflatten(flat)
        if args.dump not in rebuilt:
            near = [n for n in rebuilt if args.dump.lower() in n.lower()][:8]
            print(f"error: no rule named {args.dump}", file=sys.stderr)
            if near:
                print("did you mean: " + ", ".join(sorted(near)), file=sys.stderr)
            return 1
        index = flat.rule_names.index(args.dump)
        memoized = " (memoized)" if flat.rule_memoized[index] else ""
        print(f"{args.dump}{memoized} <- {render_peg(rebuilt[args.dump])}")
        return 0

    if args.stats:
        print(f"rules      {len(flat.rule_names)}")
        print(f"memoized   {sum(flat.rule_memoized)}")
        print(f"overridden {sum(1 for m in flat.rule_matcher if m)}")
        print(f"nodes      {len(flat.nodes)}")
        print(f"strings    {len(flat.strings)}")
        filtering = [w for w in first if w != FILTER_ALL]
        print(f"filters    {len(dict.fromkeys(first))} distinct words")
        print(
            f"           {len(filtering)} of {len(first)} nodes filter,"
            f" {sum(bin(w).count('1') for w in filtering) / max(len(filtering), 1):.1f}"
            f" bits of {FILTER_BITS} set on average"
        )
        for name in sorted(files):
            print(f"{name:<10} {len(files[name])} bytes")
        print(f"generated  {sum(len(t) for t in files.values())} bytes of Mojo")
        return 0

    GENERATED.mkdir(parents=True, exist_ok=True)
    stale = []
    for name, text in files.items():
        path = GENERATED / name
        current = path.read_text() if path.is_file() else None
        if current == text:
            continue
        if args.check:
            stale.append(name)
        else:
            path.write_text(text)

    if args.check:
        if stale:
            print(
                "error: generated grammar is stale: " + ", ".join(sorted(stale)),
                file=sys.stderr,
            )
            print("run 'python tools/gen_grammar.py' and commit the result", file=sys.stderr)
            return 1
        print(f"generated grammar is current ({len(flat.rule_names)} rules)")
        return 0

    print(
        f"wrote {len(files)} files: {len(flat.rule_names)} rules,"
        f" {len(flat.nodes)} nodes, {len(flat.strings)} strings"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
