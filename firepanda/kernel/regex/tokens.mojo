"""What a parsed pattern is made of.

The op codes are the ones `re._parser` produces, with the same names, and that
is the point of them rather than a convenience. The routing decision pandas
makes is a walk over that parser's tokens looking for three of these by name, so
a tree whose nodes do not correspond one to one with Python's cannot be checked
against Python's, and the check is the only thing that says the parser is right.

There is one node here Python does not have. Python represents a sequence as a
plain list, which an arena cannot do, so `OP_SEQ` is a node whose children are
the items in order. It carries no information and the router ignores it, which
is what keeps the correspondence exact where it matters.
"""


comptime OP_LITERAL: UInt8 = 1
"""One character, matching itself. `a` holds it in `a`."""

comptime OP_NOT_LITERAL: UInt8 = 2
"""Any character but one. Python produces this for a negated class holding a
single literal, so `[^a]` is this rather than an `OP_IN`."""

comptime OP_ANY: UInt8 = 3
"""A full stop, which is any character but a newline unless the pattern said
otherwise."""

comptime OP_IN: UInt8 = 4
"""A character class. Its children are the things in the brackets: literals,
ranges, categories, and an `OP_NEGATE` first when the class was negated."""

comptime OP_RANGE: UInt8 = 5
"""Two code points and everything between them, inclusive. `a` is the low end
and `b` is the high one."""

comptime OP_CATEGORY: UInt8 = 6
"""One of the six Perl classes. `a` says which, and what the six mean is the
single largest difference between the two engines, which is why it is a name
here and a table somewhere else."""

comptime OP_NEGATE: UInt8 = 7
"""The caret at the front of a class. A child of the `OP_IN` rather than a flag
on it, because that is where Python puts it."""

comptime OP_AT: UInt8 = 8
"""A position rather than a character. `a` says which."""

comptime OP_BRANCH: UInt8 = 9
"""Alternation. Its children are the alternatives, each an `OP_SEQ`."""

comptime OP_SEQ: UInt8 = 10
"""A run of items one after another. The one node with no counterpart in
Python's tokens, where a sequence is a bare list."""

comptime OP_SUBPATTERN: UInt8 = 11
"""A capturing group. `a` is its number, counting from one."""

comptime OP_MAX_REPEAT: UInt8 = 12
"""A greedy quantifier. `a` is the least number of times and `b` the most, which
is `MAXREPEAT` when the pattern did not say."""

comptime OP_MIN_REPEAT: UInt8 = 13
"""A lazy quantifier, which is the same counts read the other way round."""

comptime OP_POSSESSIVE_REPEAT: UInt8 = 14
"""A possessive quantifier, which Python has read since 3.11 and RE2 refuses.
That pairing is the reason it is a node rather than an error: pandas hands the
pattern to RE2 and RE2 is the one that complains."""

comptime OP_ATOMIC_GROUP: UInt8 = 15
"""`(?>...)`, which is the other construct Python reads and RE2 refuses."""

comptime OP_ASSERT: UInt8 = 16
"""A lookahead or a lookbehind. `a` is 1 for ahead and minus one for behind.
One of the three op codes that decides the routing."""

comptime OP_ASSERT_NOT: UInt8 = 17
"""A negative lookaround, with `a` read the same way. The second of the three."""

comptime OP_GROUPREF: UInt8 = 18
"""A backreference. `a` is the group number. The third of the three, and the one
that cannot be answered in linear time in general."""

comptime OP_GROUPREF_EXISTS: UInt8 = 19
"""`(?(1)yes|no)`, which asks whether a group took part. `a` is the group
number, the first child is the yes branch and the second is the no branch when
there is one."""


comptime OP_FAILURE: UInt8 = 20
"""A node that never matches. Python's parser produces this for a negative
lookaround with an empty body, since a body that always matches makes the
lookaround always fail, and that collapse is why `(?!)` is routed to RE2 and
raises an Arrow error for a pattern Python answers as a column of False."""


comptime AT_BEGINNING: UInt8 = 1
"""`^` outside multiline mode, which is the start of the text."""

comptime AT_BEGINNING_LINE: UInt8 = 2
"""`^` in multiline mode."""

comptime AT_BEGINNING_STRING: UInt8 = 3
"""`\\A`, which is the start of the text whatever the flags say."""

comptime AT_END: UInt8 = 4
"""`$` outside multiline mode. The two engines disagree about this one: Python
matches at the end of the text and also just before a newline that ends it, and
RE2 matches only at the end."""

comptime AT_END_LINE: UInt8 = 5
"""`$` in multiline mode."""

comptime AT_END_STRING: UInt8 = 6
"""`\\Z`, which is the end of the text and nothing else. pandas rewrites this to
RE2's `\\z` on the way to Arrow, which is upstream papering over one instance of
the difference `AT_END` is another instance of."""

comptime AT_BOUNDARY: UInt8 = 7
"""`\\b`, between a word character and something that is not one."""

comptime AT_NON_BOUNDARY: UInt8 = 8
"""`\\B`, anywhere `\\b` is not."""


comptime CATEGORY_DIGIT: UInt8 = 1
"""`\\d`. Every Unicode decimal digit to Python and `[0-9]` to RE2, which is the
difference document 76 opens with because it is the most written pattern there
is."""

comptime CATEGORY_NOT_DIGIT: UInt8 = 2
"""`\\D`."""

comptime CATEGORY_SPACE: UInt8 = 3
"""`\\s`."""

comptime CATEGORY_NOT_SPACE: UInt8 = 4
"""`\\S`."""

comptime CATEGORY_WORD: UInt8 = 5
"""`\\w`."""

comptime CATEGORY_NOT_WORD: UInt8 = 6
"""`\\W`."""


comptime MAXREPEAT: Int32 = 0x7FFFFFFF
"""What the upper bound of an unbounded quantifier holds. Python calls the same
thing `MAXREPEAT` and gives it a number too, for the same reason: a repeat with
no ceiling and a repeat with a very high one behave alike, so giving the first
one a number removes a case from everything downstream."""


@fieldwise_init
struct Node(Copyable, ImplicitlyCopyable, Movable):
    """One node of a parsed pattern.

    Six small integers and no owned memory, so the arena is a flat list that
    moves and copies without touching anything else. Everything a node needs to
    say beyond its op code fits in two payloads because the grammar's nodes are
    small: the widest thing here is a repeat, which is a lower bound, an upper
    bound and one child.
    """

    var op: UInt8
    """Which node this is."""

    var a: Int32
    """The first payload. A code point for a literal, a group number for a
    subpattern or a backreference, a lower bound for a repeat, a direction for
    an assertion, a which for a category or a position."""

    var b: Int32
    """The second payload, which only a range and a repeat use."""

    var first: Int32
    """The first child, or minus one."""

    var last: Int32
    """The last child, or minus one. Carried so that attaching a child is not a
    walk to the end of a linked list, which for a long alternation is the
    difference between reading a pattern once and reading it squared."""

    var next: Int32
    """The next sibling, or minus one."""
