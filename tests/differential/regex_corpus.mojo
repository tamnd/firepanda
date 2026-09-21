"""The patterns the two regular expression differentials both ask about.

One corpus rather than two because the two programs are asking different
questions about the same thing. `regex.mojo` asks which engine a pattern reaches
and `regex_match.mojo` asks what that engine then answers, and a pattern that
finds a routing mistake is exactly the kind of pattern likely to find a matching
mistake as well. Written down twice they would drift, and the one that drifted
would be the one nobody was looking at.

Three quarters of the atoms are things a person would call malformed, which is
the point in both programs. A grammar is defined as much by what it refuses as
by what it reads, pandas turns a refusal into a routing decision, and RE2 turns
a different refusal into an exception out of the caller's `str.contains`.
"""

comptime SHOWN = 30
"""How many disagreements of each kind are printed before the rest are counted.

Thirty because a systematic mistake in a grammar shows up in the first few
examples and the rest of the list is the same mistake wearing different
characters, and a report nobody scrolls to the end of is a report with its
conclusion off the screen.
"""


struct _Rng(Movable):
    """A small deterministic generator.

    Not the standard library's, because the corpus has to be the same corpus on
    every machine and on every run: a differential whose input moves is one
    where a fix cannot be told from a reshuffle.
    """

    var state: UInt64
    """Everything it knows."""

    def __init__(out self, seed: UInt64):
        """Starts at a seed.

        Args:
            seed: Where to start. Zero is turned into one, since this generator
                has nowhere to go from zero.
        """
        self.state = seed if seed != 0 else 1

    def next(mut self) -> UInt64:
        """The next number.

        Returns:
            A number, well mixed enough for choosing among a few dozen
            fragments, which is all it is asked to do.
        """
        var x = self.state
        x ^= x >> 12
        x ^= x << 25
        x ^= x >> 27
        self.state = x
        return x * 0x2545F4914F6CDD1D

    def below(mut self, bound: Int) -> Int:
        """A number under a bound.

        Args:
            bound: One past the largest wanted.

        Returns:
            Something from zero up to the bound.
        """
        return Int(self.next() >> 33) % bound


def _atoms() -> List[String]:
    """The pieces with nothing underneath them.

    Three quarters of these are things a person would call malformed, which is
    the point. A grammar is defined as much by what it refuses as by what it
    reads, and pandas turns a refusal into a routing decision, so a parser that
    is right about every valid pattern and wrong about `\\x4` sends that pattern
    to the wrong engine.

    Returns:
        The list, in no order that means anything.
    """
    var out: List[String] = [
        "a",
        "b",
        "z",
        "0",
        "9",
        "_",
        " ",
        "é",
        "漢",
        ".",
        "^",
        "$",
        "\\d",
        "\\D",
        "\\s",
        "\\S",
        "\\w",
        "\\W",
        "\\b",
        "\\B",
        "\\A",
        "\\Z",
        "\\z",
        "\\G",
        "\\1",
        "\\2",
        "\\9",
        "\\0",
        "\\12",
        "\\123",
        "\\400",
        "\\x41",
        "\\x4",
        "\\xzz",
        "\\u0041",
        "\\u00",
        "\\U00000041",
        "\\U0011FFFF",
        "\\N{BULLET}",
        "\\N{NOT A CHARACTER NAME}",
        "\\N",
        "\\Nx",
        "\\p{L}",
        "\\P{L}",
        "\\p{Lu}",
        "\\p{Nd}",
        "\\p{Greek}",
        "\\p{Arabic}",
        "\\p{Any}",
        "\\p{Cs}",
        "\\P{Nd}",
        "\\p{^Greek}",
        "\\P{^Lu}",
        "\\pL",
        "\\pN",
        "\\PL",
        "\\p{Cn}",
        "\\p{Foo}",
        "\\p{latin}",
        "\\p{IsGreek}",
        "\\p",
        "\\p{",
        "\\p{L",
        "\\k<n>",
        "\\Q",
        "\\e",
        "\\a",
        "\\f",
        "\\n",
        "\\r",
        "\\t",
        "\\v",
        "\\-",
        "\\ ",
        "\\.",
        "\\\\",
        "\\",
        "[abc]",
        "[^abc]",
        "[a-z]",
        "[z-a]",
        "[]",
        "[^]",
        "[]]",
        "[a-]",
        "[-a]",
        "[\\d]",
        "[\\w-]",
        "[[:alpha:]]",
        "[a\\]b]",
        "[\\x41-\\x5a]",
        "[^\\D]",
        "[a",
        "[a-\\d]",
        "[\\d-a]",
        "[é-漢]",
        "(?#note)",
        "(?i)",
        "(?s)",
        "(?m)",
        "(?x)",
        "(?a)",
        "(?L)",
        "(?u)",
        "(?-i)",
        "(?i-s)",
        "(?y)",
        "(?P=n)",
        "(?P=missing)",
        "(?P>n)",
        "*",
        "+",
        "?",
        "{2}",
        "{2,3}",
        "{,3}",
        "{2,}",
        "{}",
        "{2",
        "}",
        "]",
        "(",
        ")",
        "|",
        "-",
        "",
    ]
    return out^


def _quantifiers() -> List[String]:
    """The suffixes that repeat whatever is in front of them.

    The possessive ones are here even though RE2 refuses them, because refusing
    them is RE2's job and reading them is Python's, and the whole point of this
    front end is that the reading happens before the choice of engine.

    Returns:
        The list.
    """
    var out: List[String] = [
        "*",
        "+",
        "?",
        "*?",
        "+?",
        "??",
        "*+",
        "++",
        "?+",
        "{0}",
        "{2}",
        "{2,}",
        "{,2}",
        "{1,3}",
        "{1,3}?",
        "{1,3}+",
        "{3,1}",
        "{2}+",
        "{99999}",
    ]
    return out^


def _wrappers() -> List[String]:
    """The brackets that take something else inside them.

    Each is written with a `%` where the body goes, which is a stand in rather
    than a format string because there is exactly one hole and a real formatter
    would be more machinery than the thing it replaces. The conditional forms
    have two holes and are handled apart from these.

    Returns:
        The list.
    """
    var out: List[String] = [
        "(%)",
        "(?:%)",
        "(?i:%)",
        "(?-i:%)",
        "(?im:%)",
        "(?P<n>%)",
        "(?P<1n>%)",
        "(?P<>%)",
        "(?'n'%)",
        "(?=%)",
        "(?!%)",
        "(?<=%)",
        "(?<!%)",
        "(?>%)",
        "(?#%)",
        "(%",
        "%)",
    ]
    return out^


def _filled(shape: StringSlice, body: StringSlice) -> String:
    """Puts a body into a wrapper.

    Args:
        shape: The wrapper, holding one percent sign where the body goes.
        body: What to put there.

    Returns:
        The two joined, or the shape unchanged when it holds no percent sign.
    """
    var out = String()
    var seen = False
    for point in shape.codepoints():
        if not seen and point.to_u32() == UInt32(ord("%")):
            out += body
            seen = True
        else:
            out += String(point)
    return out^


def _grown(mut rng: _Rng, depth: Int) -> String:
    """One item, which at any depth above zero may hold more items.

    Args:
        rng: The generator.
        depth: How much further down it may go. At zero it returns an atom,
            which is what stops this.

    Returns:
        A fragment of a pattern.
    """
    var atoms = _atoms()
    if depth <= 0:
        return atoms[rng.below(len(atoms))]

    var roll = rng.below(100)
    if roll < 30:
        return atoms[rng.below(len(atoms))]
    if roll < 50:
        var wrappers = _wrappers()
        return _filled(
            wrappers[rng.below(len(wrappers))], _grown(rng, depth - 1)
        )
    if roll < 62:
        var quantifiers = _quantifiers()
        return _grown(rng, depth - 1) + quantifiers[rng.below(len(quantifiers))]
    if roll < 74:
        var out = String()
        var parts = 2 + rng.below(2)
        for at in range(parts):
            if at != 0:
                out += "|"
            out += _grown(rng, depth - 1)
        return out^
    if roll < 82:
        var out = String("(?(1)")
        out += _grown(rng, depth - 1)
        if rng.below(2) == 0:
            out += "|"
            out += _grown(rng, depth - 1)
        out += ")"
        return out^
    if roll < 88:
        var out = String("(?(n)")
        out += _grown(rng, depth - 1)
        out += ")"
        return out^
    if roll < 94:
        # A named group and a reference to it, which is the only way a
        # backreference by name is ever valid, and is therefore the only way to
        # reach the routing decision through one.
        var out = String("(?P<n>")
        out += _grown(rng, depth - 1)
        out += ")(?P=n)"
        return out^
    var out = String()
    var parts = 2 + rng.below(3)
    for _ in range(parts):
        out += _grown(rng, depth - 1)
    return out^


def _measured() -> List[String]:
    """The patterns the measurements in document 76 were taken on.

    Kept apart from the generated ones and always asked, because a generated
    corpus is a statement about the grammar as a whole and these are the
    individual facts the design rests on. A corpus that stops covering one of
    them after a seed changes would take a documented finding with it silently.

    Returns:
        The list.
    """
    var out: List[String] = [
        "a",
        "(?=a)",
        "(?=a)*",
        "(?=a)?",
        "(?:(?=a))*",
        "(?:(?=a))+",
        "(?:(?=a)){2}",
        "(?:(?=a))?",
        "(?:(?=a))*+",
        "(?>(?=a))",
        "(a)(?:\\1)*",
        "(a)(?:\\1)?",
        "[a](?:(?=x))*",
        "(?:(?!a))*",
        "(?:(?<=a))*",
        "((?=a))*",
        "(?:a|(?:(?=b))*)",
        "(a)(?(1)(?=b)|c)",
        "(?!)",
        "(?<!)",
        "(?=)",
        "(?<=)",
        "(?:)",
        "()",
        "(?#hi)a",
        "(?i)a",
        "(?i)(?s)a",
        "a(?i)b",
        "(?a)(?u)",
        "(?a)(?u)(?=a)",
        "(?#\\)",
        "(?#a\\\\)b",
        "\\N{GREEK SMALL LETTER ALPHA}",
        "[^a]",
        "[a]",
        "a|b",
        "a{}",
        "a{2",
        "\\p{L}",
        "(a)\\1",
        "\\1(a)",
        "(?P<n>a)(?P=n)",
        "(?P=n)",
        "(?i:(?=a))",
        "(?:ab)",
        "(ab)",
        "^*",
        "^+",
        "^?",
        "^{0}",
        "^{2}",
        "^{0,2}",
        "^*?",
        "$*",
        "$+",
        "**",
        "a**",
        "(?:a)*",
        "\\b*",
        "\\b+",
        "\\b{0}",
        "\\b{2}",
        "\\A*",
        "^*a",
        "a^*",
        "a\\b*b",
        "a^+b",
        "^**",
        "^*+",
        "(^)*",
        "\\Qa+b\\E",
        "\\Qa+b",
        "a\\Eb",
        "\\E",
        "\\Q\\E",
        "\\Q",
        "\\Qa\\E\\Qb\\E",
        "\\Q.\\E",
        "\\Q[a]\\E",
        "\\Q\\\\\\E",
        "\\Qa\\\\E",
        "\\Qab\\E*",
        "\\Qab\\E+",
        "(?i)\\Qa\\E",
        "[\\Qa\\E]",
        "\\Qa|b\\E",
        "\\Qa\\Eb",
        "\\Q(a)\\E",
        "\\QQ\\E",
        "\\QE\\E",
        "\\Q\\n\\E",
        "\\Q\\d\\E",
        "a\\Q\\Eb",
        "\\Q\\Q\\E",
        "\\Qa\\E{2}",
        "\\Q\\E*",
        "\\Q*\\E",
        "x\\Q\\E*",
        "\\Qa\\E\\E",
        "\\Q\\Ea",
        "(\\Qa)b\\E)",
        "\\Qa[b\\E",
        "\\Qa(b\\E",
        "(?:\\Qa)\\E)",
        "\\Qa\\E|b",
        "\\Q\\E\\Q\\E",
        "\\Q\\E?",
        "\\Q\\E{2}",
        "(\\Q\\E*)",
        "\\Q\\E\\Q\\E*",
        "a|\\Q\\E*",
    ]
    return out^


def corpus(cases: Int, seed: UInt64) -> List[String]:
    """The patterns to ask about.

    Args:
        cases: How many to generate on top of the measured ones.
        seed: Where the generator starts.

    Returns:
        The measured patterns first, then the generated ones.
    """
    var out = _measured()
    var rng = _Rng(seed)
    for _ in range(cases):
        out.append(_grown(rng, 1 + rng.below(3)))
    return out^


def report(title: StringSlice, patterns: List[String], total: Int):
    """Prints one kind of disagreement.

    Args:
        title: What the list is.
        patterns: The patterns in it.
        total: How many were compared, for the share.
    """
    if len(patterns) == 0:
        return
    print()
    print(len(patterns), "of", total, title)
    var shown = len(patterns) if len(patterns) < SHOWN else SHOWN
    for at in range(shown):
        print("   ", patterns[at])
    if len(patterns) > shown:
        print("   ", len(patterns) - shown, "more")


def names_a_character(pattern: StringSlice) -> Bool:
    """Whether a pattern holds a `\\N{...}` escape.

    The backslashes are counted rather than looked at one at a time, because
    `\\\\N{` is an escaped backslash followed by a plain letter and is not the
    escape at all.

    Args:
        pattern: The pattern.

    Returns:
        True when the parser will meet a named character escape.
    """
    var at = 0
    var bytes = pattern.as_bytes()
    while at < len(bytes):
        if bytes[at] != 0x5C:
            at += 1
            continue
        var run = 0
        while at < len(bytes) and bytes[at] == 0x5C:
            run += 1
            at += 1
        if run % 2 == 0:
            continue
        if at + 1 < len(bytes) and bytes[at] == 0x4E and bytes[at + 1] == 0x7B:
            return True
    return False
