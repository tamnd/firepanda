"""Tests that a parsed pattern turns into the instructions it should.

The differential next door runs thirty thousand generated patterns against
sixteen pieces of text and compares the answers with pandas, and it is the
stronger test by a wide margin. This one exists because agreeing about an answer
is not the same as building the right program: a compiler can lay out a repeat
in a shape that happens to accept the same strings and costs twice as much, or
can get the preference order of a lazy quantifier backwards in a way no boolean
answer will ever show, and both are invisible until captures arrive.

So the assertions here are programs written out rather than verdicts. A change
that moves an instruction has to come here and say so.

The refusals get the same treatment, and for them the flag being asserted is the
one that says whose refusal it is. RE2 refuses a lookaround and firepanda
refuses `(?i)`, and the difference between those two is the whole reason the
caller above can tell a pattern it will never answer from a pattern it cannot
answer yet.
"""

from std.collections.span import Span
from std.testing import TestSuite, assert_equal, assert_false, assert_true

from firepanda.kernel.regex.parse import parse_pattern
from firepanda.kernel.regex.program import (
    IN_ANY,
    IN_ANY_ALL,
    IN_AT,
    IN_CHAR,
    IN_JUMP,
    IN_MATCH,
    IN_NOT_SET,
    IN_SET,
    IN_SPLIT,
    Program,
    class_of,
    compile_program,
    in_set,
    is_word_point,
    is_word_point_unicode,
    run_bodies,
    word_ranges_unicode,
)
from firepanda.kernel.regex.tokens import (
    AT_BOUNDARY,
    AT_BOUNDARY_UNICODE,
    AT_NON_BOUNDARY,
    AT_NON_BOUNDARY_UNICODE,
)
from firepanda.kernel.regex.route import ENGINE_PYTHON, ENGINE_RE2


def shown(value: Int32) -> String:
    """A code point, as a character when it is one somebody can read.

    Args:
        value: The code point.

    Returns:
        The character itself, or the number after a hash.
    """
    if value >= 0x21 and value < 0x7F:
        return String(chr(Int(value)))
    return String("#", value)


def ranges_of(program: Program, at: Int32, count: Int32) -> String:
    """One set's ranges, as text.

    Args:
        program: The program holding the range table.
        at: Where this set starts in it.
        count: How many ranges it has.

    Returns:
        The ranges, low to high, separated by spaces.
    """
    var out = String("")
    for i in range(Int(count)):
        if i != 0:
            out += " "
        var low = program.ranges[Int(at) + i * 2]
        var high = program.ranges[Int(at) + i * 2 + 1]
        out += String(shown(low), "-", shown(high))
    return out^


def drawn(program: Program) -> String:
    """A whole program, as one line.

    Each instruction is its position, its name and whatever it needs to say,
    separated by semicolons. Positions are written out rather than left implied
    because every split and jump names one and a listing that made the reader
    count lines would not be worth having.

    Args:
        program: The compiled pattern.

    Returns:
        The listing, or an exclamation mark and the reason there is not one.
    """
    if not program.ok:
        return String("!", program.problem)
    var out = String("")
    for at in range(len(program.code)):
        if at != 0:
            out += "; "
        var it = program.code[at]
        out += String(at, " ")
        if it.op == IN_CHAR:
            out += String("char(", shown(it.a), ")")
        elif it.op == IN_SET:
            out += String("set(", ranges_of(program, it.a, it.b), ")")
        elif it.op == IN_NOT_SET:
            out += String("notset(", ranges_of(program, it.a, it.b), ")")
        elif it.op == IN_ANY:
            out += "any"
        elif it.op == IN_ANY_ALL:
            out += "anyall"
        elif it.op == IN_SPLIT:
            out += String("split(", it.a, ",", it.b, ")")
        elif it.op == IN_JUMP:
            out += String("jump(", it.a, ")")
        elif it.op == IN_AT:
            out += String("at(", it.a, ")")
        elif it.op == IN_MATCH:
            out += "match"
        else:
            out += String("op", it.op)
    return out^


def built(pattern: StringSlice) -> String:
    """The program a pattern compiles to for RE2.

    Args:
        pattern: The pattern.

    Returns:
        The listing, or the refusal.
    """
    return drawn(compile_program(parse_pattern(pattern), ENGINE_RE2))


def runs_of(pattern: StringSlice) -> String:
    """Which splits of a pattern are a repeat of one character.

    Args:
        pattern: The pattern.

    Returns:
        Each split that qualifies as its own index, a colon and the index of its
        body, separated by spaces, or `none`.
    """
    var program = compile_program(parse_pattern(pattern), ENGINE_RE2)
    var runs = run_bodies(Span(program.code))
    var out = String("")
    for at in range(len(runs)):
        if runs[at] < 0:
            continue
        if out.byte_length() != 0:
            out += " "
        out += String(at, ":", runs[at])
    if out.byte_length() == 0:
        return String("none")
    return out^


def test_a_plain_pattern_is_one_instruction_a_character() raises:
    """The simplest program there is, which fixes the shape the rest are read
    against."""
    assert_equal(built("abc"), "0 char(a); 1 char(b); 2 char(c); 3 match")


def test_a_group_leaves_nothing_behind() raises:
    """Until there are captures, a group is only a place the parser put a
    child."""
    assert_equal(built("(ab)"), "0 char(a); 1 char(b); 2 match")
    assert_equal(built("(?:ab)"), "0 char(a); 1 char(b); 2 match")


def test_a_star_prefers_going_round_again_and_a_lazy_one_prefers_leaving() raises:
    """The two arms of the split are the same two instructions in both cases and
    the order they are written in is the whole difference, which is the kind of
    thing a boolean answer cannot see."""
    assert_equal(built("a*"), "0 split(1,3); 1 char(a); 2 jump(0); 3 match")
    assert_equal(built("a*?"), "0 split(3,1); 1 char(a); 2 jump(0); 3 match")


def test_a_plus_writes_the_body_once_before_the_loop() raises:
    """`a+` is `a` followed by `a*`, which is what a machine with no counter
    has."""
    assert_equal(
        built("a+"),
        "0 char(a); 1 split(2,4); 2 char(a); 3 jump(1); 4 match",
    )


def test_a_repeat_of_one_character_is_recognised_from_the_program() raises:
    """The backtracker walks these in one step instead of three, and what it
    reads is the shape of the instructions rather than anything written down
    while they were being built.

    A lazy repeat is the same three instructions with the arms of the split the
    other way round, so it qualifies as well and the arm that is the body is
    what says which of the two it is. A body of more than one instruction does
    not qualify, because then there is nothing to collapse.
    """
    assert_equal(runs_of("a*"), "0:1")
    assert_equal(runs_of("a*?"), "0:1")
    assert_equal(runs_of("a+"), "1:2")
    assert_equal(runs_of("[^/]+"), "1:2")
    assert_equal(runs_of(".*"), "0:1")
    assert_equal(runs_of("a{2,}"), "2:3")
    assert_equal(runs_of("(?:ab)*"), "none")
    assert_equal(runs_of("a?"), "none")
    assert_equal(runs_of("abc"), "none")


def test_a_count_is_a_number_of_copies() raises:
    """There is no counter in a Thompson program, so three means three."""
    assert_equal(built("a{3}"), "0 char(a); 1 char(a); 2 char(a); 3 match")


def test_the_optional_copies_of_a_count_are_nested_and_not_side_by_side() raises:
    """`(?:ab){0,2}` has to be `(?:ab(?:ab)?)?` rather than `(?:ab)?(?:ab)?`.

    The second one matches a gap in the middle, so it accepts `abab` and also
    accepts nothing twice over in a way the first does not. For a body of one
    character the two are the same set of strings, which is exactly why this is
    written down against a body of two.
    """
    assert_equal(
        built("(?:ab){0,2}"),
        (
            "0 split(1,6); 1 char(a); 2 char(b); 3 split(4,6); 4 char(a);"
            " 5 char(b); 6 match"
        ),
    )


def test_a_count_of_zero_writes_nothing_at_all() raises:
    """Which is why the refusals are a separate walk over the tree."""
    assert_equal(built("a{0}"), "0 match")


def test_an_alternation_tries_its_arms_in_the_order_they_were_written() raises:
    """Both engines do, and it matters for `a|ab` even though it does not for
    `ab|a`."""
    assert_equal(
        built("a|b"),
        "0 split(1,3); 1 char(a); 2 jump(4); 3 char(b); 4 match",
    )


def test_a_full_stop_reads_the_flags() raises:
    """The one thing a global flag changes in this compiler today."""
    assert_equal(built("."), "0 any; 1 match")
    assert_equal(built("(?s)."), "0 anyall; 1 match")


def test_a_class_is_sorted_and_its_touching_ranges_are_merged() raises:
    """Three separate letters become one range, which is what makes membership
    a binary search worth doing."""
    assert_equal(built("[cab]"), "0 set(a-c); 1 match")
    assert_equal(built("[a-cx]"), "0 set(a-c x-x); 1 match")


def test_a_negated_class_reads_the_same_ranges_the_other_way() raises:
    """Rather than storing the complement, which would be hundreds of ranges for
    a class of one letter."""
    assert_equal(built("[^a]"), "0 notset(a-a); 1 match")
    assert_equal(built("[^a-c]"), "0 notset(a-c); 1 match")


def test_the_perl_classes_are_the_ascii_ones_re2_uses() raises:
    """The single largest difference between the two engines, as a table.

    Python's `\\s` also holds a vertical tab, and Python's `\\d` holds every
    Unicode decimal digit there is, so a column of Arabic Indic digits answers
    False here and True once the same pattern picks up a lookahead and changes
    engines.
    """
    assert_equal(built("\\d"), "0 set(0-9); 1 match")
    assert_equal(built("\\D"), "0 notset(0-9); 1 match")
    assert_equal(built("\\s"), "0 set(#9-#10 #12-#13 #32-#32); 1 match")
    assert_equal(built("\\w"), "0 set(0-9 A-Z _-_ a-z); 1 match")


def test_an_anchor_is_a_position_rather_than_a_character() raises:
    """And the multiline flag picks a different one for the same character."""
    assert_equal(built("^a$"), "0 at(1); 1 char(a); 2 at(4); 3 match")
    assert_equal(built("(?m)^a$"), "0 at(2); 1 char(a); 2 at(5); 3 match")


def anchored(pattern: StringSlice, captures: Bool = False) -> Bool:
    """Whether a pattern compiles to a program that can only match at the start.

    Args:
        pattern: The pattern.
        captures: Whether to compile it with the save instructions, which is the
            case the walk has to step over.

    Returns:
        What the compiler wrote on the program.
    """
    return compile_program(
        parse_pattern(pattern), ENGINE_RE2, captures
    ).anchored


def test_a_pattern_that_opens_with_a_start_anchor_says_so() raises:
    """The flag the two scans read to stop starting attempts they know will
    fail. `\\A` is the same promise as `^` here because the multiline flag is
    the only thing that separates them and the compiler has already spent it."""
    assert_true(anchored("^abc"))
    assert_true(anchored("\\Aabc"))
    assert_true(anchored("^https?://"))
    assert_true(anchored("^abc", captures=True))
    assert_true(anchored("^(a)(b)", captures=True))
    assert_true(anchored("(^a)"))


def test_a_pattern_that_can_match_further_along_says_nothing() raises:
    """Including the two that look anchored and are not. Under multiline `^` is
    a different position code and matches after every newline, and an
    alternation compiles to a split whether or not both of its arms are
    anchored, which this reads as unanchored rather than walking into the
    arms."""
    assert_false(anchored("abc"))
    assert_false(anchored("(?m)^abc"))
    assert_false(anchored("^a|^b"))
    assert_false(anchored("a^b"))
    assert_false(anchored("a*^b"))


def begins(pattern: StringSlice, point: Int, captures: Bool = False) -> Bool:
    """Whether a character is one the pattern says a match can begin with.

    Args:
        pattern: The pattern.
        point: The character.
        captures: Whether to compile with the save instructions, which is the
            case the walk has to step over.

    Returns:
        What the set on the program says, which is False when there is no set.
    """
    var program = compile_program(parse_pattern(pattern), ENGINE_RE2, captures)
    return in_set(
        Span(program.ranges),
        program.first_at,
        program.first_count,
        UInt32(point),
    )


def firsts(pattern: StringSlice) -> Int:
    """How many ranges the set of first characters has.

    Args:
        pattern: The pattern.

    Returns:
        The count, which is zero when the compiler decided against a set.
    """
    return Int(compile_program(parse_pattern(pattern), ENGINE_RE2).first_count)


def test_a_pattern_says_which_characters_can_begin_a_match() raises:
    """The set the scans read to step over a position without walking the
    program at it. An alternation contributes both arms, a star contributes the
    letter it repeats and the one after it, and an assertion is walked through
    rather than reasoned about."""
    assert_true(begins("abc", ord("a")))
    assert_false(begins("abc", ord("b")))
    assert_true(begins("a|b", ord("a")))
    assert_true(begins("a|b", ord("b")))
    assert_false(begins("a|b", ord("c")))
    assert_true(begins("a*b", ord("a")))
    assert_true(begins("a*b", ord("b")))
    assert_true(begins("[0-9]x", ord("5")))
    assert_false(begins("[0-9]x", ord("x")))
    assert_true(begins("\\bfoo", ord("f")))
    assert_false(begins("\\bfoo", ord("o")))
    assert_true(begins("(abc)", ord("a"), captures=True))
    assert_false(begins("(abc)", ord("b"), captures=True))
    assert_equal(firsts("abc"), 1)
    # Two letters next to each other come out as one range, because the set is
    # sorted and merged the way every other set in the table is.
    assert_equal(firsts("a|b"), 1)
    assert_equal(firsts("a|c"), 2)


def test_a_pattern_that_could_begin_anywhere_is_left_without_a_set() raises:
    """Four reasons not to have one. A pattern that can match nothing begins a
    match at every position, so there is nothing to step over. A pattern opening
    with `(?s).` accepts every character. A set holding nearly all of ASCII
    would cost a search at every position and reject almost none of them, which
    is the full stop and the negated class of one character. And an anchored
    pattern starts no attempt above position zero, so it has no position to step
    over and is left out to keep its scan the one it was."""
    assert_equal(firsts("a*"), 0)
    assert_equal(firsts(""), 0)
    assert_equal(firsts("^https?://"), 0)
    assert_equal(firsts("(?s).x"), 0)
    assert_equal(firsts(".x"), 0)
    assert_equal(firsts("[^/]x"), 0)
    assert_equal(firsts("\\b"), 0)


def alphabet(pattern: StringSlice, python: Bool = False) -> Program:
    """A program compiled with its class table on it.

    Args:
        pattern: The pattern.
        python: Whether to compile it for Python's engine, which is the one
            whose word boundary reads the wide class.

    Returns:
        The program.
    """
    var engine = ENGINE_PYTHON if python else ENGINE_RE2
    return compile_program(parse_pattern(pattern), engine, alphabet=True)


def tells_apart(program: Program, one: UInt32, two: UInt32) -> Bool:
    """Whether some instruction of the program answers differently for two
    characters.

    Every instruction that reads a character is asked about both, and so is
    every word boundary, since a boundary reads the characters around a position
    and a class table that ignored it would put a letter and a space together.

    Args:
        program: The program.
        one: The first character.
        two: The second.

    Returns:
        True when the two are different characters as far as this program is
        concerned.
    """
    var wide = word_ranges_unicode()
    for i in range(len(program.code)):
        var instruction = program.code[i]
        var first = False
        var second = False
        if instruction.op == IN_CHAR:
            first = Int32(Int(one)) == instruction.a
            second = Int32(Int(two)) == instruction.a
        elif instruction.op == IN_SET or instruction.op == IN_NOT_SET:
            first = in_set(
                Span(program.ranges), instruction.a, instruction.b, one
            )
            second = in_set(
                Span(program.ranges), instruction.a, instruction.b, two
            )
        elif instruction.op == IN_ANY:
            first = one != 0x0A
            second = two != 0x0A
        elif instruction.op == IN_AT:
            var which = UInt8(Int(instruction.a))
            if which == AT_BOUNDARY or which == AT_NON_BOUNDARY:
                first = is_word_point(one)
                second = is_word_point(two)
            elif (
                which == AT_BOUNDARY_UNICODE or which == AT_NON_BOUNDARY_UNICODE
            ):
                first = is_word_point_unicode(one, Span(wide))
                second = is_word_point_unicode(two, Span(wide))
        if first != second:
            return True
    return False


def test_the_alphabet_holds_one_class_per_set_of_characters_told_apart() raises:
    """What the table is for: `abc` has four classes rather than a million, and
    the characters below `a` and the ones above `c` are one of them, because
    nothing in that program can tell them apart."""
    var three = alphabet("abc")
    assert_equal(three.class_count, 4)
    assert_equal(class_of(three, UInt32(ord("z"))), class_of(three, UInt32(0)))
    assert_equal(
        class_of(three, UInt32(ord("z"))), class_of(three, UInt32(0x3042))
    )
    assert_true(
        class_of(three, UInt32(ord("a"))) != class_of(three, UInt32(ord("b")))
    )

    var letters = alphabet("[a-z]+")
    assert_equal(letters.class_count, 2)
    assert_equal(
        class_of(letters, UInt32(ord("a"))),
        class_of(letters, UInt32(ord("z"))),
    )

    # A full stop reads every character and tells one of them from the rest.
    var stop = alphabet(".")
    assert_equal(stop.class_count, 2)
    assert_true(
        class_of(stop, UInt32(0x0A)) != class_of(stop, UInt32(ord("a")))
    )

    # And under `(?s)` it tells nothing from anything, which is one class over
    # every code point there is.
    var all = alphabet("(?s).")
    assert_equal(all.class_count, 1)
    assert_equal(class_of(all, UInt32(0x0A)), class_of(all, UInt32(0x3042)))


def test_a_word_boundary_cuts_the_alphabet_the_way_its_own_class_does() raises:
    """A boundary reads the characters around a position rather than the one at
    it, so a table built from the character instructions alone would put a
    letter and a space in one class and answer `\\bfoo` wrongly."""
    var narrow = alphabet("\\bfoo")
    assert_true(
        class_of(narrow, UInt32(ord("a"))) != class_of(narrow, UInt32(ord(" ")))
    )
    # RE2's boundary is ASCII, so a Greek letter is on the space's side of it.
    assert_equal(
        class_of(narrow, UInt32(ord(" "))), class_of(narrow, UInt32(0x03B1))
    )

    # Python's is not, so the same pattern on the other engine puts the Greek
    # letter with the Latin one.
    var wide = alphabet("\\bfoo", python=True)
    assert_equal(
        class_of(wide, UInt32(ord("a"))), class_of(wide, UInt32(0x03B1))
    )
    assert_true(
        class_of(wide, UInt32(0x03B1)) != class_of(wide, UInt32(ord(" ")))
    )


def test_two_characters_in_one_class_are_one_character_to_the_program() raises:
    """The property the whole table rests on, asked of every pair of characters
    a handful of patterns can reach. Two in the same class answer every
    instruction alike, and two in different classes are told apart by at least
    one of them, which is what makes the count as small as it can be rather than
    merely small."""
    var patterns = List[String]()
    patterns.append(String("abc"))
    patterns.append(String("[a-z]+[0-9]*"))
    patterns.append(String("^https?://(?:www\\.)?([^/]+)/.*$"))
    patterns.append(String("\\bcat\\b|dog"))
    patterns.append(String("(?s).x"))
    var points = List[UInt32]()
    for point in range(128):
        points.append(UInt32(point))
    points.append(UInt32(0x00E9))
    points.append(UInt32(0x03B1))
    points.append(UInt32(0x3042))
    points.append(UInt32(0x10FFFF))
    for i in range(len(patterns)):
        var program = alphabet(patterns[i])
        for one in range(len(points)):
            for two in range(one + 1, len(points)):
                var apart = tells_apart(program, points[one], points[two])
                var same = class_of(program, points[one]) == class_of(
                    program, points[two]
                )
                assert_equal(apart, not same)


def test_a_program_is_compiled_without_a_class_table_unless_it_is_asked_for() raises:
    """Nothing reads the table yet and it costs several times what compiling a
    short pattern costs, so a caller has to ask. The ceiling on the length of a
    program is the other way a caller does not get one."""
    var plain = compile_program(parse_pattern("abc"), ENGINE_RE2)
    assert_equal(plain.class_count, 0)
    assert_equal(len(plain.class_ascii), 0)
    assert_equal(class_of(plain, UInt32(ord("a"))), 0)

    var long = String("(?:abcdefghijklmnopqrstuvwxyz){1000}")
    var big = alphabet(long)
    assert_true(big.ok)
    assert_true(big.sized() > 20000)
    assert_equal(big.class_count, 0)


def test_what_re2_refuses_is_not_counted_as_a_gap() raises:
    """Six constructs Python's grammar reads and RE2 has never had. A caller
    writing one of these gets an Arrow error out of pandas today, so refusing it
    here is agreement rather than a shortfall."""
    var refusals: List[String] = [
        "(?=a)",
        "(?<=a)",
        "(a)\\1",
        "(a)(?(1)b)",
        "(?>a)",
        "a*+",
    ]
    for pattern in refusals:
        var program = compile_program(parse_pattern(pattern), ENGINE_RE2)
        assert_false(program.ok)
        assert_false(program.gap)


def test_a_backreference_under_a_count_of_zero_is_still_refused() raises:
    """The pattern that made the refusals a separate walk. It compiles to
    nothing at all, so a compiler that only looked at what it emitted would
    answer where RE2 raises."""
    var program = compile_program(
        parse_pattern("(?P<n>a)(?P=n){0}"), ENGINE_RE2
    )
    assert_false(program.ok)
    assert_false(program.gap)
    assert_equal(program.problem, "RE2 has no backreference")


def test_re2_will_not_repeat_more_than_a_thousand_times() raises:
    """And the limit counts the whole way down, so two numbers neither of which
    is over it can still be over it together."""
    assert_equal(built("a{1000}").find("!"), -1)
    assert_equal(built("a{1001}"), "!RE2 will not repeat that many times")
    assert_equal(built("(?:a{10}){100}").find("!"), -1)
    assert_equal(built("(?:a{11}){91}"), "!RE2 will not repeat that many times")
    assert_false(
        compile_program(parse_pattern("a{1001}"), ENGINE_RE2).gap,
    )


def test_a_repeat_with_no_ceiling_spends_its_lower_bound() raises:
    """Which is why a star nested a thousand deep is fine and `a{1000,}` twice
    over is not."""
    assert_equal(built("(?:a*){1000}").find("!"), -1)
    assert_equal(
        built("(?:a{1000,}){2}"), "!RE2 will not repeat that many times"
    )


def test_the_four_inline_flag_letters_re2_has_never_heard_of() raises:
    """Refused in either form and whether they are turned on or off, so
    `(?-x:a)` is an error exactly as `(?x)a` is."""
    assert_equal(built("(?x)a"), "!invalid perl operator: (?x")
    assert_equal(built("(?u)a"), "!invalid perl operator: (?u")
    assert_equal(built("(?-x:a)"), "!invalid perl operator: (?x")
    assert_false(compile_program(parse_pattern("(?x)a"), ENGINE_RE2).gap)


def test_case_folding_is_spent_here_rather_than_refused() raises:
    """This used to be the one flag letter RE2 has that this compiler refused,
    on the grounds that no folding table had been written. Document 83 wrote
    one, so the flag compiles now and the set it built is what
    `test_regex_fold.mojo` reads. What is left here is the regression guard:
    the flag is no longer a refusal and no longer a gap."""
    var program = compile_program(parse_pattern("(?i)a"), ENGINE_RE2)
    assert_true(program.ok)
    assert_false(program.gap)
    assert_equal(program.problem, "")


def test_a_scoped_flag_group_carries_its_letters_on_a_node() raises:
    """The letters used to be thrown away, and a program built from that tree
    answered as though they were never written. The listing is where that shows
    with nothing running: the dollar sign inside the bracket is the end of a
    line and the one after it is the end of the text, from the same pattern."""
    assert_equal(built("(?m:a$)"), "0 char(a); 1 at(5); 2 match")
    assert_equal(built("a$"), "0 char(a); 1 at(4); 2 match")
    assert_equal(built("(?m:a$)$"), "0 char(a); 1 at(5); 2 at(4); 3 match")


def test_what_re2_reads_differently_is_refused_rather_than_answered() raises:
    """None of these three raises anywhere, which is what makes them dangerous:
    an answer out of this tree would be a column of booleans that looks exactly
    like a right one."""
    assert_equal(built("a{,2}"), "!RE2 reads this syntax differently")
    assert_equal(built("[[:alpha:]]"), "!RE2 reads this syntax differently")
    assert_equal(built("\\B"), "!RE2 reads a non boundary between bytes")
    assert_true(compile_program(parse_pattern("a{,2}"), ENGINE_RE2).gap)


def test_the_syntax_re2_refuses_that_looks_like_nothing() raises:
    """A backslash in front of a character outside ASCII is an ordinary way to
    write that character to Python and an error to RE2, and a one digit octal
    escape in a class is the character with that code to Python and an error to
    RE2, which will not read a nonzero octal escape shorter than two digits."""
    assert_equal(built("\\é"), "!RE2 has no such syntax")
    assert_equal(built("[a\\1b]"), "!RE2 has no such syntax")
    assert_false(compile_program(parse_pattern("\\é"), ENGINE_RE2).gap)
    assert_equal(built("[a\\01b]"), "0 set(#1-#1 a-b); 1 match")


def test_a_pattern_python_cannot_read_is_a_gap_and_not_a_refusal() raises:
    """Those are the patterns pandas answers with RE2 precisely because Python
    refused them, and reaching them needs a second front end."""
    var program = compile_program(parse_pattern("\\p{L}"), ENGINE_RE2)
    assert_false(program.ok)
    assert_true(program.gap)
    assert_equal(program.problem, "Python's grammar cannot read this pattern")


def test_the_two_engines_refuse_the_same_pattern_in_two_voices() raises:
    """A conditional group is refused on both sides, but the flag says
    something different on each: for RE2 the refusal agrees with upstream and
    for Python it is a shortfall here.

    A lookaround was this row's pattern until documents 93 and 94 answered both
    halves of it on Python's engine, a backreference was until document 95 did
    and an atomic group was until document 99 did, each of which left RE2 the
    only side refusing it and so left nothing for the row to compare. The
    construct here is the one that is left."""
    var theirs = compile_program(parse_pattern("(a)(?(1)b|c)"), ENGINE_RE2)
    assert_false(theirs.ok)
    assert_false(theirs.gap)
    assert_equal(theirs.problem, "RE2 has no conditional group")
    var ours = compile_program(parse_pattern("(a)(?(1)b|c)"), ENGINE_PYTHON)
    assert_false(ours.ok)
    assert_true(ours.gap)
    assert_equal(ours.problem, "this engine has no conditional group yet")


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
