# 76. Two engines and the parser that picks one

## 1. What this is and what it is not

Document 73 measured what pandas does with a regular expression and found two engines rather than one, chosen per call by handing the pattern to Python's own parser and walking what comes back. This document is the first piece of the thing that follows from that: the parser, and the routing decision it feeds.

There is no matching here. Nothing in this slice runs a pattern against a row, and none of the nine methods on the string accessor is wired to any of it. What exists is the front end both engines will share and the decision about which of them gets a given pattern, measured against pandas until the two agree on every pattern a generator could produce.

Everything below was measured against pandas 3.0.5 with pyarrow 24.0.0 on CPython 3.14.7, and the parts about how Python's parser behaves come from reading `re/_parser.py` rather than from reasoning about what it ought to do. That distinction earned its place: several of the rules in section 5 are ones nobody would arrive at by thinking about them.

## 2. Why Python's grammar, when most patterns are answered by RE2

The engine that answers the common case is RE2, and the grammar that decides is Python's. That is not a compromise, it is what pandas does, and it has a consequence worth stating on its own line: the set of patterns Python's parser refuses is part of the observable behaviour of a pandas program.

A pattern Python cannot read is not an error in pandas. `_has_unsupported_regex` catches the parse failure, answers False, and the pattern goes to Arrow. So `\p{L}`, which is RE2 syntax Python has never heard of, works in pandas today, and it works for the reason that it failed to parse. A parser here that read `\p{L}` out of helpfulness would route that pattern to the Python side and break it.

This is why the parser refuses things. Every refusal in `firepanda/kernel/regex/parse.mojo` is a pattern that must reach RE2, and a refusal that Python does not make sends a working pattern to an engine that will not run it.

## 3. The arena

The nodes live in one `List[Node]` and refer to each other by index, with a first child and a next sibling on each. Six small integers per node and no owned memory, so the whole tree moves and copies as one flat list.

The reason is a Mojo constraint met early. A node owning a `List` of children cannot be moved out of the list holding it without copying the subtree underneath, and the parser does exactly that move every time a quantifier takes the item in front of it and puts it under a repeat. An arena turns that into three integer writes.

It is also the shape the compiler behind this wants. A program counter is an index, so a walk that already holds indices does not have to invent them later.

## 4. Which of Python's collapses are reproduced

`re._parser` does not transcribe a pattern. It folds a class of one literal into that literal, drops a non capturing group by inlining its contents, and turns a negative lookaround with an empty body into a node that never matches.

The rule applied here is that a collapse is reproduced when it is visible from outside, and outside means two things and only two: whether the pattern parses, and where it routes.

Inlining a non capturing group is visible, because it lifts whatever was inside to a level the router walks. It is reproduced, though not in the same shape: Python ends up with a flat list and this ends up with a sequence node the walk steps through without counting. Different trees, same routing, which is the requirement.

Collapsing `(?!)` is visible and is the subject of section 6. Folding `[a]` into `a` is not visible, because no character class can hold a lookaround or a backreference, so it is not done, and the tree keeps the shape the caller wrote. The matching engine would rather have that shape, and nothing downstream of the router can tell.

This is also the honest statement of what the parser is checked against. It is not checked token for token against Python's tokens. It is checked on the two answers a caller can see, over a corpus large enough that agreeing on both by accident is not available.

## 5. The rules that had to be read rather than reasoned about

Each of these was a mistake in the first version of this parser, found by the generated corpus, and each is now a test in `tests/test_regex_parse.mojo`.

**A quantifier repeats the last item of the sequence, not the last thing the reader read.** `a(?#c)+` is `a+`. A comment produces nothing at all, so the plus reaches back past it. The first structure here read an item and then looked for a quantifier, which cannot express this, and the quantifier had to move into the sequence reader where the last child is known.

**A backslash inside a comment hides the character after it.** `(?#\)` is unterminated. Python reads a comment through the same tokenizer as everything else and that tokenizer hands back an escape as one token, so the closing bracket is never seen as one. Nobody wrote this rule. It falls out of the reader, and a caller writing a Windows path in a comment meets it.

**A backslash and some digits are an octal escape only under a rule with four parts.** Python takes a third digit only when the two it has are both octal and the next character is octal too, and prefers a group reference otherwise. The first version here read up to three digits and preferred a group reference when the number was small, which is close and not the same, and the corpus disagreed about it in a few hundred patterns.

**A backreference is checked where it is written and a conditional is checked at the end of the parse.** `\1(a)` is a parse error and `(?(1)a)(b)` is read. The asymmetry is real and it changes the routing of both patterns.

**A global flag group has to be at the start, and an empty alternative counts as something in front of it.** `(?i)a` reads, `a(?i)b` does not, and `|(?i)a` does not either. Python 3.11 made the position an error, which turned a set of patterns that used to be answered by Python into patterns answered by RE2.

**Flags can only be turned off in the scoped form.** `(?-i)` and `(?i-s)` are both parse errors, and neither is a global flag group with a minus sign in it. A caller writing either has written an RE2 pattern without meaning to.

**Braces that cannot be read as a count are characters.** `a{}` is three literals and `a{2` is three literals. Python does not refuse a brace it cannot parse, it puts it back, and a parser that refused here would move a pattern pandas answers with Python onto the Arrow side.

**A negative lookaround with an empty body stops being a lookaround.** `(?!)` becomes `FAILURE`, which is the whole of section 6's second finding.

## 6. What the router can and cannot see

pandas' walk looks for three op codes and recurses into two node kinds, a subpattern and a branch. There are seven node kinds that can hold another node underneath them. The five it does not enter are the three repeats, the atomic group and the conditional.

So a lookaround underneath any of those five is invisible to the decision:

```
s.str.contains(r"(?=a)")    a column of booleans, answered by Python
s.str.contains(r"(?=a)?")   ArrowInvalid: invalid perl operator: (?=
```

Making an assertion optional changes which engine runs. The caller gets an error naming a library they did not call, for a pattern that differs from a working one by a single character.

The second finding is the collapse. `(?!)` and `(?<!)` become `FAILURE` in `re._parser`, so there is no assertion left for the walk to find and both go to RE2, which refuses them. `(?=)` and `(?<=)` keep their nodes and go to Python, which answers them. One character apart, two engines, and the one that raises is the one whose body is empty.

Both are reproduced. Reproducing an upstream bug is a decision that has to be made by a person and written down rather than arrived at by accident, and the argument is that the alternative is worse in a way that is harder to find. A library that answers where pandas raises is a library whose divergence shows up only when somebody moves a program the other way, and it shows up then as a wrong answer rather than as an error. The refusal is reproduced in kind rather than in wording, since pandas' wording names Arrow.

Section 8 of document 73 recorded a third case of the same shape, `\p{L}(?=x)`, where the lookahead is invisible because the pattern never parses. That one is reproduced too and for the same reason, and it is worth noticing that all three findings are the same sentence: the switch is not "does the pattern contain a lookaround", it is "does Python's parser report a lookaround at a place this particular walk looks".

## 7. The corpus, and why it came before the tests

`tests/differential/regex.mojo` builds patterns out of the grammar rather than collecting them by hand: roughly a hundred atoms, three quarters of them malformed on purpose, nineteen quantifiers and seventeen wrappers, combined to a random depth. Both questions are asked of both front ends over thirty thousand of them, and the oracle calls pandas' own `_has_unsupported_regex` rather than a reimplementation of it, because a reimplementation of an oracle is a second copy of the thing under test wearing the same name.

The first run disagreed on eighty nine patterns out of two thousand and forty seven. Every one traced to a specific rule, and the seven distinct mistakes behind them are section 5. Not one of them would have been in a hand written test file, because a hand written test file is bounded by what somebody already knows is interesting, which is the wrong end of the problem: the routing mistakes that matter are the ones on patterns nobody would think to try, since those are the ones that reach a user before they reach a test.

The run is now ten thousand in ten thousand on thirty thousand patterns and on three seeds, with a ceiling of zero disagreements. There is no bounded disagreement worth carrying here, because a pattern routed to the wrong engine is not a refusal, it is an answer computed by the engine pandas would not have used.

Two kinds of pattern are counted and set aside rather than compared, and the counts are printed so that setting them aside is a number somebody watches rather than a silence. They are sections 8 and 9.

## 8. The one place where copying pandas was not available

```
s.str.contains(r"(?a)(?u)")
ValueError: ASCII and UNICODE flags are incompatible
```

Turning on both alphabets globally is reported at the end of the parse by `re._parser.fix_flags`, which raises `ValueError`. pandas catches `re.error` and nothing else, so the exception goes straight out through `str.contains` naming a module the caller never imported. There is no routing decision to copy, because pandas never reaches one.

This reads such a pattern as a pattern that did not parse, which sends it to Arrow, which is where every other failed parse goes. That is a divergence and it is chosen rather than accidental. The alternative is reproducing a crash, in kind, for a pattern that is merely contradictory rather than dangerous, and a library whose answer to a bad flag combination is an exception from the middle of the call is not a better library for being bug compatible about it.

Arrow's answer to `(?a)(?u)` is its own refusal, since RE2 has an inline flag group but no ASCII flag to put in one, so it reports `invalid perl operator: (?a`. The observable outcome is an error in both libraries, with a different sentence in it.

## 9. What is read approximately

`\N{GREEK SMALL LETTER ALPHA}` names a character rather than writing it, and resolving the name needs the Unicode name table. That table is not carried here.

What the parser does instead is read the braces and not what is between them. The routing answer is therefore right for every name that exists and wrong for every name that does not: Python refuses an unknown name and this accepts it. The node left behind holds a placeholder code point rather than the character, and the parse says so through `Parsed.approximate`, which exists so that the matching engine can refuse rather than leaving a pattern that quietly matches the replacement character.

This is worth the table eventually and is not worth it yet. The differential holds these patterns out of the agreement figure and prints how many it held out, which is the honest way to carry a known gap: a number that moves when the gap changes size.

## 10. A pattern whose routing depends on the interpreter

`\z` is a parse error on CPython 3.13 and an end of string anchor on CPython 3.14. firepanda supports 3.12 and up, so both interpreters are in range, and pandas therefore routes `\z` patterns differently on the two.

The visible difference needs a second ingredient. On 3.13 the pattern fails to parse and goes to Arrow, where `\z` happens to be valid RE2 syntax meaning the same thing, so the answer is the same. Put a lookaround next to it and the two part company: `\z(?=x)` is unreadable on 3.13 and goes to RE2, which refuses the lookahead, and is readable on 3.14 and goes to Python, which answers it.

This parser reads `\z`, which matches 3.14 and the version the differential runs on. A caller on 3.12 or 3.13 combining `\z` with a lookaround gets an answer here where pandas gives them an Arrow error. That is one pattern in a corner of a corner, and the alternative is a parser whose grammar depends on which interpreter is loaded, which is a far worse thing to own.

## 11. What is not here yet

The matching engines. Both of them, with two class tables and two end anchors so that the RE2 side and the Python side diverge in exactly the four places document 73 section 6 measured and nowhere else. Both have to be linear time, for the reason issue #158 gave originally, which does not weaken because the semantics being copied are Python's. The RE2 side is now document 77 and the Python side is still out.

The wiring. None of `contains`, `match`, `fullmatch`, `count` or `replace` consults any of this yet, and the gate in `python/firepanda/_pandas.py` that refuses any pattern holding one of twelve metacharacters is still the thing a caller meets.

Scoped flags. `(?i:a)` parses and the flags are dropped rather than carried on the node, because there is nothing yet that could act on them. The node it leaves is the same node `(?:a)` leaves, which is deliberate: a subpattern numbered zero would have been the obvious place to hang them and would also have been a lie, since zero is the whole match. What the parse does carry is which letters a scoped group mentioned, so that the compiler in document 77 can refuse the pattern rather than answer it as though they were never written.

The measurements document 73 section 10 listed are still not taken: the empty match rule in `replace` and `count` on each side, which of several equal length alternatives a capture group ends up holding, what `(?i)` does to a Unicode class in each, and whether RE2's leftmost first is leftmost first everywhere. Each is a place a compatibility layer can be quietly wrong, and each should be measured before the piece that depends on it is written.

## 12. Observations to file upstream

The incomplete walk in `_has_unsupported_regex`, which enters two of the seven nesting op codes, so that `(?=a)` is answered and `(?=a)?` raises.

The `(?!)` collapse to `FAILURE`, which routes a pattern to the engine that cannot run it, while `(?=)` on the same line routes to the one that can.

`(?a)(?u)` raising `ValueError` out of a `str.contains` call, since pandas catches only `re.error`.
