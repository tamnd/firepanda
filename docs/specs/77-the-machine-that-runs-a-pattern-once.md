# 77. The machine that runs a pattern once

## 1. What this is and what it is not

Document 76 built the front end both engines share and the decision about which of them gets a given pattern. This document is the next piece: the compiler that turns one of those parsed patterns into instructions, and the machine that runs them against text. It is the RE2 side only, which is the side that answers the common case.

What is here answers one question, which is whether a pattern matches somewhere in a piece of text. There are no capture groups, no replacement, no counting of matches, and none of the nine methods on the string accessor is wired to any of it. The reason for stopping there is in section 7 and it is about what can be checked rather than about what was easy.

Everything below was measured against pandas 3.0.5 with pyarrow 24.0.0 on CPython 3.14.7. The facts about RE2 in section 4 were measured the same way, by asking pyarrow through pandas and reading what came back, rather than read out of RE2's documentation, and section 4 has two places where the documentation and the behaviour are not the same thing.

## 2. Why Thompson, and why the answer is one bit

The program is a list of instructions with two branching ones and no backtracking, and every place the pattern could be after reading the same number of characters is held at once. So the text is read once and never returned to, and the work is the length of the text times the size of the program however badly the pattern is written.

This is the same choice RE2 made and it is made here for the same reason. A pattern arrives from a caller and is run against a column, so an engine that can take exponential time on `(a+)+b` is a denial of service with a friendly API in front of it. A backtracking engine would also be easier to write and would pass every test in this repository, which is what makes the choice worth writing down rather than leaving to be inferred.

There are nine instructions. One matches a code point, two match a set of ranges and its complement, two match anything and anything but a newline, two branch and jump, one checks a position rather than a character, and one says the pattern has matched. Every set in the program lives in one range table laid end to end, so a compiled pattern is two allocations however many character classes it holds, and membership is a binary search rather than a walk because a Unicode class table will have hundreds of ranges in it and the classes a caller writes have two.

The whole of the correctness argument for the machine is two details about one array. An instruction is added to a list at most once per position in the text, which is what keeps a list shorter than the program and is also what makes a repeat with a body that can match nothing terminate: `(a*)*` goes round its outer loop, arrives back at an instruction it has already added at this position, and stops. Without that, it is an infinite loop rather than a slow one. The array holds the position rather than a round number, and a thread queued for the next character is stamped with the next position, so when the search arrives there and tries to start a fresh attempt the instructions already queued are recognised. One array, two lists, and nothing to clear between positions.

The search is unanchored, so a fresh attempt is started at every position of the row, and there is one pattern shape where every one of those after the first is known to fail before it is started. A program whose first instruction is `^` outside multiline mode, or `\A`, asks whether the position is zero and dies there, so the attempt at position seven walks from instruction zero to the anchor and stops, once per character of every row. The compiler now says so on the program, because it is the only part of this that has the flags and the first instruction in the same place, and the two scans in `firepanda/kernel/regex/pike.mojo` read the flag and skip those attempts. A row that does not match then ends at the second position rather than at the last, which is the whole of what the flag buys and is worth having because ClickBench q28 reads a column of URLs with an anchored pattern. A pattern that begins with an alternation compiles to a split first and is left alone, even when both of its arms are anchored, since the flag is read as a promise about the program rather than about one path through it.

## 3. Two kinds of refusal

A compiled pattern is a value and so is a refusal, and a refusal carries a flag saying whose it is. Either RE2 refuses this too, in which case refusing is agreement and a caller gets the same Arrow error out of pandas today, or firepanda cannot do it yet, in which case the pattern is a gap.

This distinction is the one piece of design in this slice worth copying. Every decision about whether a pattern can be answered now lives in the compiler, where the facts about RE2 already are, rather than in the harness or the accessor as a heuristic over the pattern text. A differential that held out every pattern with `(?i)` in it would also hold out the ones where `(?i)` failed to parse and the pattern was therefore an ordinary RE2 pattern, and it would have to be kept in step with the compiler by hand forever.

The refusal is a walk over the whole tree rather than something noticed while instructions are being written, and that too came out of the corpus. `(?P<n>a)(?P=n){0}` holds a backreference and compiles to nothing at all, because a repeat with a count of zero writes its body zero times, so a compiler that only looked at what it emitted would answer a pattern RE2 rejects. RE2 parses the whole pattern before running any of it, and so does this.

## 4. What RE2 refuses, measured

Six of these were known from document 73 and the rest are new here, and every one was measured by writing the pattern and reading what pandas did with it.

A lookaround, a backreference, a conditional group, an atomic group and a possessive quantifier are the five constructs Python's grammar reads and RE2 has never had. Four of the seven inline flag letters are the sixth, and RE2 refuses `L`, `x`, `a` and `u` with the same complaint whether they are turned on or off and whether they are global or scoped, so `(?-x:a)` is an error exactly as `(?x)a` is. Asking for `(?a)`, which is a request for the classes RE2 already uses, is among them.

A comment group is refused, which is a fact the tree cannot carry because a comment leaves nothing behind, so the parser records it. So are `\uXXXX` and `\UXXXXXXXX`, which RE2 writes as `\x{41}` instead. So is `\b` inside a character class, which is a backspace to Python and a word boundary RE2 will not have inside brackets. So is a `\Z` that is not the last thing in the pattern, because pandas rewrites a trailing one into RE2's `\z` on the way past and leaves one anywhere else alone.

Two more were found by running patterns rather than by reading about them, and both look like nothing at all. A backslash in front of a character outside ASCII is refused, so `\漢` is an ordinary way to write a character to Python and an error to RE2, which recognises no escape outside ASCII and refuses every backslash it does not recognise. A one digit octal escape inside a class is refused, so `[\1]` is the character with code one to Python and an error to RE2, while `[\01]` and `[\12]` are the same character to both. RE2 will not read a nonzero octal escape shorter than two digits because that is how it tells one from a backreference it does not have.

The last one is a budget rather than a construct. RE2 will not repeat anything more than a thousand times, and the limit counts the whole way down: `a{1001}` is an error, and so is `(a{11}){91}`, which is 1001 copies written as two numbers neither of which is over the limit. So it is a budget that divides on the way into a repeat rather than a check on one number, the children of a sequence each get the whole of what their parent had, and a repeat with no ceiling spends its lower bound, which is why `(a*){1000}` is a pattern and `(a{1000,}){2}` is not. Python has no limit at all, so this is a refusal rather than a gap and a caller writing `a{5000}` gets an Arrow error out of pandas today.

## 5. The tables, and what a class means

The six Perl classes are ASCII to RE2 and Unicode to Python, which is the difference document 76 opens with, and it is why a column of Arabic Indic digits answers False to `str.contains(r"\d")` and True once the same pattern picks up a lookahead and moves engines.

The tables here are the measured sets rather than the documented ones. RE2 documents `\s` as tab, newline, form feed, carriage return and space, and the measurement agrees with the documentation, which is worth saying because Python's `\s` also holds a vertical tab and the documentation is the only place where the two look alike. `\w` is the digits, the two alphabets and the underscore, and nothing else.

A negated class is not stored as its complement. It is an instruction that reads the same ranges and answers the other way, because a negated class is common enough that storing the complement would double the size of the range table for no gain. A class that mixes a negation with anything else is complemented properly against the whole code point space, since there is no instruction for that shape.

The full stop excludes a newline, in both engines, and the only character either of them calls a line ending is the newline. Neither counts a carriage return or any of the Unicode line separators, which is agreement rather than accident because both say so.

`$` is the one anchor where the two part company. Python's matches at the end of the text and also just before a newline that ends it, so `re.search("a$", "a\n")` finds something and the same pattern through Arrow does not. This engine takes the RE2 reading, and Python's will be a different table for the same reason the classes are.

## 6. What is different rather than refused

Three constructs are read by both engines and read differently, and none of them raises anywhere. A pattern like this answered out of this tree gives a column of booleans that looks exactly like a right one, which is why they are refused here as gaps rather than answered.

A count with no lower bound is `a{0,2}` to Python and the five literal characters `a{,2}` to RE2. A POSIX class is a set holding a bracket, a colon and some letters to Python and every letter there is to RE2. Both are recorded by the parser, because by the time the tree exists the difference is invisible.

The third is the interesting one. RE2 asks the word boundary question between bytes rather than between characters, so `\B` matches in the middle of any character that takes more than one byte to write, and `Series(["é"], dtype="str").str.contains(r"\B")` is True on the Arrow backend and False on the Python one for the same data. `\b` is unaffected, since a byte in the middle of a character is not a word byte on either side and a boundary needs one. Reproducing it means running this machine over bytes rather than over code points, which is a larger change than this one, so `\B` is a gap with its own line in the report.

## 7. The corpus, and the three families the first run found

The differential was written before the hand written tests, for the same reason document 76 section 7 gives, and it found the same kind of thing again. It takes the corpus document 76 built, which is thirty thousand patterns generated out of the grammar, runs each of them against sixteen pieces of text through this engine and through pandas' own `str.contains`, and compares both the refusal and the sixteen answers. The texts are chosen so that each measured difference between the engines has something to bite on: Arabic Indic digits for `\d`, a vertical tab and a non breaking space for `\s`, and three texts with newlines in them for `$`.

The first run compared 8054 patterns and disagreed about 66 of them, in three families, and every one was a fact about RE2 that had not been measured.

The first family was five constructs RE2 refuses and nobody had checked: the comment group, the two named character escapes, `\b` in a class and the non trailing `\Z`. The second was the pair under `{0}`, which is section 3's reason for a separate walk. The third was `{,n}` and the POSIX class, which are section 6's first two.

A second run found the scoped flag group, which had been parsed and thrown away since document 76 and did not matter until there was an engine to act on it. A third found the two in section 4 that look like nothing, the backslash before a character outside ASCII and the one digit octal escape, both of which arrived as a single pattern in a report of two. A fourth found the repeat budget, which had been mistaken for a size limit on the program and is not one.

It now agrees with pandas on every pattern it compares, across five seeds of thirty thousand patterns each, which is about thirty eight thousand patterns compared and six hundred thousand answers. The ceiling is zero and it is zero for a reason worth restating: a wrong answer here is not a refusal a caller can see, it is a column of booleans that looks exactly like a right one.

The held out patterns are counted by reason in the report rather than passed over in silence, so that setting a pattern aside is a number somebody watches. The largest number in that report is the subject of the next section.

## 8. What is not here yet

Nineteen thousand six hundred of the thirty thousand generated patterns are held out because Python's grammar cannot read them, and that single number is the largest gap in the component. Those are the patterns pandas answers with RE2 precisely because Python refused them, which is document 76 section 2, and reaching them needs a second front end that reads RE2's own grammar. `\p{L}` is the one to think about: it works in pandas today and it is unreachable here.

Captures are not in the machine. The instruction for saving a position and the per thread slots it needs are a known shape and were left out on purpose, because the oracle for a boolean answer is `str.contains` and the oracle for a capture is a harder thing to write, and adding the feature before the check that would catch it being wrong is the habit this component is trying not to have. `findall`, `extract` and `extractall` wait on it.

Case folding is a table that does not exist here, so `(?i)` is refused rather than ignored. Ignoring it would be a wrong answer rather than a missing feature.

Scoped flags are parsed and dropped, so a pattern writing `(?m:a$)` is refused as a gap. What closes it is carrying the flags on the node the group leaves behind, which the parser deliberately does not do yet because the node it would hang them on would have to be invented rather than borrowed from Python's tokens.

The Python engine is not written. It needs the Unicode class tables, its own reading of `$`, and the three constructs RE2 does not have, and the router already sends it about five percent of the corpus.

Nothing is wired to the accessor. `contains`, `match`, `fullmatch`, `count` and `replace` still refuse every metacharacter through the literal path in `python/firepanda/_pandas.py`, and wiring them to the router is the next slice.

## 9. Observations to file upstream

RE2's `\B` is byte based, so pandas gives different answers for the same data on its two string backends, and the difference is invisible until the column holds a character outside ASCII. This one is worth filing because it is not a documented divergence between the backends and it is not the kind a caller would think to check.

The rest of the list is in document 76 section 12 and is unchanged by this slice.

## 10. What a character costs, and what a lazy DFA would buy

This section was written after issue #830 was closed and before any of issue #863 was written, because #863's first question is whether the machine should move to bytes and the honest answer to that turned out to be a measurement rather than an argument.

Everything here is one column of 200000 synthetic URLs written the way ClickBench q28's referers are, which is 7.52 million characters of ASCII, run with q28's own pattern `^https?://(?:www\.)?([^/]+)/.*$` compiled to 25 instructions. Each line is one thread doing one thing, and the figure is the user CPU of the process over three passes of the column rather than the wall clock, because the machine this was taken on carries other work and wall clock there says more about the neighbours than about the program.

| what runs | CPU over 22.56 million characters | per character |
| --- | --- | --- |
| building the column and nothing else | 0.01 s | |
| reading the bytes as code points | 0.05 s | about 2 ns |
| the same answer written by hand over the bytes | 0.02 s | about 1 ns |
| `matches`, which is the yes or no | 1.10 s | about 48 ns |
| `search` with captures, which is what a replace runs | 1.75 s | about 77 ns |

The last two lines have the decode inside them, because a scan has to read the bytes as code points before it can run, and the hand written line does not because it works in bytes and never needs one.

A sample of the yes or no scan puts 69 percent of it in `_queue`, 19 percent in the loop in `matches` that calls it, 5 percent in the decode and 3 percent in `_accepts` and the range search underneath it. The number of threads the machine holds was counted rather than guessed at: 25.68 million thread steps over 22.56 million characters, which is 1.14 threads per character.

Those two together say something that was not obvious before it was measured. The machine is not slow here because it is holding many possibilities at once, since it is holding barely more than one. It is slow because each character costs a walk of the program through a recursive function with a stamp array, a list append and a dozen arguments, and that walk is most of the run whatever the pattern is doing.

### The alphabet is not where the time is

Reading a character and asking whether an instruction accepts it is 3 percent, and turning the row's bytes into code points is 5 percent. So moving the machine to UTF-8 bytes, which is what RE2 does, would be a large change to the compiler in exchange for a few percent of the scan.

It is still worth having a cheaper alphabet, because a DFA has to key a transition on the input and a million entries per state is not a table. The decision is to keep code points and to give the compiled program a class table instead: one number per distinct set of characters the program can tell apart, computed once when the pattern is compiled, with ASCII indexed directly and anything above it found by the same kind of binary search the ranges already use. A pattern's class count is small, since it is bounded by the number of distinct sets in the program plus one for everything else, so a state is a row of a few entries rather than of a million.

What that decision gives up is `\B`, which section 6 above has as a gap because RE2 asks the word boundary question between bytes and this machine has no byte positions to ask it at. Moving to bytes would close it. It stays open, and it stays open knowing what it would have cost to close: a UTF-8 range compiler in front of every class, the fold tables reworked to match, and the three differentials rerun against a compiler that now writes several instructions where it used to write one. That is a milestone of its own to close one line in one report.

If the DFA ever does want bytes, the class table is the piece that changes and the cache above it is not, because the cache is keyed on a class number and does not know where the number came from.

### The DFA reaches the yes or no scans and not the capturing ones

A DFA replaces the 69 percent. One lookup per character in a table of states, filled when a state is first reached, instead of a walk of the program per character. The floor underneath that is the hand written line in the table, which is about 1 ns per character, and the yes or no scan today is about 48.

So `contains`, `count`, `match` and `fullmatch` are what a DFA is for, and the gain there should be most of the difference between 48 ns and something near the hand written kernel.

`replace` and `extract` are not. A DFA says whether and where a match ended and cannot say where each group started, so the shape everyone uses is to find the span with the DFA and then run the machine over the span to fill the groups in. On q28 that shape saves nothing at all, because q28's pattern ends in `.*$` and so the span is the whole row every time it matches. The scan would find the span quickly and then pay the same 77 ns per character it pays now.

That is the thing this measurement changed. The issue that asks for a DFA is motivated by q28, and a DFA on its own does not answer q28.

### What answers the capturing scans is a bounded backtracker

The reason the capturing scan costs 77 ns rather than 48 is that every thread carries a slot vector and the fresh attempt at each position copies one. The reason it cannot simply backtrack instead is the whole of section 2: a backtracking engine on `(a+)+b` does not finish.

There is a third engine that is neither of those, and RE2 carries it for exactly this case. It backtracks, and it keeps a bitmap of the instruction and position pairs it has already tried, so a pair is never tried twice and the work is bounded by the size of the program times the length of the span. That makes it safe on `(a+)+b` for the same reason this machine is safe, and it makes a match with one capture over a forty character row a couple of passes over the row with one instruction per step rather than a thread list per character. It is only affordable when the span is short enough for the bitmap, which is why it is a third engine rather than a replacement, and the span is exactly what the DFA hands it.

So the order of the work is the class table first, since both of the other two want it; then the DFA, which answers the four scans that ask yes or no; then the bounded backtracker, which is what q28 has been waiting for. Issue #863 carries all three and the measurements above are why it is in that order.
