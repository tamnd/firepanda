# 85. The flag that picks an engine

## 1. What this is

`str.contains` and `str.fullmatch` under a `flags` argument, answered out of Python's engine rather than out of RE2. Document 84 measured that those two and `count` hand any flag at all to Python's `re` upstream and left all three refused, with section 11 of that document naming the scan they were waiting for. This is the half of that scan which needs only a mask, so the two mask methods come off the refusal and `count` and `replace` stay on it. Issue #8 M6.

The slice is small in the kernel and awkward in the layer above it, and the awkwardness is the interesting part. The flag bits cannot carry the decision they look like they carry, the pattern has to be anchored differently once it has moved, and the measurement turned up a fifth place where the two engines disagree that nothing in the reading had predicted.

## 2. The bits cannot say it on their own

A flag argument means two separate things at once and only one of them is about the pattern.

It means what the letters mean, so `re.MULTILINE` moves where a dollar sign may sit and `re.DOTALL` lets a full stop cover a newline. That part is already handled, because document 84 seeds the parser with whatever flags it is given and the tree comes out carrying them.

It also means the call has moved to a different engine, because upstream routes on the presence of the argument rather than on what the argument says. That part cannot ride in the bits, because `case=False` arrives at exactly the same place as exactly the same bit. A caller who wrote `contains(pat, case=False)` stays on RE2 upstream and a caller who wrote `contains(pat, flags=re.IGNORECASE)` does not, and by the time the two reach the compiler they are one `FLAG_IGNORECASE` and are indistinguishable.

So the routing has to be told separately from the meaning. `program_for` takes an `argued: Bool` beside the flags, and it is the only thing in the function that decides between the two engines for a pattern both of them can run.

## 3. A number beside the two words

The kernel side of the accessor is three doors picked by the shape of the answer, and a mask goes through `flag`. The rule that door has always followed is that a variant of a method is a new word rather than a new argument, which is why `case=False` on a real pattern is `contains_regex_folded` and not `contains_regex` with a boolean beside it.

That rule does not scale to this. Seven flag letters in any combination is 128 words per method, which is not a list anybody wants to write down and not a list anybody wants to read. So the flags ride in a number, and the routing rides in that number being nonzero.

The `text` door already had a numeric slot of its own, a `start` that doubles as a count, so the shape is not new to the file. What is new is that the number is doing two jobs, and the function that reads it says so: `_compiled` ors the argued flags with `FLAG_IGNORECASE` when the word ended in `_folded`, and passes `argued=argued != 0` separately. A pattern folded by the word and a pattern folded by the argument produce the same seeded tree and different engines, which is what upstream does.

## 4. The seven letters numbered twice

`re` and the kernel both number the flags and they agree on nothing. `re.IGNORECASE` is 2 and `FLAG_IGNORECASE` is 1, `re.MULTILINE` is 8 and `FLAG_MULTILINE` is 4, `re.UNICODE` is 32 and `FLAG_UNICODE` is 64, and the other four are shifted about similarly. The two sets were written for different readers and neither of them is wrong, so the translation is a table of seven pairs in the Python layer with the table written out rather than computed.

The translation is also where an unknown value is caught. A caller who passes a number holding a bit that is none of the seven letters gets a refusal naming the leftover bits, rather than having them dropped. Upstream would compile such a value and let `re` complain, and the complaint is worth keeping even though nobody arrives here by accident, because silently ignoring part of an argument is the one failure a compatibility layer must not have.

## 5. Anchoring stops when Arrow stops

`match` and `fullmatch` are `contains` with a rewritten pattern on the Arrow path. The rewrite wraps the pattern in a group and pins it, and pandas does the same rewrite upstream for the same reason: Arrow answers one question and the accessor needs three.

The moment the call leaves Arrow, upstream stops rewriting. It hands the pattern the caller wrote to `re.compile` and answers with `regex.fullmatch` instead of `regex.search`, so the anchoring comes from the method being called rather than from anything in the pattern.

That matters because the Arrow rewrite is wrong under two of the flags it may now be carrying. `fullmatch` pins its tail with a dollar sign, and a dollar sign under `re.MULTILINE` is happy at the end of the first line, so `fullmatch("a", flags=re.M)` on a row holding `a` and then `b` would answer yes where upstream answers no. A dollar sign is also happy before a trailing newline whatever the flags say, and `re.fullmatch("a", "a\n")` finds nothing.

So an argued call is anchored by `python_anchored` rather than by `anchored`, and it writes `\A(...)\z` instead of `^(...)$`. Those are the two positions no flag can move: `\A` is the start of the text and `\z` is the end of it with no allowance for a newline. The leading flag group is hoisted the same way the Arrow rewrite hoists it, because `(?i)` has to stay at the front of the pattern or Python refuses to compile it at all, and nothing is cropped or stripped because upstream crops and strips nothing here either.

`match` gets `\A(...)` and no tail, and nothing reaches that branch today, because the only flags upstream lets `match` keep are the two that leave it on Arrow. It is written anyway and tested directly, because the branch it is missing from would be a silent wrong answer rather than a crash the day something changes.

## 6. A fifth difference between the two engines

The two engines differed in four places before this slice: the Perl classes are Unicode to Python and ASCII to RE2, `\b` is measured against the wider class, `$` sits before a trailing newline for one of them, and the Turkish I folds into a group of four rather than a pair. All four were reasoned about first and then measured.

The fifth was measured first. The differential in section 10 found 54 rows where `contains(r"\B", flags=...)` answered yes and pandas answered no, all of them the empty row.

Python's `\B` fails on an empty subject. It is not a consequence of any rule about word characters, because both sides of an empty string are equally not a boundary and the negation of not a boundary is a boundary. It is a special case written into CPython in 3.12, confirmed with `re.finditer` rather than read: `''` gives no matches, `' '` gives two, `'a'` gives none and `'ab'` gives one. RE2 has no such case and matches. So `str.contains(r"\B")` on an empty row answers True and the same call with a flag beside it answers False, in pandas as much as here, which is now an entry in the upstream observations list.

The fix is four lines in the matching machine, on the Python anchor only, and it is the one anchor in that function that is not the negation of its partner. After it the same differential ran clean.

## 7. Which flags are answered and which are still refused

Four of the seven letters go through. `re.IGNORECASE`, `re.MULTILINE`, `re.DOTALL` and `re.UNICODE` are read by the parser and mean what they mean, and `re.UNICODE` means nothing at all because the Python engine already reads its classes the Unicode way.

`re.LOCALE` is refused and it is not a gap. Python refuses it too, with `cannot use LOCALE flag with a str pattern`, so the refusal is a `ValueError` and it is upstream's answer rather than a missing feature.

`re.VERBOSE` and `re.ASCII` are gaps and say so. The parser does not read verbose mode, where whitespace in the pattern stops counting and a comment may run to the end of a line, and it does not narrow the Perl classes for the ascii flag. Both refuse rather than answer, because a verbose pattern read as an ordinary one is a different pattern and an ASCII `\w` read as a Unicode one is a different class.

`count` and `replace` still refuse every flag, for the reason document 84 gives and document 79 measured: the loop that counts and the loop that replaces are not the loop that asks once, and Python's versions of both differ from Arrow's in ways that produce a column of plausible wrong numbers rather than an error. `extract` is on Python's engine already and refuses a flag for the same reason it refuses one today, which is that its own scan is the one being changed.

## 8. A flag beside a search that reads no pattern

`contains(pat, flags=re.I, regex=False)` is accepted upstream and the answer comes back the same as without the flag, which is already in the upstream observations list as the flag being ignored. It is not ignored, it is spent on a route that happens to agree, and the agreement stops the moment a `case` is beside it.

Measured on a column holding a sharp s and a row reading STRASSE: `contains("ss", case=False, regex=False)` is False on the sharp s and True on STRASSE, and the same call with any flag added is True on both. The first is Arrow comparing without case and the second is Python upper casing both sides, where a sharp s upper cases to two letters. `contains("k", case=False, regex=False)` finds a Kelvin sign and the same call with a flag does not, for the other half of the same reason.

So a flag beside `regex=False` is a route rather than a no-op, and it is refused here rather than answered, because the two answers it picks between are both available and only one of them is the one this library gives. The refusal says which route it is refusing to take, which also keeps it distinguishable from the refusal a flag gets on `count`, where upstream serves a scan that is not written here.

## 9. A pattern that is not text

The type check moved. It used to sit inside the function that decides whether a pattern can take the byte search, which was the only path a non string could reach, and an argued call skips that function entirely because the engine is decided before the pattern is read. So `contains(1, flags=re.I)` would have gone to the compiler with an integer.

It is now its own two line check called at the top of both methods that take a pattern, which is also the order upstream has: `re.compile` raises `TypeError` before anything looks at the flags, so a call that is wrong in both ways gets the type error.

## 10. How it was checked

Two methods, 43 patterns, 9 flag combinations and 3 settings of `case`, over 31 rows holding newlines, the Turkish I, the Kelvin sign, the long s, the three sigmas and an empty row. That is 2322 comparisons against live pandas 3.0.5, and after the `\B` fix there are no differences and nothing skipped.

The Python suite has fifteen new tests of its own covering the pieces that a wholesale comparison cannot point at, which are which engine answered, where the anchoring came from, and what each of the three kinds of refusal says. The Mojo suite has five more on the anchoring rewrite and on the refusal rules the two engines do not share.

Three tests elsewhere in the Python suite asserted that a flag on `contains` or `fullmatch` was refused. They now assert what it answers, and they keep the `count` and `replace` refusals that are still true. That is the second slice in a row to rewrite tests in that direction and it is the honest shape of the work.

## 11. What moved

`str.contains` and `str.fullmatch` now cover the `flags` parameter, which is the one parameter either of them had left. Nothing on the conformance board moves yet, because the board's cases for this parameter sit on `replace` and `extract` and both of those are still on the L2 ceiling document 84 named. They come off it when the counting and replacing scans land.

## 12. What is not here yet

The other half of the scan, which is the counting loop and the replacing loop. Both follow `re.finditer` rules, which means advancing one character on an empty match, never cutting the text and never stepping bytes, and between them they give `count` and `replace` under `flags` and `replace` under `case=False` with a real pattern. That is what lifts `str.replace` and `str.extract` off L2.

Verbose mode in the parser, which is the only one of the seven letters that needs grammar work rather than wiring, and the ascii flag, which needs the Perl classes to be narrowable at build time rather than chosen at build time.

A scoped flag group is still refused, which is 525 held-out patterns in the differential. The five constructs the router sends to Python are still refused: lookaround, backreference, conditional, atomic group and possessive quantifier. The RE2 grammar front end is still the largest single gap in the component at 19620 of 30052 corpus patterns.
