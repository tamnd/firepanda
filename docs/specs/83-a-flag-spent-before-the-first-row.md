# 83. A flag spent before the first row

## 1. What this is

`(?i)` written inside a pattern, on every method of the `str` accessor that reads one. It is the largest single family of patterns the regular expression differential was holding out, it is the last of the seven inline flag letters that both engines have and this compiler refused, and it is the first thing in this component where a prediction that had been carried in the specification for two documents turned out to be wrong when it was finally measured. Issue #8 M6.

The argument spelled `case=False` is not this. That one is a literal search with a folding compare, it was built in document 69, and for a regular expression it is still refused. Section 10 says what it needs and why it is its own slice.

## 2. The prediction the measurement overturned

Document 81 section 11 says case folding "is also two tables, because the two engines fold different alphabets". Document 82 section 10 repeats it. Both sentences were written from the reasoning in document 81 section 4, which is that RE2 reads the three Perl classes as ASCII and Python reads them as Unicode, so the two engines plainly disagree about which characters are letters and would therefore be expected to disagree about which letters are the same letter.

They do not. There are 2927 code points that either engine considers cased, they fall into 1468 groups, and the two engines put 1464 of those groups together identically. The four code points they disagree about are `I` at U+0049, `i` at U+0069, the dotted capital `I` at U+0130 and the dotless small `i` at U+0131. Python reads all four as one letter. RE2 reads `I` and `i` as one letter and leaves the other two alone, each matching only itself.

So it is one table and a four code point exception, not two tables. The exception is not a table at all, it is nine lines in the compiler, and the reason the wrong answer survived two documents is that the reasoning behind it was sound and nobody had run the measurement. A character class being ASCII on one engine is a real difference and it turns out to be a difference about what `\w` means rather than about what folding means.

The group sizes are worth putting down because they are the shape of the problem: 41 groups of one, 1399 of two, 24 of three and 4 of four. The singletons are code points with a case mapping whose other half is more than one character long, so nothing folds onto them. The four is the Turkish family. Anything that assumed a fold was a pair would be right 95 percent of the time and wrong about Greek sigma, which has three.

## 3. Both engines asked, and the two checks that make the answer safe

Python's answer is read out of `_sre.unicode_iscased` and `_sre.unicode_tolower`, which is what `re` itself compiles a cased literal with, merged with the fifty groups in `re._casefix._EXTRA_CASES` where lowering is not enough. RE2's answer is read by running `pyarrow.compute.match_substring_regex` with a pattern of `(?i)\x{...}` for one code point at a time against an array holding every candidate. The escape rather than the character itself is deliberate, so that nothing in the measurement depends on which characters RE2 thinks need escaping.

Two checks stand between that and the committed table.

The first is closure. Every measurement above is taken against a candidate set, and a candidate set is built out of code points that `str.lower`, `str.upper`, `str.casefold` or `str.title` moves. A code point that none of those four move but that some engine still folds would be invisible to the measurement and wrong in the same direction in both halves of it, which is the failure mode a differential cannot see. So a class holding every candidate is compiled with `(?i)` and run against all 1114112 code points on both engines, and both have to answer with the candidate set exactly. Both do.

The second is the difference itself. The generator refuses to write the file unless the set of code points the two engines disagree about is exactly those four, unless Python still reads all four as one group, and unless RE2 still reads `I` and `i` as a pair and the other two as singletons. A release of either engine that moves any of that fails the generator rather than drifting quietly into the committed table.

## 4. Where the flag is spent

Nowhere near a row. A literal under the flag becomes the set of everything that folds onto it, a class has the folds of everything in it added, and the program that comes out the other end has no flag on it and the machine that runs it never learns one was set.

That is RE2's arrangement and the reason is arithmetic. Folding at run time is a table lookup per character of every row of the column. Folding at compile time is a table lookup per character of the pattern, once. For a column of a million rows and a pattern of five characters that is a million lookups against five, and the answers are the same.

It also means the whole of this slice is testable without a column. Every test in `tests/test_regex_fold.mojo` asks whether a pattern matches a row, and every one of them is really asking what the compiler put in a set.

## 5. Fold, then negate

`(?i)[^a]` does not match a capital `A`, in both engines and here.

That is the only interesting ordering question in the slice and it has exactly one right answer. Folding the set and then applying the caret gives the complement of `{a, A}`, which excludes both. Applying the caret and then folding the complement gives the complement of `{a}`, which holds `b` and `B` and `c` and `C` and every other letter, and folding that adds `A` back in through the other case of every letter that is not the one the caller wrote down. The second order does not merely differ, it makes a negated class under the flag match almost everything, which is a wrong answer that looks like a column of booleans.

The same ordering applies twice more. A one character negated class is its own node in Python's parser rather than a class, so the fold has to be spent there too or `(?i)[^a]b` is a silent hole. And a negated category, which section 6 is about, is folded and then negated rather than the other way round for the same reason.

## 6. The ASCII class that is not closed under folding

This is the second thing the measurement overturned, and it arrived as a test failure against pandas rather than as a thought.

Python's `\w` is Unicode. Folding a code point never moves it into or out of that class, so Python's three Perl classes are closed under folding and the flag has nothing to add to them. RE2's `\w` is `[0-9A-Za-z_]`, and that is not closed: the Kelvin sign at U+212A folds onto `k` and the long s at U+017F folds onto `s`, and neither of them is in an ASCII class. So `(?i)\w` on RE2 matches two code points that are not ASCII, and `str.contains(r"(?i)\w")` on a row holding the Kelvin sign is True in pandas today.

That makes the negation question real rather than academic. `(?i)\W` on RE2 has to be the complement of the widened class, which excludes the Kelvin sign. Folding the complement of the narrow class instead would add a plain `k` to `\W`, which is not a near miss, it is a word character in the class of non word characters. RE2 has a comment where it does this saying the fold case adjustments make the obvious order incorrect, and the measurement agrees with the comment.

The consequence for the code is that folding happens per item inside a character class rather than once over the union the brackets build. `[\W]` is a class holding one item and the item is a negation, so the fold is spent inside the item. That is one more place than the first implementation had, and the first implementation passed every test in the Mojo file and failed against pandas on the seventh row of a twenty pattern sweep.

## 7. The letter with no other case

A digit under the flag is still a comparison and not a set.

2927 code points out of 1114112 are cased. Everything else folds onto itself, and emitting a set of one for each of them would turn the commonest instruction in any program, matching one literal character, into a binary search over a one element table. So the literal branch folds first, looks at what came back, and only builds a set when the answer is more than the code point it started with.

This is the kind of thing that is invisible in a correctness test and is the whole reason the instruction set has a separate one character instruction at all. The test for it reads the compiled program rather than the answer, because both spellings answer the same and only one of them is the reason the fast path exists.

## 8. The table's shape, and the seventy seconds the first shape cost

The first table was the groups written out: every cased code point sorted, an offset per point, and every group written once for each member it holds. That is 2927 plus 2928 plus 5917, which is 11772 numbers in a generated Mojo file of 11817 lines.

It worked and it was correct and it cost seventy one seconds to compile. A test file that touches the compiler and runs eighteen assertions in under a tenth of a second took seventy one seconds of wall clock, essentially all of it the compiler reading a large array literal. Measuring a file holding only the first of the three arrays gave forty five seconds, so the cost is roughly linear in the number of entries with a fixed twenty seconds of everything else, and a table this size is a tax on every test file in the repository that reaches the regular expression compiler.

The shape that replaced it is the one Go's `unicode.SimpleFold` uses. Store the cycle rather than the groups: each cased code point has one successor, the next member of its group counting upwards and wrapping at the top, so a caller reads a whole group by following the successor until it comes back to where it started. That halves it at once, because a group of four is written once instead of four times.

Then write the successor as runs, since it is nearly always the same arithmetic over a long stretch of the alphabet. A run is a low, a high and a delta. One delta is not a number but a marker meaning the alphabet alternates upper and lower in pairs, so an even code point's successor is the one above it and an odd one's is the one below, and that single marker is what turns 1411 runs into 473. Forty of the 473 carry it.

So the committed table is 1419 numbers rather than 11772, and the test file costs nine seconds rather than seventy one. The generator checks the compression by walking the runs for every cased code point and comparing the group that comes back against the group it measured, so the table is verified against the measurement rather than against the code that compressed it.

The lesson is not that the first shape was careless. It is that a generated table is a compile time cost in a language that evaluates array literals at compile time, and that cost is not visible in any test, it is visible only in how long the tests take to start.

## 9. What the word boundary does, which is nothing

`\b` is not folded on either engine, and the reason it is worth a section is that the class it asks about is the one section 6 just widened.

Each engine asks the boundary question against its own word class, unfolded. RE2's stays ASCII under the flag, so the Kelvin sign is not a word character to `\b` even though `(?i)\w` now matches it, and `(?i)\bk` misses a row that is a Kelvin sign on its own and finds one that is a letter followed by a Kelvin sign. Python's word class holds it, so the same pattern over the same two rows answers the other way round. Both of those are pandas' answers today and both are asserted.

It would have been entirely reasonable to widen the boundary's class along with the category's, and it would have been wrong.

## 10. The arguments beside the flag, and a defect in pandas

`case` and `flags` as arguments to the five pattern methods are the next slice, and measuring how pandas routes them turned up something that should be filed upstream.

The routing is not one rule. `contains`, `match` and `fullmatch` with `case=False` stay on Arrow and get RE2's alphabet. `contains`, `fullmatch` and `count` with any non zero `flags` go to Python's `re`. `replace` with `case=False` or with any `flags` goes to Python. `extract` is Python always. An inline `(?i)` never moves a call between the two, which is why this slice needed no accessor change at all.

`match` is the odd one and it is the defect. `Series.str.match(pat, flags=re.I)` goes to Arrow, where every other method with a flag goes to Python. `Series.str.match(pat, flags=re.M)` raises `ValueError: Cannot pass flags that do not match pat.flags`, and so does every flag but IGNORECASE. The mechanism is three files deep: `accessor.py` has a block in `match` and nowhere else that pre-compiles the pattern with `flags | re.U` and then sets `flags` to zero, `string_arrow.py` sees a zero `flags` beside a compiled pattern that carries some and hands the call to the object path, and `object_array.py` compares `flags | re.U` against the compiled pattern's own flags and finds they differ. So an argument the caller passed correctly is rejected by a check that the accessor's own rewrite made fail.

The consequence a compatibility layer has to decide about is that `str.match("i", flags=re.I)` and `str.fullmatch("i", flags=re.I)` take different engines for the same flag and the same pattern, so they disagree about the Turkish dotted `I`. Two adjacent methods on one accessor, one argument, one letter.

Three smaller observations came out of the same measurement. `contains(pat, flags=re.I, regex=False)` silently ignores the flags. `replace(pat, repl, case=False, regex=False)` uses Python's `re` while `contains(pat, case=False, regex=False)` uses Arrow, so the two literal paths fold different alphabets for the same column. And `str.match(pat, flags=<anything>)` with a pattern that is already a compiled object behaves differently again.

## 11. What is not here yet

Scoped flags. `(?i:a)` is still refused with the sentence saying the group is not carried, because the parser reads the letters and throws them away, and a program built from that tree would answer without folding while both engines fold. Carrying the flags on the node is its own piece of work and it is what 525 held out patterns in the differential are waiting for.

The `case` and `flags` arguments, per section 10. They are what lifts `str.replace` and `str.extract` off their L2 ceiling on the conformance board, since both are capped there by an uncovered `flags` parameter rather than by anything about their answers.

The five constructs the router sends to Python are still refused by both engines: lookaround, backreference, conditional, atomic group and possessive quantifier. `findall` and `extractall` still want a list column and a `MultiIndex` respectively. The RE2 grammar front end is still the largest single gap in the component, at 19620 of the 30052 corpus patterns.

What did move is the differentials. `str.contains` now compares 7771 of the corpus and holds out 22281, `str.match` and `str.fullmatch` compare 7766 each, and `str.findall` on the other engine compares 26712 and holds out 3340, up from 26608 and down from 3444. All four agree with pandas on ten thousand cases out of ten thousand. Case folding is gone from both tallies of reasons a pattern is set aside, which is the first time a reason has left either list rather than shrunk.
