# 66. A pattern that is only itself

## 1. Four names that pandas answers with an engine we do not have

`str.contains`, `str.match`, `str.fullmatch` and `str.count` are four questions about where a pattern sits in a row. Anywhere, at the front, the whole row, and how many times. pandas reads the argument to all four as a regular expression, and there is no regular expression engine in this library. Documents 64 and 65 finished the questions about what a character is; these four open the questions about patterns, which is where the sixteen names of the accessor that are left after them live and which is a much larger piece of work than eight tables of ranges.

They ship now anyway, on a smaller promise than pandas makes, and this document is about why that promise is worth making and where its edge is.

## 2. A pattern with no metacharacter means one thing

A regular expression holding none of `.`, `^`, `$`, `*`, `+`, `?`, `{`, `}`, `[`, `]`, `\`, `|`, `(` or `)` matches exactly the characters it is written with, in that order, and nothing else. `re.search("bc", row)` and `"bc" in row` are the same question for every row there is. That is not an approximation and it does not depend on the engine, so a byte search answers it exactly.

So the Python layer reads the pattern. A pattern with none of those characters in it goes to a kernel and the answer is pandas' answer. A pattern with one of them in it is refused, by name, saying which character caused it. A caller who wrote `contains(".")` meaning a full stop is one keyword away from the answer they want and `regex=False` gives it to them; a caller who wrote `contains("^a")` is not, and the message they read lets them tell which of the two they are.

The alternative worth ruling out is searching for the metacharacter literally and saying nothing. `contains(".")` under that reading is False on nearly every row and pandas answers True on nearly every row, and a caller who ported a program would find their filter had quietly emptied. The board would read a wrong answer rather than a gap. An absent name reads as unimplemented and a refusing one reads as a failure, which is the trade document 07 describes, and a name that answers confidently and wrongly is worse than either.

## 3. Three of the four cost nothing

Contains is contains, `match` is starts with, and `fullmatch` is equality against a constant. `pattern.mojo` had all three of those before this work started, because a `LIKE` pattern is read into exactly those four shapes and the file says so in `MatchKind`. `str.match` and `str.startswith` are the same kernel call with two names, which they already were in pandas for reasons that do not survive to a column: one of them takes a regular expression and a tuple is meaningful to the other, and neither difference is about bytes.

Only the count needed a kernel, and it is the search in a loop with the cursor moved past each hit.

## 4. Matches do not overlap, and that is the only rule worth an opinion

The cursor moves by the whole pattern after a hit rather than by one byte, so `aa` appears twice in `aaaa` and not three times. A regular expression engine scanning for successive non overlapping matches answers the same, which is what `re.findall` does and what pandas answers, so there is nothing to choose here. It is asserted directly rather than only through the scalar twin, because it is the one line of the kernel a reader might think should be different.

## 5. An empty pattern is counted in bytes

This is the one place the four say something a caller would not predict.

`pandas.Series(["héllo"], dtype="str").str.count("")` is 7. `len(re.findall("", "héllo"))` is 6. The row is five characters and six bytes, and pandas is counting a match at every byte offset and one past the end. The reason is that pandas 3 holds text in Arrow and Arrow's `count_substring` counts by offset, and the offsets are bytes.

Measured across the three ways pandas can hold the same column, a column of object answers 6 and a column of `str` and a column of `ArrowDtype` both answer 7. So this is not a rule anybody chose, it is an implementation detail of one backend that became the default backend, and pandas 2 with its object columns would have said 6.

This library answers 7, and it does so for exactly the reason document 65 section 11 gives for calling a half sign a digit: where Arrow and Python disagree and pandas answers Arrow, this library answers pandas. The premise on the front of the project is that a program keeps running, and a program keeps running by getting pandas' answer rather than the defensible one. It is written into the kernel docstring, the scalar twin, the Mojo test and the Python test, because a single assertion saying seven reads like a typo and four of them reading the same way with the measurement attached reads like a decision.

It is also worth reporting upstream, and it is on the list with the others. `Series.str.count("")` changed its answer when the default dtype changed, and it changed it to a number of bytes in a method whose other answers are numbers of matches.

## 6. Two arguments refused rather than ignored

`case=False` is a case folding pass over both sides before the search, which is a kernel that does not exist yet, and `flags` is a regular expression engine's argument in a method that has no engine. Both have a default that means leave it alone and both are refused when they are set to anything else.

Ignoring either would be the worst available behaviour. A caller who wrote `contains("abc", case=False)` and got the case sensitive answer has a wrong result with no indication anywhere that anything was dropped, and it is the failure mode a compatibility layer exists to prevent. `count` has no `case` argument at all, which is the one place the four disagree about their own signatures, so it passes the check nothing.

## 7. What the board says

Six cases are armed and the board moves from 2599 passing runs to 2622, with unimplemented falling from 1762 to 1745 and divergent rising from 122 to 126. The failures stay where they were. The strings section reads 14 of 58 callables at L3 with 19 divergent and 22 unimplemented, and the L3 figure does not move, because a callable that diverges on any frame it runs on does not count as fully conforming and all four of these diverge on the null heavy frame.

Three of the six cases are new to the board rather than newly answered. The registry had `strings/match` and `strings/fullmatch` and both of them used a regular expression, so there was no literal form of either question anywhere in the corpus, which is a gap in the board that has nothing to do with what this library implements. `strings/match-literal`, `strings/fullmatch-literal` and `strings/count-empty` fill it, and pandas against pandas still answers every run, at 4502 rather than 4492.

The null row is the one divergence and it is not a new one. pandas' `str` backed column answers False for a missing row on all three masks and this library keeps the row missing, which is `engine/string-predicate-null`, and it held twelve cases before `contains`, `match` and `fullmatch` joined it. The count goes to the other entry it belongs to: `engine/string-count-width` already said that `len`, `find` and `rfind` answer int64 with a null where pandas widens the column to float64 to make room for a NaN, and how many times a pattern appears in a row that is not there is the same unanswerable question with the same answer, so `count` is the fourth case there rather than an entry of its own. The registry is still 29 entries, which is the number it should be when a slice adds no new kind of disagreement.

Before any of the four were armed on the board, the corpus was checked for a row that would fail if each implementation were wrong, which is the check the last two slices in this accessor both needed and the second of them had to add rows for. This time it came back yes without changes: the string frames hold a row where the pattern appears twice, a row where it appears at the front, a row that is the pattern exactly and a row that is empty, and those four separate all four names from each other.

## 8. What is left

Sixteen `str` names, and the shape of the remaining work is now clear rather than a list. `cat`, `join`, `partition`, `rpartition`, `split`, `rsplit`, `get_dummies` and `wrap` are about cutting a row up or putting rows together and need no engine at all, and several of them answer a column of lists, which is a type this library does not have yet and is the real blocker rather than the pattern. `extract`, `extractall`, `findall` and `replace` with `regex=True` need the engine. `normalize` needs the Unicode normalization tables, which is the one piece of character data document 65 did not build. `decode`, `encode` and `translate` are their own small problems.

`replace` is the near one. Its `regex` argument defaults to False in pandas 3, which means the default call is a literal replacement and is exactly the shape this document describes, and it is one kernel away.
