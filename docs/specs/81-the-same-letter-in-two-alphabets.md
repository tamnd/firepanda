# 81. The same letter in two alphabets

## 1. What this is

Document 76 built a parser whose grammar is Python's and a router that decides which of two engines answers a call. Document 77 built the machine, and documents 78, 79 and 80 wired five methods to it. All five of those go to RE2, because the router sends them there, so the second engine has been a name in a switch statement and nothing else since it was written.

This is the second engine. It exists because three of the accessor's methods never reach Arrow at all, and the pattern they hand to Python's `re` does not mean what the same pattern means two lines above.

Everything below was measured against pandas 3.0.5 with pyarrow 25.0.1 on CPython 3.13.12, whose Unicode is 15.1.0, while pyarrow 25.0.1 is built against Unicode 16.

## 2. The finding

```
>>> s = pandas.Series(["café"], dtype="str")
>>> s.str.count(r"\w")
0    3
>>> s.str.findall(r"\w")
0    [c, a, f, é]
```

Three and four, on one row, from one accessor, with one pattern. Neither answer is a bug.

`count` goes to `pyarrow.compute.count_substring_regex`, which is RE2, and RE2's `\w` is 63 characters of ASCII. `findall` never goes near Arrow: pandas compiles the pattern with `re` and loops in Python, and Python's `\w` is 138558 code points. The row holds four word characters to one of them and three to the other.

The same thing happens to `\d`, more sharply, because there the narrow class is ten characters:

```
>>> s = pandas.Series(["１２３"], dtype="str")
>>> s.str.count(r"\d")
0    0
>>> s.str.findall(r"\d")
0    [１, ２, ３]
```

And to `$`, which is not a class at all:

```
>>> s = pandas.Series(["a\n"], dtype="str")
>>> s.str.contains(r"a$")
0    False
>>> s.str.findall(r"a$")
0    [a]
```

## 3. Which methods take which engine

Six go to Arrow and get RE2's reading: `contains`, `match`, `fullmatch`, `count`, `replace` and `split` with `regex=True`. The first five are documents 78 through 80. `split` goes to `pc.split_pattern_regex` and is not written here yet because it wants a list column.

Three go to Python's `re` and never touch Arrow: `extract`, `extractall` and `findall`.

The routing is by method name and not by pattern, which is the part that makes this hard to notice. Document 76's router decides between the two engines for the first group, by walking the tree looking for a lookaround, a backreference or a conditional. That router is never consulted for the second group. `extract` is answered in Python whatever the pattern says, and `contains` is answered in Arrow unless the tree holds one of the three.

So there are two different reasons a call reaches Python's engine, and only one of them is a decision about the pattern.

`rsplit` deserves a line of its own. `StringMethods.rsplit` has no `regex` parameter at all, and `ArrowExtensionArray._str_rsplit` has no regex path in it. `split` has both. Splitting from the right is not the same method with an argument flipped, upstream, and a reader who assumes it is will write a call that silently splits on a literal.

## 4. What the three classes are

RE2's three are ASCII and are four ranges of literals. `\w` is `[0-9A-Z_a-z]`, which is 63 characters. `\d` is `[0-9]`. `\s` is `[\t\n\f\r ]`, which is five and does not hold the vertical tab.

Python's three are Unicode, and all three are exactly derivable from a string method, which was checked over every code point rather than assumed:

`\w` is what `str.isalnum` says yes to, plus the underscore. 138558 code points in 749 ranges.

`\d` is exactly `str.isdecimal`. 680 code points in 64 ranges. It is not `str.isdigit`, which says yes to 808, and the 128 in the difference are the superscripts and the circled numbers and everything else that is a number to read and not a digit to arithmetic. This is the one a reader gets wrong from memory.

`\s` is exactly `str.isspace`. 29 code points in 10 ranges, and it holds the no break space, which surprises people, and does not hold the zero width space, which surprises the other people.

Arrow's own string predicates are a third set again, and `str.isalnum()` and its seven siblings are answered out of those. Document 65 has them. They are not either of the two sets above: Arrow's alphanumeric class plus the underscore is 147597, which is 9039 wider than Python's `\w`, and Arrow's decimal class is 90 wider than Python's `\d`.

That gap is not a disagreement about any definition. Every one of the 9039 and every one of the 90 is unassigned in CPython's Unicode 15.1.0 and assigned in pyarrow's Unicode 16, and both Python sets are strict subsets. The two libraries were built against different releases of Unicode, and that is the whole of it. It will close when CPython's release moves and it will open again the next time pyarrow's moves first.

So the same accessor holds three different alphabets, and which one a call gets depends on whether it asked a class question, a pattern question through Arrow, or a pattern question through Python.

## 5. The other two differences

`$` outside multiline mode matches at the end of the text in RE2 and matches at the end of the text or just before a newline that ends it in Python. pandas papers over one instance of this on the way to Arrow, by rewriting a trailing `\Z` to RE2's `\z`, and leaves the much commoner `$` alone. Section 2 has the pair of answers.

`\b` is asked against whichever `\w` is running, so it moves with the class rather than being a separate fact. It is worth stating anyway because it moves in both directions. The two ends of a word in Greek are boundaries to Python and are not to RE2, which is the direction a reader expects. Between an ASCII letter and a letter outside ASCII, RE2 sees a word character next to something that is not one and puts a boundary there, and Python sees two word characters and does not. So `a\bé` matches for RE2 and fails for Python on the same two characters.

`\B` is the one place this engine refuses RE2 and answers Python. RE2 asks the boundary question between bytes rather than between characters, which document 77 recorded, and this machine walks code points and cannot reproduce it. Python asks it between characters, which is what the machine already does, so the same instruction that is a refusal for one engine is ordinary work for the other.

## 6. What was built

`tools/gen_regexclass.py` writes `firepanda/kernel/regex/classdata.mojo`, which holds the three classes as low and high pairs. It reads them out of the running CPython's `re` rather than out of the published Unicode data files, for the same reason `tools/gen_charclass.py` reads pyarrow: pandas answers these three methods by compiling the pattern with this exact module, so the module is not an approximation of the right answer, it is the right answer. The generator asserts all three definitions from section 4 over every code point before it writes, so a future CPython that widens one of them without widening the matching string method fails there rather than passing quietly.

The tables are committed rather than built, because the Mojo build has no Python in it.

`compile_program` takes the engine it always took and now does something different with it. The difference is three things settled while the pattern is being compiled, so the machine that runs the program never learns which engine asked for it. A category node reads the Unicode ranges instead of the ASCII ones. `AT_END` becomes `AT_END_TEXT`, which is the reading in section 5. `AT_BOUNDARY` and `AT_NON_BOUNDARY` become their Unicode twins.

The one cost worth naming is that the boundary cannot read the table the class reads. A class is turned into ranges once while the pattern compiles, and a boundary is asked once per position of every row, and bringing a comptime table out into memory costs a copy of six kilobytes. So `Machine` holds one copy of the word ranges, built in its constructor and only when the program actually holds one of the two Unicode boundaries, and lends it down the walk. A machine is built once per column, so the copy is paid once per column rather than once per position.

## 7. What was measured

`pixi run differential-regex-python` is new. It generates the same 30052 patterns the other three regular expression differentials generate, from the same corpus and the same seed, and runs each of them over the same sixteen pieces of text through `str.findall`. It compares both the refusals and the answers, and the ceiling is zero for the reason it is zero next door: a wrong answer here is not a refusal a caller can see, it is a list that looks exactly like a right one.

`findall` is the method asked rather than `extract`, for two reasons. It takes a pattern with no groups in it, which most of the corpus is, where `extract` raises for one. And its answer for a row is a list whose length says whether anything matched, which is the same yes or no the match differential reports and lets the two be read side by side.

It compares 26608 patterns, holds out 3444 for reasons it tallies, and disagrees on none of them.

Those two numbers are the interesting part of the run. The RE2 differential on the identical corpus compares 7668 patterns and holds out 22384, and almost all of that hold out is one reason: 19615 patterns are RE2 syntax Python's grammar cannot read, which document 77 section 8 named as the largest single gap in the component. Python's engine has no such gap, because the parser is Python's grammar, so a pattern the parser cannot read is a pattern Python cannot read either and the two refuse together. Three and a half times as much of the corpus is answerable for the engine that was written second.

What is held out is eleven reasons and every one of them is this library falling short of an engine that reads the pattern: 742 backreferences, 678 case foldings, 661 lookarounds, 399 named characters, 354 possessive quantifiers, 186 conditionals, 173 atomic groups, 96 verbose flags, 83 ascii flags, 59 repeat counts over a thousand, and 13 empty negative lookarounds.

`tests/test_regex_python.mojo` holds fifteen cases, and every one of them is a pair: the same pattern against the same text compiled for both engines. A test showing `\w` matching a letter in Greek is a test about Greek, and a test showing the same `\w` matching it for one engine and not for the other is a test about the thing that will be got wrong. The three differentials for the RE2 side all still agree at ten thousand in ten thousand, which is what says the other engine was not disturbed.

## 8. Three refusals that stopped being refusals

Writing the second engine turned three families of pattern from refused into answered, and none of them was the point of the slice.

The parser records that a pattern holds syntax RE2 has never had, which is a comment group, a `\u` or `\U` escape, a `\Z` that is not trailing, a backslash in front of a character outside ASCII, or a one digit octal escape inside a class. It records it and reads the pattern correctly anyway, into the nodes Python reads it into. That record is a fact about the other engine, so for Python's engine there is nothing to refuse and the tree already in hand is the right one.

The same goes for the two constructs both engines read and read differently. `a{,2}` is `a{0,2}` to Python and five ordinary characters to RE2. `[[:alpha:]]` is a POSIX class to one and a bracket with some letters in it to the other. In both cases the tree is Python's reading, so again only one engine has a problem and it is not the one asking.

And `(?u)` asks for the classes this engine now reads anyway, so it is taken rather than refused. The other three flag letters RE2 has never heard of still mean something. `(?x)` and `(?a)` are gaps here. `(?L)` never arrives, because Python will not take it on a pattern made of text and the parser is Python's grammar, so the caller is told the grammar cannot read the pattern, which is what pandas says too.

Each of those had been written as a refusal in RE2's voice, and each of them now says which engine is complaining. That distinction is the one document 77 section 2 built the gap flag for, and this is the first slice where both sides of it are populated.

## 9. What is not here yet

The three methods are not wired. `METHOD_FINDALL`, `METHOD_EXTRACT` and `METHOD_EXTRACTALL` are not in `method.mojo`, and none of the three has an accessor door, because each wants a shape the library does not have: `findall` wants a list column, `extract` wants a frame, and `extractall` wants a frame with a MultiIndex. The engine underneath them is this document and the doors above them are the next slice.

The constructs are the rest of the router. A lookaround, a backreference, a conditional, an atomic group and a possessive quantifier are the five things a pattern can hold that send it to Python upstream, and this engine has none of them. Until it does, the router sends patterns to an engine that refuses them, which is a gap on the board rather than a wrong answer, but it is the gap that makes the router worth having and it is still open. A backreference in particular cannot be answered in linear time in general, so it is a different machine rather than a missing instruction.

Case folding is 678 patterns of the hold out and is a table rather than an engine. It is also two tables, because the two engines fold different alphabets, and section 4 is the reason to expect them to disagree about which.

Scoped flags carried on the node, the ascii flag, verbose mode, and an RE2 grammar front end are all unchanged from document 80 section 10. The front end is still the largest single gap for the RE2 half and is now measurably not a gap for this half.

The counting scan and the replacing scan are Arrow's, and Python's own scan is a third one. `re.finditer` does not re-slice the text, moves in characters, and treats an empty match after a non-empty one differently again. `Machine.counts` implements Arrow's rules and is correct for `str.count`, and nothing yet implements Python's, because nothing yet needs it. `findall` will.

## 10. Observations to file upstream

The same class letter in the same accessor means two different sets of characters depending on which method was called, and nothing in the documentation for either method says so. `str.count(r"\w")` and `str.findall(r"\w")` on the same row differ by one on `café` and by three on a row of fullwidth digits.

`ArrowStringArray._str_extract` and `ArrowExtensionArray._str_extract` are two different implementations with different refusals. The first compiles the pattern with `re` and loops in Python. The second calls `pc.extract_regex`, raises `NotImplementedError("Only flags=0 is implemented.")` for any flags, and raises `ValueError` for a pattern with no symbolic group name in it. So the same call on the same data answers or raises depending on whether the column is `str` dtype or `ArrowDtype`, and the dtype is the thing the caller is least likely to be thinking about.

`str.rsplit` has no `regex` parameter and `str.split` does. Section 3 has it.

The rest of the list is in document 76 section 12, document 77 section 9, document 78 section 12, document 79 section 10 and document 80 section 11, and is unchanged by this slice.
