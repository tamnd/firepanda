# 67. The one pattern that needs no promise

## 1. The fifth name, and why it is the easy one

Document 66 shipped `contains`, `match`, `fullmatch` and `count` on a smaller promise than pandas makes, because pandas reads the argument to all four as a regular expression and there is no regular expression engine here. `str.replace` is the fifth question about a pattern and it needs no promise at all, because its `regex` argument defaults to False in pandas 3. The ordinary call is a literal replacement already. A caller who types `s.str.replace("a", "A")` is asking for a byte search and a rewrite and that is exactly what they get, with no narrowing anywhere and nothing refused on the way in.

`regex=True` goes through the same check the other four use and gets the same answer: a pattern with none of the twelve metacharacters in it means one thing and is served, and anything else is refused by name. So this name is the one place in the group where the default path and the narrow path are the same path, and the refusal only appears when a caller goes out of their way to ask for an engine.

What it cost instead was a kernel of a shape this file had not needed before.

## 2. The first kernel here whose answer is text

Every kernel in `pattern.mojo` before this one answered a mask or a number, which means the answer has a known size before the search runs and can be written into a column allocated up front and filled in parallel across morsels. How long a row comes out of a replace is not known until the search has run on that row, so the column cannot be allocated first and the parallel shape does not apply. It builds into a `StringBuilder` in a serial loop instead, which is the shape `text_repeat` in `edges.mojo` already had and which every text answering kernel in this library shares.

That is the whole cost, and it is worth naming because it is the reason this one name took a kernel rather than a call into one that existed. `match` is `starts_with` and `fullmatch` is equality, and both were already written; a replace is not a search wearing a different name.

Matches do not overlap, for the same reason and with the same rule as the count: the cursor moves past the whole pattern after a hit rather than by one byte, so `replace("aa", "X")` on `aaaa` is `XX` and not `Xa` or anything else. pandas agrees, and a regular expression engine scanning for successive non overlapping matches agrees too, so there is nothing to choose.

## 3. An empty pattern, counted in two different units by one accessor

Document 66 section 5 reported that `Series.str.count("")` on a row holding one accented letter answers the number of bytes plus one rather than the number of characters plus one, and called it an implementation detail of Arrow that became pandas' answer. Replacing with an empty pattern is the other half of that same argument and it goes the other way.

`pandas.Series(["héllo"], dtype="str").str.count("")` is 7, which is six bytes and one. `pandas.Series(["héllo"], dtype="str").str.replace("", "-")` is `-h-é-l-l-o-`, which is six dashes for five characters and one. Same accessor, same row, same argument, two different units.

The reason is not a decision anybody made about text. `pyarrow.compute.replace_substring` does not terminate when the pattern is empty. That is measured and not inferred: a single row holding two ASCII characters was still running after forty five seconds at a steady twenty nine percent of a core, and a non empty pattern on the same column returns immediately. It is a known bug upstream, apache/arrow#39149, and pandas carries an explicit guard for it in `pandas/core/arrays/_arrow_string_mixins.py` with the issue number in the comment, which falls back to Python's own `str.replace` for that one case. Python counts characters. `count` has no such guard because `count_substring` terminates fine, so it stays in Arrow, and Arrow counts bytes.

This library gives both answers, because both are pandas' answers and the premise on the front of the project is that a program keeps running. It is written into the kernel docstring, the scalar twin, the Mojo test, the Python test and one board case each, and the Python test asserts the two numbers next to each other in one function on purpose. Either one on its own reads like a bug. The two of them together, with the reason attached, read like what they are.

It is worth saying plainly that the difference is unstable in a way the byte counting on its own is not. If pyarrow fixes the hang and pandas drops its guard, `replace("")` will start counting bytes and this library will have to follow it. That is a risk of following pandas rather than a reason not to, and it is the kind of thing a divergence registry cannot hold because today there is nothing to disagree about.

## 4. What `n` is allowed to be

pandas spells the limit `n` and reads three ranges out of it. A negative number means every match, which is the default at minus one. Zero means no match at all and hands the row back untouched, which is the one value a reader might expect to mean the same as the default. A positive number means that many matches counted from the left, and a number larger than the row has matches is the same as the default.

All three are the same rule inside the kernel, which carries a counter that starts at the limit, stops when it reaches zero and never decrements when it started negative. Zero is checked before the loop rather than inside it, so a row that is handed back whole is handed back without a search running over it.

The empty pattern obeys the limit too, and this is the one corner where my own expectation was wrong and the measurement corrected it. `"".replace("", "-", 2)` in Python is `"-"`, not `""`: an empty row has exactly one place to put the replacement and a limit of two does not invent a second. The kernel had it right and the first test written against it did not.

## 5. Four refusals, and one of them is a gap rather than a shortfall

A callable `repl` with `regex=False` is a `ValueError` in pandas and needs an engine when `regex=True`, so it is refused here. A `repl` that is neither a string nor callable is a `TypeError` in pandas and is one here. A pattern holding a metacharacter with `regex=True` is the refusal document 66 describes. None of those three are a narrowing.

The fourth is. pandas honours `case=False` on `replace`, and it is worth being precise about how, because the other four names in the group do not get the same treatment. pandas escapes the pattern with `re.escape`, adds `re.IGNORECASE` and runs the result through the object path, so `["Abc", "abc", "ABC"]` with `replace("a", "X", case=False)` answers `['Xbc', 'Xbc', 'XBC']`. That is measured. This library refuses it, along with a non zero `flags`, for the reason document 66 section 6 gives: ignoring an argument that changes the answer is the worst thing a compatibility layer can do, and a case folding search is a kernel that does not exist yet.

So `case=False` on `replace` is a gap and not a divergence. The board reads it as a name that does not resolve rather than as a wrong answer, which is the honest reading, and it is one case folding kernel away from being neither.

## 6. A dictionary is several replacements, in the order the dictionary has them

pandas lets `pat` be a mapping, in which case `repl` must be absent and each pair is applied in turn to the result of the last. `{"a": "X", "b": "Y"}` on `abcabc` is `XYcXYc`, and it is sequential rather than simultaneous, which matters when one replacement's output contains another's pattern. That is loop in the Python layer over the pairs, each one a full pass, and it does not reach the kernel as anything special. It is written where it is because the mapping is a Python object and unpacking it before the crossing keeps the Mojo side taking two strings and not a table.

## 7. What the board says

Nine runs are armed and every one of them passes. The board moves from 2632 passing runs, up from 2622, with unimplemented falling from 1745 to 1738 and divergent staying exactly where it was at 126. The failures do not move. The registry is still 29 entries, and unlike the last three slices in this accessor that is not because a new disagreement joined an old entry: there is no disagreement. `replace` is the first name in the `str` accessor to arrive with nothing to register, including on the null heavy frame, because it answers text and a missing row stays missing on both sides rather than turning into a False the way the three mask questions do.

The strings section still reads 14 of 58 callables at L3. `str.replace` does not count as fully conforming, because `strings/replace-regex` and `strings/replace-backreference` both ask for an engine and stay unimplemented, and a callable with an unimplemented case is not at L3 no matter how many of its other cases pass.

Two things were added to the corpus rather than answered by the library, and both are disclosed here for the same reason document 66 disclosed three. `strings/replace-empty` is new, and it is on the board so that the unit difference in section 3 is visible as two rows of a table rather than only as a paragraph in a document. And `strings/replace-n` was moved off the ascii frame, or rather given a second frame beside it, because the ascii frame is the alphabet cut at every length from nothing up to twenty letters and therefore holds no row with the letter `a` in it twice. A limit of one and a limit of every match answer identically on every row of it, which means the case that exists to test `n` could not fail if `n` were ignored entirely. The pattern frame has rows holding `a` two and three times and separates them. The oracle goes from 4502 runs to 4505 and still answers every one.

## 8. What is left

Fifteen `str` names. `cat`, `join`, `partition`, `rpartition`, `split`, `rsplit`, `get_dummies` and `wrap` want a column of lists, which this library does not have and which is now plainly the largest single thing standing between this accessor and the end of it. `extract`, `extractall`, `findall` and `replace` with a real pattern want the engine. `normalize` wants the Unicode normalization tables. `decode`, `encode` and `translate` are three small separate problems, and `translate` is the nearest of the three: it is a per character mapping over a table the caller supplies, which is the same hash table shape the label half of `rename` has been waiting on since document 50.

The case folding kernel is worth calling out on its own, because it is now owed to five places rather than one. `contains`, `match`, `fullmatch` and `replace` all have a `case` argument they refuse, and `casefold` already has the mapping half of the problem solved. What is missing is a search that folds both sides as it goes rather than folding the column into a copy first, and a copy would be correct and would also double the memory the accessor touches, which is the wrong trade for a library whose premise includes the resource figure.
