# 34. The two ends of a row

Status: implemented. Nine `str` methods, one kernel file, no new arguments at the boundary.

## 1. Why these nine are one piece of work

pandas puts fifty seven names on `s.str` and they do not arrive in a sensible order. Grouping them by what a caller would call them is useless, because `strip` and `replace` both sound like editing and have nothing in common underneath. Grouping them by what the kernel has to know works much better, and under that reading nine of the fifty seven fall out together: `strip`, `lstrip`, `rstrip`, `pad`, `center`, `ljust`, `rjust`, `zfill` and `repeat` never look at the middle of a row. They take characters off the ends or they put characters on the ends, and what is between the ends passes through untouched.

That is worth being a file. The kernels next door parse, search and compare, and every one of them needs to know something about the contents of a row. These nine need to know two things only: where the character boundaries are, and how many characters there are. `firepanda/kernel/edges.mojo` is the four functions that need exactly that and nothing else, and `repeat` is in there for the duller reason that it is the other method which changes how long a row is without reading it.

Four kernel functions cover nine pandas names. `text_strip` takes the two ends as flags and covers three names, `text_pad` takes the two sides as flags and covers four, and `text_zfill` and `text_repeat` are one each. The compression is real rather than clever: `lstrip` genuinely is `strip` with one flag turned off, and writing it out three times would have given three places for the same bug to live.

## 2. A width is a count of characters

Everything here counts characters, which is the rule `chars.mojo` already argued for and which is inherited rather than rediscovered. Python counts code points, pandas inherits that from Python, and a column of accented letters centred in twelve has to come out twelve characters wide and not twelve bytes wide. The byte oriented kernels are no help at all, and a padding implementation that reached for `len(bytes)` would pass every ASCII test and be wrong on the first row of real data.

The cost is one pass over the bytes of each row to count its characters before anything is decided. That is the same cost `str.len` already pays and it is not avoidable: the number of characters in a UTF-8 row is not recorded anywhere and has to be counted. It is cheap, because counting characters is counting the bytes that are not continuation bytes and that is a comparison per byte with no branching worth the name.

## 3. What counts as whitespace

`strip` with nothing to strip removes whitespace, and Python's idea of whitespace is not ASCII's and is not the Unicode White_Space property either. It is that property plus the four C1 file, group, record and unit separators, twenty nine code points in all, and it is the same set `str.isspace` answers True for.

The table is written out by hand in `is_python_space` rather than being derived from a property database, and the reason is arithmetic. Twenty nine entries is small, it has not changed since Unicode 4.1, and a derivation would need a general property table that this library does not carry and would not carry for this alone. The risk with a hand written table is a typo, so `tests/test_edges.mojo` checks every one of the twenty nine and also checks six code points that look like they belong and do not: `0x180E`, the Mongolian vowel separator, stopped being whitespace in Unicode 6.3, and `0x200B`, the zero width space, never was one despite the name.

## 4. Why the odd character goes where it goes

Padding both sides of an odd gap leaves one character over and something has to decide which side it lands on. That decision is not in pandas' documentation, not in Python's documentation and not in the Unicode standard. It is in CPython's `unicode_center`, as one line:

```c
left = margin / 2 + (margin & width & 1);
```

That expression is reproduced rather than approximated, because every obvious reading of splitting a gap in half gets some case wrong. `"a".center(4, ".")` is `".a.."`, so the spare character goes right. `"ab".center(5, ".")` is `"..ab."`, so the spare character goes left. The two disagree, and they disagree on a term involving the width, which is not something anybody arrives at by reasoning about strings. It is arrived at by reading the C, and it is being written down here so that the next person does not have to.

## 5. Why nothing new crosses the boundary

Document 13 sets the arity ceiling: a bound method gets seven real positional arguments after `py_self`, and the `str` accessor's text door already spends six of them on a column, a word, a string, two optional positions and a step. Nine new methods with widths, fill characters, sides and strip sets would have blown through that ceiling easily if each argument had been given a slot of its own.

None of them got one. The side folds into the word, so `pad_left`, `pad_right` and `pad_both` are three values of `kind` rather than a fourth argument, and the same trick gives `strip`, `lstrip`, `rstrip` and their three `_chars` spellings. The width and the repeat count ride in the `start` slot, which already carries a number and is already allowed to be absent. The fill character and the strip set ride in the `arg` slot, which already carries a prefix or a replacement.

Six new words in the `kind` vocabulary cost nothing at runtime, because the dispatcher was already a chain of string comparisons and six more is six more comparisons on a call that is about to touch every row of a column. Eleven words now reach the text door and the chain is still shorter than the work it dispatches to.

The one thing the folding gives up is that an absent number is now ambiguous at the boundary: the `start` slot is genuinely absent for `slice(None)` and must never be absent for `zfill`. `_whole` in `firepanda/py/text.mojo` refuses the absence rather than reading it as a zero, which matters because a zero width would have made `zfill` quietly hand the column back unchanged, and a method that silently does nothing is worse than a method that raises.

## 6. Which side of the boundary each refusal lives on

pandas refuses four things here and every one of them is refused in Python, before anything crosses:

| What | What pandas raises | Where it is checked |
| --- | --- | --- |
| `fillchar` that is not a string | `TypeError` | `StringMixin._padded` |
| `fillchar` that is not one character | `TypeError` | `StringMixin._padded`, and again in the kernel |
| `width` that is not a whole number | `TypeError` | `StringMixin._width` |
| `side` that is not one of three words | `ValueError` | `StringMixin._padded` |

The order matters and is pandas' order, because a call that gets two of them wrong should report the same one pandas would have reported. The fill character is checked twice, in Python and in the kernel, and that is not redundancy: the kernel is reachable from Mojo without any Python in the picture and has to check for itself, while the Python check exists so that the message a Python caller reads is the message pandas would have given. The two messages differ, because the kernel counts the characters it found and pandas just says `str`.

`width` refuses a bool as well as refusing a string, even though Python is happy to call a bool an integer. `zfill(True)` is a mistake every time it is written.

## 7. The one thing that is not written yet

`str.repeat` in pandas is two methods wearing the same name. Given a number it repeats every row that many times, and given a sequence it repeats each row by its own count, which means a column of counts has to cross the boundary rather than a number. The first form is here and the second says so, with a `NotImplementedError` that names the difference.

Refusing it is better than the alternatives. Repeating by the first count in the sequence would be wrong on every row after the first, and quietly ignoring the sequence would be wrong on all of them. Document 07's rule is that a name must not resolve and then refuse, and this is the exception the rule allows for: the name `repeat` resolves and answers, and one particular shape of argument is the thing that is missing.

## 8. Two rules that look like details and are not

A strip set is a set of characters and not a prefix. `"abXba".strip("ab")` is `"X"` and not `"Xba"`, because every leading and trailing character that appears anywhere in the set comes off, in any order and any number of times. Everybody has been bitten by this at least once and usually in production, so it has a test with the assertion spelled out rather than only being compared against pandas.

An empty strip set and an absent strip set are two different requests. `strip()` removes whitespace and `strip("")` removes nothing at all, which follows from pandas handing both straight to Python. That is why the absence picks the word the crossing carries rather than being filled in with a default set on the way, and it is why there are six strip words at the boundary instead of three.

## 9. What it is worth

Nine names on the board, and the cases in `firepanda-compat` that they arm are `strings/strip`, `strings/lstrip`, `strings/rstrip` and their three `-chars` spellings, `strings/pad-left`, `strings/pad-both`, `strings/center`, `strings/ljust`, `strings/rjust`, `strings/zfill` and `strings/repeat`. Thirteen case ids across four string corpora, which is roughly forty five board runs, and eight of the thirteen are L3 cases because they cover a named parameter rather than only a default.

The string section had one hundred and eighty two unarmed runs before this and it is the largest single block of unarmed work left. The next pieces of it are case conversion, which needs Unicode case mapping tables and is a much larger job than it sounds because of `İstanbul` and `ß`, and the ten `is*` predicates, which need Unicode category tables for the same reason. Both of those want a property table that this library does not carry, and deciding whether to carry one is the next real decision in the accessor.
