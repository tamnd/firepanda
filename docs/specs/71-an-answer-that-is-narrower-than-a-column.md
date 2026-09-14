# 71. An answer that is narrower than a column

## 1. The other direction off the same rule

Document 70 was about the first `str` method whose answer is wider than a column. This one is about the first whose answer is narrower. `str.cat` with nothing to concatenate against folds an entire column into a single string, and a scalar is a shape no door in `firepanda/py/text.mojo` carries, so it gets a function of its own for the same reason `partition` did.

That is worth saying twice in two consecutive documents because the two look like opposite decisions and are the same one. The rule those doors follow is that the shape of the answer picks the door. Three columns is a shape and one string is a shape, and neither of them is a column, so neither of them fits through a door built to hand a column back. `translate` remains the only genuine exception in this file, and it is an exception because what makes it different is the shape of its argument.

## 2. Two operations wearing one name

pandas spells two different things `str.cat` and picks between them by whether `others` was passed.

With `others`, it lines a second column up against this one and concatenates row by row. The answer is a column as tall as the input, a missing row on either side makes the answer missing, and the separator goes between the two pieces of each row rather than between rows.

With no `others`, it folds this column into one string. The answer is a scalar, a missing row is dropped rather than propagated, and the separator goes between neighbouring rows.

Those two share a name, a `sep` argument and a `na_rep` argument, and share nothing else. They have different answer shapes, different heights, different rules for a missing row and different meanings for the same separator. Only the second is written here.

## 3. Why the other half is refused rather than approximated

The row by row half is not hard arithmetic. It is a loop over two columns appending bytes, which is less work than the fold is. What stops it is one step pandas takes before any of that: it aligns the two columns on their labels.

So a row of `others` meets the row of this column carrying the same label, not the one sitting in the same position. Hand pandas an `others` one row shorter and it does not raise, it matches what it can and fills the rest with missing rows, which is what `s.str.cat(pd.Series(["x"]))` does on a column of four. Label alignment is not written in this library yet.

An implementation that concatenated by position would be right for every caller whose two columns happen to share an index and silently wrong for everyone else, with no error to say which one they were. That is the worst of the three available options, so `others` raises `UnsupportedError` and the message names alignment as the missing piece rather than naming the method. On the board that reads as a gap, which it is, rather than as a disagreement, which it is not.

## 4. What a missing row does, which is two answers behind one argument

The rule is that a missing row is dropped, and dropped means it takes its separator with it. A column of `["a", None, "b"]` joined by `-` is `a-b`. It is not `a--b`, which is what a join written the obvious way produces, because the obvious way replaces the row with an empty string and leaves the separator standing.

Given a `na_rep` the row is not missing any more. It becomes a row holding that text, its separator comes back, and the same column joins to `a-?-b`.

pandas decides between the two by whether the argument was supplied, which means the empty string is a real request here and not a way of spelling the default. `na_rep=""` gives `a--b`: the row survives, its separator survives, and only its text is empty. Leaving the argument out gives `a-b`. Those are different answers and an implementation that read the flag back out of the string would collapse them.

That is why the crossing into Mojo carries two arguments rather than one. `text_join` takes `na_rep` and takes `skip_missing` separately, and the Python layer is the only place that knows pandas spells the flag as the absence of the string.

## 5. The empty row, which is the same output reached by the opposite rule

An empty row is readable. It is never dropped, it keeps its separator, and `["a", "", "b"]` joined by `-` is `a--b`.

So `a--b` is the answer for an empty row under the default and the answer for a missing row under `na_rep=""`, and `a-b` is the answer for a missing row under the default. Two of those three look the same and are reached by opposite rules, which is exactly the situation where a test that only checks one of them proves nothing. There is a test in `python/tests/test_str_cat.py` and another in `tests/test_chars.mojo` that put the pair side by side for that reason.

## 6. Counting before writing

A join is the one text operation whose answer length is known exactly before anything is written. Every row's byte length is already recorded, the separator's length is fixed, and the number of separators falls out of the number of rows that survive.

So `text_join` makes two passes. The first adds up the bytes and settles how many rows survive, and the second writes into a buffer allocated once at that size. The alternative appends and lets the buffer grow, which copies everything written so far every time it doubles, and for a tall column that copying is the entire cost of the operation.

The twin in `firepanda/kernel/scalar.mojo` is the growing version, which is what anybody writes first. The two are checked against each other over four separators and both missing row rules, and what they are really being checked on is the separator count, because that is the part of a join that is wrong by one when it is wrong at all.

## 7. The refusals, which are not pandas' sentences this time

Every previous document in this series could say that a refusal here repeats the sentence pandas raises. This one cannot, because pandas does not check these arguments at all.

`sep=1` falls into `str.join` and comes back as `AttributeError: 'int' object has no attribute 'join'`. `na_rep=1` comes back as `TypeError: sequence item 2: expected str instance, int found`. Both of those name pandas' implementation rather than the caller's mistake, and the second names a row index that depends on the data. Repeating them would mean repeating an accident.

So both are refused here with `DTypeError`, which reaches Python as the `TypeError` a caller would expect, and the message says which argument was wrong. The tests assert the error type rather than the sentence, and they assert it on both libraries, so the day pandas starts checking these the test still passes and the difference is in the wording only.

`join` is a third argument that pandas does not check, and there the right answer was to do nothing. It says how to line `others` up and there is no `others` to line up, so it has no effect either way, and refusing a value pandas accepts for an argument that does nothing would be a divergence invented for tidiness. It is dropped unread.

## 8. The upstream observations

Three, none of which change anything here.

pandas does not validate `sep` or `na_rep`, and the resulting sentences name `str.join` rather than the argument. The `na_rep` one leaks a row index.

pandas does not validate `join` either. `s.str.cat(other, join="bogus")` runs to completion and answers as though `join="left"` had been asked for, even though there is an `others` there for the argument to have applied to and the documentation lists exactly four values it accepts.

`str.cat` is the only name on this accessor whose answer shape depends on an argument rather than on the name, which is what makes the two operations in section 2 hard to document and hard to type. `partition` has the same problem through `expand` and at least keeps the same number of rows either way.

## 9. What is not here

`others` in every form, which is section 3 and which needs label alignment. When alignment lands this reopens as a second function, not as a branch inside this one, because the answer shape differs.

The eleven `str` names still missing after this one: `decode`, `encode`, `extract`, `extractall`, `findall`, `get_dummies`, `join`, `normalize`, `rsplit`, `split` and `wrap`. Of those, `wrap`, `normalize` and `get_dummies` need no new column type and are the next ones reachable, though `get_dummies` answers a frame whose width comes out of the data and `normalize` wants the Unicode normalization tables. Four want the regular expression engine, which is now the single largest blocker on this accessor. `join` and the default form of `split` and `rsplit` want a list column, and `encode` and `decode` want a binary one.

`str.cat` on a frame rather than on a column, which pandas does not have either.
