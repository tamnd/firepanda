# 68. A table is not a small replace

## 1. The name that looks like the one before it and is not

`str.translate` takes a table and swaps single characters out of it, and read quickly that is `str.replace` with several patterns at once. Every part of that reading is wrong, and the two places it is wrong are the two places this document spends most of its time. A key here is always exactly one character, so nothing is ever searched for and no match can overlap another, where `replace` has to find a pattern of any length and then decide what happens to the bytes after it. And every key is applied in the same pass, where a mapping given to `replace` is several replacements run one after the other over the output of the last one. The difference is easy to state and easy to see: the table that sends `a` to `b` and `b` to `a` really swaps them, and the same pair handed to `replace` turns every one of them into `a`.

That second difference is the whole reason this needs a kernel of its own rather than a loop over the one that was written last week. A single pass over the input with a lookup per character is not an optimisation of running the table entry by entry, it is a different answer, and it is the answer pandas gives.

## 2. Where pandas gets its rule from, which is not Arrow this time

The four names before this one all needed an argument about which of pandas' three string backends was answering, because the answer changed depending on whether Arrow or Python was doing the work. This one does not. pandas hands the table straight to Python's own `str.translate` and reads back whatever comes out. Python looks a character up by its ordinal, with `table[ord(character)]`, and leaves the character exactly as it was whenever that lookup raises a `LookupError`. There is no pyarrow kernel in the path at all, so there is no backend split to measure and nothing in this document about which of the three pandas is holding the column in.

That has one pleasant consequence and one awkward one. The pleasant one is that the rule is completely specified by a language that is completely specified: everything below was measured against pandas 3.0.5 and every measurement agreed with reading Python's documentation, which is not something the last four names could say. The awkward one is section 5.

## 3. Everything the rule actually says, measured

A key is a code point ordinal, an integer. A value may be an integer ordinal, a string of any length including the empty one, or `None`.

`None` and the empty string are the same request. Python documents a `None` value as deleting the character and an empty string as replacing it with nothing, and those are two names for one operation. This was measured rather than assumed, and it is why the crossing into the kernel carries no third case for a delete: the Python side turns a `None` into an empty string before it builds the column and the kernel never learns which spelling the caller used.

A value may be longer than one character, so `{ord("a"): "XY"}` on `"ab"` gives `"XYb"`. The output is never rescanned, so `{ord("a"): "aa"}` on `"ab"` gives `"aab"` and not an infinite loop, and `{ord("a"): "ba", ord("b"): "c"}` on `"a"` gives `"ba"` and not `"bc"`. This is the same one pass rule as the swap, seen from the other side.

A key the table does not hold leaves its character alone, which is the `LookupError` branch.

A key that is not an integer never matches, and neither does a negative one or one at or above `0x110000`, because no character has that ordinal. pandas does not refuse these, it simply never finds them, so `{"a": "X"}` is a table that does nothing at all and `{"a": "X", ord("b"): "Y"}` is a table that does exactly half of what it looks like it does. This library drops such keys rather than refusing them, which is the exact behaviour and not a lenient reading of it.

A value out of range raises `ValueError("character mapping must be in range(0x110000)")` and a value of some other type entirely raises `TypeError("character mapping must return integer, None or str")`. Both sentences are reproduced here word for word, because a caller who has just read one of them in pandas should not have to learn a second phrasing.

An empty table hands every row back unchanged. A missing row stays missing.

## 4. Two seats for a lookup, and a fast lane for the common table

The kernel keeps the table in two places and every key lands in exactly one of them. The first is a direct array of 128 slots indexed by the ordinal, which covers ASCII. The second is an ordered list of the keys above 127 with a binary search over it. This is the same split `_in_class` already makes in the same file for the same reason: almost every table anyone writes is ASCII, the direct array turns the lookup into a single load, and the list keeps the rest correct without making the common case pay for it.

There is one more lane on top of that. If the table holds no key above 127 and the row itself is entirely ASCII, the kernel walks the row byte by byte rather than decoding code points, because in that case a byte is a character and the decode buys nothing. Any table with a wide key, or any row with a non-ASCII byte in it, takes the general path, which decodes the row into code points and looks each one up.

The general path is why this kernel is serial rather than split across morsels, the same reason `replace` was: the length of the output is not known before the kernel runs, so there is nowhere to write morsel three until morsels one and two have finished. Text out means one pass and a builder, and that is now the shape of every name in this accessor whose answer is text.

A scalar twin sits in `scalar.mojo`, as it does for every kernel here, and it is deliberately stupid: it walks the table from the front for every single character and has no seats, no ordering requirement and no ASCII lane. If the two ever disagree then the bug is in the seats and not in the rule, which is exactly what a twin is for. Both are run on every case in the Mojo tests and asserted to agree before either answer is checked against what it should be.

## 5. The awkward consequence, and the one refusal in this name

pandas takes any object with a `__getitem__`, because it never does anything with the table except subscript it. A list works. A string works, since `"abc"[97]` would raise an `IndexError` which is a `LookupError`. A user written class with a `__getitem__` works. A mapping is only the usual case and not a required one.

This library takes a mapping and refuses everything else. The reason is the crossing. Serving the general case means holding on to a Python object and asking it about every character of every row, which is a Python call per character, and avoiding exactly that is what going into a kernel is for. A table of ten entries against a column of a million rows would be a million calls back into the interpreter to answer a question that the two seats answer in a single load.

So this is a refusal rather than a slow answer, and that is a deliberate choice about which kind of wrong the board should see. A refusal reads as a gap, which is true. Answering it at interpreter speed would read as a pass, which would also be true, and would hide the fact that the fast path had been abandoned. Of the two ways to be honest, the one that shows up on the board is better. The refusal is an `UnsupportedError` with a message that says a mapping is what this takes and that the alternative would have to be read one character at a time.

`str.maketrans` builds a mapping in all four of its forms, and it is how a table is written in practice, so the refused shapes are genuinely the unusual ones. That is an argument for the choice and not a defence of it.

## 6. A fourth door, and why it is not the thing document 07 warns about

Every other name in this accessor reaches its kernel through one of three module level functions in `firepanda/py/text.mojo`, sorted by the shape of the answer: one for names that give back text, one for names that give back a flag, one for names that give back a number. Document 07 is clear that adding a door per method is how a binding surface turns into a hundred entry points that all do the same thing, and the three doors exist so that the twenty three text names share one crossing.

`translate` gets a door of its own anyway, and the reason is not the shape of its answer, which is text like the other twenty two. It is the shape of its argument. Every one of those twenty two takes scalars, and the three doors take scalars, and a table is not a scalar. There is no way to fold a table into a string and unfold it on the other side, because a replacement may hold any character at all and so no character is available to separate one entry from the next. The table arrives as two columns, keys and values, for exactly the reason `is_in` already takes a column: a column is how a variable number of values crosses.

So the rule the three doors follow is about answer shape, this argument is a different shape of input entirely, and this sits outside the rule rather than being an exception to it. A fourth door for a fourth answer shape would be the drift document 07 is about. A door for the one argument in the namespace that is column sized is a different claim, and there is only one such argument.

The table is sorted on the Python side rather than in the kernel, and the kernel requires the keys to arrive in order and says so if they do not. A caller has one table and any number of columns, so sorting once where the table is built is less work than sorting again on every call, and the check in the kernel means the requirement is stated rather than assumed.

## 7. What the board says

A `strings/translate` case was added to the corpus against the unicode and ascii string frames, and the driver arm hands the same table to both sides. The separation check was run before arming, as it is every time now, and it came back saying the case is distinguishable from every other case already in the section, which is more than could be said for `strings/replace-n` the last time round.

Nothing here went into the divergence registry. That is the second name in a row with nothing to register, after four in a row that each had something, and the reason is section 2: with no Arrow in the path there was no second opinion available for pandas and this library to disagree about.

## 8. What is left

Fourteen names in the accessor are still unwritten and the blockers have not moved. Eight of them need a column type that holds a list per row, which is the single largest thing missing from this namespace and is now the blocker on more than half of what is left. Four of them need the regular expression engine. `normalize` needs the Unicode normalization tables, which is a data problem rather than an engine one.

The case folding kernel is still owed to five places and has not moved either: `contains`, `match`, `fullmatch` and `replace` all refuse `case=False`, and `casefold` holds the mapping half of what they need. What is missing is a search that folds both sides as it goes rather than folding the whole column into a copy first.

One thing this name shares with something outside the accessor is worth writing down. The label half of `rename` wants a per value mapping over a column, which is the same shape as this: a table of keys and values, a lookup per element, one pass, no rescanning. The two seats do not transfer, since labels are not single characters and cannot be indexed by ordinal, but the argument about how a table crosses into a kernel does, and when that kernel is written it should cross the same way.
