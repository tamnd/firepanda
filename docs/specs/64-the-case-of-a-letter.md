# The case of a letter

## 1. Five names and one table underneath

`upper`, `lower`, `isspace`, `islower` and `isupper` are the first five names of the `str` accessor that are about case rather than about position, and they ship together because they all rest on the same thing. Somewhere there is a table that says what the other case of a character is and which characters have a case at all, and every one of the five is a walk over a row asking that table a question. Which table it is turns out to be the whole content of this document, because the answer to every other question here is short.

The short answers first. A missing row stays missing through `upper` and `lower`, which is pandas and is what every other kernel in the library does. A question about case answers False on a row with no cased character in it, so a row holding `42` is neither lower case nor upper case, and False on an empty row, because the rule is that all of nothing is not enough. The three questions answer a column of bools and the two rewrites answer a column of text, which is what picks which of the three doors in `text.mojo` each of them comes through. None of the five takes an argument.

## 2. pandas has two answers and they are not the same answer

The interesting part of this slice was finding out what we were supposed to be copying. pandas has two ways of holding text, and they do not agree with each other about case.

A column held the way pandas holds text by default is an Arrow column, and `str.upper` on it is an Arrow compute kernel. A column held as object is a column of Python strings, and `str.upper` on it is Python's own method called once per row. Arrow uses what Unicode calls the simple case mappings, which are one character in and one character out, always. Python uses the full mappings, where `ß` raises to `SS` and a Greek letter with an iota written underneath raises to two or three characters.

So `pd.Series(['straße']).str.upper()` answers `STRAẞE` and `pd.Series(['straße'], dtype='object').str.upper()` answers `STRASSE`, out of the same pandas on the same afternoon. `İstanbul` lowers to `istanbul` in the first and to `i̇stanbul`, which is an i and a separate combining dot, in the second. `ΟΔΟΣ` lowers to `οδοσ` in the first and applies the final sigma rule in the second.

This library follows the first, for two reasons. It is what a caller gets from `pd.Series([...])` without asking for anything, and it is what the conformance corpus is scored against. And a case change that cannot change the length of a row is a materially different operation from one that can, which matters to a column of fixed width views far more than it matters to a list of Python objects.

That decision is written into `python/tests/test_str_case.py` as a test that asserts both pandas answers side by side, so that the choice is visible in the suite rather than implied by which dtype somebody happened to write.

## 3. Case is still not a byte

Document 30 and the header of `chars.mojo` are about one distinction, which is that a position in a string is a character and not a byte. Case needs that distinction whichever mapping is being used, because the two cases of a character are not the same number of bytes: `é` and `É` are both two bytes, but `ı` is two and `I` is one, and the Turkish pair alone is enough to rule out any rewrite that works in place.

What the simple mappings buy is that the count of characters is preserved even though the count of bytes is not. The kernels still build a new column with `StringBuilder` rather than rewriting the payload where it lies, and the tests still include a row whose length in bytes changes.

## 4. The walk is borrowed and the bytes are checked first

The Mojo standard library can already raise and lower a `StringSlice` and can already answer the three questions, so the kernels hand each element over rather than carrying a case table of their own. That is the one place in `chars.mojo` where the walking is not ours.

It costs one check. The counting kernels in that file walk the bytes themselves and stop at the end of the element whatever the bytes say, which is how a column of invalid UTF-8 gets a deterministic answer instead of an error. The borrowed walk trusts a lead byte instead, and a truncated two byte sequence at the end of an element takes the first byte of the next element with it, because the payload of a text column is one buffer with the elements laid end to end. That was measured with a probe rather than reasoned about: a one byte span holding 0xC4 came back having consumed the byte after it. Reading past the end of an element is the one thing the header of that file promises never happens.

So an element is checked for being well formed UTF-8 before it is handed over, using the validating constructor rather than the unchecked one. One that is not is copied through unchanged by the two that write text and answers False to the three that ask a question, which is the only pair of answers available: there is nothing to change the case of, and it is not text to be asked about.

## 5. Where the borrowed data is not Arrow's data

The standard library's case data is close to Arrow's and not the same, and the difference was measured rather than guessed at, by walking all 1111998 code points through both sides.

For the two that rewrite a row, they disagree on 149 code points. 39 of those are the ones where the library applies a full mapping and Arrow does not, which is the `ß` case and thirty eight others, mostly Greek letters with a diacritic that Python writes out as two characters and the Latin ligatures like `ﬁ`. The other 110 are code points the library has never heard of, because they were added to Unicode after its copy of the data was made, which includes a whole alphabet added in Unicode 16.

For the three that ask a question, the gap is larger and is a different shape. Arrow calls 29 code points whitespace and the library calls 12, and the 17 missing are U+001F, the no break space and the space separators from U+1680 through U+3000. Arrow calls 2326 code points lower case and the library calls 1489, with 841 that Arrow has and it does not and 4 that it has and Arrow does not, which are the title case digraphs. Arrow calls 1928 code points upper case and the library calls 1433, with 526 Arrow only and 31 library only, the title case letters again.

## 6. The 149 are corrected and the 1400 are not

`firepanda/kernel/casefix.mojo` is the list of those 149 code points with Arrow's answer for each, generated by `tools/gen_casefix.py` and committed. `text_case` tests an element for holding one of them before handing it over, and an element that does is written out a code point at a time with the table consulted first. Every other element takes one call as before.

The test is in two parts because the first part has to be cheap. The lowest code point in the table is U+00DF, so any of them has a lead byte of at least 0xC3, and a pass over the bytes looking for one that large rules out every ASCII element without decoding anything. An element that survives that is decoded and each code point is looked up by binary search, which rules out ordinary accented text, and only an element that really holds one of the 149 takes the slow path.

The three questions are not corrected, and the asymmetry is deliberate rather than an oversight. Correcting a mapping needs a list of 149 pairs, which is a page of generated source. Correcting the questions needs the answer for 1384 code points, and answering them properly needs the category of every code point rather than a patch list, which is the general case of carrying our own Unicode tables. The line is drawn where a list stops being a list and starts being a database.

So `upper` and `lower` agree with pandas on every code point there is, and the three questions agree with pandas everywhere except on the characters in section 5. Three of those, one for each question, are asserted in `python/tests/test_str_case.py` so that the day the standard library's data is replaced a test fails and somebody has to come and read this document instead of quietly gaining a behaviour change.

What would close the rest is firepanda carrying its own case and category data, generated from the Unicode files into tables this repository owns, with a check that the generated file is current. That is a tool, a generated source file and a decision about how many megabytes of table a dataframe library should carry, which is a piece of work in its own right rather than a paragraph at the end of this one. It is named here so that it is on the list.

## 7. What this does to the board

The three questions answer a missing row with a missing value where pandas answers False. That is `engine/string-predicate-null`, which was registered when `startswith` and `endswith` landed and which rests on the same fact: the answer in pandas is a numpy array of bools with nowhere to put a third state, so a row that was missing and a row that was really not upper case come back the same. Held as object, pandas answers None on that row and agrees with us, which is a useful check that the difference is about storage rather than about the question.

Nothing else here is a divergence. The five names are scored on all four string frames, and two rows of the unicode frame are what section 2 is about: the row holding `ß`, and the row holding the `ﬁ` ligature, both of which change length under Python's mappings and neither of which changes length under Arrow's.

## 8. What is left

The thirty one names of the accessor that are still missing, most of which need a regex engine, a splitter that can answer more than one column, or both. `casefold`, `title`, `capitalize` and `swapcase` are the four that are nearest, because they are the same shape as these five and need no new argument handling. `casefold` in particular is a third mapping rather than a variation on either of these two, and it is the one name of the four where Arrow and Python differ in a way that will need the same treatment section 6 describes.

`normalize` is worth naming separately. It is the name that exists because a visible letter can arrive composed or decomposed, which is also the reason a character count can surprise somebody, and it needs the normalization tables rather than the case ones.

The `str` accessor refuses a column that is not text at the accessor rather than at the method, which is pandas, and the message names what the column holds using this library's vocabulary rather than the one `infer_dtype` uses. That was decided when the accessor was built and is unchanged by any of this.
