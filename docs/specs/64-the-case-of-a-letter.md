# The case of a letter

## 1. Five names and one table underneath

`upper`, `lower`, `isspace`, `islower` and `isupper` are the first five names of the `str` accessor that are about case rather than about position, and they ship together because they all rest on the same thing. Somewhere there is a table that says what the other case of a character is and which characters have a case at all, and every one of the five is a walk over a row asking that table a question. Which table it is turns out to be the whole content of this document, because the answer to every other question here is short.

The short answers first. A missing row stays missing through `upper` and `lower`, which is pandas and is what every other kernel in the library does. A question about case answers False on a row with no cased character in it, so a row holding `42` is neither lower case nor upper case, and False on an empty row, because Python's rule is that all of nothing is not enough. The three questions answer a column of bools and the two rewrites answer a column of text, which is what picks which of the three doors in `text.mojo` each of them comes through. None of the five takes an argument.

## 2. Case is not a byte, and it is not quite a character either

Document 30 and the header of `chars.mojo` are about one distinction, which is that a position in a string is a character and not a byte. Case needs that distinction and then needs one more.

`ß` raises to `SS`. One character in, two out, and three bytes in against two bytes out, so neither count is preserved. A capital I with a dot over it lowers to a small i followed by a separate combining dot, which is one character in and two out again. Greek letters with an iota written underneath raise to two and sometimes three characters. So a case change is not a byte for a byte rewrite, which nobody would write, and it is also not a character for a character rewrite, which is the shape anybody would reach for first and which would be wrong on every example in this paragraph.

That is why the two rewriting kernels build a new column with `StringBuilder` rather than mapping the payload in place, and why the tests include a row that comes back longer than it went in rather than only rows that do not.

## 3. The walk is borrowed and the bytes are checked first

The standard library can already raise and lower a `StringSlice`, and can already answer the three questions, so the kernels hand each element over rather than carrying a case table of their own. That is the one place in `chars.mojo` where the walking is not ours.

It costs one check. The counting kernels in that file walk the bytes themselves and stop at the end of the element whatever the bytes say, which is how a column of invalid UTF-8 gets a deterministic answer instead of an error. The borrowed walk trusts a lead byte instead, and a truncated two byte sequence at the end of an element takes the first byte of the next element with it, because the payload of a text column is one buffer with the elements laid end to end. Reading past the end of an element is the one thing the header of that file promises never happens.

So an element is checked for being well formed UTF-8 before it is handed over. One that is not is copied through unchanged by the two that write text and answers False to the three that ask a question, which is the only pair of answers available: there is nothing to change the case of, and it is not text to be asked about.

## 4. The data underneath is not the data CPython has

pandas answers these five out of CPython, so the target is Python's Unicode data. The standard library here carries an older and smaller copy, and the difference was measured rather than guessed at, by walking all 1111998 code points through both.

For the three questions, the counts are these. Python calls 29 code points whitespace and this library calls 12, and the 17 it does not are U+001F, the no break space, and the space separators from U+1680 through U+3000. Python calls 2569 code points lower case and this library calls 1489, with 1084 that Python has and it does not, being 816 ordinary lower case letters, 266 modifier letters and 2 others, and 4 that it has and Python does not, which are the four title case digraphs. Python calls 1978 code points upper case and this library calls 1433, with 576 that Python has and it does not, being 498 capital letters and 78 circled and squared ones, and 31 that it has and Python does not, which are the title case letters again.

For the two rewrites, Python has a case mapping for 2981 code points and this library has one for 2918, and they disagree on 118 of them. 90 of the disagreements are about raising and 28 are about lowering. Most of the raising ones are the Greek letters with an iota underneath, where Python writes out the two or three character sequence that the Unicode data calls the full mapping and this library gives the single character that the simple mapping gives. The rest on both sides are letters added to Unicode after the copy here was made, including a whole alphabet.

There is one more difference that is not in any of those counts, because it is a rule rather than a table. A Greek capital sigma at the end of a word lowers to a different letter than one in the middle of a word, and Python applies that rule while this library does not, so `ΟΔΟΣ` lowers to `οδος` there and `οδοσ` here.

## 5. One of them is corrected here

The capital I with a dot over it is U+0130 and it is the one difference in that list that reaches a Latin alphabet. It is a Turkish letter, it is in ordinary Turkish text, and it is in the conformance corpus, in the word `İstanbul`. Python lowers it to a small i followed by a combining dot above, so that the dot the capital carries is not lost. The standard library here lowers it to a plain small i and the dot is gone.

So the kernel writes it out itself. Before lowering, every U+0130 in the element is replaced by the two characters it should become, and the result is lowered as normal, which leaves both of them alone because neither has a case mapping. The check for whether the element holds one at all is a byte pair test rather than a character walk, which is sound because the first byte of U+0130 is 0xC4 and no continuation byte is 0xC4, so the pair can only ever be the letter itself.

One correction rather than a hundred is a deliberate line and it is worth saying where the line is. This one is the only one of the 118 that a caller holding text in a Latin script can meet, the fix is exact rather than approximate, and it costs a scan the lowering was going to do anyway. The others need data, not code.

## 6. Why the rest is left, and what would close it

The honest description of the remaining difference is that it is small in the places most data lives and real in a few scripts. Every ASCII row, every Latin 1 row except the ordinal indicators, and every accented Latin letter agrees with Python exactly. A Greek row can disagree on a final sigma, a row holding a modifier letter can disagree about being lower case, and a row holding a character added to Unicode in the last few years can disagree about having a case at all.

Two of those are asserted in the tests rather than worked around, next to the no break space, so that the day the standard library's data is replaced the tests fail and somebody has to come and read this document instead of quietly gaining a behaviour change.

What would close it is firepanda carrying its own case data, generated from the Unicode files into a table this repository owns, with the full mappings and the conditional rules that the standard library leaves out. That is a tool, a generated source file, a check that the generated file is current, and a decision about how many megabytes of table a dataframe library should carry, which is a piece of work in its own right rather than a paragraph at the end of this one. It is named here so that it is on the list.

## 7. What this does to the board

The three questions answer a missing row with a missing value where pandas, holding the column in its own string dtype, answers False. That is `engine/string-predicate-null`, which was registered when `startswith` and `endswith` landed and which rests on the same fact: the answer in pandas is a numpy array of bools with nowhere to put a third state, so a row that was missing and a row that was really not upper case come back the same. Held as object, pandas answers None on that row and agrees with us, which is a useful check that the difference is about storage rather than about the question.

Nothing else here is a divergence. The five names are scored on the corpus, including the frame of unicode text, and the İstanbul row in it is the reason section 5 exists.

## 8. What is left

The thirty one names of the accessor that are still missing, most of which need a regex engine, a splitter that can answer more than one column, or both. `casefold`, `title`, `capitalize` and `swapcase` are the four that are nearest, because they are the same shape as these five and need no new argument handling, and `casefold` in particular is a third case mapping rather than a variation on either of these two.

`normalize` is worth naming separately. It is the name that exists because a visible letter can arrive composed or decomposed, which is also the reason a character count can surprise somebody, and it needs the normalization tables rather than the case ones.

The `str` accessor refuses a column that is not text at the accessor rather than at the method, which is pandas, and the message names what the column holds using this library's vocabulary rather than the one `infer_dtype` uses. That was decided when the accessor was built and is unchanged by any of this.
