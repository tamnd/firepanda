# The case of a letter

## 1. Eight names and one table underneath

`upper`, `lower`, `capitalize`, `swapcase`, `casefold`, `isspace`, `islower` and `isupper` are the names of the `str` accessor that are about case rather than about position, and they belong together because they all rest on the same thing. Somewhere there is a table that says what the other case of a character is and which characters have a case at all, and every one of the eight is a walk over a row asking that table a question. Which table it is turns out to be the whole content of this document, because the answer to every other question here is short.

The first five landed together, then `capitalize` and `swapcase` followed once the first five had been corrected and there was a table worth building on, and `casefold` came last because it is the one name here that is not answering the same table at all. Sections 2 through 6 are about the correction, section 7 is about the two that came after it, and section 8 is about the one that came from somewhere else.

The short answers first. A missing row stays missing through all five names that write text, which is pandas and is what every other kernel in the library does. A question about case answers False on a row with no cased character in it, so a row holding `42` is neither lower case nor upper case, and False on an empty row, because the rule is that all of nothing is not enough. The three questions answer a column of bools and the five rewrites answer a column of text, which is what picks which of the three doors in `text.mojo` each of them comes through. None of the eight takes an argument. Four of the five rewrites give back a row exactly as many characters long as the one they were given and `casefold` is the one that does not, which is section 8.

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

For the three that ask a question, the gap is larger and is a different shape. Arrow calls 29 code points whitespace and the library called 12, and the 17 missing are U+001F, the no break space and the space separators from U+1680 through U+3000. Arrow calls 2326 code points lower case and the library calls 1489, with 841 that Arrow has and it does not and 4 that it has and Arrow does not, which are the title case digraphs. Arrow calls 1928 code points upper case and the library calls 1433, with 526 Arrow only and 31 library only, the title case letters again.

## 6. The 149 are a list and the 1384 are a table

`firepanda/kernel/casefix.mojo` is the list of those 149 code points with Arrow's answer for each, generated by `tools/gen_casefix.py` and committed. `text_case` tests an element for holding one of them before handing it over, and an element that does is written out a code point at a time with the table consulted first. Every other element takes one call as before.

The test is in two parts because the first part has to be cheap. The lowest code point in the table is U+00DF, so any of them has a lead byte of at least 0xC3, and a pass over the bytes looking for one that large rules out every ASCII element without decoding anything. An element that survives that is decoded and each code point is looked up by binary search, which rules out ordinary accented text, and only an element that really holds one of the 149 takes the slow path.

The three questions were not corrected here, and the asymmetry was deliberate rather than an oversight. Correcting a mapping needs a list of 149 pairs, which is a page of generated source. Correcting the questions needs the answer for 1384 code points, and answering them properly needs the class of every code point rather than a patch list. The line was drawn where a list stops being a list and starts being a table.

They are corrected now, in `firepanda/kernel/charclass.mojo`, which is that table and which turned out to be eleven kilobytes rather than the megabytes this section expected, because a class is dense in runs and can be held as ranges. Document 65 is the whole of it. The three questions agree with pandas on every code point there is, the three rows this section used to name as the asserted differences all agree, and the test that named them is a sweep of all 1111998 of them instead.

So every name in this document agrees with pandas on every code point, and `upper` and `lower` still reach their answer the way section 4 describes, which is unchanged by any of it. The two files underneath are different shapes for a reason: `casefix.mojo` is pairs because a mapping is a pair, and `charclass.mojo` is ranges because a class is a run.

## 7. The two names the library has no call for

`capitalize` and `swapcase` are not variations on `upper` and `lower` that the standard library can be asked for, so they are walked here in full. The reward for having built the table in section 6 is that neither of them needs anything past it, and both are exact against pandas.

Capitalising is the first character raised and every other character dropped. That is all it is, and the part worth stating is that it is not what the name suggests to most people: it has no idea what a word is, so a row of several words comes back with one capital in it, and it does not leave a character that is already a capital alone, so a row that arrived shouting comes back whispering after the first letter. It also starts at the first character rather than the first letter, so `1abc def` comes back unchanged. All three are pandas, and all three have a test.

The reason it needs nothing new is that both halves of it are the mappings from section 6, which are already right. That claim is checked rather than assumed: `tools/gen_casefix.py` runs the rule against Arrow over every code point on its own and over four thousand words built out of the cased ones, and refuses to write a table if a single answer differs.

Swapping case is the interesting one, because it has to know which case a character is already in before it can write the other one, and that is exactly the question section 6 says we are still wrong about for more than a thousand code points. It turns out not to need the question. The mappings answer it on their own: a character that lowers to something else was upper, a character that raises to something else was lower, and a character that neither raises nor lowers has no case to swap. Since the mappings are corrected, so is the classification, for free.

The exceptions are the thirty one titlecase characters. A titlecase character is in neither case, so there is no other case to write it in and Arrow leaves it exactly as it is, but both of its mappings would move it. `ǅ` raises to `Ǆ` and lowers to `ǆ`, and the right answer for swapping it is `ǅ`. So `casefix.mojo` carries a fourth table, `KEPT_BY_SWAP`, which is those thirty one code points and nothing else, and the generator derives it from Arrow rather than from a list somebody typed: a code point belongs in it if Arrow's own swapcase kernel leaves it alone while at least one of its mappings does not.

With that list the rule is exact. Both names were then run against a live pandas over every code point in Unicode on its own, 1112064 rows each, and over sixty thousand random multi character words, with no row differing.

There is one more thing worth writing down, which is that `swapcase` deliberately does not use `islower` and `isupper` even though they are sitting in the same file and would have looked like the obvious way to write it. They are the two names this document says are still wrong about 1384 code points, and a kernel built on them would have inherited every one of those. Asking the mappings instead is both cheaper, since it is a table lookup rather than a category lookup, and right.

An ASCII element does not go near any of this. `swapcase` checks for a byte with its top bit set and, finding none, flips one bit per letter, because the two cases of an ASCII letter differ in exactly that bit and nothing else in the range has a case at all.

## 8. Folding is not a case

`casefold` looks like a third case and it is not one. Nobody writes text in it and nobody reads it. Its one promise is that two rows a reader would call the same come out as the same bytes, which is what you want when you are comparing rows rather than showing them, and everything odd about it follows from that. `Straße` and `STRASSE` both fold to `strasse`, where lowering leaves the first as `straße` and the second as `strasse` and so says the two are different. The German sharp s is the row in the corpus that makes the difference visible and it is not a curiosity: it is the reason the name exists.

The price of that promise is that a folded row can be longer than the row that went in. `ß` folds to two letters, `ﬁ` folds to two letters, and 104 of the code points in the table fold to more than one, none to more than three. That is the opposite of the rule sections 2 through 6 spend their length on, where the whole point was that an Arrow backed `upper` never makes a row longer, and it is worth being clear that this is not an inconsistency anybody chose.

It is what pandas does. pyarrow has no casefold kernel. The case related compute functions it does have are `ascii_capitalize`, `ascii_is_title`, `ascii_swapcase`, `ascii_title`, `case_when`, `utf8_capitalize`, `utf8_is_title`, `utf8_swapcase` and `utf8_title`, and no folding anywhere in that list, so a pandas text column falls back to Python's own `str.casefold` for this one method while every other case method stays in Arrow. Both pandas backends therefore give the same answer here, which is the only name in this group where that is true, and Python is the oracle rather than the thing being corrected. Follow Arrow for `upper` and follow Python for `casefold` and you are not being inconsistent, you are copying what pandas actually answers in each case, which is the only rule this library has.

The table is small for a reason worth stating. 353 code points fold to something other than their lower case, and everything else in Unicode folds to exactly what it lowers to, so `casefold.mojo` is a difference table rather than a copy of the folding database. The kernel asks it first and falls through to the corrected lower case path of section 6 for everything else, which means the 149 corrections are carrying their weight a third time. A capital theta symbol is not in the fold table at all, because folding it is lowering it, and it still comes out right because the lower case path knows about it.

The table holds rows of different lengths, so it is three arrays rather than one: the code points, an offset per code point with the end on the tail, and the answers end to end. There is also no byte test on this path, which is a difference from `upper` and `lower` worth naming. Those can rule out an element by looking for a byte at or above 0xC3, since the lowest code point they correct is U+00DF, and a good deal of ordinary accented text leaves on that test. The lowest code point here is the micro sign at U+00B5, whose lead byte is 0xC2, and 0xC2 is the lowest lead byte any non ASCII character can have, so asking whether an element could hold one of these and asking whether it is not ASCII are the same question. The kernel asks the second one because it is the cheaper spelling of it, and an ASCII element folds by lowering, which is one call.

Two rows are worth having in your head. `İstanbul` folds to a plain i followed by a combining dot and then `stanbul`, which is Python's answer and is longer in characters than the row that went in, where the same row lowered by Arrow is `istanbul` with no dot at all. And the three letters of a titlecase family, `Ǆ` and `ǅ` and `ǆ`, all fold to the last of them, where `swapcase` in section 7 leaves the middle one exactly as it found it. Both of those are in the tests, because they are the two places where folding and the rest of this document visibly part company.

Verified the same way as everything else here: over all 1112064 code points on their own and over sixty thousand random words against a live pandas, with no row differing. The generator also checks the one rule the walk rests on, which is that folding a row is folding each of its characters and sticking the answers together, and it refuses to write a table if that stops holding.

## 9. `title` is the name that is not here

`title` is the last member of this shape and it is missing on purpose. It needs to know where a word starts, pandas and Arrow agree that a word starts at a character that is cased, and whether a character is cased is the category question rather than the mapping question. Measured the same way as everything else here: the boundary rule built out of the corrected mappings disagrees with Arrow on 1295 code points, all of them letters that are cased and have no case mapping at all, `ĸ` and `ƍ` and their kind.

That is not a list, it is the same table section 6 declined to build. That table is built now and document 65 is about it, so `title` no longer waits on data. Whether a character is cased is the lower class and the titlecase class together and needs nothing new. What it waits on instead is a rule about what comes before a character rather than about the character, which is the first thing in this part of the library that cannot be answered a code point at a time.

`istitle` is on the same footing, and `isalpha`, `isalnum`, `isnumeric` and `isdecimal` need four more classes of exactly the kind document 65 section 9 sizes. Those are the only names of this group still missing.

## 10. What this does to the board

The three questions answer a missing row with a missing value where pandas answers False. That is `engine/string-predicate-null`, which was registered when `startswith` and `endswith` landed and which rests on the same fact: the answer in pandas is a numpy array of bools with nowhere to put a third state, so a row that was missing and a row that was really not upper case come back the same. Held as object, pandas answers None on that row and agrees with us, which is a useful check that the difference is about storage rather than about the question.

Nothing else here is a divergence. All eight names are scored on all four string frames, and two rows of the unicode frame are what section 2 is about: the row holding `ß`, and the row holding the `ﬁ` ligature, both of which change length under Python's mappings and neither of which changes length under Arrow's.

## 11. What is left

The twenty eight names of the accessor that are still missing, most of which need a regex engine, a splitter that can answer more than one column, or both. `title` is covered in section 9 and is the nearest of them, and since document 65 it is nearest to a rule rather than to a table.

`normalize` is worth naming separately. It is the name that exists because a visible letter can arrive composed or decomposed, which is also the reason a character count can surprise somebody, and it needs the normalization tables rather than the case ones.

The `str` accessor refuses a column that is not text at the accessor rather than at the method, which is pandas, and the message names what the column holds using this library's vocabulary rather than the one `infer_dtype` uses. That was decided when the accessor was built and is unchanged by any of this.
