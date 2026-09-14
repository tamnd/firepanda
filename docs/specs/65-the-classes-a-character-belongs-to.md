# 65. The classes a character belongs to

## 1. A different question from the one document 64 answers

Document 64 is about what a character maps to. This one is about what a character is. `upper` needs to know that `a` becomes `A`, which is a pair, and a list of pairs is a list. `islower` needs to know that `a` is lower case and `ª` is not and `ǅ` is neither, which is a class, and a class is not a list of anything a reader would want to read. The two questions look alike from the outside and the data underneath them is a different shape, which is why document 64 could correct 149 code points out of a page of generated source and could not do the same thing for the three questions it left alone.

## 2. The 1384

Measured by walking all 1111998 code points through Arrow and through the Mojo standard library, the two disagree about the three questions as follows. Arrow calls 29 code points whitespace and the standard library called 12, missing the four ASCII separators at U+001C to U+001F, the no break space, and the space separators from U+1680 through U+3000. Arrow calls 2326 code points lower case and the standard library called 1489, with 841 Arrow has that it did not and 4 it had that Arrow does not. Arrow calls 1928 code points upper case and the standard library called 1433, with 526 Arrow only and 31 its own. The two sets it had that Arrow does not are the titlecase characters, which it counted as both cases at once and Arrow counts as neither.

1384 code points across three questions is not a patch list. It is whole blocks of Unicode on one side and not the other, plus a disagreement about what case means that no number of individual corrections would settle. So this is the table document 64 declined to build, built.

## 3. The tables are Arrow's and not the Unicode database's

Issue [#748](https://github.com/tamnd/firepanda/issues/748) proposed generating these from the published Unicode data files, and that is not what happened. pandas answers these three names out of Arrow. Arrow is therefore not an approximation of the right answer here, it is the right answer, and a table generated from the Unicode files would be a second opinion this library would then have to reconcile with the first every time utf8proc and the release we happened to read drifted apart. Reading pyarrow directly means a code point cannot disagree, ever, and it means the generator is twenty lines rather than a parser for a file format.

The cost of that choice is that the generator needs pyarrow, which the Mojo build does not have, so `firepanda/kernel/charclass.mojo` is committed rather than built. That is the same arrangement `casefix.mojo` and `casefold.mojo` already have and the same reason.

## 4. Ranges, held flat, read by parity

Unicode hands out properties in blocks, so a class is dense in runs and sparse in code points. The lower case letters are 2326 code points in 667 runs, the upper case are 1928 in 657, the titlecase are 31 in 10 and the spaces are 29 in 10. A table with a row per code point would be 1114112 rows per class whatever the class holds. A table of runs is 1344 numbers for the largest of the four.

The runs are written flat, as a start, one past an end, the next start and so on, which is an even number of entries in order. A code point is in the class when the number of entries at or below it is odd, which is exactly what a lower bound search leaves behind, so membership is one binary search and one parity test and there is no second comparison of the kind `_listed_at` needs to tell a hit from a miss.

Below 128 there is no search. Each class carries its first 128 code points as two 64 bit words and an ASCII character is a shift and a mask. That is worth having here in a way it is not in the mapping kernels, because those can rule out an entire element with a pass over its bytes and skip the decoding altogether, and a question about a class has to look at every character whatever the answer turns out to be.

All four classes together are 2688 numbers and eight words, which is under eleven kilobytes of the binary.

## 5. What the three questions actually are

None of the three is the loop a reader expects, and the two about case are not each other's opposite in two separate ways.

A row is whitespace when it has a character in it and every character is in the space class, so the empty row is not whitespace. A row is lower case when at least one of its characters is in the lower class and none of its characters is in the upper class or the titlecase class. A row is upper case the same way with the two case classes exchanged. So a row of digits is neither, because it has no cased character to be in a case, and a row of `abc1` is lower case, because the digit is in none of the three classes and does not count against anything.

That rule was not assumed. The generator builds it in Python out of the same four sets it is about to write and checks it against pyarrow over every code point on its own and over sixty thousand random words drawn from the classes that make it interesting, and it refuses to write a table if pyarrow answers differently on a single row.

## 6. The third case

A titlecase character is one that has a different form at the start of a word from the form it has in the middle, which in practice is the four Croatian digraphs like `ǅ` and twenty seven Greek letters with an iota written under them. Arrow calls them neither lower nor upper. That is one of the few places where the standard library was not merely out of date but disagreed, since it called them both.

They are their own class here, `TITLE_ONLY`, and both case questions read it even though neither is named after it. A row holding one is not lower case and not upper case, and so is a row holding one next to an ordinary letter, which is the answer `ǅa` gets and is not the answer any rule built out of the other two classes would give.

`TITLE_ONLY` is the same 31 code points `KEPT_BY_SWAP` in `casefix.mojo` already carries for `swapcase`, arrived at from the other direction. That file derived them from the mappings, because a character both of whose mappings move it and which is in neither case can only be the third one, and this one reads them from the class directly. The two lists agreeing is a check on both.

## 7. What it is worth

The three questions now agree with pandas on every code point in Unicode, asserted by a sweep in `python/tests/test_str_case.py` rather than by a sample, and on sixty thousand random words besides. The test that used to name three specific rows where this library and pandas differed, written to fail the day the data was replaced, did fail and is now that sweep.

A million rows through either library is a fraction of a second, so a test that asserts the whole measurement costs about two seconds and there is no reason to assert a sample of it instead.

## 8. What this does to the board

Nothing moves, and that is the point worth writing down. The three questions were already scored on all four string frames and already passed, because the corpus does not contain a non breaking space or a Croatian digraph in a row that is asked one of these questions. They were correct on the corpus and wrong in the general case, document 64 section 6 said so in as many words, and a conformance board cannot tell those two states apart. The sweep is what tells them apart.

The one divergence these three carry is unchanged and is `engine/string-predicate-null`, which is that a missing row answers a missing value here and False in pandas, because the answer in pandas is a numpy array of bools with nowhere to put a third state.

## 9. What is left

The classes here are the four the three existing questions need. The other class questions in the accessor need four more of the same kind, which are the alphabetic characters, the numeric ones, the digits and the decimal digits, and those are 684, 146, 136 and 72 runs respectively. `isalnum` needs no table at all, since a character is alphanumeric exactly when it is alphabetic or numeric, which was measured rather than assumed. `isascii` has no Arrow kernel, which is worth knowing before anybody writes it.

`istitle` and `title` need one thing these four classes do not give, which is whether a character is cased at all, and that is the union of the lower class and the titlecase class and needs no fifth table either. What they do need is a rule about what comes before a character rather than about the character, which is the first question in this part of the library that is not answerable one code point at a time.
