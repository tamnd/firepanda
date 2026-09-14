# 69. The fold a search is not the fold a reader gets

## 1. The argument that was refused five times

`contains`, `match`, `fullmatch` and `replace` all take a `case` argument, and until this slice all four refused it when it was False. The refusal was correct and it was recorded as a gap rather than as a divergence, because ignoring an argument that changes the answer is the worst thing a compatibility layer can do and answering it slowly is the second worst. Document 66 section 6 made that argument and document 67 section 4 made it again for `replace`, where it stung more because pandas does honour `case=False` there.

`casefold` landed in the case group and looked like the missing half. It is not, and finding out why is most of this document.

## 2. Two folds, and the measurement that separates them

`str.casefold` is allowed to make a row longer. `ß` casefolds to `ss`, `ﬁ` casefolds to `fi`, `ﬄ` casefolds to `ffl`. That is what the Unicode full case folding tables say and it is what a reader of a row wants, because a fold is supposed to answer the question of whether two words are the same word.

A search cannot afford it. If the pattern were folded and every row were folded, a match found in the folded row would cover a number of bytes that has nothing to do with the number of bytes it was found in, and a replacement would have nowhere to put itself. So Arrow does not do it, and pandas answers three of these four names out of Arrow.

The measurement is short and it decided the whole design. `pyarrow.compute.match_substring(rows, "straße", ignore_case=True)` says that `STRASSE` does not hold `straße`, and `match_substring(rows, "ﬁance", ignore_case=True)` says that `FIANCE` does not hold `ﬁance`. pandas gives the same two answers on a column held the way pandas 3 holds one by default. So the fold a case insensitive search compares through is simple case folding, one code point to exactly one code point, and it is not `casefold`.

It is also not `lower`, which is the other obvious guess and the one that is wrong quietly rather than loudly. Lowercasing misses four pairs that a search calls equal: final sigma against medial sigma against capital sigma, the micro sign at U+00B5 against Greek mu, long s against s, and the Kelvin sign at U+212A against k. Each of those was measured through Arrow and each of them matches.

## 3. The rule, recovered rather than read

There is a Unicode data file with the simple folds in it. This library does not read it, because what has to be matched here is what pandas does and not what Unicode recommends, and the two have parted company before in this project.

So the rule was derived from measurements and then checked back against the thing it was derived from. It is: the full fold, when the full fold happens to be a single code point, which covers 1426 code points; otherwise the lower case, when that is a single code point different from the character itself, which covers 28 more and is the branch that gets `ẞ` to `ß`; otherwise the character itself, which is the 76 that a search leaves exactly as they are. Three pairs sit outside that rule and were found by checking every pair inside a full fold class by hand: U+1FD3 to U+0390 and U+1FE3 to U+03B0, which are Greek letters Unicode encodes twice, and U+FB06 to U+FB05, which are the two `st` ligatures and have no simple fold written down anywhere.

That is 1457 entries with 1427 distinct targets. `tools/gen_searchfold.py` writes them into `firepanda/kernel/searchfold.mojo` and verifies rather than trusts: every entry is asked of pyarrow in both directions, and then every one of the 1427 targets is swept against all 1.1 million code points to check that the set of things Arrow folds onto it is exactly the set this table says it should be. That sweep found nothing wrong and took 92 seconds, which is why the file is committed instead of being built.

## 4. The fourth method, which pandas answers out of a different language

`replace` is not answered out of Arrow. pandas' `_str_replace` in `_arrow_string_mixins.py` raises `NotImplementedError` when `case` is False and falls back to the object path, which escapes the pattern with `re.escape`, adds `re.IGNORECASE` and runs it through Python's regular expression engine.

So one of these four methods has its answer decided by a different implementation in a different language, and there is no reason in principle for the two to agree. They were checked. Every pair of code points that simple folding calls equal was tested against `re.IGNORECASE` and every one of them agrees, and the three extra merges are the same three. That is why `pattern.mojo` has one folded search rather than two, and it is a fact about pandas 3.0.5 and pyarrow 25.0.1 rather than a guarantee: the generator checks it on every run and will stop if it stops being true.

## 5. Why the pattern is folded once and the row is not folded at all

The obvious implementation folds both sides into new strings and runs the search that already exists. It is correct and it doubles the memory the accessor touches, because folding a column means writing a second column as tall as the first.

What ships instead folds the pattern once, into a small list of code points, and then walks each row a character at a time, folding as it reads and comparing to that list. Nothing column sized is allocated for the three questions, which are fixed width answers, and `replace` allocates what its output needs and nothing more.

There is a cost and it is worth naming. The case sensitive search has a skip table: it looks at the last byte of a window and jumps forward by however far that byte cannot possibly be part of a match. That is a statement about bytes, and this search compares code points that the bytes in front of it do not hold, so the skip does not apply and the folded search is the naive one. The same reasoning kills the wide scan. A folded match may also cover a different number of bytes than the pattern it matched, because `ſ` is two bytes and is compared as the one byte `s`, which is why the inner function answers the offset where a match ended rather than a yes or a no, and why the folded `fullmatch` cannot decide a row by comparing lengths the way its sibling does.

The ASCII half of the fold is not in the table. The lowest code point in it is the micro sign at U+00B5, whose lead byte is 0xC2, and 0xC2 is the lowest lead byte any non ASCII character can have, so a byte below 128 can be folded with a subtraction and the binary search is only reached by characters that might actually be in the table.

## 6. A second entry point rather than a flag

The kernel has `text_contains_folded` beside `text_contains` rather than a `fold` parameter on one function, and the crossing into Python spells the choice as a word: `contains_folded`, `match_folded`, `fullmatch_folded`, `replace_folded`. That is not a fourth door and it is not a per method door. It is the same thing `strip` and `strip_chars` already are, and `pad_left` and `pad_right` and `pad_both`: one pandas name whose argument picks which kernel runs, spelled out on the Mojo side so that the dispatch reads as a list of kernels rather than as a list of flags. The three doors are still sorted by the shape of the answer and there are still three of them.

The Python layer decides which word to send. `_fold_word` is what used to be `_plain`, and it now returns a suffix instead of refusing: the empty string when `case` is absent or True, `_folded` when it is False. `flags` is still refused by that same function, because every flag is a statement about a regular expression and there is still no engine to read one with, and `re.IGNORECASE` written as a flag is refused even though it is the same request `case=False` makes, since accepting one spelling of it would mean accepting every other flag beside it.

`count` has no `case` argument at all, which is the one place the four disagree about their own signature, so it passes `None` and the suffix comes back empty.

## 7. The awkward finding, which is pandas disagreeing with itself

`n=0` means no replacements to `str.replace` and every replacement to `str.replace` with `case=False`. That is the same method on the same column with the same argument, answering two different things depending on another argument entirely.

The cause follows directly from section 4. The Arrow path takes the number at its word and does nothing. The fallback path hands it to `re.sub`, where a count of zero has meant unlimited since long before pandas existed. So turning a search insensitive silently turns a request for nothing into a request for everything. It was measured on pandas 3.0.5 and it is not documented anywhere.

This library matches it, because matching pandas is the job, and the widening is done in the Python layer rather than in the kernel so that the number still means what it says everywhere below the accessor. It is worth filing upstream and it is the kind of thing that only turns up when somebody sweeps the argument space rather than testing the interesting values.

## 8. What the board says

Four names stop refusing an argument. `strings/contains-case-false` was already sitting in the corpus unarmed and is armed now, and cases for `match`, `fullmatch` and `replace` with `case=False` are new, along with one on a frame whose rows separate simple folding from `casefold` and from lowercasing, since the corpus as it stood could not have told the three apart.

The null rows behave as they did before. The three questions hand back a missing row where pandas fills False into a bool column, which is `engine/string-predicate-null` and is unchanged by folding, and `replace` keeps a missing row missing exactly as pandas does.

## 9. What is left

`case=False` is no longer owed anywhere. What the four names still refuse is a pattern holding one of the twelve regular expression metacharacters, and a non zero `flags`, and both of those are the same missing engine.

Fourteen names in the accessor are unwritten and the shape of the pile has not changed: eight want a column of lists, four want the engine, `normalize` wants the Unicode normalization tables and `encode` and `decode` are two small separate problems. The one thing this slice adds to that list is a table, and `searchfold.mojo` sits beside `casefold.mojo` as the second piece of Unicode data this library carries rather than derives.
