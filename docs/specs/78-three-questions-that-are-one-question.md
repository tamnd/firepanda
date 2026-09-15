# 78. Three questions that are one question

## 1. What this is

Documents 76 and 77 built a parser, a router and an engine, and left them with nothing calling them. This is the wiring: `str.contains`, `str.match` and `str.fullmatch` now send a pattern with a metacharacter in it to the engine instead of refusing it. Nothing else on the accessor moved, and section 9 says why `count` and `replace` stayed where they were.

The interesting part of this slice is not the wiring, which is four short layers and one of them is three lines. It is that `match` and `fullmatch` are not modes of the engine. pandas answers them by rewriting the pattern and asking `contains`, the rewrite is written in Python in pandas rather than being anything Arrow does, and copying it exactly is the whole job. A rewrite that is off by one bracket is not an error a caller sees, it is a column of booleans that looks like a right one.

Everything below was measured against pandas 3.0.5 with pyarrow 24.0.0 on CPython 3.14.7, and the pandas source quoted is `pandas/core/arrays/_arrow_string_mixins.py` and `pandas/core/arrays/arrow/array.py` at that version.

## 2. The rewrite, and what it is visible in

`str.match("a|b")` asks whether a row starts with `a` or starts with `b`. It does not ask whether a row starts with `a` or holds a `b` somewhere. That is the rewrite showing through: upstream puts what the caller wrote inside a group first and the anchor on afterwards, so the pattern the engine runs is `^(a|b)` and not `^a|b`. Those are different questions and the difference is an answer rather than an error.

So the rewrite is copied character for character rather than reimplemented from its description, and `tests/test_regex_method.mojo` asserts the pattern text rather than the answer. The differential compares answers, which is the stronger check and is blind in exactly one direction: a rewrite this library gets wrong in the same way for every pattern would still agree with pandas whenever the engine agreed with RE2, and a rewrite that quietly refused a family would agree with pandas on every pattern it did not refuse. Asserting the text closes both.

## 3. The order, which is not an order anything is indifferent to

pandas does three things to a pattern and it does them in this order.

It decides which engine gets the pattern, by reading the pattern the caller wrote. Then it rewrites a trailing `\Z` into `\z`. Then it anchors. Only the Arrow branch does the last two at all, because the Python branch hands `re` the pattern as written.

Each step moves patterns across a line and doing them in another order loses some. Routing on the rewritten pattern is wrong for `(?i)(?=a)`, which pandas reads and answers out of `re` and which stops being a pattern Python's grammar will read once a group is wrapped around it, so the pattern would be sent to an engine that has never heard of a lookahead and refused with the wrong reason. Rewriting `\Z` after anchoring is too late, because anchoring puts a bracket after it and RE2 refuses a `\Z` that is not the last thing in the pattern, so `str.match(r"ab\Z")` would be refused where pandas answers it. That one was found by the differential rather than by reading, and it was nine patterns per sweep at two thousand cases.

`program_for` in `firepanda/kernel/regex/method.mojo` is those three steps in that order and nothing else.

## 4. What `\Z` costs and why it is a rewrite rather than a rule

Python spells the end of the text `\Z` and RE2 spells it `\z`, and RE2 has no `\Z` at all. pandas rewrites one at the end of a pattern and leaves one anywhere else alone, which is why a `\Z` in the middle of a pattern is an Arrow error today and a `\Z` at the end is not.

Only a real escape is rewritten. `\\Z` is a backslash and then a capital letter, which is not an end of text to anybody, so the backslashes in front are counted and only an odd run means the `Z` is escaped. That is what upstream does and it is copied here rather than approximated, because approximating it is a wrong answer on `a\\Z` and a rare enough pattern that nothing would catch it.

## 5. The four branches of `fullmatch`

`_str_fullmatch` upstream is four cases and then a call to `_str_match`, and they are the four combinations of the pattern already carrying an anchor at each end. A pattern with both is left alone. A pattern with neither has a group and a `$` put around it. A pattern with only one of them has that one taken off first and then the same treatment. Then `_str_match` strips one leading `^` if there is one, wraps what is left in a group, and puts a `^` in front.

So `fullmatch("a")` runs `^((a)$)` and `fullmatch("^a$")` runs `^(a$)`, which is the same question asked with a different number of brackets.

Two details in there are copied rather than tidied. A trailing `$` counts as an anchor only when it is not written `\$`, which is how pandas tells an anchor from a dollar sign somebody wanted printed, and which pandas gets wrong for a pattern ending in a literal backslash followed by a dollar. One leading `^` is stripped and no more, so `^^a` becomes `^(^a)` and still asserts the same position twice. Neither changes an answer and both are in the tests, because a tidied version of either would drift away from upstream the next time somebody read it.

## 6. The flag group hoist, and the `(?m)` case that paid for it

There is one place this library does not copy the rewrite, and it exists because pandas hands its rewrite straight to Arrow while this library parses its own rewrite with Python's grammar first.

Python's grammar wants a global flag group first in the pattern and refuses one anywhere else. So `^((?s)a.b)`, which is what pandas writes for `str.match("(?s)a.b")`, is a pattern this library's front end cannot read, and `match` would refuse a pattern `contains` answers. The fix is to move the group in front of the rewrite, so `(?s)a.b` becomes `(?s)\A(a.b)`.

Moving it changes what it covers, and what it now covers that it did not before is exactly the two anchors the rewrite adds. That is harmless for six of the seven flag letters and it is not harmless for `m`. `fullmatch("(?m)")` in pandas is `[False, False, True, False]` over `['a', 'a\n', '', 'b\nc']`, because upstream leaves the flag inside the group and the added `^` and `$` are outside it, so they are the ends of the row. A hoist that let the flag reach them would make them the ends of a line and answer True for the first row, which is a different column and not a slower one.

So a hoisted rewrite writes the anchors it adds as `\A` and `\z`, which are the same two positions with no flag able to touch them, and it leaves any `^` the caller wrote where it was rather than stripping it, since upstream only strips one from the front of the whole pattern and a pattern opening with a flag group has none there. `(?m)^b` becomes `(?m)\A(^b)` and the caller's caret stays where the flag can reach it, which is where the caller put it. Both rules are written unconditionally rather than only for `m`, because a hoist that is sound for six letters out of seven is a hoist somebody has to keep rechecking.

The differential found the `(?m)` case as ten disagreements under `fullmatch` and it found nothing at all under `contains`, which is the sweep that was there before this slice. That is the argument for running the corpus three times over rather than once.

## 7. A pattern the grammar cannot read is refused as written

The rewrite adds brackets, and brackets can rescue a pattern that was not a pattern. `)a` closes a bracket nobody opened and Python refuses it, and `^()a)` is a perfectly good pattern. So a compiler that parsed only the rewritten pattern would answer `str.match(")a")` where pandas raises, since pandas reads the pattern as written when it picks an engine and never gets past that.

`program_for` therefore refuses on the original tree when the original tree did not parse, and only compiles the rewrite when it did. This was one of two disagreements at thirty thousand cases and it is the only one of the two that was real.

The other one is worth recording because of what it was. `[^abc])|(\a?` looked like a second bug of the same shape under `fullmatch` and was not a bug at all: the differential binary was being run directly out of `build/differential/` instead of through `pixi run -e differential`, which resolves a different Python and a different pandas, and the oracle was answering with the previous method's answers. The same binary run both ways gives different answers. The harness is only an oracle when it is run the way the pixi task runs it.

## 8. What a refusal becomes in Python

The compiler hands back a program or a refusal, and a refusal carries the flag from document 77 section 3 saying whose it is. `firepanda/py/text.mojo` is where that bit turns into an exception, and it turns into two different ones.

RE2 refuses this too, so pandas refuses it as well, out of Arrow, as a `ValueError`. That is what a caller gets here, so `a*+` and `(?#note)a` and `a\Zb` raise `ValueError` in both libraries and a program written against pandas keeps working. The wording is this library's own rather than Arrow's, because a caller here did not call Arrow.

firepanda cannot answer it yet, so it is a `NotImplementedError`, which is what `UnsupportedError` is built on. A lookaround or a backreference is answered by pandas out of Python's `re` and is a gap here, and a caller who catches `ValueError` around a pattern they know to be good should not be told they wrote a bad one.

`case=False` with a metacharacter is the third case and is a gap. pandas serves it by handing RE2 its own ignore case flag and there is no folding table here yet, so it is refused rather than quietly dropped. A literal pattern with `case=False` has a folding path already and keeps it, so what is refused is the pair rather than either half.

## 9. Where the literal path stops

`python/firepanda/_pandas.py` decides whether a pattern needs an engine at all, and a pattern holding none of the twelve metacharacters does not. That path was the whole of these three methods before this slice and it is still the path most patterns a program writes take, because it is a byte search and a great deal faster than anything that reads instructions.

`count` and `replace` did not move. Both need to know where a match ends so they can start looking for the next one, and the engine answers whether there is a match rather than where it is, so both still refuse a metacharacter by name. Answering `count(".")` as a search for a full stop would be wrong on every row of a column pandas counts everywhere, and refusing is the honest shape of the gap.

## 10. What was measured

`pixi run differential-regex-match` now runs three sweeps over the same corpus, one per method, and reports each separately. At thirty thousand cases it generates 30052 patterns, compares 7668 of them under `contains` and 7664 under each of the other two, and agrees on every text of every one of them, with zero disagreements in each sweep. The largest held out reason is the same one document 77 section 8 named: about 19600 patterns Python's grammar cannot read, which are the ones pandas answers out of RE2 precisely because Python refused them.

Beside that, `tests/test_regex_method.mojo` asserts eighteen rewrites as text, `tests/test_regex_column.mojo` checks the column kernel against its scalar twin across a morsel boundary and over rows chosen for what they leave behind rather than for what they match, and `python/tests/test_str_regex.py` checks sixty accessor calls against pandas plus the refusals and the two paths.

## 11. What is not here yet

The list from document 77 section 8 is unchanged except for its last paragraph. Captures, case folding, scoped flags, the Python engine and an RE2 grammar front end are all still open, and the front end is still the largest single gap in the component at about nineteen thousand six hundred patterns out of thirty thousand.

What this slice adds to that list is the rest of the accessor. `findall`, `extract` and `extractall` wait on captures. `count` and `replace` wait on knowing where a match ends, which is the same instruction captures need. `split` and `rsplit` want their regex form and also want a list column, which is a different gap in a different part of the library.

## 12. Observations to file upstream

A pattern Python's grammar refuses reaches Arrow and comes back as a `ValueError` rather than an `re.error`, because the router catches the parse failure and treats it as a pattern for the other engine. So pandas never reports a syntax error from its own parser for a string column, and a caller who writes `)a` is told by Arrow that they wrote something Arrow does not like rather than being told where the bracket is.

`_str_fullmatch` reads a pattern ending in a literal backslash followed by a dollar as ending in an escaped dollar, so the anchor it should have added is not added. This is section 5 and it is one line.

The rest of the list is in document 76 section 12 and document 77 section 9 and is unchanged by this slice.
