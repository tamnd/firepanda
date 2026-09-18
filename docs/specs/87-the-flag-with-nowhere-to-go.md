# 87. The flag with nowhere to go

## 1. What this is

`str.extract` under a `flags` argument, which is the sixth and last pattern method on the accessor to read one. Document 86 section 12 named it and named the reason it was left: it hands back a frame of groups rather than a column, so it crosses the accessor's door by a shape that takes no flags, and the letters would have to travel beside the group count. That was right about the wiring and wrong about why the wiring looks that way, which is the one thing in this slice worth a document. Issue #8 M6.

The change is fifteen lines of plumbing and no kernel at all. `program_for(METHOD_EXTRACT, pattern, flags)` has read the letters since the day it was written, because it compiles for Python's engine before anything routes, so every flag this library can carry was already honoured in the machine and nothing above the machine was passing one down.

## 2. The argument that says one thing where the others say two

Everywhere else on this accessor a `flags` argument is two facts wearing one name. The letters say what the pattern means, and the fact that any letter at all was passed says the call has left Arrow. Document 85 built the first version of that, document 86 pulled the second fact out of the number and put it back in the word because `replace` reaches Python's engine with zero flags, and the shape that came out of those two slices is a word carrying the engine and a number carrying the letters.

`extract` has no second fact. pandas never sends it to Arrow under any argument, because `ArrowStringArray._str_extract` and `ArrowExtensionArray._str_extract` are two different implementations and the one a string column gets is the object one, which is `re.compile(pat, flags=flags)` and then a walk. So there is no route for the argument to carry, no engine for it to pick, and nothing for the word at the door to say that the word was not already saying. The letters cross on their own beside the pattern, which is what a flags argument looks like when the only thing it does is mean something.

This is also the one pattern method with no `case` argument, so it is the one where the two spellings of a fold cannot be asked to agree with each other. Document 84 exists because `case=False` and `flags=re.IGNORECASE` and a written `(?i)` are one fact that arrives by three routes. Here there are two routes and no argument to compare them against.

## 3. Reading the letters before counting the groups

The order inside `extract` was the one decision with two defensible answers. The accessor refuses a pattern that opens no group, with pandas' own sentence, and it refuses a flag this library cannot carry, with its own. A pattern that is both has to get one of them.

Upstream compiles the pattern and then asks the compiled object how many groups it has, so a bad flag is refused over the pattern and never reaches the group count. This library now parses with the flags seeded, checks `tree.groups` off the parse, and then compiles. That puts the flag refusal first for the same reason upstream has it first, and it is not only about matching a message. A group is opened by a bracket, and verbose mode is the one flag that changes what the characters around a bracket mean, so counting groups before reading the flags would be counting them under the wrong grammar the first time somebody carries `(?x)`. The order is cheap now and it is the order that stays correct later.

## 4. The check that moved, for the second time in three slices

Document 85 section 11 moved the type check on the pattern to the top of the mask methods, because it had been living on the path a pattern took to reach the byte search and a flagged call was a second path that skipped it. The same check on `extract` had the same shape and this slice is the second path for it, so it moved too.

Naming it twice is the point. A validation that sits where the work happens rather than where the call arrives is correct exactly as long as there is one place the work happens, and every slice in this component has been adding a second place. `_a_pattern` is now called at the entrance of all six, and a seventh name that grows a route will not find this particular bug waiting for it.

The deletion that came with it is `_no_flags`, the helper that turned a flags argument into a refusal. It had one caller left.

## 5. What the seven letters do here

`re.IGNORECASE`, `re.MULTILINE`, `re.DOTALL` and `re.UNICODE` are answered, and they are answered by the same tree seeding documents 83 and 84 built, so there is nothing new in the engine to describe. The rows that tell them apart were produced by running pandas 3.0.5 over a candidate set rather than recalled, which is the procedure the last slice's note adopted after getting the same wrong example into a test five times.

`^([a-z])` over `"1\nab"` is missing without a flag and `"a"` under multiline. `(a.b)` over `"a\nb1"` is missing without a flag and `"a\nb"` under dotall. `(k)` over the Kelvin sign is missing without a flag and the Kelvin sign under ignore case, which is the row that separates a fold done through a lowercase table from a fold done through the whole cycle. Each of those is one cell of a frame rather than one cell of a mask, so a wrong answer shows up as a missing value where text was expected, which is the failure this method can have that the five above it cannot.

`re.LOCALE` is a `ValueError` on both sides, and it is upstream's own refusal rather than a copy: pandas hands the pattern to `re`, and `re` will not take the locale flag with a text pattern. `re.VERBOSE` and `re.ASCII` are refused here and answered upstream, which is this library's gap and is the same gap document 86 section 12 recorded. They are now the only two of the seven letters that any name on this accessor turns down, and the reason is the parser rather than anything about routes.

A value that is not one of the seven is refused rather than dropped, which is the rule document 85 section 5 set. `re.DEBUG` is the one worth knowing about, because it is a real flag that upstream answers by printing the parsed pattern to standard output and then giving the ordinary result. This library refuses it.

## 6. The mask that thought it already knew everything

Writing the test for that last refusal is what found a bug in it, and the bug was two slices old and belonged to all six names rather than to this one.

The check is a mask of the seven letters ored together and then a test that the caller's value has no bit outside it. The seven letters are members of `re.RegexFlag`, which is an `IntFlag`, and oring `IntFlag` members gives an `IntFlag`. The complement of an `IntFlag` is bounded by the bits that enumeration defines rather than by the integer, so `~mask` came back holding only the flag bits `re` names and none of the ones it does not. Every bit outside the enumeration read as already known and was dropped without a word.

The part that made it survive two slices is that it was wrong in exactly the direction nobody tests. `re.DEBUG` and `re.TEMPLATE` are bits `re` names, so they were caught correctly the whole time, and those are the two anybody would reach for while writing this test by hand. `1024` is not a flag in the module and it is the value that shows the defect, and it got into the test only because the rows for this slice were produced by running candidates through both libraries rather than chosen from what seemed worth asserting.

The fix is to build the mask out of plain integers and to read the caller's value as an index once at the top, so the arithmetic below is arithmetic. The test asserts it across all six names, since they share the one helper and so they shared the one defect.

## 7. How it was checked

336 cells, which is eight rows through seven patterns under six flag combinations, compared against live pandas by value. Then five tests that each pin one letter to the discriminating row above, the two refusals, and the pattern type check under a flag.

The two tests in other files that asserted `extract` refuses a flag were the visible cost. Both were renamed to assert the opposite, which is the fourth slice running where the test that proved a gap became the test that proves it closed, and the count of those is the honest measure of how much of document 85's plan was a plan rather than a prediction.

The Mojo suite, the Python suite, ruff, mypy and the generated bindings check were run before the merge.

## 8. What moved

Every pattern method on the `str` accessor now reads `flags`. `contains`, `match`, `fullmatch`, `count`, `replace` and `extract` is the whole list, and it took five slices across documents 83 through 87 because the first four of them were about the engine underneath rather than the argument on top.

On the conformance board the `flags` parameter is exercised for the last of the six, which was the last thing standing between the strings section and a parameter reading that reflects the accessor rather than the order the work happened in.

## 9. What is not here yet

Verbose mode in the parser and the ascii flag, which are now the only two letters refused for a reason inside this library and are unchanged from document 85 section 12.

`findall` and `extractall`, which are the two names on this accessor with no engine path at all. `findall` wants a list column, which does not exist, and a third scan. `extractall` wants a `MultiIndex` whose names are `[None, 'match']` and drops the rows that did not match. They are the next thing here and they are a bigger slice than this one by a long way.

A scoped flag group is still refused, which is 525 held-out patterns in the differential. The five constructs the router sends to Python are still refused: lookaround, backreference, conditional, atomic group and possessive quantifier. The RE2 grammar front end is still the largest single gap in the component at 19620 of 30052 corpus patterns.

The replace differential still sets aside 186 sweeps that it no longer needs to set aside, which document 86 section 12 named as the cheapest reach this component has left and which this slice did not take either.

`text_extract_regex` is the one text kernel still running down its rows one at a time. `stack_payloads` is the half of the work it needs and it exists, so this is a smaller job than it was before the two replaces were made parallel.
