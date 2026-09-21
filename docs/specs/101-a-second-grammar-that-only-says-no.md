# 101. A second grammar that only says no

## 1. What this is

Two patterns in three that reach this accessor are patterns nobody can run. Of the thirty thousand generated patterns in the corpus, 19615 are ones Python's grammar cannot read, which is exactly why they end up in front of Arrow in pandas at all, and 17976 of those are ones RE2 will not read either. For every one of those the whole of the answer a caller gets is a refusal, and until this slice landed firepanda gave the wrong one.

Wrong in a way a caller can trip over rather than wrong in a way only a report notices. pandas hands the pattern to Arrow, Arrow refuses it, and the refusal comes back as `pyarrow.lib.ArrowInvalid`, which is a subclass of `ValueError`. firepanda could not read the pattern either, but it had no way to tell a pattern that is broken from a pattern that is merely beyond this library, so it said `NotImplementedError` for both. A caller who wrapped the call in `except ValueError` caught the one and not the other.

So there is a second grammar in `firepanda/kernel/regex/re2.mojo` now. It reads RE2's syntax and answers one question, whether RE2 would read this pattern, and when the answer is no it says why. It compiles nothing and it runs nothing. It exists so that a refusal can be the right class with the right reason on it.

The measurement at the end of the slice, on `str.contains` over the same thirty thousand patterns and sixteen texts the other differentials use:

```
compared 27431 patterns          (was 9623)
held out 2621 patterns           (was 20429)
    1807 Python's grammar cannot read this pattern    (was 19615)
    390 a named character is not resolved yet
    218 RE2 reads this syntax differently
    111 RE2 reads a non boundary between bytes
    40 this engine has no lookaround beside a backreference yet
    25 this engine has no capture inside a lookahead yet
    17 this engine has no lookaround beside an atomic group yet
    10 this engine will not repeat that many times
    3 this engine has no lookaround beside a conditional group yet
agreement 10000 in ten thousand, 0 disagreements
```

Roughly 17800 patterns moved from a wrong `NotImplementedError` to the right `ValueError`, carrying RE2's own reason in this library's voice. `str.match` and `str.fullmatch` moved the same way, to 27529 and 27530 compared.

## 2. Why the existing grammar could not answer this

There is already a parser in `parse.mojo` and it already says no to plenty. The trouble is that what it says no to is not the same set.

This library's parser reads Python's grammar, because Python's grammar is what a pattern has to pass before either engine here can compile it. RE2's grammar overlaps it but neither contains the other. `\Q a\E` is RE2 and not Python. `\p{L}` is RE2 and not Python. `(?P=name)` is Python and not RE2. `a{,2}` is a repeat in Python and a literal in RE2. `[a-\d]` is a class in Python and a refusal in RE2. The two grammars disagree about several hundred corpus patterns in each direction, so a no from one is not evidence of anything about the other.

Reusing the tree was not an option for a second reason, which is that the tree does not exist when the question is asked. The question only comes up on the branch where parsing failed, so there is nothing to walk. The reader takes the pattern text and starts over.

## 3. What the reader is not

It is not a compiler. `compile_program(tree, ENGINE_RE2, ...)` is still what turns a pattern into something RE2's target can run, and this reader is consulted only on the branch where that never happens. It emits no instructions and has no opinion about what a pattern means.

It is not an engine. It never sees a row of text.

It is not a fourth target. `ENGINE_RE2` and `ENGINE_PYTHON` are still the two things a program can be compiled for.

It is one function, `re2_reads(pattern)`, answering a struct of three fields: whether RE2 reads it, why not if not, and whether the reader declined to judge. Everything else in the file is the recursive descent underneath it.

## 4. The text pandas actually hands Arrow

This turned out to be the part most worth measuring rather than assuming, because the pattern a caller writes is not always the pattern RE2 is asked about.

pandas anchors before it routes. `str.contains` hands Arrow the pattern as written. `str.match` hands it `^(` plus the pattern plus `)`. `str.fullmatch` hands it `^((` plus the pattern plus `)$)`. That is not a detail that stays hidden, because the wrapping changes the answer.

`str.match("?")` answers rather than raising. A bare `?` has nothing to repeat and RE2 says so, but `^(?)` is not a repeat at all: it is a group opening with `?`, which RE2 reads as a flag group naming no flags, which is legal and matches everything. Meanwhile `str.contains("?")` raises.

The wrapping is visible in the refusals too. `str.contains(")a")` comes back quoting `)a`, and `str.match(")a")` comes back quoting `^()a)`, so the parenthesis that is unexpected is a different one. `str.fullmatch("a)")` quotes `^((a))$)`.

That is why `anchored` grew a third parameter. The function already existed to wrap a pattern per method, but it also hoisted leading flag groups, which is firepanda's own improvement and not something pandas does. The reader wants pandas' text exactly, so the call site passes `hoist=False`:

```mojo
var read = re2_reads(anchored(method, preprocessed(pattern), False))
```

`preprocessed` stays on, because that is a transformation pandas also performs.

## 5. The grammar, and how it was measured

None of this was read out of RE2's source. It was measured, one pattern at a time, against a running Arrow through `pyarrow.compute.match_substring_regex` on a one element column, which is the same RE2 a caller will meet. Several of the rules are ones nobody would guess.

**Escapes are two different sets depending on where they are.** Outside a class RE2 takes `a b d f n r s t v w z A B C D Q S W` plus `p P x`. Inside a class it takes `a d f n r s t v w D S W` plus `p P x`, and `b z A B C Q E` are refusals there. Every ASCII punctuation mark, space and tab after a backslash is a literal. A non ASCII character after a backslash is not: `\é` is refused.

**Octal has a shape.** `\0`, `\00`, `\000`, `\0000`, `\07`, `\08`, `\12`, `\101`, `\400`, `\777` and `\1000` are all read. `\1`, `\8`, `\9` and `\18` are not. A leading zero makes it octal; without one it needs at least two digits.

**A brace that does not open a well formed count is a literal.** `a{,}`, `a{}`, `a{a}`, `a{2,`, `a{2`, `a{-1}` and `a{1,2,3}` are all patterns about a literal brace. Nothing is refused.

**RE2 checks a repeat in a fixed order**, and the order is what decides which complaint a pattern with two problems gets. First, is the brace a well formed count, and if not it is a literal and we are done. Second, consume a trailing `?` as the ungreedy marker. Third, is this a repeat on a repeat, which is `bad repetition operator`. Fourth, is the size out of range, which is `invalid repetition size`. Fifth, is there anything to repeat, which is `no argument for repetition operator`. Getting that order backwards was twenty of the twenty one disagreements on the first differential run.

**RE2 builds on a stack and a repeat takes whatever is on top of it.** A flag group pushes nothing, so a repeat written after one reaches past it and grabs what was there before. `(?i)*` is a refusal because it stands at the front of the pattern and there is nothing underneath, but `a(?i)*` repeats the `a`, and `(?im:(?s))(?i-s){1,3}?` is a pattern RE2 reads. The three places that leave nothing on the stack are the start of the pattern, a bar, and the opening of a group. That one rule is the other disagreement from the first run, and it is not something a grammar written from the syntax alone would produce.

**The count limit is 1000 and a nest of counts shares it.** A single count above 1000 is refused. Nested counts multiply and spend the same budget, so `(a{100}){10}` is read at exactly 1000 and `(a{100}){11}` is refused, `(a{31}){32}` is read at 992, `((a{10}){10}){10}` is read at 1000 and `((a{10}){10}){11}` is refused. Branches of a bar take the larger rather than the sum, so a budget spent down one arm is not also spent down the other.

**Group names are looser than Python's.** `(?P<1>a)`, `(?P<1n>a)`, `(?P<é>a)` and two groups with the same name are all read. `(?P<n n>`, `(?P<>` and `(?P<n->` are `invalid named capture group`, and `(?P'n'a)` and `(?Pn>a)` are `invalid perl operator`, which is a different complaint for what looks like a neighbouring mistake.

**A class has its own rules.** A `]` first in a class is a literal, so `[]a]` and `[^]a]` are classes. A `-` at either end is a literal. `[\d-a]` and `[\d-\w]` are read, because a set on one end of what looks like a range means it is not a range. `[a-\d]` is `invalid escape sequence` rather than a range complaint. `[b-a]` and `[漢-é]` are backwards ranges. `[[:alpha]]` and `[[:]]` are read as ordinary classes, and only `[[:foo:]]` is refused.

The whole measured grammar is what `tools/regex_re2_oracle.py` produces on demand, and it is worth saying why that helper does not go through pandas' accessor: the router would send half of these patterns to `re` and they would never reach RE2, so most of the grammar would be unreachable from the place a caller stands.

## 6. Where the bias goes

The two ways of being wrong are not the same size, so the reader is deliberately lopsided.

Saying no to a pattern RE2 reads is a new wrong refusal. Somebody had a working column and now has an exception. That is a regression this slice would be introducing on its own account, and the ceiling for it is zero.

Saying yes to a pattern RE2 refuses leaves the caller exactly where they already were, which is holding the `NotImplementedError` they were holding before. It is a miss rather than a break.

So when the reader is not sure, it says yes. There is one construct it is not sure about, which is `\p{...}` and `\P{...}`. Telling a script or category name RE2 knows from one it does not needs a table this library has not got, so a pattern holding either spelling sets an `unsure` flag, and an unsure pattern comes back as read no matter what else was found in it. That is 166 corpus patterns RE2 refuses and the reader declines to judge, and they stay held out rather than becoming wrong.

The same bias is why the reader keeps the first refusal it finds rather than the last. RE2 reports the first problem it hits and so does this.

## 7. The differential

`tests/differential/regex_re2.mojo`, run by `pixi run differential-regex-re2`, asks RE2 about every pattern in the corpus and asks the reader the same thing, and compares both the verdict and the reason.

Comparing the reason takes a mapping, because the reader speaks in this library's voice. `invalid perl operator` is not a sentence about anything a caller wrote. The reader says `RE2 has no group written that way` instead, and `kind_of` in the differential maps the twelve sentences back onto RE2's eleven kinds so they can be compared. The mapping lives in the harness rather than in the reader, so the reader owes the harness nothing. Two sentences map to one kind, because RE2 uses `invalid character class range` both for a class name it does not know and for a range written backwards, and telling those apart is worth doing even though RE2 does not.

Two ceilings are enforced and both are zero: a pattern RE2 reads and the reader refuses, and a pattern both refuse for reasons that differ. The other two buckets are counted and printed and do not fail the run.

```
corpus 30052 patterns
RE2 reads and the reader reads 8339
RE2 refuses and the reader refuses 21547
    12747 invalid perl operator
    4247 invalid escape sequence
    1495 no argument for repetition operator
    615 bad repetition operator
    535 missing ]
    523 invalid named capture group
    432 unexpected )
    385 missing )
    236 invalid repetition size
    224 invalid character class range
    108 trailing \
RE2 refuses and the reader declined to judge 166
RE2 refuses and the reader read it 0
agreement 10000 in ten thousand, 0 disagreements
```

Those numbers are the ones this reader was born with and they have moved twice since. The line counting what the reader declined to judge is gone entirely, because document 103 built the table of Unicode names that was the only thing it ever counted, and a bucket that can only print zero is worse than no bucket. As of that document the run is 8373 read, 21679 refused with matching reasons and nothing set aside, still at zero in both ceilings.

The zero on the second to last line is the one that was not expected. The reader was written to be allowed to miss, and over thirty thousand patterns it missed nothing it was willing to judge.

The first run of this differential had 21 disagreements and both causes were rules rather than slips, which is the argument for having built the differential before wiring anything up. One was the flag group and the stack. The other was the repeat check order, which accounted for the other twenty and which no amount of reading the syntax would have produced.

## 8. Where it is wired in

One branch, in `program_for` in `method.mojo`, the one taken when this library's own parse failed:

```mojo
var out = compile_program(tree, ENGINE_RE2, minor=minor)
var read = re2_reads(anchored(method, preprocessed(pattern), False))
if not read.ok:
    out.problem = read.problem.copy()
    out.gap = False
return out^
```

`gap` is the field that picks the class. False is a `ValueError` and true is a `NotImplementedError`, which is the pairing `tagged` already had. So a pattern RE2 also refuses keeps RE2's reason and becomes a `ValueError`, matching what `ArrowInvalid` is, and a pattern RE2 would have read stays a gap with the message it had, because that one really is a thing this library cannot do yet.

Nothing else moved. The order of the engines, the router, the parse and the compile are all where they were.

## 9. The twelve sentences

`RE2 has no group written that way`. `RE2 has no such escape`. `there is nothing here for that repeat to repeat`. `RE2 will not repeat a repeat`. `a class is never closed`. `RE2 will not take that group name`. `a bracket is closed that nothing opened`. `a bracket is opened that nothing closes`. `RE2 will not repeat that many times`. `RE2 has no such character class`. `a range in a class runs backwards`. `a backslash is the last thing in the pattern`.

Eleven kinds and twelve sentences, for the class range reason given above. They are sentences about the pattern the caller wrote rather than about RE2's internals, which is the same choice every other refusal in this accessor makes, and it is the reason the mapping in the differential has to exist at all.

## 10. What is left

The 1807 patterns still held out on `contains` are the other half of the job, which is the patterns Arrow takes and this library cannot read. They are a construct at a time and they are known: `\Q...\E` is 208 of them, `\p` and `\P` is 298, `{,n}` read as a literal is 166, `\z` is 23, `[[:...:]]` is 18, a leading `]` in a class is 11, and after those come a flag group anywhere in the pattern rather than only at the front, a repeat on an assertion, `[\d-a]`, the octal forms and a group name starting with a digit.

The `\p` work is the one that pays twice, because a table of script and category names is also what would let the reader stop being unsure about the 166 patterns it declines to judge now. That is document 103, and it did pay twice.

Beyond those are the 218 patterns where RE2 reads the syntax differently rather than not at all, and the 111 where RE2 reads a non boundary between bytes. Those are not grammar questions and they do not belong to this reader.

The reader itself is not measured for speed and does not need to be yet. It runs once per compile, on a branch that ends in an exception, over a pattern of a few dozen characters. If a pattern that is going to be refused ever becomes hot, the compile in front of it is the larger cost.
