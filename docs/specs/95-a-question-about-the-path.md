# 95. A question about the path

## 1. What this is

The engine that copies Python now reads a backreference. `(\w)\1` matches a character followed by the same character again, `(?P<x>a)(?P=x)` is the same instruction under a name, and `^(.)(.)\2\1$` is a four letter palindrome. It is the largest of the five constructs Python has and RE2 has not, at 839 of the 30052 held out patterns, and it is the one document 93 promised would need a document about an engine rather than a document about a construct.

Here is why. Every other construct in this library is a question about the program and the position. A set asks what character is here. A position test asks what is on either side of here. A lookaround asks whether a second search starting here succeeds, and a second search is still a question about a program and a position. A backreference is the first one that asks what the path that arrived matched, and there are three engines under the compiler, and two of them are built on that question never being asked.

## 2. Why two of the three engines cannot be asked

The state cache in `dfa.mojo` builds a state out of a set of instructions. That is the whole idea of it: every path that can be at one of those instructions at this position is represented by the state, and the transition out of the state depends on the state and the next character and nothing else, which is what lets a transition be worked out once and then read out of a table on every row after. A backreference has a different answer for two paths that are both at that instruction, so there is no transition to store.

The Pike machine in `pike.mojo` keeps a list of threads and merges two the moment they stand at the same instruction and the same position. That merge is the whole of its bound: the work is the length of the row times the size of the program precisely because a pair cannot be in the list twice. A backreference is a question the two merged threads answer differently, so either the merge goes, and with it the bound, or the construct does.

The bounded backtracker in `backtrack.mojo` follows one path at a time and keeps the slots that path wrote. So it can be asked. It is the only one of the three that can, and that is the whole reason this slice is about engines.

## 3. What it costs the bitmap

The backtracker is not a plain backtracking engine. It is a plain one with a bitmap over it, and the bitmap's justification is the same sentence the other two engines rest on, written out in the document on the file itself: a pair of an instruction and a position may be dropped on a second visit because whether a match can be found from there does not depend on how the search arrived. With a backreference in the program it does depend on that, exactly and only because the reference reads a slot the arrival wrote.

The first version of this slice turned the bitmap off for such a program and left a plain backtracking walk, which is what every other regular expression engine with a backreference in it also is, including CPython's. That is not what is here, because turning it off took something else off with it, and section 11 has how the corpus said so.

What is here is the narrower rule the sentence above actually allows. A pair of an instruction and a position does not say everything about what is left to do, but a pair and the slots do, because the slots are the whole of what a reference can read. So the bitmap is kept and everything in it is forgotten the moment a slot changes value. Two arrivals separated by no change of any slot are the same state in every respect, and the second one is dropped. Two arrivals with a slot change between them are two states, and the second one is walked.

Forgetting is the length of a list rather than the length of the bitmap. `Bounded.marks` holds the cells set since the last change, `_forget` clears those and empties it, and `_write` is the one door every slot write goes through, the one that opens a group and the one that puts back what the group held before it.

The comparison in `_write` is not a saving, it is the point. A repeat over a body that matches nothing writes the same numbers into the same slots on every turn of it, so counting each of those writes as a change would forget the bitmap on every turn and leave the loop with nothing to stop it. Comparing first makes the second turn a state the bitmap has already seen, and the loop ends there.

A row too long for the bitmap is the one case left. Everywhere else that is a `GAVE_UP` and the machine takes the row, and here there is no machine to take it, so such a row runs with `stamped` off and the step count of section 6 is the whole of the bound. That is the only path that reaches the count on an ordinary pattern.

Two smaller designs were tried on paper and are written down here so that they are not tried again. Keying the bitmap on the live slots as well as the instruction and the position is correct and is a key too large to be worth building, and the rule above is that key with the slots factored out into a clear. A reachability pass marking the instructions from which no `IN_REF` is reachable, so that those could still be memoised unconditionally, is also correct and buys close to nothing, because a reference is usually near the end of a pattern and almost every instruction precedes it.

## 4. The instruction

`IN_REF` is instruction 13. Its `a` is the slot the group opened at, which is `2 * k` for group `k`, so the text to read again lies between `slots[a]` and `slots[a + 1]`. Its `b` is 1 when the two are compared with the ASCII case dropped and 0 when they are compared as they are.

A group that never took part is a slot pair still at minus one, and the instruction fails rather than matching nothing. That is upstream's answer and it is worth a sentence because the two cases look alike and are not. `re.match(r"(a)?\1b", "b")` is None, because the group was skipped and never took part. `re.match(r"(a?)\1b", "b")` matches, because the group ran and matched nothing, so the reference reads nothing and finds it. The test file carries both.

The width is the group's width and not the pattern's. `(a*)\1` against four `a` characters matches all four, because the group takes two and the reference reads two, and it gets there by the group taking four first, the reference failing, and the group giving characters back. That is ordinary backtracking and it is what the bitmap was making unnecessary everywhere else.

## 5. Where the slots come from

`IN_REF` reads a slot, and a caller asking whether a row contains `(\w)\1` never asked for the groups. So the pattern turns the slots on for itself. `compile_program` computes `slotted = captures or _holds_ref(tree.nodes, tree.root)` before the builder is made, because the save instructions bracket the whole program and have to be decided before the first instruction is written.

This is the one place in the compiler where the shape of a program is decided by the pattern rather than by the question the caller asked, and it is a cost the pattern brings rather than one the question does. `_holds_ref` walks the whole tree rather than the part that gets emitted, so `(a)(?P=1){0}` carries slots for a reference that is never written. That is a few wasted slots and not a wrong answer, and the alternative is deciding the shape of the program from something the walk that writes it has not reached yet.

## 6. The bound, and what a caller is told

Every other place this library gives up has a second engine underneath it. A row too long for the bitmap goes to the machine. A state cache that runs out of states goes to the machine. Here there is nothing underneath, because the machine is the thing that cannot answer the question at all.

So the bound is a count of steps. `MAX_STEPS` is 1 << 22, four million, which is sixteen times the number of visits `MAX_CELLS` allows. The step is charged in `_attempt` at the point the walk moves rather than at a push, because what is being bounded is the walking and a push that is never popped costs nothing. Running out sets `Bounded.overrun`, which stops the scan at that row rather than trying the later positions, since nothing later in the row is going to be cheaper than what has already been given up on.

It is a backstop rather than the working bound. The bitmap of section 3 is what stops an ordinary walk, and after that rule went in no pattern in the corpus reaches this count at all. What is left for it is the two shapes the bitmap cannot cover: a row too long to have one, and a program where a slot really does change on every turn of a loop, which is the exponential backtracking a backreference makes possible and which no bitmap was ever going to fix.

`searched` and `located` then raise, with `this pattern is taking too long on this row`. A sentence about the pair rather than about the pattern, because a backreference that is cheap on most rows and ruinous on one is the ordinary case rather than the odd one.

That is a divergence and it is deliberate. Upstream says nothing at all here and keeps going, which on the same pair is a call that does not come back. A library that can be made to hang inside a kernel by one cell of one column is worse than a library that says it gave up, and the number is high enough that no ordinary row reaches it.

The plumbing this needed is a `raises` on eight kernel entry points and four morsel bodies. `parallel_morsels` already propagates the first failure out of a morsel body, and a job small enough to fit in one morsel runs inline so its error comes off the caller's own stack, so nothing new was needed under the kernels.

## 7. The ignore case flag

A backreference under `(?i)` does not compare the two characters the way a literal under `(?i)` does, and that is measured rather than assumed. A literal is compared by its fold orbit and a reference is compared by simple lowercase, so on CPython 3.13 `(?i)ss` matches the long s and `(?i)(s)\1` does not, `(?i)(σ)\1` does not match sigma followed by final sigma, and `(?i)(é)\1` and `(?i)(İ)\1` and `(?i)(ß)\1` all do match their case pairs.

Under the ASCII alphabet that is a subtraction: `(?ai)(a)\1` matches `aA`, and the twenty six letters are the whole of it. That is what `IN_REF` with `b` set to 1 does and it is written.

Under the wide alphabet it wants a simple lowercase table, which is `Py_UNICODE_TOLOWER` over roughly 2927 cased code points. This library carries the fold tables and not that one, and folding where upstream lowers would be answering the question wrongly rather than answering it. So `(?i)(a)\1` is refused as a gap, with `this engine has no backreference under the ignore case flag yet`, and the table is the next slice.

The flag is read in `_emit_node` rather than in `_check_node`, which is the only construct in the file that does that. The reason is that the flag is scoped and the checking walk enters bodies with a scope already taken off it. `(?i:(a)\1)` has the reference inside the scope and is refused, and `(?i:(a))\1` has it outside and is answered, and only the walk that emits knows which of the two it is standing in.

## 8. Reading the text again

`_reads_again` compares the two stretches code point by code point. It reads the row through `point_at`, the same accessor every other instruction in the file uses, so a lead of unreadable bytes in front of a cut row is handled the same way it is handled everywhere else.

Characters and not bytes. The two ends of a group are positions in characters, so `(ß)\1` compares one character against one character, where a width in bytes would take half of the sharp s and compare it against nothing.

## 9. The bracket that used to renumber

`match` and `fullmatch` on this engine are answered by a rewritten pattern, because upstream answers them with `re.match` and `re.fullmatch`, which anchor from outside the pattern, and this library has to put the anchor inside it. `python_anchored` writes `\A(` and `)` around the pattern, and that bracket used to be a capturing one.

A capturing bracket around the whole pattern numbers every group the caller wrote one higher, and a reference follows the numbering. So `(a)\1` came out as `\A((a)\1)\Z` and the reference named the wrapper instead of the group beside it. The wrapper is still open where the reference stands, so it reads as a group that never took part, and the pattern quietly matches nothing at all. That is a wrong answer and not a refusal.

It never showed, because a backreference was refused on both engines until this slice, and it stopped being harmless the moment one of them could read it. The bracket is now `(?:`, which changes nothing anywhere else and removes the hazard rather than working around it. This is this library's own defect and is written down here rather than filed anywhere, because nobody outside could reach it.

## 10. What the other engines do about it

RE2 has no such syntax, so `compile_program` on the RE2 engine refuses it with `RE2 has no backreference` and that refusal is agreement rather than a shortfall. `holds_unsupported` sees a backreference in the tree, so a call naming one is routed to Python's engine without the caller passing a flag, which is the same route the lookaround takes.

The state cache refuses `IN_REF` while it is scanning the program, with `the pattern asks about what it matched before`. The Pike machine drops the thread in `_queue`, and drops it rather than answering it for the reason section 2 gives. Neither refusal is ever reached through the compiler, since the backtracker takes every such program and never hands one back, and they are written where a reader walking either file will meet them.

One combination is refused. A pattern holding both a lookaround and a backreference has nowhere to go: the machine runs the inner search of a lookaround and the backtracker is the only engine that reads a reference, and the backtracker has one stack and one bitmap and no place to put a search inside a search. So `(?=a)(b)\1` is refused with `this engine has no lookaround beside a backreference yet`, as a gap. The check is made over the emitted instructions rather than over the tree, because a construct under a repeat of zero is in the tree and not in the program, so `(?=a){0}(b)\1` is answered.

## 11. What this cost the corpus

The sweep asks pandas about 30052 patterns over sixteen texts, through `str.findall`, and compares every answer. After this slice it compares 28729 of them and holds out 1323, with no disagreement in any of the ones it compares.

The 1323 break down as 399 for a named character, 393 for a possessive quantifier, 202 for a conditional group, 196 for an atomic group and 69 for a repeat count this engine will not honour, which are the five constructs that were already refused and are not this document's business. The other three are this slice's own: 28 for a lookaround standing beside a backreference, which is section 10, 10 for a backreference under the wide reading of the ignore case flag, which is section 7, and 26 for a capture inside a lookahead, which is document 93's and is unchanged.

Before this slice it compared 27957 and held out 2095, of which 839 were held out for a backreference. So 772 patterns moved from held out to compared, which is what the two totals moved by. Of the 839, 772 compile and answer, 38 are refused for one of the two new reasons above, and the remaining 29 hold a second construct this engine has not got and are held out for that one instead, which is the same bookkeeping document 94 wrote down and the reason the other buckets each moved by a few. Nothing that used to be compared stopped being compared and nothing that used to agree stopped agreeing.

The three patterns that made section 3 what it is came out of this sweep. With the bitmap turned off rather than stamped, `(?P<n>(|))(?P=n)*`, `(?P<n>\b)(?P=n)*` and `(?P<n>\b)(?P=n)+?[a-z]` all walked to the four million step bound and were reported as having run out of steps, and all three are patterns Python answers at once. Each of them is a repeat with no bound over a body that matches nothing, and a bitmap that drops the second arrival at an instruction and a position had been quietly ending that loop everywhere else in the library for as long as there had been one. With the stamped bitmap in place all three are answered and no pattern in the corpus reaches the step count at all.

That is the reason the narrower rule is worth its extra twenty lines rather than the bound being left to catch them. A count that no ordinary pattern reaches is a backstop. A count that three patterns in thirty thousand reach is a rule, and a rule that says a loop is a timeout is the wrong rule.

## 12. What is not here

The simple lowercase table, which is section 7 and is the next slice. It wants a generator beside `tools/gen_regexfold.py` emitting `Py_UNICODE_TOLOWER` over the cased code points, and until it exists the wide reading of `(?i)` beside a backreference is a gap.

The lookaround beside a backreference, which is section 10 and which wants the backtracker to be able to run a nested search, or the two engines to be able to hand a row between them in the middle of a match rather than only at the start of one.

The three constructs that are left after this slice, which are the named character at 399 patterns, the possessive quantifier at 393, the conditional group at 200 and the atomic group at 194, with 70 more held out for a repeat count this engine will not honour. None of those is a question about the path, so none of them is a document about an engine.

The measurement of what turning the bitmap off costs. It is not free and it is not measured, and the honest statement is that a program with a reference in it is slower than one without by an amount nobody here has put a number on. The bound says it cannot be unbounded and says nothing about what it is.

The sixteen texts, which every document since 90 has named and which are still sixteen.
