# 100. An alternation nobody chooses

## 1. What this is

The engine that copies Python now answers the conditional group. `(a)?(?(1)b|c)` reads a `b` when group one took part and a `c` when it did not, so the pattern picks its own arm out of what the path has already done rather than out of what the text in front of it says.

It is the third construct in this accessor that decides which engine answers rather than what the answer is, and it is the first reason again rather than the second. Document 95 was a question two of the three engines cannot answer, because a state there is a set of instructions and a merge there is two threads at the same instruction and the same position, and neither of those holds a path. Document 99 was an instruction two of the three engines cannot obey, because there are no separable choices in a walk that is following all of them at once. This one is a question, so it goes where the backreference went and runs under the arrangement the backreference already runs under.

What it does not share with a backreference is the cost. A backreference reads the text a second time and compares it. A test looks at two numbers and goes one way or the other, so it is a comparison and a branch and nothing else, and it costs no backtracking of its own at all.

## 2. Why it is only reachable through a flag

The same door document 99 opened. pandas picks the engine for a pattern method by walking the pattern and asking whether it holds a lookaround or a backreference. A conditional holds neither, so a call with no flags on it goes to Arrow, Arrow is RE2, and RE2 has never had the construct. `Series.str.contains(r"(a)(?(1)b|c)")` on pandas 3.0.6 raises `ArrowInvalid: Invalid regular expression: invalid perl operator: (?(`. This library agrees, in its own words, because the router in `route.mojo` is a copy of that walk and the construct was already refused on RE2 with a sentence of its own.

Naming a flag is what moves the call, because a non empty `flags` is what moves pandas onto `re`, and `re` has answered this construct since long before any version this library supports. `case=False` is not that door, for the reason it is not that door next to it: pandas turns it into something Arrow can read rather than into a move onto `re`, and the measurement above with `case=False` on the end raises the same `ArrowInvalid`.

`extract` is the one name that needs no flag, because `extract` never reaches Arrow on either library. pandas compiles the pattern with `re` and loops in Python, and `program_for` here compiles for Python's engine before it asks the router anything. So `Series.str.extract(r"(a)?(?(1)b|c)")` answers on both with no argument at all, which is the one place in this slice a caller meets the construct without knowing about flags.

## 3. An alternation nobody chooses

The shape written into the program is an alternation with the choosing taken out. A test is written first. The arm for a group that took part is written after it, so that falling through to the next instruction is the first arm. A jump over the rest ends that arm. The arm for a group that did not take part is what the test jumps to.

`IN_TEST` carries the first slot of the group asked about in `a`, which is the group number doubled and is the same number an `IN_REF` carries, and the pc for a group that did not take part in `b`. Falling through for the yes arm rather than jumping to it is what keeps the arm the caller wrote first written first, which matters to nothing but a person reading the disassembly, and is free.

The difference from `_emit_branch` next door is the whole reason the construct costs no backtracking. A branch pushes the arm not taken so that a failure later can come back for it, and patches a jump at the end of each arm. A test pushes nothing. Exactly one arm is entered and the other is never looked at again. Nothing is patched twice.

A conditional written with one arm has the other arm empty, so the test jumps to the end of the construct and the pattern goes on. `(a)?(?(1)b)` therefore matches the empty string wherever there is no `ab`, which is upstream's answer for it and is the shape that makes the counting and replacing scans the interesting half of the test file.

## 4. Two ways a group can fail to hold text

The test asks whether the group took part, which is not the same question as whether it holds any text, and the two measurements that separate those are the two rows worth keeping.

A group that matched nothing has taken part. `(a*)` matches the empty string in front of a `b`, so both of its slots are written and both are zero, so `(a*)(?(1)b|c)` takes the first arm and reads the `b`. Against `ac` there is then no way to reach the second arm at all, and upstream agrees that the pattern does not match there.

A group the pattern has not opened yet has not taken part. `(?(1)a|b)(x)` asks about a group that is opened after it and takes the second arm every time. Nothing in the compiler says so: the slots start at minus one and the test reads them, so a group that has not been reached reads exactly like one that was skipped. That is the same reading `IN_REF` gives an untaken group, and it is worth saying out loud that it is the same reading rather than a second rule.

A group the pattern does not have at all is not a pattern. `(?(2)a|b)` is refused by the parser here and refused by upstream at compile time with `invalid group reference 2`, and the two refusals turn out not to be the same class. `re.error` is a subclass of `Exception` rather than of `ValueError`, so a pattern `re` cannot compile comes out of pandas as something a caller catching `ValueError` does not catch, while every pattern this parser turns down is a `ValueError`. That is an older divergence than this slice and it is on the list to file upstream, and the test row that measures it says so rather than papering over it.

## 5. What it costs the bitmap

Everything a backreference costs it, and the witness is short enough to write down in full.

`(?:(a)|a)(?(1)b|c)` against `ac` has two ways to read the leading `a`. The first arm of the bracket sets group one and the second does not, and both of them arrive at the test at position one. The first to arrive takes the yes arm, wants a `b`, and fails. If the arrival were remembered, the second one would be dropped as a repeat of a question already answered, the no arm would never be entered, and the `c` would never be read. Upstream matches `ac` there. So the arrival is not the same question, the pair of an instruction and a position does not say what happens next, and `memo` goes off for a program holding a test exactly as it goes off for one holding a reference.

That is the opposite of what document 99 found for the cut, and the two arguments are worth reading next to each other. What an atomic group matches from a position is a function of the position, so a second arrival really is the same question. Which arm a conditional takes from a position is not a function of the position at all.

The step count underneath is what bounds the walk instead, which is the arrangement document 95 built and needs nothing added to it. `(a)?(?(1)b|c)*` is a repeat with no bound over a body that can match nothing and it answers rather than running away, because the bound that stops it is the count rather than the bitmap.

The slots come on whether the caller asked for them or not, by the line that already did that for a backreference. `_holds_ref` is now a walk for either construct, since an instruction that reads a slot and a slot that was never written are not two things to put together and hope. The function has a name from before there were two of them.

## 6. The three refusals

The state cache refuses a program holding a test with a sentence of its own, `the pattern asks whether a group took part`, which sits beside the two it already had. A set of instructions is not a path and there is nothing in one to ask.

The Pike machine drops the thread, which is what it does for a reference and for a cut, and the comment there says the same thing a third time: two merged threads can answer this differently, so the test would have to be taken after the merge, and the merge is the thing there is to keep.

RE2 refuses it at the check over the tree, in the words it used before this slice, `RE2 has no conditional group`. That refusal is load bearing rather than left over, because it is what a caller who wrote no flags gets and it is what keeps this library routing the way pandas routes.

The compiler says all of this on a flag rather than in a sentence. `Program.asks` is the third of the three and it is set by the same scan over the emitted code that sets the other two.

## 7. The lookaround beside it, and the width inside one

A lookaround beside a conditional is refused, with a sentence of its own, `this engine has no lookaround beside a conditional group yet`. The reason is the reason the other two pairings are refused. A lookaround is a search inside a search and the backtracker has one stack to run it on, so a program holding one goes to the machine, and the machine cannot answer a test. A pattern holding both has nowhere to go.

The naming is in the order the constructs landed rather than in the order they appear in the pattern, so a pattern holding a reference as well is named by the reference. Nothing but a person reads the sentence, and a person told about either of the two has been told what to take out.

The width walk inside a lookbehind is the part of this that is not obvious. A lookbehind is run from a position worked out by subtracting the width of its body, so a body without one is refused, and a conditional is as wide as its arms when they agree and has no width when they do not. That was measured against a running 3.13: `(?P<n>a)(?<=(?(1)b|c))` compiles, `(?P<n>a)(?<=(?(1)b|cc))` and `(?P<n>a)(?<=(?(1)b))` do not.

Every one of those three is refused here in the end, so it would be easy to read the width case as dead code and delete it. It is not, because the two refusals are not the same refusal. The width one is a `ValueError` and is agreement with a rule of the language. The other is a gap and is this library falling short. Asking the width first is what keeps a pattern upstream refuses out of the gap bucket, and what keeps a pattern upstream compiles out of the `ValueError` bucket, and those two buckets are the thing the differential counts.

## 8. What the corpus said

200 of the 30052 generated patterns were held out with the conditional group as the thing in the way. 194 of them are compared now and 0 disagree, and the other 6 moved to the new lookaround pairing, which is the honest place for them rather than a win.

The whole run reads 29462 patterns compared, 590 held out and 0 disagreements against a ceiling of 0. The held out buckets are 399 for a named character, 77 for a repeat counted higher than this compiler will unroll, 41 each for a lookaround beside a backreference and beside an atomic group, 26 for a capture inside a lookahead, and the 6 above.

A much larger number was measured on the way to that one and is worth writing down so that the next person to count does not think they have found something. 7145 corpus patterns hold the three characters `(?(` and only 200 of them are conditional groups this library had anything to say about. The rest are patterns Python's grammar cannot read, mostly a conditional naming a group the pattern never opens, and they were already being compared rather than held out, because a pattern this parser cannot read is a pattern upstream cannot read and both sides raise. Counting the construct by looking for it in the text of a pattern overcounts it by a factor of thirty five here.

## 9. What is not here

The lookaround beside it, which is now the third pairing waiting on the same thing the other two wait on: either a backtracker that can run a nested search without giving up its stack, or two engines that can hand a row between them in the middle of a match.

No measurement of what the new instruction costs a program that has none. It is one more comparison in a dispatch chain that three engines share, and document 96 is the document that found such a comparison costing about 1.4 times on the Pike machine, so the question is a real one even though the answer here is almost certainly nothing. It is asked in the same breath as the one document 99 left open about the mark and the cut.

The fast path on `case=False` that document 86 asked for and documents 97 and 99 repeated. It is still owed and this slice did not move it.

What is left after this on the engine that copies Python is the named character at 399 patterns, the repeat counted higher than the compiler will unroll at 77, the capture inside a lookahead at 26, and the three lookaround pairings at 41, 41 and 6. The RE2 grammar front end is still much the largest single gap in the accessor and is not on this engine at all.
