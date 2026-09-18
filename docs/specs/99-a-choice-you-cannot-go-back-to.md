# 99. A choice you cannot go back to

## 1. What this is

The engine that copies Python now answers the atomic group and the possessive quantifier. `(?>a*)b` says take every `a` there is and never give one of them back, `a*+b` says the same thing in two characters fewer, and both of them are 589 of the 30052 held out patterns, which is the largest thing left after the named character. They were the two largest remaining and they are one construct, so they are one slice.

Document 95 was about a construct two of the three engines cannot answer. This one is about a construct two of the three engines cannot obey, and the difference between those two sentences is the whole of why the document exists. A backreference is a question: what did the path that arrived match. A cut is an instruction: the ways this group could have matched instead are gone. The first needs an engine that keeps a path so that there is something to ask. The second needs an engine that keeps its choices somewhere you can reach into, so that there is something to throw away.

## 2. Why it is only reachable through a flag

pandas picks the engine for a pattern method by walking the pattern and asking whether it holds a lookaround or a backreference. An atomic group holds neither, so a call with no flags on it goes to Arrow, Arrow is RE2, and RE2 has never had either construct. `Series.str.contains(r"(?>a*)b")` on pandas 3.0.6 raises `ArrowInvalid: Invalid regular expression: invalid perl operator: (?>`, and `a*+b` raises `bad repetition operator: *+`. This library agrees, in its own words, because the router in `route.mojo` is a copy of that walk and the two constructs were already refused on RE2 with a sentence each.

Naming a flag is what moves the call. pandas hands a call with a non empty `flags` to `re`, and `re` has answered both constructs since Python 3.11. So `Series.str.contains(r"(?>a*)b", flags=re.IGNORECASE)` answers on pandas and now answers here, and that is the whole of the door. The router does not change and must not, because changing it would make this library answer a call pandas refuses.

`case=False` is not that door, which is worth writing down because it looks like it should be. pandas turns `case=False` into something Arrow can read rather than into a move onto `re`, so a call that says only `case=False` still reaches RE2 and still raises. It behaves differently for a backreference, where the pattern itself is what moved the call, and the difference is upstream's rather than this library's.

## 3. One construct, two spellings

`a*+` is `(?>a*)` and is compiled as exactly that: a mark, a greedy repeat, and a cut. That is upstream's reading of it too, and taking the same reading means there is one mechanism to get right rather than two that have to be kept agreeing. `_emit_node` has four lines for `OP_POSSESSIVE_REPEAT` and three of them are the same three `OP_ATOMIC_GROUP` has.

The counted forms come out of it for free. `a{2,}+` is a mark, the repeat `_emit_repeat` already writes for `a{2,}`, and a cut, and the repeat budget `_check_node` spends on a counted repeat is spent on the possessive one the same way, since a possessive repeat is unrolled into copies exactly as a greedy one is.

## 4. The cut on an explicit stack

The backtracker holds its choices on two parallel lists, `jobs_pc` and `jobs_at`. An entry is an instruction to walk, or, when the instruction is negative, a slot to put back and the value to put back into it. One test at the top of the loop tells those apart.

A mark is a third kind of entry written the same way, `MARKED`, which is a number far below any slot a program could have, so it costs the loop no second comparison. `IN_MARK` pushes it and then pushes the instruction after it. `IN_CUT` walks down to the nearest `MARKED` and compacts everything above it in place.

What is thrown away is the choices, which is exactly what the construct says. What is kept is the saves, in the order they were made, and that is the part a reader should slow down for. A save is not a choice: it is what a slot held before the group wrote to it, and it is still owed to the pattern outside the group, because the group as a whole can still fail on what comes after it. `(?>(a+))b` against `aac` has the group match `aa`, the cut fire, the `b` fail, and the whole pattern fail, and group one has to read as untaken afterwards rather than as `aa`. Keeping the saves in order and dropping everything else is one pass over the group's part of the stack.

The nearest mark is always the right mark, and nothing has to be numbered. A group nested inside this one has either reached its own cut, which took its mark off, or failed, which popped its mark off, and either way it is gone before the outer cut runs. `(?>a(?>b)c)` and `(?>(?>a)|b)c` were the two shapes that were checked against that argument, and `(?>(?>ab|a)b)c` is the row in the test file that fails if a cut ever reaches past its own mark.

A choice made before the mark is below the mark and survives. That is why `(?>a)?ab` matches `ab`: the question mark belongs to the pattern rather than to the group, so the arm that skips the group was pushed before the mark was.

## 5. What it does not cost the bitmap

A backreference cost the bitmap most of what it was worth, and this does not cost it anything, for a reason worth stating rather than assuming.

The bitmap drops a second arrival at an instruction and a position because the pair says everything about what is left to do. A backreference breaks that because the instruction reads a slot the arrival wrote. A cut does not, because what an atomic group matches from a position is a function of the position and nothing else: that is what being atomic means. The second arrival at the group's first instruction really is the same question as the first, and the answer it gets is the answer the first one got. So `memo` stays on for a program holding a cut, and `(?>a*)*b` is answered rather than looped on, which is a repeat with no bound over a body that can match nothing and would otherwise never end.

The one thing that does change is the row that is too long for the bitmap. Everywhere else that is a `GAVE_UP` and the machine takes the row. There is no machine to take this one, for the same reason there is none for a backreference and by a different argument, so `Bounded` now carries `alone`, which is `program.refs or program.cuts` and means there is nothing underneath. A row too long for the bitmap on such a program runs with no bitmap at all and `MAX_STEPS` is the whole of the bound.

That splits the per row state into four flags where there were two. `memo` is per program and says whether an arrival may be dropped outright. `bitmap` is per row and says whether there is a bitmap at all. `stamped` is per row and says whether the bitmap is in use under the backreference's narrower rule. `counting` is per row and says whether the step count is bounding this walk, which is whenever the bitmap is not bounding it outright. All four are read in the loop as fields rather than worked out there, so the ordinary program with none of this in it pays nothing.

## 6. The three refusals

The state cache and the Pike machine both turn a program holding a cut down, in their own words, and the words say what kind of thing is in the way rather than naming the syntax.

The cache says the pattern throws away a choice it could have made. A state there is a set of instructions and the set is where every choice the pattern has lives at once, so throwing some of them away and keeping the rest is not a thing that can be said about a set with no paths in it.

The machine kills the thread, with a comment rather than a message, because it has no way of reporting anything mid walk. The reason is the merge: the threads that stand for the choices a cut would throw away are the other threads in the same list, and some of them belong to paths that never entered the group. Obeying the cut would mean knowing which, and knowing which is the merge undone, and the merge is the whole of the machine's bound.

The third refusal is the compiler's, and it is not a message. `Program.cuts` is set by the same post emit scan that sets `Program.refs`, and `column.mojo` reads the pair to decide that the row goes to the backtracker rather than being asked of the cache and handed on.

## 7. The lookaround beside it

A lookaround is a search inside a search, and the backtracker has one stack and one bitmap to run it on, so a program holding one goes to the machine. The machine cannot obey a cut. A program holding both has nowhere to go and is refused, which is exactly the shape document 95 section 10 left behind for the backreference, arrived at from the other side.

The refusal says which of the two was in the way, `this engine has no lookaround beside an atomic group yet` and `this engine has no lookaround beside a backreference yet`, because a reader who has just written a pattern wants to know which half to take out. It is 41 patterns of the corpus, and the backreference's is now 33, and the two together are what a nested search would buy.

## 8. What the corpus said

Before this slice the Python differential compared 28739 of the 30052 patterns and held out 1313. After it, it compares 29268 and holds out 784, at zero disagreements.

The 589 patterns that were the atomic group and the possessive quantifier did not all become comparable, and where the rest of them went is the interesting half of the number. 41 went to the new lookaround bucket. 7 went to the conditional group, which is now 209, and 5 to the lookaround beside a backreference, which is now 33, and 7 to the repeat this compiler will not unroll, which is now 76. Those are patterns that held two refused things and were being reported under the first one met. That leaves 529 newly compared, and 28739 plus 529 is 29268.

One of them found a defect that was already there. `\-{2}+|(?<=漢)|(?P<n>\123)(?P=n)|0|(?<=(?(1)^))` has a conditional group inside a lookbehind, and `_fixed_width` had no case for a conditional, so it reported the lookbehind as having no fixed width and refused the pattern with a `ValueError` where upstream compiles it happily. It had been hidden behind the possessive quantifier at the front of the pattern. A conditional is an alternation between two arms, so it has a width when its arms agree, and the arm nobody wrote reads nothing. `re.compile(r"(?P<n>a)(?<=(?(1)^))")` is fine upstream and `re.compile(r"(?P<n>a)(?<=(?(1)b))")` is not, and those two rows are the rule. This library still refuses the construct further down, and the point of asking the width is that the refusal a caller sees is the one about the conditional rather than one about a lookbehind upstream had no quarrel with.

Five differentials at zero disagreements, as before.

## 9. What is not here

The lookaround beside a cut, which is section 7 and is 41 patterns, and the lookaround beside a backreference, which is 33. Both want the same thing, which is either a nested search on the backtracker or a way for two engines to hand a row between them mid match.

A measurement of what the mark and the cut cost a program that has neither. The claim in section 5 is that they cost nothing, and the argument is that the two new instructions are never emitted and the four new flags are read as fields, but it is an argument rather than a number.

A measurement of what the cut costs a program that has one. The compaction is one pass over the group's part of the stack, so a pattern that enters an atomic group many times pays that many passes, and whether that is ever worse than the backtracking it saves has not been measured. The construct is usually written to save work rather than to spend it, which is a reason to expect it is not, not a reason to know.

The conditional group, which is 209 patterns and is the largest thing here after the named character now, and the named character itself at 399.

A fast path for a folded literal, which document 86 section 7 named and documents 97 and this one still name.

The sixteen texts, which every document since 90 has named and which are still sixteen.
