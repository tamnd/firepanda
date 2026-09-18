# 94. A question about the text behind

## 1. What this is

The engine that copies Python now reads a lookbehind. `(?<=a)b` matches a `b` that has an `a` in front of it, and the `a` is not consumed, so the match is one character wide and it starts at the `b`. `(?<!a)b` is the same question with the answer inverted, and it holds at the front of a row where there is nothing in front of the `b` at all, which is the case that makes a negative lookbehind different from `[^a]b` rather than a shorter spelling of it.

Document 93 landed the other half of this construct and said what this half would need that that one did not. This document is the answer to that sentence, and it is short for a reason worth saying at the top: almost none of the machinery is new. The instruction is a copy of the one next door with one number added to it, the second machine is the same second machine with a different seed, and the emitter is the same emitter. What is new is one function in the compiler, and that one function is the whole of the slice.

## 2. The one thing that is different

A lookahead starts its body where the thread is standing. That is why document 93 could answer one without the compiler knowing anything at all about the body: the position is to hand, the body either matches from it or does not, and the answer comes back a boolean.

A lookbehind has to start its body somewhere else. The body has to end where the thread is standing, and a Pike machine walks forwards, so the only way to run a body that ends at a known place is to start it at a known place, which means knowing how far back to start. That is a number the compiler has to produce, and it can only produce it for a body that always reads the same number of characters however it matches.

Python asks the same question and refuses the same bodies, with `look-behind requires fixed-width pattern`. So this is not a limitation this library is choosing on grounds of implementation. It is a rule of the language being copied, it is visible to every caller of `re`, and a lookbehind engine that accepted `(?<=a*)b` would be answering a pattern pandas raises for, which is a divergence in the direction this library never takes.

## 3. The width analysis

`_fixed_width` in `program.mojo` walks one node and answers how many characters it always reads, or minus one when the answer is not always the same number. `_children_width` beside it sums that over a node's children, which is the list `_emit_children` writes out, and every caller with a body rather than an atom goes through it.

The rules are short enough to list and each one is a place to be wrong.

A literal, a negated literal, a dot, a set, a range and a category all read one character. The members of a set are not walked into, because a set reads one character whatever is written inside it, and `[abcdef]` is one and not six.

A position test reads nothing. So does an assertion of either kind, including a lookahead and another lookbehind, and so does the node the parser writes for an empty negative lookaround. That last one is why `(?<=(?!))` compiles here, and upstream compiles it too.

A sequence, a capturing group, a scoped flag group and an atomic group are the sum over their children, and any child without a fixed width takes the whole thing with it. That is why `(?<=ab*c)d` is refused although two of its three pieces are perfectly fixed.

An alternation is fixed when every arm is fixed at the same number. `(?<=a|b)c` reads and `(?<=a|bc)d` does not, and `(?<=ab|cd)e` reads, which is the row that separates counting the arms from comparing them.

A repeat is fixed when its two bounds are the same number, and then it is that number of copies of its body. `a{2}` and `a{2,2}` are fixed and `a{2,3}` is not, and `?`, `*`, `+` and the lazy and possessive spellings of all of them are not, because each of those is a repeat whose bounds differ.

Anything else is minus one. That is the right default for a walk whose failure mode is silently handing out a start position that does not exist.

The refusal set this produces was diffed against Python 3.13 shape by shape over thirty bodies chosen to sit on every edge of the rules above, and it matched on all thirty. That diff is the evidence for this section, and the test file carries the same shapes so that the day either side moves is a day something fails.

## 4. The instruction

`IN_BEHIND` is instruction 12. Its `a` is where the body starts, the same as `IN_LOOK`'s, and its `b` carries two numbers, because an instruction has two payloads and this one wants three. The width is `b >> 1` and the sign is `b & 1`.

Packing them is worth a paragraph because the alternative looks cleaner. `Instruction` is an op and two `Int32` payloads, twelve bytes, and every instruction in every program in the library is that size. Adding a third payload for the benefit of one instruction would grow every program by a third, including every program that holds no lookbehind at all, which is nearly all of them. A shift and a mask in the one branch that reads the field is a better trade than four bytes on every instruction ever emitted, and the field is documented where the constant is declared so that nobody has to work the encoding out from the branch.

A separate instruction rather than a bit on `IN_LOOK` is the opposite call from the one document 93 made about the two directions of a lookahead, and the reason is that these two carry different things. The positive and negative forms of one direction carry identical payloads and differ in a comparison, so one instruction and a bit is right. The two directions carry different payloads and are decoded differently, so two instructions is right. The rule is about the shape of what is stored and not about how alike the constructs look on the page.

## 5. The same second machine

`_looks` in `pike.mojo` answers both directions and does not know which one it is answering. It takes a position and a body and says whether the body matches starting exactly there, and the seed is the whole of the difference: a lookahead hands it the position the thread is standing at, and a lookbehind hands it that position less the width.

Nothing in the machine checks that the body ends where the thread is. It does not have to. The width is the number of characters the body always reads, so a body that matches at all from `position - width` ends at `position` by arithmetic, and a check would be asserting the thing the compiler has already refused every counterexample to.

A position with less text behind it than the width has nowhere to start, and that is answered without running anything: the guard is `position - width >= 0` in the branch, a False, and the sign then decides. This is not an optimisation, it is the only correct answer. `(?<!ab)c` holds against `bc` because there is no room behind the `c` for an `ab` to be, and a machine that started the body at a negative position would be reading something that is not there.

A body of no width is a width of zero and starts where the thread already is, which is how `(?<=^)` and `(?<=\b)` come out right by the ordinary rule rather than as cases of their own. That is the payoff for making an assertion contribute zero in section 3 rather than refusing it, and it is also why `(?<=(?=a)a)b` reads: the inner lookahead is worth nothing, the `a` is worth one, and the body is started one character back.

## 6. The early exit

`_looks` used to walk to the end of the row whichever way the answer was going. The loop ended when the position ran off the end, so a body that had died three characters in kept stepping over the rest of the row asking nothing of it.

A lookbehind makes that expensive in a way a lookahead did not quite. The body of a lookbehind has a known width, so it can never survive past `width` steps, and every step after that is a walk over a row length of text to learn something the second step already knew. On a long row with a lookbehind at every position that is the row length squared, for a construct whose whole cost should be the width.

The fix is one clause: when no thread survives a pass, there is nothing to revive and the answer is False. That is true of the outer machine too and the outer machine cannot use it, because the outer machine seeds a fresh attempt at every position and so always has something live. The nested one seeds once and so has a reason to stop. A lookahead wins the same way whenever its body is short, which most bodies are, and this is the one performance change in the slice.

## 7. What the other engines do about it

The DFA cache refuses any program holding an `IN_BEHIND`, in its constructor, alongside the `IN_LOOK` it already refused. The two are one branch now and the message says the text around rather than the text ahead. The reason is the one document 93 gave: the instruction runs a second machine over text outside the window before deciding whether a thread goes on, and a DFA that could do that is a DFA with the text in it.

The backtracker refuses it in its constructor too, in the same branch and for the same reason, which is that it holds one stack and one bitmap and a lookaround is a search inside a search. A program holding either instruction is handed straight back and the Pike machine answers it. Refusing at the moment the program is sized is again what matters, because the alternative is the instruction falling through to the branch that asks whether the character at the cursor is accepted, which would answer something rather than nothing.

`_first_ranges` has a catch-all that returns no optimisation for any instruction it does not recognise, and `IN_BEHIND` falls into it. That is correct and not a missed opportunity, for a sharper reason than it was for the lookahead: a lookbehind can be the first thing in a pattern and the characters it permits are not characters the pattern reads at all, they are characters already behind it.

RE2 refuses a lookaround of either direction, with the message it has always had, and the router sends every pattern holding one to Python because pandas does. Nothing in this slice touches either of those. The refusal on RE2's engine is what makes the router's decision safe rather than merely correct.

## 8. Where the refusal sits

The width refusal is a `ValueError` and not a gap, and it is the only refusal on this engine that is not one.

That distinction is the whole of what the `gap` flag on a program is for. A gap is a pattern pandas answers and this library does not, which is a shortfall here and reaches a caller as a `NotImplementedError` saying so. A refusal that is not a gap is a pattern nobody answers, which reaches a caller as a `ValueError`, and telling such a caller that the feature is not implemented yet would be telling them to wait for something that is never coming.

Every other construct this engine refuses is a gap. This one is agreement. A caller who writes `(?<=a*)b` gets a `ValueError` here and a `re.PatternError` from pandas, and the difference in class is the divergence document 86 registered rather than anything this slice introduced, since `re.PatternError` has `Exception` as its only base and this library raises a `ValueError` for every pattern either engine refuses.

The capture refusal is a gap, and it is the same rule the lookahead has and for the same reason, which is that the nested machine carries no slots and reporting nothing would be a wrong answer rather than a refusal. The message names the direction now rather than always saying lookahead, so a caller with one of each in a pattern is told which one it is about.

## 9. What this cost the corpus

The sweep asks 30052 patterns over 16 texts through `str.findall` and compares every answer against pandas. Before this slice it compared 27643 and held out 2409, of which 336 were held out for a lookbehind. It now compares 27957 and holds out 2095, and none of the 2095 is a lookbehind.

Of the 336, 290 compile and answer, 26 are refused for a body without one width, which the sweep counts as agreement because pandas refuses them too, and the remaining 20 turn out to hold a second construct this engine has not got and are held out for that one instead. That last group is why the other buckets each went up by a few while the total went down by 314: a pattern refused for its lookbehind is now refused for whatever the next thing in it is, and the sweep counts the first reason it finds.

Zero disagreements on 27957 patterns over 16 texts is the number that matters, and it is the same number this engine has reported since document 90.

## 10. What is not here

The capture inside a lookaround, in either direction, which needs the nested machine to carry slots and to merge them back into the outer thread's on success, and which is only reachable through `extract`.

The buffers, which document 93 section 6 named and this slice did not touch. They are still allocated per call and the early exit in section 6 above makes the allocation a larger share of what a nested call costs rather than a smaller one, which strengthens the case for measuring it and does not change the fact that it is a measurement rather than a design.

The three constructs that are left. After this slice the Python engine refuses 839 patterns for a backreference, 399 for a named character it has no table for, 393 for a possessive quantifier, 200 for a conditional group, 194 for an atomic group and 70 for a repeat count it will not honour. The backreference is the big one and it is the one document 93 explained why a Pike machine cannot have, so it is the next document about an engine rather than the next document about a construct.

The sixteen texts. Documents 90, 91, 92 and 93 all said it and it is still true, and a lookbehind is if anything more exposed to it than a lookahead was, because the interesting cases are about what is behind a position and the front of a row is a place every one of the sixteen texts has exactly one of.
