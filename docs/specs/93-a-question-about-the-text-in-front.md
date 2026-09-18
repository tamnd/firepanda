# 93. A question about the text in front

## 1. What this is

The engine that copies Python now reads a lookahead. `a(?=b)` matches an `a` that has a `b` after it, and the `b` is not consumed, so the match is one character wide and the position after it is the position after the `a`. `a(?!b)` is the same question with the answer inverted, and it holds at the end of a row where there is nothing after the `a` at all, which is the case that makes a negative lookahead different from `a[^b]` rather than a shorter spelling of it.

This is the first construct to land that RE2 has not got. Every earlier slice on this engine was about a flag, and a flag is a modifier on syntax both engines already read. A construct is a piece of syntax one engine reads and the other refuses, and the routing rule that document 82 wrote down was built for exactly this: the word carries the engine, and a pattern holding a lookaround is sent to Python because pandas sends it to Python. Until this slice the router sent it there and the compiler refused it, so the route existed and led nowhere. 700 of the 30052 patterns in the held out corpus were being turned down for that one message, plus 13 more for the empty negative form, which made it the largest block the Python engine was refusing that did not need a new table or a new invariant.

## 2. Why this one and not one of the other four

Python has five constructs RE2 has not got: a lookaround, a backreference, a conditional group, an atomic group and a possessive quantifier. The sweep counts them as 700 lookaround, 786 backreference, 392 possessive, 199 conditional and 188 atomic, so the backreference is the bigger number and it is not the one that landed.

A backreference asks what a group matched earlier in this attempt, and the machine in `pike.mojo` runs every live position in lockstep precisely so that it never has to ask that question, because a Pike machine has one thread per position and not one per path and there is no such thing as what this thread matched earlier. Reading a backreference means either backtracking, which is the thing document 77 refused to build on the grounds that a pattern arrives from a caller and `(a+)+b` on a backtracker is a denial of service with a friendly API in front of it, or carrying a copy of the slots down every branch and giving up the lockstep. That is a different engine and it wants its own document.

A lookahead wants none of that. It is a question about a position, the same shape of question as `^` and `\b` and every other thing `IN_AT` already answers, and the only difference is that answering it takes a program rather than a comparison. Nothing about the outer walk changes: the thread either goes on to the next instruction or dies, and either way the position does not move.

## 3. The instruction

`IN_LOOK` is instruction 11 in `program.mojo`. Its `a` is where the body starts and its `b` is 1 for `(?=...)` and 0 for `(?!...)`. It is resolved in the epsilon closure in `_queue`, next to `IN_AT` and `IN_SPLIT` and `IN_JUMP`, which is to say before any character is read and without advancing the position. The branch runs the body, compares the answer to `b`, and queues the next instruction when they agree.

Writing the polarity as a bit on one instruction rather than as two instructions is the whole of the difference between the positive and the negative form in this engine. The alternative, a separate `IN_LOOK_NOT`, would have been two branches in the closure doing the same work with one comparison flipped, and it would have to be kept in step forever by whoever touches either.

## 4. Where the body lives

The body is compiled into the same instruction list as everything else, immediately after the `IN_LOOK`, and it ends in an `IN_MATCH` of its own. An `IN_JUMP` is written in front of it and patched to the instruction after the body, so the outer walk steps over the body and never falls into it from the front. The emitter is six lines and it is `_emit_lookahead`.

Keeping the body in the same list rather than in a program of its own is worth a sentence, because a program of its own is the obvious design and it is worse. A separate program means a second set of ranges, a second table of slots and a second thing to serialise, and the only thing it buys is that the body is reachable by name. It is already reachable by index, which is what `IN_LOOK`'s `a` is, and an index into a list the machine is already holding costs nothing to follow.

The `IN_MATCH` at the end of the body is what the nested machine stops on. It is the same instruction the outer program ends on and it means the same thing, which is that the program this machine was given has been satisfied. Nothing distinguishes the two except which machine is looking at it.

## 5. The second machine, and why it reads the whole row

`_looks` in `pike.mojo` is a small Pike machine. It takes the program, the instruction the body starts at, the text, and the position the outer machine has got to, and it answers whether the body matches starting exactly there.

Two things about it are decisions rather than details.

It is seeded with one attempt, at the body's first instruction and at the caller's position, and it never seeds another. The outer machine seeds a fresh attempt at every position because it is looking for a match somewhere. This one is not: a lookahead asks whether the body matches **here**, and a machine that kept seeding would answer whether the body matches anywhere at or after here, which is a different and much weaker question. `(?=b)a` against `ab` is the row that separates them. There is a `b` ahead, so the weaker question says yes, and the real answer is no because the position the pattern has got to holds an `a`.

It reads the whole text rather than a slice of it starting at the position. Cutting the text off at the position and handing the tail to the nested machine is tempting, since the body cannot see behind itself anyway, and it is wrong in both directions. `a(?=b$)` against `abc` would say yes, because the `$` would be at the end of the slice. `(?=^a)a` against `ab` would say yes for the same reason in reverse, and it happens to be right there and wrong the moment the position is not zero. Anchors are questions about the row, so the nested machine has to be looking at the row.

Nothing comes back out except the boolean. The nested machine has its own slots and its own carry buffer and they are discarded when it returns, which is exactly why a capturing group inside a lookahead is refused rather than answered.

## 6. Nesting

A lookahead can hold a lookahead, and this works by recursion rather than by a case: the nested machine runs the same closure the outer one does, so when it reaches an `IN_LOOK` it calls `_looks` again. `(?=a(?=b))ab` is an ordinary pattern upstream and there is no reason for it to be special here.

The cost is that the buffers are allocated per call. The nested machine needs its own visited stamps and its own two lists of live threads, and it cannot borrow the outer machine's because the outer machine is in the middle of using them. Holding one set of buffers per nesting depth on the machine and indexing into them is the obvious fix and it is not here, because the depth is not known until the program is walked and the number of patterns that nest at all is small enough that the allocation may not show up at all. This is a thing to measure rather than a thing to guess, and it is named here so that whoever measures it knows it was left on purpose.

## 7. What is refused, and why each one

A lookbehind is refused, on Python's engine, with a message that says lookbehind rather than lookaround so that a caller can tell which half arrived. It is a different question and not a harder version of this one. Answering `(?<=a)b` means running the body against the text ending at the position, and since a Pike machine walks forwards that means knowing how wide the body is so that the start can be computed. Python computes that width at compile time and raises `look-behind requires fixed-width pattern` when it cannot, so reading a lookbehind means building a width analysis over the tree and reproducing that refusal exactly, including which shapes Python thinks have a fixed width. That is a slice.

A capturing group inside a lookahead is refused, and only when the caller asked for the groups. Upstream, a group inside a lookahead keeps what it matched and is readable afterwards even though the text it matched was not consumed. The nested machine here carries no slots, so it cannot report that, and an engine that quietly reported nothing would be giving a wrong answer rather than a refusal. The refusal is conditional on `captures` because the only caller who can see the difference is one who asked for the groups, and every other caller is answered. Through the accessor that is `extract`, which needs the groups because they are the answer, and `count` and `replace` on this engine, which need the slot holding the start of the whole match because Python's rule for where to look next is written in terms of it. So `contains("(?=(a))a")` answers and `extract("(?=(a))a")` raises.

A lookaround of any kind is still refused on RE2's engine, with the message it already had. That is agreement with pandas rather than a shortfall: a pattern holding one never reaches Arrow, because the router looks for the construct and sends the pattern to Python, and one that somehow did reach Arrow would raise there. The refusal on that engine is what makes the router's decision safe rather than merely correct.

## 8. The empty negative lookaround

`(?!)` is a negative lookahead with nothing in it, which means a question that is always false, which means a pattern that never matches anything. Upstream reads it as a pattern rather than as a mistake, and so does the parser here: CPython collapses it into a node of its own and this parser copies that, which is why it arrives as `OP_FAILURE` rather than as an `OP_ASSERT_NOT` with an empty body.

It compiles, on Python's engine, to an `IN_SET` with no ranges in it. A set with no members is a character test nothing passes, and `in_set` with a count of zero returns false without reading anything, so the thread dies at that instruction and the pattern never matches. That is four characters of change and it is the right four, because the alternative is a `IN_FAIL` instruction that exists to do what an empty set already does.

RE2's engine still refuses it, and the router still sends it to Arrow, because the router looks for assertions by name and after the collapse there is no assertion in the tree to find. So `contains("(?!)")` raises on both libraries and `extract("(?!)")` answers on both, and the two facts have the same cause. That is an upstream oddity rather than one of ours and it is in the observations list.

## 9. What the change cost the four places that read a program

The backtracker refuses one too, in its constructor, the same way and for a reason of a different shape. It is not that it could not answer a lookahead in principle, since it follows one path at a time and knows where it is, which is more than the DFA has. It is that it holds one stack and one bitmap and a lookahead is a search inside a search, so running one means either a second set of both or a walk that is two walks interleaved, and neither of those is a small change to a file whose whole argument is that the constants are small. A program holding an `IN_LOOK` is handed straight back and the machine answers it, which is the arrangement a row too long for the bitmap already has and is the reason that arrangement is there. Refusing at the moment the program is sized is what matters: the alternative is the instruction falling through to the branch that asks whether the character at the cursor is accepted, which would answer something rather than nothing.

The DFA cache refuses any program holding an `IN_LOOK`, outright, in its constructor, alongside the two shapes it already refused. The reason is plainer than either of those: that instruction runs a second machine over the rest of the row before deciding whether a thread goes on, so the answer at a position depends on text the box has not read and will not read again, and there is nothing to fold into a state. A DFA that could do this is a DFA with the text in it.

`_first_ranges`, which works out the set of characters a match can start with so that the machine can skip positions, has a catch-all that returns an empty list for any instruction it does not recognise, and an empty list means no optimisation. `IN_LOOK` falls into it and that is the correct answer rather than a missed one, since a lookahead can be the first thing in a pattern and the characters it permits are not the characters it consumes.

`Program.anchored`, which notices a pattern that can only match at the start, looks for a leading `^` and is untouched, because a leading lookahead is not one.

The one structural change inside the machine is that `_queue` now takes the `Program` rather than its `List[Instruction]`. It needs the ranges and the size to run the nested machine, and passing three more arguments down a function that already passes eleven was worse than passing the thing they all came out of.

The structural change outside it is in `program_for`, and it is the one the accessor tests found rather than the Mojo ones. There were two ways a call could land on Python's engine and they were two branches: a call with a `flags` argument was rewritten with Python's anchors and compiled with whatever captures the method needs, and a call routed there by its own syntax was compiled over the caller's tree, unanchored and without captures. The second branch was correct only because it always refused. Every construct that routes was refused, and a refusal does not care whether there is an `\A` in front of it or a slot behind it.

A lookahead ended that, and it ended it quietly. `contains("a(?=b)")` was right, because contains anchors nothing and asks for no groups. `match("a(?=b)")` came back with the unanchored answer, which is to say the wrong one, and looked exactly like a right one. `count("a(?=b)")` and `replace("a(?=b)", "#")` came back with a program that had no slot zero in it and trapped on the first match, because Python's counting rule is written in terms of where a match started. So the two branches are one branch now, keyed on whether the engine is Python rather than on which of the two things put it there, and the one thing that still distinguishes them is that a pattern the grammar cannot read is never routed by a walk over a tree that does not exist.

Folding them turned up a thing the old shape had been hiding, which is that the rewrite for `match` and `fullmatch` puts a group around the whole pattern and so numbers every group in it one higher than the caller wrote it. A backreference in such a pattern names the wrapper rather than the group the caller meant, and the grammar refuses that outright because the wrapper is still open where the backreference stands. So `match(r"(a)\1")` came back saying the grammar could not read a pattern the grammar reads perfectly well, as a `ValueError` where pandas answers with a column. Upstream never meets this because `regex.match` anchors from outside the pattern and writes no bracket at all.

Nothing reaches it today, since a backreference is refused on both engines, and the two rewriting methods now ask the compiler about the caller's own pattern before rewriting it and hand back that answer when it is a refusal. That keeps a still refused construct refused with the message and the gap flag it earned. The day a backreference lands, the rewrite has to stop renumbering, which probably means anchoring by instruction rather than by text, and that will be somebody's document.

The general form is worth saying, since this slice produced two of them: a branch that is correct because the value it computes is always thrown away is a branch that becomes wrong the first time somebody stops throwing it away, and nothing about it looks conditional.

## 10. The third one, which was not a branch at all

The two above were branches that were correct because nobody used them. The third was not correct at all and had never been, and it took a lookahead to produce a pattern that could tell.

Document 86 wrote Python's scanning rule down as four lines. Look from the cursor, take the end of the match, and step one character further on when the match had no width. That is what both Python scans in this library did, the counting one in `count.mojo` and the replacing one in `replace.mojo`, and it is not the rule. Upstream changed it in 3.7 and what it does now is this: a match of no width leaves the cursor where it is, and the next search is run with the end of the pattern refused at that one position. So the same place gets a second look, an arm of the pattern that reads a character can win there after an arm that reads nothing has already answered, and only when that second look comes back with nothing does the search move along, which it does by itself.

The two rules agree for every pattern that cannot prefer to match nothing where it could have matched something. That is nearly every pattern anybody writes, it is every pattern document 86 was checked against, and it is every pattern in the 30052 the corpus generates that reached these scans before this slice. What tells them apart is an alternation whose first arm can match nothing, or a lazy repeat, or a zero width assertion written in front of a branch that reads. `count(r"(?!x)|\s")` on `a b` is five upstream and was four here: the arm that reads nothing wins at each of the four positions, and upstream then asks again at the space and takes the arm that reads it. The replacing scan made the same mistake visibly, leaving the space in the row with a replacement written on each side of it.

`a*?` and `b*|a` would have shown this without any lookahead in them, through any `flags` argument, since a flag has routed a call to this engine since document 92. Nothing caught it because the pattern lists both scans are swept over held no lazy repeat and no alternation with an empty first arm, and because the differential that compares these scans against pandas only ever asked patterns with no flags beside them, which go to Arrow and run Arrow's loop. The lookahead is what made a pattern with no flags in it reach Python's loop, and the count differential failed on 212 of them at once.

The fix is one argument. `Machine.search` takes an `advance` flag meaning the end of the pattern is refused at the starting position, the thread that reaches it is skipped rather than taken and the threads behind it are left running, which is what upstream's backtracker does when it fails a `SUCCESS` under the same condition. A thread at the starting position has read nothing by definition, so the test is the position and needs no slots. Both scans then set the flag from whether the last match had width, and the replacing scan loses the second cursor document 86 section 5 was about, since there is no longer a character for the two to straddle.

This library's own backtracker takes the same argument and skips the same instruction, and the one thing worth writing down about it is the bitmap, which is not cleared between attempts at different positions and is the whole reason that engine is linear. A refusal is the first thing in it that reads where the attempt began, so the question is whether refusing a match at one position can poison a cell a later attempt wants. It cannot: the refusal only ever lands on the attempt that starts at the cursor and only ever at the position it starts at, and an attempt further along the row starts later than that and only ever moves forward, so the cell it would write is a cell it can never reach. The test that says so asks every pattern that prefers an empty match of every cursor of every row and compares the two engines slot by slot.

## 11. What is not here

The lookbehind, which is section 7 and is the next half of this construct.

The buffers, which is section 6 and is a measurement rather than a design.

The capture inside a lookahead, which needs the nested machine to carry slots and to merge them back into the outer thread's on success, and which is only reachable through `extract` and so is worth doing when somebody asks.

The other four constructs. After this slice the Python engine refuses 828 patterns for a backreference, 399 for a named character it has no table for, 392 for a possessive quantifier, 336 for a lookbehind, 199 for a conditional group, 188 for an atomic group and 67 for a repeat count it will not honour. Several of those numbers went up, which is not a regression: a pattern that was refused for its lookaround is now refused for whatever the next thing in it is, and the sweep counts the first reason it finds.

The sixteen texts. Document 90 said it and 91 and 92 repeated it and it is still true, which is that a sweep of 30052 patterns over 16 texts is a wide test of patterns and a narrow test of text, and a corpus of texts built the way the corpus of patterns was built does not exist. A lookahead is a construct whose interesting cases are about what comes after the position, so it is more exposed to that gap than a flag was.
