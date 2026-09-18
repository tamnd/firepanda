# 96. Three steps a character and two of them spare

## 1. What this is

ClickBench q28 rewrites a column of URLs with `^https?://(?:www\.)?([^/]+)/.*$`, and after #863 and #889 the regex engine is about nine tenths of what that query does per row. Document 80 counted the three passes around the engine and found nothing left in them worth having, so the query is the engine now and the engine is a step count.

The count was 226256004 steps over the 921225 rows q28 keeps, which is about 245 steps a row against rows that average 86 characters, so about 2.85 steps per character. This document is why that number is close to three rather than close to one, what was done about it, and the thing that was tried first and taken out again.

## 2. Where the three steps come from

A Thompson program has no counter and no loop instruction. `[^/]+` is written out as the class once, then a split, the class again, and a jump back to the split:

```
0 notset(/)
1 split(2,4)
2 notset(/)
3 jump(1)
4 ...
```

So every character after the first costs three instructions: the split that decides whether to go round again, the class that reads the character, and the jump that goes back. Two of the three are bookkeeping. The same is true of `.*` on the tail of the pattern, which on this column is most of the row: an 86 byte Referer with a 20 character host name spends about 50 steps getting through the scheme and the host and all the rest of its 245 on the tail, and every step of that tail is a split, a test that can only say yes, and a jump.

That is not a fault of the compiler. It is what a Thompson program is, and the Pike machine next door needs exactly those instructions, because the split is where the machine puts two threads into its list and the thread list is the whole of how it stays linear.

## 3. The instruction RE2 has, and why it is not here

RE2 answers this with a character class loop: one instruction that consumes a maximal run of a class and writes down where the run ended, so a host name of twenty characters is one step rather than sixty. The issue that started this work proposed exactly that.

It was tried on paper and turned down, for a reason that is about the backtracker rather than about the instruction. The whole of what makes `backtrack.mojo` safe is that it visits each pair of an instruction and a position at most once, which gives a bound of the program size times the row length however the pattern is written. An instruction that consumes a run in one step and then backs off inside that run visits the positions of the run again on the way back, and it does so once per position the run was entered at, so a pattern with two of them next to each other is the row length squared. RE2 can afford it because RE2's backtracker is the fallback for short rows and its DFA carries the long ones. Here the backtracker answers every row of q28 and gives up to the machine only above a quarter of a million bitmap cells, so a quadratic path in it is a quadratic path in the product.

So the fusion here is the smaller one that keeps the bound exactly: three instructions into one step, with the decision still taken once per character. The split reads the character itself and comes straight back to the split one position along, and the class and the jump behind it are never visited. One visit per split per position, which is what the bitmap already promised.

## 4. The order is the whole of the correctness

A greedy split pushes the arm that leaves the repeat and then the arm that goes round, so the arm that goes round comes off the stack first and is the path the pattern prefers. Fused, that is the arm that leaves pushed as before, and then, if the character at this position is in the class, the split itself pushed at the next position. A lazy split is the same two pushes the other way round, which is what the lazy split already did.

Captures are untouched, because the body of a run is a bare class and a class writes no slots. The end of the row is untouched, because the character test is the same test with the same guard. What changes is only which cells of the bitmap get marked: the body and the jump no longer get one. Nothing else in the program jumps into either of them, and if something did, arriving there would be walked the ordinary way and would give the same answer, more slowly and still bounded.

## 5. The thing that was tried first

The first version wrote the shape into the program. Two opcodes were added, `IN_RUN` and `IN_RUN_LAZY`, `_emit_star` rewrote the split's opcode when it saw that the body was a single character instruction, and the three engines each grew a branch for them.

It worked and it was worth what it should have been worth, and it made the Pike machine about 1.4 times slower. On the real Referer column the machine went from 4068 to 6166 ms before to 6351 to 6608 ms after, reproduced three times back to back. The reason is in document 77 section 10, which measured `_queue` at about 69 percent of the machine's time: adding two opcodes to a program means adding two comparisons to the front of the dispatch chain that every character instruction falls through, and a character instruction is what the common case is. Rewriting the branch so the payload order matched the split exactly made the branch body identical and changed nothing, which is what said the cost was the chain and not the branch.

The general shape of that is worth writing down, because it will come up again. A note in the program is read by every engine and wanted by one. There are three engines over one instruction set here and they do not pay the same way: the state cache builds each state once per program, so a comparison there is free, and the machine walks the dispatch chain per instruction per position, so a comparison there is a tax on a column.

## 6. What was done instead

`run_bodies` in `program.mojo` takes the instructions and answers, for each one, whether it is a split of that shape and where its body is. It reads the shape back out of the program rather than being told: a split at `s` qualifies when one arm is `s + 1`, the instruction there reads one character, the instruction after it is a jump back to `s`, and the other arm is `s + 3`. Greedy and lazy both come out of it, and which arm the body is on is what says which of the two it is.

`Bounded` calls it once in its constructor and keeps the answer, which is one `Int32` per instruction for the life of a column and is nothing beside the bitmap. The fused walk lives inside the existing split branch of `_attempt`, behind a read of that list.

The compiled program is byte for byte what it was. `pike.mojo`, `dfa.mojo`, the first character set analysis and the program listing the tests assert against are all untouched, and the machine pays nothing at all. The detection sits immediately below the emit it recognises, so a change to one is in front of the reader of the other, and a test asserts the pairs it fires on and three shapes it does not.

## 7. What it is worth

Measured on the real Referer column of the 1M file, the same seven section harness #889 used, paired back to back against the same tree with the change reverted. The machine the numbers were taken on is shared and was loaded, so the minima of three pairs are what is quoted and the spread is wide:

| | before | after |
|---|---|---|
| decode and backtracker, serial | 2031 ms | 1126 ms |
| the kernel q28 calls, parallel | 740 ms | 472 ms |

About 1.8 times on the backtracker and about 1.6 times on the kernel around it. The Pike machine is unchanged and measured unchanged, within a noise band that on this machine was a factor of four.

The step count printed by the harness does not move, and that is not a mistake. It counts an inline copy of the walk kept beside the engine as a control, so it says what the program costs rather than what the engine now does with it. What the engine does is the time.

## 8. What is left

`.*` at the end of an anchored pattern is still a run, and it is still walked one character at a time even though nothing can follow it. A rule that recognised a trailing run under an end anchor could answer it with one jump to the end of the row. That is a different piece of work and it belongs to the compiler rather than to either engine.

The instruction RE2 has is still the larger prize and it is still not available to a backtracker that has to stay bounded. If a lazy DFA ever carries the long rows on this path, the trade changes and it is worth reopening.
