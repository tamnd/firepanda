# 102. A flag that belongs to a place

## 1. What this is

Document 101 left 1805 patterns in the corpus that RE2 reads and this library cannot. They are not one problem. Counted by the sentence Python's own grammar refused them with, which is the sentence this library repeats because this library reads Python's grammar, they are:

```
of 30052 patterns, 1805 are read by RE2 and not here
    633 bad escape                                          \p 176, \P 216, \Q 181, the rest small
    267 missing :
    202 nothing to repeat
    173 bad character in group name
    162 invalid group reference
    148 bad character range
    120 octal escape value outside of range 0-0o377
     85 global flags not at the start of the expression
     10 redefinition of group name
      4 multiple repeat
      1 unknown extension ?
```

This slice takes the 267 and the 85. They look like two buckets and they are one fact, which is that a flag in RE2 belongs to a place in the pattern and a flag in Python belongs to the pattern. Everything else on that list needs a table or a separate rule and is a later slice. The 392 patterns that want `\p` and `\P` are the largest single item and they want a Unicode script and category name table, which is the same table that would let the reader in document 101 stop being unsure about a pattern, so those two pieces of work are one piece of work and it is the one after this.

352 patterns is nineteen and a half per cent of what is left, and it is the largest thing on the list that needs no data at all.

## 2. The two spellings, and why Python has only one of them

Python reads `(?i)` and `(?i:a)` and they are not the same construct wearing different brackets. The second is a group and the flag applies to what is inside it. The first produces nothing at all and the flag applies to the whole pattern, which is why Python added a rule in 3.11 that it may only be written at the very start. `a(?i)b` is a parse failure with `global flags not at the start of the expression`, and it is a parse failure because the alternative would be a flag that says it is about the whole pattern while sitting in the middle of it.

Python has a second rule in the same place and it is the one that catches more patterns. A flag may only be turned off in the scoped form. `(?-i)` and `(?i-s)` are both parse failures, and the sentence is `missing :`, because what Python is doing when it sees the minus is looking for the colon that makes it a group. Neither of those is a global flag group with a minus in it. There is no such thing.

RE2 has one rule and it is neither of Python's. A flag group with no colon in it turns the flag on or off from that point to the end of the enclosing group. Position is the whole of its meaning. So `a(?i)b` is read, `(?-i)` is read, `(?i-s)` is read, and a caller who writes any of them has written an RE2 pattern whether they meant to or not, because pandas decides which engine answers by asking whether Python's `re` will compile the pattern and `re` will not compile any of these.

## 3. What the scope actually covers, measured

Guessing at this would have got it wrong twice, so all of it is measured against the RE2 inside pyarrow rather than read off a description of it. The rows are given with each answer.

It runs forward and not backward.

```
a(?i)b      on ab aB AB    ->  1 1 0
```

`ab` and `aB` match and `AB` does not, so the flag reached the `b` after it and not the `a` before it.

It crosses a bar.

```
a(?i)b|c    on c C         ->  1 1
x(?i)x|c    on c C         ->  1 1
```

The second one is the clean case, since nothing in it matches either row except the `c` branch, and `C` matches. A flag written in one alternative is still on in the next one.

It does not run backward across a bar either.

```
c|x(?i)x    on c C         ->  1 0
```

Which is what makes it a fact about position rather than about the alternation. The bar is not a boundary, it is simply more pattern.

It stops at the closing bracket of the group it is in.

```
((?i))c        on c C            ->  1 0
(?-i:a(?i)b)c  on abc aBc abC    ->  1 1 0
```

The first shows a flag set inside a capturing group not reaching past it. The second shows one set inside a scoped group that had turned the same letter off, reaching the `b` beside it and not the `c` outside it.

It leaves nothing behind for a repeat.

```
(?i)*        ->  refused, no argument for repetition operator
a(?i)*b      on b ab aab aB aaB  ->  1 1 1 1 1
a(?i){2}     on b ab aab aB aaB  ->  0 0 1 0 1
```

The first is refused because at the start of a pattern there is nothing on the parser's stack. The third is the one that settles it: `aab` and `aaB` match and `ab` does not, so the `{2}` counted the `a` and not the flag group. A repeat written after a flag group reaches back past it to whatever was there before, which is the same rule document 101 section 5 recorded from the other side.

## 4. The letters are not the same set either

This was measured one letter at a time over both cases, and it is short enough to print whole.

| letter | RE2 `(?X)` | RE2 `(?X:a)` | Python `(?X:a)` |
| --- | --- | --- | --- |
| `a` | no | no | reads |
| `i` | reads | reads | reads |
| `m` | reads | reads | reads |
| `s` | reads | reads | reads |
| `u` | no | no | reads |
| `x` | no | no | reads |
| `U` | reads | reads | no |

RE2 has four letters and Python has six and they share three. Two of the differences matter here and one does not.

`x` is the one that matters most, because it is the one a person would assume is shared. RE2 has no verbose mode at all, so `(?x)a b` is refused by RE2 and read by Python, which is the opposite direction from everything else in this document. The reader in document 101 already knows this and nothing in this slice changes it.

`U` is RE2's and swaps which quantifiers are greedy. The corpus generates it once in thirty thousand patterns, so it is not this slice. It is recorded here because a later slice that reads it will need section 3 to already be true.

`a` and `u` are Python's alphabet letters. They are not a gap, because a pattern writing one of them is a pattern Python reads, so pandas answers it in Python and RE2 is never asked.

## 5. How the tree carries it

The parser in `firepanda/kernel/regex/parse.mojo` reads Python's grammar and that is not changing, because the routing depends on it. `method.mojo` decides which engine answers by asking this parser whether the pattern reads, exactly as pandas decides by asking `re`, and a parser that started reading RE2-only syntax as though it were ordinary would move patterns to the wrong engine.

So the tree gains a flag rather than the parser gaining a mode. `Parsed` already carries `re2_refuses` for Python syntax RE2 has never had and `re2_differs` for syntax RE2 reads a different way. This adds the mirror of the first one, `python_refuses`, for RE2 syntax Python has never had, and the flag group in a place Python will not take it is the first thing to set it.

The parse then reads the construct rather than giving up on it, and the compiler is what refuses. `compile_program` on `ENGINE_PYTHON` refuses a tree with `python_refuses` set and says the sentence Python would have said, so a caller who passed a `flags` argument and therefore landed on Python's engine sees no change at all. `compile_program` on `ENGINE_RE2` builds the program. That is the same division of labour the other two flags already use and it is there for the same reason, which is that the parser is the only thing that sees the text and the compiler is the only thing that knows who is asking.

One line of that paragraph was wrong when it was written, and this is the amendment. The router does not ask the compiler anything. It asks the parse whether the pattern reads, and the whole point of this section is that a pattern holding this construct now reads, so the router started sending to Python's engine every pattern that held a flag group in a place beside a lookaround or a backreference, which pandas hands to Arrow. `\p{L}` and `^*` inherited the same mistake when documents 103 and 104 set the same flag. The fix is one condition in `holds_unsupported` and one in `reads_as_python`: both ask whether Python's grammar read the pattern rather than whether this one did. It took three more slices to find, because the misrouted patterns are the ones holding two awkward constructs at once and no test file has one, and because neither of the two comparisons that would have caught it is run by any workflow.

Two sentences are needed rather than one, because Python does not give the same reason for the two spellings. A colon-less group in a place is `global flags not at the start of the expression`. A colon-less group turning a letter off is `missing :`, wherever it is written. The tree records which, because inventing a single sentence for both would mean an argued call getting a message Python never gives.

## 6. Splitting a sequence rather than wrapping one

The node that comes out is `OP_SCOPE`, which already exists and already carries the letters turned on and the letters turned off, because the scoped form has always produced one. Nothing in the three engines needs to learn anything new. What changes is where the node goes.

A sequence reading `ab(?i)cd` comes out as a sequence of `a`, `b`, and a scope holding a sequence of `c` and `d`. The flag group splits the sequence it is in rather than wrapping it, and the rest of the sequence is read into the new inner one. That is what makes the scope end at the closing bracket without anything having to say so, since the inner sequence ends where the outer one would have.

The repeat rule in section 3 falls out of this only if the split is careful. `_seq` tracks the last item it read so that a quantifier knows what to repeat, and a flag group must not become that item, or `a(?i){2}` would count the scope rather than the `a`. So the split moves the attachment point and leaves the last item alone, which is the same thing the parser already does for a comment group, and for the same reason: `a(?#c)+` repeats the `a` too.

Crossing a bar needs one more thing, because a sequence that has ended cannot read the next alternative. `_branch` carries the letters the previous alternative turned on and off and wraps each later alternative in a scope of its own. `a(?i)b|c` becomes an alternation of `a` and a scope holding `b`, and a scope holding `c`. That is not the same tree as `a(?i:b|c)`, which would change which alternatives there are, and it is not the same tree as `a(?i:b)|c`, which would not reach the second alternative at all. Distributing the scope over the alternatives that follow is the transformation that leaves the alternation alone and gets the flags right.

## 7. What the reader in document 101 already says

The reader is consulted only when the parse fails, so a pattern this slice teaches the parser to read stops reaching it. That is worth stating because it means the reader needs no change and gets one anyway, in the form of fewer callers.

It already answers these correctly, which was checked rather than assumed: it reads `a(?i)b` and `(?-i)` and `(?i-s)`, and it refuses `(?x)` and `(?-)` and `(?i-)` and `a(?i`, which is what RE2 does. The one thing it has always got right about this construct is the one that took the longest to find, which is that a flag group pushes nothing on the parser's stack, so `(?i)*` is refused at the front of a pattern and repeats the `a` in `a(?i)*`. Section 3 measured it again from the matching side and the two agree.

## 8. Where the bias goes

The same place document 101 put it. A pattern this slice reads wrongly is a column of booleans that looks like a right one, and a pattern it refuses is a `NotImplementedError` a caller can read. So anything not certain stays refused.

In practice that is one rule. A flag letter RE2 takes and this parser does not know is not guessed at, it sets the tree's existing refusal and the pattern stays a gap. `U` is the whole of that set today and it is one pattern in thirty thousand.

## 9. The differential

`tests/differential/regex_match.mojo` and the four beside it are the measurement, and no new program is needed, because the corpus already generates these patterns in quantity and they are already counted. What changes is which bucket they are in.

On `str.contains`, which is the widest of the five because it is the one pandas hands to Arrow with the pattern as written:

```
                     before   after
compared              27431   27752
held out               2621    2300
  grammar bucket       1807    1472
agreement per 10000   10000   10000
disagreements             0       0
```

335 patterns moved rather than the 352 section 1 predicted, and the shortfall is not a shortfall. A held out pattern is counted once under the first reason it hit, so a pattern writing both a flag group in a place and something else this library still cannot read was already being counted somewhere and stays counted there. The grammar bucket fell by 335 and the compared count rose by exactly 321, because fourteen of the patterns that now parse land in a narrower held out bucket instead: the named character bucket went from 390 to 399, `RE2 reads this syntax differently` from 218 to 221, and `RE2 reads a non boundary between two bytes` from 111 to 113. That is the right direction. A pattern moving out of `Python's grammar cannot read this pattern` and into a bucket that names the actual construct is a pattern whose reason is now true.

`str.match` and `str.fullmatch` moved the same way, to 27853 compared and 2199 held out with 1372 and 1371 in the grammar bucket. `str.count` and `str.replace` both sit at 27739 compared and 2313 held out. `regex_python` is 29462 compared and 590 held out and did not move at all, which is the point of section 5: a caller on Python's engine sees no change.

`tests/differential/regex_re2.mojo` is unchanged and still at zero, since a parser that now reads a construct must not have changed what the reader says about it.

All five are at 10000 agreements in ten thousand with zero disagreements, which is where they were before.

## 10. What is left

The list in section 1, less this slice. In the order their size argues for:

The 392 that want `\p` and `\P`, which want a Unicode script and category name table, and which are the same table that would let document 101's reader stop marking a pattern unsure. That is the next slice and it is the largest one left.

The 202 that repeat a zero width assertion, as in `^*` and `\b*` and `\B{0}`. RE2 repeats an assertion and Python refuses it, so this is the same shape as this slice, a construct one grammar has and the other has not, and it needs no data either.

The 181 that write `\Q` and `\E`, which is a span of text to be read as literal characters and nothing more.

The 162 that write a backreference RE2 has never had, where `\1` is an octal escape rather than a reference, together with the 120 that write an octal escape Python calls out of range. Those two are one rule about how far a backslash and a digit reach.

The 173 and the 10 about group names, which are two rules about which characters a name may hold and whether a name may be used twice.

The 148 about a range in a class, as in `[\d-a]`, which Python refuses and RE2 reads as three members rather than a range.

And after all of those, the 218 patterns RE2 reads differently rather than refuses and the 111 where RE2 puts a non boundary between two bytes of one character, which are both about what a pattern means rather than about whether it reads, and which document 90 section 10 has been waiting on for a while.

The sweep still asks sixteen texts and a corpus of texts still does not exist, which is document 90 section 10 and is repeated here because it is still true.
