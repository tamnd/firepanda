# 104. A repeat on a position

## 1. What this is

The third slice of the patterns RE2 reads and this library did not. Document 102 took the flag group written in a place and document 103 took the Unicode name, and what those two left behind, counted by the sentence Python's own grammar refused them with, was:

```
    202 nothing to repeat
    181 \Q and \E
    173 bad character in group name
    162 invalid group reference
    148 bad character range
    120 octal escape value outside of range 0-0o377
     10 redefinition of group name
      4 multiple repeat
      1 unknown extension ?
```

This slice takes the largest of those, and it is the first one in the run that needs neither a table nor a new node. `^*` is a pattern RE2 reads and Python refuses, and everything else follows from working out what RE2 thinks it means.

Not all 202 are this construct. `*` and `a|*` and `(*)` are patterns with genuinely nothing in front of the quantifier, and RE2 refuses those too, so they are already answered correctly and stay where they are. What moves is the subset where the thing in front of the quantifier is a position test.

## 2. The one sentence, and the two readings under it

Python has a rule that a quantifier must follow something a quantifier can be applied to, and a position test is not one of those things. `re.compile("^*")` is `nothing to repeat`, and so are `$+`, `\b{0}`, `\B*` and `\A?`. That has been true in every version this project supports.

RE2 has no such rule. A position test goes on the stack like anything else and a quantifier written after one takes it. So `^*` compiles, and the question that matters is what it matches.

The answer is the one a person gets to by thinking about it for a minute, and it was measured anyway, because the last two slices both turned up a rule that thinking about it for a minute got wrong. A position test asks a question about where the engine is standing. Asking it twice in a row cannot fail if asking it once succeeded, and asking it zero times cannot fail at all. So a quantifier on one collapses to one of exactly two things: the test itself, when the quantifier insists on at least one, and nothing at all, when it will settle for none.

Measured against the RE2 inside pyarrow 24.0.0 over the seven texts `""`, `"a"`, `"ab"`, `"a b"`, `"\na"`, `"a\nb"` and `"aa"`:

```
    ^*      ^+      ^?      ^{0}    ^{2}    ^{0,2}  ^*?
    empty   ^       empty   empty   ^       empty   empty

    $*      $+      \b*     \b+     \b{0}   \b{2}   \b{1000}
    empty   $       empty   \b      empty   \b      \b
```

Every one of those was compared column for column against the pattern in the row under it, and every one is identical. `^*` is `^` written nowhere, `^+` is `^` written once, and `\b{1000}` is `\b` written once.

Two of the pairs are worth pointing at. `\b*` and `\b+` differ, which is what shows the collapse is decided by the lower bound and not by which quantifier was used. And `^?` is empty rather than `^`, which is the same rule and reads oddly until you notice that a `?` is allowed to take none.

## 3. What it is not

Three constructs in the neighbourhood are unchanged, and each is unchanged for its own reason.

`(^)*` and `(?:^)*` are patterns Python reads, because the quantifier follows a group and a group is a thing a quantifier can be applied to. They went down the ordinary path before this slice and they go down it now. The check added to the compiler asks the narrow question, whether the body is exactly one position test, rather than the wide one, whether the body reads any characters, precisely so that these keep the path they had.

`(?=a)*` is a quantifier on a lookahead. Python reads it, RE2 has no lookahead at all, and neither half of that is this slice.

`^*+` and `^**` and `^*{2}` are refused by RE2, and for a reason that is about the second quantifier rather than the first. RE2 reads `^*` and puts a repeat on its stack, and `**` is a repeat on a repeat, which RE2 calls `bad repetition operator`. That is why the parser here builds the repeat node rather than collapsing at parse time: collapsing would leave the anchor on the stack, or leave nothing there, and the next quantifier would then be refused for the wrong reason or accepted when it should not be. The collapse belongs to the compiler, where the tree is already whole.

## 4. Where it lands

Two edits and no new op.

In `_seq` in `parse.mojo`, the branch that gave up with `nothing to repeat` when the last item was an `OP_AT` now records `python_refuses` and Python's own sentence and falls through to build the repeat. That is the third use of the field document 102 added, after the flag group and the Unicode name, and it is the first one that needed no discussion about what the sentence should be, since Python gives exactly one sentence for every spelling of this.

In `_emit_repeat` in `program.mojo`, a repeat whose body is exactly one `OP_AT` writes the body once when the lower bound is at least one and writes nothing otherwise, instead of copying the body a count of times. Copying it would have been a loop around something that reads no characters, which is the one shape a Thompson program built by copying cannot express.

Nothing else moved. The engines learned nothing, the router is untouched, and the count limit that refuses `^{1001}` is the same one that refuses `a{1001}`, spent in the same walk.

## 5. What the reader already knew

The reader in `re2.mojo` needed no change at all, which is worth recording because the last two slices both needed one.

It already read a quantifier on a position test, because it was written from RE2's grammar rather than from Python's and RE2's grammar has no rule against it. It already refused `^**` and `^*{2}` as a repeat on a repeat and `^*+` as an operator RE2 has not got, and it already applied RE2's own stack rule, which is the one that says a quantifier written after a flag group reaches past it because a flag group pushes nothing while a position test pushes something. That rule was measured for document 101 and it is the rule this slice depends on.

So the differential that compares the reader with RE2 does not move. It was exact after document 103 and it is exact now.

## 6. Where the bias goes

Nowhere new. The narrow check in the compiler is the bias: it asks whether the body is one position test rather than whether the body reads nothing, so a body this slice has not thought about keeps the treatment it had. A pattern that goes down the old path is a pattern that was already right.

## 7. The differential

The corpus generated two patterns in this family, `^*` and `\b*`, and now generates twenty four, chosen to cover each claim in section 2 and each exception in section 3: all four quantifiers on `^`, both bounded counts, the lazy marker, `$` and `\A` and `\b` as well, an anchor repeat with text on either side of it, and the three RE2 refuses.

As with the last slice the corpus moved in the same commit, so it was measured twice.

Against the corpus document 103 left, which is 30052 patterns, `str.contains` goes from 28410 compared and 1642 held out to 28461 and 1591. The grammar bucket falls from 916 patterns to 841. 75 left it. 51 became compared answers and the other 24 moved to a bucket that names their actual construct: 2 to the named character bucket, 6 to `RE2 reads this syntax differently` and 16 to `RE2 reads a non boundary between bytes`. That last group is the `\B` family and it is the most useful thing on this line, which section 8 has more about.

75 is the honest count of this construct in that corpus, against the 202 section 1 begins with. The other 127 are the patterns with genuinely nothing in front of the quantifier, which RE2 refuses as well and which were already being answered with the refusal RE2 gives.

Against the widened corpus, which is 30070 patterns and is what the differential asks from now on, `str.contains` is 28479 compared and 1591 held out, `str.match` and `str.fullmatch` are both 28529 and 1541, `str.count` and `str.replace` are both 28465 and 1605, and the Python engine differential is 29596 and 474. All five are at 10000 agreements in ten thousand with zero disagreements.

`regex_re2` does not move, which is section 5. It is 8389 read and 21681 refused with matching reasons over the 30070, with nothing read that RE2 refuses and nothing set aside, which is the same exactness document 103 reached over the 30052.

## 8. What is left

The list in section 1, less this slice and less the part of the 202 that was never this slice.

The 181 that write `\Q` and `\E`, which is a span of text read as literal characters and nothing more, and which is now the largest item.

The 162 that write a backreference RE2 has never had, where `\1` is an octal escape rather than a reference, together with the 120 that write an octal escape Python calls out of range. Those two are one rule about how far a backslash and a digit reach.

The 173 and the 10 about group names, which are two rules about which characters a name may hold and whether a name may be used twice.

The 148 about a range in a class, as in `[\d-a]`, which Python refuses and RE2 reads as three members rather than a range.

And after those, the patterns RE2 reads differently rather than not at all and the ones where RE2 puts a non boundary between two bytes of one character, which are both about what a pattern means rather than about whether it reads.

The non boundary bucket is the one this slice made larger, and that is the right direction. `\B*` and `\B{2}` used to be held out for a sentence about Python's grammar, which is true and says nothing a caller can act on. They are held out now for the reason they are actually held out, which is that RE2 asks the word boundary question between bytes and this engine asks it between characters. The pattern is no more answered than it was, and the reason it is not answered is no longer wrong.

The table from document 103 still holds 163 script names the corpus exercises two of, and the sweep still asks sixteen texts, which is document 90 section 10 and is still true.
