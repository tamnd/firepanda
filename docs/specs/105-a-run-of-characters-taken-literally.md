# 105. A run of characters taken literally

## 1. What this is

The fourth slice of the patterns RE2 reads and this library did not. Documents 102, 103 and 104 took the flag group written in a place, the Unicode name and the repeat on a position test, and what those three left behind, counted by the sentence Python's own grammar refused them with, was:

```
    181 \Q and \E
    173 bad character in group name
    162 invalid group reference
    148 bad character range
    120 octal escape value outside of range 0-0o377
     10 redefinition of group name
      4 multiple repeat
      1 unknown extension ?
```

This slice takes the largest of those. `\Q` opens a run of characters to be taken literally and `\E` closes it, so `\Qa+b\E` matches the three characters `a` and `+` and `b` rather than a repeated `a` followed by a `b`. Python has neither escape and calls both `bad escape`, which is what routes every pattern holding one to Arrow.

It is the smallest slice of the four to state and the one with the most rules under it, because almost nothing a reader would assume about it is true.

## 2. The run, measured

Every line in this section was measured against the RE2 inside pyarrow 24.0.0, by compiling the pattern and comparing its column against the column of the pattern claimed to be equivalent, over fifteen texts chosen to tell the two apart.

`\Q` opens the run and the run ends at the first `\E`, or at the end of the pattern when nobody wrote one. An unterminated run is not a refusal: `\Qa+b` is `a\+b`, exactly as `\Qa+b\E` is.

`\E` with no run open is a refusal, and this is the first surprise. RE2 says `invalid escape sequence: \E`, so `a\Eb` and `\E` alone and `\Qa\E\E` are all patterns RE2 turns down. The close is not a harmless no op outside a run, which is what a reader who thinks of these as a pair of markers would expect.

Neither half is allowed inside a character class. `[\Qa\E]` is `invalid escape sequence: \Q`, so the run is a thing written between atoms rather than an item like the Unicode name of document 103 was.

Both halves are gone from what the pattern means. `\Q\E` is empty and matches every row, and so is `\Q` written alone.

## 3. Nothing inside the run is an escape

This is the rule that would have been got wrong, and it was measured in four spellings because getting it wrong is invisible until a row has a backslash in it.

The closer is found by looking for a backslash followed by an `E` as two literal characters. Nothing between the two is read as an escape first. So:

```
    \Qa\\E      is  a  and one backslash
    \Q\\\E      is  two backslashes
    \Q\Q\E      is  a backslash and the letter Q
    \Q\n\E      is  a backslash and the letter n
```

The first of those is the one worth staring at. A reader who thinks the run understands escapes reads `\Qa\\E` as the letter `a` and an escaped backslash with the run still open. RE2 reads it as the letter `a`, then finds the close at the second backslash and the `E`, and the run is one character long. The two readings differ on any row holding a backslash, and `\Q\n\E` shows the same thing from the other side: a newline is what it means in Python's grammar and two characters here.

There is no way to write a literal `\E` inside a run, which follows from the above rather than being a separate rule, and which is worth saying because every other quoting construct in every other language has an answer to that question.

## 4. The run is not a group

The characters in the run are still just characters once the run is over, which shows up in two places.

A quantifier written after the run takes the last character of it and not the whole run. `\Qab\E*` is `ab*` and not `(?:ab)*`, and `\Qa\E{2}` is `aa`. That is measured and it is also what falls out of the run leaving one item per character on RE2's stack, which is the rule document 101 measured and documents 102 and 104 both leaned on.

An empty run leaves nothing there at all. So `\Q\E*` is `no argument for repetition operator`, and `x\Q\E*` is `x*` with the star reaching back past the run to the `x`. That is exactly the flag group's behaviour from document 102 section 4, reached by a completely different construct, which is the second piece of evidence that the stack rule is the rule rather than a description of one case.

The run also crosses syntax it would otherwise end at, which is the other half of not being a group:

```
    (\Qa)b\E)   is  (a\)b)
    \Qa|b\E     is  a\|b
    \Qa[b\E     is  a\[b
    \Qa(b\E     is  a\(b
```

A closing bracket inside a run does not close the group the run is in. That is a rule about reading order: the run is taken before the grammar sees any of it, so a bracket counter that walks the pattern without knowing about `\Q` will count wrongly.

One thing is not special. `(?i)\QAB\E` is `(?i)AB`, so folding applies to the characters of a run the way it applies to any other literal, and there is no line anywhere that makes that true.

## 5. Where it lands

One block in `_seq` in `parse.mojo`, placed where an atom would have been read, and no new op.

The block records `python_refuses` with the sentence `bad escape \Q`, which is the field document 102 added and which this is the fourth use of. Then it walks the run and appends one `OP_LITERAL` node per character, updating the sequence's last child each time exactly as the atom reader would have done.

That is the whole implementation, and every rule in sections 3 and 4 is a consequence of it rather than a line in it. The quantifier takes the last character because the last character is what the sequence's last child points at. The empty run leaves nothing because a loop that runs zero times attaches nothing. The run crosses a bracket because the walk consumes characters without consulting the grammar. The fold applies because an `OP_LITERAL` is an `OP_LITERAL`.

The compiler was not touched. The engines were not touched. The router was not touched.

`\E` with no run open is left where it was, which is the escape reader giving up. RE2 refuses it too, so the parse failing is the right outcome and the reader in `re2.mojo` is what supplies the caller's `ValueError`. Inside a class the same thing happens to both halves, for the same reason.

## 6. The bracket Arrow adds

The slice turned up one thing that is not about `\Q` at all, and it turned up because the corpus widened and the replace differential broke.

`pyarrow.compute.replace_substring_regex` compiles the pattern twice. One copy has a capturing bracket round the whole of it and is what finds the extent of a match, and the other is the pattern as the caller wrote it and is what the group numbers in the replacement refer to. That is why a `\1` in a replacement names the caller's own first group rather than the bracket Arrow added, and why a `\2` in a replacement for a pattern with one group raises `Rewrite schema requests 2 matches`. Both copies are compiled, the bracketed one first, and a caller sees a refusal from either.

For every pattern but one shape the bracket is invisible, because a pattern RE2 reads is a pattern RE2 reads with a bracket round it. The shape is a pattern that ends inside a run. `\Qa` swallows the bracket Arrow wrote, so the group is never closed, and `str.replace("\Qa", "-")` raises `missing ): (\Qa)` while `str.count("\Qa")` and `str.contains("\Qa")` both answer. The quoted pattern in that message is not the pattern the caller passed.

This had been true since pyarrow had the function and was unreachable from here, because a pattern holding an unterminated run did not parse and was held out of every comparison. The slice made it parse, and 138 of the widened corpus's patterns turned into replaces this library answered and pandas refused.

So `method.mojo` now asks the RE2 reader about `(pattern)` before it asks about anything else, for `replace` and for no other method. The compile still uses the pattern as written, which is what keeps the capture numbering the caller's. The order matters only for the wording: `)a` is refused by both copies and the message quotes `()a)`, with a bracket nobody wrote in it, so asking about the bracketed copy first is what reproduces the message upstream gives.

It is worth filing. A caller cannot be expected to know that one of the five string methods validates a pattern they did not write.

## 7. What the reader needed

Less than document 103 and more than document 104.

The reader in `re2.mojo` already walked a run correctly, already looked for the closer as two literal characters, already let an unterminated run reach the end of the pattern, and already refused `\E` on its own and both halves inside a class. That was written for document 101 from RE2's grammar and it was right.

What it did not have was the stack rule of section 4. Every escape leaves one item behind, so the atom reader set the flag that says a repeat has something to take before reading any escape at all, and an empty `\Q\E` is the one escape that leaves nothing. The reader took `\Q\E*` when RE2 refuses it. That is the direction section 8 cares about, so it is the one worth having found.

The fix is three lines and a field. The run records whether it held any characters, and the atom reader puts the flag back the way it found it when the answer is no. `x\Q\E*` still reads, because the flag it puts back is the one the `x` set.

## 8. Where the bias goes

The same place it has gone in 101 through 104. A pattern read wrongly is a column of booleans that looks like a right one, and a pattern refused is something a caller can read and act on.

The reader defect in section 7 is the shape of mistake that bias is meant to catch, and it is worth naming that the unit tests did not catch it and the corpus did. A reader that takes one pattern too many is a reader that hands a program to a compiler which has no business building one, and nothing about the pattern says so.

## 9. The differential

The corpus generated no pattern in this family deliberately, so the measurement has a clean before: every pattern that moved, moved because of this slice.

The measured list goes from 73 patterns to 114, the 41 added covering each claim in sections 2, 3 and 4, including all four spellings of the escape rule, the four bracket crossings, both halves of the stack rule and the five refusals.

On the corpus document 104 measured, unwidened, the `str.contains` grammar bucket falls from 841 patterns to 703. Of the 138 that leave it, 136 become compared answers and 2 move to the bucket that names the character they could not resolve. That is 138 against the 181 the list in section 1 counted, and the other 43 hold a `\Q` beside a second construct this library still cannot read, so they stay where they were and will come out with a later slice.

On the widened corpus of 30111 patterns, `str.contains` is 28641 compared and 1470 held out, `str.match` and `str.fullmatch` are both 28405 and 1706, `str.count` is 28627 and 1484, `str.replace` is 28960 and 1151, and the Python engine differential through `str.findall` is 29637 and 474. The routing differential compares 29280. The reader in `re2.mojo` reads 8420 and refuses 21691 with matching reasons and nothing wrong in either direction. All six are at 10000 agreements in ten thousand with zero disagreements.

The one bucket that differs between the methods for a reason belonging to this slice is the grammar one, which is 701 for `str.replace` and 703 for `str.contains` and `str.count`. The two patterns are the ones section 6 moved: the bracketed copy refuses them with RE2's own sentence, so they stop being a gap this library owns and become a refusal that matches the one pandas gives.

## 10. What is left

The list in section 1, less this slice.

The 162 that write a backreference RE2 has never had, where `\1` is an octal escape rather than a reference, together with the 120 that write an octal escape Python calls out of range. Those two are one rule about how far a backslash and a digit reach, and they are the largest item now.

The 173 and the 10 about group names, which are two rules about which characters a name may hold and whether a name may be used twice.

The 148 about a range in a class, as in `[\d-a]`, which Python refuses and RE2 reads as three members rather than a range.

And after those, the patterns RE2 reads differently rather than not at all and the ones where RE2 puts a non boundary between two bytes of one character, which are both about what a pattern means rather than about whether it reads.

The table from document 103 still holds 163 script names the corpus exercises two of, and the sweep still asks sixteen texts, which is document 90 section 10 and is still true.
