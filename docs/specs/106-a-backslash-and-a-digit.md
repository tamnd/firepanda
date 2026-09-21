# 106. A backslash and a digit

## 1. What this is

The fifth slice of the patterns RE2 reads and this library did not. Documents 102 through 105 took the flag group written in a place, the Unicode name, the repeat on a position test and the run of characters taken literally, and what those four left behind, counted by the sentence Python's own grammar refused them with, was:

```
    173 bad character in group name
    162 invalid group reference
    148 bad character range
    120 octal escape value outside of range 0-0o377
     10 redefinition of group name
      4 multiple repeat
      1 unknown extension ?
```

This slice takes the second and the fourth, which are 282 patterns together and the largest item on the list, because they are one rule rather than two. Both are about how far a backslash and a digit reach and what the digits mean once they have been read, and neither can be moved without the other.

The short version is that Python has a backreference and RE2 does not, so the same two characters are a reference to a group in one grammar and a character in the other, and which of the two a caller gets depends on how many groups they happened to write earlier in the pattern.

## 2. Python's rule

Python's own comment on this rule is `octal escape *or* decimal group reference (sigh)`, and the parenthesis is doing real work.

One digit is read. If the next character is a digit it is read too. Then, and only if both digits read so far are octal and the next character is an octal digit as well, a third is read and the three are an octal character, with a ceiling of `0o377`. Anything else is a decimal group reference made of the one or two digits, and the reference is checked where it is written, so a number larger than the count of groups opened so far is `invalid group reference`.

That is why `\777` is an octal escape and out of range, `\778` is a reference to group 77, and `\788` is a reference to group 78. A rule of "three digits means octal" would read the last two the same as the first and would be wrong about both.

Inside a character class there is nothing for a reference to refer from, so the digits are octal there or the escape is bad, and the same ceiling applies.

## 3. RE2's rule

RE2 has no backreference and never has had one, because a backreference is the construct that costs it its time bound. So a backslash and a digit is an octal escape there or it is nothing at all.

A leading zero is always octal. A leading digit from one to seven is octal only when an octal digit follows it. An `8` or a `9` is never anything. Up to three octal digits are taken, and the ceiling is `0o777`.

The middle rule is the one worth stating slowly, because it is RE2 telling an octal escape from the reference it refuses to read, by looking at the character after the first digit. `\1` alone is `invalid escape sequence: \1`. `\12` is the character with code ten. `\18` is also `invalid escape sequence: \1`, and the message quotes one character of the two the caller wrote, which is worth knowing before reading a bug report about it.

## 4. Where they part

Every line here was measured against the RE2 inside pyarrow 24.0.0 and against CPython's `re`, one pattern at a time, by compiling the pattern under both and comparing the two columns over thirteen texts chosen to tell the readings apart.

```
    \1      invalid group reference    invalid escape sequence: \1
    \9      invalid group reference    invalid escape sequence: \9
    \12     invalid group reference    the character with code 10
    \77     invalid group reference    the character with code 63
    \18     invalid group reference    invalid escape sequence: \1
    \78     invalid group reference    invalid escape sequence: \7
    \99     invalid group reference    invalid escape sequence: \9
    \123    the character with code 83 the character with code 83
```

So the patterns that move are the ones where two octal digits follow the backslash and Python has no group to point them at. `\1` and `\18` and `\99` are refused by both grammars, for reasons that have nothing to do with each other, and they stay refused.

The pair to stare at is `\12` and `\123`. They are one digit apart, and the shorter one, which is the one that looks more like a reference, is the one that is a reference. Going from two digits to three moves the pattern from a construct the two grammars disagree about into one they agree about exactly, because three octal digits are an octal character to Python as well.

The other thing worth naming is that this is the first construct in five slices where which grammar reads a pattern depends on something other than the characters in front of the reader. `\12` is a reference when twelve groups have been opened and a character when eleven have, and nothing about the two characters says which.

## 5. The ceiling

Python's octal escape is a byte and RE2's is a character.

Python refuses anything above `0o377` with `octal escape value outside of range 0-0o377`, which is 255 and is the largest byte. RE2 takes three octal digits and reads them as a character, so `\400` is U+0100 and `\777` is U+01FF, and both were measured matching the rows holding those characters and nothing else.

The same holds inside a class, where `[\400]` is a refusal to Python and a set of one character to RE2.

That is the whole of the 120, and it is the simplest thing in this document: one comparison changes from a refusal into a literal that carries Python's sentence beside it.

One case runs the other way and was already right. `[\1]` is the character with code one to Python and `invalid escape sequence` to RE2, because RE2 will not read a nonzero octal escape shorter than two digits even in a class where there is no reference for it to be confused with. The parser already recorded that with `re2_refuses`, which is the flag for a construct Python reads and RE2 does not, and nothing in this slice touches it.

## 6. The reference the router missed

The slice turned up a case that is not about the grammar at all, and it is a consequence of an upstream bug this library reproduces on purpose.

The walk pandas does to decide which engine answers a call enters a subpattern and a branch and nothing else, so a repeat node is opaque to it, and document 76 recorded that. A pattern holding a backreference is supposed to be answered by Python. A pattern holding a backreference inside a repeat is handed to Arrow instead.

Every such pattern raised there until now, because a reference short enough to write by hand is one or two digits and RE2 refuses `\1`. With twelve groups in front of it the reference is `\12`, and `\12` is a character RE2 reads, so the call comes back with a column:

```
    (abcdefghijkl as twelve groups)\12   answered by Python, matches a repeated l
    ((the same twelve groups)\12)+       answered by RE2, matches a newline
```

Both of those were measured against pandas. The second one is a column of booleans that has nothing to do with what the caller wrote, and it is the answer pandas gives, so refusing it here would be a refusal upstream does not give.

The parser now leaves RE2's reading of the digits on the reference node, one higher so that zero means there is no reading, and the compiler emits that character rather than refusing when the engine is RE2. A one digit reference still refuses, because RE2 has no reading of `\1` to fall back on.

## 7. Where it lands

Three edits in `_digit_escape` in `parse.mojo`, one helper beside it and two branches in `program.mojo`.

The helper is four lines and answers one question: what does RE2 make of the digits Python read as a reference. Only one and two digit runs ever reach it, because Python takes a third digit only when all three are octal and returns a literal when it does, so a run that got as far as being called a reference is a run RE2 stops at after two.

The two ceiling checks, one in the class branch and one outside it, stop giving up and record `python_refuses` with Python's own sentence instead, which is the field document 102 added and this is the fifth use of.

The reference branch asks the helper before it gives up. When RE2 has a reading, the node is a literal and Python's refusal is recorded beside it. When RE2 has none, the refusal stands exactly as it did, and the message a caller sees comes from the RE2 reader on the path document 101 built, which is the right message because it is the one Arrow gives.

The engines were not touched. The router was not touched.

## 8. What the reader needed

Nothing at all, for the second slice running.

The reader in `re2.mojo` was written from RE2's grammar rather than from Python's, so it has read `\12` as a character and refused `\1` and `\18` since the day it was written. Its differential does not move except by the patterns added to the corpus: 8453 read and 21703 refused with matching reasons over 30156 patterns, with nothing wrong in either direction.

That is the second piece of evidence that writing the reader from the other grammar rather than from this parser was the right call. A reader derived from the parser would have had to be changed in the same place at the same time, and a reader that changes with the thing it checks is not a check.

## 9. Where the bias goes

The same place it has gone in 101 through 105. A pattern read wrongly is a column of answers that looks like a right one, and a pattern refused is something a caller can read and act on.

Section 6 is the one place in this slice where the bias points at reading rather than refusing, and the reason it is safe to follow it there is that the answer was measured against pandas first. The rule stays what it has been: read a construct when the answer upstream gives has been seen, and refuse it when it has not.

## 10. The differential

The corpus generated patterns in this family already, which means this slice has a clean before on the corpus the last one measured.

On that corpus the `str.contains` grammar bucket falls from 703 patterns to 458. Of the 245 that leave it, 234 become compared answers and 11 move to buckets that name their actual construct, three of them to the named character bucket and six to the one for syntax RE2 reads differently. That is 245 against the 282 the list in section 1 counted, and the other 37 hold a digit escape beside a second construct this library still cannot read.

The measured list in the corpus goes from 114 patterns to 159, the 45 added covering each line of section 4, both halves of the ceiling, the class forms and the four reference shapes of section 6.

On the widened corpus of 30156 patterns, `str.contains` is 28920 compared and 1236 held out, `str.match` and `str.fullmatch` are both 28683 and 1473, `str.count` is 28906 and 1250, `str.replace` is 29246 and 910, and the Python engine differential through `str.findall` is 29682 and 474. The routing differential compares 29325. All six are at 10000 agreements in ten thousand with zero disagreements.

## 11. What is left

The list in section 1, less this slice.

The 173 and the 10 about group names, which are two rules about which characters a name may hold and whether a name may be used twice, and which are now the largest item.

The 148 about a range in a class, as in `[\d-a]`, which Python refuses and RE2 reads as three members rather than a range.

The 4 about a repeat on a repeat and the 1 about an unknown extension, which are small enough to fold into whichever slice reaches them first.

And after those, the roughly 240 patterns RE2 reads differently rather than not at all and the roughly 113 where RE2 puts a non boundary between two bytes of one character, which are both about what a pattern means rather than about whether it reads, and which are now the two largest things standing between this library and the whole of the string accessor.

The table from document 103 still holds 163 script names the corpus exercises two of, and the sweep still asks sixteen texts, which is document 90 section 10 and is still true.
