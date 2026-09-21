# 109. A count with no lower bound

## 1. What this is

The eighth slice of the patterns RE2 reads and this library did not, and the last one of any size. Documents 102 through 108 took the flag group written in a place, the Unicode name, the repeat on a position test, the run of characters taken literally, the backslash followed by a digit, the name for a group and the dash after a set, and what those seven left behind, counted by the sentence Python's own grammar refuses them with, was:

```
    126 nothing to repeat
     10 multiple repeat
      3 unknown extension ?<
```

This slice takes the first two, which are 136 patterns and are one construct seen from two sides. The construct is `{,3}`, a count written with no lower bound.

The short version is that Python has read `{,n}` as `{0,n}` since 3.11 and RE2 has never read it as a count at all, so when there is nothing for Python's count to go on there is only one reading left and it is RE2's.

## 2. The two readings

Python reads `{,n}` as `{0,n}`. That is a change made in 3.11 and it is the whole of Python's rule, so `a{,3}` is `a` repeated up to three times and `{,3}` with nothing in front of it is a count with nothing to repeat.

RE2 reads `{,n}` as four characters. A count to RE2 opens with a digit, and a brace that is not followed by one is an ordinary character, so `a{,3}` matches the text `a{,3}` and `{,3}` matches the text `{,3}`.

Both were measured rather than read off a description, against the RE2 inside pyarrow 24.0.0 and against a running CPython.

```
    a{,3}       Python counts, RE2 spells
    {,3}        Python refuses, RE2 spells
    {,}         Python refuses, RE2 spells
    ^{,3}       Python refuses, RE2 spells
    9*{,3}      Python refuses, RE2 spells
    {2}         both refuse
    {2,}        both refuse
    {}          both spell
    a{3}        both count
```

## 3. Why the first row is a different question from the rest

The first row is a pattern with two readings and no refusal in it. Python answers one column and RE2 answers another, and this library holds one tree rather than one per engine, so a tree cannot carry both. That row is marked as one the two grammars do not agree about and held out, which is where it already was and where it stays.

Every other row where the two part has a refusal on Python's side. A pattern Python refuses has no Python column to lose, so RE2's reading is the only reading and the tree can hold it. That is the same argument documents 101 through 108 all made, and it is why the slice is drawn where it is: not at the construct, which is one construct, but at whether the construct leaves Python anything to say.

## 4. What Python refuses it with

Two sentences, and which one depends on what is in front of the brace rather than on the brace.

With nothing in front of it at all, or with a position test such as `^` or `\b` in front of it, Python says there is nothing to repeat. With a repeat in front of it, as in `9*{,3}`, Python says it is a multiple repeat. Both are recorded as written, because the caller who reached Python's engine passed a `flags` argument and Python's sentence is the one they are owed.

The position test row is document 104 arriving at the same place from the other side. That slice read `^*` into the tree as a repeat on an anchor because RE2 repeats a position test, and this one reads `^{,3}` into the tree as an anchor and four characters, because RE2 has no count there to repeat anything with.

## 5. Where it lands

`_counted` in `parse.mojo` now reports which of the two forms it read rather than deciding what to do about it, because what to do depends on what is in front of the brace and that is a fact the count reader does not have. The sequence reader makes the decision, in the branch it already had for a quantifier with nothing to repeat.

When the count has no lower bound and Python has nothing to put it on, the cursor goes back to the opening brace and the characters go in one node each. That is the same shape the literal run of document 105 builds, and it buys the same thing for free: a count written after this one finds the closing brace in front of it and repeats that one character, so `{,3}{2,}` is three characters and a repeat of the fourth. Nothing was written down to make that happen.

The compiler was not touched, the engines were not touched and the router was not touched.

## 6. What the reader needed

Nothing, for the fifth time in eight slices. `re2.mojo` reads a brace that is not followed by a digit as an ordinary character, which is RE2's rule and was already written down.

## 7. Where the bias goes

The same place it has gone in 101 through 108. A pattern read wrongly is a column of answers that looks like a right one, and a pattern refused is something a caller can read and act on.

Section 3 is the place this slice declines to collect. `a{,3}` could be answered by reading it RE2's way, and it is not, because Python's reading of it is a real reading that a caller who passes a `flags` argument is owed, and a tree that held RE2's would be wrong for them. A held out pattern is a refusal with a sentence, which is the safe side.

## 8. The differential

The corpus generated patterns in this family already, so this slice has a clean before on the corpus the last one measured.

On that corpus the `str.contains` grammar bucket falls from 139 patterns to 6, which is 3 about the other spelling of a named group and 3 about a braceless count written after one that had already been read as Python's, as in `\.{,2}{,2}`. That last three is the case section 3 cannot reach without unwinding a reading it has already committed to, and it is left.

The measured list in the corpus goes from 239 patterns to 279 and the corpus from 30236 to 30276. On the widened corpus the same bucket is 7, the extra one being a fourth pattern of the shape section 8 already names.

On the widened corpus the seven comparisons read:

```
    routing         compared 29445, held out 831
    str.contains    compared 29453, held out 823
    str.match       compared 29209, held out 1067
    str.fullmatch   compared 29209, held out 1067
    str.count       compared 29439, held out 837
    str.replace     compared 29796, held out 480
    str.findall     compared 29802, held out 474
```

Every one of the seven answered ten thousand agreements in ten thousand with no disagreements, and the RE2 grammar reader read 8544, refused 21732 and read nothing RE2 refuses.

## 9. What is left

The list in section 1, less this slice, is 3 patterns about `(?<name>a)`, which is a named group to RE2 and `unknown extension ?<n` to Python, and the ones that section 8 names. That is the end of the list: on this corpus there are seven patterns left that RE2 reads and this library does not, out of thirty thousand.

What remains after that is not about whether a pattern reads but about what it means, and it is two families. The larger is what RE2 reads differently, which the POSIX class and the first row of section 2 are both in. The other is where RE2 puts a non boundary between two bytes of one character.

The table from document 103 still holds 163 script names the corpus exercises two of, and the sweep still asks sixteen texts, which is document 90 section 10 and is still true.
