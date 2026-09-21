# 108. A dash after a set

## 1. What this is

The seventh slice of the patterns RE2 reads and this library did not. Documents 102 through 107 took the flag group written in a place, the Unicode name, the repeat on a position test, the run of characters taken literally, the backslash followed by a digit and the name for a group, and what those six left behind, counted by the sentence Python's own grammar refuses them with, was:

```
    143 bad character range
    126 nothing to repeat
     10 multiple repeat
      3 unknown extension ?<
```

This slice takes the first, which is the largest item on the list and is one shape written one way. It is `[\d-a]`, a class holding a set of characters, a dash and a character, which Python refuses and RE2 reads.

The short version is that a range needs one character on the left of it, and the two grammars find that out at different times.

## 2. Python's rule

Python reads the left hand item, and then, if the next character is a dash and the dash is not the last thing in the class, it commits to a range and reads the item after it. Only then does it look at what it has. If either end is a set of characters rather than one character, or if the two ends run backwards, the class is refused with `bad character range` and the text of the range it was trying to build.

The commitment is the whole of it. By the time Python knows that `\d` is a set it has already decided a range is being written, and it has nothing to do with that decision except refuse. The same line refuses `[\d-a]`, `[a-\d]`, `[\d-\w]` and `[b-a]`, which are four different complaints in any description of the grammar and one branch in the implementation.

## 3. RE2's rule

RE2 asks before it commits. A dash opens a range only when what was last read inside the class was one character, and when it was a set of characters the dash is a dash.

So `[\d-a]` is three members, the digits and a dash and a letter, and the item after the dash is read as a fresh item rather than as the top of a range. That last part is worth writing down because it is visible: `[\d-a-z]` is the digits, a dash and the range `a` to `z`, not the digits and a dash and two letters, because once the dash has been taken as a member the next dash has a single character in front of it again.

On the other side RE2 is stricter than Python rather than looser. A range has already been opened when the top of it is read, and an escape that names a set is not something RE2 has a reading for there, so `[a-\d]` is `invalid escape sequence: \d` rather than a complaint about the range. Both grammars refuse it, for reasons that do not resemble each other.

## 4. Where they part

Both grammars were asked about the same classes, against the RE2 inside pyarrow 24.0.0 and against a running CPython.

```
    [\d-a]          Python refuses it, RE2 reads it
    [\D-a]          Python refuses it, RE2 reads it
    [\s-a]          Python refuses it, RE2 reads it
    [\d-\w]         Python refuses it, RE2 reads it
    [^\d-a]         Python refuses it, RE2 reads it
    [\d-a-z]        Python refuses it, RE2 reads it
    [a-\d]          both refuse it, for different reasons
    [b-a]           both refuse it
    [a-\n]          both refuse it
    [\d-]           both read it
    [a-b-c]         both read it
```

One direction, and only one. There is no class in this family that Python reads and RE2 does not, which makes this the simplest slice since the literal run and the only one where the whole difference is a single branch taken or not taken.

The last two rows are the ones that keep the branch honest. A dash at the end of a class is a member to both grammars and always was, and a dash after a completed range is a member to both, so this slice is only about a dash after a set and not about the dash generally.

## 5. Where it lands

One branch in `_class` in `parse.mojo`, taken when the left hand item is not a literal and there is a dash after it. The dash becomes a member, the loop goes round again, and the item after the dash is read by the next turn of it, which is what gives `[\d-a-z]` its range for free rather than by a second rule.

The sentence Python refuses with is recorded on the cursor rather than given up on, which is the same idiom documents 101 through 107 all used and the eighth place it appears. The compiler was not touched, the engines were not touched and the router was not touched, because a class holding a member and a range is the node the compiler already emits.

The docstring on `_class` had a fourth rule written on it that called this a refusal rather than a permission. That paragraph is now the place the disagreement is written down, and it is the only prose in the file that had to change.

## 6. What the reader needed

Nothing, for the fourth time in seven slices.

`re2.mojo` has had this rule since it was written. It carries two constants for what was last read inside a class, one for a single character and one for a set, and `_escape_is_class` with a docstring saying that a set cannot be one end of a range so the hyphen after one is a hyphen. The reader has been reading `[\d-a]` correctly the whole time and refusing `[a-\d]` with RE2's own sentence, and the corpus wrote `[\d-a]` often enough that the reader's differential had been confirming it on every run. The parser was the half that was wrong.

That is the opposite of document 107, where the corpus had never written the character that would have caught the reader, and it is the argument for keeping the two statements of the grammar separate rather than sharing one: the disagreement between them is what finds the error, and which of the two is wrong is not something either of them knows.

## 7. Where the bias goes

The same place it has gone in 101 through 107, and this slice collects rather than pays. A pattern read wrongly is a column of answers that looks like a right one, and a pattern refused is something a caller can read and act on. Every row in section 4 that moved is a refusal here becoming the answer pandas gives, and nothing in it moves the other way.

## 8. The differential

The corpus generated patterns in this family already, in the shape `[\d-a]`, so this slice has a clean before on the corpus the last one measured.

On that corpus the `str.contains` grammar bucket falls from 282 patterns to 139, and the `bad character range` line goes to nothing rather than to a smaller number, because the corpus only ever wrote one shape of it and every other shape in section 4 is one both grammars refuse or both grammars read.

The measured list in the corpus goes from 199 patterns to 239 and the corpus from 30196 to 30236, the 40 added covering each row of section 4, the negated class, the class that opens with a bracket, the dash against each of the six Perl classes, the pairings with a repeat and a backreference and a lookaround, and the two shapes that are a different complaint rather than this one.

On the widened corpus of 30236 patterns, `str.contains` is 29288 compared and 948 held out, `str.match` and `str.fullmatch` are both 29044 and 1192, `str.count` is 29274 and 962, `str.replace` is 29629 and 607, `str.findall` is 29762 and 474, the routing differential compares 29405, and the RE2 reader reads 8508 and refuses 21728 with matching reasons and nothing wrong in either direction. All seven are at 10000 agreements in ten thousand with zero disagreements.

The gap between the grammar bucket on `str.contains`, which is 139, and the one on `str.match`, which is 385, is worth a line, because it is not this slice and it is the next one. `match` strips one leading caret and wraps what is left, so `^*` becomes `^(*)`, and a repeat that had a position in front of it now has nothing in front of it. That is 244 of the 246, the other two holding a backreference and landing in this bucket by a different road, and every one of the 244 is a pattern the `{,3}` slice will reach.

## 9. What is left

The list in section 1, less this slice.

The 126 about a repeat with nothing in front of it, which is almost all `{,3}`, a brace pair Python reads as a count from zero and RE2 reads as four literal characters, and which is the largest item now and the last one of any size.

The 10 about a repeat on a repeat.

The 3 about `(?<name>a)`, which is a named group to RE2 and `unknown extension ?<n` to Python, and which is left over from document 107 on purpose because it is a question about how a group is written rather than about the name inside it.

After those the list of patterns RE2 reads and this library does not is empty on this corpus, and what remains is the two families that are about what a pattern means rather than whether it reads: the 251 where RE2 reads the syntax differently, which the POSIX class is the largest part of, and the 122 where RE2 puts a non boundary between two bytes of one character.

The table from document 103 still holds 163 script names the corpus exercises two of, and the sweep still asks sixteen texts, which is document 90 section 10 and is still true.
