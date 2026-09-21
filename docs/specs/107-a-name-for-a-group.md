# 107. A name for a group

## 1. What this is

The sixth slice of the patterns RE2 reads and this library did not. Documents 102 through 106 took the flag group written in a place, the Unicode name, the repeat on a position test, the run of characters taken literally and the backslash followed by a digit, and what those five left behind, counted by the sentence Python's own grammar refuses them with, was:

```
    184 bad character in group name
    140 bad character range
    120 nothing to repeat
      9 multiple repeat
      5 redefinition of group name
```

This slice takes the first and the last, which are 189 patterns together and the largest item on the list, because they are the two halves of one question: which names may a group have, and may two groups have the same one.

The short version is that Python asks whether a name is an identifier and RE2 asks a wider question, and that Python will not let two groups share a name while RE2 does not care.

## 2. Python's rule

Python's rule is one line of its own source: it reads every character up to the terminator, and then asks `name.isidentifier()` of what it read.

Both halves of that matter. The name is read to the terminator rather than checked as it goes, so `(?P<n\>>a)` is the name `n\` and a complaint about the name rather than a complaint about the backslash, and `(?P<a b>a)` is a complaint about `a b` rather than about the space. And the test is Python's identifier test rather than a regular expression rule at all, so the answer for a name written outside ASCII is whatever the Unicode identifier tables say and has nothing to do with the pattern it is in.

The identifier test has a front and a back. The first character must be one Unicode allows an identifier to start with and the rest must be ones it allows an identifier to continue with, and the difference between the two sets is what makes `(?P<n1>a)` a name Python takes and `(?P<1n>a)` a name it does not.

A name may also only be used once. `(?P<n>a)(?P<n>b)` is `redefinition of group name 'n' as group 2; was group 1`, and the check is against every name opened so far rather than against the enclosing group, so `(?P<n>a)|(?P<n>b)` is refused as well even though no subject can be inside both.

## 3. RE2's rule

RE2 asks one question of each character and nothing about the name as a whole.

The rule was measured over every code point below U+11000 rather than read off a description, and it is that a character may appear in a name when its Unicode general category is a letter, a non spacing or a spacing mark, a decimal or a letter number, or connector punctuation. The underscore is in that last category, so it needs no rule of its own.

An enclosing mark is not one of those, and no other kind of number is either, so `(?P<x½>a)` and `(?P<x€>a)` and `(?P<x·>a)` are all refused while `(?P<xé>a)` and `(?P<x١>a)` and `(?P<x‿>a)` are read. There is no front and no back, so a digit may lead. And there is no rule at all about a name appearing twice.

## 4. Where they part

Both grammars were asked about the same one character names, one code point at a time, over every point below U+11000 with a stride above it, against the RE2 inside pyarrow 24.0.0 and against CPython's `re`.

```
    (?P<n1>a)       both read it
    (?P<1n>a)       Python refuses it, RE2 reads it
    (?P<1>a)        Python refuses it, RE2 reads it
    (?P<١>a)        Python refuses it, RE2 reads it
    (?P<é>a)        both read it
    (?P<x½>a)       both refuse it
    (?P<x·>a)       Python reads it, RE2 refuses it
    (?P<a b>a)      both refuse it
    (?P<>a)         both refuse it
```

Three of those four outcomes were already handled. The one that was not is the second row and the two under it, which is a name RE2 takes and Python does not, and which is what the 184 patterns are.

The last row worth pointing at is the middle dot. Unicode allows it in an identifier and puts it in a category RE2 does not look at, so it is the only direction here that runs the other way, and it is a pattern this library reads and marks as one RE2 will not take rather than a pattern it refuses.

## 5. The name used twice

The five is smaller than the 184 and needed less code than the sentence describing it.

Python refuses a name that is already taken and RE2 has no such rule, so `(?P<n>a)(?P<n>b)` and `(?P<a>x)(?P<a>y)(?P<a>z)` and `(?P<n>(?P<n>a))` are all patterns Arrow answers and this library refused.

The parser now records Python's sentence and keeps the first binding, which is the one a use of the name would have resolved to. Nothing can reach that binding, because RE2 has neither `(?P=name)` nor a conditional group and refuses both on sight, so a pattern holding a duplicate name and a use of it is refused by both grammars and stays refused.

## 6. Where it lands

One new rule in `parse.mojo`, written as five small functions, and two fields on the cursor that were already there.

`_name_point` is RE2's question and reads the same category table `\p{Nd}` reads. `_identifier_point` and `_identifier_start` are Python's, and each is the same categories plus or minus a short list of code points Unicode keeps for identifiers that would otherwise be spelled two ways. Those lists were measured rather than copied, and they are short: twenty one ranges on the set Unicode allows after the first character and eighteen on the set it allows at the front. They are the whole difference between the category rule and what `str.isidentifier` answers.

`_name` itself now reads to the terminator and judges afterwards, which is Python's own order and is what section 2 is about. A name that is not an identifier is recorded with `python_refuses` when it is a name RE2 takes and a definition rather than a use, and refused outright otherwise. A name that is an identifier and holds a character RE2 will not take sets `re2_refuses`, which is the flag for syntax Python reads and RE2 has never had.

The compiler was not touched, the engines were not touched and the router was not touched.

## 7. What the reader needed

One function, and it was wrong in a way nothing had caught.

`_is_name_point` in `re2.mojo` said that a name character was an ASCII word character or anything at all outside ASCII, with a comment saying that had been measured. What had been measured was that `(?P<é>a)` is a name RE2 takes, and the rule written down from that one observation was too wide by every punctuation mark, symbol and space outside ASCII. The corpus never wrote one, so the reader's differential never moved.

It reads the same categories the parser does now. The two are written out separately rather than shared, because the reader is meant to be an independent statement of RE2's grammar and a reader that imports its answer from the thing it checks is not a check, which is the argument documents 105 and 106 both leaned on when the reader needed nothing.

## 8. The table and its version

The category table in `unicodedata.mojo` was generated for the Unicode name classes and is older than the one CPython carries here.

That is the right table for RE2's half, and it was confirmed to be: over every code point below U+11000 the reader and the live RE2 agree on whether a name is taken, including six code points CPython knows as letters and RE2 does not, which are the ones Unicode added after RE2's table was built.

It is the wrong table for Python's half, and the cost is 119 code points. Each of them is a character CPython allows in an identifier and this library does not, so a name holding one is refused here where Python reads it. For a pattern that is only a named group this changes nothing a caller sees, because pandas sends such a pattern to RE2, and RE2 refuses the name for the same reason the table does. It only shows as a gap when the pattern also holds a lookaround or a backreference, which is what sends it to Python instead, and there the answer is a refusal where pandas gives a column. That is 119 code points none of which are in the corpus, and the fix is a newer table rather than a rule.

## 9. Where the bias goes

The same place it has gone in 101 through 106. A pattern read wrongly is a column of answers that looks like a right one, and a pattern refused is something a caller can read and act on.

Section 8 is the one place this slice pays the bias rather than collects it, and it pays it in the safe direction: a refusal where upstream answers, rather than an answer where upstream refuses.

## 10. The differential

The corpus generated patterns in this family already, in the shape `(?P<1n>%)`, which means this slice has a clean before on the corpus the last one measured.

On that corpus the `str.contains` grammar bucket falls from 458 patterns to 279. 189 of them hold one of the two constructs this slice takes, 10 of those hold a second construct the same bucket still names, which is a range in a class or a repeat with nothing in front of it, and 179 leave the bucket.

The measured list in the corpus goes from 159 patterns to 199, the 40 added covering each row of section 4, the duplicate in three shapes, the names neither grammar takes, the name only Python takes, the unterminated forms and the three patterns section 11 leaves behind.

On the widened corpus of 30196 patterns, `str.contains` is 29113 compared and 1083 held out, `str.match` and `str.fullmatch` are both 28874 and 1322, `str.count` is 29099 and 1097, `str.replace` is 29454 and 742, and the Python engine differential through `str.findall` is 29722 and 474. The routing differential compares 29365, and the RE2 reader reads 8477 and refuses 21719 with matching reasons and nothing wrong in either direction. All seven are at 10000 agreements in ten thousand with zero disagreements.

## 11. What is left

The list in section 1, less this slice.

The 140 about a range in a class, as in `[\d-a]`, which Python refuses and RE2 reads as three members rather than a range, and which is the largest item now.

The 120 about a repeat with nothing in front of it, which is almost all `{,3}`, a brace pair Python reads as a count from zero and RE2 reads as four literal characters.

The 9 about a repeat on a repeat, which is small enough to fold into whichever slice reaches it first.

And after those, the roughly 240 patterns RE2 reads differently rather than not at all and the roughly 113 where RE2 puts a non boundary between two bytes of one character, which are both about what a pattern means rather than about whether it reads.

One spelling of a named group is left over on purpose. `(?<name>a)` is a named group to RE2 and `unknown extension ?<n` to Python, and it is a different question from this one, since it is about how the group is written rather than about the name inside it. It is three patterns of the corpus now, and it goes with the other group spellings rather than with the names.

The table from document 103 still holds 163 script names the corpus exercises two of, and the sweep still asks sixteen texts, which is document 90 section 10 and is still true.
