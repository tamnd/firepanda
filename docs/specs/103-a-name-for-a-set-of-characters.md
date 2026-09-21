# 103. A name for a set of characters

## 1. What this is

Document 101 built a reader for RE2's grammar and document 102 took the first slice of patterns RE2 reads and this library did not. What that slice left behind, counted by the sentence Python's own grammar refused them with, was:

```
    633 bad escape                    \P 216, \Q 181, \p 176, the rest small
    202 nothing to repeat
    173 bad character in group name
    162 invalid group reference
    148 bad character range
    120 octal escape value outside of range 0-0o377
     10 redefinition of group name
      4 multiple repeat
      1 unknown extension ?
```

This slice takes the 392 that write `\p` or `\P`, which is the largest single item on the list and the one document 102 named as coming next. It is also the only one that closes a second thing at the same time, because the reader in document 101 has one construct it refuses to judge and this is that construct.

## 2. Why the table came from Arrow and not from CPython

`tools/gen_regexclass.py` next door reads the three Perl classes out of a running CPython, and this generator reads pyarrow, which is the opposite. The reason is the whole of why the table exists.

`\p` is not Python syntax. CPython's `re` calls it `bad escape \p` and always has, in every version this project supports and every version before them. So pandas routes every pattern holding one to Arrow, and the RE2 inside Arrow is the only engine that will ever run it. Asking CPython what `\p{L}` covers would be asking a library that has no opinion about the question. Asking the published Unicode data files would be asking a third party that neither library consults at run time.

That matters more here than it did for the Perl classes, because RE2 and CPython are not built against the same Unicode release and this table is large enough that the skew shows. Measured while this was written, `\p{L}` and CPython's `unicodedata` disagree about 4302 code points and `\p{Lo}` about 4243, and every one of those is a code point one release has assigned and the other has not. A table copied from CPython would have been wrong about four thousand characters on the day it was written.

So the generator asks the exact engine the caller's pattern will reach. It is one `pyarrow.compute.match_substring_regex` call per name over a column holding every code point that can appear in a row, which is 1112064 of them, and the answer comes back as one boolean per row and folds into runs. Around two hundred calls, a few minutes, and not something to run in a loop.

## 3. What the names are, measured

There is no way to ask an engine for the list of names it knows, so the list is written down and the generator refuses to write anything if RE2 turns one of them down. That is what turns a release that drops a name into a failure in the generator rather than a wrong table shipped quietly.

196 names. 32 general categories, 163 scripts, and `Any`. Between them they are 5803 ranges.

The 32 break into 26 two letter categories and the 6 one letter ones, and only the 26 are measured. The 6 are built as unions of their parts and then checked against RE2, because a union that turns out to be wrong is a fact worth failing on rather than a table worth shipping.

## 4. The two rules nobody would guess

Both were measured and both would have been got wrong.

`\p{C}` does not hold the unassigned code points. It is exactly `Cc` and `Cf` and `Co` and `Cs`, and `\p{Cn}` is not a name RE2 has at all. A reader coming to this from Unicode rather than from RE2 expects a category called other to hold the code points nothing has been assigned to, and it does not, and the gap is 819533 characters wide. This is why the one letter names are generated as unions and checked rather than measured and trusted: had `C` been measured directly the table would have been right and nobody would have known why, and had the union been trusted without the check the table would have been wrong by the same 819533.

`\p{Cs}` is a real name that can never match anything. It is the surrogates, a surrogate has no UTF-8 encoding, and every row reaching the engine arrived as UTF-8. So the name reads, and compiles, and is a set with nothing in it. It is in the table with no ranges under it rather than left out, because leaving it out would make it a name RE2 has not got, and it is a name RE2 has.

Three more rules are smaller and are asserted in the generator so that a release cannot move them quietly. A name is case sensitive, so `latin` is not `Latin` and is not a name. The `Is` prefix several other engines take is not one RE2 takes, so `IsLatin` is not a name either. And `Cn` stays refused.

## 5. The four spellings and the one negation

`\p{Greek}` is the one people write. `\pL` is the braceless form and takes exactly one character, so `\pLu` is the letter category followed by a literal `u` rather than the uppercase letter category. That is RE2's reading and it is not the one the braced form beside it suggests, which is the whole reason it is worth a test of its own.

There are two ways to negate and they mean the same thing. `\P{L}` is the one people write and `\p{^L}` is the other, and RE2 reads them as one construct, so both set the same bit on one node and the tree cannot tell which was written. Writing both, as `\P{^L}`, cancels, which was measured rather than assumed.

All four spellings work inside a character class, where a name is an item like any other. `[\p{L}\p{N}]` is a union and `[^\p{L}a]` is a complement of one, and both fall out of the class machinery already there rather than needing a rule.

## 6. What the ignore case flag does to a name

It widens it, and by a lot. `\p{Lu}` is 1831 code points and `(?i)\p{Lu}` is 3212, measured against the RE2 inside pyarrow.

That is not a special rule, it is the same rule the Perl classes already follow, and it follows for the same reason: none of these sets is closed under folding. A set of uppercase letters that has been folded holds the lowercase ones too. So a name goes through the same `_folded` the classes go through, before the negation is applied and not after it, which is the order document 97 settled and which `[\d\D]` and `[^\W]` already depend on.

What is worth saying is that it costs nothing to get right here because it was got right there. The one line that compiles a name is the one line that compiles a category with a different table under it.

## 7. How it lands in the tree and the program

A new op, `OP_UNICODE`, where `a` is the name's index in the table and `b` is one when the name was negated. Python's parser has no such node because Python's parser has no such escape, which makes this the first op in the file that exists for one engine only.

The tree also records `python_refuses` with the sentence `bad escape \p`, which is the field document 102 added one slice ago for exactly this shape of problem. So a caller who passed a `flags` argument, and therefore landed on the engine that copies Python, gets Python's own words rather than an answer Python would not have given. That field was built for the flag group and is used twice now, which is the first evidence that it was the right shape rather than a second exception.

A name RE2 has not got is not refused with a sentence of this library's own. The parse gives up exactly as it did before the table existed, the compiler never sees a tree, and the reader is asked instead. That is the next section.

## 8. What the reader in document 101 stops declining

Document 101 section 8 recorded one construct the reader would not judge. `\p{...}` was read as something RE2 takes even when it was plainly malformed, because telling a name RE2 knows from one it does not needs exactly this table and there was none. The differential had a bucket for it and the bucket had 166 patterns in it.

The table settles it, so the reader looks the name up and refuses `\p{Cn}` and `\p{Foo}` and `\p{latin}` and `\p{` with the sentence it already had for a class name RE2 has not got. That is what carries a bad name through to the caller as the `ValueError` pandas raises rather than as a `NotImplementedError`, because pandas hands the pattern to Arrow and Arrow says `invalid character class range`.

It also means the reader has nothing left that it is unsure about. The `unsure` field on `Re2Read` and on the cursor, the branch in `re2_reads` that turned a refusal beside something unjudged back into a reading, and the differential bucket that counted them are all gone. A field nothing can set and a bucket that can only print zero are worse than nothing, because a reader believes them.

The bias itself is not gone and is not weakened. It is still the rule every new rule in that file is written under. It just has nothing left to apply to.

## 9. Where the bias goes

The same place it went in 101 and 102. A pattern read wrongly is a column of booleans that looks like a right one, and a pattern refused is something a caller can read and act on.

Here that turns into one rule about the list in section 3. The list is written down rather than discovered, so a name RE2 gains in a later release is a name this library refuses until somebody adds it, which leaves that caller where they already were. A name RE2 loses is a table this library would be wrong about, and that is the direction that matters, so it is the direction the generator asserts on and fails.

## 10. The differential

The corpus generated two `\p` patterns, `\p{L}` and `\P{L}`, which is not enough to measure a table with 196 names in it. It generates twenty now, chosen to cover each thing the section above claims: a category and a script and `Any` and `Cs`, both braced negations and the braceless form, a real Unicode category RE2 has not got, a name in the wrong case, the `Is` prefix, and the three truncations `\p` and `\p{` and `\p{L`. The sixteen texts already held Greek letters and Arabic Indic digits, so a name that covers one of those is told apart from a name that does not without touching them.

That makes this the one slice whose before and after are not the same measurement, because the corpus moved in the same commit as the code. So it was measured twice.

Against the old corpus, which is the same 30052 patterns document 102 reported on, `str.contains` goes from 27752 compared and 2300 held out to 28182 and 1870. The grammar bucket falls from 1472 patterns to 1032. 440 left it, ten of them landing in a bucket that names their actual construct, the named character bucket going from 399 to 404 and `RE2 reads this syntax differently` from 221 to 226, and the other 430 becoming compared answers. That is more than the 392 section 1 predicted, and for the ordinary reason: 392 counted the patterns whose first refusal was the `\p`, and a pattern can hold one beside something else that was being counted somewhere else.

Against the widened corpus, which is what the differential asks from now on, `str.contains` is 28410 compared and 1642 held out with 916 in the grammar bucket, `str.match` is 28489 and 1563, `str.fullmatch` is 28490 and 1562, `str.count` and `str.replace` are both 28396 and 1656, and the Python engine differential is 29578 and 474. All five are at 10000 agreements in ten thousand with zero disagreements.

`regex_re2` is where the second half of this slice shows. It was 8339 patterns read, 21547 refused with matching reasons, 166 declined to judge and 0 wrong in either direction. It is now 8373 read, 21679 refused with matching reasons, 0 declined and 0 read that RE2 refuses, still at zero disagreements. The reader and RE2 now agree about every one of the 30052 patterns, on whether it reads and on why not, with nothing set aside.

## 11. What is left

The list in section 1, less this slice. In the order their size argues for:

The 202 that repeat a zero width assertion, as in `^*` and `\b*` and `\B{0}`. RE2 repeats an assertion and Python refuses it, so this is the shape document 102 was, a construct one grammar has and the other has not, and it needs no table.

The 181 that write `\Q` and `\E`, which is a span of text to be read as literal characters and nothing more.

The 162 that write a backreference RE2 has never had, where `\1` is an octal escape rather than a reference, together with the 120 that write an octal escape Python calls out of range. Those two are one rule about how far a backslash and a digit reach.

The 173 and the 10 about group names, which are two rules about which characters a name may hold and whether a name may be used twice.

The 148 about a range in a class, as in `[\d-a]`, which Python refuses and RE2 reads as three members rather than a range.

And after those, the patterns RE2 reads differently rather than refuses and the ones where RE2 puts a non boundary between two bytes of one character, which are both about what a pattern means rather than about whether it reads, and which document 90 section 10 has been waiting on.

One thing this slice could have done and did not. The table holds 163 script names and the corpus exercises two of them, so the evidence that the other 161 are right is the generator's own check that RE2 still takes each name and the fact that every one of them was measured the same way in the same pass. That is good evidence and it is not the differential, and widening the corpus to name more scripts would cost nothing but a longer list.

The sweep still asks sixteen texts and a corpus of texts still does not exist, which is document 90 section 10 and is repeated here because it is still true.
