# 96. Two ways to drop a case

## 1. What this is

The engine that copies Python now answers a backreference under `(?i)`, which was the one thing document 95 left behind. The slice is a table and a comparison. The table is `firepanda/kernel/regex/lowerdata.mojo`, written by `tools/gen_regexlower.py`, and it says what one code point's simple lowercase is. The comparison is three lines in `Bounded._reads_again`.

It is a small slice with a long reason, and the reason is the whole of why it is a document rather than a paragraph in the changelog. Case insensitivity in this library was, until now, a thing that happened entirely while a pattern was being compiled. This is the first piece of it that happens while a row is being walked, and the two do not agree.

## 2. Two questions that look like one

Ask what `(?i)` means and the natural answer is that it makes the pattern ignore case. That is one sentence covering two different mechanisms, and upstream implements them with two different functions that give different answers on real text.

A literal under the flag is widened. The compiler takes the character, looks up every other character that folds onto it, and emits a set holding all of them. `(?i)s` becomes the set of `s`, `S` and the long s, because Unicode says those three are one letter for folding purposes. That is document 76's work and `folddata.mojo` is its table.

A backreference under the flag cannot be widened, because what it is going to be compared against is not known until the row is being walked. There is nothing to put in a set. So it is compared at run time instead, and upstream compares the two characters by lowering each of them with `sre_lower_unicode`, which is `Py_UNICODE_TOLOWER`, and asking whether the results are equal.

Those two are not the same relation. Folding groups the long s with `s`. Lowering sends `s` to `s` and the long s to the long s, which are two different characters. So the two mechanisms disagree, and they disagree inside a single pattern:

    re.search(r"(?i)ss", "sſ")      matches
    re.search(r"(?i)(s)\1", "sſ")   does not

Measured on CPython 3.13.12. The sigma pair goes the same way, `(?i)σσ` matching sigma followed by final sigma and `(?i)(σ)\1` not. The dotless i goes the other way round, `(?i)i` matching it because Python folds all four members of that family together and `(?i)(ı)\1` not matching it followed by a plain `i` because it lowers to itself.

This is not a defect upstream and it is not being reported as one. Lowering is a reasonable thing for a run time comparison to do and folding is a reasonable thing for a compile time widening to do, and a library that wants the same answers as upstream has to have both.

## 3. The rows that were measured

Everything below was measured against a running CPython 3.13.12 rather than read out of the Unicode data files, and the same rows are the test file's assertions.

| pattern | text | answer | why |
| --- | --- | --- | --- |
| `(?i)ss` | `sſ` | matches | the long s is in the fold orbit of `s` |
| `(?i)(s)\1` | `sſ` | no | the long s lowers to itself |
| `(?i)(σ)\1` | `σς` | no | final sigma lowers to itself |
| `(?i)(é)\1` | `éÉ` | matches | the capital lowers onto the small |
| `(?i)(İ)\1` | `İi` | matches | the dotted capital lowers onto a plain `i` |
| `(?i)(ı)\1` | `ıi` | no | the dotless i lowers to itself |
| `(?i)(ß)\1` | `ßẞ` | matches | the capital sharp s lowers onto the small one |
| `(?i)(k)\1` | `k` and the Kelvin sign | matches | the Kelvin sign lowers onto a `k` |
| `(?ai)(k)\1` | `k` and the Kelvin sign | no | the ASCII rule is the alphabet and nothing else |
| `(?ai)(é)\1` | `éÉ` | no | the same |

The last two are section 7's point in two rows. `(?ai)` is not `(?i)` with a narrower table, it is a different rule.

## 4. The table

1433 code points on CPython 3.13.12 have a simple lowercase that is not themselves. The generator asks `_sre.unicode_tolower` about every code point in Unicode, which is the same function the regular expression engine calls, rather than asking `str.lower` or `unicodedata`, because `str.lower` is the full lowercase and turns one character into three where the simple one does not.

Written as runs of a low, a high and a delta, the same shape the fold table has, that is 668 runs. Most of the waste is one particular layout: long stretches of the alphabet are laid out as an upper code point immediately followed by its lower, so over such a stretch the even code point moves by one and the odd one does not move at all. An ordinary run cannot say that, so each letter becomes a run of its own.

The fold table has a marker for a stretch like that, `FOLD_EVEN_ODD`, meaning the even one steps up and the odd one steps down, because folding is a cycle and both directions are in it. Lowering is not a cycle and only one direction is in it, so this table has its own marker, `LOWER_EVEN_ONLY`, meaning the even one steps up and the odd one stays. Twenty seven runs carry it and the table comes to 214 runs, of which 187 are ordinary. Of those 187, sixty two take a delta of one, which is the same step the marker stands for, left as a plain run wherever the stretch of pairs was too short to be worth marking.

Both markers are `-0x40000000`, which is a number no real delta can be.

## 5. What the generator checks

Two things, and the first is the one that matters.

It walks its own runs for every code point in Unicode and compares the answer against `_sre.unicode_tolower`. Every code point rather than every code point that moves, because the failure a run encoding makes is a run that swallows a neighbour, and a neighbour that does not move is invisible to a check that only asks about the ones that do.

It also checks that every code point that lowers onto something is one `_sre.unicode_iscased` says is cased. That is the join between this table and the fold table: they are built over the same code points, because upstream groups the fold orbits by lowercase in the first place, and a code point that lowered without being cased would be one this engine folds one way and lowers another with nothing to notice it.

## 6. Where it is read

`simple_lower` in `backtrack.mojo` is a binary search of the runs, and it is the only caller of the table.

The table is brought out of the compiler's world the same way the word ranges are, by `materialize` and a copy, and it is held on `Bounded` rather than copied per row. It is two and a half kilobytes, so `__init__` only asks for it when the program it was handed actually holds an `IN_REF` with the wide comparison on it. A pattern with no backreference in it, or with one under no flag or under `(?ai)`, does not pay for the table at all.

That check happens in the loop `__init__` already walks for the two Unicode word boundaries, so it costs no second pass.

## 7. Three comparisons, not two

`IN_REF`'s `b` used to be a flag: one for folded, zero for not. It is now one of three constants, and the three are three rules rather than one rule at two widths.

`REF_EXACT` compares the two runs code point by code point. `REF_NARROW` is `(?ai)`, and is the twenty six ASCII letters and nothing else: the Kelvin sign is not a `k` and the long s is not an `s`, which is the same rule `(?a)` imposes on a folded literal and is written out in `_fold_one` for the same reason. `REF_WIDE` is `(?i)`, and is the table.

The choice is made in `_emit_node` rather than in `_check_node`, for the reason document 95 section 7 gives: the flag is scoped, and only the walk that emits knows which scope it is standing in. `(?i:(a)\1)` has the reference inside the scope and folds. `(?i:(a))\1` has it outside and does not, so it wants the two characters to be the same character even though the group that caught the first one was folding. Both are answered now where before the first was refused.

A pair that is already equal is not lowered. That is a saving rather than a rule, and it is worth naming because it makes the wide comparison cost nothing on the ordinary row where the two characters match outright.

## 8. Which interpreter this was generated against

CPython 3.13.12, carrying Unicode 15.1.0, which is the interpreter `folddata.mojo` was generated against.

The skew is real and it is the same skew the fold table already lives with. On CPython 3.14.7, carrying Unicode 16.0.0, 1460 code points lower rather than 1433, and the fold table would have 2981 cased code points in 1521 groups where 3.13 has 2927 in 1494. So both tables are the answer of the version named in their headers rather than of whichever version happens to be running, and picking a second version for the second table would have made two skews where there is one.

Choosing this rather than 3.14 is worth a sentence because the pixi environment here is 3.14.7 and the differential oracle runs on it. The differential compares against `re` on whichever interpreter it is given, so a disagreement caused by the skew would show up there rather than hiding, and none did, because the corpus does not reach the code points the two versions differ over.

## 9. What the corpus said

Before this slice the Python differential compared 28729 of the 30052 patterns and held out 1323, of which 10 were `this engine has no backreference under the ignore case flag yet`. After it, it compares 28739 and holds out 1313, the bucket is gone, and every other bucket is to the pattern what it was: 399 for a named character, 393 for a possessive quantifier, 202 for a conditional group, 196 for an atomic group, 69 for a repeat count, 28 for a lookaround beside a backreference and 26 for a capture inside a lookahead. Nothing that used to be compared stopped being compared and nothing that used to agree stopped agreeing.

Ten patterns is the smallest move any of these slices has made and that is the point of it. The work was the table rather than the patterns, and the ten are the corpus noticing that a rule which had been guessed at is now answered.

Five differentials at zero disagreements, as before.

## 10. What is not here

The full lowercase, which turns the capital sharp s into two characters and the Turkish dotted capital into two, and which `str.lower` does and `sre_lower_unicode` does not. Nothing in a regular expression asks for it.

Upper case. There is no construct in the engine that needs to raise a character's case, and adding a table nothing reads would be adding a table nothing reads.

The locale reading of the flag, which is `re.LOCALE`, and which is refused at compile time along with every other use of that flag and is not this slice's business.

A fast path for a folded literal, which document 86 section 7 named and which is still named. `case=False` on a replace spends the engine where it used to spend a byte search, and that is about literals rather than about references.

The lookaround beside a backreference, which is document 95 section 10 and is still 28 patterns.

The sixteen texts, which every document since 90 has named and which are still sixteen.
