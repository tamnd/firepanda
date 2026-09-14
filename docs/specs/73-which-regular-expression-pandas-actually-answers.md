# 73. Which regular expression pandas actually answers

## 1. Why this document exists before any engine does

Issue #158 says the engine to write is RE2, on the grounds that pandas answers its pattern methods out of Arrow and Arrow uses RE2. Writing an RE2 and stopping would produce wrong answers on five of the nine methods and would produce wrong answers on the other four for patterns that look entirely ordinary.

This document is what the measurements say instead. Everything in it was measured against pandas 3.0.5 with pyarrow 25.0.1, and the parts about how pandas decides come from reading `ArrowStringArray` rather than from guessing.

The short version is that pandas runs two regular expression engines with different semantics, picks between them per call using a static property of the pattern, and the choice is visible in the answer rather than only in which patterns are refused.

## 2. The switch

`ArrowStringArray._str_contains` and its four siblings begin like this:

```python
if (
    flags
    or self._is_re_pattern_with_flags(pat)
    or (regex and self._has_unsupported_regex(pat))
):
    return super()._str_contains(pat, case, flags, na, regex)
```

`super()` is `ObjectStringArrayMixin`, which is Python's own `re` applied row by row. If none of the three conditions holds, the pattern goes to Arrow, which is RE2.

`_has_unsupported_regex` parses the pattern with `re._parser`, walks the token tree, and answers True if it finds `ASSERT`, `ASSERT_NOT` or `GROUPREF`, which is lookahead, lookbehind and a backreference. If `re._parser` cannot parse the pattern at all it answers False and lets Arrow have it.

So the rule is: a pattern Python can read, which contains a lookaround or a backreference, is answered by Python. Everything else is answered by RE2.

## 3. Which methods can reach RE2 and which cannot

Five consult the switch: `contains`, `count`, `match`, `fullmatch` and `replace`.

Four never reach Arrow at all, because `ObjectStringArrayMixin` defines them and nothing overrides: `extract`, `extractall`, `findall`, and `split` and `rsplit` in their regular expression form.

That second group is the part that makes an RE2 only engine wrong rather than incomplete. `extract` and `extractall` and `findall` are the three methods issue #158 describes as being entirely about capture groups, and all three are Python's `re` in pandas, with Python's semantics, always. An engine written to RE2's rules answers them wrongly and no amount of finishing it will help.

## 4. The same question, two answers, in one accessor

A column of one Arabic-Indic three, one fullwidth three, one ASCII three and one `é`, each followed by an `x`:

```
s.str.contains(r"\d")   [False, False, True, False]
s.str.findall(r"\d")    [['٣'], ['３'], ['3'], []]
```

`contains` says the first row holds no digit. `findall` on the same column with the same pattern finds one and hands it back. The two disagree because `contains` can reach RE2, where `\d` is `[0-9]`, and `findall` cannot, so it is Python's `\d`, which is every Unicode decimal digit.

This is not a corner. `\d` is the most written pattern there is.

## 5. The same method, two answers, decided by a lookahead that changes nothing

On the same column:

```
s.str.contains(r"\d")       [False, False, True, False]
s.str.contains(r"\d(?=x)")  [True,  True,  True, False]
```

Every row ends in `x`, so the lookahead excludes nothing. It changes the answer anyway, because it is a lookaround, so the call leaves RE2 and `\d` stops meaning `[0-9]` and starts meaning every Unicode decimal digit.

Adding an assertion that cannot fail changes which characters are digits. That is the single sharpest way to state what this surface is, and any library claiming compatibility has to reproduce it.

## 6. The four differences that are visible in answers

For syntax both engines accept, these are the ones measured so far. The RE2 column is a pattern with no lookaround, the Python column is the same pattern with a lookahead appended that matches the empty string.

| Pattern | Rows | RE2 | Python |
|---|---|---|---|
| `^\d$` | `٣`, `3` | `[False, True]` | `[True, True]` |
| `^\w$` | `é`, `a`, `_` | `[False, True, True]` | `[True, True, True]` |
| `^\s$` | no-break space, space | `[False, True]` | `[True, True]` |
| `^a$` | `a\n`, `a` | `[False, True]` | `[True, True]` |

The first three are the same difference three times: RE2's Perl classes are ASCII by default and Python's are Unicode aware for a `str` pattern. The fourth is a different one. Python's `$` matches at the end of the text and also just before a newline that ends the text. RE2's `$` outside multiline mode matches only at the end.

Several things agree and are worth recording as agreeing, because each was a plausible place to differ: `.` excludes a newline in both, `(?s)` and `(?m)` mean the same in both, alternation is leftmost first in both rather than leftmost longest, and `(?i)` folds `K` onto `k` in both and does not fold `ß` onto `ss` in either.

`\Z` agrees only because pandas rewrites it. `_preprocess_re_pattern` rewrites a trailing `\Z` to `\z` before handing the pattern to Arrow, counting the backslashes before the `Z` to be sure the `\Z` is not itself escaped. Python's `\Z` and RE2's `\z` both mean end of text, and RE2's `\Z` does not exist. That rewrite is upstream papering over one instance of the same class of problem the row above is another instance of.

## 7. The patterns RE2 refuses, which pandas does not catch

Python 3.11 added possessive quantifiers and atomic groups, so `re._parser` reads `a*+` and `(?>a)` without complaint. Neither is a lookaround or a backreference, so the detector says the pattern is fine and hands it to RE2, which rejects both:

```
s.str.contains("a*+")   ArrowInvalid: Invalid regular expression: bad repetition operator: *+
s.str.contains("(?>a)") ArrowInvalid: Invalid regular expression: invalid perl operator: (?>
```

So pandas raises an Arrow error for a pattern Python's own engine would have run, and the error names Arrow rather than naming pandas or the argument. These are refusals to copy in kind rather than in wording.

## 8. The corner where the switch cannot see

`\p{L}` is RE2 syntax that Python cannot parse. `_has_unsupported_regex` catches the `re.error`, answers False, and the pattern goes to RE2, which runs it. That works and is deliberate.

Now put a lookahead in it:

```
s.str.contains(r"\p{L}(?=x)")
ArrowInvalid: Invalid regular expression: invalid perl operator: (?=
```

Python still cannot parse the pattern, so the detector still never sees the lookahead, so the pattern still goes to RE2, which rejects the lookahead it was never told about. A pattern using one engine's syntax plus the other engine's feature is answerable by neither and pandas cannot route it.

The switch is therefore not "does the pattern contain a lookaround". It is "does Python's parser report a lookaround", and a lookaround standing behind syntax Python cannot read is invisible to it.

## 9. What this means for what gets written here

Two engines, because that is what the surface is. An RE2 with RE2's classes and RE2's `$`, and a Perl style engine with Python's classes and Python's `$`, with a shared parser front end where the syntax overlaps and different class tables and different end anchors behind it.

Both have to be linear time. The original argument in issue #158 for RE2 was that a pattern coming out of user data must not be able to hang a process running it over a million rows, and that argument does not weaken because the semantics being reproduced are Python's. Lookaround and backreferences are the two constructs that make linear time hard, and they are exactly the two that route to the Python side, so the Python side is where the real work is. Backreferences genuinely cannot be done in linear time in general. Bounded lookaround can.

The routing rule has to be reproduced exactly, including the two things about it that read like bugs. A pattern must be parsed before it is run so that the presence of a lookaround or a backreference can be decided, and that decision then picks the class tables and the end anchor for the whole call. The parse that decides has to be Python's grammar, since it is Python's parser making the call upstream, and a pattern this parser cannot read must route to the RE2 side even when the reason it could not read it is a lookaround.

Nothing here is a divergence to register yet, because none of it is implemented. When it is, the parts to argue about are whether to copy the `\p{L}` with a lookahead corner and whether to copy pandas raising Arrow's sentence for a possessive quantifier. The answer to the first is probably yes, because a caller writing that pattern is confused in a way an error should reflect. The answer to the second is probably no in wording and yes in kind.

## 10. What is not measured yet

The differences in section 6 are the ones found by asking the obvious questions. The list is not closed and should not be treated as closed. Not yet measured: whether the two engines agree on the empty match rule in `replace` and `count`, whether they agree on which of several equal length alternatives a capture group ends up holding, what `(?i)` does to a Unicode class in each, and whether RE2's leftmost first is leftmost first in every case or only in the ones asked about here.

Each of those is a place a compatibility layer can be quietly wrong, and each should be measured before the corresponding piece is written rather than after a case fails.
