# 89. A letter with a reach

## 1. What this is

Document 88 read the last two of the seven inline flag letters and ended by naming the one thing about a flag this library still turned down for a reason of its own. That was the scoped form, `(?i:a)`, where a letter applies to part of a pattern rather than to all of it. 525 patterns in the held-out set were counted under it and this is the slice that carries them.

Every other way of writing a flag says the same thing about the whole pattern. `(?i)` at the front, `flags=re.IGNORECASE` beside it and `case=False` beside that are three spellings and one reach, and the compiler could read all three off one field because there was only ever one answer to the question of whether a letter was on. The scoped form is the first one where `a` and `b` in the same pattern can be read under different rules, and a field with one value in it cannot hold that.

## 2. Why a wider field was not the fix

The parse already had a `scoped` field. It held every letter any scoped group mentioned, on or off, flattened over the whole pattern, and it existed so that the RE2 compiler could refuse `(?x:a)` for the same reason it refuses `(?x)a`. That field is a report of which letters were written somewhere and it says nothing about where, which is exactly enough to refuse on and exactly not enough to answer on.

The temptation is to make it two fields, an on set and an off set, and to apply them the way the global flags are applied. That is wrong in a way that is quiet. `(?i:a)b` would fold the `b`, which no engine does, and the pattern would come back with an answer rather than with an error, so the mistake would be a wrong column rather than a raised exception. A letter with a reach has to be carried at the place it reaches, which means on a node.

The field stays anyway, and it stays for the job it was built for. RE2 reads `(?i:`, `(?m:` and `(?s:` and has never heard of `(?x:` or `(?a:` or `(?u:` in any position, so the RE2 compiler still has to answer the question of whether a letter it refuses was written anywhere in the pattern, and answering that by walking the tree looking for scope nodes would be the same question asked the long way round.

## 3. The node Python does not have

CPython's parser hangs a scoped group on a `SUBPATTERN` token whose group number is `None`, with the two flag sets sitting in the second and third slots. That shape cannot be borrowed here. `OP_SUBPATTERN` in this arena carries a capture number in its first payload and the number is a payload rather than an option, so a subpattern with no number would mean teaching every reader of that op code that zero might mean two different things.

So there is an `OP_SCOPE`, with the letters turned on in `a` and the letters turned off in `b` and one child. Document 88's tokens file opens by saying that the op codes are Python's names on purpose, because the routing decision pandas makes is a walk over Python's tokens and a tree whose nodes do not correspond one to one with Python's cannot be checked against Python's. This is the second node that breaks the correspondence, after `OP_SEQ`, and it breaks it in the same direction: Python folds two things into one token and this splits them, which is safe, where folding two of Python's tokens into one would not be.

The node carries every letter that was written and the compiler then ignores one of them, which is the subject of the next section.

## 4. Two kinds of letter, and only one of them reaches the compiler

Six of the seven letters are questions the compiler asks about a character. Does this literal fold, does this full stop take a newline, does this dollar sign mean the end of a line, which alphabet does `\w` read from. All six are read off the builder at the moment a node is emitted, so scoping them is a matter of setting the builder's flags before the children are emitted and putting them back afterwards.

Verbose mode is not one of those. It decides which characters there are at all, by throwing whitespace and comments away before the grammar sees them, and that happens while the pattern is being read rather than while it is being compiled. By the time a program is built there is nothing left of the letter to act on. So the parser turns it on and off around the reading of the inner branch, and the compiler skips it.

The node carries it anyway. Dropping the letter at the point the node is built would make the node a report of what the compiler cares about rather than of what the caller wrote, and the tree has a second reader: the router, which is section 8. A node that is honest about the pattern is worth more than a node that is convenient for one of its readers.

## 5. Nesting is a save and a restore, and one combination is not

`(?i:(?-i:a)b)` on `aB` is True. That is the row that says what the restore has to put back. When the inner group ends, the letters go back to what the enclosing group had, not to what the pattern started with, so the `b` after it is folded because the outer group said so and the `a` before it is not because the inner group said otherwise.

Both halves of the implementation are written that way and for the same reason. The parser saves the cursor's flags, sets them, reads the branch and restores them. The compiler saves the builder's flags, sets them, emits the children and restores them. Neither clears anything, and a version that cleared would pass every test with one level of nesting in it.

Three of the seven letters do not combine that way, and this is the one place a scope is more than the global form with a smaller reach. `a`, `u` and `L` name an alphabet, and naming one of them clears all three before the named one goes back on, so `(?u:\w)` inside a pattern carrying a global ascii flag reads the wide alphabet rather than carrying both letters at once. Upstream spells that as `_combine_flags` in the compiler and it is copied here rather than reasoned about, because the reasoning is short enough to get right and the copy is shorter.

The other four letters are independent of each other and are a plain on and off, so there is one line of exception and not a table. It lives in the compiler and nowhere else. The parser carries the alphabet letters in its cursor too and never reads them, since which alphabet a class comes out of is settled while the program is built, and writing the same rule in a second place where it could never be seen to be wrong is how a rule comes to have two spellings that drift.

There is one field that is recomputed rather than saved. The builder keeps a `narrow` boolean, which is a reading of whether this is Python's engine under the ascii flag, and it is recomputed from the flags after they are set and after they are put back rather than being saved and restored alongside them. Two fields that can disagree are a worse thing to own than one line that cannot, and this is a field whose whole content is a reading of the other one.

## 6. The one cell where pandas and this library part company

`Series.str.contains("(?u:\w)", flags=re.ASCII)` on a row holding an e-acute is True here and False in pandas, and the reason is a defect in CPython rather than a difference of opinion about what the pattern means.

`re.search` on a pattern that opens with a scoped group widening the alphabet skips positions the pattern matches. `re.fullmatch` on the same pattern and the same row answers correctly, and so does the same pattern with an alternation bar after it, and so does the same pattern with a literal in front of it. That is the shape of the first character optimisation: the set of characters a match could start with is computed from the pattern's own flags rather than from the flags in force inside the group, so the scan skips every position holding a character the outer alphabet does not have.

pandas inherits it on `contains`, `count` and `replace`, which search, and not on `match` and `fullmatch`, which anchor. So the same accessor answers the same question two ways depending on which name asked it, and there is no reading of the pattern under which both are right.

This library answers it the one way. Reproducing the other would mean building the first character optimisation, which this engine does not have, and then putting a defect into it, and an optimisation that a defect can reach is an optimisation that has become part of the semantics. The refusals upstream reproduces on purpose, in the router and elsewhere, are reproduced because they are cheap and because the alternative is a wrong answer where pandas raises. This is a wrong answer either way round, and the choice is between owning the right one and owning an optimiser.

So it is a registered divergence with a test that asserts both sides, which means the day CPython fixes it is a day this library's tests fail rather than a day nobody notices. It is three cells of the accessor sweep and one pattern, and the pattern has to open with the widening group for the defect to bite.

## 7. What the other engine has and what it never had

RE2 reads `(?i:...)`, `(?-i:...)`, `(?m:...)`, `(?s:...)` and `(?U:...)`, and pandas gets all of those out of Arrow without ever looking at them. It refuses `(?x` and `(?a` with `invalid perl operator`, which is the same refusal it gives the global forms, so a scoped letter RE2 never had is an error upstream exactly as much as a global one is and refusing it here is agreement rather than a shortfall.

That means the RE2 compiler needed no new reasoning at all. The letters it has were already read off its builder in one place each, so putting them on a node and restoring them afterwards carries them for free, and the letters it does not have were already refused by the flat field described in section 2 and are still refused by it.

Python refuses one letter in the scoped form for a reason of its own, which is locale. `(?L:x)` is `bad inline flags: cannot use 'L' flag with a str pattern` and every pattern this library carries is made of text, so the letter never reaches a tree. That refusal is in the parser, with Python's own sentence, and it was already there.

## 8. The router had to be taught the node

pandas decides which engine answers a pattern by walking `re._parser`'s tokens looking for a lookaround or a backreference, and it recurses into a subpattern. A scoped group is a subpattern to CPython, so upstream's walk enters one, and `_has_unsupported_regex("(?i:(?=a))")` is True. Measured, not assumed.

A node of this library's own has to be entered on purpose or the same pattern routes to RE2, which has never heard of `(?=`, and raises an Arrow error for a pattern pandas answers as a column of booleans. So `OP_SCOPE` joins the three node kinds the walk already recurses into. The walk stays incomplete in every other way it was already incomplete, because reproducing upstream's blind spots is a decision document 76 made and wrote down.

## 9. What came off the refusal list

`a scoped flag group is not carried yet` is gone from the compiler and from the held-out breakdown, and the compared set grows by 434 patterns, from 7771 to 8205 of 30052. The other 91 of the 525 that reason used to cover turn out to hold a lookaround or a backreference or a named character as well, which the scoped refusal was reported ahead of and was therefore hiding, so they are still held out and they are held out for something true. The parser's refusals for the scoped form are all still there and all still Python's, which are turning off one of the three alphabet letters, writing the locale letter, writing the ascii and unicode letters together, and writing `(?-i)` without a colon.

Two of those are worth naming because they look like they should have changed and did not. The ascii and unicode letters conflict only at the top of a pattern: `(?a:(?u:x))` is a pattern upstream and `(?a)(?u)x` is a `ValueError`, and the check here fires at the end of the parse against the pattern's own flags, which the parser restores after every scope, so it only ever sees the global form. That is right by construction rather than by a case, which is the better of the two ways to be right.

And `(?x:a#c)b` is still `missing ), unterminated subpattern`, because a comment in verbose mode ends at a newline or at the end of the pattern and a closing bracket is neither. Document 88 section 5 met that trap from the anchoring side, where a rewrite glued `)\z` onto a pattern ending in a comment. This is the same rule met from the caller's side, and it is a parse error in CPython too.

## 10. How it was checked

`tests/test_regex_scoped_flags.mojo` is fifteen tests and every row in it was asked of a running CPython 3.13.12 before it was written down, with the RE2 rows asked of a running pandas. The rows that matter are the ones where a plausible wrong implementation gives a different answer: `(?i:a)b` on `aB`, which a flattened field answers True; `(?i:a)b` on `Ab`, which a dropped field answers False; `(?i:(?-i:a)b)` on `aB`, which a clear rather than a restore answers False; and `(?i:a)+b` on `AaB`, which a builder field left set after the last pass answers True.

That sweep is also what found the alphabet rule in section 5. `(?u:\w)` under `re.ASCII` on an e-acute is the single cell of it that separates the right combination from the obvious one, and the obvious one was what I had written. It is in the Mojo tests now as a case with a name of its own, along with the two nested spellings that say which of the two letters wins.

`python/tests/test_str_scoped_flags.py` is the same question asked through the accessor, twice over. Once with no flags, which is the call pandas sends to Arrow, across `contains`, `match`, `fullmatch`, `count` and `replace`. Once with a flag beside the pattern, which moves the call to Python's engine, across four of those five names and both of the letters RE2 never had. `match` is left out of the flagged sweep because upstream keeps that one name's flags on Arrow and raises for every pattern in it, which is a fact about pandas already recorded elsewhere.

All five sweeps of the three regex differentials agree with pandas ten thousand times in ten thousand with no disagreements, over a corpus of 30052 generated patterns and sixteen texts. That is the instrument that says the 434 patterns which came off the refusal list came off it with the right answers rather than merely with answers, and it is also the instrument that cannot see the divergence in section 6, because every cell of it asks pandas without a flag and a call without a flag goes to Arrow.

Three existing tests asserted the refusal and were rewritten into assertions about the answer. That is the honest way to retire a refusal test: the row that used to say `this is turned down` becomes the row that says what it does instead, in the same file, so that the file's history is readable.

## 11. What moved

`firepanda/kernel/regex/tokens.mojo` gained `OP_SCOPE`.

`firepanda/kernel/regex/parse.mojo` saves and restores the cursor's flags around the inner branch in `_flags` and builds the node, and its `scoped` field's docstring now says what the field is for rather than what it was a substitute for.

`firepanda/kernel/regex/program.mojo` saves and restores the builder's flags around the children in `_emit_node`, recomputes `narrow` from them, and lost the refusal.

`firepanda/kernel/regex/route.mojo` recurses into the node.

## 12. What is not here yet

Nothing about a flag. Every one of the seven letters is read in both forms now, but for locale, which neither engine takes on a pattern made of text, and the four letters RE2 refuses are refused here in agreement with it.

The gaps that are left are the ones the flags work was never about. Five constructs route to Python's engine and are refused there: lookaround, backreference, conditional, possessive quantifier and atomic group. `findall` and `extractall` want a list column and a third scan. And the RE2 grammar front end is still the largest single gap in the corpus by a wide margin.
