# 86. The other half of the scan

## 1. What this is

`str.count` and `str.replace` answered out of Python's engine, which is the loop half of the work document 85 started. Section 12 of that document named it exactly: the counting loop and the replacing loop, both following `re.finditer` rules, which between them give `count` and `replace` under `flags` and `replace` under `case=False` with a real pattern. All of that is here, and three refusals that had nothing to do with flags came off with it. Issue #8 M6.

The kernel half is eighty lines and the two loops are short enough to read in one sitting. The interesting part is above them, where the rule document 85 used to pick an engine turns out not to survive contact with `replace`, and where measuring what upstream actually does with `case=False` on this one name undoes an assumption that had been carried through four documents.

## 2. The routing cannot ride in the bits after all

Document 85 section 3 says the flags ride in a number and the routing rides in that number being nonzero. That was true while `contains` and `fullmatch` were the only names served, and it is false for `replace`.

`replace` reaches Python's engine with no flags at all on two shapes of call. The first is a replacement holding `\g<`, because pandas reads a replacement naming a group by name out of `re` and Arrow's rewrite grammar cannot spell one. The second is an empty pattern with `regex=True`, because pyarrow used not to terminate on one and pandas works around it with `re.sub`. Neither call has a `case` argument or a flag anywhere in it, and both of them have to arrive at the other engine.

So the engine goes back into the word, which is where every other choice on this accessor already lives, and the number is left holding only what the letters mean. Five new words: `contains_regex_python`, `match_regex_python`, `fullmatch_regex_python`, `count_regex_python` and `replace_regex_python`. The three masks did not need the change and get it anyway, because one rule for five names is worth more than a shorter diff.

There is a small piece of luck in the spelling. `_python` and `_folded` are both seven bytes, so the function that strips the suffix off the word takes the same slice either way and the two cases differ only in what they do afterwards.

## 3. The engine is a property of the program

The two loops live under the column walk, four layers below the door, and the walk has to know which loop to run. Threading a boolean down through `text_count_regex`, `text_replace_regex`, `chars_count_regex` and `chars_replace_regex` would have worked and would have been four signatures changed to carry one bit that none of the four does anything with.

Instead the bit rides on the compiled program. `Program` has a `python` field, set by the compiler from the engine it was asked for, and the walk reads it. That is not a convenience, it is where the fact belongs: the two engines walk a row looking for a second match by different rules, and the rules belong to whoever compiled the pattern rather than to whoever is walking.

It also picks the grammar the replacement is read by, which is the one place in this library where a fact about the pattern decides something about an argument that is not the pattern. That is upstream's arrangement rather than one made here. `re.sub` reads its template Python's way and `replace_substring_regex` reads its rewrite RE2's way, and which of the two a call reaches is the same routing decision that picked the engine.

## 4. One rule where Arrow has three

Document 79 measured Arrow's counting loop and found three rules, all of which read wrong and all of which are what pandas answers. The text is cut rather than searched from an offset, so the front of the row moves after every match. The cursor moves in bytes, so an empty pattern counts the bytes of a row rather than its characters. And the cursor moves to where a match ended rather than past it, so a match of no width found ahead of the cursor is counted twice.

Python has one rule for both of its loops, and it is four lines:

```
pos = end
if start == end:
    pos += 1
```

Look from the cursor into the whole row. Take the end of the match. If the match had no width, step one character further on. The text is never cut, so an anchor is judged against the row rather than against what is left of it, and the cursor is in characters rather than in bytes.

The three differences that come out of that are all visible in ordinary answers, and all three were measured rather than reasoned:

`count("^")` on a row of three letters is four without a flag and one with `re.M`. Four, because Arrow makes the rest of the row into a new text after every match and there are four texts including the empty one at the end. One, because Python looks at one row.

`count("")` on a row holding a sharp s is three without a flag and two with. Three, because the sharp s is two bytes. Two, because it is one character.

`count("\\B")` on an empty row is zero on this engine and is refused on the other, because RE2's boundary is asked between bytes and Python's has a special case in CPython that fails on an empty subject, which document 85 section 6 found with a differential.

## 5. Two cursors, and the limit that needs them

The replacing loop needs a second cursor and it is the only subtle thing in it.

`pos` is the end of the last match, and it is where untouched text resumes. `p` is where the next attempt may start. They hold the same number except immediately after a match of no width, when `p` is one character further on. The character between them is not skipped: it is copied across by the next round's copy of the text in front of the next match.

That is also what makes a count come out right. A scan stopped by its limit writes out the rest of the row from `pos`, so the character it was about to step over is still there. `str.replace("a*", "#", n=2, case=False)` on `abc` is `##bc` upstream, and a loop keeping one cursor would have written `##c`.

A single cursor would have given the same answer on every unlimited scan, which is most of them, so this is the kind of thing that is either measured or wrong.

## 6. The replacement has a second grammar

Arrow's rewrite grammar is `\1` through `\9` and `\\`, with everything else after a backslash refused. Python's template grammar is longer and was measured escape by escape against a compiled pattern with two groups, because none of it is written down as a rule anywhere.

Four things may follow the backslash and they are tried in Python's own order.

A `g` opens a name in angle brackets. The name may be a number, so `\g<1>` and `\1` are the same reference, and `\g<0>` is the whole match, which Arrow's grammar cannot spell at all. A name that is not a number is resolved against the pattern's group labels.

A `0` opens an octal escape of up to three digits, masked to a byte. So `\0` is a NUL, `\01` is one, and `\0000` is a NUL followed by the character zero.

Any other digit is a group reference of one or two digits, unless all three of it, the digit after it and the digit after that are octal, in which case it was an octal escape all along and is capped at `0o377`. That is the one rule in this grammar that reads ahead, and it is why `\123` is the letter S and `\12` is a reference to group twelve.

Anything else is a control character if Python has one for it, which is the seven of `abfnrtv` plus the backslash. A letter Python does not know is an error. Anything that is neither a letter nor a digit keeps the backslash as well as the character, so `\-` is two characters and not one. That last one reads like a bug and is what `re` does.

## 7. `case=False` on this one name is not the fold it is everywhere else

This is the part that changed while it was being written.

The four names above `replace` answer `case=False` out of a fold that maps one character to one character, which document 69 measured and which is why `STRASSE` does not hold `straße`. The plan for this slice was to leave `replace` on that path and only move the calls carrying flags.

Reading `ObjectStringArrayMixin._str_replace` says otherwise. It begins `if case is False: flags |= re.IGNORECASE`, and then, if there are any flags at all, it escapes the pattern when `regex` is False and compiles it. So `case=False` is not a second kernel upstream. It is the flag, spelled differently, and both settings of `regex` end up in the same engine with the same fold.

Which fold that is matters. The engine's fold takes every code point that folds onto the one written down, so a pattern of `k` matches a Kelvin sign and a pattern of `i` matches a dotted capital I. The one to one table already carries the Kelvin sign and the long s, so the two folds agree about more than they look like they would, and the four Turkish I code points are the whole of the difference. They show it in both directions and both directions are asserted: `replace("i", "#", case=False)` swaps a dotted capital I and `contains("i", case=False)` does not find one.

So `case=False` on `replace` now goes to Python's engine for both settings of `regex`, which is simpler than the alternative as well as being what upstream does. One consequence is that `replace_folded` is no longer reachable from the pandas layer. The word stays in the kernel door for the Mojo API, where a caller asking for a one to one fold can still have one.

The cost is real and is named here rather than discovered later. A `case=False` literal replacement used to be a byte search and is now a compiled pattern and a scan. It is correct and it is slower, and a folded literal fast path could be restored for the calls where the two folds provably agree.

## 8. Three refusals that were never about flags

`replace` carried three refusals that all said the same thing in different words, which was that Python's engine was not written. All three are gone.

A replacement holding `\g<` was refused. It is now routed, and the test that asserted the refusal now asserts something more interesting, because upstream's check for it does not look at `regex` and reads as though it moves a literal replacement too. It does not: the call it moves lands in a branch that asks `regex or flags or callable(repl)` before it reads the replacement as a template, and a literal call answers no to all three, so it comes back out as a plain `str.replace`. Measured: `replace("X", r"\g<0>", regex=False)` on `aXb` gives `a\g<0>b` and the same call with `regex=True` gives `aXb`. The routing test here is narrowed to the calls it actually changes.

An empty pattern with a backslash in the replacement was refused, for the same reason one step removed: pandas answers an empty pattern out of `re.sub`, so the replacement is read by Python's grammar there and the two grammars only agree while there is no backslash to disagree about. Now that both grammars are written, the call is routed instead.

A count beside a flag was refused, and half of that refusal was a workaround rather than a rule. `n=0` means no replacements to the Arrow path and all of them to `re.sub`, which reads a count of zero as unlimited and has done since long before pandas existed. The old code widened the zero when a fold was on, with a comment explaining why. Now the widening is where it belongs, in the one branch that hands a count to `re.sub`, and the kernel's number still means what it says.

What is not gone is the Arrow refusal document 80 put in. A count with a real pattern and no flag and no `case` still lands on Arrow's bounded loop, which replaces nothing after the first match and raises on a pattern of no width, and is still refused rather than copied.

## 9. The exception class nobody can match

A bad replacement template is an error on both sides and the classes do not line up.

`re.PatternError.__mro__` is `(PatternError, Exception, BaseException, object)`. It is not a `ValueError`, so a caller catching `ValueError` around `str.replace` catches a bad RE2 rewrite upstream and does not catch a bad Python template. An unknown group name is worse: it comes out of `re` as an `IndexError`.

This library raises `InvalidArgumentError`, which is a `ValueError`, for a bad replacement on either engine. That is one class where upstream has three, and it is a divergence rather than a bug. Matching it would mean inventing a class that is an `Exception` and not a `ValueError` for one of the two grammars and an `IndexError` for one branch of it, which is a worse surface than the one refusal message everything else on this accessor gives.

## 10. How it was checked

Twenty two patterns, seven flag combinations, four settings of `n` and twenty three rows, which is 154 columns of counts and 616 columns of replaced text compared row for row against live pandas 3.0.5. Nothing is held out. A missing row is missing on both sides for both names, which is the one place `count` agrees with pandas where the three masks do not.

The rows carry the four engine differences document 85 found plus the ones a loop needs, which are a row of spaces around a letter where an empty match and a boundary both land several times, a row of three repeated letters where a greedy star and a cut row part company, and an empty row where a scan that steps before it looks runs off the end.

The Mojo suite has ten new tests on the two loops and the template grammar, with each of the three counting differences asserted as a pair so that the test says which loop ran rather than only what it answered. The Python suite has fifteen new tests on the routing, the two folds, the count of zero, and the six refusals the template grammar keeps.

Seven tests elsewhere in the Python suite asserted a refusal that is now an answer. They assert the answer instead, and four of them got a second assertion showing the pair the move creates, which is worth more than the refusal was. That is the third slice in a row to rewrite tests in that direction and it goes on being the honest shape of the work.

## 11. What moved

`str.count` and `str.replace` now cover the `flags` parameter, and `str.replace` covers `case`, a named group in the replacement, an empty pattern, and a count beside any of those. That is every parameter `count` has and all but `repl` being callable on `replace`.

Both come off the L2 ceiling document 84 named. The board's cases for the `flags` parameter sit on `replace` and `extract`, so this is the first slice in the three where the conformance board can move at all.

## 12. What is not here yet

`extract` still refuses a flag. It is on Python's engine already and the reason is now a different one: it hands back a frame of groups rather than a column, so it crosses by a door that takes no flags, and the flags would have to travel beside the group count. That is wiring rather than a scan.

Verbose mode in the parser and the ascii flag are unchanged from document 85 section 12 and are the only two of the seven letters still refused for a reason inside this library.

`findall` and `extractall` are the two names on this accessor with no engine path at all. `findall` wants a list column, which does not exist, and a third scan. `extractall` wants a `MultiIndex` whose names are `[None, 'match']` and drops the rows that did not match.

A scoped flag group is still refused, which is 525 held-out patterns in the differential. The five constructs the router sends to Python are still refused: lookaround, backreference, conditional, atomic group and possessive quantifier. The RE2 grammar front end is still the largest single gap in the component at 19620 of 30052 corpus patterns.

A folded literal fast path for `replace` is the one performance item this slice created, and section 7 says what it would have to prove before it could be taken.

The replace differential sets aside 186 sweeps because pandas answered them out of the other engine, and it set them aside because this library had no other engine to answer them with. It has one now, so those 186 are comparable rather than held out, and the harness has not been told. That is the cheapest reach this component has left and it is a change to `tests/differential/` rather than to anything under `firepanda/`.
