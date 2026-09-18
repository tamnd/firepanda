# 88. A grammar and an alphabet

## 1. What this is

`re.VERBOSE` and `re.ASCII`, the last two of the seven letters any name on the `str` accessor still turned down. Document 85 section 12 called them gaps and document 87 section 9 said they were the only two refused for a reason inside this library rather than upstream, and they are taken together here because that is the one thing they have in common. Issue #8 M6.

They have nothing else in common and the document is really two documents. Verbose mode changes what the characters of a pattern mean and is spent entirely in the parser, in one place, in nine lines. The ascii flag changes what six classes and a fold and a word boundary cover and is spent entirely in the compiler, in seven places, none of them new code so much as a second Bool arriving where one already was. Neither of them is visible to the machine that runs the program, which is now true of all five letters this library answers.

## 2. Verbose mode is a rule about where, not about what

The obvious reading of verbose mode is that whitespace and comments are thrown away, and the obvious implementation is a pass over the pattern text before it is parsed. That implementation is wrong, and the way it is wrong is not a corner case. `[a b]` under verbose mode still matches a space. So the rule is not about which characters are discarded, it is about where in the parse the discarding happens, and everything surprising about verbose mode follows from that one fact.

CPython does the skip at the top of the main item loop in `_parse` and nowhere the item loop calls into. The class loop is a separate read, so `[a b]` keeps its space. The counted scan that reads `{1,2}` is a separate read, so `a{1, 2}` is not a repeat. The peek that looks for a lazy marker after a quantifier is a separate read, so the `?` in `a * ?` is a second quantifier rather than a modifier on the first. Three behaviours a reader would call inconsistent, all of them the same line of code.

This library puts the skip in the same place, at the top of the `while True:` loop in `_seq` in `firepanda/kernel/regex/parse.mojo`, before the character that decides what the next item is has been looked at. The three behaviours come out of that for free, and the tests in `tests/test_regex_verbose_ascii.mojo` pin all three, because the whole risk in this slice is that somebody later moves the skip somewhere more convenient.

## 3. The four rows that separate the rule from a guess

Each of these was asked of a running CPython 3.13.12 before it was written down, on the procedure document 87 adopted.

The whitespace set is exactly the six characters `re` spells `" \t\n\r\v\f"`, which is `0x20`, `0x09`, `0x0A`, `0x0D`, `0x0B` and `0x0C`. It is not Unicode whitespace and it is not `str.isspace`, and the character that shows the difference is the file separator at `0x1c`, which `str.isspace` calls whitespace and verbose mode keeps. A pattern holding one still means something. So the helper that answers this question writes the six out by hand rather than asking any general whitespace test, and its docstring says why.

A backslash in front of a space is a literal space. That is the only way to write one outside a class, and it is a second reason the skip cannot be a pass over the pattern before it is parsed, because at that point nothing knows which backslashes are escapes.

A `#` outside a class runs to the end of the line and consumes the newline. `\#` and `[a#b]` put the character back, the same two ways a space is put back. A comment that reaches the end of the pattern without a newline is fine.

`a{1, 2}` is the one worth reading twice. It does not match `aa`, which is the part everybody expects once they know the counted scan does not skip. What it does match is the literal text `a{1,2}` written without the space, because the `{` degrades to a literal, the run after it is read as literals, and the spaces inside that literal run are then skipped by the item loop after all. The skip is not disabled for the failed construct, it just arrives later.

## 4. Reading the letter out of the pattern

`(?x)` can turn the flag on partway through a parse, and CPython handles that by raising an internal `Verbose` exception and restarting the whole parse from the beginning. This library does not restart. It reads `c.flagged & FLAG_VERBOSE` off the cursor at the top of each pass of the item loop rather than once before the loop, so the flag simply becomes true from the next item onward.

That reaches the same answer, and the reason is a rule the parser already enforced. A global flag group has to come before anything that produced a node, so the only thing that can sit behind a `(?x)` is another global flag group or a comment, and neither of those is affected by whether verbose mode was on while it was read. `(?x) (?i)a` matches `A`, `(?x)#c\n(?i)a` matches `A`, and ` (?x)a b` and `|(?x)a b` are both the `global flags not at the start` error. All four were measured upstream and all four are asserted here.

## 5. The rewrite that stopped being safe

`fullmatch` on Python's engine is answered by handing the engine `\A(pattern)\z`, which is a rewrite of the pattern text. Document 85 section 5 wrote that rewrite and gave the reason for the two anchors it uses. It has been correct for every pattern this library could carry, and verbose mode is the first flag that breaks it.

A verbose pattern is allowed to end in the middle of a comment, because a comment ends at a newline or at the end of the pattern and both are endings. `a # c` is a good pattern. Glue `)\z` onto it and the bracket is inside the comment, where the grammar never sees it, so the group is never closed and a pattern the caller wrote correctly comes back as one the grammar cannot read.

The fix is a newline in front of the closing bracket, put there only under verbose mode. A newline is the only thing that ends a comment and verbose mode throws a newline away, so it changes the answer nowhere else. The flag is read off the parsed tree rather than off the argument, because `(?x)` written into the pattern turns it on just as much as the letter passed beside it does.

Upstream has no such trouble and the reason is worth keeping, because it is the same reason twice. `re.fullmatch` anchors from outside the pattern and never writes a bracket at all, which is why document 85 needed a second rewrite for this engine in the first place: a glued anchor sits inside the pattern where a flag can reach it, and an anchor `re` applies sits outside where none can. Verbose mode reaches further inside than the flags before it did. It reaches the brackets.

## 6. Three alphabets where there were two

The compiler threads one Bool through the three places an alphabet matters, `_category_ranges`, `_folded` and `_at_value`, and that Bool was called `python` and meant the wide alphabet. The tempting move for `re.ASCII` is to pass `python=False` and reuse RE2's tables, and it is wrong by exactly one character. Python's ASCII `\s` is `[0x09-0x0D, 0x20]` and holds the vertical tab. RE2's `\s` is `[0x09, 0x0A, 0x0C, 0x0D, 0x20]` and never did. Two of the three narrow classes are the same as RE2's and the third is not.

So the parameter became a pair. `python` still says which engine, `narrow` says whether the ascii flag is set, and it is only ever true on Python's engine because RE2 refuses the letter before anything compiles. `wide` is `python and not narrow`, and the spaces branch writes all three sets out rather than deriving one from another, because two of them are four ranges of literal numbers and the third is a generated table.

`narrow` is stored once on `_Builder` as `python and (flags & FLAG_ASCII) != 0` rather than being read out of the flags at each of the three places, for the same reason `python` is.

The discriminating row for this whole letter in a pandas frame is one character wide. `str.contains(r"\s", flags=re.ASCII)` on a row holding `a\x0bb` is True and `str.contains(r"\s")` on the same row is False, because the second one is RE2. Every other cell either agrees or is about `\w` and `\d`, which narrow the way anybody would guess.

## 7. The fold narrows and the boundary follows it

Under `re.ASCII` the ignorecase fold is only A-Z against a-z. The Kelvin sign stops folding onto `k`, the long s stops folding onto `s`, the two Turkish letters stop folding onto `i`, and an e-acute stops folding onto its own capital. That makes `(?ia)` a third answer rather than either of the two this library already had, since RE2's fold under ignorecase is wider than ASCII in its own way.

`_fold_one` gained a `narrow` parameter and an early return that emits the one partner a letter has, and `_folded` still walks its input range by range so that a class keeps being folded before it is negated. Document 83 section 6 has why that order matters, and the row that checks it survived the narrowing is `(?i)[^k]` against the Kelvin sign, which matches under the letter and does not without it.

`\b` and `\B` ask about a word character, so the letter that says which characters those are moves them too. `\b` moves onto the branch RE2 already uses, because the ASCII word class is one set and both engines have it.

`\B` does not, and this is the one thing in the slice that went out wrong and came back. Python fails a `\B` on an empty row and RE2 matches one, which document 85 found and which is a special case written into CPython about the row rather than a consequence of any rule about word characters. It survives the narrowing: `re.search(r"\B", "", re.ASCII)` is None. Sending `\B` to RE2's `AT_NON_BOUNDARY` under the letter took RE2's answer for the empty row along with RE2's alphabet, so the ASCII column of the flags sweep disagreed with pandas on the empty row and on nothing else.

`AT_NON_BOUNDARY_ASCII` is the fix, and it is a third value rather than a reuse because the two halves of the word boundary are not the same shape. `\b` is a question and nothing else, so RE2's value is the whole of the narrow reading of it. `\B` is a question with a special case attached, and only the question narrows.

## 8. What came off the refusal list

`_refused_flags_python` lost both of its remaining branches and is now only the locale one, which is upstream's own `ValueError` rather than a shortfall here. Document 87 section 5 listed five letters answered and two refused, and it is now six and one, with the one being a letter Python will not take on a pattern made of text either.

Scoped flags are deliberately unchanged. Removing the two branches means `(?x:a b)` now falls through to `a scoped flag group is not carried yet` with `gap=True`, which is the right refusal and the same one every other letter gets in a scoped group. What is missing there is a place to hang a flag on a node, which document 77 section 8 named and which is one slice of its own.

Both alphabets at once is still refused, and the wording is Python's. `(?a)(?u)a` and `(?u)a` with `re.ASCII` beside it both come back as `ASCII and UNICODE flags are incompatible`. Upstream raises a bare `ValueError` for this and pandas does not catch it, which document 76 section 8 wrote down and which is unchanged.

## 9. How it was checked

`tests/test_regex_verbose_ascii.mojo` is fifteen tests and is the document in assertions. Every non-ASCII character in it is written as a `\uXXXX` escape, so the file has no non-ASCII source bytes at all, which matters because two of the rows are about characters a reader cannot tell apart by eye.

`tests/test_regex_python.mojo` had a test asserting that verbose mode was not read yet, which is now a test that the three letters RE2 never had all mean something here. On the Python side the three sweeps grew: the flags sweep is 27 patterns by 12 flag values where it was 24 by 8, the extract sweep is 8 by 9 where it was 7 by 6, and the count and replace sweep is 24 by 11 where it was 22 by 7. Each sweep compares every cell against a running pandas, and the flags sweep gained a row as well: a letter, a vertical tab and a letter, which is the only row in that list that discriminates a whole letter on its own, because Python's ASCII `\s` holds a vertical tab and RE2's does not.

The sweep is what found both of the defects written up here. The `\B` one in section 7 turned up on the empty row, which is the one cell of 648 that could have found it. The closing bracket one in section 5 turned up on `a # c`, which is the one pattern of 27 that ends in a comment and which was put into the list because a comment reaching the end of a pattern is a rule worth checking, not because anybody suspected the anchoring. Neither defect is the kind a reader finds by reading, and neither would have been caught by any test written from the design.

`tests/test_regex_method.mojo` gained a test for the rewrite itself, which asserts the text that comes out rather than only the answer that comes back, because the text is the part that is easy to change by accident later.

The three regex differentials were run because this slice changes the parser, which means it could move an answer for a pattern holding no flag at all. All nine sweeps are 10000 in ten thousand with 0 disagreements. Two lines left the held-out breakdown and no line joined it, which is what a slice that only adds answers is supposed to look like.

The held-out count for `str.contains` is 22281 of 30052, and `verbose mode is not read yet` and `the ascii flag is not read yet` are no longer two of the reasons in it.

## 10. What moved

`firepanda/kernel/regex/parse.mojo` gained `_is_verbose_space` and nine lines at the top of `_seq`'s item loop. `firepanda/kernel/regex/tokens.mojo` gained `AT_NON_BOUNDARY_ASCII` and `firepanda/kernel/regex/pike.mojo` gained four lines to run it. `firepanda/kernel/regex/program.mojo` threads `narrow` through `_category_ranges`, `_fold_one`, `_folded`, `_at_value` and `_Builder`, and `_refused_flags_python` is down to one branch. `firepanda/kernel/regex/method.mojo` gained a `verbose` argument on `python_anchored`, fed from the parsed tree by `program_for`, which is the newline in section 5. Nothing in `firepanda/py/text.mojo` changed and nothing in `python/firepanda/_pandas.py` changed, because `_DOOR_FLAGS` already carried both letters and had been mapping them onto bits the kernel then refused.

## 11. What is not here yet

Scoped flags, which is now the only thing about a flag this library turns down for a reason of its own. 525 patterns in the held-out set are waiting on it.

The five constructs the router sends to Python's engine and Python's engine refuses: lookaround at 621 patterns, backreference at 689, conditional at 86, possessive quantifier at 22 and atomic group at 7.

`findall` and `extractall`, which want a list column and a `MultiIndex` respectively and are the two pattern names left on the accessor.

The RE2 grammar front end, still the largest single gap in the component at 19615 of the 30052 corpus patterns.
