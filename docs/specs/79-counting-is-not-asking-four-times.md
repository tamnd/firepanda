# 79. Counting is not asking four times

## 1. What this is

Document 78 wired three `str` methods to the regular expression engine and left `count` and `replace` refusing a metacharacter. This is `count`. It is the fourth method and the first one that is not a yes or no question, and the difference turns out to be the whole slice: the engine needed one new thing, and the loop around the engine needed three rules that nobody would guess and that no documentation states.

The short version is that `str.count` does not run the pattern once. It runs it, counts the match, cuts the text somewhere, and runs it again on what is left, and every one of those words hides a decision. Where it cuts is not where the match started. What the pattern sees afterwards is not the rest of a longer text, it is a new text with its own beginning. And the cut moves in bytes even though the pattern matches characters. All three are Arrow's, none of them is RE2's or Python's, and all three change answers on patterns people write.

Everything below was measured against pandas 3.0.5 with pyarrow 25.0.1 on CPython 3.13.12. The pandas source quoted is `ArrowStringArray._str_count` in `pandas/core/arrays/string_arrow.py` at that version, which is five lines and hands the whole question to `pyarrow.compute.count_substring_regex` once it has decided the pattern is RE2's.

## 2. The loop nobody wrote down

`_str_count` routes exactly the way `_str_contains` routes. If there are flags or the pattern holds a lookaround or a backreference it goes to Python's `re`, and otherwise the pattern is preprocessed and handed to `count_substring_regex`. There is no anchoring and no rewrite, so `count` is the plainest of the four to route and the only one where the router is the entire Python side.

What `count_substring_regex` then does is the part that is not written down anywhere. Reading it out of the answers rather than out of the source, it is this.

```
pos = 0
while pos <= n_bytes:
    m = leftmost first match of the pattern in text[pos:], treated as a whole text
    if there is none: break
    count += 1
    new = pos + (where m ended, relative to pos)
    if new == pos: new = pos + 1
    pos = new
```

Three lines of that are surprising and they are sections 3, 4 and 5. The fourth surprising thing is that `pc.replace_substring_regex`, which is the obvious sibling and is what `str.replace` will need, does not share this loop at all. Replacing every `\A` in a five character row marks one position and replacing every `\b` marks the two real boundaries, which is the right answer and not this one. So `replace` cannot ride on any of this and is its own slice with its own measurements, which is why it is still refusing a metacharacter after this one.

## 3. The rest of the row becomes the text

`pandas.Series(["aaa"], dtype="str").str.count("^a")` is 3. Python's `re` says 1 for the same pattern and the same text, and so would anybody asked to guess.

It is 3 because the text handed to the engine after a match is a fresh piece of text rather than an offset into the old one, so the start of the text moves with the scan and `^` is true again at every position the scan stops at. `\A` does the same, `(?m)^` does the same, and `\b` sees the character before the cut as absent rather than as whatever it actually was.

The consequence that is easiest to check is that `count("\\A")` on any row is the number of bytes in the row plus one. Four for `abc`, seven for `héllo`, five for a two character Arabic word written in four bytes. An anchor that means the beginning of the text turns into a counter of positions.

The end does not move, because the end of the remaining text is the end of the row wherever the scan is. So `$` and `\z` are the one pair that stays put, and they still count 2 rather than 1 for a reason that belongs to section 5.

## 4. The cursor moves in bytes

`count("a*")` is 7 on `héllo` and 5 on a two character Arabic word. The first is five characters and six bytes, the second is two characters and four bytes, and in both cases the answer is the byte count plus one.

This is the same rule the literal path already had, which document 66 measured when `str.count("")` turned out to count bytes where Python's `re` counts characters. It was a curiosity there because the empty string is the only literal that can show it. Here it is every pattern that can match nothing, which is a large family: `a*`, `a?`, `a{0}`, `(?:x)?`, and any alternation with an empty arm.

It also means the scan can stop in the middle of a character, which is a state the engine had never been in. A scan that has just counted an empty match at byte 1 of `héllo` resumes at byte 2, which is the second byte of the accented letter. Arrow hands RE2 a byte string and RE2 reads that byte as an invalid UTF-8 byte, matches nothing against it, and carries on. The engine here reads code points rather than bytes, so it had to be taught the same thing, which section 6 covers.

## 5. Where the cursor goes after a match

The first two rules are the ones that get quoted. The third is the one that took longest to find, and two simpler rules that each fit most of the data are both wrong.

The first guess was that the cursor moves past the match, and one byte forward if the match was empty. That predicts `count("\\b")` on `  a  ` is 1 and the answer is 2.

The second guess was that the cursor moves past the match and the engine keeps the whole text as context. That predicts `count("\\Aa")` on `aaa` is 1 and the answer is 3.

The rule that fits is that the cursor moves to where the match ended, unless the match ended exactly where the cursor already was, in which case it moves one byte. Nothing about where the match started comes into it. So a match of no width that the engine found further along the text is counted once where it was found, and then counted again on the next pass because the cursor is now sitting on it. That is why `\b` on `  a  ` is 2 rather than 1 or 3, why `\b` on `ab ba` is 5 where Python's `re` says 4, and why `$` on any row is 2.

Checked against every measured row afterwards, which is about sixty numbers: all nine `\b` rows, all nine `\B` rows, `\A`, `$`, `x*`, `é*`, `l*`, `a{0}` and the alternations. It fits all of them and neither of the first two guesses does.

## 6. What the engine learned

Everything above is a loop and the loop needs one thing from the engine that document 77 did not build: where a match ends.

`Machine.matches` answers whether. `Machine.find` answers where, and it is the harder of the two, because a pattern that can match in several places has to end where the caller's engine says it ends rather than wherever the machine happened to notice first. `a|aa` on `aa` counts 2 and `aa|a` counts 1, which is leftmost first and is what both RE2 and Python do, and a leftmost longest engine would answer 1 for both.

The answer to both halves of leftmost first turned out to be the order the thread list was already in. A thread added earlier is preferred, the start of a fresh attempt is added last, and those two facts come from the depth first walk `_queue` already did and from the order `find` adds the start thread in. So the rules are: once a match has been recorded, stop injecting start threads, since a later attempt cannot beat an earlier one. And a thread that reaches the match instruction records the end and cuts every thread behind it in the list while leaving the ones in front running, since a thread in front is a preferred way of matching the same or an earlier attempt and might still get there. That is nine lines on top of `matches` and no new instruction.

The other engine change is for section 4. A scan that resumes in the middle of a character hands the machine a text whose first position is not a character, and the machine now carries a `lead` count of bytes standing in front of the first code point it was given. Those positions read as `UNREADABLE`, which nothing matches and which is not a word character, so a boundary assertion sees them for what they are. This is the faithful reading of what RE2 sees rather than a convenience: `é*` on `héllo` counts 6, which is one real match and five of no width, and an engine that quietly skipped the half character would also count 6 and would be wrong about which six.

`Machine.counts` is section 2's loop written out, with the byte walk that section 4 needs, and `text_count_regex` in `firepanda/kernel/regex/column.mojo` runs it down a column with one machine per morsel. The cost is one pass over a row per match plus one to find there are no more, where `contains` stops at the first, which means an empty pattern against a long row is the expensive case here exactly as it is in Arrow.

## 7. What was measured

`pixi run differential-regex-count` is new. It generates the same 30052 patterns the other two regular expression differentials generate from the same corpus and the same seed, runs each of them over the same sixteen pieces of text through `str.count` on both sides, and compares the numbers. It compares 7668 patterns, holds out 22384 for reasons it tallies, and disagrees on none of them. That is 122688 counts against pandas with a ceiling of zero.

The held out reasons are the same six as document 78 in the same proportions, and the largest is still the same one: 19615 patterns Python's grammar cannot read, which is the RE2 front end that document 77 section 8 named as the largest gap in the component.

The count differential is a separate program from the match differential rather than a fourth sweep inside it, because the answer has a different shape. Whether is a bit and how many is a number, a count can be larger than the text is long, and all three ways the loop can be wrong produce numbers that look perfectly reasonable.

Beside the differential, `tests/test_regex_count.mojo` holds each of the three rules with the case that pins it and nothing else, and every number in that file was read off pandas rather than worked out from the rules. That order is the point: the rules were written after the numbers, and a test written from the rules would have agreed with each wrong guess in section 5 in turn. `python/tests/test_str_count_regex.py` runs the same twenty patterns document 78 uses through the accessor against pandas, plus the two refusals and the type error.

## 8. Where the literal path stops now

`python/firepanda/_pandas.py` still sends a pattern holding none of the twelve metacharacters to the byte search, which is faster and is the path most patterns a program writes take. What changed is that the pattern which fails that check now goes to the engine instead of being refused.

`replace` is the only one of the five left refusing. It needs the text a match covered rather than only where the match ended, which is a capture slot rather than a position, and section 2 says its loop is a different loop that nobody has measured yet. Those are two separate pieces of work and neither of them is this one.

## 9. What is not here yet

The list from document 78 section 11 with `count` struck off. Captures, case folding, scoped flags, the Python engine and an RE2 grammar front end are all still open, and the front end is still the largest single gap in the component at about nineteen thousand six hundred patterns out of thirty thousand.

`replace` now has a second thing waiting on it, which is its own set of measurements of how `replace_substring_regex` handles an empty match. Document 73 section 10 listed that as deliberately deferred and it is now the thing standing directly in front of a method.

`\B` is refused here as it is everywhere else in the component, because RE2 answers it between the bytes of a character as well as between characters, and that refusal is the reason `\B` appears in the measurements of section 5 and not in the tests.

## 10. Observations to file upstream

`count_substring_regex` loses the text context after each match, so `str.count("^a")` on a row of three letters is three and `str.count("\\A")` counts the bytes of the row. `replace_substring_regex` on the same patterns behaves correctly, with `\A` marking only position 0 and `\b` marking the real boundaries, so the two kernels disagree about what the same pattern means in the same library. Whichever of the two is intended, they should agree.

`count_substring_regex` counts a zero width match found ahead of the cursor twice, once where it was found and once from there. `str.count("$")` is 2 for every row, which is hard to defend under any reading.

`str.count` reads the routing of `_has_unsupported_regex` and not `_is_re_pattern_with_flags`, unlike the three predicates beside it. Nothing measured here turns on the difference, but it is one method out of four reading a different pair of checks.

The rest of the list is in document 76 section 12, document 77 section 9 and document 78 section 12, and is unchanged by this slice.
