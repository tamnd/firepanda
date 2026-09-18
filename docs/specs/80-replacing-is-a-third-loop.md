# 80. Replacing is a third loop

## 1. What this is

Document 79 wired `count` to the regular expression engine and left `replace` as the one `str` name of the five still refusing a metacharacter. This is `replace`. It is the fifth method and the first one whose answer is text rather than a bit or a number, and that single difference is what made it the last of the five rather than the second.

Two things fall out of the answer being text. The first is that a replacement can name a group, so the engine has to say where each group matched rather than only whether the pattern matched, which is a different program and a different machine. The second is that Arrow replaces down a row with a loop that is not the loop it counts with, in the same library, on the same pattern, and the three ways the two differ are none of them written down anywhere.

Everything below was measured against pandas 3.0.5 with pyarrow 25.0.1 on CPython 3.13.12. The pandas source quoted is `ArrowStringArray._str_replace` in `pandas/core/arrays/string_arrow.py` at that version.

## 2. Which calls reach the engine

`_str_replace` routes to Python's `re.sub` if the pattern is a compiled pattern, if the replacement is callable, if `case` is False, if there are flags, if the replacement holds `\g<`, or if the pattern holds a lookaround or a backreference. Otherwise the pattern is preprocessed, which rewrites a trailing `\Z` to `\z`, and handed to `pyarrow.compute.replace_substring_regex` with `max_replacements` set to None for a negative count and to the count otherwise.

That list has one entry the other four do not have, which is the replacement. `\g<` is Python's way of naming a group and RE2 has no spelling for it, so a call is sent to a different engine on the strength of a string the pattern knows nothing about.

The consequence worth stating plainly is that `regex=True` has to reach the engine even when the pattern holds no metacharacter at all. `pandas.Series(["abc"], dtype="str").str.replace("a", "\\\\", regex=True)` is one backslash and a `bc`, and the same call with `regex=False` is two backslashes and a `bc`. The pattern is the letter `a` in both. What changed is which grammar read the replacement, so a router that sent a metacharacter free pattern to the byte search would answer the first of those wrongly. The other four names can decide on the pattern alone and this one cannot.

## 3. The scan is not the scan that counts

Document 79 section 2 wrote out the loop `count_substring_regex` runs and noted that `replace_substring_regex` does not share it. Here is what it does instead, read out of the answers the same way.

```
p = 0
lastend = -1
while p <= n_chars:
    m = leftmost first match at or after p, with the whole text as context
    if there is none: break
    copy text[p:m.start]
    if m.start == lastend and m.start == m.end:
        copy one character
        p += 1
        continue
    write the replacement
    p = m.end
    lastend = p
copy text[p:]
```

Three things differ from the count loop and all three change answers.

It does not cut the row. RE2 is given an offset rather than a fresh text, so an anchor keeps seeing the row it started with. `str.replace("^a", "#")` on a row of four letters replaces once, where `str.count("^a")` on the same row is four. `str.replace("\\A", "#")` puts one marker at the front of the row, where `str.count("\\A")` is the number of bytes in the row plus one. `(?m)^` marks the start of each line and `^` marks only the start of the row, which is the same pair of answers Python would give and the opposite of what counting gives.

It moves in characters. `str.replace("x*", "#")` on a five character word written in six bytes puts six markers in, where `str.count("x*")` on the same row is seven. The count moves a byte at a time and can stop in the middle of a character, and this one cannot.

It refuses an empty match that lands exactly where the last match ended, and because it is writing text rather than counting it has to do something instead, which is to copy one character across. `str.replace("a*", "#", regex=True)` on `abc` is `#b#c#`. Python's `re.sub` of the same pattern on the same text is `##b#c#`, because the empty match just after the `a` is a match like any other there. The count loop has the same refusal and simply steps, which is why the same pattern counts one more than it replaces.

The subtlety in the loop above is that the cursor does not advance on the pass that keeps an empty match. It advances on the next pass, through the skip branch, because the match it finds then is the same empty match and `lastend` now equals it. A port that tried to advance on both passes drops a character.

## 4. The replacement has a grammar

Arrow reads the replacement with RE2's rewrite grammar, which is small enough to state completely. `\0` is the whole match. `\1` through `\9` are the groups, numbered by the order their brackets opened. A pair of backslashes is one backslash. Anything else after a backslash is an error, and so is a backslash at the end. A group that did not take part in the match contributes nothing rather than raising, so `(a)(b)?` against `aaaa` with the replacement `<\1|\2>` writes four copies of `<a|>`.

There is no multiple digit group. `\10` is group one followed by the character zero, which is worth knowing because Python's `re` reads the same three characters as group ten.

The three errors and their exact Arrow messages are these. A trailing backslash gives `Invalid replacement string: Rewrite schema error: '\' not allowed at end.` A backslash followed by anything that is not a digit or a backslash gives `Invalid replacement string: Rewrite schema error: '\' must be followed by a digit or '\'.` A group number larger than the pattern has gives `Invalid replacement string: Rewrite schema requests N matches, but the regexp only has M parenthesized subexpressions.` All three arrive in Python as a `ValueError`, so the firepanda refusals are `ValueError` too.

## 5. What the engine learned

Everything in section 4 needs the text a group covered, and the engine document 77 built answers where a match ended and nothing else. So the program and the machine both grew a second mode.

`compile_program` takes a `captures` flag. When it is set, the builder puts a save instruction on either side of every group and one around the whole match, and `Program.slots` says how many slots the result wants. When it is not set, nothing is emitted and `slots` is zero, which is the mode the other four methods still compile in. This matters because the slots are the expensive part: every thread in flight carries a copy of them, so a program built with captures is the slower of the two and only the callers that need the text of a match ask for one.

The machine carries the slots alongside the thread list, and `_queue` became recursive to do it. A save instruction has to write a slot, walk on, and then put the slot back the way it was, because the walk it kicked off is one way of matching and the thread sitting behind it is another. A stack of pending positions cannot express that without a copy of the slots per entry, and a recursive walk expresses it with two lines and a local variable. The dedup stamp keeps the thread that was added first, which is the higher priority one under leftmost first, so the slots that survive are the right ones.

`find` and the new `search` are the same loop with two arguments. `search` starts at a position rather than at zero and fills a caller's list with the slots of the match it found, and `find` starts at zero and passes a list nobody reads. The scan in section 3 needs both halves of that: a first position, because it walks the row, and the slots, because the replacement asks for them.

`byte_width` in `pike.mojo` stopped being private, because the replace scan works in characters and the row is bytes, and the table that turns one into the other was already there for the count scan.

## 6. Where the loop lives

`firepanda/kernel/regex/replace.mojo` holds the `Rewrite` struct, which is a parsed replacement, `parse_rewrite`, which reads one or says why not, and `replaced`, which is section 3's loop written out. A `Rewrite` is a list of literal runs each followed by a group number, with a final run whose group is minus one, so writing a replacement out is a walk of two short lists and no branching on characters.

`text_replace_regex` in `firepanda/kernel/regex/column.mojo` runs it down a column. It is the first kernel there that is not split into morsels, because a `StringBuilder` is one buffer with one cursor and handing four threads a share of it is a different design rather than a flag. The literal `text_replace` in `pattern.mojo` is serial for exactly the same reason, so this is the established shape here rather than a new compromise. It also means there is no scalar twin, since the twin next door exists to check the morsel split and the null repair and there is neither of those here.

That last paragraph stopped being true after tamnd/firepanda#830 measured what it cost, and the change is the one section 10 asked for: a payload per morsel written by the thread that owns the morsel, and one pass at the end that puts the payloads end to end and moves the long views onto them. `stack_payloads` in `firepanda/array/strings.mojo` is that pass. There is still no scalar twin, and what stands in for one is a test that runs the same rows short and then tiled past a morsel and asks whether row `i` still says what it said. The literal `text_replace` in `pattern.mojo` and the folded `text_replace_folded` beside it took the same route straight afterwards, since the join was the whole of what either of them was waiting for. `text_extract_regex` in this file went last and runs the join once per capturing group. It is the only one of the four that builds a validity bitmap rather than copying the input's, because a row of its answer is missing when the input was null, when the row matched nothing, and when the group took no part in the match it was in. A morsel is a whole number of bytes of that bitmap, so a thread setting a bit for one of its own rows is never writing a byte another thread has a row in.

Everything the row costs is still paid once per column. The pattern is compiled once, the replacement is read once, and the machine, the offsets, the slots and the output buffer are made once and handed to every row.

## 7. The two refusals

A count and a pattern together are refused. Arrow answers that call out of a different loop again, which finds a match with `RE2::FindAndConsume` and then asks `RE2::Replace` to replace inside the text it found, with no rule about a match of no width and no cursor to move. `str.replace("a*", "#", n=5)` on a row of five characters puts five markers at the front of a row it then leaves untouched, and `str.replace("\\b", "#", n=1)` raises `ArrowInvalid: Regex found, but replacement failed` on every row. Copying that would be copying a bug, so the call is a gap on the board instead.

The refusal is narrowed to the calls that would have reached the engine, which is a pattern holding a metacharacter or a replacement holding a backslash. A count with an ordinary pattern and an ordinary replacement is correct upstream and still takes the literal path, so nothing that worked before this slice stopped working.

An empty pattern with a backslash in the replacement is refused. pandas has a guard that sends an empty pattern to Python's own `re.sub` elementwise, because `pyarrow.compute.replace_substring` does not terminate on one, and the two grammars only agree while there is no backslash to disagree about. With one, `\0` is a null character on that path rather than the whole match, `\n` is a newline rather than an error, and `\1` is a `re.PatternError` rather than a rewrite schema error. Document 67 already recorded the guard and this is the second thing that falls out of it.

## 8. What was measured

`pixi run differential-regex-replace` is new. It generates the same 30052 patterns the other three regular expression differentials generate from the same corpus and the same seed, and runs each of them over the same sixteen pieces of text with three replacements: a plain marker, a marker holding the whole match, and a bare group reference. It compares 7668 patterns, holds out 22384 for reasons it tallies, sets aside 186 sweeps where pandas answered out of Python's engine rather than Arrow's, and disagrees on none of them.

The three replacements are three different questions. The plain marker is the scan on its own with the grammar kept out of the way. The marker holding the whole match says how much of the row each match covered, which a scan that is right about the ends and wrong about the starts would fail. The bare group reference is the capture slots, and it is also the one that is refused when the pattern has no group, so the refusals are compared as well as the rows.

The rows cross the boundary as hexadecimal. A row can hold a newline, a space or a tab, and the answers have to arrive as one string, so both sides write two characters a byte and the comparison is a comparison of digits.

The held out reasons are the same six as documents 78 and 79 in the same proportions, and the largest is still 19615 patterns Python's grammar cannot read, which is the RE2 front end that document 77 section 8 named as the largest gap in the component.

Beside the differential, `tests/test_regex_replace.mojo` holds each rule with the case that pins it and nothing else, and every expected string in that file was read off pandas before the loop was written rather than worked out from the rules afterwards. `python/tests/test_str_replace_regex.py` runs twenty three patterns through the accessor against pandas, the six replacements that exercise the grammar, and the four refusals.

## 9. Where the literal path stops now

`python/firepanda/_pandas.py` still sends `regex=False` to the byte search, which is the default for this name in pandas 3 and is the path most calls take. What changed is that `regex=True` now goes to the engine always, rather than being refused when the pattern held a metacharacter and taking the byte search when it did not, and section 2 says why the middle option was never correct.

All five of the `str` names that take a pattern now have an engine. What is left in the accessor is `case=False`, which is a folding table rather than an engine, and this method's count.

## 10. What is not here yet

The list from document 79 section 9 with `replace` struck off and captures struck off, since a capture slot was the thing `findall`, `extract` and `extractall` were waiting on as much as this method was. Those three now need a list or a frame to put their answers in rather than anything from the engine.

`split` and `rsplit` in their regular expression form want a list column, which is the same gap. Case folding, scoped flags carried on the node, the Python engine with its own class tables, and an RE2 grammar front end are all still open, and the front end is still the largest single gap in the component.

A bounded replace is open and may stay open. Section 7 says what Arrow does with one and there is nothing there worth copying, so the honest answer is a gap on the board until somebody decides which of the two loops a caller actually wants.

A parallel replace is open. Section 6 says what stands in the way, which is one builder with one cursor, and the shape that closes it is a builder per morsel and a join at the end. The same change would make the literal `text_replace` parallel, so it is one piece of work for both.

That was written before there was a number on it. The number arrived from ClickBench q28, where this kernel used 0.87 cores on a machine where the kernel next door used 3.40, and the shape above is what landed. Section 6 has the second half. The literal `text_replace` and the folded one beside it followed immediately, and `text_extract_regex` after them with the join run once per capturing group, so nothing of that paragraph is open any more.

## 11. Observations to file upstream

`count_substring_regex` and `replace_substring_regex` disagree about what the same pattern means in the same library. The first loses the text context after each match and the second does not, so `\A` counts once per byte and replaces once per row. The first moves in bytes and the second moves in characters, so a pattern that can match nothing counts one more than it replaces on any row holding a character wider than a byte. Whichever reading is intended, the two kernels should agree.

`str.replace` with any count of one or more and a pattern that can match nothing is wrong in two different ways depending on the pattern. `str.replace("a*", "#", n=5)` inserts five markers at position zero and leaves the row alone. `str.replace("\\b", "#", n=1)` raises `ArrowInvalid: Regex found, but replacement failed`. Both come from the bounded loop having neither the empty match rule nor the cursor that the unbounded one has.

`str.replace(pat, repl, n=0)` still replaces once on an empty row for a pattern that can match nothing, and `str.replace("", repl, n=0)` is unlimited rather than none, because the empty pattern guard does not pass the count on.

`str.replace("\\B", "#")` on a row holding a character wider than a byte raises `ArrowException: Unknown error: Wrapping ... failed`, because RE2 reads a non boundary between the bytes of a character and the replacement is written between them, which produces something that is not UTF-8. The differential in section 8 carries a marker for the same shape arriving as a column pyarrow did not catch, which is the same bug one layer further along.

The rest of the list is in document 76 section 12, document 77 section 9, document 78 section 12 and document 79 section 10, and is unchanged by this slice.
