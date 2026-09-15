# 84. The same fold asked for from outside

## 1. What this is

`case=False` on a pattern that is a real pattern, and the `flags` argument that sits beside it. Document 83 spent `(?i)` while the pattern compiled and left the argument spelling of the same request refused, with section 10 of that document saying the argument was its own slice. This is that slice. Issue #8 M6.

It is a small piece of implementation sitting on top of a large piece of measurement. The implementation is two default arguments, one seeded field and a word ending in `_folded`. The measurement is what those two arguments mean upstream, which turned out to be stranger than either the documentation or the reasoning suggested, and which split what was planned as one slice into two.

## 2. What upstream does with the two arguments

pandas turns `flags` into a compiled pattern and then forgets it. `Series.str.match` is the clearest of the four to read and the rule is the same in all of them:

```python
if flags is not lib.no_default:
    flags = flags | re.U
    pat = re.compile(pat, flags=flags)
    flags = 0
```

After that the accessor asks whether the pattern it now holds should go to Arrow or to Python's `re`, and the test it uses is about the compiled object rather than about the argument that built it. A compiled pattern carrying nothing but `IGNORECASE` and `UNICODE` is not a pattern with flags to that test, because `UNICODE` is on for every string pattern in Python 3 and `IGNORECASE` is the one flag Arrow's engine has of its own. So a `flags=re.I` on `match` reaches Arrow.

`case` is turned into the same thing from the other end. When the caller passed nothing, `case` is read back out of the compiled pattern, and when the caller passed something, it is compared to what was read back out and a disagreement raises. Either way what reaches the array is one boolean, and the array hands it to Arrow as `ignore_case`.

So the two arguments are one argument by the time anything acts on them, and `(?i)` written inside the pattern is a third spelling of it. All three have to answer alike and all three do, here and upstream.

## 3. The measurement that split the slice in two

The plan was one slice covering both arguments on all six methods. The measurement was taken against pandas 3.0.5 before any of it was written, and it says something the plan did not expect: the four methods do not agree about which engine a flag sends them to.

`contains`, `fullmatch` and `count` go to Python's `re` for any flag at all, ignore case included. Their Arrow paths raise `NotImplementedError` the moment `flags` is nonzero and the accessor catches that and falls back. `match` alone does the compile quoted above before the routing test runs, so its flag has already stopped being a flag and it stays on Arrow. `replace` goes to Python for `case=False` as well as for any flag, and `extract` is on Python whatever anybody passes.

That matters because Python's `re` does not scan the way Arrow scans. Document 79 measured the difference for counting and replacing and it is not small: Arrow cuts the text after each match so `count("^a")` on `"aaa"` is 3 to Arrow and 1 to Python, Arrow steps `\b` in bytes so `count(r"\b")` on `"héllo"` is 5 to Arrow and 2 to Python, and the two disagree about an empty match at the end of a row. Answering a flagged `count` out of Arrow would produce a column of numbers that looks exactly like a right one.

So this slice is `case` everywhere and `flags` on `match`, which needs no new scan because nothing it touches leaves RE2. The Python engine's own scan is the next slice and section 11 says what it is.

## 4. Seeding rather than merging

A flag that arrived as an argument and a flag that was written `(?i)` have to be the same fact before anything reads either, or every check and every compile step has to learn about the second way of asking.

`parse_pattern` takes the flags as an argument and seeds the cursor with them before the pattern is walked. Nothing in the parser reads that field before the walk and the only write to it is an or, so a flag turned on by the caller stays on and a flag the pattern writes is added to it. What comes out is a tree carrying one set of flags and no memory of where they came from, which is the same shape the tree has always had.

Seeding rather than merging afterwards also means the two checks that read flags see the argument. A pattern written `(?u)` and handed `ASCII` beside it is refused for the reason a pattern writing both is refused, rather than for no reason at all. Nothing upstream sends that combination and the point is that the check cannot be walked around rather than that the combination matters.

`program_for` passes the flags to both of its parses. That second parse is the one `match` and `fullmatch` do on a pattern this library built rather than on the one the caller wrote, and a flag dropped on the way there would leave `contains` folding and the other two not. That is one method out of three quietly disagreeing with pandas, which is the failure this component is built to make impossible, so it has a test of its own.

## 5. A word rather than an argument

The accessor sends the kernel a word and the word is the method. `case=False` on a regular expression is `contains_regex_folded` rather than `contains_regex` with a second argument, which is the rule `firepanda/py/text.mojo` has followed since it was written and which `strip` and `strip_chars` follow for the same reason. Three new words, one per mask method, and the compiler strips the suffix and seeds `FLAG_IGNORECASE`.

`count` and `replace` do not get a folded word, because neither of them can be asked for one. `count` has no `case` argument at all, and `replace` with `case=False` goes to Python upstream and so is refused.

The fold the word asks for is not the fold the byte search does. The search maps one character to one character and the engine takes every code point that folds onto the one written down, so `STRASSE` holds `straße` to neither and a row holding a Kelvin sign matches a folded `k` to both. They agree on everything measured here and they are not the same rule, which is why the tests run half their patterns down each path.

## 6. The two refusals that belong to match

`match` is the only one of the four that takes a flag and it pays for that with two `ValueError`s nobody else has. Both are upstream's and both are reproduced, because a caller catching one of them today is catching something real.

The first is a `case` that disagrees with the `flags` beside it. pandas reads the compiled pattern's own ignore case bit back out and compares it to the argument, so `match(pat, case=True, flags=re.I)` raises and `match(pat, case=False, flags=re.I)` runs. The message names a compiled regular expression object even though the caller passed a string, because upstream compiled one on the way past.

The second is a flag beyond ignore case. `match(pat, flags=re.M)` raises `Cannot pass flags that do not match pat.flags`, where `fullmatch` with the same argument answers. That is a defect rather than a rule: the accessor zeroed the flags after compiling them in, and a check further down compares the zero it left behind against the pattern it built. It is in the upstream observations list and it is matched here in the meantime.

## 7. Which of the two fires first

A call can trip both checks and only one message comes out, so the order is part of the behaviour.

`match(pat, case=False, flags=re.M)` gives the case-sensitivity message and not the flags one. The reason is where the two checks live: the comparison between `case` and the compiled pattern is in the accessor and the flags check is further down in the array, so the accessor's check runs first and the array is never reached. The order is measured rather than read, and it is asserted against live pandas rather than against this paragraph.

## 8. What the other four are waiting for

`contains`, `fullmatch` and `count` refuse any flag, and `replace` and `extract` refuse any flag as well. Five refusals, one reason: the call belongs to Python's engine upstream and that engine's scan is not written here.

The refusal says so rather than saying the argument is unsupported. A caller who passed `flags=re.I` to `contains` is one keyword away from `case=False`, which is answered, and a caller who passed `flags=re.M` is not, and the message should let them tell which of the two they are. That is the same standard the literal refusal in `_literal` holds itself to.

`count` used to give a different message from the other three because it went through a different check. It goes through the same one now, so all four answer a caller who passes a flag with the same sentence.

## 9. What `match`'s default had to become

`match` is the only name on the accessor whose `case` starts off as nothing rather than as `True`. It has to, because it is the only one that can tell the two apart: a caller who passed nothing gets the fold read out of the flags, and a caller who passed `True` beside `re.I` gets a `ValueError`. Upstream draws the same distinction with its own sentinel.

The other three keep `case=True`, because nothing they do can distinguish a default from an argument that says the same thing.

## 10. What moved

Nine compat cases went on the board in firepanda-compat for the written flag, and ten runs moved from unimplemented to pass. The section totals did not move, because every parameter those cases touch was already covered, so they are depth rather than reach. The argument spelling adds no cases yet for the same reason the written one added no coverage: `case` was already a covered parameter on all three mask methods.

Four tests in the Python suite asserted that something here was refused and now assert what it answers instead. That is the honest shape of a slice like this one and the tests were rewritten rather than deleted, because the claim they were making is still worth making in the other direction.

## 11. What is not here yet

The Python engine's own scan, which is the next slice and the larger one. It wants a second counting loop and a second replacing loop following `re.finditer` rules, which means advancing one character on an empty match, never cutting the text and never stepping bytes. It gives `contains`, `fullmatch`, `count` and `replace` under `flags`, and `replace` under `case=False` with a real pattern, and it is what lifts `str.replace` and `str.extract` off their L2 ceiling on the board.

A scoped flag group is still refused, which is 525 held-out patterns in the differential. `(?i:a)` parses and the letters are thrown away, so a program built from that tree would answer without folding while both engines fold.

The five constructs the router sends to Python are still refused: lookaround, backreference, conditional, atomic group and possessive quantifier. The RE2 grammar front end is still the largest single gap in the component at 19620 of 30052 corpus patterns.

`findall` and `extractall` are the two names on this accessor that reach the engine and are not wired to it. Both want something the column layer does not have yet, a list column for one and a `MultiIndex` for the other.
