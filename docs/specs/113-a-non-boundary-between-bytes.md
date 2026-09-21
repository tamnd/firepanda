# 113. A non boundary between bytes

## 1. What this is

The larger of the two families document 111 ended on, and the largest single regular expression gap this library has left. A hundred and twenty four corpus patterns are held out of every comparison with the sentence `RE2 reads a non boundary between bytes`, and the comment beside the refusal says the fix is to run the engine over bytes rather than code points and that it is a larger change than the one it sits in.

That is still true and this document does not do it. What it does is measure the rule exactly, and then take the part of the gap that turns out not to be a gap at all.

## 2. The rule, measured

RE2 runs on bytes. A word byte is one of `[0-9A-Za-z_]` and nothing else, so every byte of every character outside ASCII is a non word byte on both sides of every position inside it.

```
    \b at byte position i   is_word(byte i-1) is not is_word(byte i)
    \B at byte position i   is_word(byte i-1) is     is_word(byte i)
```

with a position off either end counting as not a word byte. Measured against the RE2 inside pyarrow 24.0.0 by asking `replace_substring_regex` to mark every position it matched at, over all ninety six texts of document 112 and both patterns, and the model above reproduces every one of them.

The clearest single piece of evidence is that `\B` produces invalid UTF-8 and `\b` never does.

```
    "é"    \b  ->  'é'                         no match anywhere
    "é"    \B  ->  b'|\xc3\xa9|'               a match at 0 and at 2
    "aé"   \B  ->  b'a\xc3|\xa9|'              a match at 2, between two bytes
```

The third line cannot be decoded. Arrow hands back a string that is not UTF-8 at all, and `pyarrow` itself raises `UnicodeDecodeError` reading it back. That is what a non boundary between bytes means in practice.

`\b` is unaffected, which the comment in the compiler already claimed and which the measurement confirms: a boundary needs a word byte on one side of it, a byte inside a character is never one, so `\b` never matches at a position that is not also a character position.

## 3. The scan rule

Separate from the predicate and needed to predict anything longer than one match. After an empty match, RE2 advances by one decoded rune: the length of the UTF-8 sequence at that position, or one byte if the byte there is not a valid lead. So a match at the start of `中` advances three and a match between two of its bytes advances one.

```
    "中"     \B  ->  matches at 0 and 3, not at 1 or 2
    "a中b"   \B  ->  matches at 2 and 3
```

Both fall out of the same rule. In the first the match at 0 is on a character boundary and steps over the whole character, and in the second the match at 2 is on a continuation byte and steps one.

## 4. What `count` does, and why it is not this

`count_substring_regex` does not agree with either model, and it does not agree with its own `replace`. `\b` over `"abc"` marks two positions and counts three. `\b` over `"a"` marks two positions and counts one.

The rule it follows is that each match restarts the scan on the bytes that are left, so the byte before the new starting position is forgotten and the new start looks like the start of a text. `"abc"` matches at 0, then `"bc"` is scanned fresh and `b` looks like the beginning of a word, then `"c"` the same, which is three.

That is Arrow losing the text context, it is a defect rather than a rule, and it is already on the list of things to report upstream. It is written down here because anybody measuring this family will meet it within about a minute and should not spend an afternoon trying to derive a predicate that explains it.

## 5. The part that is not a gap

Every position RE2 has and this engine has not is strictly inside a character. Three things follow from that and each one is a fact about the bytes rather than a guess.

Nothing can be read at such a position. Every character instruction this compiler emits matches a whole character, and a position inside one is not where a whole character begins.

No position test holds there except the non boundary. It is above zero so no start anchor holds, it is below the end so no end anchor holds, the byte on either side of it is a lead byte or a continuation byte and therefore never a newline, so neither `(?m)^` nor `(?m)$` holds, and neither side is a word byte, so `\b` does not hold.

So the only match RE2 can have at one of those positions is the empty string, found by a path through non boundaries and nothing else.

## 6. Two questions that cannot see it

Every one of those positions is above zero. A program that can only ever match at position zero therefore never attempts one, and answers the same thing whichever reading is used.

That is exactly `str.match` and `str.fullmatch`. Both are built by wrapping the pattern in an anchor, `^(...)` for one and `\A(...)\z` for the other, and both want the leftmost match, which has to start where the anchor says. Interior positions are never attempted, so the two engines cannot part on them.

The compiler already worked out whether a program is anchored, in `_anchored`, because the scan wants to know. It works it out after the whole program is emitted, and the refusal was being made while it was still being emitted, which is why the question had never been asked here. The refusal is now a note on the builder and the decision is made where the answer is known.

A hundred and twenty four patterns were held out of each of those two comparisons and none are now.

## 7. The other four

They are unanchored, so they attempt interior positions and the argument above does not reach them. A narrower one does.

An interior match is the empty string through non boundaries. A program that has no such path has nothing to find there, whatever it can do elsewhere, and answers the same on both readings. So `\Ba` is answered, because something has to be read after the assertion and nothing can be read there, while `\B` on its own is held out, and `\Ba|\B` is held out because one of its branches is bare.

The test is a reachability walk over the emitted instructions rather than a simulation. It steps through the branching and the saves, follows a non boundary, and stops at anything that consumes. The position tests it stops at are named one code at a time rather than taken as everything that is not the non boundary, so a position test added later is a refusal until somebody has thought about whether it holds inside a character.

This is worth less than the anchored argument, because the family is mostly bare. A hundred and six of the hundred and twenty four are `\B` alone or an alternation with a bare `\B` in it.

## 8. What the differential says

```
                    held out before   after
    routing               831           831
    str.contains          572           554
    str.match             638           514
    str.fullmatch         638           514
    str.count             586           568
    str.replace           260           248
    str.findall           474           474
```

Two hundred and ninety six held out patterns answered, all seven comparisons still at ten thousand agreements in ten thousand with no disagreements, over the ninety six text corpus rather than the sixteen this would have been measured against a slice ago.

`findall` is unchanged because it is Python's engine, where the boundary question is asked between characters and there was never anything to refuse.

## 9. What is left

A hundred and six patterns, and the fix for them is the one the original comment named. `\B` alone needs the engine to stand at a byte position inside a character, which means walking bytes, which means every character instruction becoming a sequence of byte instructions and every range table becoming a byte automaton. That is the shape of the change and it is not a slice.

There is a cheaper answer for one of the four. `str.contains` with a program that can match empty through non boundaries is true whenever the text holds any character outside ASCII, because such a character has an interior position and the empty match is always available there. That is a rule about the scan rather than about the engine and it would take `contains` to zero. It does not help `count` or `replace`, which need the positions themselves and not just their existence, and a fix that moves one of four is worth writing down rather than writing.

The measurement in sections 2 and 3 is the part of this document that outlives it. Whoever does the byte engine needs the predicate and the rune advance stated exactly, and now they are, against a corpus of ninety six texts rather than against an argument.
