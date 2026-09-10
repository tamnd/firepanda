# 22. What a message is for

Document 14 decided which class an error arrives as. This one is about the string it carries, which turned out to be a separate question with a separate answer, and which nobody had written down until four messages were found wrong at once.

## The class is not the whole product

A user meets a failure twice. The first time is the `except` clause, and document 14 is about that: an `except KeyError` that stops firing is a process that dies, so the class has to be the one pandas raises. The second time is the traceback on a screen, and what happens next is not a language feature. They read the last line, they do not recognise it, and they paste it into a search box.

That second meeting is the one this document is about, and the thing it turns on is that firepanda's documentation does not exist yet and pandas' does. A user who searches `Already tz-aware, use tz_convert to convert` lands on a page that explains the difference between localizing and converting, which is exactly what they needed and which nobody at firepanda had to write. A user who searches `this column is already on UTC` lands on nothing. The two messages say the same thing to a person reading carefully. They are worth very different amounts.

So a message on the pandas facing path is not judged by whether it is true or by whether it is clear. It is judged by whether it contains the string that finds the explanation.

## Pandas' sentence first, ours after it

The rule is not to copy pandas' message. It is to lead with it.

Almost every one of these messages had something in it worth keeping. `tz_convert has nothing to convert this column from, because it carries no zone` says which column and why. `no common type for int64 and string` says what the promotion table decided. Throwing that away to gain a search phrase trades one thing for another when both fit in a single string, and the string is not short of room.

So the shape is pandas' sentence, then ours:

```
Already tz-aware, use tz_convert to convert. This column is already on UTC, and
tz_convert is the one that reads it against another clock.
```

The first sentence is copied because it is the index key. The second is ours because it is the part that knows which column this was. Neither has to be sacrificed.

Where we know something pandas does not, it goes in the second half. `astype` on text that will not read says `invalid literal for int() with base 10: 'x' at row 1`, and the row number is not in the pandas message. That is the one direction in which being different is free: the phrase somebody searches for is still there, and the extra fact costs them nothing to ignore.

## What is not copied

Three kinds of pandas message are deliberately not reproduced.

The first is a message that is not pandas'. `pd.Series([1,2,3]) + "x"` raises `UFuncTypeError: ufunc 'add' did not contain a loop with signature matching types (dtype('int64'), dtype('<U1')) -> None`. That is numpy's internals arriving through pandas without pandas having decided anything about it. Copying it would be copying a leak, and it would tie firepanda's messages to a library firepanda does not use. The constant path keeps its own message for that reason.

The second is a message that would be false. Our dtype spellings are ours: firepanda prints a text column as `string` where pandas 3 prints `str`, because the type has an Arrow name and pandas does not get to rename it in a message. The searchable part of `operation 'add' not supported for dtype 'int64' with dtype 'str'` is `not supported for dtype`, and that is what gets matched. The dtype names are facts about firepanda and stay firepanda's.

The third is a message that is worse than ours. `bool - bool` is refused with numpy's own sentence naming `bitwise_xor` as the operator that works, which tells somebody what to write instead. A general purpose message about two dtypes with no operation between them would have replaced a fix with a diagnosis. Where our message is more useful, it wins, and the code has to be able to tell the two cases apart without reading its own strings.

## Deciding which failure this is, without reading the message

The last point is the implementation constraint and it is worth stating as a rule, because the obvious way to write any of this is to catch an error and test whether its message starts with a known prefix.

That is a dependency between two files through a string literal, and nothing checks it. A message reworded in the kernel silently stops matching in the binding, the rewrite silently stops happening, and the only symptom is a conformance case that was passing and is not. There is no compiler error and no test failure anywhere near the edit.

So the question is asked again instead. `binary_failure` in `firepanda/py/ops.mojo` calls `promote` on the two dtypes, which is the same function the core consulted, is a function of the two types alone, and cannot disagree with what the core decided. If it refuses, the failure is the promotion failure and the sentence goes on. If it answers, the failure was something else and the cause is kept whole.

This is the same argument document 14 makes about not guessing a class from a message. A message is for a person. Anything the code needs to know has to be asked for.

## Where this applies

The pandas facing path only, which is `firepanda/py/` and the kernel messages that reach a user through it. The native Arrow facing API keeps Arrow's vocabulary, for the reason `firepanda/py/cast.mojo` gives about `astype`: a user calling the Mojo API directly is not searching pandas documentation and is not helped by a sentence written for somebody who is.

Four messages were fixed under this rule in the change that wrote it, and the fixes are in the changelog. The rule is here so the fifth one does not have to rediscover it.
