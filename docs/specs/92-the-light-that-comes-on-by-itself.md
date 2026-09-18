# 92. The light that comes on by itself

## 1. What this is

Documents 90 and 91 each end by naming the same missing piece. This is that piece, and it is four dozen lines of Python and a step in a workflow file rather than anything in the engine.

Two rules that this library copies out of CPython's `re` moved in 3.14. Each is written down in the regular expression compiler as a version number, and the compiler compares the interpreter it was handed against one of them. That arrangement is right, and it has a failure mode that nothing in the repository would catch.

When the next release moves a third rule, nothing raises. The library goes on answering, with the rules of whatever version was current when somebody last measured, and the wrong answers are ordinary looking columns. The two rules already found were found by a differential sweep that happened to run under a newer interpreter than the accessor tests, which is luck rather than a procedure, and document 90 says so at the time.

So the next release has to be a red light on purpose.

## 2. Why this was deferred once

Document 90 built the version number and did not build this, and the reason it gave is worth keeping because it was right. There was one rule. A mechanism for keeping a list of one honest is a mechanism nobody maintains, and it would have been written, passed forever, and been believed without ever having been seen to fail.

That document also set the condition for building it, which was a second rule. Document 91 is the second rule. So this is next rather than someday, and the interval between the two is the whole of the argument for writing the condition down instead of the intention.

## 3. What is checked

Three things, all of them about the constants rather than about any pattern. The script reads the constants out of the compiler and the supported floor out of `pixi.toml`, so neither is spelled twice.

The running interpreter is not newer than `PYTHON_NEWEST`. That is the point of the exercise. A newer one means a release has happened that nobody has measured, and the message says what to do about it: run the corpus sweep across versions, write down what moved, add a constant for each rule that did, raise the number, and that documents 90 and 91 are what doing it looks like.

No rule constant is above `PYTHON_NEWEST`. A threshold in a version nobody has measured is a guess or a typo. This one is cheap rather than important.

No rule constant is at or below the floor. This is the one that does work nobody would otherwise do. When the floor rises past a threshold, every supported interpreter is already above it, the branch behind the constant can never run, and the rule has retired. The constant, the branch, the tests that asked for it and the paragraphs that explain it are all dead, and the only thing that would ever notice is somebody reading the file for another reason. So retiring a rule is a failing check with a message saying what to delete, which is how these are meant to leave.

## 4. Where it runs, which is twice

Under the pixi interpreter as a CI step, and again inside the accessor test suite.

That looks like belt and braces and is not. Document 90's whole finding is that this project has two test suites standing in two different Pythons and neither of them says which. The pixi environment is what the differential sweeps measure against, so its `re` is the oracle. The accessor suite runs under whatever `uv` was told to fetch, and that is the interpreter a contributor's pandas comparison is actually being made against. Either one can be upgraded without the other, and an upgrade to either is the event this is watching for.

Checking one and not the other would leave exactly the hole that produced document 90.

## 5. Why the failing paths are tested rather than only the healthy one

A check that has only ever seen a correct repository is a check whose failure message nobody has read. Each of the three complaints is asked for directly with constants written into the test, so the messages exist in a passing run and can be read by whoever is about to need them.

The row about the actual interpreter is separate and is the only one that looks at the repository as it is.

## 6. What this does not do

It does not know what changed. It says a version has arrived that nobody has measured, and the measuring is a person running the corpus sweep across interpreters. That sweep is still a thing somebody has to run rather than a thing that runs itself, and it is the larger gap of the two.

It also asks only about the interpreter it is standing in. A wheel installed on a newer Python than either of the two suites ever ran under gets nothing from this, because there is no check in the installed package and adding one would mean a library that refuses to import on a version it would mostly have answered correctly. That is a worse trade than a wrong answer on a handful of patterns, so it is not made, and it is the reason this is a development check rather than a runtime one.

## 7. What moved

A script in `tools/`, a pixi task, a CI step in the job that already had pixi, and a test file in the accessor suite. `PYTHON_NEWEST` now says in its own docstring that something checks it, since a constant whose whole risk is being forgotten should name the thing that remembers.
