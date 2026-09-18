# 90. A rule with a date on it

## 1. What this is

The slice document 89 describes ended with a red light. One of the six differential sweeps came back with 106 disagreements out of the ten thousand patterns it samples, against a ceiling of zero, and every pattern it named held a `\B`. The first thing to do with a red light at the end of a slice that touched the compiler is to assume it is the slice, and the second thing is to check, and for a bare `\B` with no bracket anywhere in it not one line of that slice's new code runs.

It was not the slice. `re.search(r"\B", "")` is None on 3.9, 3.10, 3.11, 3.12 and 3.13, and is a match on 3.14. CPython 3.14 took out the special case that failed a `\B` on an empty subject and made `\B` the plain negation of `\b`, which is what RE2 has always had and what every other engine has always had. The accessor tests run under 3.13 and asserted the old answer and passed. The sweep runs under the pixi environment, which is 3.14, and failed. The two suites were never disagreeing with each other about anything. They were asking two different interpreters.

That is the part worth keeping and it is bigger than the one rule. This library copies `re`, and a rule copied off a running interpreter is a measurement with a date on it. The measurements have been going into the code and the dates have not. This slice puts one date in, for the rule that moved, and builds the place the next one goes.

## 2. What actually moved, measured

Asked of a running interpreter rather than of a changelog. `re.search(r"\B", "")` is None on 3.9 through 3.13 and a match on 3.14.7. `re.search(r"\b", "")` is None on every one of them, which is the half that did not move and could not have: a boundary needs something on one side of it and an empty subject has nothing on either.

So the pair `\b` and `\B` were not a pair up to 3.13. The positive half was a question about two neighbouring characters and the negative half was that question negated with one row carved out of it, and 3.14 removed the carve. Nothing else about either of them changed, under any alphabet, at any position of any row that holds a character.

## 3. How much of the corpus this is, and what else is in there

The held-out corpus is 30052 patterns. Every one of them was run against the sixteen texts the sweeps use, under 3.12, 3.13 and 3.14, and the three sets of answers compared cell by cell.

3.12 and 3.13 agree on all 30052. 3.13 and 3.14 disagree on 280, and those 280 fall into exactly two families and no others.

130 of them are this one, where a pattern answered `n` on the empty text under 3.13 and answers `y` under 3.14 with every other text unchanged. The other 150 are a different change in the same release: `\z` became a legal escape in 3.14 and is a `bad escape \z` on 3.13 and before, so a pattern holding one moves from an error to an answer. That second family is a slice of its own and section 10 says why it cannot ride along with this one.

Two families in one release is the argument against fixing this with a flag. One rule changing is a special case. Two rules changing in the same release, in a library this one copies, in a range of versions the project supports, is a pattern.

## 4. Why not a boolean

The obvious shape is a `Bool` on the builder saying whether the empty row case is on. It would work, it is one bit, and it is wrong for the reason a reader would find it wrong six months from now.

A reader who arrives at `b.empty_row_case` has to go somewhere else to learn which interpreters have it, and the somewhere else is a comment that will not be updated when the next rule moves. A reader who arrives at `b.minor < PYTHON_PLAIN_NON_BOUNDARY` has the whole question in front of them: which interpreter, and which rule, and the name of the release the rule changed in. The second rule that moves gets a second constant beside the first and nothing else has to change. The second rule that moves in the boolean version gets a second boolean, and then the third, and at some point somebody bundles them into a struct and calls it a version.

`pixi.toml` says this project supports Python 3.12 and up. That is a range, not a version, and a number is what a range is made of.

## 5. Why not a fourth position code

This library already had two position codes for `\B`, because the two alphabets are a real difference. `AT_NON_BOUNDARY` asks the question against the ASCII 63 and `AT_NON_BOUNDARY_UNICODE` asks it against 138558, and which of the two a pattern gets is settled while it is being compiled rather than once per position of every row, which is what document 81 is about.

Document 88 added a third, `AT_NON_BOUNDARY_ASCII`, and the reasoning it gave was that `\B` is a question with a special case attached and only the question narrows. The code was right and the reasoning was upside down. Writing the empty row case as a third position code says that the answer for an empty row is a fact about which characters are word characters. It is not. `(?a)` narrows which characters count and says nothing at all about a row that holds none of any kind, and the proof is that the answer under `(?a)` and the answer without it were the same answer on every interpreter, before and after the change.

The case is a question about the row. So it is an instruction about the row: `AT_TEXT_NOT_EMPTY`, which reads the length rather than the position, and which the compiler writes in front of a `\B` when the interpreter it is compiling beside is one of the ones that has the case. Two instructions for one node. A thread has to satisfy both to go on, so an empty row dies on the first and a row with something in it reads the second and gets the plain answer.

With the case out of the codes, the narrow reading of `\B` is RE2's code again and the third value is gone. `\b` has two codes and `\B` has two codes and each pair differs only in which characters are word characters, which is what the pair was always supposed to mean.

## 6. Where the version comes from

`firepanda/py/text.mojo` reads `sys.version_info.minor` off the interpreter the call arrived in and hands it to `program_for`, which hands it to `compile_program`, which puts it on the builder. It is read on every compile rather than kept, because there is no module level mutable state here and because reading an attribute off an already imported module is not the expensive part of compiling a pattern.

It is read there rather than passed in from further out because it is not a thing a caller chose. A caller chose a pattern and some flags. Which interpreter their pandas is running is a fact about the process, and the door is where this library touches the process.

Every other caller gets `PYTHON_NEWEST`, which is 14, which is the newest CPython this has been measured against. That is the right default for a kernel test or a differential harness that has no interpreter to ask, and it is a constant with a docstring rather than a literal 14 in six places, so the next time the newest one moves the number moves once.

## 7. RE2 has not got a version of Python

RE2 gets none of this and the reason is older than the slice. RE2 asks the word boundary question between bytes rather than between characters, so this library refuses `\B` on that engine outright rather than answering it wrongly, and has since long before any of this. The refusal reads `RE2 reads a non boundary between bytes` and it is one of the gaps the board counts.

That refusal is what makes the version question Python's alone, and it is worth a test row of its own rather than a comment, because the day the RE2 front end learns to answer `\B` is a day somebody has to think about whether the empty row travels with it.

## 8. What it looks like compiled

The listing is where the shape shows with nothing running. `\B` on Python's engine beside 3.13 compiles to `at(12); at(11); match` and beside 3.14 to `at(11); match`. Under `(?a)` it is `at(12); at(8); match` and `at(8); match`. The extra instruction is the same instruction under both alphabets, which is the whole claim of section 5 written as four lines of a test.

It sits in front of the boundary rather than in front of the program, so a `\B` that is not the first thing in a pattern still only asks about the row it is in. `a|\B` on an empty row is the row that says so.

## 9. What moved

`firepanda/kernel/regex/tokens.mojo` lost `AT_NON_BOUNDARY_ASCII` and gained `AT_TEXT_NOT_EMPTY` in its place, and the docstrings on `AT_NON_BOUNDARY` and `AT_NON_BOUNDARY_UNICODE` now say they are a pair. `firepanda/kernel/regex/pike.mojo` answers the new code with a length test and has lost the alphabet crossed empty case. `firepanda/kernel/regex/program.mojo` gained `PYTHON_NEWEST` and `PYTHON_PLAIN_NON_BOUNDARY`, a `minor` field on the builder and a `minor` argument on `compile_program`, and its `_at_value` narrow branch is one line where it was two. `firepanda/kernel/regex/method.mojo` threads the number through `program_for`. `firepanda/py/text.mojo` gained `_python_minor`.

`tests/test_regex_boundary_empty.mojo` is the new file. Two existing rows moved because they had the old answer written into them: one in `tests/test_regex_verbose_ascii.mojo`, which was the row document 88 caught the alphabet mistake on and which now asserts the thing that is true under both interpreters, and one in `tests/test_regex_python_scan.mojo`, where `count` on an empty row went from zero to one. `python/tests/test_str_boundary_empty.py` is the new accessor file and one test in `python/tests/test_str_flags_python_engine.py` now computes its expected value off the running `re` rather than spelling it.

## 10. What is not done

The `\z` family is the other 150 patterns and it is not here. It is a harder shape than this one, because the refusal has to be driven off what the caller wrote rather than off the tree that reaches the compiler: `python_anchored` writes this library's own `\A(...)\z` around a pattern on the way to Python's engine, so by the time a program is built every pattern has a `\z` in it and the question of whether the caller wrote one has already been answered wrongly. That wants a field on the parse recording a fact about the caller's text, which is the shape `scoped` has, and it is a slice.

The sweep this came out of only ever asks sixteen texts. Sixteen texts found one rule because the first of the sixteen is the empty string, and a rule that only shows on a row nobody thought to include is a rule that is still in here. What would find those is a corpus of texts built the way the corpus of patterns was, which does not exist.

Nothing checks that `PYTHON_NEWEST` is the newest. A CI job that fails when the interpreter it is running under is newer than the constant would turn the next release into a red light rather than into a quiet wrong answer, and that is worth doing the next time a version rule lands rather than now, when there is one of them.
