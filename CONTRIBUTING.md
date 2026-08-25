# Contributing

## Where the project is

Specification stage. There is no implementation yet, so the useful work right now is
in `docs/specs/`, and the most useful contribution is disagreement.

Three kinds of contribution are worth more than code at this point:

**Confirming or refuting a [verify] claim.** `docs/specs/01-research-2026.md` marks
every claim that came from a secondary source and could not be confirmed against
Modular's own documentation. Mojo 1.0 is recent and moved a lot on the way there.
If you have the toolchain installed and can check one, that is a real contribution.

**Finding a wrong premise.** The specification makes load-bearing assumptions: that
Mojo's `Dict` is not fast enough for group by, that the Arrow C Data Interface can be
implemented in Mojo with no copying, that monomorphizing every kernel over every dtype
does not blow up compile time. If one of these is wrong, finding out now is worth
several milestones.

**Saying what you would actually use.** The parity checklist in
`docs/specs/06-pandas-parity.md` is ordered by a guess about what matters. A pandas
program you actually run, with a note about which parts of it firepanda would need to
support, is better evidence than the guess.

## Once there is code

### Before you start

Every change belongs to a milestone in `docs/specs/08-milestones.md`. Say which one in
the pull request. Work that does not belong to a milestone is either a bug fix or a
conversation to have in an issue first.

### Setup

```sh
curl -fsSL https://pixi.sh/install.sh | bash
pixi install
pixi run build
pixi run test
```

### The rules that are not negotiable

**Public API names match pandas 3.0 exactly**, including where pandas is inconsistent.
This is the product, not a style preference. Somebody's muscle memory is the thing
being preserved, and our aesthetic preferences lose to it every time.

**Every kernel has a scalar twin.** A new kernel in `firepanda/kernel/` needs a
matching straightforward implementation in `firepanda/kernel/scalar/` and an entry in
the differential fuzz harness. The twin is never called in production; it exists to be
the specification the vectorized version is checked against. Write it first.

**Every public symbol has a docstring with a runnable example.** `mojo test` runs
docstring examples, which makes them the only documentation that cannot rot.

**No new shared mutable state outside `firepanda/exec/shared.mojo`.** Mojo has no race
detector. That file is the substitute: it holds every atomic and every mutable global
in the library, each with the invariant it maintains, and CI greps to enforce it. A
change that adds shared state without adding it there does not merge.

**Public signatures use stable standard library types only.** `String`, `List`, `Span`,
`Optional`, `Bool` and our own types. No `Dict`, no `Variant`, nothing from
`algorithm`, nothing from `max`. Those are unstable at Mojo 1.0, and pinning a public
API to them turns a compiler upgrade into a breaking release.

**Divergences from pandas are documented where they bite.** If your change makes
firepanda behave differently from pandas, it goes in the table in
`docs/specs/04-python-dx.md` section 6 and in the migration guide row for that
function — not in a general caveats section nobody reads.

### Benchmarks

Anything touching a kernel, the hash table or the executor needs numbers in the pull
request: the machine, the Mojo toolchain version, and the before and after. The
toolchain version matters because the ABI is not stable and the codegen changes
release to release, so a result that does not say which compiler produced it is not
reproducible.

CI runs a regression gate against main's recorded results. It will catch you.

### Floating point

Vectorized summation changes the association order, so results are not bit-identical
to a naive scalar loop, and the vectorized version is generally more accurate. Assert
closeness with a documented tolerance, never equality. Never write a single-pass
sum-of-squares variance.

## Commit messages

Conventional commits: `feat:`, `fix:`, `perf:`, `docs:`, `test:`, `ci:`, `refactor:`,
`spec:`. The changelog is generated from them.

Reference the milestone where it is not obvious: `feat(kernel): radix-partitioned group by (M5)`.

## Code of conduct

[CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md). It is the Contributor Covenant and it is
enforced.

## License

Contributions are licensed under Apache-2.0, matching the project.
