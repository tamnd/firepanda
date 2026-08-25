<!-- Keep this short. Delete anything that does not apply. -->

## What this changes

## Why

## Checks

- [ ] Milestone this belongs to, from `docs/specs/08-milestones.md`:
- [ ] New public symbols have a docstring with a runnable example (`mojo test` runs it)
- [ ] New kernels have a scalar twin in `firepanda/kernel/scalar` and are in the differential fuzz
- [ ] Public API names match pandas 3.0, including where pandas is inconsistent
- [ ] No new `Atomic` or mutable global outside `firepanda/exec/shared.mojo`
- [ ] Public signatures use stable stdlib types only — no `Dict`, no `Variant`, nothing from `max`
- [ ] Any divergence from pandas is in the table in `docs/specs/04-python-dx.md` section 6

## Benchmarks

<!-- Required for anything touching a kernel, the hash table, or the executor.
     Numbers with the machine and the toolchain version, or say why not applicable. -->
