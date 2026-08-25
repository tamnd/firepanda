# Changelog

All notable changes are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and versions follow
[semantic versioning](https://semver.org/spec/v2.0.0.html).

Pre-1.0, minor versions may break the API. Every break appears here with the
migration, not just with a note that it happened.

The Mojo toolchain version is part of a release's identity and is recorded with each
entry, because the Mojo ABI is not stable within 1.x and a binary artifact built
against one runtime is not guaranteed to load against another.

## [Unreleased]

### Added

- The specification: twelve documents in `docs/specs/`, written against Mojo 1.0,
  pandas 3.0.5, Polars 1.43 and Arrow 25.0.0 as of August 2026.
- Milestone issues M0 through M11 covering the work to a defensible 1.0.
- CI: spec conformance checks, a guarded build and test matrix, wheel builds with
  clean-install verification, a nightly Mojo canary that files an issue on failure,
  a benchmark regression gate, and workflow, dependency and Scorecard auditing.
- Release: PyPI publishing through Trusted Publishing with build provenance
  attestations. No PyPI token exists in this repository.

Nothing is implemented. See the status notice in the README.
