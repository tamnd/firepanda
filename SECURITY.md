# Security policy

## Reporting

Report vulnerabilities privately through
[GitHub Security Advisories](https://github.com/tamnd/firepanda/security/advisories/new).
Please do not open a public issue.

Expect an acknowledgement within a few days. If a report is confirmed, the fix, the
advisory and the CVE request go out together.

## Supported versions

While the project is pre-1.0, only the latest release is supported. There are no
backports.

## What counts

firepanda parses untrusted input by design. CSV, NDJSON, Parquet, IPC and Arrow C
Data Interface structures may all come from somewhere the user does not control, and
the parsers are written in a language with `UnsafePointer` and manual buffer
management. The following are in scope:

- Memory corruption, out-of-bounds access or a crash triggered by a malformed file in
  any supported format.
- Anything reachable through the Arrow C Data Interface, where the memory belongs to
  another process's library and the lifetime is governed by a foreign release callback.
- An interpreter crash rather than a Python exception coming out of the extension
  module. Every exported function is supposed to convert internal failure into a
  raised exception; one that aborts the interpreter instead is a bug and may be a
  security bug.
- Path traversal or unintended file access through a dataset path, particularly in
  Hive-partitioned scanning where directory names become column values.

## What does not

- Resource exhaustion from a file the user chose to open. Reading a 500 GB Parquet
  file into memory is the user's decision; a memory budget is a feature request.
- `firepanda.sql()` executing the query it was given. It is a query engine.
- Anything requiring a malicious Mojo function to be compiled into the pipeline. Code
  you compile into your own process is code you trust.

## Supply chain

Releases are published to PyPI through Trusted Publishing, using OIDC. No API token
for PyPI exists in this repository's secrets.

Every published artifact carries a build provenance attestation, verifiable with:

```sh
gh attestation verify firepanda-<version>-<platform>.whl --repo tamnd/firepanda
```

Wheels vendor the Mojo runtime, because the Mojo ABI is not stable within 1.x and a
wheel is a binary artifact. The toolchain version is recorded in the wheel metadata.
A vulnerability in the vendored runtime is a vulnerability in the wheel, and it will
be handled as one: report it here and we will rebuild and re-release.

Workflow definitions are audited by `zizmor` on every pull request.
