#!/usr/bin/env bash
# Builds the differential programs, several at a time.
#
# They used to be built one after another by a chain of && in the pixi task,
# which was fine at three of them and stopped being fine at eight. Five of them
# took five minutes and nine seconds on a CI runner, so eight was past the job's
# eight minute ceiling and every pull request's differential job began timing
# out during the build, before it had compared anything.
#
# None of them depends on any of the others, so they are built several at once
# and the step costs about what the slowest one costs rather than the sum.
#
# `FIREPANDA_BUILD_JOBS` overrides the width, which is what to reach for on a
# machine that is short of memory rather than short of cores.

set -euo pipefail

cd "$(dirname "$0")/.."

if [ -n "${FIREPANDA_BUILD_JOBS:-}" ]; then
  jobs=$FIREPANDA_BUILD_JOBS
elif command -v nproc > /dev/null 2>&1; then
  jobs=$(nproc)
elif command -v sysctl > /dev/null 2>&1; then
  jobs=$(sysctl -n hw.ncpu)
else
  jobs=2
fi
# A Mojo compile is itself parallel and holds around a gigabyte while it runs,
# so the width here is bounded by memory rather than by cores.
[ "$jobs" -gt 4 ] && jobs=4
[ "$jobs" -lt 1 ] && jobs=1

mkdir -p build/differential

# The five regex programs read a table from the directory they live in, so they
# need that directory on the import path as well as the repository root.
printf '%s\0' \
  "mojo build -I . tests/differential/main.mojo -o build/differential/frames" \
  "mojo build -I . tests/differential/sql.mojo -o build/differential/sql" \
  "mojo build -I . tests/differential/sql_generated.mojo -o build/differential/generated" \
  "mojo build -I . tests/differential/semantics.mojo -o build/differential/semantics" \
  "mojo build -I . tests/differential/answers.mojo -o build/differential/answers" \
  "mojo build -I . -I tests/differential tests/differential/regex.mojo -o build/differential/regex" \
  "mojo build -I . -I tests/differential tests/differential/regex_match.mojo -o build/differential/regex-match" \
  "mojo build -I . -I tests/differential tests/differential/regex_count.mojo -o build/differential/regex-count" \
  "mojo build -I . -I tests/differential tests/differential/regex_replace.mojo -o build/differential/regex-replace" \
  "mojo build -I . -I tests/differential tests/differential/regex_python.mojo -o build/differential/regex-python" \
  "mojo build -I . tests/differential/tpch.mojo -o build/differential/tpch" |
  xargs -0 -P "$jobs" -I COMMAND bash -c COMMAND
