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
#
# The first argument picks a group. There are twelve programs and CI runs six of
# them: the six regular expression comparisons are run by hand, from the
# specification documents that introduce them, and never from a workflow. They
# were being compiled on every pull request all the same, which was about half
# of a step that took six and a half minutes and was the longest pole left in
# the pipeline after the test job was split up.
#
#   core    the six the workflow runs
#   regex   the six it does not
#   all     every one of them, which is the default and what a developer wants

set -euo pipefail

cd "$(dirname "$0")/.."

group=${1:-all}
case "$group" in
  core | regex | all) ;;
  *)
    echo "usage: $0 [core|regex|all]" >&2
    exit 2
    ;;
esac

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

core=(
  "mojo build -I . tests/differential/main.mojo -o build/differential/frames"
  "mojo build -I . tests/differential/sql.mojo -o build/differential/sql"
  "mojo build -I . tests/differential/sql_generated.mojo -o build/differential/generated"
  "mojo build -I . tests/differential/semantics.mojo -o build/differential/semantics"
  "mojo build -I . tests/differential/answers.mojo -o build/differential/answers"
  "mojo build -I . tests/differential/tpch.mojo -o build/differential/tpch"
)

# Six of the seven regex programs read a table from the directory they live in,
# so they need that directory on the import path as well as the repository root.
# The seventh does not and is built the same way anyway, because a build line
# that differs from its neighbours for a reason that is not visible in it is a
# line somebody copies wrong.
regex=(
  "mojo build -I . -I tests/differential tests/differential/regex.mojo -o build/differential/regex"
  "mojo build -I . -I tests/differential tests/differential/regex_match.mojo -o build/differential/regex-match"
  "mojo build -I . -I tests/differential tests/differential/regex_count.mojo -o build/differential/regex-count"
  "mojo build -I . -I tests/differential tests/differential/regex_replace.mojo -o build/differential/regex-replace"
  "mojo build -I . -I tests/differential tests/differential/regex_python.mojo -o build/differential/regex-python"
  "mojo build -I . -I tests/differential tests/differential/regex_re2.mojo -o build/differential/regex-re2"
  "mojo build -I . -I tests/differential tests/differential/regex_property.mojo -o build/differential/regex-property"
)

case "$group" in
  core) commands=("${core[@]}") ;;
  regex) commands=("${regex[@]}") ;;
  all) commands=("${core[@]}" "${regex[@]}") ;;
esac

printf '%s\0' "${commands[@]}" |
  xargs -0 -P "$jobs" -I COMMAND bash -c COMMAND
