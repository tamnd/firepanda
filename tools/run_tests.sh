#!/usr/bin/env bash
# Runs every unit test file.
#
# Mojo 1.0 has no `mojo test`. A test file is a program whose `main` calls
# `TestSuite.discover_tests[__functions_in_module()]().run()`, which exits
# non-zero if anything fails, so the runner is a loop with a failure tally rather
# than anything clever.
#
# The loop runs several files at once, because each one is a separate `mojo run`
# that compiles the library again and a test file spends most of its wall clock
# in the compiler rather than in the assertions. Serially this step was the
# largest single cost in the pull request pipeline.
#
# Output is collected per file and printed in filename order once that file
# finishes, so the log reads the same as the serial one did rather than as four
# test suites interleaved. Every file is run even after one fails, because a
# compile error in one module usually means the same mistake is in three, and
# finding all three in one round trip is worth the extra seconds.
#
# `FIREPANDA_TEST_JOBS` overrides the width. Set it to 1 to get the old
# behaviour when a failure is confusing enough to want a clean serial log.

set -uo pipefail

cd "$(dirname "$0")/.."

if [ -n "${FIREPANDA_TEST_JOBS:-}" ]; then
  jobs=$FIREPANDA_TEST_JOBS
elif command -v nproc > /dev/null 2>&1; then
  jobs=$(nproc)
elif command -v sysctl > /dev/null 2>&1; then
  jobs=$(sysctl -n hw.ncpu)
else
  jobs=2
fi
# Each `mojo run` is itself parallel, so handing it every core twice over makes
# the machine slower rather than faster.
[ "$jobs" -gt 8 ] && jobs=8
[ "$jobs" -lt 1 ] && jobs=1

files=(tests/test_*.mojo)
if [ ! -e "${files[0]}" ]; then
  echo "no test files found under tests/" >&2
  exit 1
fi

# BSD mktemp wants a template, so the macOS job needs one too.
logs=$(mktemp -d "${TMPDIR:-/tmp}/firepanda.XXXXXXXX")
trap 'rm -rf "$logs"' EXIT

echo "running ${#files[@]} test files, $jobs at a time"

run_one() {
  local file=$1 logs=$2
  local base=${file##*/}
  if mojo run -I . "$file" > "$logs/$base.log" 2>&1; then
    : > "$logs/$base.ok"
  fi
}
export -f run_one

printf '%s\0' "${files[@]}" \
  | xargs -0 -P "$jobs" -I {} bash -c 'run_one "$1" "$2"' _ {} "$logs"

failed=0
for file in "${files[@]}"; do
  base=${file##*/}
  echo "=== $file"
  cat "$logs/$base.log"
  [ -e "$logs/$base.ok" ] || failed=$((failed + 1))
done

echo
if [ "$failed" -ne 0 ]; then
  echo "$failed of ${#files[@]} test files failed"
  exit 1
fi
echo "${#files[@]} test files passed"
