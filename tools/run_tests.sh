#!/usr/bin/env bash
# Runs every unit test file.
#
# Mojo 1.0 has no `mojo test`. A test file is a program whose `main` calls
# `TestSuite.discover_tests[__functions_in_module()]().run()`, which exits
# non-zero if anything fails, so the runner is a loop with a failure tally rather
# than anything clever.
#
# Every file is run even after one fails, because a compile error in one module
# usually means the same mistake is in three, and finding all three in one round
# trip is worth the extra seconds.

set -uo pipefail

cd "$(dirname "$0")/.."

failed=0
ran=0

for file in tests/test_*.mojo; do
  [ -e "$file" ] || continue
  ran=$((ran + 1))
  echo "=== $file"
  if ! mojo run -I . "$file"; then
    failed=$((failed + 1))
  fi
done

if [ "$ran" -eq 0 ]; then
  echo "no test files found under tests/" >&2
  exit 1
fi

echo
if [ "$failed" -ne 0 ]; then
  echo "$failed of $ran test files failed"
  exit 1
fi
echo "$ran test files passed"
