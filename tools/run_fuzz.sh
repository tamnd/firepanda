#!/usr/bin/env bash
# Runs every fuzzer, all of them at once, and passes its arguments to each.
#
# The four fuzzers are independent programs with no shared state, and each one
# spends its first several seconds in the compiler. Running them in sequence made
# the pull request wait for the sum of four compiles when the machine could do
# them at the same time.
#
# `--max-total-time=N` means N seconds per fuzzer rather than N in total, which
# is the same as it meant when they ran in sequence.
#
# Output is collected per fuzzer and printed when everything is done, in a fixed
# order, because four fuzzers writing progress to the same terminal is unreadable.

set -uo pipefail

cd "$(dirname "$0")/.."

fuzzers=(main kernel hash join)

# BSD mktemp wants a template, so the macOS job needs one too.
logs=$(mktemp -d "${TMPDIR:-/tmp}/firepanda.XXXXXXXX")
trap 'rm -rf "$logs"' EXIT

for name in "${fuzzers[@]}"; do
  (
    if mojo run -I . "tests/fuzz/$name.mojo" "$@" > "$logs/$name.log" 2>&1; then
      : > "$logs/$name.ok"
    fi
  ) &
done
wait

failed=0
for name in "${fuzzers[@]}"; do
  echo "=== tests/fuzz/$name.mojo"
  cat "$logs/$name.log"
  [ -e "$logs/$name.ok" ] || failed=$((failed + 1))
done

echo
if [ "$failed" -ne 0 ]; then
  echo "$failed of ${#fuzzers[@]} fuzzers failed"
  exit 1
fi
echo "${#fuzzers[@]} fuzzers passed"
