#!/usr/bin/env bash
# Rebuilds `tools/test_costs.txt` from a CI run's shard logs.
#
# The table is what the test shards in `.github/workflows/ci.yml` divide the
# file list on, and it has to be measured on the machine that runs it. A
# workstation's numbers do not transfer: the first version of the table came off
# a sixteen core machine on the theory that only the ratios matter, and the
# ratios do not hold, so shards balanced to within half a per cent there came out
# spanning 423 to 622 seconds on a runner.
#
# `pixi run test-costs` measures a local run, which is the right thing when there
# is no CI run to read. This is the right thing when there is one, and it is free
# because a sharded run already prints the time next to every file.
#
#   tools/test_costs_from_ci.sh 35593555863
#
# The argument is a run id, which `gh run list --workflow CI` prints. A run with
# a failed shard is missing that shard's files and is refused below rather than
# silently writing a short table.
#
# The run also has to be a cold one, and this is the part that is easy to get
# wrong because a warm run looks better in every way. The time a shard prints
# next to a file is wall clock for compiling and running it, and the compiler
# cache is restored per shard slot, so on a run where the slots already held the
# right files most of that number is a cache hit rather than a compile. A table
# built from one of those is not a table of what the files cost, it is a table
# of what was cached, and it will balance the shards on the wrong thing.
#
# It is worth being concrete about the size of it, because the numbers are not
# subtly off. A table regenerated from a warm run came out at 5076 seconds
# against the 12086 the suite really costs, which reads as the suite having got
# more than twice as fast. Checked against a cold run afterwards, the old table
# was within two per cent and the warm one was out by a median factor of five
# and a maximum of twenty: `tests/test_category_compare.mojo` was written down
# as 7 seconds and actually costs 142.
#
# A run is cold when the shard slots did not already hold the files they were
# given, which in practice means the first run after the shard count changes, or
# after the caches are pruned. The check at the bottom refuses a table that came
# out far cheaper than the one it is replacing, which is what a warm run looks
# like from here.

set -euo pipefail

cd "$(dirname "$0")/.."

run=${1:-}
if [ -z "$run" ]; then
  echo "usage: $0 <run-id>" >&2
  exit 2
fi

repo=${FIREPANDA_REPO:-tamnd/firepanda}
out=${2:-tools/test_costs.txt}

ids=$(gh api --paginate "repos/$repo/actions/runs/$run/jobs" \
  --jq '.jobs[] | select(.name | startswith("Tests (ubuntu-latest")) | "\(.id) \(.conclusion)"')

if [ -z "$ids" ]; then
  echo "run $run has no ubuntu test shards" >&2
  exit 1
fi

if echo "$ids" | grep -qv ' success$'; then
  echo "run $run has a test shard that did not succeed, so its files have no times:" >&2
  echo "$ids" | grep -v ' success$' >&2
  exit 1
fi

rows=$(mktemp)
trap 'rm -f "$rows"' EXIT

while read -r id _; do
  # `--allow-escape-sequences` because the log is what a terminal saw, colour
  # codes and all, and gh refuses to print it otherwise. The nulls come from the
  # same place.
  gh api --allow-escape-sequences "repos/$repo/actions/jobs/$id/logs" \
    | tr -d '\000' \
    | grep -aoE '=== tests/test_[a-z_0-9]+\.mojo \([0-9]+s\)' \
    | sed -E 's/=== (tests[^ ]*) \(([0-9]+)s\)/\2 \1/' >> "$rows"
done <<< "$ids"

count=$(wc -l < "$rows" | tr -d ' ')
unique=$(cut -d' ' -f2 "$rows" | sort -u | wc -l | tr -d ' ')
present=$(ls tests/test_*.mojo | wc -l | tr -d ' ')

if [ "$count" != "$unique" ]; then
  echo "run $run timed $count files but only $unique distinct ones, so two shards overlap" >&2
  exit 1
fi
if [ "$unique" != "$present" ]; then
  echo "run $run timed $unique files and the checkout has $present, so the run is not this tree" >&2
  exit 1
fi

before=""
if [ -e "$out" ]; then
  before=$(awk '!/^#/ && NF >= 2 { s += $1 } END { print s + 0 }' "$out")
  cp "$out" "$out.before"
fi

{
  echo "# Seconds per test file, written by \`tools/test_costs_from_ci.sh\` out of"
  echo "# run $run. The test shards in .github/workflows/ci.yml divide the"
  echo "# list up on these numbers."
  echo "#"
  echo "# Measured on the CI runner itself, four cores running four files at a"
  echo "# time, which is what these numbers have to come from. A workstation's do"
  echo "# not transfer: the ratios between files are not the same on sixteen cores"
  echo "# as on four, and a table measured on one balanced the shards to within"
  echo "# half a per cent there and to 423 against 622 seconds here."
  echo "#"
  echo "# These have to come off a cold run. A warm one reports cache hits rather"
  echo "# than compiles and reads as a suite less than half the size of the real"
  echo "# one. The header of tools/test_costs_from_ci.sh has the detail."
} > "$out"
sort -k1,1nr -k2,2 "$rows" >> "$out"

total=$(awk '{s+=$1} END {print s}' "$rows")

# What a warm run looks like from here, which is a suite that appears to have
# got dramatically cheaper without anything having been made faster. A real
# improvement of more than a third between two runs is not something that
# happens to 162 files at once, so it is refused and has to be forced.
if [ -n "$before" ] && [ "$before" -gt 0 ] && [ "$total" -lt $((before * 2 / 3)) ]; then
  if [ -z "${COSTS_ANYWAY:-}" ]; then
    mv "$out.before" "$out"
    echo "run $run gives $total seconds against $before in the table it would have" >&2
    echo "replaced, which is too much cheaper to be real and is what a warm compiler" >&2
    echo "cache looks like from here. The table is left alone. Measure a cold run, or" >&2
    echo "set COSTS_ANYWAY=1 if you have checked that this one is cold." >&2
    exit 1
  fi
  echo "run $run gives $total seconds against $before, kept because COSTS_ANYWAY is set" >&2
fi
rm -f "$out.before"

echo "wrote $out, $unique files, $total seconds in total"
