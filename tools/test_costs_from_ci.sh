#!/usr/bin/env bash
# Rebuilds `tools/test_costs.txt` from a CI run's shard logs.
#
# The table is what the ten test shards in `.github/workflows/ci.yml` divide the
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
# The argument is a run id, which `gh run list --workflow CI` prints. Any run
# whose test shards all succeeded will do; a run with a failed shard is missing
# that shard's files and is refused below rather than silently writing a short
# table.

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

{
  echo "# Seconds per test file, written by \`tools/test_costs_from_ci.sh\` out of"
  echo "# run $run. The ten test shards in .github/workflows/ci.yml divide the"
  echo "# list up on these numbers."
  echo "#"
  echo "# Measured on the CI runner itself, four cores running four files at a"
  echo "# time, which is what these numbers have to come from. A workstation's do"
  echo "# not transfer: the ratios between files are not the same on sixteen cores"
  echo "# as on four, and a table measured on one balanced the shards to within"
  echo "# half a per cent there and to 423 against 622 seconds here."
} > "$out"
sort -k1,1nr -k2,2 "$rows" >> "$out"

echo "wrote $out, $unique files, $(awk '{s+=$1} END {print s}' "$rows") seconds in total"
