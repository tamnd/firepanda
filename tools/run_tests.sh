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
#
# `FIREPANDA_TEST_SHARDS` and `FIREPANDA_TEST_SHARD` cut the list of files into
# pieces so that several machines can each run one. One machine with four cores
# was taking thirty nine minutes of the fifty one this step's job cost, and the
# reason is that a test file spends most of its wall clock compiling the library
# again rather than running assertions, so there is nothing to share between
# files and the only way to go faster is more machines. `.github/workflows/ci.yml`
# runs ten of them.
#
# The split is by measured time, out of `tools/test_costs.txt`, largest first
# into whichever shard has least so far. That is the standard greedy schedule
# and it needs real numbers: the first version of this balanced on file size
# instead, on the reasoning that compile time tracks how much code there is, and
# the ten shards came out between 249 and 662 seconds. Size is not the cost. The
# cost is how much of the library a file's imports pull in and instantiate, and
# two files of the same length disagree about that by a factor of three.
#
# `FIREPANDA_TEST_WRITE_COSTS` names a file to write the table to, which is what
# `pixi run test-costs` does. It needs an unsharded run, because a table written
# from one shard lists a tenth of the files. A file the table has not heard of
# is treated as an average one until somebody regenerates it.
#
# The logs go under `build/` rather than under `TMPDIR`, which they used to. On
# macOS `TMPDIR` is a per-session directory under `/var/folders` that the system
# is free to reap, and it does: a run of this script reported twenty four of a
# hundred and fifty five files failed with every one of those twenty four saying
# only that its log did not exist, and the next run reported all hundred and
# fifty five the same way, with the tests themselves passing when run one at a
# time straight afterwards. A tally that says a test failed when the test passed
# is worse than no tally, so the logs now live somewhere nothing else prunes.
#
# A missing log is also reported as what it is. It is not a test failure and
# saying so sent someone looking at the wrong thing for an afternoon.

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

shards=${FIREPANDA_TEST_SHARDS:-1}
shard=${FIREPANDA_TEST_SHARD:-1}
if [ "$shards" -lt 1 ] || [ "$shard" -lt 1 ] || [ "$shard" -gt "$shards" ]; then
  echo "FIREPANDA_TEST_SHARD=$shard is not between 1 and $shards" >&2
  exit 1
fi

costs=${FIREPANDA_TEST_COSTS:-tools/test_costs.txt}

total=${#files[@]}
if [ "$shards" -gt 1 ]; then
  if [ ! -e "$costs" ]; then
    echo "no cost table at $costs, so the shards cannot be balanced" >&2
    exit 1
  fi
  # Every shard runs this over the whole list and keeps the files that fall to
  # it, so they agree on the assignment without talking to each other.
  #
  # `read` in a loop rather than `mapfile`, because the macOS runner's bash is
  # 3.2 and does not have it.
  picked=()
  while IFS= read -r file; do
    picked[${#picked[@]}]=$file
  done < <(
    printf '%s\n' "${files[@]}" \
      | awk -v costs="$costs" '
          BEGIN {
            while ((getline line < costs) > 0) {
              if (split(line, f, " ") < 2) continue
              cost[f[2]] = f[1] + 0
              sum += f[1] + 0
              seen++
            }
            # A file the table has never heard of is assumed to be an average
            # one, which is the least wrong thing to assume about a file that
            # was added after the table was written.
            mean = (seen > 0) ? sum / seen : 0
          }
          {
            printf "%d %s\n", (($0 in cost) ? cost[$0] : mean), $0
          }' \
      | sort -k1,1nr -k2,2 \
      | awk -v n="$shards" -v mine="$shard" '
          {
            # Ties go to the shard holding fewest files, which makes this
            # degrade to round robin rather than to "everything in shard one"
            # if the table ever comes back all zeroes.
            least = 1
            for (s = 2; s <= n; s++) {
              if (load[s] < load[least]) least = s
              else if (load[s] == load[least] && count[s] < count[least]) least = s
            }
            load[least] += $1
            count[least]++
            if (least == mine) print $2
          }'
  )
  if [ "${#picked[@]}" -eq 0 ]; then
    echo "shard $shard of $shards has no files out of $total, which cannot be right" >&2
    exit 1
  fi
  files=("${picked[@]}")
fi

# The process id is in the name because two runs of this in the same checkout at
# once is a normal thing to want and they must not share a log directory.
logs=build/testlogs.$$
rm -rf "$logs"
mkdir -p "$logs"
trap 'rm -rf "$logs"' EXIT

if [ "$shards" -gt 1 ]; then
  echo "running ${#files[@]} of $total test files, shard $shard of $shards, $jobs at a time"
else
  echo "running ${#files[@]} test files, $jobs at a time"
fi

run_one() {
  local file=$1 logs=$2
  local base=${file##*/}
  # `SECONDS` rather than anything finer because macOS `date` has no `%N`, and
  # one second is plenty: the files run from three seconds to a minute and this
  # number is only ever used to decide which shard a file belongs in.
  SECONDS=0
  if mojo run -I . "$file" > "$logs/$base.log" 2>&1; then
    : > "$logs/$base.ok"
  fi
  echo "$SECONDS" > "$logs/$base.time"
}
export -f run_one

printf '%s\0' "${files[@]}" \
  | xargs -0 -P "$jobs" -I {} bash -c 'run_one "$1" "$2"' _ {} "$logs"

failed=0
lost=0
for file in "${files[@]}"; do
  base=${file##*/}
  if [ -e "$logs/$base.time" ]; then
    echo "=== $file ($(cat "$logs/$base.time")s)"
  else
    echo "=== $file"
  fi
  if [ -e "$logs/$base.log" ]; then
    cat "$logs/$base.log"
    [ -e "$logs/$base.ok" ] || failed=$((failed + 1))
  else
    echo "no output was captured for this file, so it did not run to a result"
    lost=$((lost + 1))
  fi
done

# The cost table the sharding above reads. Writing it is opt in and takes a
# whole unsharded run, because a table written from one shard would list a tenth
# of the files and the other nine tenths would fall to the mean on the next run
# and undo the balance. `pixi run test-costs` is the way to do it.
if [ -n "${FIREPANDA_TEST_WRITE_COSTS:-}" ]; then
  if [ "$shards" -gt 1 ]; then
    echo "refusing to write a cost table from shard $shard of $shards" >&2
    exit 1
  fi
  {
    for file in "${files[@]}"; do
      base=${file##*/}
      [ -e "$logs/$base.time" ] || continue
      printf '%s %s\n' "$(cat "$logs/$base.time")" "$file"
    done
  } | sort -k1,1nr -k2,2 > "$FIREPANDA_TEST_WRITE_COSTS"
  echo "wrote $FIREPANDA_TEST_WRITE_COSTS"
fi

echo
if [ "$lost" -ne 0 ]; then
  echo "$lost of ${#files[@]} test files produced no log, so this run says nothing"
  echo "the log directory was $logs"
  exit 1
fi
if [ "$failed" -ne 0 ]; then
  echo "$failed of ${#files[@]} test files failed"
  exit 1
fi
if [ "$shards" -gt 1 ]; then
  echo "${#files[@]} test files passed, shard $shard of $shards"
else
  echo "${#files[@]} test files passed"
fi
