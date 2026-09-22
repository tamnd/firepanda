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
# `--fast`, or `FIREPANDA_TEST_FAST`, leaves out the expensive files. The
# distribution is lopsided enough that one number separates the suite cleanly:
# fifty five of the hundred and sixty two files cost a minute or more and between
# them they are seventy three per cent of the run, so leaving them out keeps two
# files in three for a quarter of the time. It is for the loop somebody is in
# while they are writing code, and it says so in its own output, because a run
# that skipped a third of the suite is not the thing that decides whether a
# branch is good.
#
# `--changed` leaves out the files that cannot have broken, which is a different
# cut of the same idea and a sharper one. A test file cannot break unless
# something it imports changed, Mojo imports are static, so the set is readable
# straight off the source. `tools/affected.py` works it out and records what it
# is worth: the median library file selects sixteen per cent of the suite, the
# SQL front end files select twelve, and a quarter of the files select under
# three. The other quarter are the ones everything imports and select the lot.
# It is the same loop `--fast` is for, and like `--fast` it is not what decides
# whether a branch is good.
#
# `FIREPANDA_TEST_ONLY` names the files to run outright, which is how a machine
# that has no `.git` to diff gets told what a selection came to.
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
# runs eight of them.
#
# Unsharded, the files go out longest first for the same reason the shards are
# packed that way. Alphabetical order puts `test_sql_run.mojo` near the end and
# the last lane runs it alone while the other seven have nothing left to do.
#
# The split is by measured time, out of `tools/test_costs.txt`, largest first
# into whichever shard has least so far. That is the standard greedy schedule
# and it needs real numbers: the first version of this balanced on file size
# instead, on the reasoning that compile time tracks how much code there is, and
# the ten shards came out between 249 and 662 seconds. Size is not the cost. The
# cost is how much of the library a file's imports pull in and instantiate, and
# two files of the same length disagree about that by a factor of three.
#
# The numbers also have to come off the machine that runs them, which was the
# second thing this got wrong. They were measured on a sixteen core workstation
# on the theory that only the ratios matter, the shards balanced to within half
# a per cent there, and on a four core runner the same assignment came out
# spanning 423 to 622 seconds. `tools/test_costs_from_ci.sh` reads the table
# back out of a CI run's logs, which is where it should come from.
#
# What the shards cannot do anything about is that a shard does not finish until
# its longest file does. On a runner that file is `tests/test_sql_run.mojo` at
# 596 seconds of the 12086 the suite costs, so six shards and twenty shards have
# the same wall clock and the whole job sits a little above a floor set by one
# compile. The count in the workflow is eight, which is the smallest that keeps
# the busiest unfloored shard well clear of that floor: eight puts it at 384
# against 596, and runners differ from each other by about a third.
#
# The obvious way out is to compile the library once and point the test files at
# the result, and it does not work. `mojo precompile firepanda -o
# build/firepanda.mojoc` takes 37 seconds, and on the workstation
# `mojo run -I build tests/test_sql_run.mojo` takes 206 seconds against 193 for
# `mojo run -I .` on the same commit and the same idle machine. All 422 tests
# pass either way, so the package is correct and simply does not save anything:
# what the file pays for is instantiating generics, and those are instantiated
# into whichever program uses them whether or not the package was built first.
# This was measured once before on files costing fourteen and twenty one seconds,
# where there was nothing to see, so it is recorded here against the file that
# actually sets the floor.
#
# Splitting that file is the other obvious way out and it was tried in #933.
# Three parts cost 201, 175 and 185 seconds against 204 for the whole. The
# reason is worth stating exactly, because the obvious reading of it is wrong.
# It is not that each part still imports `firepanda.sql.run`: importing that
# module and calling nothing costs 15 seconds on the workstation, where the file
# costs 195. It is that each part still *calls* `run`, and one call is the whole
# price. Measured by generating a file that imports `run` and calls it n times:
#
#   n = 0    15 s        n = 16   120 s
#   n = 1   124 s        n = 64   120 s
#   n = 4   123 s
#
# The first query costs 109 seconds and every query after it is free, because
# what is being paid for is instantiating the engine at the point of use. So a
# test file is a fixed price for touching a part of the library at all plus
# almost nothing per test, which is why three parts cost three times one part,
# and why a split can only win if a part calls nothing.
#
# The thing that does work is the compiler's own cache, which is restored in
# `.github/workflows/ci.yml`. Warm, the same file takes 22 seconds rather than
# 195, and 61 when the change under test reaches `run` itself.
#
# `FIREPANDA_TEST_WRITE_COSTS` names a file to write the table to, which is what
# `pixi run test-costs` does, and is the fallback when there is no run to read.
# It needs an unsharded run, because a table written from one shard lists a
# eighth of the files. A file the table has not heard of is treated as an average
# one until somebody regenerates it.
#
# Every file's time is printed next to its name in the report below, sharded or
# not, which is what makes reading the table back out of a run's logs possible.
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
# saying so sent someone looking at the wrong thing for an afternoon. A file
# whose compiler was killed is reported the same way and for the same reason:
# on a shared machine the thing that kills it is the kernel reclaiming memory,
# which says nothing at all about the code under test.
#
# `tools/run_tests_remote.sh` runs one shard of this on each of several hosts,
# which is the only lever that works and is why the sharding above exists.

set -uo pipefail

cd "$(dirname "$0")/.."

fast=0
changed=0
for arg in "$@"; do
  case $arg in
    --fast) fast=1 ;;
    --changed) changed=1 ;;
    *)
      echo "unknown argument $arg, expected --fast or --changed" >&2
      exit 2
      ;;
  esac
done
[ -n "${FIREPANDA_TEST_FAST:-}" ] && fast=1
[ -n "${FIREPANDA_TEST_CHANGED:-}" ] && changed=1

# What `--fast` calls expensive. The distribution is lopsided enough that one
# number separates the suite cleanly: 55 of the 162 files are at or above a
# minute and between them they are 73 per cent of the whole run, so leaving them
# out keeps two files in three and costs a quarter of the time.
expensive=60

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

# And cores are not always the binding constraint. A `mojo run` of a test file
# wants somewhere around three quarters of a gigabyte while it is compiling, and
# eight of those on a machine that has already given its memory to something
# else do not run eight times faster, they page. Measured on a laptop with three
# other projects building on it: free memory at 60 megabytes, swap at 7
# gigabytes of 8, forty three million page outs, and `tests/test_sql_run.mojo`
# taking 76 minutes against the 522 seconds the cost table has for it. So the
# width is capped by what there is to run in as well as by what there is to run
# on. A CI runner, which is the case that matters, has its memory to itself and
# never reaches this.
#
# The reading is taken two ways because the two systems this runs on count
# spare memory differently and neither command exists on the other. It was
# macOS only for a while, which was the wrong half: the machine that pages is
# whichever one is shared, and the shared ones here are Linux servers. One of
# them went to a load of 36 with one gigabyte of eleven left while running this,
# and the width it had picked was eight.
spare=
if [ -z "${FIREPANDA_TEST_JOBS:-}" ]; then
  if command -v vm_stat > /dev/null 2>&1; then
    spare=$(vm_stat | awk '
      /page size of/ { for (i = 1; i <= NF; i++) if ($i == "of") size = $(i + 1) }
      /Pages free/ || /Pages inactive/ || /Pages speculative/ { gsub(/\./, "", $NF); pages += $NF }
      END { if (size > 0) print int(pages * size / 1048576) }')
  elif [ -r /proc/meminfo ]; then
    # `MemAvailable` is the kernel's own estimate of what can be handed out
    # without swapping, which is the question being asked. `MemFree` is not: on
    # a server that has been up for months it is near zero and almost all of the
    # difference is page cache that would be given back on demand.
    spare=$(awk '/^MemAvailable:/ { print int($2 / 1024) }' /proc/meminfo)
  fi
fi
if [ -n "${spare:-}" ] && [ "$spare" -gt 0 ] && [ $((spare / 768)) -lt "$jobs" ]; then
  echo "only ${spare}MB of memory is going spare, so $((spare / 768)) at a time rather than $jobs"
  jobs=$((spare / 768))
fi

[ "$jobs" -lt 1 ] && jobs=1

files=(tests/test_*.mojo)
if [ ! -e "${files[0]}" ]; then
  echo "no test files found under tests/" >&2
  exit 1
fi

# `FIREPANDA_TEST_ONLY` names the files to run and nothing else. It is how
# `tools/run_tests_remote.sh` sends a selection to a machine that cannot work
# one out for itself, since the tree it rsyncs over does not carry `.git`.
if [ -n "${FIREPANDA_TEST_ONLY:-}" ]; then
  picked=()
  for name in $FIREPANDA_TEST_ONLY; do
    if [ ! -e "$name" ]; then
      echo "FIREPANDA_TEST_ONLY names $name, which is not here" >&2
      exit 1
    fi
    picked[${#picked[@]}]=$name
  done
  files=("${picked[@]}")
  echo "running the ${#files[@]} test files that were asked for, which is not" \
    "the whole suite and does not decide whether a branch is good"
  changed=0
fi

# `--changed` runs only the files that import something this branch touched.
# `tools/affected.py` explains how much that is worth and when it refuses to
# answer, and when it refuses the list is left alone, which is the whole suite.
#
# The comparison is against the merge base rather than against `main`, so that
# somebody else's commits landing while this branch was open do not select the
# suite. The working tree is asked as well as the branch, because the file being
# edited is the reason for running this at all and is usually not committed.
#
# Nothing here is allowed to fail quietly. A `git` that does not answer, an
# `affected.py` that cannot tell, an empty answer that might be an empty answer
# or might be a broken one, all of them end up running everything.
if [ "$changed" -eq 1 ]; then
  base=$(git merge-base origin/main HEAD 2> /dev/null)
  touched=$(
    {
      [ -n "$base" ] && git diff --name-only "$base" HEAD
      git diff --name-only HEAD
      git ls-files --others --exclude-standard
    } 2> /dev/null | sort -u
  )
  if [ -z "$base" ] || [ -z "$touched" ]; then
    echo "nothing to compare against, so this runs the whole suite"
  # The answer is taken into a variable rather than read off a pipe, because a
  # pipe hands back the exit status of the loop reading it and the exit status
  # is how this says it could not tell.
  elif ! answer=$(python3 tools/affected.py $touched 2> /dev/null); then
    echo "the change reaches further than an import graph can say, so this" \
      "runs the whole suite"
  elif [ -z "$answer" ]; then
    echo "nothing this branch changed reaches a test file"
    exit 0
  else
    picked=()
    while IFS= read -r line; do
      [ -n "$line" ] && picked[${#picked[@]}]=$line
    done <<< "$answer"
    files=("${picked[@]}")
    echo "running the ${#files[@]} test files this branch can reach, which is" \
      "not the whole suite and does not decide whether a branch is good"
  fi
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
              if (line ~ /^#/) continue
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

# `--fast` leaves out the files the cost table calls expensive, and it needs the
# table for the same reason the sharding does: which files those are is a
# measurement and not something worth guessing at. It is not the gate. It says
# so on the way in and it says how many it skipped, because the expensive files
# are the frame arithmetic, the chunk agreement sweep and the SQL planner, which
# is where a mistake in a kernel actually shows up.
left_out=0
if [ "$fast" -eq 1 ]; then
  if [ ! -e "$costs" ]; then
    echo "no cost table at $costs, so --fast has nothing to go on" >&2
    exit 1
  fi
  kept=()
  while IFS= read -r file; do
    kept[${#kept[@]}]=$file
  done < <(printf '%s\n' "${files[@]}" | awk -v costs="$costs" -v limit="$expensive" '
      BEGIN { while ((getline line < costs) > 0) { if (line ~ /^#/) continue
                                                   if (split(line, f, " ") < 2) continue
                                                   cost[f[2]] = f[1] + 0 } }
      # A file the table has never heard of is kept. It is new, it is probably
      # ordinary, and the failure worth avoiding is silently not running it.
      { if (!($0 in cost) || cost[$0] < limit) print }')
  left_out=$(( ${#files[@]} - ${#kept[@]} ))
  files=("${kept[@]}")
fi

# Longest first. The sharding above already sorts, because it has to, but an
# unsharded run went out alphabetically and the spread is wide enough to matter:
# 522 seconds at the top against a median of 36, so a lane that picks the slow
# one up last finishes long after every other lane has run out of work. A file
# the table has not heard of goes last, which is the right guess about a file
# that is more likely to be ordinary than to be one of the fifty one.
if [ "$shards" -eq 1 ] && [ -e "$costs" ]; then
  ordered=()
  while IFS= read -r file; do
    ordered[${#ordered[@]}]=$file
  done < <(printf '%s\n' "${files[@]}" | awk -v costs="$costs" '
      BEGIN { while ((getline line < costs) > 0) { if (line ~ /^#/) continue
                                                   if (split(line, f, " ") < 2) continue
                                                   cost[f[2]] = f[1] + 0 } }
      { printf "%d %s\n", ($0 in cost) ? cost[$0] : 0, $0 }' | sort -k1,1nr -k2,2 | cut -d' ' -f2)
  files=("${ordered[@]}")
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
if [ "$left_out" -ne 0 ]; then
  echo "--fast left out the $left_out files costing ${expensive}s or more, so this run is not the gate"
fi

run_one() {
  local file=$1 logs=$2
  local base=${file##*/}
  # `SECONDS` rather than anything finer because macOS `date` has no `%N`, and
  # one second is plenty: the files run from one second to five minutes and this
  # number is only ever used to decide which shard a file belongs in.
  SECONDS=0
  local status=0
  mojo run -I . "$file" > "$logs/$base.log" 2>&1 || status=$?
  if [ "$status" -eq 0 ]; then
    : > "$logs/$base.ok"
  elif [ "$status" -gt 128 ]; then
    # A status above 128 is a signal, and the signal that turns up here is the
    # kernel killing the compiler because the machine ran out of memory. That is
    # not a test failure and must not be counted as one. See the report below.
    echo "$status" > "$logs/$base.killed"
  fi
  echo "$SECONDS" > "$logs/$base.time"
}
export -f run_one

printf '%s\0' "${files[@]}" \
  | xargs -0 -P "$jobs" -I {} bash -c 'run_one "$1" "$2"' _ {} "$logs"

# Back into filename order to be read. The lanes were handed the files longest
# first, which is a fact about the scheduling rather than something a person
# reading the log wants it sorted by.
reading=()
while IFS= read -r file; do
  reading[${#reading[@]}]=$file
done < <(printf '%s\n' "${files[@]}" | sort)

failed=0
lost=0
for file in "${reading[@]}"; do
  base=${file##*/}
  if [ -e "$logs/$base.time" ]; then
    echo "=== $file ($(cat "$logs/$base.time")s)"
  else
    echo "=== $file"
  fi
  if [ -e "$logs/$base.log" ]; then
    cat "$logs/$base.log"
    if [ -e "$logs/$base.ok" ]; then
      :
    elif [ -e "$logs/$base.killed" ]; then
      # Counted with the lost files rather than with the failures. A file whose
      # compiler was killed says nothing about the code, and calling it a
      # failure sends whoever reads the report looking for a bug that is not
      # there. This turned up on a shared server with seven gigabytes going
      # spare and six files compiling at once, where five of the first thirty
      # one were killed and every one of them passed on its own afterwards.
      # The width backs off from what is spare, which makes this rarer, but the
      # memory can go away after the width is picked.
      echo "the compiler was killed by signal $(($(cat "$logs/$base.killed") - 128)), so this file did not run to a result"
      lost=$((lost + 1))
    else
      failed=$((failed + 1))
    fi
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
  # Same reasoning as the shard refusal. A table written from a `--fast` run
  # lists only the cheap files, and the expensive ones, which are the whole
  # reason the sharding needs a table, would all fall to the mean.
  if [ "$fast" -eq 1 ]; then
    echo "refusing to write a cost table from a --fast run" >&2
    exit 1
  fi
  # The header is written first and only the rows go through `sort`, because a
  # header sorted along with the data lands in the middle of the file. The
  # reader above skips a comment wherever it appears, so this is about the file
  # being readable rather than about it working.
  {
    echo "# Seconds per test file, written by \`pixi run test-costs\` on $(uname -m),"
    echo "# $jobs files at a time. The ten test shards in .github/workflows/ci.yml"
    echo "# divide the list up on these numbers."
    echo "#"
    echo "# Prefer \`tools/test_costs_from_ci.sh\` when there is a CI run to read."
    echo "# The ratios between files are not the same on sixteen cores as on four,"
    echo "# so a table measured off the runner balances the runner badly."
  } > "$FIREPANDA_TEST_WRITE_COSTS"
  for file in "${files[@]}"; do
    base=${file##*/}
    [ -e "$logs/$base.time" ] || continue
    printf '%s %s\n' "$(cat "$logs/$base.time")" "$file"
  done | sort -k1,1nr -k2,2 >> "$FIREPANDA_TEST_WRITE_COSTS"
  echo "wrote $FIREPANDA_TEST_WRITE_COSTS"
fi

echo
if [ "$lost" -ne 0 ]; then
  echo "$lost of ${#files[@]} test files did not run to a result, so this run says nothing"
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
