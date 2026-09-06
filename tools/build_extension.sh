#!/usr/bin/env bash
#
# Builds the CPython extension and puts everything it needs beside it.
#
# The output directory is self contained. Nothing in it points at the pixi
# environment it was built in, so it can be copied to a machine that has never
# had a Mojo toolchain and imported there. That claim is the whole of M3's
# distribution problem and `python/tests/test_extension.py` is what checks it did
# not quietly stop being true.
#
# Usage: tools/build_extension.sh [output directory]
#
# The default output is build/extension. Run it through pixi, because `mojo`
# needs the environment and `mojo build` refuses to run outside a pixi project.

set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
out="${1:-$root/build/extension}"

# Where the toolchain keeps its shared libraries. `mojo` is on the path inside
# the pixi environment and the libraries sit one directory up from the binary,
# which is a more reliable way to find them than asking pixi for its prefix.
mojo_bin="$(command -v mojo)"
lib="$(cd "$(dirname "$mojo_bin")/../lib" && pwd)"

case "$(uname -s)" in
  Darwin) suffix="dylib" ;;
  Linux)
    suffix="so"
    # Both of these are checked up front rather than where they are used. Not
    # having patchelf fails as `command not found` from inside a function several
    # steps after the build succeeded, and not having ldconfig is worse than
    # that: the system library test below would answer no every time and the
    # script would cheerfully vendor libc.
    command -v patchelf > /dev/null || {
      echo "patchelf is needed to relocate the runtime libraries on Linux" >&2
      echo "install it with: apt-get install -y patchelf" >&2
      exit 1
    }
    # `/sbin` is on root's path and not always on everyone else's, so look there
    # by hand rather than trusting PATH.
    ldconfig_bin="$(command -v ldconfig || echo /sbin/ldconfig)"
    [ -x "$ldconfig_bin" ] || {
      echo "ldconfig is needed to tell a system library from a Mojo one" >&2
      exit 1
    }
    # Once, rather than once per candidate library. This is a few thousand lines.
    system_libraries="$("$ldconfig_bin" -p)"
    ;;
  *) echo "unsupported platform: $(uname -s)" >&2; exit 1 ;;
esac

# The libraries a binary needs that the machine it lands on will not already
# have. Bare file names, so the rest of this script does not care which platform
# it is on, but the two implementations decide membership differently and the
# difference matters.
#
# On macOS the reference itself says. The Mojo runtime libraries name each other
# as `@rpath/...` and name the system C++ library as `/usr/lib/libc++.1.dylib`,
# an absolute path, so following only the loader relative ones is exactly right.
# The first version of this matched on file name alone and vendored the pixi
# environment's own copy of libc++, which is 1.17 MB of a library the host
# process already has loaded, and two libc++ in one process is the kind of
# mistake that ends in a crash rather than in an error message.
#
# On Linux an ELF `NEEDED` entry is a bare name either way, so there is nothing
# in the reference to read. The rule there is that a library the system loader
# can already find is the system's, and only what is left is ours.
needed() {
  case "$(uname -s)" in
    Darwin)
      otool -L "$1" | tail -n +2 | awk '{print $1}' \
        | grep -E '^@(rpath|loader_path)/' | xargs -n1 basename
      ;;
    Linux)
      objdump -p "$1" | awk '/NEEDED/ {print $2}' | while read -r name; do
        # `ldconfig -p` prints one indented `name (flags) => path` per line, so
        # anchoring on the leading whitespace and the trailing space is what
        # keeps `libc.so.6` from matching `libcrypto.so.6`.
        printf '%s\n' "$system_libraries" \
          | grep -qE "^[[:space:]]+$name " || echo "$name"
      done
      ;;
  esac
}

# Throw away the local symbol names, which are most of the file.
#
# A Mojo generic is compiled once per set of type arguments and the name of each
# copy carries those arguments in full, spelled as MLIR types. One of them in the
# extension is over four thousand characters long, and there are thousands of
# them, because every dtype pair that a kernel is instantiated for gets its own.
# Measured on the build that first reached the arithmetic kernels from Python:
# 7,858,352 bytes, of which the code was 2,998,272 and the symbol table was
# 4,849,664. Discarding the local names takes the file to 3,064,128, which is 61
# percent off for nothing given up that a shipped artefact uses.
#
# What is given up is function names in a crash trace, which for a release build
# is the ordinary trade and is not the way anybody debugs this: a developer has
# the unstripped file in `build/` from their own `mojo build` and it is that copy
# they run under a debugger.
#
# Only the extension is stripped. The four runtime libraries come out of the
# toolchain already stripped, so the same call takes nothing off them and the
# signature it forces a fresh copy of costs twelve kilobytes each, which is a net
# loss of fifty one kilobytes for no gain at all.
shrink() {
  command -v strip > /dev/null || {
    echo "no strip on PATH, leaving the symbol table in" >&2
    return 0
  }

  echo "discarding local symbols"
  # Local symbols only. The dynamic symbol table is what the loader reads and
  # `PyInit__firepanda` lives in it, so `-x` is the whole of what is safe here
  # and `-s` would take the entry point out and leave a file Python cannot import.
  strip -x "$1"

  # Stripping edits the file and therefore invalidates its signature, with the
  # SIGKILL consequence the codesign comment below describes. `relocate` signs
  # this same file again a moment later and that is fine: `--force` replaces, two
  # signatures never coexist, and the file is the same size either way. Signing
  # here anyway is what keeps this function correct on its own rather than
  # correct because of what happens to run after it.
  case "$(uname -s)" in
    Darwin) codesign --force --sign - "$1" ;;
  esac
}

# Point a binary at the directory it is sitting in, and at nothing else.
#
# Only absolute rpaths are removed. The four runtime libraries carry several
# rpaths each, but every one of them is already `@loader_path` relative, mostly
# Bazel leftovers naming directories that do not exist in a pixi environment and
# do no harm; the extension is the only file with an absolute rpath, and it is
# the absolute path of the pixi environment it was built in. Deleting the
# harmless ones as well was the first version of this and it was worse than a
# waste, for the reason in the codesign comment below.
relocate() {
  local edited=0

  case "$(uname -s)" in
    Darwin)
      while read -r path; do
        case "$path" in /*) ;; *) continue ;; esac
        install_name_tool -delete_rpath "$path" "$1"
        edited=1
      done < <(otool -l "$1" \
        | awk '/LC_RPATH/ {found=1} found && /path / {print $2; found=0}')

      if ! otool -l "$1" | grep -q '@loader_path'; then
        install_name_tool -add_rpath "@loader_path" "$1"
        edited=1
      fi

      # Every binary on Apple silicon carries a signature, at minimum an ad hoc
      # one, and editing load commands invalidates it. The loader's answer to an
      # invalid signature is to kill the process: no ImportError, no traceback,
      # nothing on any stream, just SIGKILL and exit status 137 from a `python -c
      # "import firepanda"` that printed absolutely nothing. It is the least
      # informative failure in this milestone, and it is why the rule above is to
      # touch only the files that have to be touched.
      if [ "$edited" = 1 ]; then
        codesign --force --sign - "$1"
      fi
      ;;
    Linux)
      patchelf --set-rpath '$ORIGIN' "$1"
      ;;
  esac
}

rm -rf "$out"
mkdir -p "$out"

echo "building the extension"
mojo build --emit shared-lib -I "$root" \
  -o "$out/_firepanda.so" "$root/firepanda/py/module.mojo"

# Copy what the extension asks for, then what those ask for, until nothing new
# turns up. Which libraries the Mojo runtime is made of is not a fact this script
# should carry, because it is a fact about the toolchain and it will change.
echo "vendoring the runtime"
pending=("$out/_firepanda.so")
while [ ${#pending[@]} -gt 0 ]; do
  current="${pending[0]}"
  pending=("${pending[@]:1}")
  while read -r name; do
    [ -n "$name" ] || continue
    case "$name" in *".$suffix") ;; *) continue ;; esac
    [ -e "$out/$name" ] && continue
    [ -e "$lib/$name" ] || continue
    cp "$lib/$name" "$out/$name"
    chmod u+w "$out/$name"
    pending+=("$out/$name")
  done < <(needed "$current")
done

shrink "$out/_firepanda.so"

for file in "$out"/*; do
  relocate "$file"
done

echo
echo "$out"
ls -l "$out" | awk 'NR > 1 {printf "  %9d  %s\n", $5, $NF}'
echo
case "$(uname -s)" in
  Darwin) total=$(find "$out" -type f -exec stat -f %z {} + | paste -sd+ - | bc) ;;
  Linux)  total=$(find "$out" -type f -exec stat -c %s {} + | paste -sd+ - | bc) ;;
esac
echo "  total $total bytes"
