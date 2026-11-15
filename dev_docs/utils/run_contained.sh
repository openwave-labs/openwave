#!/bin/sh
# Run one command inside a throwaway work directory with very little in reach.
# macOS only (sandbox-exec, deny by default).
#
#   dev_docs/utils/run_contained.sh WORKDIR COMMAND [ARG ...]
#   dev_docs/utils/run_contained.sh --selftest
#
# WORKDIR must be a folder made for this run under the system temp directory,
# already holding a copy of everything the command needs. The command gets:
#   network      denied
#   programs     only the interpreter named by CONTAINED_PATH, files in WORKDIR,
#                and a few shell basics. No `open`, `osascript`, `launchctl`,
#                `curl`, `security` or any other system tool can be started
#   services     no lookup of system services (clipboard, keychain, AppleScript)
#   writes       WORKDIR only, plus the null device
#   reads        the operating system and the interpreter, and WORKDIR. Nothing
#                in a home directory, on another volume, in the temp directories
#                outside WORKDIR, or in the logs, preferences and applications
#   environment  six fixed variables, nothing inherited
#
# What it is not: a virtual machine. The command still talks to the kernel, and
# code that ships its own compiled binary, or drives system libraries directly,
# can reach services this profile does not name (launchd among them). It reduces
# what a script can reach; it does not make an unread script safe. It is the
# last step of the safety gate in PR_REVIEW_STANDARDS.md section 3.1: after the
# screen, after every line that will execute has been read, and after the
# maintainer's go for this execution. For code that cannot be read in full, or
# that carries binaries, use a virtual machine with nothing of value on it.
set -eu
SELF=$(cd "$(dirname "$0")" && pwd -P)/$(basename "$0")
if [ "${1:-}" = "--selftest" ]; then
  T=$(mktemp -d /private/tmp/contained-selftest.XXXXXX) # allow-local-path
  echo sibling > "$T.sibling"
  fail=0
  blocked() { if "$SELF" "$T" /bin/bash -c "$2" >/dev/null 2>&1; then echo "OPEN     $1"; fail=1; else echo "blocked  $1"; fi; }
  works()   { if "$SELF" "$T" /bin/bash -c "$2" >/dev/null 2>&1; then echo "works    $1"; else echo "BROKEN   $1"; fail=1; fi; }
  refused() { if "$SELF" "$2" /bin/ls >/dev/null 2>&1; then echo "ACCEPTED $1"; fail=1; else echo "refused  $1"; fi; }
  works   "write and read inside the work folder" "echo x > inside.txt && /bin/cat inside.txt"
  blocked "network (TCP connect)"              "exec 3<>/dev/tcp/1.1.1.1/53"
  blocked "list the home directories"          "/bin/ls /Users"
  blocked "read a sibling temp file"           "/bin/cat $T.sibling"
  blocked "list the shared temp directory"     "/bin/ls /private/tmp"
  blocked "list the applications"              "/bin/ls /Applications"
  blocked "write outside the work folder"      "echo x > $T.sibling"
  blocked "start a system tool (curl)"         "/usr/bin/curl --version"
  blocked "launch an application (open)"       "/usr/bin/open -a Calculator"
  blocked "read the clipboard"                 "/usr/bin/pbpaste"
  blocked "AppleScript"                        "/usr/bin/osascript -e 'return 1'"
  blocked "keychain tool"                      "/usr/bin/security list-keychains"
  blocked "launchd tool"                       "/bin/launchctl print system"
  refused "a work folder in a system location" "/Applications"
  refused "the shared temp directory itself"   "/private/tmp"
  count=$("$SELF" "$T" /usr/bin/env | /usr/bin/wc -l | /usr/bin/tr -d ' ')
  if [ "$count" -eq 6 ]; then echo "works    environment holds 6 fixed variables"; else echo "OPEN     environment ($count variables)"; fail=1; fi
  if [ "$fail" -eq 0 ]; then echo "selftest: every denial holds"; else echo "selftest: FAILED"; fi
  echo "left for you to delete: $T and $T.sibling"
  exit "$fail"
fi
[ "$#" -ge 2 ] || { echo "usage: $0 WORKDIR COMMAND [ARG ...]   |   $0 --selftest" >&2; exit 64; }
[ "$(uname)" = "Darwin" ] || { echo "macOS only: elsewhere use a container or a virtual machine with the network off" >&2; exit 69; }
[ -d "$1" ] || { echo "refusing: WORKDIR does not exist" >&2; exit 65; }
WORK=$(cd "$1" && pwd -P); shift
case "$WORK" in
  /private/tmp/?*|/private/var/folders/?*/?*/?*/?*) ;; # allow-local-path
  *) echo "refusing: WORKDIR must be a folder made for this run under the system temp directory" >&2; exit 65 ;;
esac
case "$WORK" in *'"'*|*'\'*) echo "refusing: WORKDIR path holds a quote or a backslash" >&2; exit 65 ;; esac
if git -C "$WORK" rev-parse --show-toplevel >/dev/null 2>&1; then
  echo "refusing: WORKDIR is inside a git repository" >&2; exit 65
fi
PREFIX=""
if [ -n "${CONTAINED_PATH:-}" ]; then
  PREFIX=$(cd "$CONTAINED_PATH/.." && pwd -P)
  case "$PREFIX" in /Users/*|/home/*|*'"'*|*'\'*) echo "refusing: the interpreter lives in a home directory, which the sandbox cannot read" >&2; exit 65 ;; esac
fi
mkdir -p "$WORK/.home" "$WORK/.tmp"
PROFILE="(version 1)
(deny default)
(allow process-fork)
(allow process-exec* (subpath \"$WORK\") (literal \"/bin/sh\") (literal \"/bin/bash\") (literal \"/bin/cat\") (literal \"/bin/ls\") (literal \"/usr/bin/env\") (literal \"/usr/bin/wc\") (literal \"/usr/bin/tr\"))
${PREFIX:+(allow process-exec* (subpath \"$PREFIX\"))}
(allow signal (target same-sandbox))
(allow process-info* (target self))
(allow sysctl-read)
(allow ipc-posix-sem)
(allow file-read*)
(deny file-read* (subpath \"/Users\") (subpath \"/Volumes\") (subpath \"/home\")) ; allow-local-path
(deny file-read* (subpath \"/private/tmp\") (subpath \"/private/var/tmp\") (subpath \"/private/var/folders\")) ; allow-local-path
(deny file-read* (subpath \"/private/var/root\") (subpath \"/private/var/log\") (subpath \"/private/var/db/dslocal\")) ; allow-local-path
(deny file-read* (subpath \"/Library/Keychains\") (subpath \"/Library/Preferences\") (subpath \"/Applications\"))
(allow file-read-metadata (literal \"/private/tmp\") (literal \"/private/var/folders\")) ; allow-local-path
(allow file-read* (subpath \"$WORK\"))
(allow file-write* (subpath \"$WORK\") (literal \"/dev/null\"))
(deny network*)"
cd "$WORK"
exec /usr/bin/env -i \
  PATH="${CONTAINED_PATH:+$CONTAINED_PATH:}/usr/bin:/bin" \
  HOME="$WORK/.home" TMPDIR="$WORK/.tmp" LANG=en_US.UTF-8 \
  PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 \
  /usr/bin/sandbox-exec -p "$PROFILE" "$@"
