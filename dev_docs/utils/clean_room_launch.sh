#!/bin/sh
# Headless clean-room launch (dev_docs/CLEAN_ROOM_STANDARDS.md § 3.4). macOS only: needs sandbox-exec.
#
# Usage: clean_room_launch.sh <room-dir> <opening-prompt-file> <transcript.jsonl> [model]
# Env:   CLEAN_ROOM_PYTHON      interpreter for the room (default: python3 on PATH)
#        CLEAN_ROOM_PACKAGE     the platform package (default: openwave), used both as the module name
#                               that must NOT import from the room and as the name that must NOT appear
#                               in the interpreter's environment
#        CLEAN_ROOM_SETUP_ONLY  set to 1 to build ./py, run every probe, and stop before the session
#
# The room holds only the audited packet. The session loads no instruction file, memory, MCP server
# or skill, has four tools, denies anything that would prompt, and runs code only through ./py.
#
# ./py starts the interpreter with a fixed environment under a sandbox that denies by default:
#   network   refused
#   programs  only the interpreter itself can be started (a script may run sys.executable again);
#             no shell, no system tool, nothing else in the interpreter's install
#   services  no system service can be looked up, so no clipboard, keychain or application launch
#   reads     the system's libraries and frameworks, the interpreter's install, and the room
#   writes    the room, except ./py and .room.sb themselves
#   clock     UTC, whatever the machine's zone
# ./py and .room.sb are also locked with the user-immutable flag, so the session's file tools
# cannot change them either. They are hashed before the session and checked after it.
# Each denial is probed below before the session starts. The outside-read, outside-write,
# program, service, network and wrapper-write probes each have a control that must succeed with
# no sandbox, so that such a probe cannot pass for a reason other than the sandbox. The others
# only check that the action is refused.
#
# What stays visible to a room script: the names at the top of the file system, whether a given
# path exists, the machine's host name and kernel version, and the interpreter's install, which
# means the NAMES of every package installed there. Files in the install that record where a
# package came from are closed. So the interpreter comes from an environment kept for rooms alone
# (§ 3.2). The launch aborts if the platform's name is in that environment or if any package there
# was installed from a path on this machine; what else the environment holds is the operator's to keep.
#
# Teardown: the lock must come off before the room can be deleted:
#   chflags nouchg <room-dir>/py <room-dir>/.room.sb
set -eu
abort() { printf 'ABORT: %s\n' "$1"; exit 2; }
[ $# -ge 3 ] || { echo "usage: $0 <room-dir> <opening-prompt-file> <transcript.jsonl> [model]"; exit 64; }
[ "$(uname)" = "Darwin" ] || abort "macOS only: the room interpreter needs sandbox-exec"
[ -d "$1" ] || abort "the room directory does not exist: $1"
[ -f "$2" ] || abort "the opening prompt file does not exist: $2"
ROOM=$(cd "$1" && pwd -P)
PROMPT=$(cd "$(dirname "$2")" && pwd -P)/$(basename "$2")
[ -d "$(dirname "$3")" ] || abort "the transcript's folder does not exist: $(dirname "$3")"
OUT=$(cd "$(dirname "$3")" && pwd -P)/$(basename "$3")
MODEL=${4:-opus}
PY=$(command -v "${CLEAN_ROOM_PYTHON:-python3}") || abort "the interpreter was not found: ${CLEAN_ROOM_PYTHON:-python3}"
PKG=${CLEAN_ROOM_PACKAGE:-openwave}
PYREAL=$("$PY" -c "import os, sys; print(os.path.realpath(sys.executable))") || abort "the interpreter does not start: $PY"
PREFIX=$("$PY" -c "import sys; print(sys.base_prefix)") || abort "the interpreter did not report its install folder"
SITE=$("$PY" -c "import sysconfig; print(sysconfig.get_paths()['purelib'])") || abort "the interpreter did not report its package folder"
case "$OUT" in "$ROOM"/*) abort "the transcript must be written outside the room: $OUT" ;; esac
# the package folder of the install the interpreter was built on: the same folder as SITE, except
# for a virtual environment, where it is the shared install underneath, which the room can read
BASESITE=$("$PY" -c "import sys, sysconfig; print(sysconfig.get_path('purelib', vars={'base': sys.base_prefix, 'platbase': sys.base_prefix}))") || abort "the interpreter did not report its base package folder"
for p in "$ROOM" "$PY" "$PYREAL" "$PREFIX" "$SITE" "$BASESITE"; do
  case "$p" in
    *'"'* | *'\'* | *'`'* | *'$'*) abort "a path holds a quote, backslash, backtick or dollar sign, which would break the room interpreter" ;;
  esac
done

# § 3.1: no agent-instruction file anywhere on the room's ancestor path
d=$ROOM
while [ "$d" != "/" ]; do
  for f in "$d/CLAUDE.md" "$d/.claude/CLAUDE.md"; do
    [ -e "$f" ] && abort "instruction file on the room's ancestor path: $f"
  done
  d=$(dirname "$d")
done

# The folders above the room and the interpreter may be looked at (not listed, not read), so that
# their paths resolve.
ancestors() {
  p=$1
  while [ "$p" != "/" ]; do
    p=$(dirname "$p")
    [ "$p" = "/" ] || printf ' (literal "%s")' "$p"
  done
}
META="$(ancestors "$ROOM")$(ancestors "$PY")$(ancestors "$PYREAL")$(ancestors "$PREFIX")$(ancestors "$SITE")"

# a re-launch into the same room rewrites the two files, so the lock comes off first
chflags nouchg "$ROOM/py" "$ROOM/.room.sb" 2>/dev/null || true
chmod u+w "$ROOM/py" "$ROOM/.room.sb" 2>/dev/null || true

cat > "$ROOM/.room.sb" <<EOF
(version 1)
(deny default)
(allow process-fork)
(allow process-exec* (literal "$PYREAL") (literal "$PY"))
(allow signal (target same-sandbox))
(allow process-info* (target self))
(allow sysctl-read)
(allow ipc-posix-sem)
(allow file-read-metadata (literal "/")$META)
(allow file-read* (subpath "/System") (subpath "/usr/lib") (subpath "/usr/share") (subpath "/Library/Apple"))
(allow file-read* (literal "/") (literal "/dev/null") (literal "/dev/zero") (literal "/dev/random") (literal "/dev/urandom") (literal "/dev/dtracehelper"))
(deny file-read* (subpath "/System/Volumes/Data"))
(allow file-read* (subpath "$PREFIX") (subpath "$SITE") (subpath "$ROOM"))
(deny file-read* (subpath "$PREFIX/conda-meta") (regex #"/__editable__[^/]*\$") (regex #"\.dist-info/direct_url\.json\$"))
(allow file-write* (subpath "$ROOM") (literal "/dev/null"))
(deny file-write* (literal "$ROOM/py") (literal "$ROOM/.room.sb"))
(deny network*)
EOF

cat > "$ROOM/py" <<EOF
#!/bin/sh
# room interpreter: fixed environment, OS-sandboxed to this room, no site processing, one thread, low priority
exec /usr/bin/env -i PATH=/usr/bin:/bin HOME=/var/empty USER=room LOGNAME=room LANG=en_US.UTF-8 TZ=UTC \\
  PYTHONPATH="$SITE" OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \\
  VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \\
  /usr/bin/nice -n 15 /usr/bin/sandbox-exec -f "$ROOM/.room.sb" "$PY" -S "\$@"
EOF
chmod 755 "$ROOM/py"
seal() { cat "$ROOM/py" "$ROOM/.room.sb" | shasum -a 256 | cut -d' ' -f1; }
SEAL=$(seal)

# § 3.2: the platform package must not import from the room, the interpreter's environment must
# be one kept for rooms, and every denial must hold.
# A probe that passes says the action was refused. Its control runs the same action with no
# sandbox and must succeed.
cd "$ROOM"
inside() { ./py -c "$1" >/dev/null 2>&1; }
outside() { "$PY" -S -c "$1" >/dev/null 2>&1; }

inside "pass" || abort "the room interpreter does not start under the sandbox (an interpreter that is a link into a closed folder cannot be used)"
inside "import $PKG" && abort "$PKG is importable from the room"
# The environment itself. A room script can list the interpreter's package folder and, under a
# virtual environment, the shared one beneath it, so what is installed there is visible by name.
# First: the platform's name must not be there, as an install record under any name, as a module
# an install record declares, or as the name of any entry in those folders or in the folders that
# hold the interpreter's commands. The check reads names at the top of each folder and the install
# records; it does not read inside packages.
# Exit 0: not there. Exit 10: there. Anything else: the check did not run, which also aborts.
rc=0
"$PY" -I -S -c "
import importlib.metadata as m, os, re, sys
norm = lambda s: re.sub('[-_.]+', '_', s.lower())
key = norm(sys.argv[1])
found = False
for site in sorted(set(sys.argv[2:])):
    if not os.path.isdir(site):
        continue
    found = found or any(key in norm(e) for e in os.listdir(site))
    for d in m.distributions(path=[site]):
        tops = [norm(t) for t in (d.read_text('top_level.txt') or '').split()]
        found = found or norm(d.metadata.get('Name') or '') == key or key in tops
sys.exit(10 if found else 0)
" "$PKG" "$SITE" "$BASESITE" "$(dirname "$PY")" "$PREFIX/bin" >/dev/null 2>&1 || rc=$?
[ "$rc" = "10" ] && abort "$PKG is present in the room interpreter's environment, so its name is visible to the room (§ 3.2). Set CLEAN_ROOM_PYTHON to an environment kept for rooms"
[ "$rc" = "0" ] || abort "the check for $PKG in the interpreter's environment did not run (exit $rc)"
# Second: no package installed from a path on this machine, which is how a project under
# development gets into an environment. An install record counts when it names a local path that
# exists here, or cannot be read; a record that names the folder a package was built in on
# another machine does not. Old-style link files count.
LINKED=$("$PY" -I -S -c "
import json, os, sys
from urllib.parse import unquote, urlparse
n = 0
for site in sorted(set(sys.argv[1:])):
    if not os.path.isdir(site):
        continue
    for e in os.listdir(site):
        f = os.path.join(site, e, 'direct_url.json')
        if e.endswith('.egg-link'):
            n += 1
        if not (e.endswith('.dist-info') and os.path.exists(f)):
            continue
        try:
            with open(f) as h:
                url = json.load(h)['url']
            if not isinstance(url, str):
                raise TypeError
            u = urlparse(url)
            local =u.scheme.lower() == 'file' and u.netloc in ('', 'localhost') and os.path.exists(unquote(u.path))
        except Exception:
            local = True
        n += 1 if local else 0
print(n)
" "$SITE" "$BASESITE") || abort "the count of locally installed packages did not run"
[ "$LINKED" = "0" ] || abort "$LINKED package(s) in the room interpreter's environment were installed from a path on this machine, so the environment is not kept for rooms alone and their names are visible to the room (§ 3.2). Set CLEAN_ROOM_PYTHON to an environment kept for rooms"

outside "import os; os.listdir('$HOME')" || abort "the outside-read probe fails even unsandboxed, so it cannot test the sandbox"
inside "import os; os.listdir('$HOME')" && abort "the sandbox did not refuse an outside read"
inside "import os; os.listdir('/private/etc')" && abort "the sandbox did not refuse a read of system configuration" # allow-local-path

PROBE="$ROOM/../.clean_room_write_probe"
outside "import os; open('$PROBE', 'w').close(); os.remove('$PROBE')" || abort "the outside-write probe fails even unsandboxed (the room's parent folder is not writable), so it cannot test the sandbox"
inside "open('$PROBE', 'w')" && abort "the sandbox did not refuse a write outside the room"
inside "import os; open('.clean_room_write_probe', 'w').write('x'); os.remove('.clean_room_write_probe')" || abort "the room interpreter cannot write inside the room"
# the two files are still unlocked here, so these two probes test the sandbox rule alone
outside "open('$ROOM/py', 'a').close(); open('$ROOM/.room.sb', 'a').close()" || abort "the wrapper-write probe fails even unsandboxed, so it cannot test the sandbox"
inside "open('py', 'a')" && abort "the sandbox did not refuse a write to ./py"
inside "open('.room.sb', 'a')" && abort "the sandbox did not refuse a write to .room.sb"
# now the lock, which is what stops the session's own file tools
chmod 555 "$ROOM/py"
chmod 444 "$ROOM/.room.sb"
chflags uchg "$ROOM/py" "$ROOM/.room.sb"
(printf '' >> "$ROOM/py") 2>/dev/null && abort "./py is not locked against the session's file tools"
(printf '' >> "$ROOM/.room.sb") 2>/dev/null && abort ".room.sb is not locked against the session's file tools"

outside "import subprocess; subprocess.run(['/usr/bin/true'], check=True)" || abort "the program probe fails even unsandboxed, so it cannot test the sandbox"
inside "import subprocess; subprocess.run(['/usr/bin/true'], check=True)" && abort "the sandbox did not refuse to start another program"
inside "import subprocess, sys; subprocess.run([sys.executable, '-S', '-c', 'pass'], check=True)" || abort "a room script cannot start its own interpreter again"

outside "import os, pwd; pwd.getpwuid(os.getuid())" || abort "the service probe fails even unsandboxed, so it cannot test the sandbox"
inside "import os, pwd; pwd.getpwuid(os.getuid())" && abort "the sandbox did not refuse a system-service lookup"

NET="import socket; socket.create_connection(('1.1.1.1', 53), timeout=3).close()"
if outside "$NET"; then
  inside "$NET" && abort "the sandbox did not refuse a network connection"
else
  echo "note: the network probe has no control (this machine has no route out); the profile denies network"
fi

EXTRA=$(./py -c "
import os
known = {'PATH', 'HOME', 'USER', 'LOGNAME', 'LANG', 'TZ', 'PYTHONPATH', 'OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
         'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS', '__CF_USER_TEXT_ENCODING'}
print(' '.join(sorted(set(os.environ) - known)))
") || abort "the environment probe did not run"
[ -z "$EXTRA" ] || abort "the room interpreter inherited environment variables: $EXTRA"

inside "import time, sys; sys.exit(0 if time.tzname[0] == 'UTC' else 1)" || abort "the room clock is not UTC"
inside "import numpy" || echo "note: numpy does not import in the room"
[ "$(seal)" = "$SEAL" ] || abort "./py or .room.sb changed during the probes"

if [ "${CLEAN_ROOM_SETUP_ONLY:-0}" = "1" ]; then
  echo "setup only: ./py is built and locked, and every probe passed; no session was started"
  echo "teardown: chflags nouchg \"$ROOM/py\" \"$ROOM/.room.sb\""
  exit 0
fi

claude -p "$(cat "$PROMPT")" --model "$MODEL" --restricted --tools "Read,Edit,Write,Bash" \
  --strict-mcp-config --disable-slash-commands --permission-prompts none \
  --allowedTools "Edit(./**)" "Write(./**)" "Bash(./py *)" \
  --no-session-persistence --output-format stream-json --verbose < /dev/null > "$OUT"
echo "done: transcript at $OUT"
if [ "$(seal)" = "$SEAL" ]; then
  echo "room interpreter unchanged: sha256 $SEAL over ./py and .room.sb"
else
  echo "WARNING: ./py or .room.sb changed during the session. The run's code may have run unsandboxed: treat the room as void."
  exit 3
fi
echo "teardown: chflags nouchg \"$ROOM/py\" \"$ROOM/.room.sb\""
