"""Find this room's own background python process running item1.py (by argv) - macOS sysctl."""
import ctypes, ctypes.util, os, signal, sys
libc = ctypes.CDLL(ctypes.util.find_library("c"))
CTL_KERN, KERN_ARGMAX, KERN_PROCARGS2 = 1, 8, 49


def argv_of(pid):
    mib = (ctypes.c_int * 3)(CTL_KERN, KERN_PROCARGS2, pid)
    size = ctypes.c_size_t(0)
    if libc.sysctl(mib, 3, None, ctypes.byref(size), None, 0) != 0:
        return None
    buf = ctypes.create_string_buffer(size.value)
    if libc.sysctl(mib, 3, buf, ctypes.byref(size), None, 0) != 0:
        return None
    raw = buf.raw[4:size.value]
    parts = [p for p in raw.split(b"\0") if p]
    return [p.decode(errors="replace") for p in parts[:4]]   # executable + argv only


me = os.getpid()
hits = []
for pid in range(1, 100000):
    if pid == me:
        continue
    try:
        os.kill(pid, 0)
    except (ProcessLookupError, PermissionError, OSError):
        continue
    a = argv_of(pid)
    if a and any(x.endswith("item1.py") for x in a):
        hits.append((pid, a))
print(hits)
if "--kill" in sys.argv and len(hits) == 1:
    os.kill(hits[0][0], signal.SIGTERM)
    print("sent SIGTERM to", hits[0][0])
