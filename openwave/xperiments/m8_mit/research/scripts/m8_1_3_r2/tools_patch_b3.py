"""one-off edit helper: replaces the main-loop tail of sB3.py (kept for the record)."""
import os
HERE = os.path.dirname(os.path.abspath(__file__))
p = os.path.join(HERE, "sB3.py")
s = open(p).read()
a = s.index('    Ms = sim(Mw)\n')
b = s.index('if PLANT is None:\n    json.dump(results')
new = open(os.path.join(HERE, "tools_b3_tail.txt")).read()
s = s[:a] + new + s[b:]
open(p, "w").write(s)
print("patched")
