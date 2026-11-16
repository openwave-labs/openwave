"""one-off edit: in planted mode, s4_boxes.py runs only box O1 R4 and skips the point comparison."""
import os
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
p = os.path.join(ROOT, "s4_boxes.py"); s = open(p).read()
a = 'for bname, box in inp["boxes"]["boxes"].items():\n    name, s = bname.split()\n'
assert a in s
s = s.replace(a, a + '    if PLANT and bname != "O1 R4":\n        continue\n')
b = 'pts_exact = json.load(open(os.path.join(OUT, "C4_points.json")))\n'
assert b in s
s = s.replace(b, b + 'if PLANT:\n    print("failures:", n_fails()); sys.exit(0)\n')
open(p, "w").write(s)
print("patched")
