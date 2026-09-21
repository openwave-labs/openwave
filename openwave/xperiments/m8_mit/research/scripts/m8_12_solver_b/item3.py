"""Item 3: the six dimension-1 classes are critical points of r-hat_6 on the unit sphere.

The argument is in RETURN.md (symmetric criticality); this script computes the
values exactly and checks the tangential gradient grad N(u) - 4 N(u) u = 0 exactly.
"""
import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sympy as sp
import core
from core import report, IDX

HERE = os.path.dirname(os.path.abspath(__file__))
ok = True


def vec(d):
    c = [sp.Integer(0)] * 7
    for m, val in d.items():
        c[IDX[m]] += sp.sympify(val)
    return c


PTS = {"v0": vec({0: 1}), "v1": vec({1: 1}), "v2": vec({2: 1}), "v3": vec({3: 1}),
       "hex v3+v-3": vec({3: 1, -3: 1}), "oct v2-v-2": vec({2: 1, -2: -1})}
out = {}
for name, c in PTS.items():
    g, r, Nv = core.sphere_grad(c)
    val = core.rhat6(c)
    ok &= report("%s: N(u/|u|) equals r6 computed from rho_6 directly" % name, sp.simplify(Nv - val) == 0)
    ok &= report("%s: tangential gradient of r6 vanishes exactly" % name, g == sp.zeros(14, 1))
    print("  %s: r6 = %s ~ %.15f" % (name, val, float(val)))
    out[name] = str(val)

# the same check at a non-critical point must fail (shows the PASS line above can fail)
c = vec({3: 1, 1: 1, -2: -1})
g, r, Nv = core.sphere_grad(c)
gn2 = core.exact(sum(t ** 2 for t in g))
print("control (v3+v1-v-2)/sqrt3: |grad|^2 =", gn2)
ok &= report("control: the gradient check fires at a non-critical point", gn2 != 0)
json.dump(out, open(os.path.join(HERE, "out", "item3.json"), "w"), indent=1)
print("ITEM3", "ALLPASS" if ok else "SOMEFAIL")
