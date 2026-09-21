"""Item 0: a CG value, Theta^2, and two exact values of r-hat_6."""
import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sympy as sp
import mpmath
import core
from core import report, IDX, MS

HERE = os.path.dirname(os.path.abspath(__file__))
out = {}
ok = True

# --- CG value, three routes: Racah formula (core.cg), stretched closed form, library
v = core.CG6[(3, -3)]
closed = sp.sqrt(sp.Rational(sp.binomial(6, 6) * sp.binomial(6, 0), sp.binomial(12, 6)))
print("<3 3; 3 -3 | 6 0> =", v, " (= 1/sqrt(924))")
ok &= report("CG(3,3;3,-3|6,0) equals stretched closed form sqrt(C(6,6)C(6,0)/C(12,6))",
             sp.simplify(v - closed) == 0)
# whole table: stretched closed form sqrt(C(6,3+m1)C(6,3+m2)/C(12,6+Q)) (all positive)
bad = 0
for (m1, m2), c in core.CG6.items():
    cf = sp.sqrt(sp.Rational(sp.binomial(6, 3 + m1) * sp.binomial(6, 3 + m2), sp.binomial(12, 6 + m1 + m2)))
    bad += sp.simplify(c - cf) != 0
ok &= report("all 49 <3 m1;3 m2|6 Q> equal the stretched closed form", bad == 0, "mismatches=%d" % bad)
ok &= report("Condon-Shortley normalisation <3 3;3 3|6 6> = +1", core.CG6[(3, 3)] == 1)
# orthonormality of the J=6 column against all other J (built by the same Racah routine)
worst = 0
for Q in range(-6, 7):
    for JJ in range(abs(Q), 7):
        s = sum(core.cg(3, m1, 3, Q - m1, 6, Q) * core.cg(3, m1, 3, Q - m1, JJ, Q)
                for m1 in MS if (Q - m1) in IDX)
        target = 1 if JJ == 6 else 0
        worst = max(worst, abs(core.exact(s) - target))
ok &= report("Racah CG: J=6 column orthonormal to J=0..6 columns (exact)", worst == 0, "worst=%s" % worst)
# library comparison (declared in manifest)
from sympy.physics.quantum.cg import CG as LibCG
libbad = sum(sp.simplify(c - LibCG(3, m1, 3, m2, 6, m1 + m2).doit()) != 0 for (m1, m2), c in core.CG6.items())
ok &= report("own CG agree with sympy.physics.quantum.cg.CG on all 49 entries", libbad == 0, "mismatches=%d" % libbad)
out["cg_33_3m3_60"] = str(v)

# --- Theta^2
a = sp.symbols("a0:7")
b = sp.symbols("b0:7", real=True)
u = [a_ + sp.I * b_ for a_, b_ in zip(sp.symbols("p0:7", real=True), b)]
tt = core.theta(core.theta(u))
diff = [sp.simplify(sp.expand(x - y)) for x, y in zip(tt, u)]
ok &= report("Theta(Theta u) = u for symbolic u (exact)", all(d_ == 0 for d_ in diff))
out["theta_squared"] = "Theta(Theta u) = +u"

# --- r-hat_6 values
def vec(d):
    c = [sp.Integer(0)] * 7
    for m, val in d.items():
        c[IDX[m]] += sp.sympify(val)
    return c

pts = {"2v3+v1-3v-2": vec({3: 2, 1: 1, -2: -3}), "v3+i v0+2v-1": vec({3: 1, 0: sp.I, -1: 2})}
out["rhat6"] = {}
for name, c in pts.items():
    ex = core.rhat6(c)
    agree = []
    for dps in (50, 100):
        mpmath.mp.dps = dps
        mv = core.rhat6_mp([mpmath.mpc(sp.re(x), sp.im(x)) for x in c], mpmath)
        err = abs(mv - mpmath.mpf(ex.p) / ex.q)
        agree.append(err)
        ok &= report("r6(%s): exact %s vs mpmath at %d digits" % (name, ex, dps), err < mpmath.mpf(10) ** (-(dps - 5)),
                     "|diff|=%s" % mpmath.nstr(err, 5))
    print("r6(%s) = %s ~ %s" % (name, ex, sp.N(ex, 30)))
    out["rhat6"][name] = str(ex)

os.makedirs(os.path.join(HERE, "out"), exist_ok=True)
json.dump(out, open(os.path.join(HERE, "out", "item0.json"), "w"), indent=1)
print("ITEM0", "ALLPASS" if ok else "SOMEFAIL")
