"""Item A.4, high-precision route for Q: the defining integral, by the 50-digit quadrature of quad.py,
at R2's eight points, against the exact h(u) = sum_J q_J f_J(u).
Usage: ./py sA4_quad.py [--plant=point]
"""
import sys, json
import mpmath as mpm
from lib import *
from quartic import fJ
import quad

PLANT = next((a.split("=", 1)[1] for a in sys.argv[1:] if a.startswith("--plant=")), None)
chk = Checker()
n = 7; n1 = 8; j = sp.Rational(7, 2)
q = {int(J): sp.Rational(v) for J, v in json.load(open(rel("out_A1_R2.json")))["qJ"].items()}
d = json.load(open(rel("out_slot_R2.json")))
eta = sp.Matrix(n1, 2, lambda r, c: sp.sympify(d["eta"][r][c]))
s = sp.sympify(d["s"])
pts = json.load(open(rel("inputs.json")))["R2 points"]["points"]
mpm.mp.dps = 50
nodes = quad.haar_D(n, 32, 16)
eta_mp = [[mpm.mpmathify(sp.N(eta[k, a], 70)) for a in range(2)] for k in range(n1)]
out = {}
for name, comp in sorted(pts.items()):
    w = [sp.Integer(0)] * n1
    for m, v in comp.items():
        w[int(sp.Rational(m) + j)] = sp.sympify(v)
    Qe = sp.radsimp(sp.expand(sum(q[J] * v for J, v in fJ(n, w).items())))
    wm = [mpm.mpmathify(sp.N(x, 70)) for x in w]
    if PLANT == "point":
        wm[0] += mpm.mpf("0.001")
    (a2, a4), = quad.averages(nodes, eta_mp, [wm])
    nw2 = mpm.fsum(abs(x) ** 2 for x in wm)
    Qn = a4 / ((mpm.mpf(s.p) / s.q) ** 2 * nw2 ** 2)
    dq = abs(Qn - mpm.mpmathify(sp.N(Qe, 60)))
    print(f"{name:20s} Q exact = {str(Qe):10s} quadrature = {mpm.nstr(Qn, 40)}  |diff| = {mpm.nstr(dq, 3)}")
    chk(f"A.4 {name}: exact Q = 50-digit quadrature of the integral", dq < mpm.mpf(10) ** -40)
    out[name] = dict(Q=str(Qe), Q_quad=mpm.nstr(Qn, 45))
if PLANT is None:
    json.dump(out, open(rel("out_A4_quad.json"), "w"), indent=1)
chk.summary()
