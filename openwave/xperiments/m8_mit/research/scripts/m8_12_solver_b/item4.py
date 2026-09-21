"""Item 4: unite the critical points of items 2 and 3 modulo rotations and phase.

Every pair of critical points with equal r6 is decided either by an exhibited exact
rotation g with D(g) s proportional to t (tested by the Cauchy-Schwarz equality
|<t, D s>|^2 = |t|^2 |s|^2, exactly), or by an exact rotation-invariant that differs.
"""
import os, sys, json, itertools
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sympy as sp
import numpy as np
import core, rot
from core import report, IDX, MS

HERE = os.path.dirname(os.path.abspath(__file__))
ok = True
pi = sp.pi
al = sp.symbols("alpha", real=True)


def vec(d):
    c = [sp.Integer(0)] * 7
    for m, val in d.items():
        c[IDX[m]] += sp.sympify(val)
    return c


def M(c):
    return sp.Matrix(c)


def proportional(s, t):
    s, t = M(s), M(t)
    ip = (t.H * s)[0, 0]
    lhs = sp.expand(ip * sp.conjugate(ip))
    rhs = sp.expand((t.H * t)[0, 0] * (s.H * s)[0, 0])
    diff_ = lhs - rhs
    if diff_.has(al):                      # circle families: compare in exponential form
        diff_ = sp.expand(diff_.rewrite(sp.exp))
    return sp.simplify(sp.radsimp(diff_)) == 0


# critical points (representatives in the charts of item 2, plus item 3)
s3_10, s5_6, s2_5, s10_23 = sp.sqrt(sp.Rational(3, 10)), sp.sqrt(sp.Rational(5, 6)), sp.sqrt(sp.Rational(2, 5)), sp.sqrt(sp.Rational(10, 23))
CP = {
    "v3": vec({3: 1}), "v-3": vec({-3: 1}),
    "v2": vec({2: 1}), "v-2": vec({-2: 1}),
    "v1": vec({1: 1}), "v-1": vec({-1: 1}),
    "v0": vec({0: 1}),
    "oct": vec({2: 1, -2: -1}),
    "hex": vec({3: 1, -3: 1}),
    "L12 circle": vec({1: 1, -2: sp.exp(sp.I * al) / 2}),
    "L23 circle": vec({2: 1, -3: sp.sqrt(sp.Rational(12, 13)) * sp.exp(sp.I * al)}),
    "L22 circle": vec({2: 1, -2: sp.exp(sp.I * al)}),
    "L33 circle": vec({3: 1, -3: sp.exp(sp.I * al)}),
    "U max +": vec({0: 1, 2: s3_10, -2: s3_10}), "U max -": vec({0: 1, 2: -s3_10, -2: -s3_10}),
    "U min +": vec({0: 1, 2: sp.I * s5_6, -2: sp.I * s5_6}), "U min -": vec({0: 1, 2: -sp.I * s5_6, -2: -sp.I * s5_6}),
    "W sad +": vec({0: 1, 3: sp.I * s2_5, -3: sp.I * s2_5}), "W sad -": vec({0: 1, 3: -sp.I * s2_5, -3: -sp.I * s2_5}),
    "W min +": vec({0: 1, 3: s10_23, -3: s10_23}), "W min -": vec({0: 1, 3: -s10_23, -3: -s10_23}),
}
vals = {}
for k, c in CP.items():
    vals[k] = core.exact(core.rhat6(c))
    ok &= report("%s: r6 = %s (independent of the circle parameter alpha)" % (k, vals[k]), not vals[k].has(al))

# ---------------------------------------------------------------- rotations
Rx = sp.zeros(7)
for m in MS:
    Rx[IDX[-m], IDX[m]] = -1
Dz = rot.Dz
ROT = {}
ROT[("v3", "v-3")] = ("R_x(pi)", Rx)
ROT[("v2", "v-2")] = ("R_x(pi)", Rx)
ROT[("v1", "v-1")] = ("R_x(pi)", Rx)
ROT[("oct", "L22 circle")] = ("R_z((alpha - pi)/4)", Dz((al - pi) / 4))
ROT[("hex", "L33 circle")] = ("R_z(alpha/6)", Dz(al / 6))
ROT[("U max +", "U max -")] = ("R_z(pi/2)", Dz(pi / 2))
ROT[("U min +", "U min -")] = ("R_z(pi/2)", Dz(pi / 2))
ROT[("W sad +", "W sad -")] = ("R_z(pi/3)", Dz(pi / 3))
ROT[("W min +", "W min -")] = ("R_z(pi/3)", Dz(pi / 3))
n_ = (sp.Integer(1) / sp.sqrt(3), sp.Integer(1) / sp.sqrt(3), -sp.Integer(1) / sp.sqrt(3))
ROT[("hex", "U max +")] = ("rotation by 4pi/3 about (1,1,-1)/sqrt3", rot.D3_exact(n_, 4 * pi / 3))
# oct -> W saddle: R1 = rotation about (1,-1,0)/sqrt2 by arccos(1/sqrt3) (takes the 3-fold axis
# (1,1,1)/sqrt3 of the octahedral state to the z axis), then a z rotation fixed exactly below.
R1 = rot.D3_exact((1 / sp.sqrt(2), -1 / sp.sqrt(2), 0), sp.acos(1 / sp.sqrt(3)))
s = (R1 * M(CP["oct"])).applyfunc(core.exact)
print("R1 . oct =", list(s))
ok &= report("R1 . oct lies in span{v3, v0, v-3} (3-fold axis now along z)",
             all(s[IDX[m]] == 0 for m in (2, 1, -1, -2)))
a0, b3, c3 = s[IDX[0]], s[IDX[3]], s[IDX[-3]]
w = core.exact(sp.I * s2_5 * a0 / b3)       # required e^{-3 i beta}
ok &= report("required phase w = e^{-3 i beta} has |w| = 1 and fixes the v-3 component too",
             sp.simplify(w * sp.conjugate(w) - 1) == 0 and sp.simplify(sp.conjugate(w) * c3 / a0 - sp.I * s2_5) == 0,
             "w = %s" % w)
ok &= report("w = e^{-3 i pi/4}, so beta = pi/4", sp.simplify(w - sp.expand_complex(sp.exp(-3 * sp.I * pi / 4))) == 0)
ROT[("oct", "W sad +")] = ("R_z(pi/4) R1, R1 = rotation about (1,-1,0)/sqrt2 by arccos(1/sqrt3)", Dz(pi / 4) * R1)

# control: the proportionality test must reject a wrong target
ok &= report("control: proportionality test rejects [hex] -> [oct] under the hex->U rotation",
             not proportional(list(ROT[("hex", "U max +")][1] * M(CP["hex"])), CP["oct"]))
rot_out = {}
for (a, b), (desc, D) in ROT.items():
    src = CP[a]
    img = (D * M(src)).applyfunc(lambda q: sp.expand(q.rewrite(sp.cos)) if not q.has(al) else sp.expand(q))
    good = proportional(list(img), CP[b])
    ok &= report("rotation %s carries [%s] onto [%s] (exact)" % (desc, a, b), good)
    Dn = np.array(sp.N(D.subs(al, sp.Rational(7, 10)), 30), dtype=complex)
    sn = np.array(sp.N(M(src).subs(al, sp.Rational(7, 10)), 30), dtype=complex).ravel()
    tn = np.array(sp.N(M(CP[b]).subs(al, sp.Rational(7, 10)), 30), dtype=complex).ravel()
    resid = 1 - abs(np.vdot(tn, Dn @ sn)) ** 2 / (np.vdot(tn, tn).real * np.vdot(sn, sn).real)
    ok &= report("  float cross-check at alpha = 0.7: 1 - |<t,Ds>|^2/(|t|^2|s|^2) < 1e-12", abs(resid) < 1e-12,
                 "%.2e" % resid)
    rot_out["%s -> %s" % (a, b)] = desc

# ---------------------------------------------------------------- invariants
Jz, Jp, Jm, Jx, Jy = core.Jmats()


def Jvec2(c):
    v = M(c); n2 = (v.H * v)[0, 0]
    comps = [sp.simplify((v.H * J * v)[0, 0] / n2) for J in (Jx, Jy, Jz)]
    return core.exact(sum(sp.Abs(q) ** 2 for q in comps))


def I3(c):
    """cubic invariant <J>_a <J>_b <Q_ab>, Q_ab = (J_a J_b + J_b J_a)/2 - 4 delta_ab."""
    v = M(c); n2 = (v.H * v)[0, 0]
    Js = (Jx, Jy, Jz)
    Jv = [sp.simplify((v.H * J * v)[0, 0] / n2) for J in Js]
    tot = 0
    for a_ in range(3):
        for b_ in range(3):
            Q = (Js[a_] * Js[b_] + Js[b_] * Js[a_]) / 2 - (4 if a_ == b_ else 0) * sp.eye(7)
            tot += Jv[a_] * Jv[b_] * (v.H * Q * v)[0, 0] / n2
    return core.exact(tot)


def circle_stabiliser(c, a, b):
    """For u = v_a + (nonzero) v_b with <J> != 0 along z: projective stabiliser is C_|a-b|.
    Returns |a-b| after checking <J> is a nonzero multiple of e_z."""
    v = M(c); n2 = (v.H * v)[0, 0]
    Jv = [sp.simplify((v.H * J * v)[0, 0] / n2) for J in (Jx, Jy, Jz)]
    assert Jv[0] == 0 and Jv[1] == 0 and Jv[2] != 0
    return abs(a - b), Jv[2]


INV = {}
for k in ["v3", "v2", "W min +", "v1", "U min +", "L12 circle", "L23 circle", "oct", "v0", "hex"]:
    c = [q.subs(al, 0) for q in CP[k]]
    INV[k] = {"|<J>|^2": str(Jvec2(c)), "I3": str(I3(c))}
    for kk in range(0, 7):
        INV[k]["r%d" % kk] = str(core.rhat_k(c, kk))
    print("%-11s" % k, INV[k])

for k, (a, b) in (("L12 circle", (1, -2)), ("L23 circle", (2, -3))):
    order_, jz = circle_stabiliser([q.subs(al, 0) for q in CP[k]], a, b)
    INV[k]["projective stabiliser"] = "C_%d (<J> = %s e_z)" % (order_, jz)
    print(k, INV[k]["projective stabiliser"])

# separations of equal-valued pairs that are not joined by a rotation
SEP = [("v1", "U min +"), ("L12 circle", "L23 circle")]
sep_out = {}
for a, b in SEP:
    ok &= report("[%s] and [%s] have the same r6" % (a, b), vals[a] == vals[b], str(vals[a]))
    differ = {q: (INV[a][q], INV[b][q]) for q in INV[a] if INV[a][q] != INV[b][q]}
    ok &= report("[%s] and [%s] separated by an exact rotation invariant" % (a, b), len(differ) > 0, str(differ))
    sep_out["%s | %s" % (a, b)] = differ

# every equal-value pair accounted for (union-find over rotations; separated pairs checked)
parent = {k: k for k in CP}


def find(k):
    while parent[k] != k:
        k = parent[k]
    return k


for (a, b) in ROT:
    parent[find(b)] = find(a)
orbits = {}
for k in CP:
    orbits.setdefault(find(k), []).append(k)
undecided = []
for r1, r2 in itertools.combinations(orbits, 2):
    if vals[r1] == vals[r2]:
        pair_ok = any({find(a), find(b)} == {r1, r2} for a, b in SEP)
        if not pair_ok:
            undecided.append((r1, r2))
ok &= report("every pair of distinct orbits with equal r6 is separated by an invariant", not undecided, str(undecided))
print("\nORBITS (%d):" % len(orbits))
orb_out = []
for r in sorted(orbits, key=lambda k: float(vals[k])):
    print("  r6 = %-8s ~ %.10f  members: %s" % (vals[r], float(vals[r]), orbits[r]))
    orb_out.append({"value": str(vals[r]), "members": orbits[r]})
json.dump({"orbits": orb_out, "rotations": rot_out, "separations": sep_out, "invariants": INV},
          open(os.path.join(HERE, "out", "item4.json"), "w"), indent=1)
print("ITEM4", "ALLPASS" if ok else "SOMEFAIL")
